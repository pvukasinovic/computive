"""MLASIC Stage 5: RTL Generation.

Generates synthesizable SystemVerilog RTL from a scheduled IR graph.
Produces model-specific parameters.svh and accelerator_top.sv, copies
the module library, and assembles a complete build directory.

Two paths:
  - MLP path: Single shared MAC array, time-multiplexed across layers.
    (original Stage 5, unchanged for backward compatibility)
  - Tile fabric path: Per-tile compute modules with static routing.
    (Stage 5.1, for CNN/Transformer/arbitrary DAG models)

See docs/rtl-interface-spec.md §9.2.
"""

from __future__ import annotations

import logging
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from mlasic.ir import (
    FusedConvAttrs,
    FusedLinearAttrs,
    Graph,
    HardwareConstraints,
    OpType,
    PoolAttrs,
)
from mlasic.rtl_emit import emit_module
from mlasic.rtl_lowering import lower_mlp_top, render_mlp_parameters_svh
from mlasic.rtl_lowering_tile import (
    lower_asic_rom_module,
    lower_tile_accelerator_top,
    lower_tile_fabric_module,
    render_tile_parameters_svh,
)
from mlasic.tile_mapper import FabricConfig, TileConfig, TileType

logger = logging.getLogger(__name__)

# Root of the RTL module library (relative to this file)
_RTL_LIB_DIR = Path(__file__).resolve().parent.parent.parent / "rtl"

# TileType enum value → numeric TILE_TYPE parameter for tile.sv
_TILE_TYPE_ID: dict[TileType, int] = {
    TileType.MAC: 0,
    TileType.ALU: 1,
    TileType.NORM: 2,
    TileType.SOFTMAX: 3,
    TileType.ACTIVATION: 4,
    TileType.POOL: 5,
    TileType.RESHAPE: 6,
}


@dataclass
class RTLGenerator:
    """Generate SystemVerilog RTL from a scheduled graph.

    Takes a scheduled graph (stage="scheduled" or "scheduled_dag") with
    fused_attrs per node, and generates a complete RTL output directory.

    Two paths selected automatically:
      - MLP path: pure FusedLinear/FusedLinearReLU chains → existing arch
      - Tile fabric path: anything else → tile-based fabric
    """

    graph: Graph
    weight_dir: Path
    output_dir: Path
    constraints: HardwareConstraints = field(default_factory=HardwareConstraints)
    fabric_config: Optional[FabricConfig] = None
    target: str = "fpga"  # "fpga" or "asic"

    def __post_init__(self) -> None:
        self.weight_dir = Path(self.weight_dir)
        self.output_dir = Path(self.output_dir)
        if self.graph.stage not in ("scheduled", "scheduled_dag"):
            raise ValueError(
                f"RTLGenerator requires 'scheduled' or 'scheduled_dag' stage, "
                f"got '{self.graph.stage}'"
            )

    def _is_mlp_linear_chain(self) -> bool:
        """Return True only for pure FusedLinear/FusedLinearReLU chains.

        This selects the original MLP path for backward compatibility.
        Uses tile fabric path when fabric_config is provided or graph
        was DAG-scheduled, even for pure MLP models.
        """
        if self.fabric_config is not None:
            return False
        ordered_names = self.graph.topological_order()
        for name in ordered_names:
            node = self.graph.nodes[name]
            if node.op_type not in (OpType.FUSED_LINEAR, OpType.FUSED_LINEAR_RELU):
                return False
            if not isinstance(node.fused_attrs, FusedLinearAttrs):
                return False
        return True

    def generate_all(self) -> Path:
        """Generate complete RTL output. Returns output directory path."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("RTL Generation: output_dir=%s", self.output_dir)

        if self._is_mlp_linear_chain():
            # Original MLP path — identical output for backward compat
            logger.info("MLP linear chain detected — using MLP path")
            self.copy_module_library()
            self.copy_weight_files()
            self.generate_parameters_svh()
            self.generate_accelerator_top()
            self.validate()
        else:
            # Tile fabric path
            logger.info("Non-MLP model — using tile fabric path")
            self.copy_module_library()
            self.copy_weight_files()
            self.generate_lut_files()
            self.generate_requant_rom_files()
            self.generate_tile_parameters_svh()
            self.generate_tile_fabric_sv()
            self.generate_tile_accelerator_top()
            if self.target == "asic":
                self.generate_asic_rom_modules()
            self.validate_tile_fabric()

        logger.info("RTL Generation complete: %s", self.output_dir)
        return self.output_dir

    # ------------------------------------------------------------------
    # Module library copy
    # ------------------------------------------------------------------

    def copy_module_library(self) -> None:
        """Copy all .sv and .xdc files from the RTL library to output."""
        lib_dir = _RTL_LIB_DIR
        if not lib_dir.is_dir():
            raise FileNotFoundError(f"RTL library not found at {lib_dir}")

        # Copy subdirectories: compute, memory, layer, interface, constraints, tile
        for subdir in ["compute", "memory", "layer", "interface", "constraints", "tile"]:
            src = lib_dir / subdir
            if not src.is_dir():
                continue
            dst = self.output_dir / subdir
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                if f.suffix in (".sv", ".xdc"):
                    shutil.copy2(f, dst / f.name)
                    logger.debug("Copied %s -> %s", f, dst / f.name)

        logger.info("Copied RTL module library (%s)", lib_dir)

    # ------------------------------------------------------------------
    # Weight file copy
    # ------------------------------------------------------------------

    def copy_weight_files(self) -> None:
        """Copy .mem files from Stage 4 weight directory to output."""
        if not self.weight_dir.is_dir():
            raise FileNotFoundError(f"Weight directory not found: {self.weight_dir}")

        for f in self.weight_dir.iterdir():
            if f.suffix == ".mem":
                shutil.copy2(f, self.output_dir / f.name)
                logger.debug("Copied %s -> %s", f, self.output_dir / f.name)

        logger.info("Copied weight files from %s", self.weight_dir)

    # ------------------------------------------------------------------
    # parameters.svh generation
    # ------------------------------------------------------------------

    def generate_parameters_svh(self) -> Path:
        """Generate model-specific parameters.svh from schedule and graph.

        Thin shim around ``rtl_lowering.render_mlp_parameters_svh`` — the
        parameters.svh artifact is constants-only, so it isn't modeled as
        an RTLModule. The renderer lives next to the IR lowering pass for
        cohesion.
        """
        text = render_mlp_parameters_svh(self.graph, self.constraints)
        out_path = self.output_dir / "parameters.svh"
        out_path.write_text(text)
        logger.info("Generated %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # accelerator_top.sv generation
    # ------------------------------------------------------------------

    def generate_accelerator_top(self) -> Path:
        """Generate model-specific ``accelerator_top.sv`` via the RTL IR.

        Delegates to ``rtl_lowering.lower_mlp_top`` to build a structured
        ``RTLModule``, then to ``rtl_emit.emit_module`` for serialization.
        """
        module = lower_mlp_top(self.graph, self.constraints)
        sv = emit_module(module)
        out_path = self.output_dir / "accelerator_top.sv"
        out_path.write_text(sv)
        logger.info("Generated %s (via RTL IR)", out_path)
        return out_path

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Post-generation validation: check all expected files exist."""
        expected_sv = [
            "compute/mac_array.sv",
            "compute/requantize.sv",
            "compute/activation_relu.sv",
            "memory/sram_bank.sv",
            "memory/ping_pong_buffer.sv",
            "layer/fused_linear_relu.sv",
            "layer/byte_select.sv",
            "layer/bias_unpack.sv",
            "interface/axi_stream_in.sv",
            "interface/axi_stream_out.sv",
            "interface/axi_lite_ctrl.sv",
            "constraints/constraints.xdc",
        ]

        missing = []
        for f in expected_sv:
            if not (self.output_dir / f).is_file():
                missing.append(f)

        # Check generated files
        for f in ["parameters.svh", "accelerator_top.sv"]:
            if not (self.output_dir / f).is_file():
                missing.append(f)

        # Check weight files
        for f in ["weight_bank.mem", "bias_bank.mem"]:
            if not (self.output_dir / f).is_file():
                missing.append(f)

        if missing:
            raise FileNotFoundError(f"RTL generation incomplete, missing files: {missing}")

        logger.info("Validation passed: all %d expected files present", len(expected_sv) + 4)

    # ==================================================================
    # Tile Fabric Path — Stage 5.1
    # ==================================================================

    def _get_tile_configs(self) -> list[TileConfig]:
        """Get tile configs from fabric_config or build minimal list from graph."""
        if self.fabric_config is not None:
            return self.fabric_config.tiles
        # Fallback: one tile per node, infer type
        from mlasic.tile_mapper import OP_TO_TILE

        tiles = []
        for idx, name in enumerate(self.graph.topological_order()):
            node = self.graph.nodes[name]
            tile_type = OP_TO_TILE.get(node.op_type, TileType.RESHAPE)
            weight_bytes = 0
            bias_bytes = 0
            if node.dag_schedule is not None:
                weight_bytes = node.dag_schedule.weight_bytes
                bias_bytes = node.dag_schedule.bias_bytes
            tiles.append(
                TileConfig(
                    tile_id=idx,
                    tile_type=tile_type,
                    operator_assignment=name,
                    weight_bytes=weight_bytes,
                    bias_bytes=bias_bytes,
                )
            )
        return tiles

    def _get_routes(self) -> list:
        """Get routes from fabric_config or empty list."""
        if self.fabric_config is not None:
            return self.fabric_config.routes
        return []

    def _ensure_fabric_config(self) -> FabricConfig:
        """Return ``self.fabric_config`` or synthesize one from the graph.

        The IR-side tile-fabric lowering passes need a ``FabricConfig`` —
        this lets callers invoke the tile-* generators without supplying one
        explicitly (they get a one-tile-per-node fallback).
        """
        if self.fabric_config is not None:
            return self.fabric_config
        tiles = self._get_tile_configs()
        return FabricConfig(
            tiles=tiles,
            routes=[],
            total_weight_bytes=sum(t.weight_bytes for t in tiles),
            total_bias_bytes=sum(t.bias_bytes for t in tiles),
            total_tiles=len(tiles),
        )

    # ------------------------------------------------------------------
    # LUT file generation
    # ------------------------------------------------------------------

    def generate_lut_files(self) -> list[Path]:
        """Generate 256-entry .mem LUT files for nonlinear activations.

        Generates: gelu_lut.mem, silu_lut.mem, exp_lut.mem, rsqrt_lut.mem
        All use quantization-aware computation (INT8 input → INT8/INT16 output).
        """
        generated: list[Path] = []

        # GELU LUT: INT8 input [-128..127] → INT8 output
        gelu_table = np.zeros(256, dtype=np.int8)
        for i in range(256):
            x_int = np.int8(i if i < 128 else i - 256)
            x_fp = float(x_int) / 128.0  # rough dequant to [-1, 1]
            gelu_fp = 0.5 * x_fp * (1.0 + math.erf(x_fp / math.sqrt(2.0)))
            gelu_table[i] = np.int8(np.clip(np.floor(gelu_fp * 128.0 + 0.5), -128, 127))
        gelu_path = self.output_dir / "gelu_lut.mem"
        self._write_lut_mem(gelu_path, gelu_table)
        generated.append(gelu_path)

        # SiLU LUT: INT8 input [-128..127] → INT8 output
        silu_table = np.zeros(256, dtype=np.int8)
        for i in range(256):
            x_int = np.int8(i if i < 128 else i - 256)
            x_fp = float(x_int) / 128.0
            silu_fp = x_fp / (1.0 + math.exp(-x_fp))
            silu_table[i] = np.int8(np.clip(np.floor(silu_fp * 128.0 + 0.5), -128, 127))
        silu_path = self.output_dir / "silu_lut.mem"
        self._write_lut_mem(silu_path, silu_table)
        generated.append(silu_path)

        # Exp LUT: INT8 input [0..255] → INT16 output (for softmax)
        # exp(x) where x is (val - max), shifted to [0..255]
        exp_table = np.zeros(256, dtype=np.uint16)
        for i in range(256):
            # i=255 is x-max=0 (max val), i=0 is x-max=-255
            x = i - 255  # range [-255, 0]
            exp_val = math.exp(x / 32.0)  # scaled exponential
            exp_table[i] = min(int(exp_val * 256 + 0.5), 65535)
        exp_path = self.output_dir / "exp_lut.mem"
        self._write_lut_mem(exp_path, exp_table)
        generated.append(exp_path)

        # Reciprocal-sqrt LUT: 8-bit variance index → 16-bit rsqrt
        rsqrt_table = np.zeros(256, dtype=np.uint16)
        for i in range(256):
            if i == 0:
                rsqrt_table[i] = 65535  # max value for zero variance
            else:
                rsqrt_table[i] = min(int(256.0 / math.sqrt(float(i)) + 0.5), 65535)
        rsqrt_path = self.output_dir / "rsqrt_lut.mem"
        self._write_lut_mem(rsqrt_path, rsqrt_table)
        generated.append(rsqrt_path)

        logger.info("Generated %d LUT files", len(generated))
        return generated

    @staticmethod
    def _write_lut_mem(path: Path, data: np.ndarray) -> None:
        """Write a .mem file with one hex value per line."""
        width = data.dtype.itemsize * 2  # hex chars
        lines = []
        for val in data:
            # Convert to unsigned for hex formatting
            if data.dtype == np.int8:
                uval = int(val) & 0xFF
            elif data.dtype == np.uint16:
                uval = int(val) & 0xFFFF
            else:
                uval = int(val)
            lines.append(f"{uval:0{width}x}")
        path.write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # Per-channel requant ROM generation
    # ------------------------------------------------------------------

    def generate_requant_rom_files(self) -> list[Path]:
        """Generate per-channel requantization ROM .mem files for conv tiles.

        For each conv MAC tile, writes tile_{id}_requant.mem with one 32-bit
        M_fixed per output channel in hex. Used by tile.sv conv path to feed
        per-channel scales to the MAC array's requantize module.
        """
        tiles = self._get_tile_configs()
        generated: list[Path] = []

        for tile in tiles:
            if tile.tile_type != TileType.MAC:
                continue

            node = self.graph.nodes.get(tile.operator_assignment)
            if node is None:
                continue

            attrs = node.fused_attrs
            if not isinstance(attrs, FusedConvAttrs):
                continue

            if not attrs.requant_scale_fixed:
                continue

            # Per-channel M_fixed values
            scales = attrs.requant_scale_fixed
            tid = tile.tile_id

            # Write one 32-bit hex value per line (one per output channel)
            lines = []
            for m_fixed in scales:
                lines.append(f"{m_fixed & 0xFFFFFFFF:08x}")

            out_path = self.output_dir / f"tile_{tid}_requant.mem"
            out_path.write_text("\n".join(lines) + "\n")
            generated.append(out_path)
            logger.debug("Generated requant ROM %s (%d channels)", out_path, len(scales))

        logger.info("Generated %d per-channel requant ROM files", len(generated))
        return generated

    # ------------------------------------------------------------------
    # Tile parameters.svh generation
    # ------------------------------------------------------------------

    def generate_tile_parameters_svh(self) -> Path:
        """Generate ``tile_parameters.svh`` via the IR-side renderer.

        Thin shim around ``rtl_lowering_tile.render_tile_parameters_svh``;
        the constants header isn't an ``RTLModule`` (no body), so it lives
        as a renderer next to the tile-fabric lowering pass.
        """
        fabric = self._ensure_fabric_config()
        text = render_tile_parameters_svh(self.graph, fabric, self.constraints)
        out_path = self.output_dir / "tile_parameters.svh"
        out_path.write_text(text)
        logger.info(
            "Generated %s (%d tiles, via IR renderer)",
            out_path,
            len(fabric.tiles),
        )
        return out_path

    # ------------------------------------------------------------------
    # Tile fabric SystemVerilog generation
    # ------------------------------------------------------------------

    def generate_tile_fabric_sv(self) -> Path:
        """Generate model-specific ``tile_fabric.sv`` via the RTL IR.

        Delegates to ``rtl_lowering_tile.lower_tile_fabric_module``.
        """
        fabric = self._ensure_fabric_config()
        module = lower_tile_fabric_module(self.graph, fabric, self.constraints)
        sv = emit_module(module)
        out_path = self.output_dir / "tile_fabric.sv"
        out_path.write_text(sv)
        logger.info(
            "Generated %s (via RTL IR, %d tiles)", out_path, len(fabric.tiles)
        )
        return out_path

    # ------------------------------------------------------------------
    # Tile accelerator_top.sv generation
    # ------------------------------------------------------------------

    def generate_tile_accelerator_top(self) -> Path:
        """Generate tile-based ``accelerator_top.sv`` via the RTL IR.

        Delegates to ``rtl_lowering_tile.lower_tile_accelerator_top``.
        """
        fabric = self._ensure_fabric_config()
        module = lower_tile_accelerator_top(self.graph, fabric, self.constraints)
        sv = emit_module(module)
        out_path = self.output_dir / "accelerator_top.sv"
        out_path.write_text(sv)
        logger.info("Generated tile-based %s (via RTL IR)", out_path)
        return out_path

    # ------------------------------------------------------------------
    # ASIC ROM modules
    # ------------------------------------------------------------------

    def generate_asic_rom_modules(self) -> list[Path]:
        """Generate per-tile ``rom_tile_<id>.sv`` via the RTL IR.

        Each tile's hardcoded weight ROM is lowered to an ``RTLModule`` by
        ``rtl_lowering_tile.lower_asic_rom_module`` and serialized through
        ``rtl_emit.emit_module``. Only used for ASIC target; the FPGA path
        uses ``sram_bank`` with ``$readmemh``.
        """
        tiles = self._get_tile_configs()
        parallelism = self.constraints.max_parallelism
        generated: list[Path] = []

        for tile in tiles:
            module = lower_asic_rom_module(tile, self.graph, parallelism)
            if module is None:
                continue
            sv = emit_module(module)
            out_path = self.output_dir / f"rom_tile_{tile.tile_id}.sv"
            out_path.write_text(sv)
            generated.append(out_path)

        logger.info("Generated %d ASIC ROM modules (via RTL IR)", len(generated))
        return generated

    # ------------------------------------------------------------------
    # Tile fabric validation
    # ------------------------------------------------------------------

    def validate_tile_fabric(self) -> None:
        """Post-generation validation for tile fabric path."""
        expected = [
            "tile_parameters.svh",
            "tile_fabric.sv",
            "accelerator_top.sv",
        ]

        # Module library files (same as MLP path)
        expected_sv = [
            "compute/mac_array.sv",
            "compute/requantize.sv",
            "compute/activation_relu.sv",
            "compute/conv_engine.sv",
            "compute/softmax_unit.sv",
            "compute/layer_norm_unit.sv",
            "compute/activation_unit.sv",
            "compute/pool_unit.sv",
            "memory/sram_bank.sv",
            "memory/ping_pong_buffer.sv",
            "memory/rom_tile.sv",
            "tile/tile.sv",
            "tile/tile_fabric.sv",
            "layer/fused_linear_relu.sv",
            "layer/byte_select.sv",
            "layer/bias_unpack.sv",
            "interface/axi_stream_in.sv",
            "interface/axi_stream_out.sv",
            "interface/axi_lite_ctrl.sv",
            "constraints/constraints.xdc",
        ]

        # LUT files
        expected_lut = [
            "gelu_lut.mem",
            "silu_lut.mem",
            "exp_lut.mem",
            "rsqrt_lut.mem",
        ]

        missing = []
        for f in expected + expected_sv + expected_lut:
            if not (self.output_dir / f).is_file():
                missing.append(f)

        if missing:
            raise FileNotFoundError(
                f"Tile fabric RTL generation incomplete, missing files: {missing}"
            )

        logger.info("Tile fabric validation passed")
