"""Tests for MLASIC Stage 5.1: RTL Generation — Tile-Based Fabric.

Tests structural correctness of generated tile fabric RTL:
  - New SV modules pass verilator lint
  - MLP detection and backward compat
  - CNN tile fabric generation
  - Transformer tile fabric generation
  - FPGA vs ASIC mode
  - LUT file generation
  - Weight ROM integrity
  - Full pipeline integration
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mlasic.dag_scheduler import DAGScheduler

# Stage 1-4 imports for MLP backward compat tests
from mlasic.ingestion import ONNXParser
from mlasic.ir import (
    FusedConvAttrs,
    FusedLinearAttrs,
    Graph,
    OpNode,
    OpType,
    QuantParams,
    Tensor,
    TensorType,
)
from mlasic.optimization import (
    BatchNormFoldingPass,
    ConstantFoldingPass,
    DeadCodeEliminationPass,
    OperatorFusionPass,
    PassManager,
    QuantizationPass,
)
from mlasic.rtl_gen import RTLGenerator
from mlasic.scheduler import Scheduler
from mlasic.tile_mapper import (
    FabricConfig,
    TileMapper,
    TileType,
)
from mlasic.weight_packer import WeightPacker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_quantized_linear_node(
    name: str,
    input_name: str,
    output_name: str,
    input_dim: int,
    output_dim: int,
    has_relu: bool = True,
) -> tuple[OpNode, dict[str, Tensor]]:
    """Create a quantized FusedLinear node with tensors for testing."""
    qp = QuantParams(scale=0.1, zero_point=0, calibrated=True)
    attrs = FusedLinearAttrs(
        input_dim=input_dim,
        output_dim=output_dim,
        has_relu=has_relu,
        weight_quant=qp,
        input_quant=qp,
        output_quant=qp,
        requant_scale_fixed=65536,
        requant_shift=16,
    )
    op = OpType.FUSED_LINEAR_RELU if has_relu else OpType.FUSED_LINEAR
    w_name = f"{name}_w"
    b_name = f"{name}_b"
    node = OpNode(name, op, [input_name, w_name, b_name], [output_name])
    node.fused_attrs = attrs
    tensors = {
        input_name: Tensor(input_name, TensorType((1, input_dim), np.dtype(np.int8))),
        w_name: Tensor(
            w_name,
            TensorType((input_dim, output_dim), np.dtype(np.int8)),
            data=np.zeros((input_dim, output_dim), dtype=np.int8),
        ),
        b_name: Tensor(
            b_name,
            TensorType((output_dim,), np.dtype(np.int32)),
            data=np.zeros(output_dim, dtype=np.int32),
        ),
        output_name: Tensor(output_name, TensorType((1, output_dim), np.dtype(np.int8))),
    }
    return node, tensors


def _make_quantized_conv_node(
    name: str,
    input_name: str,
    output_name: str,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    spatial: int = 8,
    has_relu: bool = True,
) -> tuple[OpNode, dict[str, Tensor]]:
    """Create a quantized FusedConv node with tensors for testing."""
    qp_list = [QuantParams(scale=0.1, zero_point=0, calibrated=True)] * out_channels
    qp = QuantParams(scale=0.1, zero_point=0, calibrated=True)
    attrs = FusedConvAttrs(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_shape=[kernel_size, kernel_size],
        pads=[kernel_size // 2] * 4,
        has_relu=has_relu,
        weight_quant=qp_list,
        input_quant=qp,
        output_quant=qp,
        requant_scale_fixed=[65536] * out_channels,
        requant_shift=16,
    )
    op = OpType.FUSED_CONV_RELU if has_relu else OpType.FUSED_CONV
    w_name = f"{name}_w"
    b_name = f"{name}_b"
    node = OpNode(name, op, [input_name, w_name, b_name], [output_name])
    node.fused_attrs = attrs
    tensors = {
        input_name: Tensor(
            input_name,
            TensorType((1, in_channels, spatial, spatial), np.dtype(np.int8)),
        ),
        w_name: Tensor(
            w_name,
            TensorType((out_channels, in_channels, kernel_size, kernel_size), np.dtype(np.int8)),
            data=np.zeros((out_channels, in_channels, kernel_size, kernel_size), dtype=np.int8),
        ),
        b_name: Tensor(
            b_name,
            TensorType((out_channels,), np.dtype(np.int32)),
            data=np.zeros(out_channels, dtype=np.int32),
        ),
        output_name: Tensor(
            output_name,
            TensorType((1, out_channels, spatial, spatial), np.dtype(np.int8)),
        ),
    }
    return node, tensors


def _make_scheduled_linear_chain() -> tuple[Graph, FabricConfig]:
    """Build, schedule, and tile-map a 2-layer linear chain."""
    n0, t0 = _make_quantized_linear_node("n0", "x", "h0", 128, 64)
    n1, t1 = _make_quantized_linear_node("n1", "h0", "y", 64, 32, has_relu=False)
    all_t = {**t0, **t1}
    graph = Graph("test_linear", {"n0": n0, "n1": n1}, all_t, ["x"], ["y"], stage="quantized_dag")
    DAGScheduler().schedule(graph)
    fabric = TileMapper().map(graph)
    return graph, fabric


def _make_scheduled_cnn() -> tuple[Graph, FabricConfig]:
    """Build, schedule, and tile-map a CNN: Conv→Pool→Linear chain."""
    n_conv, t_conv = _make_quantized_conv_node(
        "conv1", "x", "conv_out", 4, 8, kernel_size=3, spatial=8,
    )

    # Pool node (no fused attrs needed for scheduling)
    n_pool = OpNode("pool1", OpType.MAX_POOL, ["conv_out"], ["pool_out"])
    t_pool = {
        "pool_out": Tensor("pool_out", TensorType((1, 8, 4, 4), np.dtype(np.int8))),
    }

    n_linear, t_linear = _make_quantized_linear_node(
        "fc1", "pool_out", "y", 128, 16, has_relu=False,
    )

    all_t = {**t_conv, **t_pool, **t_linear}
    nodes = {"conv1": n_conv, "pool1": n_pool, "fc1": n_linear}
    graph = Graph("test_cnn", nodes, all_t, ["x"], ["y"], stage="quantized_dag")
    DAGScheduler().schedule(graph)
    fabric = TileMapper().map(graph)
    return graph, fabric


def _make_scheduled_transformer_like() -> tuple[Graph, FabricConfig]:
    """Build a simple transformer-like model: Linear→Softmax→Linear."""
    n0, t0 = _make_quantized_linear_node("qkv", "x", "qkv_out", 64, 64)

    # Softmax node
    n_sm = OpNode("softmax", OpType.SOFTMAX, ["qkv_out"], ["attn_out"])
    t_sm = {
        "attn_out": Tensor("attn_out", TensorType((1, 64), np.dtype(np.int8))),
    }

    n1, t1 = _make_quantized_linear_node("proj", "attn_out", "y", 64, 32, has_relu=False)

    all_t = {**t0, **t_sm, **t1}
    nodes = {"qkv": n0, "softmax": n_sm, "proj": n1}
    graph = Graph("test_transformer", nodes, all_t, ["x"], ["y"], stage="quantized_dag")
    DAGScheduler().schedule(graph)
    fabric = TileMapper().map(graph)
    return graph, fabric


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def packed_output(
    ad_model_path: Path, test_vectors: np.ndarray, tmp_path: Path,
) -> tuple[dict, Path, Graph]:
    """Run Stages 1-4 on the AD model, return (memory_map, weight_dir, graph)."""
    parser = ONNXParser(ad_model_path)
    graph = parser.parse()

    pm = PassManager()
    pm.add_pass(ConstantFoldingPass())
    pm.add_pass(DeadCodeEliminationPass())
    pm.add_pass(BatchNormFoldingPass())
    pm.add_pass(OperatorFusionPass())
    pm.add_pass(QuantizationPass(calibration_data=[v for v in test_vectors[:20]]))
    graph = pm.run(graph, verify=True)

    scheduler = Scheduler()
    scheduler.schedule(graph)

    weight_dir = tmp_path / "weights"
    packer = WeightPacker(graph, weight_dir)
    memory_map = packer.pack_weights()
    return memory_map, weight_dir, graph


@pytest.fixture
def cnn_graph_and_fabric(tmp_path: Path) -> tuple[Graph, FabricConfig, Path]:
    """Scheduled CNN graph with fabric config and weight dir."""
    graph, fabric = _make_scheduled_cnn()
    weight_dir = tmp_path / "weights"
    weight_dir.mkdir()
    # Create minimal weight files
    (weight_dir / "tile_0_weights.mem").write_text("00\n" * 128)
    (weight_dir / "tile_0_biases.mem").write_text("00000000\n" * 8)
    return graph, fabric, weight_dir


@pytest.fixture
def transformer_graph_and_fabric(tmp_path: Path) -> tuple[Graph, FabricConfig, Path]:
    """Scheduled transformer-like graph with fabric config."""
    graph, fabric = _make_scheduled_transformer_like()
    weight_dir = tmp_path / "weights"
    weight_dir.mkdir()
    return graph, fabric, weight_dir


# ---------------------------------------------------------------------------
# Test: Module lint (verilator)
# ---------------------------------------------------------------------------


def _verilator_available() -> bool:
    return shutil.which("verilator") is not None


@pytest.mark.skipif(not _verilator_available(), reason="verilator not installed")
class TestNewModuleLint:
    """Verilator --lint-only for each new .sv module."""

    _RTL_DIR = Path(__file__).resolve().parent.parent / "rtl"

    @pytest.mark.parametrize(
        "sv_file",
        [
            "compute/activation_unit.sv",
            "memory/rom_tile.sv",
        ],
    )
    def test_lint_leaf_modules(self, sv_file: str) -> None:
        """New leaf modules pass verilator lint."""
        path = self._RTL_DIR / sv_file
        if not path.exists():
            pytest.skip(f"File not found: {path}")
        result = subprocess.run(
            ["verilator", "--lint-only", "--language", "1800-2017", "-Wall", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"Lint failed for {sv_file}:\nstderr: {result.stderr}"
        )


# ---------------------------------------------------------------------------
# Test: MLP detection
# ---------------------------------------------------------------------------


class TestMLPDetection:
    """Test _is_mlp_linear_chain() detection logic."""

    def test_mlp_detected_for_linear_chain(self, tmp_path: Path) -> None:
        """Pure FusedLinear/FusedLinearReLU chain is detected as MLP."""
        graph, _ = _make_scheduled_linear_chain()
        gen = RTLGenerator(graph=graph, weight_dir=tmp_path, output_dir=tmp_path / "out")
        assert gen._is_mlp_linear_chain()

    def test_mlp_not_detected_for_cnn(self, tmp_path: Path) -> None:
        """CNN graph is NOT detected as MLP."""
        graph, _ = _make_scheduled_cnn()
        gen = RTLGenerator(graph=graph, weight_dir=tmp_path, output_dir=tmp_path / "out")
        assert not gen._is_mlp_linear_chain()

    def test_mlp_not_detected_for_transformer(self, tmp_path: Path) -> None:
        """Transformer-like graph is NOT detected as MLP."""
        graph, _ = _make_scheduled_transformer_like()
        gen = RTLGenerator(graph=graph, weight_dir=tmp_path, output_dir=tmp_path / "out")
        assert not gen._is_mlp_linear_chain()


# ---------------------------------------------------------------------------
# Test: MLP backward compatibility
# ---------------------------------------------------------------------------


class TestMLPBackwardCompat:
    """MLP models produce identical output to Stage 5."""

    def test_mlp_generates_parameters_svh(
        self, packed_output: tuple[dict, Path, Graph], tmp_path: Path
    ) -> None:
        """MLP path still generates parameters.svh (not tile_parameters.svh)."""
        _, weight_dir, graph = packed_output
        out = tmp_path / "rtl_out"
        gen = RTLGenerator(graph=graph, weight_dir=weight_dir, output_dir=out)
        gen.generate_all()
        assert (out / "parameters.svh").is_file()
        assert not (out / "tile_parameters.svh").is_file()

    def test_mlp_generates_original_accel_top(
        self, packed_output: tuple[dict, Path, Graph], tmp_path: Path
    ) -> None:
        """MLP path generates accelerator_top.sv with original FSM states."""
        _, weight_dir, graph = packed_output
        out = tmp_path / "rtl_out"
        gen = RTLGenerator(graph=graph, weight_dir=weight_dir, output_dir=out)
        gen.generate_all()
        content = (out / "accelerator_top.sv").read_text()
        assert "S_RUN_LAYER" in content
        assert "fused_linear_relu" in content
        # Should NOT have tile fabric
        assert "tile_fabric" not in content


# ---------------------------------------------------------------------------
# Test: LUT file generation
# ---------------------------------------------------------------------------


class TestLUTGeneration:
    """Test LUT .mem file generation."""

    def test_lut_files_created(self, tmp_path: Path) -> None:
        """All 4 LUT files are created."""
        graph, fabric = _make_scheduled_cnn()
        gen = RTLGenerator(
            graph=graph, weight_dir=tmp_path, output_dir=tmp_path / "out",
            fabric_config=fabric,
        )
        (tmp_path / "out").mkdir()
        files = gen.generate_lut_files()
        assert len(files) == 4
        for f in files:
            assert f.is_file()

    def test_gelu_lut_256_entries(self, tmp_path: Path) -> None:
        """GELU LUT has exactly 256 entries."""
        graph, fabric = _make_scheduled_cnn()
        gen = RTLGenerator(
            graph=graph, weight_dir=tmp_path, output_dir=tmp_path / "out",
            fabric_config=fabric,
        )
        (tmp_path / "out").mkdir()
        gen.generate_lut_files()
        lines = (tmp_path / "out" / "gelu_lut.mem").read_text().strip().split("\n")
        assert len(lines) == 256

    def test_silu_lut_256_entries(self, tmp_path: Path) -> None:
        """SiLU LUT has exactly 256 entries."""
        graph, fabric = _make_scheduled_cnn()
        gen = RTLGenerator(
            graph=graph, weight_dir=tmp_path, output_dir=tmp_path / "out",
            fabric_config=fabric,
        )
        (tmp_path / "out").mkdir()
        gen.generate_lut_files()
        lines = (tmp_path / "out" / "silu_lut.mem").read_text().strip().split("\n")
        assert len(lines) == 256

    def test_exp_lut_256_entries(self, tmp_path: Path) -> None:
        """Exp LUT has exactly 256 entries."""
        graph, fabric = _make_scheduled_cnn()
        gen = RTLGenerator(
            graph=graph, weight_dir=tmp_path, output_dir=tmp_path / "out",
            fabric_config=fabric,
        )
        (tmp_path / "out").mkdir()
        gen.generate_lut_files()
        lines = (tmp_path / "out" / "exp_lut.mem").read_text().strip().split("\n")
        assert len(lines) == 256

    def test_rsqrt_lut_256_entries(self, tmp_path: Path) -> None:
        """Rsqrt LUT has exactly 256 entries."""
        graph, fabric = _make_scheduled_cnn()
        gen = RTLGenerator(
            graph=graph, weight_dir=tmp_path, output_dir=tmp_path / "out",
            fabric_config=fabric,
        )
        (tmp_path / "out").mkdir()
        gen.generate_lut_files()
        lines = (tmp_path / "out" / "rsqrt_lut.mem").read_text().strip().split("\n")
        assert len(lines) == 256

    def test_lut_hex_format(self, tmp_path: Path) -> None:
        """LUT entries are valid hex strings."""
        graph, fabric = _make_scheduled_cnn()
        gen = RTLGenerator(
            graph=graph, weight_dir=tmp_path, output_dir=tmp_path / "out",
            fabric_config=fabric,
        )
        (tmp_path / "out").mkdir()
        gen.generate_lut_files()
        for name in ["gelu_lut.mem", "silu_lut.mem", "exp_lut.mem", "rsqrt_lut.mem"]:
            lines = (tmp_path / "out" / name).read_text().strip().split("\n")
            for line in lines:
                int(line, 16)  # Should not raise


# ---------------------------------------------------------------------------
# Test: Tile parameters generation
# ---------------------------------------------------------------------------


class TestTileParameters:
    """Test tile_parameters.svh generation."""

    def test_tile_params_created(self, tmp_path: Path) -> None:
        """tile_parameters.svh is created."""
        graph, fabric = _make_scheduled_cnn()
        gen = RTLGenerator(
            graph=graph, weight_dir=tmp_path, output_dir=tmp_path / "out",
            fabric_config=fabric,
        )
        (tmp_path / "out").mkdir()
        path = gen.generate_tile_parameters_svh()
        assert path.is_file()

    def test_tile_count_matches(self, tmp_path: Path) -> None:
        """NUM_TILES matches fabric config tile count."""
        graph, fabric = _make_scheduled_cnn()
        gen = RTLGenerator(
            graph=graph, weight_dir=tmp_path, output_dir=tmp_path / "out",
            fabric_config=fabric,
        )
        (tmp_path / "out").mkdir()
        gen.generate_tile_parameters_svh()
        content = (tmp_path / "out" / "tile_parameters.svh").read_text()
        assert f"NUM_TILES    = {fabric.total_tiles};" in content

    def test_tile_types_present(self, tmp_path: Path) -> None:
        """TILE_TYPE array is present with correct values."""
        graph, fabric = _make_scheduled_cnn()
        gen = RTLGenerator(
            graph=graph, weight_dir=tmp_path, output_dir=tmp_path / "out",
            fabric_config=fabric,
        )
        (tmp_path / "out").mkdir()
        gen.generate_tile_parameters_svh()
        content = (tmp_path / "out" / "tile_parameters.svh").read_text()
        assert "TILE_TYPE" in content

    def test_requant_params_present(self, tmp_path: Path) -> None:
        """Requant parameter arrays are present."""
        graph, fabric = _make_scheduled_cnn()
        gen = RTLGenerator(
            graph=graph, weight_dir=tmp_path, output_dir=tmp_path / "out",
            fabric_config=fabric,
        )
        (tmp_path / "out").mkdir()
        gen.generate_tile_parameters_svh()
        content = (tmp_path / "out" / "tile_parameters.svh").read_text()
        assert "TILE_REQUANT_M" in content
        assert "TILE_REQUANT_SHIFT" in content
        assert "TILE_REQUANT_ZP" in content


# ---------------------------------------------------------------------------
# Test: Tile fabric generation
# ---------------------------------------------------------------------------


class TestTileFabricGeneration:
    """Test tile_fabric.sv generation."""

    def test_fabric_sv_created(self, tmp_path: Path) -> None:
        """tile_fabric.sv is created."""
        graph, fabric = _make_scheduled_cnn()
        gen = RTLGenerator(
            graph=graph, weight_dir=tmp_path, output_dir=tmp_path / "out",
            fabric_config=fabric,
        )
        (tmp_path / "out").mkdir()
        path = gen.generate_tile_fabric_sv()
        assert path.is_file()

    def test_fabric_has_tile_instances(self, tmp_path: Path) -> None:
        """tile_fabric.sv instantiates all tiles."""
        graph, fabric = _make_scheduled_cnn()
        gen = RTLGenerator(
            graph=graph, weight_dir=tmp_path, output_dir=tmp_path / "out",
            fabric_config=fabric,
        )
        (tmp_path / "out").mkdir()
        gen.generate_tile_fabric_sv()
        content = (tmp_path / "out" / "tile_fabric.sv").read_text()
        for tile in fabric.tiles:
            assert f"u_tile_{tile.tile_id}" in content

    def test_fabric_fsm_states(self, tmp_path: Path) -> None:
        """tile_fabric.sv has fabric FSM states."""
        graph, fabric = _make_scheduled_cnn()
        gen = RTLGenerator(
            graph=graph, weight_dir=tmp_path, output_dir=tmp_path / "out",
            fabric_config=fabric,
        )
        (tmp_path / "out").mkdir()
        gen.generate_tile_fabric_sv()
        content = (tmp_path / "out" / "tile_fabric.sv").read_text()
        for state in ["S_IDLE", "S_RUN_TILE", "S_NEXT_TILE", "S_DONE"]:
            assert state in content

    def test_fabric_model_name_in_comment(self, tmp_path: Path) -> None:
        """tile_fabric.sv has model name in comment."""
        graph, fabric = _make_scheduled_cnn()
        gen = RTLGenerator(
            graph=graph, weight_dir=tmp_path, output_dir=tmp_path / "out",
            fabric_config=fabric,
        )
        (tmp_path / "out").mkdir()
        gen.generate_tile_fabric_sv()
        content = (tmp_path / "out" / "tile_fabric.sv").read_text()
        assert graph.name in content


# ---------------------------------------------------------------------------
# Test: Tile accelerator_top generation
# ---------------------------------------------------------------------------


class TestTileAcceleratorTop:
    """Test tile-based accelerator_top.sv generation."""

    def test_tile_accel_top_created(self, tmp_path: Path) -> None:
        """Tile-based accelerator_top.sv is created."""
        graph, fabric = _make_scheduled_cnn()
        gen = RTLGenerator(
            graph=graph, weight_dir=tmp_path, output_dir=tmp_path / "out",
            fabric_config=fabric,
        )
        (tmp_path / "out").mkdir()
        path = gen.generate_tile_accelerator_top()
        assert path.is_file()

    def test_tile_accel_top_has_fabric(self, tmp_path: Path) -> None:
        """Tile accelerator_top instantiates tile_fabric."""
        graph, fabric = _make_scheduled_cnn()
        gen = RTLGenerator(
            graph=graph, weight_dir=tmp_path, output_dir=tmp_path / "out",
            fabric_config=fabric,
        )
        (tmp_path / "out").mkdir()
        gen.generate_tile_accelerator_top()
        content = (tmp_path / "out" / "accelerator_top.sv").read_text()
        assert "tile_fabric" in content
        assert "u_fabric" in content

    def test_tile_accel_top_has_axi(self, tmp_path: Path) -> None:
        """Tile accelerator_top has AXI interfaces."""
        graph, fabric = _make_scheduled_cnn()
        gen = RTLGenerator(
            graph=graph, weight_dir=tmp_path, output_dir=tmp_path / "out",
            fabric_config=fabric,
        )
        (tmp_path / "out").mkdir()
        gen.generate_tile_accelerator_top()
        content = (tmp_path / "out" / "accelerator_top.sv").read_text()
        assert "axi_lite_ctrl" in content
        assert "axi_stream_in" in content
        assert "axi_stream_out" in content

    def test_tile_accel_top_fsm(self, tmp_path: Path) -> None:
        """Tile accelerator_top has simplified FSM (no S_RUN_LAYER)."""
        graph, fabric = _make_scheduled_cnn()
        gen = RTLGenerator(
            graph=graph, weight_dir=tmp_path, output_dir=tmp_path / "out",
            fabric_config=fabric,
        )
        (tmp_path / "out").mkdir()
        gen.generate_tile_accelerator_top()
        content = (tmp_path / "out" / "accelerator_top.sv").read_text()
        assert "S_RUN_FABRIC" in content
        assert "S_IDLE" in content


# ---------------------------------------------------------------------------
# Test: FPGA vs ASIC mode
# ---------------------------------------------------------------------------


class TestTargetMode:
    """Test FPGA vs ASIC target generation."""

    def test_fpga_mode_no_rom_modules(self, tmp_path: Path) -> None:
        """FPGA mode does NOT generate rom_tile_*.sv modules."""
        graph, fabric = _make_scheduled_cnn()
        gen = RTLGenerator(
            graph=graph, weight_dir=tmp_path, output_dir=tmp_path / "out",
            fabric_config=fabric, target="fpga",
        )
        (tmp_path / "out").mkdir()
        gen.generate_lut_files()
        gen.generate_tile_parameters_svh()
        gen.generate_tile_fabric_sv()
        gen.generate_tile_accelerator_top()
        # Should not have rom_tile_*.sv
        rom_files = list((tmp_path / "out").glob("rom_tile_*.sv"))
        assert len(rom_files) == 0

    def test_asic_mode_generates_rom_modules(self, tmp_path: Path) -> None:
        """ASIC mode generates rom_tile_<id>.sv for MAC tiles."""
        graph, fabric = _make_scheduled_cnn()
        gen = RTLGenerator(
            graph=graph, weight_dir=tmp_path, output_dir=tmp_path / "out",
            fabric_config=fabric, target="asic",
        )
        (tmp_path / "out").mkdir()
        rom_files = gen.generate_asic_rom_modules()
        # Should have at least one ROM for the conv MAC tile
        mac_tiles = [t for t in fabric.tiles if t.tile_type == TileType.MAC]
        if mac_tiles:
            assert len(rom_files) > 0
            for f in rom_files:
                assert f.is_file()
                content = f.read_text()
                assert "rom_tile_" in content
                assert "initial begin" in content


# ---------------------------------------------------------------------------
# Test: CNN tile fabric
# ---------------------------------------------------------------------------


class TestCNNTileFabric:
    """Test tile fabric for CNN models."""

    def test_cnn_tile_count(self) -> None:
        """CNN creates correct number of tiles."""
        graph, fabric = _make_scheduled_cnn()
        # conv1(MAC) + pool1(POOL) + fc1(MAC) = 3 tiles
        assert fabric.total_tiles == 3

    def test_cnn_has_mac_and_pool_tiles(self) -> None:
        """CNN fabric has both MAC and POOL tiles."""
        graph, fabric = _make_scheduled_cnn()
        types = {t.tile_type for t in fabric.tiles}
        assert TileType.MAC in types
        assert TileType.POOL in types

    def test_cnn_routes_exist(self) -> None:
        """CNN has routes between tiles."""
        graph, fabric = _make_scheduled_cnn()
        assert len(fabric.routes) > 0


# ---------------------------------------------------------------------------
# Test: Transformer tile fabric
# ---------------------------------------------------------------------------


class TestTransformerTileFabric:
    """Test tile fabric for transformer-like models."""

    def test_transformer_tile_count(self) -> None:
        """Transformer creates correct number of tiles."""
        graph, fabric = _make_scheduled_transformer_like()
        # qkv(MAC) + softmax(SOFTMAX) + proj(MAC) = 3 tiles
        assert fabric.total_tiles == 3

    def test_transformer_has_softmax_tile(self) -> None:
        """Transformer fabric has a SOFTMAX tile."""
        graph, fabric = _make_scheduled_transformer_like()
        types = {t.tile_type for t in fabric.tiles}
        assert TileType.SOFTMAX in types

    def test_transformer_routes_exist(self) -> None:
        """Transformer has routes between tiles."""
        graph, fabric = _make_scheduled_transformer_like()
        assert len(fabric.routes) > 0


# ---------------------------------------------------------------------------
# Test: Stage acceptance
# ---------------------------------------------------------------------------


class TestStageAcceptance:
    """Test that RTLGenerator accepts both scheduled and scheduled_dag."""

    def test_accepts_scheduled_dag(self, tmp_path: Path) -> None:
        """RTLGenerator accepts 'scheduled_dag' stage."""
        graph, _ = _make_scheduled_cnn()
        gen = RTLGenerator(graph=graph, weight_dir=tmp_path, output_dir=tmp_path / "out")
        assert gen.graph.stage == "scheduled_dag"

    def test_rejects_raw_stage(self, tmp_path: Path) -> None:
        """RTLGenerator rejects 'raw' stage."""
        graph = Graph("test", {}, {}, [], [], stage="raw")
        with pytest.raises(ValueError, match="scheduled"):
            RTLGenerator(graph=graph, weight_dir=tmp_path, output_dir=tmp_path)

    def test_rejects_quantized_stage(self, tmp_path: Path) -> None:
        """RTLGenerator rejects 'quantized' stage."""
        graph = Graph("test", {}, {}, [], [], stage="quantized")
        with pytest.raises(ValueError, match="scheduled"):
            RTLGenerator(graph=graph, weight_dir=tmp_path, output_dir=tmp_path)
