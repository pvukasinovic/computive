"""MLASIC Stage 4: Weight Packing & Memory Layout.

Packs quantized weights and biases into .mem files matching the SRAM layout
expected by RTL, and generates a memory_map.json describing the layout.

SRAM organization (from rtl-interface-spec.md §5.5):
  - WEIGHT_BANK: 1536 rows × 1024 bits (128 INT8 per row)
  - BIAS_BANK:   32 rows × 1024 bits (32 INT32 per row)

RTL addressing (from rtl-interface-spec.md §6):
  weight_addr = WEIGHT_BASE[layer] + tile_idx * IN_DIM + input_idx
  bias_addr   = BIAS_BASE[layer] + tile_idx * 4 + bias_row_idx

.mem files use $readmemh format: one hex line per SRAM row, 256 hex digits
(1024 bits), MSB-first. Internal byte order is little-endian (byte 0 at LSB).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mlasic.ir import (
    FusedAttentionAttrs,
    FusedConvAttrs,
    Graph,
    HardwareConstraints,
    OpType,
    QuantParams,
    Tensor,
    TensorType,
)

logger = logging.getLogger(__name__)


@dataclass
class LayerMemoryInfo:
    """Memory layout info for one layer, used in memory_map.json."""

    layer_index: int
    layer_name: str
    input_dim: int
    output_dim: int
    num_tiles: int
    parallelism: int
    weight_start_row: int
    weight_rows: int
    weight_size_bytes: int
    bias_start_row: int
    bias_rows: int
    bias_size_bytes: int
    total_size_bytes: int


class WeightPacker:
    """Pack quantized weights and biases into .mem files for RTL synthesis.

    Takes a scheduled graph (stage="scheduled") and generates:
      - Per-layer weight/bias .mem files
      - Combined weight_bank.mem and bias_bank.mem for $readmemh
      - memory_map.json with SRAM addresses and sizes
    """

    def __init__(
        self,
        graph: Graph,
        output_dir: Path | str,
        constraints: HardwareConstraints | None = None,
    ) -> None:
        if graph.stage != "scheduled":
            raise ValueError(f"WeightPacker requires 'scheduled' stage, got '{graph.stage}'")
        self.graph = graph
        self.output_dir = Path(output_dir)
        self.constraints = constraints or HardwareConstraints()

    def pack_weights(self) -> dict:
        """Pack all weights/biases to .mem files and write memory_map.json.

        Returns:
            Memory map dict (also written to memory_map.json).
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ordered_names = self.graph.topological_order()
        ordered_nodes = [self.graph.nodes[n] for n in ordered_names]

        all_weight_rows: list[np.ndarray] = []
        all_bias_rows: list[np.ndarray] = []
        layer_infos: list[LayerMemoryInfo] = []

        for node in ordered_nodes:
            ls = node.schedule_info
            attrs = node.fused_attrs

            # Extract quantized weight tensor — must be INT8
            weight_tensor = self.graph.tensors[node.inputs[1]]
            w_q = weight_tensor.data
            if w_q.dtype != np.int8:
                raise ValueError(f"Layer {ls.layer_name}: expected INT8 weights, got {w_q.dtype}")

            # Extract quantized bias tensor — must be INT32
            if len(node.inputs) > 2:
                bias_tensor = self.graph.tensors[node.inputs[2]]
                b_q = bias_tensor.data
                if b_q.dtype != np.int32:
                    raise ValueError(f"Layer {ls.layer_name}: expected INT32 bias, got {b_q.dtype}")
            else:
                b_q = np.zeros(attrs.output_dim, dtype=np.int32)

            # Pack weight rows for RTL SRAM layout
            layer_weight_rows = self._pack_layer_weights(
                w_q, ls.num_tiles, ls.parallelism, attrs.input_dim
            )

            # Pack bias rows
            layer_bias_rows = self._pack_layer_biases(b_q)

            # Sanity checks against schedule
            if len(layer_weight_rows) != ls.weight_rows:
                raise ValueError(
                    f"Layer {ls.layer_name}: packed {len(layer_weight_rows)} weight rows, "
                    f"schedule expects {ls.weight_rows}"
                )
            expected_bias_rows = math.ceil(attrs.output_dim / self.constraints.biases_per_row)
            if len(layer_bias_rows) != expected_bias_rows:
                raise ValueError(
                    f"Layer {ls.layer_name}: packed {len(layer_bias_rows)} bias rows, "
                    f"expected {expected_bias_rows}"
                )

            # Write per-layer .mem files
            weight_file = self.output_dir / f"weights_layer{ls.layer_index}.mem"
            bias_file = self.output_dir / f"biases_layer{ls.layer_index}.mem"
            _write_int8_mem(layer_weight_rows, weight_file)
            _write_int32_mem(layer_bias_rows, bias_file)

            all_weight_rows.extend(layer_weight_rows)
            all_bias_rows.extend(layer_bias_rows)

            layer_infos.append(
                LayerMemoryInfo(
                    layer_index=ls.layer_index,
                    layer_name=ls.layer_name,
                    input_dim=attrs.input_dim,
                    output_dim=attrs.output_dim,
                    num_tiles=ls.num_tiles,
                    parallelism=ls.parallelism,
                    weight_start_row=ls.weight_start_row,
                    weight_rows=ls.weight_rows,
                    weight_size_bytes=ls.weight_bytes,
                    bias_start_row=ls.bias_start_row,
                    bias_rows=ls.bias_rows,
                    bias_size_bytes=ls.bias_bytes,
                    total_size_bytes=ls.weight_bytes + ls.bias_bytes,
                )
            )

            logger.info(
                "Layer %d (%s): %d weight rows, %d bias rows, %d bytes total",
                ls.layer_index,
                ls.layer_name,
                len(layer_weight_rows),
                len(layer_bias_rows),
                ls.weight_bytes + ls.bias_bytes,
            )

        # Write combined bank .mem files
        _write_int8_mem(all_weight_rows, self.output_dir / "weight_bank.mem")
        _write_int32_mem(all_bias_rows, self.output_dir / "bias_bank.mem")

        # Generate and write memory map
        memory_map = _generate_memory_map(layer_infos)
        map_path = self.output_dir / "memory_map.json"
        with open(map_path, "w") as f:
            json.dump(memory_map, f, indent=2)

        logger.info(
            "Weight packing complete: %d layers, %d weight rows, %d bias rows, %d total bytes",
            len(layer_infos),
            len(all_weight_rows),
            len(all_bias_rows),
            memory_map["total_bytes"],
        )

        return memory_map

    def _pack_layer_weights(
        self,
        w_q: np.ndarray,
        num_tiles: int,
        parallelism: int,
        input_dim: int,
    ) -> list[np.ndarray]:
        """Pack weight matrix into SRAM rows matching RTL access pattern.

        RTL addressing: weight_addr = BASE + tile_idx * IN_DIM + input_idx
        Each row: W[input_idx, tile_start : tile_start + parallelism]
        """
        rows: list[np.ndarray] = []
        for tile in range(num_tiles):
            col_start = tile * parallelism
            col_end = col_start + parallelism
            for i in range(input_dim):
                row_data = w_q[i, col_start:col_end]
                if len(row_data) < parallelism:
                    padded = np.zeros(parallelism, dtype=np.int8)
                    padded[: len(row_data)] = row_data
                    row_data = padded
                rows.append(row_data)
        return rows

    def _pack_layer_biases(self, b_q: np.ndarray) -> list[np.ndarray]:
        """Pack bias vector into SRAM rows (32 INT32 values per row)."""
        biases_per_row = self.constraints.biases_per_row
        num_rows = math.ceil(len(b_q) / biases_per_row)
        rows: list[np.ndarray] = []
        for r in range(num_rows):
            start = r * biases_per_row
            end = start + biases_per_row
            row_data = b_q[start:end]
            if len(row_data) < biases_per_row:
                padded = np.zeros(biases_per_row, dtype=np.int32)
                padded[: len(row_data)] = row_data
                row_data = padded
            rows.append(row_data)
        return rows


# ---------------------------------------------------------------------------
# .mem file I/O
# ---------------------------------------------------------------------------


def _write_int8_mem(rows: list[np.ndarray], filepath: Path) -> None:
    """Write INT8 weight rows to $readmemh-compatible .mem file.

    Each row: 128 INT8 values → 256 hex digits per line.
    Little-endian byte order: byte[0] at LSB, MSB-first hex string.
    """
    with open(filepath, "w") as f:
        for row in rows:
            row_uint = row.view(np.uint8)
            hex_line = "".join(f"{b:02x}" for b in reversed(row_uint))
            f.write(hex_line + "\n")


def _write_int32_mem(rows: list[np.ndarray], filepath: Path) -> None:
    """Write INT32 bias rows to $readmemh-compatible .mem file.

    Each row: 32 INT32 values → 128 bytes → 256 hex digits per line.
    Little-endian byte order within each INT32 and across the row.
    """
    with open(filepath, "w") as f:
        for row in rows:
            row_bytes = row.astype(np.int32).view(np.uint8)
            hex_line = "".join(f"{b:02x}" for b in reversed(row_bytes))
            f.write(hex_line + "\n")


def load_weight_mem(filepath: Path, values_per_row: int = 128) -> list[np.ndarray]:
    """Load weight .mem file back into list of INT8 row arrays.

    Inverse of _write_int8_mem — for roundtrip verification.
    """
    rows: list[np.ndarray] = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Parse hex pairs, MSB first
            bytes_msb = np.array(
                [int(line[i : i + 2], 16) for i in range(0, len(line), 2)],
                dtype=np.uint8,
            )
            # Reverse to get LSB-first (byte 0 at index 0)
            row_uint = bytes_msb[::-1].copy()
            rows.append(row_uint[:values_per_row].view(np.int8).copy())
    return rows


def load_bias_mem(filepath: Path, values_per_row: int = 32) -> list[np.ndarray]:
    """Load bias .mem file back into list of INT32 row arrays.

    Inverse of _write_int32_mem — for roundtrip verification.
    """
    rows: list[np.ndarray] = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            bytes_msb = np.array(
                [int(line[i : i + 2], 16) for i in range(0, len(line), 2)],
                dtype=np.uint8,
            )
            row_bytes = bytes_msb[::-1].copy()
            row_int32 = row_bytes.view(np.int32).copy()
            rows.append(row_int32[:values_per_row])
    return rows


# ---------------------------------------------------------------------------
# Memory map generation
# ---------------------------------------------------------------------------


def _generate_memory_map(layer_infos: list[LayerMemoryInfo]) -> dict:
    """Generate memory map dict for JSON serialization."""
    layers = []
    total_weight_bytes = 0
    total_bias_bytes = 0

    for info in layer_infos:
        layers.append(
            {
                "layer_index": info.layer_index,
                "layer_name": info.layer_name,
                "input_dim": info.input_dim,
                "output_dim": info.output_dim,
                "num_tiles": info.num_tiles,
                "parallelism": info.parallelism,
                "weight": {
                    "sram_bank": "WEIGHT_BANK",
                    "start_row": info.weight_start_row,
                    "num_rows": info.weight_rows,
                    "size_bytes": info.weight_size_bytes,
                },
                "bias": {
                    "sram_bank": "BIAS_BANK",
                    "start_row": info.bias_start_row,
                    "num_rows": info.bias_rows,
                    "size_bytes": info.bias_size_bytes,
                },
                "total_size_bytes": info.total_size_bytes,
            }
        )
        total_weight_bytes += info.weight_size_bytes
        total_bias_bytes += info.bias_size_bytes

    return {
        "weight_bank": {
            "total_rows": sum(info.weight_rows for info in layer_infos),
            "row_width_bytes": 128,
            "total_bytes": total_weight_bytes,
        },
        "bias_bank": {
            "total_rows": sum(info.bias_rows for info in layer_infos),
            "row_width_bytes": 128,
            "total_bytes": total_bias_bytes,
        },
        "total_bytes": total_weight_bytes + total_bias_bytes,
        "layers": layers,
    }


# ---------------------------------------------------------------------------
# Conv Weight Packing
# ---------------------------------------------------------------------------


class ConvWeightPacker:
    """Pack conv weight tensors into SRAM rows matching RTL access pattern.

    Layout: [OC_tile × IC × KH × KW] packed per tile's MAC array access pattern.
    Supports standard, depthwise, and grouped convolutions.
    """

    def __init__(self, parallelism: int = 128, row_bytes: int = 128):
        self.parallelism = parallelism
        self.row_bytes = row_bytes

    def pack_conv_weights(
        self,
        w_q: np.ndarray,
        attrs: FusedConvAttrs,
    ) -> list[np.ndarray]:
        """Pack conv weight tensor into SRAM rows.

        Args:
            w_q: INT8 weight tensor, shape [OC, IC/group, KH, KW].
            attrs: FusedConvAttrs with kernel/stride/group info.

        Returns:
            List of INT8 row arrays (each `row_bytes` wide).
        """
        oc = w_q.shape[0]
        group = attrs.group
        ic_per_group = w_q.shape[1]
        kernel_shape = attrs.kernel_shape

        parallelism = min(self.parallelism, oc)
        # Adjust parallelism to evenly divide OC
        while parallelism > 1 and oc % parallelism != 0:
            parallelism -= 1
        num_oc_tiles = math.ceil(oc / parallelism)

        rows: list[np.ndarray] = []

        if group == oc and ic_per_group == 1:
            # Depthwise: each OC uses 1 IC, pack kernel directly
            return self._pack_depthwise(w_q, num_oc_tiles, parallelism, kernel_shape)

        # Standard / grouped conv
        for tile in range(num_oc_tiles):
            oc_start = tile * parallelism
            oc_end = min(oc_start + parallelism, oc)
            tile_width = oc_end - oc_start

            # For each input position (IC × KH × KW), pack `parallelism` OC values
            for ic in range(ic_per_group):
                for kh in range(kernel_shape[0]):
                    kw_range = kernel_shape[1] if len(kernel_shape) > 1 else 1
                    for kw in range(kw_range):
                        row = np.zeros(self.row_bytes, dtype=np.int8)
                        for i in range(tile_width):
                            if len(kernel_shape) > 1:
                                row[i] = w_q[oc_start + i, ic, kh, kw]
                            else:
                                row[i] = w_q[oc_start + i, ic, kh]
                        rows.append(row)

        return rows

    def _pack_depthwise(
        self,
        w_q: np.ndarray,
        num_oc_tiles: int,
        parallelism: int,
        kernel_shape: list[int],
    ) -> list[np.ndarray]:
        """Pack depthwise conv weights: each channel has its own kernel."""
        oc = w_q.shape[0]
        rows: list[np.ndarray] = []

        for tile in range(num_oc_tiles):
            oc_start = tile * parallelism
            oc_end = min(oc_start + parallelism, oc)
            tile_width = oc_end - oc_start

            for kh in range(kernel_shape[0]):
                kw_range = kernel_shape[1] if len(kernel_shape) > 1 else 1
                for kw in range(kw_range):
                    row = np.zeros(self.row_bytes, dtype=np.int8)
                    for i in range(tile_width):
                        if len(kernel_shape) > 1:
                            row[i] = w_q[oc_start + i, 0, kh, kw]
                        else:
                            row[i] = w_q[oc_start + i, 0, kh]
                    rows.append(row)

        return rows

    def unpack_conv_weights(
        self,
        rows: list[np.ndarray],
        oc: int,
        ic_per_group: int,
        kernel_shape: list[int],
        group: int = 1,
    ) -> np.ndarray:
        """Unpack conv weight rows back into [OC, IC/group, KH, KW] tensor."""
        parallelism = min(self.parallelism, oc)
        while parallelism > 1 and oc % parallelism != 0:
            parallelism -= 1
        num_oc_tiles = math.ceil(oc / parallelism)

        kh = kernel_shape[0]
        kw = kernel_shape[1] if len(kernel_shape) > 1 else 1

        if len(kernel_shape) > 1:
            w_q = np.zeros((oc, ic_per_group, kh, kw), dtype=np.int8)
        else:
            w_q = np.zeros((oc, ic_per_group, kh), dtype=np.int8)

        is_depthwise = (group == oc and ic_per_group == 1)
        row_idx = 0

        for tile in range(num_oc_tiles):
            oc_start = tile * parallelism
            oc_end = min(oc_start + parallelism, oc)
            tile_width = oc_end - oc_start

            if is_depthwise:
                for ikh in range(kh):
                    for ikw in range(kw):
                        for i in range(tile_width):
                            if len(kernel_shape) > 1:
                                w_q[oc_start + i, 0, ikh, ikw] = rows[row_idx][i]
                            else:
                                w_q[oc_start + i, 0, ikh] = rows[row_idx][i]
                        row_idx += 1
            else:
                for ic in range(ic_per_group):
                    for ikh in range(kh):
                        for ikw in range(kw):
                            for i in range(tile_width):
                                if len(kernel_shape) > 1:
                                    w_q[oc_start + i, ic, ikh, ikw] = rows[row_idx][i]
                                else:
                                    w_q[oc_start + i, ic, ikh] = rows[row_idx][i]
                            row_idx += 1

        return w_q


# ---------------------------------------------------------------------------
# Attention Weight Packing
# ---------------------------------------------------------------------------


class AttentionWeightPacker:
    """Pack attention Q/K/V/O projection weights into .mem files."""

    def __init__(self, parallelism: int = 128, row_bytes: int = 128):
        self.parallelism = parallelism
        self.row_bytes = row_bytes

    def pack_attention_weights(
        self,
        q_weight: np.ndarray,
        k_weight: np.ndarray,
        v_weight: np.ndarray,
        o_weight: np.ndarray,
        biases: dict[str, np.ndarray] | None = None,
    ) -> dict[str, list[np.ndarray]]:
        """Pack Q/K/V/O weights into SRAM rows.

        Args:
            q_weight, k_weight, v_weight, o_weight: INT8 [in, out] matrices.
            biases: Optional dict {"q": bias_q, "k": bias_k, ...} as INT32.

        Returns:
            Dict mapping projection name to packed rows.
        """
        biases = biases or {}
        result = {}
        for name, w in [("q", q_weight), ("k", k_weight), ("v", v_weight), ("o", o_weight)]:
            rows = self._pack_linear_weights(w)
            result[f"{name}_weight"] = rows
            if name in biases:
                b_rows = self._pack_biases(biases[name])
                result[f"{name}_bias"] = b_rows
        return result

    def _pack_linear_weights(self, w_q: np.ndarray) -> list[np.ndarray]:
        """Pack 2D INT8 weight matrix [input_dim, output_dim] into SRAM rows."""
        input_dim, output_dim = w_q.shape
        parallelism = min(self.parallelism, output_dim)
        while parallelism > 1 and output_dim % parallelism != 0:
            parallelism -= 1
        num_tiles = math.ceil(output_dim / parallelism)

        rows: list[np.ndarray] = []
        for tile in range(num_tiles):
            col_start = tile * parallelism
            col_end = col_start + parallelism
            for i in range(input_dim):
                row = np.zeros(self.row_bytes, dtype=np.int8)
                chunk = w_q[i, col_start:min(col_end, output_dim)]
                row[: len(chunk)] = chunk
                rows.append(row)
        return rows

    def _pack_biases(self, b_q: np.ndarray) -> list[np.ndarray]:
        """Pack INT32 bias vector into rows (32 biases per row)."""
        biases_per_row = 32
        num_rows = math.ceil(len(b_q) / biases_per_row)
        rows: list[np.ndarray] = []
        for r in range(num_rows):
            start = r * biases_per_row
            end = start + biases_per_row
            row = np.zeros(biases_per_row, dtype=np.int32)
            chunk = b_q[start:end]
            row[: len(chunk)] = chunk
            rows.append(row)
        return rows

    def write_attention_mem(
        self,
        packed: dict[str, list[np.ndarray]],
        output_dir: Path,
        layer_name: str,
    ) -> dict[str, str]:
        """Write packed attention weights to .mem files.

        Returns dict mapping projection name to file path.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        file_map = {}
        for proj_name, rows in packed.items():
            filename = f"{layer_name}_{proj_name}.mem"
            filepath = output_dir / filename
            if "bias" in proj_name:
                _write_int32_mem(rows, filepath)
            else:
                _write_int8_mem(rows, filepath)
            file_map[proj_name] = str(filepath)
        return file_map


# ---------------------------------------------------------------------------
# INT4 Quantization + Packing
# ---------------------------------------------------------------------------


class INT4Quantizer:
    """INT4 symmetric weight quantization (4-bit, signed, range [-7, 7]).

    Supports mixed precision: INT4 weights + INT8 activations.
    Two INT4 values are packed per byte.
    """

    @staticmethod
    def quantize_weights(
        w_fp: np.ndarray,
        per_channel: bool = True,
    ) -> tuple[np.ndarray, list[QuantParams]]:
        """Quantize FP32 weights to INT4 (symmetric, zero_point=0).

        Args:
            w_fp: FP32 weight tensor, any shape. First dim is output channels.
            per_channel: If True, one scale per output channel.

        Returns:
            (w_q_int4, quant_params) where w_q_int4 is int8 array
            with values in [-7, 7], and quant_params has bit_width=4.
        """
        w = w_fp.astype(np.float32)
        oc = w.shape[0]
        w_flat = w.reshape(oc, -1)
        w_q = np.zeros_like(w_flat, dtype=np.int8)
        params: list[QuantParams] = []

        if per_channel:
            for c in range(oc):
                max_abs = float(max(abs(w_flat[c].min()), abs(w_flat[c].max())))
                if max_abs == 0:
                    max_abs = 1e-8
                scale = max_abs / 7.0
                q = np.floor(w_flat[c] / scale + 0.5)
                w_q[c] = np.clip(q, -7, 7).astype(np.int8)
                params.append(QuantParams(
                    scale=scale, zero_point=0, bit_width=4, signed=True, calibrated=True
                ))
        else:
            max_abs = float(max(abs(w.min()), abs(w.max())))
            if max_abs == 0:
                max_abs = 1e-8
            scale = max_abs / 7.0
            q = np.floor(w_flat / scale + 0.5)
            w_q = np.clip(q, -7, 7).astype(np.int8)
            params = [QuantParams(
                scale=scale, zero_point=0, bit_width=4, signed=True, calibrated=True
            )] * oc

        return w_q.reshape(w.shape), params

    @staticmethod
    def pack_int4(values: np.ndarray) -> np.ndarray:
        """Pack INT4 values (in int8 array) into bytes: 2 values per byte.

        Packing: byte = (high_nibble << 4) | (low_nibble & 0x0F)
        Values must be in [-7, 7] (stored as 4-bit signed).
        """
        flat = values.flatten().astype(np.int8)
        # Pad to even length
        if len(flat) % 2 != 0:
            flat = np.append(flat, np.int8(0))

        packed = np.zeros(len(flat) // 2, dtype=np.uint8)
        for i in range(len(packed)):
            lo = flat[2 * i] & 0x0F
            hi = flat[2 * i + 1] & 0x0F
            packed[i] = (hi << 4) | lo

        return packed

    @staticmethod
    def unpack_int4(packed: np.ndarray, count: int) -> np.ndarray:
        """Unpack bytes back to INT4 values (in int8 array).

        Args:
            packed: Packed uint8 array.
            count: Number of INT4 values to extract.
        """
        result = np.zeros(count, dtype=np.int8)
        for i in range(min(count, len(packed) * 2)):
            byte_idx = i // 2
            if i % 2 == 0:
                nibble = packed[byte_idx] & 0x0F
            else:
                nibble = (packed[byte_idx] >> 4) & 0x0F
            # Sign-extend 4-bit to 8-bit
            if nibble & 0x08:
                result[i] = np.int8(nibble | 0xF0)
            else:
                result[i] = np.int8(nibble)
        return result

    def pack_int4_rows(
        self,
        w_q_int4: np.ndarray,
        parallelism: int = 128,
        row_bytes: int = 128,
    ) -> list[np.ndarray]:
        """Pack INT4 weight matrix into SRAM rows (256 INT4 values per 128-byte row)."""
        if w_q_int4.ndim == 1:
            w_q_int4 = w_q_int4.reshape(1, -1)

        input_dim = w_q_int4.shape[0]
        output_dim = w_q_int4.shape[1] if w_q_int4.ndim > 1 else 1

        par = min(parallelism, output_dim)
        while par > 1 and output_dim % par != 0:
            par -= 1
        num_tiles = math.ceil(output_dim / par)

        # 256 INT4 values per row (128 bytes × 2 nibbles/byte)
        values_per_row = row_bytes * 2

        rows: list[np.ndarray] = []
        for tile in range(num_tiles):
            col_start = tile * par
            col_end = min(col_start + par, output_dim)
            for i in range(input_dim):
                chunk = w_q_int4[i, col_start:col_end]
                # Pad to values_per_row
                padded = np.zeros(values_per_row, dtype=np.int8)
                padded[: len(chunk)] = chunk
                packed = self.pack_int4(padded)
                rows.append(packed.view(np.int8))
        return rows


# ---------------------------------------------------------------------------
# Per-Tile Weight Files
# ---------------------------------------------------------------------------


class PerTileWeightWriter:
    """Generate per-tile .mem files and a tile_weight_map.json.

    For ASIC flow: each hardware tile gets its own ROM image.
    """

    def __init__(self, output_dir: Path | str, row_bytes: int = 128):
        self.output_dir = Path(output_dir)
        self.row_bytes = row_bytes

    def write_per_tile_files(
        self,
        layer_name: str,
        w_q: np.ndarray,
        parallelism: int,
    ) -> dict:
        """Split a weight matrix into per-tile .mem files.

        Args:
            layer_name: Name prefix for files.
            w_q: INT8 weight matrix [input_dim, output_dim].
            parallelism: Number of output channels per tile.

        Returns:
            Tile map dict: {tile_id: {file, start_row, num_rows, rom_address}}
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        input_dim = w_q.shape[0]
        output_dim = w_q.shape[1] if w_q.ndim > 1 else 1

        par = min(parallelism, output_dim)
        while par > 1 and output_dim % par != 0:
            par -= 1
        num_tiles = math.ceil(output_dim / par)

        tile_map = {}
        rom_address = 0

        for tile in range(num_tiles):
            col_start = tile * par
            col_end = min(col_start + par, output_dim)

            rows: list[np.ndarray] = []
            for i in range(input_dim):
                row = np.zeros(self.row_bytes, dtype=np.int8)
                chunk = w_q[i, col_start:col_end]
                row[: len(chunk)] = chunk
                rows.append(row)

            filename = f"{layer_name}_tile{tile}.mem"
            filepath = self.output_dir / filename
            _write_int8_mem(rows, filepath)

            tile_map[f"tile_{tile}"] = {
                "file": str(filepath),
                "start_row": 0,
                "num_rows": len(rows),
                "rom_address": rom_address,
                "oc_range": [col_start, col_end],
            }
            rom_address += len(rows) * self.row_bytes

        # Write tile map
        map_path = self.output_dir / f"{layer_name}_tile_weight_map.json"
        with open(map_path, "w") as f:
            json.dump(tile_map, f, indent=2)

        return tile_map

    def write_per_tile_binary(
        self,
        layer_name: str,
        w_q: np.ndarray,
        parallelism: int,
    ) -> dict:
        """Generate per-tile ROM images in raw binary format (for ASIC flow)."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        input_dim = w_q.shape[0]
        output_dim = w_q.shape[1] if w_q.ndim > 1 else 1

        par = min(parallelism, output_dim)
        while par > 1 and output_dim % par != 0:
            par -= 1
        num_tiles = math.ceil(output_dim / par)

        tile_map = {}

        for tile in range(num_tiles):
            col_start = tile * par
            col_end = min(col_start + par, output_dim)

            # Build raw binary: concatenated row data
            raw_data = bytearray()
            for i in range(input_dim):
                row = np.zeros(self.row_bytes, dtype=np.int8)
                chunk = w_q[i, col_start:col_end]
                row[: len(chunk)] = chunk
                raw_data.extend(row.view(np.uint8).tobytes())

            filename = f"{layer_name}_tile{tile}.bin"
            filepath = self.output_dir / filename
            with open(filepath, "wb") as f:
                f.write(raw_data)

            tile_map[f"tile_{tile}"] = {
                "file": str(filepath),
                "size_bytes": len(raw_data),
                "oc_range": [col_start, col_end],
            }

        return tile_map
