"""MLASIC ROM Mapper — generate per-tile weight ROM images for ASIC.

Extracts weights from the graph, packs them per tile's access pattern,
and writes binary ROM images. Also provides die area estimation and
weight compression.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from mlasic.ir import Graph
from mlasic.tile_mapper import FabricConfig, TileType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WeightROMMapper
# ---------------------------------------------------------------------------


class WeightROMMapper:
    """Generate per-tile weight ROM binary images.

    For each MAC tile in FabricConfig, extract the weight tensor from
    the graph, pack per the tile's access pattern, and write binary files.
    """

    def generate(
        self,
        graph: Graph,
        fabric: FabricConfig,
        output_dir: Optional[Path] = None,
        parallelism: int = 128,
    ) -> dict:
        """Generate ROM images for all MAC tiles.

        Args:
            graph: The scheduled graph with weight tensors.
            fabric: Tile fabric configuration.
            output_dir: Directory to write binary files. If None, returns
                        data in-memory only.
            parallelism: MAC array parallelism (used for .mem row width).

        Returns:
            Manifest dict with per-tile ROM info.
        """
        manifest: dict = {"tiles": [], "total_weight_bytes": 0, "total_bias_bytes": 0}

        for tile in fabric.tiles:
            if tile.tile_type != TileType.MAC:
                continue

            node_name = tile.operator_assignment
            node = graph.nodes.get(node_name)
            if node is None:
                continue

            weight_data = None
            bias_data = None

            # Extract weight tensor (inputs[1]) and bias (inputs[2])
            if len(node.inputs) >= 2:
                wt = graph.tensors.get(node.inputs[1])
                if wt and wt.is_constant and wt.data is not None:
                    weight_data = wt.data

            if len(node.inputs) >= 3:
                bt = graph.tensors.get(node.inputs[2])
                if bt and bt.is_constant and bt.data is not None:
                    bias_data = bt.data

            # Apply partitioning if needed
            if tile.partition_info and weight_data is not None:
                start, end = tile.partition_info["output_range"]
                # Slice along output dimension (dim 0 for weights)
                weight_data = weight_data[start:end]
                if bias_data is not None:
                    bias_data = bias_data[start:end]

            tile_info = {
                "tile_id": tile.tile_id,
                "node_name": node_name,
                "weight_bytes": weight_data.nbytes if weight_data is not None else 0,
                "bias_bytes": bias_data.nbytes if bias_data is not None else 0,
            }

            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)

                if weight_data is not None:
                    # Binary file
                    w_path = output_dir / f"tile_{tile.tile_id}_weights.bin"
                    weight_data.tofile(str(w_path))
                    tile_info["weight_file"] = str(w_path)

                    # .mem file for FPGA ($readmemh)
                    w_mem_path = output_dir / f"tile_{tile.tile_id}_weights.mem"
                    self._write_weight_mem(w_mem_path, weight_data, parallelism)
                    tile_info["weight_mem_file"] = str(w_mem_path)

                if bias_data is not None:
                    # Binary file
                    b_path = output_dir / f"tile_{tile.tile_id}_biases.bin"
                    bias_data.tofile(str(b_path))
                    tile_info["bias_file"] = str(b_path)

                    # .mem file for FPGA ($readmemh)
                    b_mem_path = output_dir / f"tile_{tile.tile_id}_biases.mem"
                    self._write_bias_mem(b_mem_path, bias_data)
                    tile_info["bias_mem_file"] = str(b_mem_path)

            manifest["tiles"].append(tile_info)
            manifest["total_weight_bytes"] += tile_info["weight_bytes"]
            manifest["total_bias_bytes"] += tile_info["bias_bytes"]

        if output_dir is not None:
            manifest_path = output_dir / "rom_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            logger.info("ROM manifest written to %s", manifest_path)

        return manifest

    @staticmethod
    def _write_weight_mem(
        path: Path,
        weight_data: np.ndarray,
        parallelism: int = 128,
    ) -> None:
        """Write weight data as .mem file for $readmemh.

        Each row contains PARALLELISM INT8 values packed into hex.
        Row width = PARALLELISM * 2 hex chars (256 hex chars for P=128).
        Byte ordering: byte 0 in LSB position (LE), MSB-first hex.
        """
        flat = weight_data.flatten().astype(np.int8)
        row_width = parallelism
        num_rows = math.ceil(len(flat) / row_width)

        lines = []
        for row in range(num_rows):
            start = row * row_width
            end = min(start + row_width, len(flat))
            row_bytes = flat[start:end]
            # Pad if needed
            if len(row_bytes) < row_width:
                row_bytes = np.concatenate(
                    [row_bytes, np.zeros(row_width - len(row_bytes), dtype=np.int8)]
                )
            # LE byte order: byte[0] at LSB → hex string built right-to-left
            # Each byte as unsigned 2-digit hex, concatenated MSB-first
            hex_str = "".join(f"{int(b) & 0xFF:02x}" for b in reversed(row_bytes))
            lines.append(hex_str)

        path.write_text("\n".join(lines) + "\n")

    @staticmethod
    def _write_bias_mem(path: Path, bias_data: np.ndarray) -> None:
        """Write bias data as .mem file for $readmemh.

        Each line: one INT32 bias value in 8-char hex.
        """
        flat = bias_data.flatten()
        lines = []
        for val in flat:
            # INT32 as unsigned hex
            uval = int(val) & 0xFFFFFFFF
            lines.append(f"{uval:08x}")
        path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# DieAreaEstimator
# ---------------------------------------------------------------------------


# SRAM bit cell areas by process node (um² per bit)
PROCESS_BIT_CELL_AREA: dict[str, float] = {
    "7nm": 0.050,
    "5nm": 0.035,
    "14nm": 0.080,
    "28nm": 0.120,
}

# Overhead factor for SRAM (periphery, sense amps, decoders)
SRAM_OVERHEAD_FACTOR = 1.4


@dataclass
class DieAreaEstimate:
    """Die area breakdown for weight ROM."""

    process_node: str
    total_bits: int
    bit_cell_area_um2: float
    overhead_factor: float
    raw_area_um2: float
    total_area_um2: float
    total_area_mm2: float
    weight_bits: int
    bias_bits: int

    def to_json(self) -> dict:
        return {
            "process_node": self.process_node,
            "total_bits": self.total_bits,
            "bit_cell_area_um2": self.bit_cell_area_um2,
            "overhead_factor": self.overhead_factor,
            "raw_area_um2": self.raw_area_um2,
            "total_area_um2": self.total_area_um2,
            "total_area_mm2": self.total_area_mm2,
            "weight_bits": self.weight_bits,
            "bias_bits": self.bias_bits,
        }


class DieAreaEstimator:
    """Estimate die area for weight ROM at various process nodes."""

    def estimate(
        self,
        total_weight_bytes: int,
        total_bias_bytes: int,
        process_node: str = "7nm",
    ) -> DieAreaEstimate:
        """Estimate die area for storing weights and biases in SRAM.

        Formula: total_bits * bit_cell_area * overhead_factor

        Args:
            total_weight_bytes: Total weight storage in bytes.
            total_bias_bytes: Total bias storage in bytes.
            process_node: Process technology ("7nm", "5nm", "14nm", "28nm").

        Returns:
            DieAreaEstimate with mm² breakdown.
        """
        bit_cell_area = PROCESS_BIT_CELL_AREA.get(process_node)
        if bit_cell_area is None:
            raise ValueError(
                f"Unknown process node: {process_node}. "
                f"Supported: {list(PROCESS_BIT_CELL_AREA.keys())}"
            )

        weight_bits = total_weight_bytes * 8
        bias_bits = total_bias_bytes * 8
        total_bits = weight_bits + bias_bits

        raw_area_um2 = total_bits * bit_cell_area
        total_area_um2 = raw_area_um2 * SRAM_OVERHEAD_FACTOR
        total_area_mm2 = total_area_um2 / 1_000_000.0

        return DieAreaEstimate(
            process_node=process_node,
            total_bits=total_bits,
            bit_cell_area_um2=bit_cell_area,
            overhead_factor=SRAM_OVERHEAD_FACTOR,
            raw_area_um2=raw_area_um2,
            total_area_um2=total_area_um2,
            total_area_mm2=total_area_mm2,
            weight_bits=weight_bits,
            bias_bits=bias_bits,
        )


# ---------------------------------------------------------------------------
# WeightCompressor
# ---------------------------------------------------------------------------


@dataclass
class CompressionResult:
    """Result of weight compression."""

    strategy: str
    original_bytes: int
    compressed_bytes: int
    ratio: float  # original / compressed (>1 means savings)


class WeightCompressor:
    """Compress weight tensors for ROM storage.

    Three strategies:
      1. LUT encoding: <=16 unique values → 4-bit indices (up to 2x)
      2. Zero elimination: sparse weights → index+value pairs
      3. Huffman encoding: frequency-based variable-length codes
    """

    def compress_lut(self, data: np.ndarray) -> CompressionResult:
        """LUT encoding: if <= 16 unique values, use 4-bit indices.

        Each weight is stored as a 4-bit index into a 16-entry LUT.
        Compression ratio up to 2x for INT8 data.
        """
        original_bytes = data.nbytes
        unique_vals = np.unique(data)

        if len(unique_vals) > 16:
            return CompressionResult("lut", original_bytes, original_bytes, 1.0)

        # LUT table: 16 entries * element_size
        lut_bytes = 16 * data.dtype.itemsize
        # Indices: 4 bits per element = 0.5 bytes per element
        index_bytes = math.ceil(data.size * 4 / 8)
        compressed_bytes = lut_bytes + index_bytes

        ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 1.0
        return CompressionResult("lut", original_bytes, compressed_bytes, ratio)

    def compress_zero_elimination(self, data: np.ndarray) -> CompressionResult:
        """Zero elimination: store only non-zero values with indices.

        Format: (index: uint16, value: int8) pairs = 3 bytes per non-zero.
        """
        original_bytes = data.nbytes
        flat = data.flatten()
        nonzero_count = np.count_nonzero(flat)

        # Header (4 bytes: total_elements) + 3 bytes per non-zero
        compressed_bytes = 4 + nonzero_count * 3

        ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 1.0
        return CompressionResult("zero_elimination", original_bytes, compressed_bytes, ratio)

    def compress_huffman(self, data: np.ndarray) -> CompressionResult:
        """Huffman encoding estimate.

        Estimates compressed size based on entropy of the value distribution.
        """
        original_bytes = data.nbytes
        flat = data.flatten()

        if flat.size == 0:
            return CompressionResult("huffman", original_bytes, original_bytes, 1.0)

        # Count value frequencies
        counts = Counter(flat.tolist())
        total = flat.size

        # Compute entropy (bits per symbol)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Huffman coding achieves close to entropy
        # Add 10% overhead for codebook storage
        compressed_bits = flat.size * entropy
        codebook_bits = len(counts) * (8 + 16)  # value + code length estimate
        total_bits = compressed_bits + codebook_bits

        compressed_bytes = math.ceil(total_bits / 8)
        ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 1.0
        return CompressionResult("huffman", original_bytes, compressed_bytes, ratio)

    def auto_compress(self, data: np.ndarray) -> CompressionResult:
        """Try all strategies and return the best compression ratio."""
        results = [
            self.compress_lut(data),
            self.compress_zero_elimination(data),
            self.compress_huffman(data),
        ]
        return max(results, key=lambda r: r.ratio)
