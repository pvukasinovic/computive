"""Tests for ROM Mapper (Phase 3 ROM mapping and die area estimation)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mlasic.dag_scheduler import DAGScheduler
from mlasic.ir import (
    FusedLinearAttrs,
    Graph,
    OpNode,
    OpType,
    QuantParams,
    Tensor,
    TensorType,
)
from mlasic.rom_mapper import (
    DieAreaEstimator,
    WeightCompressor,
    WeightROMMapper,
)
from mlasic.tile_mapper import TileMapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scheduled_graph():
    """Build, schedule, and tile-map a 2-layer graph."""
    qp = QuantParams(scale=0.1, zero_point=0, calibrated=True)
    attrs0 = FusedLinearAttrs(
        input_dim=128,
        output_dim=64,
        has_relu=True,
        weight_quant=qp,
        input_quant=qp,
        output_quant=qp,
        requant_scale_fixed=65536,
        requant_shift=16,
    )
    attrs1 = FusedLinearAttrs(
        input_dim=64,
        output_dim=32,
        has_relu=False,
        weight_quant=qp,
        input_quant=qp,
        output_quant=qp,
        requant_scale_fixed=65536,
        requant_shift=16,
    )
    n0 = OpNode("n0", OpType.FUSED_LINEAR_RELU, ["x", "n0_w", "n0_b"], ["h0"])
    n0.fused_attrs = attrs0
    n1 = OpNode("n1", OpType.FUSED_LINEAR, ["h0", "n1_w", "n1_b"], ["y"])
    n1.fused_attrs = attrs1

    rng = np.random.RandomState(42)
    tensors = {
        "x": Tensor("x", TensorType((1, 128), np.dtype(np.int8))),
        "n0_w": Tensor(
            "n0_w",
            TensorType((128, 64), np.dtype(np.int8)),
            data=rng.randint(-127, 128, (128, 64), dtype=np.int8),
        ),
        "n0_b": Tensor(
            "n0_b",
            TensorType((64,), np.dtype(np.int32)),
            data=rng.randint(-1000, 1000, (64,), dtype=np.int32),
        ),
        "h0": Tensor("h0", TensorType((1, 64), np.dtype(np.int8))),
        "n1_w": Tensor(
            "n1_w",
            TensorType((64, 32), np.dtype(np.int8)),
            data=rng.randint(-127, 128, (64, 32), dtype=np.int8),
        ),
        "n1_b": Tensor(
            "n1_b",
            TensorType((32,), np.dtype(np.int32)),
            data=rng.randint(-1000, 1000, (32,), dtype=np.int32),
        ),
        "y": Tensor("y", TensorType((1, 32), np.dtype(np.int8))),
    }

    graph = Graph("test", {"n0": n0, "n1": n1}, tensors, ["x"], ["y"], stage="quantized_dag")
    sched = DAGScheduler().schedule(graph)
    fabric = TileMapper().map(graph, sched)
    return graph, fabric


# ---------------------------------------------------------------------------
# TestWeightROMMapper
# ---------------------------------------------------------------------------


class TestWeightROMMapper:
    def test_generates_manifest(self):
        """ROM mapper produces a manifest with tile entries."""
        graph, fabric = _make_scheduled_graph()
        mapper = WeightROMMapper()
        manifest = mapper.generate(graph, fabric)

        assert "tiles" in manifest
        assert len(manifest["tiles"]) == 2  # 2 MAC tiles
        assert manifest["total_weight_bytes"] > 0
        assert manifest["total_bias_bytes"] > 0

    def test_writes_binary_files(self, tmp_path):
        """ROM mapper writes .bin files to disk."""
        graph, fabric = _make_scheduled_graph()
        mapper = WeightROMMapper()
        manifest = mapper.generate(graph, fabric, output_dir=tmp_path)

        # Check weight files exist
        for tile_info in manifest["tiles"]:
            if tile_info["weight_bytes"] > 0:
                assert "weight_file" in tile_info
                assert (tmp_path / f"tile_{tile_info['tile_id']}_weights.bin").exists()
            if tile_info["bias_bytes"] > 0:
                assert "bias_file" in tile_info
                assert (tmp_path / f"tile_{tile_info['tile_id']}_biases.bin").exists()

        # Check manifest file
        assert (tmp_path / "rom_manifest.json").exists()
        with open(tmp_path / "rom_manifest.json") as f:
            loaded = json.load(f)
        assert loaded["total_weight_bytes"] == manifest["total_weight_bytes"]

    def test_weight_bytes_correct(self):
        """Individual tile weight bytes match expected sizes."""
        graph, fabric = _make_scheduled_graph()
        mapper = WeightROMMapper()
        manifest = mapper.generate(graph, fabric)

        # n0: 128*64 = 8192 bytes, n1: 64*32 = 2048 bytes
        tile_weights = {t["node_name"]: t["weight_bytes"] for t in manifest["tiles"]}
        assert tile_weights["n0"] == 128 * 64
        assert tile_weights["n1"] == 64 * 32


# ---------------------------------------------------------------------------
# TestDieAreaEstimator
# ---------------------------------------------------------------------------


class TestDieAreaEstimator:
    def test_7nm_ad_model(self):
        """AD model die area sanity check at 7nm.

        ~100K params * 1 byte = 100KB weights + ~3.5KB biases
        ≈ 100,000 * 8 * 0.05 * 1.4 = ~0.056 mm²
        """
        estimator = DieAreaEstimator()
        # AD model: ~100K weight bytes + ~3.5K bias bytes
        result = estimator.estimate(100_000, 3_500, "7nm")

        # Sanity check: should be around 0.058 mm²
        assert 0.05 < result.total_area_mm2 < 0.07
        assert result.process_node == "7nm"
        assert result.total_bits == (100_000 + 3_500) * 8

    def test_different_process_nodes(self):
        """Larger process nodes have larger die area."""
        estimator = DieAreaEstimator()
        area_7nm = estimator.estimate(10_000, 1_000, "7nm")
        area_28nm = estimator.estimate(10_000, 1_000, "28nm")

        assert area_28nm.total_area_mm2 > area_7nm.total_area_mm2

    def test_unknown_process_raises(self):
        """Unknown process node raises ValueError."""
        estimator = DieAreaEstimator()
        with pytest.raises(ValueError, match="Unknown process"):
            estimator.estimate(1000, 100, "3nm")

    def test_to_json(self):
        """DieAreaEstimate.to_json() returns valid structure."""
        estimator = DieAreaEstimator()
        result = estimator.estimate(10_000, 1_000, "7nm")
        data = result.to_json()
        assert "process_node" in data
        assert "total_area_mm2" in data
        assert "weight_bits" in data


# ---------------------------------------------------------------------------
# TestWeightCompressor
# ---------------------------------------------------------------------------


class TestWeightCompressor:
    def test_lut_with_few_unique(self):
        """LUT encoding achieves ~2x for <= 16 unique values."""
        compressor = WeightCompressor()
        # 1000 elements with only 4 unique values
        data = np.array([0, 1, -1, 2] * 250, dtype=np.int8)
        result = compressor.compress_lut(data)
        assert result.strategy == "lut"
        assert result.ratio > 1.5  # Close to 2x for INT8

    def test_lut_too_many_unique(self):
        """LUT fails gracefully with > 16 unique values."""
        compressor = WeightCompressor()
        data = np.arange(100, dtype=np.int8)  # 100 unique values
        result = compressor.compress_lut(data)
        assert result.ratio == 1.0  # No compression

    def test_zero_elimination_sparse(self):
        """Zero elimination saves space for sparse weights."""
        compressor = WeightCompressor()
        data = np.zeros(1000, dtype=np.int8)
        data[0] = 1  # Only 1 non-zero
        result = compressor.compress_zero_elimination(data)
        assert result.strategy == "zero_elimination"
        assert result.ratio > 10.0  # 1000 bytes → ~7 bytes

    def test_huffman_compression(self):
        """Huffman achieves some compression on skewed distribution."""
        compressor = WeightCompressor()
        rng = np.random.RandomState(42)
        # Highly skewed: most values near 0
        data = np.clip(rng.normal(0, 0.5, 1000), -3, 3).astype(np.int8)
        result = compressor.compress_huffman(data)
        assert result.strategy == "huffman"
        assert result.ratio >= 1.0  # Should at least not expand

    def test_auto_picks_best(self):
        """Auto mode selects the strategy with best ratio."""
        compressor = WeightCompressor()
        # Very sparse data — zero elimination should win
        data = np.zeros(1000, dtype=np.int8)
        data[0] = 42
        result = compressor.auto_compress(data)
        assert result.ratio > 1.0
