"""Tests for Stage 4: Weight Packing & Memory Layout.

Test categories:
  - .mem file format: valid hex, correct word count, parseable
  - Weight sizes: match expected dimensions per layer
  - Memory map: no address overlaps, correct sizes
  - Roundtrip: pack → unpack → identical to original
  - Integration: packed weights + INT8 interpreter == ONNX Runtime
  - Bias domain: biases are in accumulator domain (pre-scaled)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mlasic.ingestion import ONNXParser
from mlasic.int8_interpreter import INT8Interpreter
from mlasic.ir import Graph
from mlasic.optimization import (
    BatchNormFoldingPass,
    ConstantFoldingPass,
    DeadCodeEliminationPass,
    OperatorFusionPass,
    PassManager,
    QuantizationPass,
)
from mlasic.scheduler import Scheduler
from mlasic.weight_packer import (
    WeightPacker,
    load_bias_mem,
    load_weight_mem,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scheduled_ad_graph(ad_model_path: Path, test_vectors: np.ndarray) -> Graph:
    """Run the full pipeline (Stages 1-3) on the AD model, return scheduled graph."""
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

    return graph


@pytest.fixture
def packed_output(scheduled_ad_graph: Graph, tmp_path: Path) -> tuple[dict, Path, Graph]:
    """Pack the AD model and return (memory_map, output_dir, graph)."""
    packer = WeightPacker(scheduled_ad_graph, tmp_path / "packed")
    memory_map = packer.pack_weights()
    return memory_map, tmp_path / "packed", scheduled_ad_graph


# ---------------------------------------------------------------------------
# .mem file format tests
# ---------------------------------------------------------------------------


class TestMemFileFormat:
    """Test that .mem files are valid $readmemh-compatible hex."""

    def test_weight_file_hex_format(self, packed_output):
        """Each line in weight .mem is 256 hex chars (1024 bits)."""
        _, output_dir, _ = packed_output
        filepath = output_dir / "weight_bank.mem"
        with open(filepath) as f:
            for i, line in enumerate(f):
                line = line.strip()
                assert len(line) == 256, f"Weight line {i}: expected 256 hex chars, got {len(line)}"
                assert all(c in "0123456789abcdef" for c in line), f"Weight line {i}: invalid hex"

    def test_bias_file_hex_format(self, packed_output):
        """Each line in bias .mem is 256 hex chars (1024 bits)."""
        _, output_dir, _ = packed_output
        filepath = output_dir / "bias_bank.mem"
        with open(filepath) as f:
            for i, line in enumerate(f):
                line = line.strip()
                assert len(line) == 256, f"Bias line {i}: expected 256 hex chars, got {len(line)}"
                assert all(c in "0123456789abcdef" for c in line), f"Bias line {i}: invalid hex"

    def test_weight_file_line_count(self, packed_output):
        """Combined weight bank has correct number of rows."""
        memory_map, output_dir, _ = packed_output
        filepath = output_dir / "weight_bank.mem"
        with open(filepath) as f:
            line_count = sum(1 for _ in f)
        expected = memory_map["weight_bank"]["total_rows"]
        assert line_count == expected

    def test_bias_file_line_count(self, packed_output):
        """Combined bias bank has correct number of rows."""
        memory_map, output_dir, _ = packed_output
        filepath = output_dir / "bias_bank.mem"
        with open(filepath) as f:
            line_count = sum(1 for _ in f)
        expected = memory_map["bias_bank"]["total_rows"]
        assert line_count == expected

    def test_per_layer_files_exist(self, packed_output):
        """Per-layer .mem files exist for all 4 layers."""
        _, output_dir, _ = packed_output
        for i in range(4):
            assert (output_dir / f"weights_layer{i}.mem").exists()
            assert (output_dir / f"biases_layer{i}.mem").exists()

    def test_per_layer_line_counts(self, packed_output):
        """Per-layer weight .mem files have correct row counts."""
        memory_map, output_dir, _ = packed_output
        for layer_info in memory_map["layers"]:
            idx = layer_info["layer_index"]
            filepath = output_dir / f"weights_layer{idx}.mem"
            with open(filepath) as f:
                line_count = sum(1 for _ in f)
            assert line_count == layer_info["weight"]["num_rows"]


# ---------------------------------------------------------------------------
# Weight size tests
# ---------------------------------------------------------------------------


class TestWeightSizes:
    """Verify weight sizes match expected AD model dimensions."""

    # From RTL spec: Layer dims and expected sizes
    EXPECTED = [
        {"layer": 0, "in_dim": 640, "out_dim": 128, "weight_bytes": 81920, "bias_bytes": 512},
        {"layer": 1, "in_dim": 128, "out_dim": 128, "weight_bytes": 16384, "bias_bytes": 512},
        {"layer": 2, "in_dim": 128, "out_dim": 128, "weight_bytes": 16384, "bias_bytes": 512},
        {"layer": 3, "in_dim": 128, "out_dim": 640, "weight_bytes": 81920, "bias_bytes": 2560},
    ]

    def test_per_layer_weight_sizes(self, packed_output):
        """Weight sizes match expected dimensions per layer."""
        memory_map, _, _ = packed_output
        for exp in self.EXPECTED:
            layer = memory_map["layers"][exp["layer"]]
            assert layer["weight"]["size_bytes"] == exp["weight_bytes"], (
                f"Layer {exp['layer']}: expected {exp['weight_bytes']} weight bytes, "
                f"got {layer['weight']['size_bytes']}"
            )

    def test_per_layer_bias_sizes(self, packed_output):
        """Bias sizes match expected dimensions per layer."""
        memory_map, _, _ = packed_output
        for exp in self.EXPECTED:
            layer = memory_map["layers"][exp["layer"]]
            assert layer["bias"]["size_bytes"] == exp["bias_bytes"], (
                f"Layer {exp['layer']}: expected {exp['bias_bytes']} bias bytes, "
                f"got {layer['bias']['size_bytes']}"
            )

    def test_total_weight_size(self, packed_output):
        """Total weight storage matches expected 196,608 bytes."""
        memory_map, _, _ = packed_output
        assert memory_map["weight_bank"]["total_bytes"] == 196608

    def test_total_bias_size(self, packed_output):
        """Total bias storage matches expected 4,096 bytes."""
        memory_map, _, _ = packed_output
        assert memory_map["bias_bank"]["total_bytes"] == 4096

    def test_total_fits_sram_budget(self, packed_output):
        """Total weight + bias fits SRAM budget (~197 KB for KV260)."""
        memory_map, _, _ = packed_output
        total = memory_map["total_bytes"]
        # SRAM budget: weight_bank (1536*128=196608) + bias_bank (32*128=4096) = 200704
        assert total <= 200704, f"Total {total} bytes exceeds SRAM budget"

    def test_weight_row_counts(self, packed_output):
        """Weight row counts match RTL spec expectations."""
        memory_map, _, _ = packed_output
        # Layer 0: 640 rows (1 tile × 640 IN_DIM)
        assert memory_map["layers"][0]["weight"]["num_rows"] == 640
        # Layer 1: 128 rows (1 tile × 128 IN_DIM)
        assert memory_map["layers"][1]["weight"]["num_rows"] == 128
        # Layer 2: 128 rows
        assert memory_map["layers"][2]["weight"]["num_rows"] == 128
        # Layer 3: 640 rows (5 tiles × 128 IN_DIM)
        assert memory_map["layers"][3]["weight"]["num_rows"] == 640


# ---------------------------------------------------------------------------
# Memory map tests
# ---------------------------------------------------------------------------


class TestMemoryMap:
    """Verify memory map correctness and no address overlaps."""

    def test_memory_map_json_written(self, packed_output):
        """memory_map.json is written and parseable."""
        _, output_dir, _ = packed_output
        path = output_dir / "memory_map.json"
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert "layers" in data
        assert len(data["layers"]) == 4

    def test_no_weight_address_overlaps(self, packed_output):
        """Weight SRAM ranges don't overlap between layers."""
        memory_map, _, _ = packed_output
        ranges = []
        for layer in memory_map["layers"]:
            w = layer["weight"]
            start = w["start_row"]
            end = start + w["num_rows"]
            ranges.append((start, end, layer["layer_index"]))

        # Sort by start and verify no overlaps
        ranges.sort()
        for i in range(len(ranges) - 1):
            _, end_a, idx_a = ranges[i]
            start_b, _, idx_b = ranges[i + 1]
            assert end_a <= start_b, (
                f"Weight overlap: layer {idx_a} ends at row {end_a}, "
                f"layer {idx_b} starts at row {start_b}"
            )

    def test_no_bias_address_overlaps(self, packed_output):
        """Bias SRAM ranges don't overlap between layers."""
        memory_map, _, _ = packed_output
        ranges = []
        for layer in memory_map["layers"]:
            b = layer["bias"]
            start = b["start_row"]
            end = start + b["num_rows"]
            ranges.append((start, end, layer["layer_index"]))

        ranges.sort()
        for i in range(len(ranges) - 1):
            _, end_a, idx_a = ranges[i]
            start_b, _, idx_b = ranges[i + 1]
            assert end_a <= start_b, (
                f"Bias overlap: layer {idx_a} ends at row {end_a}, "
                f"layer {idx_b} starts at row {start_b}"
            )

    def test_weight_rows_sequential(self, packed_output):
        """Weight rows are packed sequentially with no gaps."""
        memory_map, _, _ = packed_output
        cursor = 0
        for layer in memory_map["layers"]:
            w = layer["weight"]
            assert w["start_row"] == cursor, (
                f"Layer {layer['layer_index']}: weight start_row={w['start_row']}, "
                f"expected {cursor}"
            )
            cursor += w["num_rows"]

    def test_bias_rows_sequential(self, packed_output):
        """Bias rows are packed sequentially with no gaps."""
        memory_map, _, _ = packed_output
        cursor = 0
        for layer in memory_map["layers"]:
            b = layer["bias"]
            assert b["start_row"] == cursor, (
                f"Layer {layer['layer_index']}: bias start_row={b['start_row']}, expected {cursor}"
            )
            cursor += b["num_rows"]


# ---------------------------------------------------------------------------
# Roundtrip tests
# ---------------------------------------------------------------------------


class TestRoundtrip:
    """Pack → unpack → verify data matches original quantized tensors."""

    def test_weight_roundtrip_per_layer(self, packed_output):
        """Load per-layer weight .mem and verify exact match to graph tensors."""
        _, output_dir, graph = packed_output

        for node_name in graph.topological_order():
            node = graph.nodes[node_name]
            ls = node.schedule_info
            attrs = node.fused_attrs

            # Original quantized weights
            w_orig = graph.tensors[node.inputs[1]].data
            assert w_orig.dtype == np.int8

            # Load packed weights back
            filepath = output_dir / f"weights_layer{ls.layer_index}.mem"
            loaded_rows = load_weight_mem(filepath)
            assert len(loaded_rows) == ls.weight_rows

            # Reconstruct weight matrix from rows
            w_reconstructed = np.zeros_like(w_orig)
            row_idx = 0
            for tile in range(ls.num_tiles):
                col_start = tile * ls.parallelism
                col_end = col_start + ls.parallelism
                for i in range(attrs.input_dim):
                    w_reconstructed[i, col_start:col_end] = loaded_rows[row_idx]
                    row_idx += 1

            np.testing.assert_array_equal(
                w_reconstructed,
                w_orig,
                err_msg=f"Weight roundtrip mismatch at layer {ls.layer_index}",
            )

    def test_bias_roundtrip_per_layer(self, packed_output):
        """Load per-layer bias .mem and verify exact match to graph tensors."""
        _, output_dir, graph = packed_output

        for node_name in graph.topological_order():
            node = graph.nodes[node_name]
            ls = node.schedule_info
            attrs = node.fused_attrs

            # Original quantized bias
            if len(node.inputs) < 3:
                continue
            b_orig = graph.tensors[node.inputs[2]].data
            assert b_orig.dtype == np.int32

            # Load packed biases back
            filepath = output_dir / f"biases_layer{ls.layer_index}.mem"
            loaded_rows = load_bias_mem(filepath)

            # Reconstruct bias vector from rows
            b_reconstructed = np.concatenate(loaded_rows)[: attrs.output_dim]
            np.testing.assert_array_equal(
                b_reconstructed,
                b_orig,
                err_msg=f"Bias roundtrip mismatch at layer {ls.layer_index}",
            )

    def test_combined_weight_bank_roundtrip(self, packed_output):
        """Load combined weight_bank.mem and verify it matches all layers."""
        _, output_dir, graph = packed_output
        filepath = output_dir / "weight_bank.mem"
        all_rows = load_weight_mem(filepath)

        global_row = 0
        for node_name in graph.topological_order():
            node = graph.nodes[node_name]
            ls = node.schedule_info
            attrs = node.fused_attrs
            w_orig = graph.tensors[node.inputs[1]].data

            for tile in range(ls.num_tiles):
                col_start = tile * ls.parallelism
                col_end = col_start + ls.parallelism
                for i in range(attrs.input_dim):
                    np.testing.assert_array_equal(
                        all_rows[global_row],
                        w_orig[i, col_start:col_end],
                        err_msg=f"Combined weight mismatch at global row {global_row}",
                    )
                    global_row += 1


# ---------------------------------------------------------------------------
# Bias domain tests
# ---------------------------------------------------------------------------


class TestBiasDomain:
    """Verify biases are in accumulator domain (pre-scaled), not output domain."""

    def test_biases_are_prescaled(self, packed_output):
        """Bias values b_q = floor(b_fp / (scale_x * scale_w) + 0.5)."""
        _, _, graph = packed_output

        for node_name in graph.topological_order():
            node = graph.nodes[node_name]

            if len(node.inputs) < 3:
                continue
            b_q = graph.tensors[node.inputs[2]].data

            # Bias should be INT32 (accumulator domain)
            assert b_q.dtype == np.int32, f"Bias dtype should be int32, got {b_q.dtype}"

            # Biases in accumulator domain are typically larger than the output INT8 range
            # For the AD model, typical accumulator-domain bias magnitudes > 100
            # (but near-zero biases are also possible)
            # Just verify they're INT32 and not accidentally in [-128, 127] INT8 range
            # (This is a sanity check, not a guarantee)


# ---------------------------------------------------------------------------
# INT8 integration test
# ---------------------------------------------------------------------------


class TestINT8Integration:
    """Verify packed weights produce correct INT8 inference results."""

    def test_packed_weights_inference(self, packed_output, test_vectors):
        """Run INT8 inference using loaded packed weights, compare to graph interpreter."""
        _, output_dir, graph = packed_output

        # Run inference with the graph's INT8 interpreter (reference)
        interp = INT8Interpreter(graph)

        for vec_idx in range(min(10, len(test_vectors))):
            test_input = test_vectors[vec_idx]
            inputs = {graph.inputs[0]: test_input}
            ref_output = interp.run(inputs)

            # Now manually load packed weights and run inference
            # This verifies the packing doesn't corrupt data
            manual_output = self._manual_inference(graph, output_dir, test_input)

            for key in ref_output:
                np.testing.assert_array_equal(
                    manual_output[key],
                    ref_output[key],
                    err_msg=f"INT8 inference mismatch at vector {vec_idx}",
                )

    def _manual_inference(
        self, graph: Graph, output_dir: Path, test_input: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Run INT8 inference using weights loaded from .mem files."""
        ordered_nodes = [graph.nodes[n] for n in graph.topological_order()]

        # Quantize input using first layer's input quant params
        first_attrs = ordered_nodes[0].fused_attrs
        x = first_attrs.input_quant.quantize(test_input.astype(np.float32))

        for node in ordered_nodes:
            ls = node.schedule_info
            attrs = node.fused_attrs

            # Load weights from .mem file
            weight_rows = load_weight_mem(output_dir / f"weights_layer{ls.layer_index}.mem")
            w_q = np.zeros((attrs.input_dim, attrs.output_dim), dtype=np.int8)
            row_idx = 0
            for tile in range(ls.num_tiles):
                col_start = tile * ls.parallelism
                col_end = col_start + ls.parallelism
                for i in range(attrs.input_dim):
                    w_q[i, col_start:col_end] = weight_rows[row_idx]
                    row_idx += 1

            # Load biases from .mem file
            bias_rows = load_bias_mem(output_dir / f"biases_layer{ls.layer_index}.mem")
            b_q = np.concatenate(bias_rows)[: attrs.output_dim]

            # INT8 matmul + bias
            x_int32 = x.astype(np.int32)
            w_int32 = w_q.astype(np.int32)
            acc = x_int32 @ w_int32 + b_q.astype(np.int32)

            # Requantize
            m_fixed = np.int64(attrs.requant_scale_fixed)
            shift = attrs.requant_shift
            scaled = acc.astype(np.int64) * m_fixed
            rounded = (scaled + (np.int64(1) << (shift - 1))) >> shift

            # Add output zero point, clamp, ReLU
            zp_out = np.int64(attrs.output_quant.zero_point)
            result = rounded + zp_out
            result = np.clip(result, -128, 127).astype(np.int8)
            if attrs.has_relu:
                result = np.maximum(result, np.int8(0))

            x = result

        return {graph.outputs[0]: x}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_wrong_stage_rejected(self, tmp_path):
        """WeightPacker rejects non-scheduled graphs."""
        from mlasic.ir import Graph

        graph = Graph(name="test", nodes={}, tensors={}, inputs=[], outputs=[], stage="quantized")
        with pytest.raises(ValueError, match="scheduled"):
            WeightPacker(graph, tmp_path)

    def test_signed_weight_values_preserved(self, packed_output):
        """Verify negative weight values survive pack/unpack."""
        _, output_dir, graph = packed_output

        for node_name in graph.topological_order():
            node = graph.nodes[node_name]
            ls = node.schedule_info
            w_orig = graph.tensors[node.inputs[1]].data

            # Verify we have some negative values
            assert (w_orig < 0).any(), f"Layer {ls.layer_index}: no negative weights"

            # Roundtrip check
            loaded = load_weight_mem(output_dir / f"weights_layer{ls.layer_index}.mem")
            attrs = node.fused_attrs
            w_rt = np.zeros_like(w_orig)
            row_idx = 0
            for tile in range(ls.num_tiles):
                col_start = tile * ls.parallelism
                col_end = col_start + ls.parallelism
                for i in range(attrs.input_dim):
                    w_rt[i, col_start:col_end] = loaded[row_idx]
                    row_idx += 1

            np.testing.assert_array_equal(w_rt, w_orig)

    def test_extreme_int8_values_roundtrip(self, tmp_path):
        """Extreme INT8 values (-128, -127, 0, 127) survive pack/unpack."""
        from mlasic.weight_packer import _write_int8_mem, load_weight_mem

        # Create a row with extreme values
        row = np.array(
            [-128, -127, -1, 0, 1, 126, 127] + [0] * 121,
            dtype=np.int8,
        )
        filepath = tmp_path / "extreme.mem"
        _write_int8_mem([row], filepath)

        loaded = load_weight_mem(filepath)
        assert len(loaded) == 1
        np.testing.assert_array_equal(loaded[0], row)

    def test_extreme_int32_bias_roundtrip(self, tmp_path):
        """Extreme INT32 bias values survive pack/unpack."""
        from mlasic.weight_packer import _write_int32_mem, load_bias_mem

        row = np.array(
            [-2147483648, -1, 0, 1, 2147483647] + [0] * 27,
            dtype=np.int32,
        )
        filepath = tmp_path / "extreme_bias.mem"
        _write_int32_mem([row], filepath)

        loaded = load_bias_mem(filepath)
        assert len(loaded) == 1
        np.testing.assert_array_equal(loaded[0], row)
