"""End-to-end tests with large models (100M–1B parameters).

Tests the full compiler pipeline (Stages 1–6) at scale to verify:
  - Integer overflow safety in scheduling/weight packing
  - Correct output tiling with many tiles (40–128)
  - RTL generation with large parameter arrays
  - Memory map correctness at scale
  - Golden vector generation for large models
"""

from __future__ import annotations

import gc
import math
import time
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from mlasic.golden_vectors import GoldenVectorGenerator
from mlasic.ingestion import ONNXParser
from mlasic.int8_interpreter import INT8Interpreter
from mlasic.ir import HardwareConstraints
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
from mlasic.testbench_gen import VerifConfig, VerifGenerator
from mlasic.weight_packer import WeightPacker, load_bias_mem, load_weight_mem


# ---------------------------------------------------------------------------
# Model builder (memory-efficient: generates weights lazily)
# ---------------------------------------------------------------------------


def build_large_mlp(
    layer_dims: list[int],
    name: str = "large_mlp",
    seed: int = 42,
    has_bn: bool = True,
) -> onnx.ModelProto:
    """Build a large MLP model with the given layer dimensions.

    Uses Xavier initialization for realistic weight distributions.
    Only adds BatchNorm on hidden (non-last) layers.
    """
    rng = np.random.RandomState(seed)
    all_nodes = []
    all_inits = []
    all_vis = []

    input_name = "input"
    current = input_name
    num_layers = len(layer_dims) - 1

    for i in range(num_layers):
        in_dim = layer_dims[i]
        out_dim = layer_dims[i + 1]
        is_last = i == num_layers - 1
        prefix = f"layer{i}"

        # Weight: Xavier initialization
        fan_avg = (in_dim + out_dim) / 2
        scale = np.sqrt(1.0 / fan_avg)
        w_data = (rng.randn(in_dim, out_dim) * scale).astype(np.float32)
        w_name = f"{prefix}_weight"
        all_inits.append(numpy_helper.from_array(w_data, name=w_name))

        # MatMul
        mm_out = f"{prefix}_matmul_out"
        all_nodes.append(
            helper.make_node("MatMul", [current, w_name], [mm_out], name=f"{prefix}_MatMul")
        )
        all_vis.append(helper.make_tensor_value_info(mm_out, TensorProto.FLOAT, [1, out_dim]))

        # Bias
        b_data = (rng.randn(out_dim) * 0.01).astype(np.float32)
        b_name = f"{prefix}_bias"
        all_inits.append(numpy_helper.from_array(b_data, name=b_name))

        # Add
        add_out = f"{prefix}_add_out"
        all_nodes.append(
            helper.make_node("Add", [mm_out, b_name], [add_out], name=f"{prefix}_Add")
        )
        all_vis.append(helper.make_tensor_value_info(add_out, TensorProto.FLOAT, [1, out_dim]))
        current = add_out

        # BatchNorm (hidden layers only)
        if has_bn and not is_last:
            bn_scale = np.ones(out_dim, dtype=np.float32)
            bn_bias = np.zeros(out_dim, dtype=np.float32)
            bn_mean = rng.randn(out_dim).astype(np.float32) * 0.1
            bn_var = np.abs(rng.randn(out_dim).astype(np.float32)) + 0.5

            names = [f"{prefix}_bn_{p}" for p in ["scale", "bias", "mean", "var"]]
            for arr, n in zip([bn_scale, bn_bias, bn_mean, bn_var], names):
                all_inits.append(numpy_helper.from_array(arr, name=n))

            bn_out = f"{prefix}_bn_out"
            all_nodes.append(
                helper.make_node(
                    "BatchNormalization",
                    [current, *names],
                    [bn_out],
                    name=f"{prefix}_BN",
                    epsilon=1e-5,
                )
            )
            all_vis.append(
                helper.make_tensor_value_info(bn_out, TensorProto.FLOAT, [1, out_dim])
            )
            current = bn_out

        # ReLU (hidden layers only)
        if not is_last:
            relu_out = f"{prefix}_relu_out"
            all_nodes.append(
                helper.make_node("Relu", [current], [relu_out], name=f"{prefix}_Relu")
            )
            all_vis.append(
                helper.make_tensor_value_info(relu_out, TensorProto.FLOAT, [1, out_dim])
            )
            current = relu_out

    output_name = current
    in_dim = layer_dims[0]
    out_dim = layer_dims[-1]

    graph = helper.make_graph(
        nodes=all_nodes,
        name=name,
        inputs=[helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, in_dim])],
        outputs=[helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, out_dim])],
        initializer=all_inits,
        value_info=all_vis,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def count_params(model: onnx.ModelProto) -> int:
    """Count total parameters in an ONNX model (weights + biases)."""
    total = 0
    for init in model.graph.initializer:
        total += int(np.prod(numpy_helper.to_array(init).shape))
    return total


def _weight_rows(in_dim: int, out_dim: int, parallelism: int = 128) -> int:
    """Calculate weight SRAM rows needed for a single layer."""
    par = min(out_dim, parallelism)
    while par > 1:
        if out_dim % par == 0:
            break
        par -= 1
    num_tiles = out_dim // par
    return num_tiles * in_dim


def _hw_constraints_for_model(layer_dims: list[int]) -> HardwareConstraints:
    """Build HardwareConstraints large enough for the given model."""
    num_layers = len(layer_dims) - 1
    total_weight_rows = sum(
        _weight_rows(layer_dims[i], layer_dims[i + 1])
        for i in range(num_layers)
    )
    total_bias_rows = sum(
        math.ceil(layer_dims[i + 1] / 32) for i in range(num_layers)
    )
    max_dim = max(layer_dims)

    return HardwareConstraints(
        weight_bank_depth=total_weight_rows + 100,
        bias_bank_depth=total_bias_rows + 100,
        act_buffer_bytes=max_dim,
    )


def run_pipeline_stages_1_to_4(
    model_path: Path,
    output_dir: Path,
    layer_dims: list[int],
    n_cal: int = 10,
):
    """Run Stages 1-4, return (graph, schedule, memory_map)."""
    # Stage 1: Ingestion
    parser = ONNXParser(model_path)
    graph = parser.parse()

    # Stage 2: Optimization
    rng = np.random.RandomState(42)
    cal_data = [rng.randn(1, layer_dims[0]).astype(np.float32) for _ in range(n_cal)]

    pm = PassManager()
    pm.add_pass(ConstantFoldingPass())
    pm.add_pass(DeadCodeEliminationPass())
    pm.add_pass(BatchNormFoldingPass())
    pm.add_pass(OperatorFusionPass())
    pm.add_pass(QuantizationPass(calibration_data=cal_data))
    graph = pm.run(graph, verify=False)  # Skip verify for speed on large models

    # Stage 3: Scheduling
    hw = _hw_constraints_for_model(layer_dims)
    scheduler = Scheduler(constraints=hw)
    schedule = scheduler.schedule(graph)

    # Stage 4: Weight Packing
    weight_dir = output_dir / "weights"
    packer = WeightPacker(graph, weight_dir, constraints=hw)
    memory_map = packer.pack_weights()

    return graph, schedule, memory_map, hw


# ---------------------------------------------------------------------------
# 100M Model: 5120→5120→5120→5120→5120 (≈105M params)
# ---------------------------------------------------------------------------


class TestModel100M:
    """Full pipeline test with ~100M parameter MLP."""

    LAYER_DIMS = [5120, 5120, 5120, 5120, 5120]
    EXPECTED_PARAMS_MIN = 100_000_000
    EXPECTED_PARAMS_MAX = 120_000_000

    def test_param_count(self, tmp_path: Path):
        """Verify model has ~105M parameters."""
        model = build_large_mlp(self.LAYER_DIMS, name="mlp_100m")
        total = count_params(model)
        assert self.EXPECTED_PARAMS_MIN <= total <= self.EXPECTED_PARAMS_MAX, (
            f"Expected ~105M params, got {total:,}"
        )

    def test_full_pipeline_stages_1_4(self, tmp_path: Path):
        """Stages 1-4 complete correctly on ~100M model."""
        model = build_large_mlp(self.LAYER_DIMS, name="mlp_100m")
        model_path = tmp_path / "mlp_100m.onnx"
        onnx.save(model, str(model_path))
        del model
        gc.collect()

        t0 = time.time()
        graph, schedule, memory_map, hw = run_pipeline_stages_1_to_4(
            model_path, tmp_path, self.LAYER_DIMS
        )
        elapsed = time.time() - t0

        # Verify stage progression
        assert graph.stage == "scheduled"
        assert len(graph.nodes) == 4  # 4 fused layers (3 hidden + 1 output)

        # Verify schedule
        for node_name in graph.topological_order():
            node = graph.nodes[node_name]
            ls = node.schedule_info
            assert ls is not None
            assert ls.parallelism == 128  # 5120 / 128 = 40 tiles
            assert ls.num_tiles == 40
            assert ls.cycles_per_tile > 0

        # Verify memory map
        assert memory_map["weight_bank"]["total_rows"] > 0
        assert memory_map["bias_bank"]["total_rows"] > 0

        # Weight rows per layer: 40 tiles × 5120 in_dim = 204,800
        for layer_info in memory_map["layers"]:
            assert layer_info["weight"]["num_rows"] == 204_800
            assert layer_info["num_tiles"] == 40

        print(f"\n100M model pipeline completed in {elapsed:.1f}s")
        print(f"Total weight rows: {memory_map['weight_bank']['total_rows']:,}")
        print(f"Total bytes: {memory_map['total_bytes']:,}")

    def test_rtl_generation(self, tmp_path: Path):
        """Stage 5: RTL generation works for 100M model."""
        model = build_large_mlp(self.LAYER_DIMS, name="mlp_100m")
        model_path = tmp_path / "mlp_100m.onnx"
        onnx.save(model, str(model_path))
        del model
        gc.collect()

        graph, schedule, memory_map, hw = run_pipeline_stages_1_to_4(
            model_path, tmp_path, self.LAYER_DIMS
        )

        # Stage 5: RTL Generation
        rtl_dir = tmp_path / "rtl_output"
        gen = RTLGenerator(
            graph=graph,
            output_dir=rtl_dir,
            weight_dir=tmp_path / "weights",
            constraints=hw,
        )
        gen.generate_all()

        # Verify output files exist
        assert (rtl_dir / "parameters.svh").exists()
        assert (rtl_dir / "accelerator_top.sv").exists()

        # Verify parameters.svh has correct values
        params = (rtl_dir / "parameters.svh").read_text()
        assert "NUM_LAYERS" in params
        assert "5120" in params  # input_dim or output_dim

    def test_golden_vectors(self, tmp_path: Path):
        """Stage 6: Golden vector generation for 100M model."""
        model = build_large_mlp(self.LAYER_DIMS, name="mlp_100m")
        model_path = tmp_path / "mlp_100m.onnx"
        onnx.save(model, str(model_path))
        del model
        gc.collect()

        graph, schedule, memory_map, hw = run_pipeline_stages_1_to_4(
            model_path, tmp_path, self.LAYER_DIMS
        )

        # Generate a small number of golden vectors
        gvg = GoldenVectorGenerator(graph)
        vectors = gvg.generate_random_vectors(n=5, seed=42)

        assert vectors.num_vectors == 5
        for tv in vectors.vectors:
            for out_name in graph.outputs:
                assert tv.expected_output[out_name].dtype == np.int8
                # Output should be in valid INT8 range
                assert tv.expected_output[out_name].min() >= -128
                assert tv.expected_output[out_name].max() <= 127

        # Verify bitwise reproducibility
        passes, total, mismatches = gvg.verify_vectors(vectors)
        assert passes == total, f"Bitwise mismatch on {len(mismatches)} vectors"

    def test_weight_roundtrip(self, tmp_path: Path):
        """Packed weights roundtrip correctly for 100M model."""
        model = build_large_mlp(self.LAYER_DIMS, name="mlp_100m")
        model_path = tmp_path / "mlp_100m.onnx"
        onnx.save(model, str(model_path))
        del model
        gc.collect()

        graph, schedule, memory_map, hw = run_pipeline_stages_1_to_4(
            model_path, tmp_path, self.LAYER_DIMS
        )

        weight_dir = tmp_path / "weights"
        for node_name in graph.topological_order():
            node = graph.nodes[node_name]
            ls = node.schedule_info
            attrs = node.fused_attrs

            w_orig = graph.tensors[node.inputs[1]].data
            w_rows = load_weight_mem(weight_dir / f"weights_layer{ls.layer_index}.mem")

            w_rt = np.zeros_like(w_orig)
            row_idx = 0
            for tile in range(ls.num_tiles):
                cs = tile * ls.parallelism
                ce = cs + ls.parallelism
                for i in range(attrs.input_dim):
                    w_rt[i, cs:ce] = w_rows[row_idx]
                    row_idx += 1

            np.testing.assert_array_equal(
                w_rt, w_orig, err_msg=f"Layer {ls.layer_index} roundtrip failed"
            )

    def test_int8_inference_consistency(self, tmp_path: Path):
        """INT8 interpreter produces consistent results."""
        model = build_large_mlp(self.LAYER_DIMS, name="mlp_100m")
        model_path = tmp_path / "mlp_100m.onnx"
        onnx.save(model, str(model_path))
        del model
        gc.collect()

        graph, *_ = run_pipeline_stages_1_to_4(
            model_path, tmp_path, self.LAYER_DIMS
        )

        interp = INT8Interpreter(graph)

        # Same input should produce identical output
        rng = np.random.RandomState(42)
        test_input = rng.randn(1, self.LAYER_DIMS[0]).astype(np.float32)

        out1 = interp.run({graph.inputs[0]: test_input})
        out2 = interp.run({graph.inputs[0]: test_input})

        for name in graph.outputs:
            np.testing.assert_array_equal(out1[name], out2[name])


# ---------------------------------------------------------------------------
# 200M Model: 7168→7168→7168→7168→7168 (≈206M params)
# ---------------------------------------------------------------------------


class TestModel200M:
    """Full pipeline test with ~200M parameter MLP."""

    LAYER_DIMS = [7168, 7168, 7168, 7168, 7168]
    EXPECTED_PARAMS_MIN = 200_000_000
    EXPECTED_PARAMS_MAX = 220_000_000

    def test_param_count(self, tmp_path: Path):
        """Verify model has ~206M parameters."""
        model = build_large_mlp(self.LAYER_DIMS, name="mlp_200m")
        total = count_params(model)
        assert self.EXPECTED_PARAMS_MIN <= total <= self.EXPECTED_PARAMS_MAX, (
            f"Expected ~206M params, got {total:,}"
        )

    def test_full_pipeline_stages_1_4(self, tmp_path: Path):
        """Stages 1-4 complete correctly on ~200M model."""
        model = build_large_mlp(self.LAYER_DIMS, name="mlp_200m")
        model_path = tmp_path / "mlp_200m.onnx"
        onnx.save(model, str(model_path))
        del model
        gc.collect()

        t0 = time.time()
        graph, schedule, memory_map, hw = run_pipeline_stages_1_to_4(
            model_path, tmp_path, self.LAYER_DIMS
        )
        elapsed = time.time() - t0

        assert graph.stage == "scheduled"
        assert len(graph.nodes) == 4

        # 7168 / 128 = 56 tiles
        for layer_info in memory_map["layers"]:
            assert layer_info["num_tiles"] == 56
            assert layer_info["weight"]["num_rows"] == 56 * 7168

        print(f"\n200M model pipeline completed in {elapsed:.1f}s")
        print(f"Total weight rows: {memory_map['weight_bank']['total_rows']:,}")
        print(f"Total bytes: {memory_map['total_bytes']:,}")

    def test_rtl_generation(self, tmp_path: Path):
        """Stage 5: RTL generation works for 200M model."""
        model = build_large_mlp(self.LAYER_DIMS, name="mlp_200m")
        model_path = tmp_path / "mlp_200m.onnx"
        onnx.save(model, str(model_path))
        del model
        gc.collect()

        graph, schedule, memory_map, hw = run_pipeline_stages_1_to_4(
            model_path, tmp_path, self.LAYER_DIMS
        )

        rtl_dir = tmp_path / "rtl_output"
        gen = RTLGenerator(
            graph=graph,
            output_dir=rtl_dir,
            weight_dir=tmp_path / "weights",
            constraints=hw,
        )
        gen.generate_all()

        assert (rtl_dir / "parameters.svh").exists()
        assert (rtl_dir / "accelerator_top.sv").exists()

        params = (rtl_dir / "parameters.svh").read_text()
        assert "7168" in params

    def test_golden_vectors_and_verify(self, tmp_path: Path):
        """Golden vectors pass bitwise verification."""
        model = build_large_mlp(self.LAYER_DIMS, name="mlp_200m")
        model_path = tmp_path / "mlp_200m.onnx"
        onnx.save(model, str(model_path))
        del model
        gc.collect()

        graph, *_ = run_pipeline_stages_1_to_4(
            model_path, tmp_path, self.LAYER_DIMS
        )

        gvg = GoldenVectorGenerator(graph)
        vectors = gvg.generate_random_vectors(n=3, seed=42)
        passes, total, mismatches = gvg.verify_vectors(vectors)
        assert passes == total


# ---------------------------------------------------------------------------
# 500M Model: 11264→11264→11264→11264→11264 (≈508M params)
# ---------------------------------------------------------------------------


class TestModel500M:
    """Full pipeline test with ~500M parameter MLP."""

    LAYER_DIMS = [11264, 11264, 11264, 11264, 11264]
    EXPECTED_PARAMS_MIN = 500_000_000
    EXPECTED_PARAMS_MAX = 520_000_000

    def test_param_count(self, tmp_path: Path):
        """Verify model has ~508M parameters."""
        model = build_large_mlp(self.LAYER_DIMS, name="mlp_500m")
        total = count_params(model)
        del model
        gc.collect()
        assert self.EXPECTED_PARAMS_MIN <= total <= self.EXPECTED_PARAMS_MAX, (
            f"Expected ~508M params, got {total:,}"
        )

    def test_full_pipeline_stages_1_4(self, tmp_path: Path):
        """Stages 1-4 complete correctly on ~500M model."""
        model = build_large_mlp(self.LAYER_DIMS, name="mlp_500m")
        model_path = tmp_path / "mlp_500m.onnx"
        onnx.save(model, str(model_path))
        del model
        gc.collect()

        t0 = time.time()
        graph, schedule, memory_map, hw = run_pipeline_stages_1_to_4(
            model_path, tmp_path, self.LAYER_DIMS
        )
        elapsed = time.time() - t0

        assert graph.stage == "scheduled"
        assert len(graph.nodes) == 4

        # 11264 / 128 = 88 tiles
        for layer_info in memory_map["layers"]:
            assert layer_info["num_tiles"] == 88
            assert layer_info["weight"]["num_rows"] == 88 * 11264

        print(f"\n500M model pipeline completed in {elapsed:.1f}s")
        print(f"Total weight rows: {memory_map['weight_bank']['total_rows']:,}")
        print(f"Total bytes: {memory_map['total_bytes']:,}")

    def test_rtl_generation(self, tmp_path: Path):
        """Stage 5: RTL generation works for 500M model."""
        model = build_large_mlp(self.LAYER_DIMS, name="mlp_500m")
        model_path = tmp_path / "mlp_500m.onnx"
        onnx.save(model, str(model_path))
        del model
        gc.collect()

        graph, schedule, memory_map, hw = run_pipeline_stages_1_to_4(
            model_path, tmp_path, self.LAYER_DIMS
        )

        rtl_dir = tmp_path / "rtl_output"
        gen = RTLGenerator(
            graph=graph,
            output_dir=rtl_dir,
            weight_dir=tmp_path / "weights",
            constraints=hw,
        )
        gen.generate_all()

        assert (rtl_dir / "parameters.svh").exists()
        assert (rtl_dir / "accelerator_top.sv").exists()

        params = (rtl_dir / "parameters.svh").read_text()
        assert "11264" in params


# ---------------------------------------------------------------------------
# 1B Model: 16384→16384→16384→16384→16384 (≈1.07B params)
# ---------------------------------------------------------------------------


class TestModel1B:
    """Full pipeline test with ~1B parameter MLP.

    This is an extremely large model. Memory usage during compilation
    is expected to be ~8GB+. Tests are marked as slow.
    """

    LAYER_DIMS = [16384, 16384, 16384, 16384, 16384]
    EXPECTED_PARAMS_MIN = 1_000_000_000
    EXPECTED_PARAMS_MAX = 1_100_000_000

    def test_param_count(self, tmp_path: Path):
        """Verify model has ~1.07B parameters."""
        model = build_large_mlp(self.LAYER_DIMS, name="mlp_1b")
        total = count_params(model)
        del model
        gc.collect()
        assert self.EXPECTED_PARAMS_MIN <= total <= self.EXPECTED_PARAMS_MAX, (
            f"Expected ~1.07B params, got {total:,}"
        )

    def test_full_pipeline_stages_1_4(self, tmp_path: Path):
        """Stages 1-4 complete correctly on ~1B model."""
        model = build_large_mlp(self.LAYER_DIMS, name="mlp_1b")
        model_path = tmp_path / "mlp_1b.onnx"
        onnx.save(model, str(model_path))
        del model
        gc.collect()

        t0 = time.time()
        graph, schedule, memory_map, hw = run_pipeline_stages_1_to_4(
            model_path, tmp_path, self.LAYER_DIMS
        )
        elapsed = time.time() - t0

        assert graph.stage == "scheduled"
        assert len(graph.nodes) == 4

        # 16384 / 128 = 128 tiles
        for layer_info in memory_map["layers"]:
            assert layer_info["num_tiles"] == 128
            assert layer_info["weight"]["num_rows"] == 128 * 16384

        print(f"\n1B model pipeline completed in {elapsed:.1f}s")
        print(f"Total weight rows: {memory_map['weight_bank']['total_rows']:,}")
        print(f"Total bytes: {memory_map['total_bytes']:,}")

    def test_rtl_generation(self, tmp_path: Path):
        """Stage 5: RTL generation works for 1B model."""
        model = build_large_mlp(self.LAYER_DIMS, name="mlp_1b")
        model_path = tmp_path / "mlp_1b.onnx"
        onnx.save(model, str(model_path))
        del model
        gc.collect()

        graph, schedule, memory_map, hw = run_pipeline_stages_1_to_4(
            model_path, tmp_path, self.LAYER_DIMS
        )

        rtl_dir = tmp_path / "rtl_output"
        gen = RTLGenerator(
            graph=graph,
            output_dir=rtl_dir,
            weight_dir=tmp_path / "weights",
            constraints=hw,
        )
        gen.generate_all()

        assert (rtl_dir / "parameters.svh").exists()
        assert (rtl_dir / "accelerator_top.sv").exists()

        params = (rtl_dir / "parameters.svh").read_text()
        assert "16384" in params

    def test_golden_vectors(self, tmp_path: Path):
        """Golden vectors work for 1B model (small set)."""
        model = build_large_mlp(self.LAYER_DIMS, name="mlp_1b")
        model_path = tmp_path / "mlp_1b.onnx"
        onnx.save(model, str(model_path))
        del model
        gc.collect()

        graph, *_ = run_pipeline_stages_1_to_4(
            model_path, tmp_path, self.LAYER_DIMS
        )

        gvg = GoldenVectorGenerator(graph)
        vectors = gvg.generate_random_vectors(n=2, seed=42)
        passes, total, mismatches = gvg.verify_vectors(vectors)
        assert passes == total


# ---------------------------------------------------------------------------
# Cross-scale validation tests
# ---------------------------------------------------------------------------


class TestCrossScale:
    """Verify compiler correctness properties hold across all scales."""

    @pytest.mark.parametrize(
        "layer_dims,label",
        [
            ([2048, 2048, 2048, 2048, 2048], "~17M"),
            ([4096, 4096, 4096, 4096, 4096], "~67M"),
            ([5120, 5120, 5120, 5120, 5120], "~105M"),
        ],
    )
    def test_schedule_cycle_formula(self, tmp_path: Path, layer_dims: list, label: str):
        """Verify cycle formula: cycles_per_tile = IN_DIM + 149 at all scales."""
        model = build_large_mlp(layer_dims, name=f"mlp_{label}")
        model_path = tmp_path / f"mlp_{label}.onnx"
        onnx.save(model, str(model_path))
        del model
        gc.collect()

        graph, schedule, memory_map, hw = run_pipeline_stages_1_to_4(
            model_path, tmp_path / label, layer_dims
        )

        for node_name in graph.topological_order():
            node = graph.nodes[node_name]
            ls = node.schedule_info
            attrs = node.fused_attrs
            expected_cycles_per_tile = attrs.input_dim + 149
            assert ls.cycles_per_tile == expected_cycles_per_tile, (
                f"Layer {ls.layer_name}: expected {expected_cycles_per_tile}, "
                f"got {ls.cycles_per_tile}"
            )

    @pytest.mark.parametrize(
        "layer_dims",
        [
            [2048, 2048, 2048, 2048, 2048],
            [4096, 4096, 4096, 4096, 4096],
        ],
    )
    def test_no_weight_address_overlaps(self, tmp_path: Path, layer_dims: list):
        """Weight SRAM addresses don't overlap across layers."""
        model = build_large_mlp(layer_dims, name="overlap_test")
        model_path = tmp_path / "overlap_test.onnx"
        onnx.save(model, str(model_path))
        del model
        gc.collect()

        graph, schedule, memory_map, hw = run_pipeline_stages_1_to_4(
            model_path, tmp_path, layer_dims
        )

        layers = memory_map["layers"]
        for i in range(len(layers) - 1):
            cur_end = layers[i]["weight"]["start_row"] + layers[i]["weight"]["num_rows"]
            next_start = layers[i + 1]["weight"]["start_row"]
            assert cur_end == next_start, (
                f"Weight address gap or overlap between layers {i} and {i+1}: "
                f"cur_end={cur_end}, next_start={next_start}"
            )

    @pytest.mark.parametrize(
        "layer_dims",
        [
            [2048, 2048, 2048, 2048, 2048],
            [4096, 4096, 4096, 4096, 4096],
        ],
    )
    def test_testbench_generation(self, tmp_path: Path, layer_dims: list):
        """Stage 6: Testbench generation works at scale."""
        model = build_large_mlp(layer_dims, name="tb_test")
        model_path = tmp_path / "tb_test.onnx"
        onnx.save(model, str(model_path))
        del model
        gc.collect()

        graph, schedule, memory_map, hw = run_pipeline_stages_1_to_4(
            model_path, tmp_path, layer_dims
        )

        rtl_dir = tmp_path / "rtl_output"
        gen = RTLGenerator(
            graph=graph,
            output_dir=rtl_dir,
            weight_dir=tmp_path / "weights",
            constraints=hw,
        )
        gen.generate_all()

        # Stage 6: Verification
        tb_dir = tmp_path / "testbench"
        verif_config = VerifConfig(
            graph=graph,
            output_dir=tb_dir,
            weight_dir=tmp_path / "weights",
            num_random_vectors=3,
            num_adversarial_vectors=5,
            constraints=hw,
        )
        verif_gen = VerifGenerator(verif_config)
        verif_gen.generate_all()

        assert (tb_dir / "tb_accelerator.sv").exists()
        assert (tb_dir / "coverage.json").exists()
        assert (tb_dir / "cocotb" / "test_accelerator.py").exists()
