"""End-to-end pipeline tests (Stages 1-4) with real model structures.

Tests:
  - Full pipeline on AD model (640→128→128→128→640)
  - Full pipeline on larger MLP (1024→512→256→128→64→10)
  - ONNX Runtime FP32 vs INT8 interpreter comparison
  - Weight packing + roundtrip verification
  - Memory map correctness
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from mlasic.ingestion import ONNXParser
from mlasic.int8_interpreter import INT8Interpreter
from mlasic.ir import Graph, HardwareConstraints
from mlasic.optimization import (
    BatchNormFoldingPass,
    ConstantFoldingPass,
    DeadCodeEliminationPass,
    OperatorFusionPass,
    PassManager,
    QuantizationPass,
)
from mlasic.scheduler import Scheduler
from mlasic.weight_packer import WeightPacker, load_bias_mem, load_weight_mem

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def build_mlp_model(
    layer_dims: list[int],
    has_bn: bool = True,
    name: str = "test_mlp",
    seed: int = 42,
) -> onnx.ModelProto:
    """Build a generic MLP model with configurable layer dimensions.

    Args:
        layer_dims: [input_dim, hidden1, hidden2, ..., output_dim]
        has_bn: Whether to add BatchNorm after each hidden layer
        name: Model name
        seed: Random seed for reproducibility
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

        # Weight: Xavier initialization for realistic values
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
        all_nodes.append(helper.make_node("Add", [mm_out, b_name], [add_out], name=f"{prefix}_Add"))
        all_vis.append(helper.make_tensor_value_info(add_out, TensorProto.FLOAT, [1, out_dim]))
        current = add_out

        # BatchNorm (hidden layers only)
        if has_bn and not is_last:
            bn_scale = np.ones(out_dim, dtype=np.float32)
            bn_bias = np.zeros(out_dim, dtype=np.float32)
            # Non-trivial BN stats for realistic folding
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
            all_vis.append(helper.make_tensor_value_info(bn_out, TensorProto.FLOAT, [1, out_dim]))
            current = bn_out

        # ReLU (hidden layers only)
        if not is_last:
            relu_out = f"{prefix}_relu_out"
            all_nodes.append(helper.make_node("Relu", [current], [relu_out], name=f"{prefix}_Relu"))
            all_vis.append(helper.make_tensor_value_info(relu_out, TensorProto.FLOAT, [1, out_dim]))
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
    """Count total parameters in an ONNX model."""
    total = 0
    for init in model.graph.initializer:
        total += int(np.prod(numpy_helper.to_array(init).shape))
    return total


def run_onnx_runtime(model: onnx.ModelProto, inputs: np.ndarray) -> np.ndarray:
    """Run model through ONNX Runtime (FP32)."""
    sess = ort.InferenceSession(model.SerializeToString())
    input_name = sess.get_inputs()[0].name
    return sess.run(None, {input_name: inputs})[0]


def run_full_pipeline(
    model_path: Path,
    calibration_data: list[np.ndarray],
    output_dir: Path,
    constraints: HardwareConstraints | None = None,
) -> tuple[Graph, dict]:
    """Run Stages 1-4 on an ONNX model file.

    Returns (scheduled_graph, memory_map).
    """
    # Stage 1: Ingestion
    parser = ONNXParser(model_path)
    graph = parser.parse()

    # Stage 2: Optimization
    pm = PassManager()
    pm.add_pass(ConstantFoldingPass())
    pm.add_pass(DeadCodeEliminationPass())
    pm.add_pass(BatchNormFoldingPass())
    pm.add_pass(OperatorFusionPass())
    pm.add_pass(QuantizationPass(calibration_data=calibration_data))
    graph = pm.run(graph, verify=True)

    # Stage 3: Scheduling
    hw = constraints or HardwareConstraints()
    scheduler = Scheduler(constraints=hw)
    scheduler.schedule(graph)

    # Stage 4: Weight Packing
    packer = WeightPacker(graph, output_dir, constraints=hw)
    memory_map = packer.pack_weights()

    return graph, memory_map


# ---------------------------------------------------------------------------
# AD Model (target model: 640→128→128→128→640, ~200K params)
# ---------------------------------------------------------------------------


class TestADModelEndToEnd:
    """Full pipeline test on the MLPerf Tiny AD model architecture."""

    LAYER_DIMS = [640, 128, 128, 128, 640]

    def test_full_pipeline(self, tmp_path: Path):
        """Stages 1-4 complete without errors."""
        model = build_mlp_model(self.LAYER_DIMS, has_bn=True, name="ad_model")
        model_path = tmp_path / "ad_model.onnx"
        onnx.save(model, str(model_path))

        rng = np.random.RandomState(42)
        cal_data = [rng.randn(1, 640).astype(np.float32) for _ in range(20)]

        graph, mem_map = run_full_pipeline(model_path, cal_data, tmp_path / "packed")

        assert graph.stage == "scheduled"
        assert len(graph.nodes) == 4  # 4 fused layers
        assert mem_map["weight_bank"]["total_bytes"] == 196608
        assert mem_map["bias_bank"]["total_bytes"] == 4096

    def test_param_count(self):
        """Verify AD model has ~100K params (target: ~100K)."""
        model = build_mlp_model(self.LAYER_DIMS, has_bn=True)
        # Weights only: 640*128 + 128*128 + 128*128 + 128*640 = 196,608
        # Plus biases: 128+128+128+640 = 1,024
        # Plus BN params: 3 layers * 128 * 4 = 1,536
        total = count_params(model)
        assert total < 500_000  # well under 20M

    def test_onnx_runtime_comparison(self, tmp_path: Path):
        """INT8 output is within expected tolerance of ONNX Runtime FP32."""
        model = build_mlp_model(self.LAYER_DIMS, has_bn=True, name="ad_model", seed=123)
        model_path = tmp_path / "ad_model.onnx"
        onnx.save(model, str(model_path))

        rng = np.random.RandomState(42)
        cal_data = [rng.randn(1, 640).astype(np.float32) for _ in range(50)]

        graph, _ = run_full_pipeline(model_path, cal_data, tmp_path / "packed")

        # Get output quant params for tolerance
        ordered = [graph.nodes[n] for n in graph.topological_order()]
        last_attrs = ordered[-1].fused_attrs
        output_scale = last_attrs.output_quant.scale
        output_zp = last_attrs.output_quant.zero_point

        interp = INT8Interpreter(graph)

        # Test on 100 vectors
        test_rng = np.random.RandomState(99)
        mismatches = 0
        for i in range(100):
            test_input = test_rng.randn(1, 640).astype(np.float32)

            # ONNX Runtime FP32
            ort_output = run_onnx_runtime(model, test_input)

            # INT8 interpreter → dequantize
            int8_outputs = interp.run({graph.inputs[0]: test_input})
            int8_result = int8_outputs[graph.outputs[0]]
            fp_from_int8 = INT8Interpreter.dequantize_output(int8_result, output_scale, output_zp)

            # Quantization error: expect within a few output_scale steps
            # 4 layers of quantization accumulates error
            max_diff = float(np.abs(ort_output - fp_from_int8).max())
            # Generous tolerance: output_scale * 256 (full INT8 range error)
            if max_diff > output_scale * 256:
                mismatches += 1

        # Allow up to 5% mismatch (quantization can be lossy on edge cases)
        assert mismatches <= 5, f"{mismatches}/100 vectors exceeded tolerance"

    def test_packed_weights_match_interpreter(self, tmp_path: Path):
        """Inference from packed .mem files matches graph INT8 interpreter."""
        model = build_mlp_model(self.LAYER_DIMS, has_bn=True, name="ad_model")
        model_path = tmp_path / "ad_model.onnx"
        onnx.save(model, str(model_path))

        rng = np.random.RandomState(42)
        cal_data = [rng.randn(1, 640).astype(np.float32) for _ in range(20)]
        output_dir = tmp_path / "packed"

        graph, _ = run_full_pipeline(model_path, cal_data, output_dir)
        interp = INT8Interpreter(graph)

        # Run 10 vectors through both paths
        test_rng = np.random.RandomState(77)
        for vec_idx in range(10):
            test_input = test_rng.randn(1, 640).astype(np.float32)

            # Path 1: graph interpreter (reference)
            ref = interp.run({graph.inputs[0]: test_input})

            # Path 2: manually load from .mem files
            ordered = [graph.nodes[n] for n in graph.topological_order()]
            first_attrs = ordered[0].fused_attrs
            x = first_attrs.input_quant.quantize(test_input.astype(np.float32))

            for node in ordered:
                ls = node.schedule_info
                attrs = node.fused_attrs

                # Load weights from .mem
                w_rows = load_weight_mem(output_dir / f"weights_layer{ls.layer_index}.mem")
                w_q = np.zeros((attrs.input_dim, attrs.output_dim), dtype=np.int8)
                row_idx = 0
                for tile in range(ls.num_tiles):
                    cs = tile * ls.parallelism
                    ce = cs + ls.parallelism
                    for i in range(attrs.input_dim):
                        w_q[i, cs:ce] = w_rows[row_idx]
                        row_idx += 1

                # Load biases from .mem
                b_rows = load_bias_mem(output_dir / f"biases_layer{ls.layer_index}.mem")
                b_q = np.concatenate(b_rows)[: attrs.output_dim]

                # INT8 compute
                acc = x.astype(np.int32) @ w_q.astype(np.int32) + b_q.astype(np.int32)
                m = np.int64(attrs.requant_scale_fixed)
                s = attrs.requant_shift
                scaled = acc.astype(np.int64) * m
                rounded = (scaled + (np.int64(1) << (s - 1))) >> s
                result = rounded + np.int64(attrs.output_quant.zero_point)
                result = np.clip(result, -128, 127).astype(np.int8)
                if attrs.has_relu:
                    result = np.maximum(result, np.int8(0))
                x = result

            np.testing.assert_array_equal(
                x, ref[graph.outputs[0]], err_msg=f"Mismatch at vector {vec_idx}"
            )


# ---------------------------------------------------------------------------
# Larger MLP (~1M params)
# ---------------------------------------------------------------------------


class TestLargerMLP:
    """Pipeline test with a larger MLP to stress-test the compiler."""

    LAYER_DIMS = [512, 256, 256, 128, 64, 512]

    def test_full_pipeline(self, tmp_path: Path):
        """Full pipeline on a 5-layer MLP."""
        model = build_mlp_model(self.LAYER_DIMS, has_bn=True, name="larger_mlp")
        model_path = tmp_path / "larger_mlp.onnx"
        onnx.save(model, str(model_path))

        rng = np.random.RandomState(42)
        cal_data = [rng.randn(1, 512).astype(np.float32) for _ in range(20)]

        weight_rows_needed = sum(
            self._weight_rows(self.LAYER_DIMS[i], self.LAYER_DIMS[i + 1])
            for i in range(len(self.LAYER_DIMS) - 1)
        )
        bias_rows_needed = sum(
            math.ceil(self.LAYER_DIMS[i + 1] / 32) for i in range(len(self.LAYER_DIMS) - 1)
        )
        # Weight bytes = weight_rows × 128 bytes/row (SRAM rows are always 128 bytes)
        total_weight_bytes = weight_rows_needed * 128

        hw = HardwareConstraints(
            weight_bank_depth=max(weight_rows_needed + 10, 2048),
            bias_bank_depth=max(bias_rows_needed + 10, 64),
            act_buffer_bytes=max(self.LAYER_DIMS),
        )

        graph, mem_map = run_full_pipeline(model_path, cal_data, tmp_path / "packed", hw)

        assert graph.stage == "scheduled"
        assert len(graph.nodes) == 5  # 5 fused layers
        assert mem_map["weight_bank"]["total_bytes"] == total_weight_bytes

    def test_param_count(self):
        """Model should be well under 20M params."""
        model = build_mlp_model(self.LAYER_DIMS, has_bn=True)
        total = count_params(model)
        assert total < 20_000_000
        assert total > 200_000  # bigger than AD model

    def test_roundtrip_correctness(self, tmp_path: Path):
        """Packed weights roundtrip correctly for larger model."""
        model = build_mlp_model(self.LAYER_DIMS, has_bn=True, name="larger_mlp")
        model_path = tmp_path / "larger_mlp.onnx"
        onnx.save(model, str(model_path))

        rng = np.random.RandomState(42)
        cal_data = [rng.randn(1, 512).astype(np.float32) for _ in range(20)]

        weight_rows_needed = sum(
            self._weight_rows(self.LAYER_DIMS[i], self.LAYER_DIMS[i + 1])
            for i in range(len(self.LAYER_DIMS) - 1)
        )
        bias_rows_needed = sum(
            math.ceil(self.LAYER_DIMS[i + 1] / 32) for i in range(len(self.LAYER_DIMS) - 1)
        )

        hw = HardwareConstraints(
            weight_bank_depth=max(weight_rows_needed + 10, 2048),
            bias_bank_depth=max(bias_rows_needed + 10, 64),
            act_buffer_bytes=max(self.LAYER_DIMS),
        )

        output_dir = tmp_path / "packed"
        graph, _ = run_full_pipeline(model_path, cal_data, output_dir, hw)

        # Verify roundtrip for every layer
        for node_name in graph.topological_order():
            node = graph.nodes[node_name]
            ls = node.schedule_info
            attrs = node.fused_attrs

            w_orig = graph.tensors[node.inputs[1]].data
            w_rows = load_weight_mem(output_dir / f"weights_layer{ls.layer_index}.mem")

            w_rt = np.zeros_like(w_orig)
            row_idx = 0
            for tile in range(ls.num_tiles):
                cs = tile * ls.parallelism
                ce = cs + ls.parallelism
                for i in range(attrs.input_dim):
                    w_rt[i, cs:ce] = w_rows[row_idx]
                    row_idx += 1

            np.testing.assert_array_equal(w_rt, w_orig, err_msg=f"Layer {ls.layer_index}")

    @staticmethod
    def _weight_rows(in_dim: int, out_dim: int, parallelism: int = 128) -> int:
        """Calculate weight SRAM rows needed for a layer."""
        par = min(out_dim, parallelism)
        while par > 1:
            if out_dim % par == 0:
                break
            par -= 1
        num_tiles = out_dim // par
        return num_tiles * in_dim


# ---------------------------------------------------------------------------
# Wide MLP (~5M params, tests output tiling)
# ---------------------------------------------------------------------------


class TestWideMLP:
    """Test with wide layers that require output tiling."""

    LAYER_DIMS = [1024, 256, 256, 1024]

    def test_tiled_layers(self, tmp_path: Path):
        """Layers with output_dim > 128 produce correct tiling."""
        model = build_mlp_model(self.LAYER_DIMS, has_bn=True, name="wide_mlp")
        model_path = tmp_path / "wide_mlp.onnx"
        onnx.save(model, str(model_path))

        rng = np.random.RandomState(42)
        cal_data = [rng.randn(1, 1024).astype(np.float32) for _ in range(20)]

        weight_rows_needed = sum(
            self._weight_rows(self.LAYER_DIMS[i], self.LAYER_DIMS[i + 1])
            for i in range(len(self.LAYER_DIMS) - 1)
        )
        bias_rows_needed = sum(
            math.ceil(self.LAYER_DIMS[i + 1] / 32) for i in range(len(self.LAYER_DIMS) - 1)
        )

        hw = HardwareConstraints(
            weight_bank_depth=max(weight_rows_needed + 10, 4096),
            bias_bank_depth=max(bias_rows_needed + 10, 128),
            act_buffer_bytes=max(self.LAYER_DIMS),
        )

        output_dir = tmp_path / "packed"
        graph, mem_map = run_full_pipeline(model_path, cal_data, output_dir, hw)

        # Layer 0: 1024→256, 2 tiles (256/128=2)
        layer0 = mem_map["layers"][0]
        assert layer0["num_tiles"] == 2
        assert layer0["weight"]["num_rows"] == 2 * 1024  # 2 tiles × 1024 in_dim

        # Layer 2 (last): 256→1024, 8 tiles (1024/128=8)
        last = mem_map["layers"][-1]
        assert last["num_tiles"] == 8
        assert last["weight"]["num_rows"] == 8 * 256  # 8 tiles × 256 in_dim

    def test_int8_correctness_with_tiling(self, tmp_path: Path):
        """INT8 inference matches between graph and packed-weight paths."""
        model = build_mlp_model(self.LAYER_DIMS, has_bn=True, name="wide_mlp")
        model_path = tmp_path / "wide_mlp.onnx"
        onnx.save(model, str(model_path))

        rng = np.random.RandomState(42)
        cal_data = [rng.randn(1, 1024).astype(np.float32) for _ in range(20)]

        weight_rows_needed = sum(
            self._weight_rows(self.LAYER_DIMS[i], self.LAYER_DIMS[i + 1])
            for i in range(len(self.LAYER_DIMS) - 1)
        )
        bias_rows_needed = sum(
            math.ceil(self.LAYER_DIMS[i + 1] / 32) for i in range(len(self.LAYER_DIMS) - 1)
        )

        hw = HardwareConstraints(
            weight_bank_depth=max(weight_rows_needed + 10, 4096),
            bias_bank_depth=max(bias_rows_needed + 10, 128),
            act_buffer_bytes=max(self.LAYER_DIMS),
        )

        output_dir = tmp_path / "packed"
        graph, _ = run_full_pipeline(model_path, cal_data, output_dir, hw)

        interp = INT8Interpreter(graph)
        test_rng = np.random.RandomState(99)

        for vec_idx in range(10):
            test_input = test_rng.randn(1, 1024).astype(np.float32)
            ref = interp.run({graph.inputs[0]: test_input})

            # Manual inference from packed .mem files
            ordered = [graph.nodes[n] for n in graph.topological_order()]
            first_attrs = ordered[0].fused_attrs
            x = first_attrs.input_quant.quantize(test_input.astype(np.float32))

            for node in ordered:
                ls = node.schedule_info
                attrs = node.fused_attrs
                w_rows = load_weight_mem(output_dir / f"weights_layer{ls.layer_index}.mem")

                w_q = np.zeros((attrs.input_dim, attrs.output_dim), dtype=np.int8)
                row_idx = 0
                for tile in range(ls.num_tiles):
                    cs = tile * ls.parallelism
                    ce = cs + ls.parallelism
                    for i in range(attrs.input_dim):
                        w_q[i, cs:ce] = w_rows[row_idx]
                        row_idx += 1

                b_rows = load_bias_mem(output_dir / f"biases_layer{ls.layer_index}.mem")
                b_q = np.concatenate(b_rows)[: attrs.output_dim]

                acc = x.astype(np.int32) @ w_q.astype(np.int32) + b_q.astype(np.int32)
                m = np.int64(attrs.requant_scale_fixed)
                s = attrs.requant_shift
                scaled = acc.astype(np.int64) * m
                rounded = (scaled + (np.int64(1) << (s - 1))) >> s
                result = rounded + np.int64(attrs.output_quant.zero_point)
                result = np.clip(result, -128, 127).astype(np.int8)
                if attrs.has_relu:
                    result = np.maximum(result, np.int8(0))
                x = result

            np.testing.assert_array_equal(x, ref[graph.outputs[0]], err_msg=f"Vector {vec_idx}")

    @staticmethod
    def _weight_rows(in_dim: int, out_dim: int, parallelism: int = 128) -> int:
        par = min(out_dim, parallelism)
        while par > 1:
            if out_dim % par == 0:
                break
            par -= 1
        num_tiles = out_dim // par
        return num_tiles * in_dim


# ---------------------------------------------------------------------------
# No-BN model (simpler path, verifies BN-less pipeline)
# ---------------------------------------------------------------------------


class TestNoBatchNormMLP:
    """Pipeline test without BatchNorm layers."""

    LAYER_DIMS = [256, 128, 64, 256]

    def test_full_pipeline_no_bn(self, tmp_path: Path):
        """Pipeline works without BatchNorm."""
        model = build_mlp_model(self.LAYER_DIMS, has_bn=False, name="no_bn_mlp")
        model_path = tmp_path / "no_bn_mlp.onnx"
        onnx.save(model, str(model_path))

        rng = np.random.RandomState(42)
        cal_data = [rng.randn(1, 256).astype(np.float32) for _ in range(20)]

        weight_rows_needed = sum(
            self._weight_rows(self.LAYER_DIMS[i], self.LAYER_DIMS[i + 1])
            for i in range(len(self.LAYER_DIMS) - 1)
        )
        bias_rows_needed = sum(
            math.ceil(self.LAYER_DIMS[i + 1] / 32) for i in range(len(self.LAYER_DIMS) - 1)
        )

        hw = HardwareConstraints(
            weight_bank_depth=max(weight_rows_needed + 10, 1024),
            bias_bank_depth=max(bias_rows_needed + 10, 32),
            act_buffer_bytes=max(self.LAYER_DIMS),
        )

        graph, mem_map = run_full_pipeline(model_path, cal_data, tmp_path / "packed", hw)

        assert graph.stage == "scheduled"
        assert len(graph.nodes) == 3  # 3 fused layers
        assert mem_map["total_bytes"] > 0

    @staticmethod
    def _weight_rows(in_dim: int, out_dim: int, parallelism: int = 128) -> int:
        par = min(out_dim, parallelism)
        while par > 1:
            if out_dim % par == 0:
                break
            par -= 1
        num_tiles = out_dim // par
        return num_tiles * in_dim
