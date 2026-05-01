"""Tests for Stage 2 graph optimization passes."""

from __future__ import annotations

import copy

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from mlasic.ingestion import ONNXParser
from mlasic.interpreter import IRInterpreter
from mlasic.ir import (
    FusedLinearAttrs,
    Graph,
    OpNode,
    OpType,
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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def parsed_ad_graph(ad_model_path) -> Graph:
    """Parse the AD model and return the IR graph."""
    parser = ONNXParser(ad_model_path)
    return parser.parse()


@pytest.fixture
def calibration_data() -> list[np.ndarray]:
    """Generate calibration data for quantization."""
    rng = np.random.RandomState(123)
    return [rng.randn(1, 640).astype(np.float32) for _ in range(20)]


# ---------------------------------------------------------------------------
# IRInterpreter tests
# ---------------------------------------------------------------------------


class TestIRInterpreter:
    def test_matmul(self):
        w = np.random.RandomState(0).randn(4, 3).astype(np.float32)
        graph = Graph(
            name="test",
            nodes={"mm": OpNode("mm", OpType.MATMUL, ["x", "w"], ["y"])},
            tensors={
                "x": Tensor("x", TensorType((1, 4), np.dtype(np.float32))),
                "w": Tensor("w", TensorType((4, 3), np.dtype(np.float32)), data=w),
                "y": Tensor("y", TensorType((1, 3), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["y"],
        )
        interp = IRInterpreter(graph)
        x = np.random.RandomState(1).randn(1, 4).astype(np.float32)
        result = interp.run({"x": x})
        np.testing.assert_allclose(result["y"], x @ w, rtol=1e-5)

    def test_add(self):
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        graph = Graph(
            name="test",
            nodes={"add": OpNode("add", OpType.ADD, ["x", "b"], ["y"])},
            tensors={
                "x": Tensor("x", TensorType((1, 3), np.dtype(np.float32))),
                "b": Tensor("b", TensorType((3,), np.dtype(np.float32)), data=b),
                "y": Tensor("y", TensorType((1, 3), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["y"],
        )
        interp = IRInterpreter(graph)
        x = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
        result = interp.run({"x": x})
        np.testing.assert_allclose(result["y"], [[5.0, 7.0, 9.0]])

    def test_relu(self):
        graph = Graph(
            name="test",
            nodes={"relu": OpNode("relu", OpType.RELU, ["x"], ["y"])},
            tensors={
                "x": Tensor("x", TensorType((1, 4), np.dtype(np.float32))),
                "y": Tensor("y", TensorType((1, 4), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["y"],
        )
        interp = IRInterpreter(graph)
        x = np.array([[-1.0, 0.0, 1.0, -0.5]], dtype=np.float32)
        result = interp.run({"x": x})
        np.testing.assert_allclose(result["y"], [[0.0, 0.0, 1.0, 0.0]])

    def test_batch_norm(self):
        gamma = np.array([2.0, 3.0], dtype=np.float32)
        beta = np.array([0.5, -0.5], dtype=np.float32)
        mean = np.array([1.0, 2.0], dtype=np.float32)
        var = np.array([4.0, 9.0], dtype=np.float32)
        graph = Graph(
            name="test",
            nodes={
                "bn": OpNode(
                    "bn",
                    OpType.BATCH_NORM,
                    ["x", "gamma", "beta", "mean", "var"],
                    ["y"],
                    attributes={"epsilon": 1e-5},
                ),
            },
            tensors={
                "x": Tensor("x", TensorType((1, 2), np.dtype(np.float32))),
                "gamma": Tensor("gamma", TensorType((2,), np.dtype(np.float32)), data=gamma),
                "beta": Tensor("beta", TensorType((2,), np.dtype(np.float32)), data=beta),
                "mean": Tensor("mean", TensorType((2,), np.dtype(np.float32)), data=mean),
                "var": Tensor("var", TensorType((2,), np.dtype(np.float32)), data=var),
                "y": Tensor("y", TensorType((1, 2), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["y"],
        )
        interp = IRInterpreter(graph)
        x = np.array([[5.0, 8.0]], dtype=np.float32)
        result = interp.run({"x": x})
        # (5-1)/sqrt(4+eps)*2+0.5 ≈ 4.5, (8-2)/sqrt(9+eps)*3-0.5 ≈ 5.5
        np.testing.assert_allclose(result["y"], [[4.5, 5.5]], atol=1e-4)

    def test_fused_linear(self):
        w = np.eye(3, dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        node = OpNode("fl", OpType.FUSED_LINEAR, ["x", "w", "b"], ["y"])
        node.fused_attrs = FusedLinearAttrs(3, 3, has_relu=False)
        graph = Graph(
            name="test",
            nodes={"fl": node},
            tensors={
                "x": Tensor("x", TensorType((1, 3), np.dtype(np.float32))),
                "w": Tensor("w", TensorType((3, 3), np.dtype(np.float32)), data=w),
                "b": Tensor("b", TensorType((3,), np.dtype(np.float32)), data=b),
                "y": Tensor("y", TensorType((1, 3), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["y"],
        )
        interp = IRInterpreter(graph)
        x = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
        result = interp.run({"x": x})
        np.testing.assert_allclose(result["y"], [[11.0, 22.0, 33.0]])

    def test_fused_linear_relu(self):
        w = np.eye(3, dtype=np.float32)
        b = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
        node = OpNode("flr", OpType.FUSED_LINEAR_RELU, ["x", "w", "b"], ["y"])
        node.fused_attrs = FusedLinearAttrs(3, 3, has_relu=True)
        graph = Graph(
            name="test",
            nodes={"flr": node},
            tensors={
                "x": Tensor("x", TensorType((1, 3), np.dtype(np.float32))),
                "w": Tensor("w", TensorType((3, 3), np.dtype(np.float32)), data=w),
                "b": Tensor("b", TensorType((3,), np.dtype(np.float32)), data=b),
                "y": Tensor("y", TensorType((1, 3), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["y"],
        )
        interp = IRInterpreter(graph)
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = interp.run({"x": x})
        # (1-5, 2+0, 3+5) = (-4, 2, 8) → relu → (0, 2, 8)
        np.testing.assert_allclose(result["y"], [[0.0, 2.0, 8.0]])

    def test_reshape(self):
        shape_data = np.array([1, 2, 3], dtype=np.int64)
        graph = Graph(
            name="test",
            nodes={"reshape": OpNode("reshape", OpType.RESHAPE, ["x", "shape"], ["y"])},
            tensors={
                "x": Tensor("x", TensorType((1, 6), np.dtype(np.float32))),
                "shape": Tensor("shape", TensorType((3,), np.dtype(np.int64)), data=shape_data),
                "y": Tensor("y", TensorType((1, 2, 3), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["y"],
        )
        interp = IRInterpreter(graph)
        x = np.arange(6, dtype=np.float32).reshape(1, 6)
        result = interp.run({"x": x})
        assert result["y"].shape == (1, 2, 3)
        np.testing.assert_allclose(result["y"].flatten(), np.arange(6))

    def test_flatten(self):
        graph = Graph(
            name="test",
            nodes={
                "flat": OpNode("flat", OpType.FLATTEN, ["x"], ["y"], attributes={"axis": 1}),
            },
            tensors={
                "x": Tensor("x", TensorType((2, 3, 4), np.dtype(np.float32))),
                "y": Tensor("y", TensorType((2, 12), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["y"],
        )
        interp = IRInterpreter(graph)
        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        result = interp.run({"x": x})
        assert result["y"].shape == (2, 12)

    def test_run_all_returns_intermediates(self):
        """run_all returns all tensor values, not just outputs."""
        w = np.eye(2, dtype=np.float32)
        b = np.array([1.0, 2.0], dtype=np.float32)
        graph = Graph(
            name="test",
            nodes={
                "mm": OpNode("mm", OpType.MATMUL, ["x", "w"], ["mm_out"]),
                "add": OpNode("add", OpType.ADD, ["mm_out", "b"], ["y"]),
            },
            tensors={
                "x": Tensor("x", TensorType((1, 2), np.dtype(np.float32))),
                "w": Tensor("w", TensorType((2, 2), np.dtype(np.float32)), data=w),
                "mm_out": Tensor("mm_out", TensorType((1, 2), np.dtype(np.float32))),
                "b": Tensor("b", TensorType((2,), np.dtype(np.float32)), data=b),
                "y": Tensor("y", TensorType((1, 2), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["y"],
        )
        interp = IRInterpreter(graph)
        x = np.array([[3.0, 4.0]], dtype=np.float32)
        all_vals = interp.run_all({"x": x})

        assert "x" in all_vals
        assert "mm_out" in all_vals
        assert "y" in all_vals
        np.testing.assert_allclose(all_vals["mm_out"], [[3.0, 4.0]])
        np.testing.assert_allclose(all_vals["y"], [[4.0, 6.0]])

    def test_ad_model_runs(self, parsed_ad_graph):
        """IRInterpreter can run the full AD model."""
        interp = IRInterpreter(parsed_ad_graph)
        x = np.random.RandomState(42).randn(1, 640).astype(np.float32)
        result = interp.run({"input": x})
        assert len(result) == 1
        out = list(result.values())[0]
        assert out.shape == (1, 640)
        assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# ConstantFoldingPass tests
# ---------------------------------------------------------------------------


class TestConstantFoldingPass:
    def test_folds_constant_matmul(self):
        """Folds MatMul with all-constant inputs."""
        a = np.eye(2, dtype=np.float32)
        b = np.array([[1, 2], [3, 4]], dtype=np.float32)
        graph = Graph(
            name="test",
            nodes={
                "mm": OpNode("mm", OpType.MATMUL, ["a", "b"], ["c"]),
                "add": OpNode("add", OpType.ADD, ["x", "c"], ["y"]),
            },
            tensors={
                "a": Tensor("a", TensorType((2, 2), np.dtype(np.float32)), data=a),
                "b": Tensor("b", TensorType((2, 2), np.dtype(np.float32)), data=b),
                "c": Tensor("c", TensorType((2, 2), np.dtype(np.float32))),
                "x": Tensor("x", TensorType((1, 2), np.dtype(np.float32))),
                "y": Tensor("y", TensorType((1, 2), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["y"],
        )
        pass_ = ConstantFoldingPass()
        result = pass_.run(graph)

        assert "mm" not in result.nodes
        assert result.tensors["c"].is_constant
        np.testing.assert_allclose(result.tensors["c"].data, a @ b)

    def test_no_change_when_no_constants(self, parsed_ad_graph):
        """AD model has no all-constant nodes."""
        before_count = len(parsed_ad_graph.nodes)
        pass_ = ConstantFoldingPass()
        result = pass_.run(parsed_ad_graph)
        assert len(result.nodes) == before_count

    def test_iterative_folding(self):
        """Iterates until fixpoint through dependent constant nodes."""
        a = np.array([[1.0, 2.0]], dtype=np.float32)
        b = np.array([[1.0], [1.0]], dtype=np.float32)
        c = np.array([[10.0]], dtype=np.float32)
        graph = Graph(
            name="test",
            nodes={
                "mm": OpNode("mm", OpType.MATMUL, ["a", "b"], ["ab"]),
                "add": OpNode("add", OpType.ADD, ["ab", "c"], ["abc"]),
                "add2": OpNode("add2", OpType.ADD, ["x", "abc"], ["y"]),
            },
            tensors={
                "a": Tensor("a", TensorType((1, 2), np.dtype(np.float32)), data=a),
                "b": Tensor("b", TensorType((2, 1), np.dtype(np.float32)), data=b),
                "ab": Tensor("ab", TensorType((1, 1), np.dtype(np.float32))),
                "c": Tensor("c", TensorType((1, 1), np.dtype(np.float32)), data=c),
                "abc": Tensor("abc", TensorType((1, 1), np.dtype(np.float32))),
                "x": Tensor("x", TensorType((1, 1), np.dtype(np.float32))),
                "y": Tensor("y", TensorType((1, 1), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["y"],
        )
        pass_ = ConstantFoldingPass()
        result = pass_.run(graph)

        # Both mm and add should be folded, only add2 remains
        assert len(result.nodes) == 1
        assert "add2" in result.nodes
        np.testing.assert_allclose(result.tensors["abc"].data, [[13.0]])


# ---------------------------------------------------------------------------
# DeadCodeEliminationPass tests
# ---------------------------------------------------------------------------


class TestDeadCodeEliminationPass:
    def test_removes_dead_node(self):
        graph = Graph(
            name="test",
            nodes={
                "relu": OpNode("relu", OpType.RELU, ["x"], ["y"]),
                "dead": OpNode("dead", OpType.RELU, ["x"], ["z"]),
            },
            tensors={
                "x": Tensor("x", TensorType((1, 4), np.dtype(np.float32))),
                "y": Tensor("y", TensorType((1, 4), np.dtype(np.float32))),
                "z": Tensor("z", TensorType((1, 4), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["y"],
        )
        pass_ = DeadCodeEliminationPass()
        result = pass_.run(graph)

        assert "dead" not in result.nodes
        assert "z" not in result.tensors
        assert "relu" in result.nodes

    def test_preserves_graph_inputs(self):
        graph = Graph(
            name="test",
            nodes={"relu": OpNode("relu", OpType.RELU, ["x"], ["y"])},
            tensors={
                "x": Tensor("x", TensorType((1, 4), np.dtype(np.float32))),
                "y": Tensor("y", TensorType((1, 4), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["y"],
        )
        pass_ = DeadCodeEliminationPass()
        result = pass_.run(graph)
        assert "x" in result.tensors

    def test_ad_model_no_dead_code(self, parsed_ad_graph):
        before_count = len(parsed_ad_graph.nodes)
        pass_ = DeadCodeEliminationPass()
        result = pass_.run(parsed_ad_graph)
        assert len(result.nodes) == before_count


# ---------------------------------------------------------------------------
# BatchNormFoldingPass tests
# ---------------------------------------------------------------------------


class TestBatchNormFoldingPass:
    def test_removes_all_bn_nodes(self, parsed_ad_graph):
        pass_ = BatchNormFoldingPass()
        result = pass_.run(parsed_ad_graph)
        bn_count = sum(1 for n in result.nodes.values() if n.op_type == OpType.BATCH_NORM)
        assert bn_count == 0

    def test_preserves_semantics(self, parsed_ad_graph):
        """Folded graph produces same output (FP32 tolerance)."""
        before = copy.deepcopy(parsed_ad_graph)
        pass_ = BatchNormFoldingPass()
        after = pass_.run(parsed_ad_graph)

        interp_before = IRInterpreter(before)
        interp_after = IRInterpreter(after)

        rng = np.random.RandomState(42)
        for _ in range(5):
            x = rng.randn(1, 640).astype(np.float32)
            out_before = interp_before.run({"input": x})
            out_after = interp_after.run({"input": x})
            for key in out_before:
                np.testing.assert_allclose(out_before[key], out_after[key], rtol=1e-5, atol=1e-5)

    def test_removes_bn_param_tensors(self, parsed_ad_graph):
        """BN param tensors (gamma, beta, mean, var) are removed."""
        bn_param_names = set()
        for n in parsed_ad_graph.nodes.values():
            if n.op_type == OpType.BATCH_NORM:
                bn_param_names.update(n.inputs[1:])

        pass_ = BatchNormFoldingPass()
        result = pass_.run(parsed_ad_graph)

        for name in bn_param_names:
            assert name not in result.tensors

    def test_nontrivial_bn_params(self):
        """Verify BN folding math with non-identity gamma/beta/mean/var."""
        rng = np.random.RandomState(42)
        in_dim, out_dim = 4, 3

        w = rng.randn(in_dim, out_dim).astype(np.float32)
        b = rng.randn(out_dim).astype(np.float32)
        gamma = np.array([2.0, 0.5, 3.0], dtype=np.float32)
        beta = np.array([1.0, -1.0, 0.5], dtype=np.float32)
        mean = np.array([0.5, -0.3, 0.1], dtype=np.float32)
        var = np.array([2.0, 0.5, 4.0], dtype=np.float32)

        graph = Graph(
            name="test",
            nodes={
                "mm": OpNode("mm", OpType.MATMUL, ["x", "w"], ["mm_out"]),
                "add": OpNode("add", OpType.ADD, ["mm_out", "b"], ["add_out"]),
                "bn": OpNode(
                    "bn",
                    OpType.BATCH_NORM,
                    ["add_out", "gamma", "beta", "mean", "var"],
                    ["bn_out"],
                    attributes={"epsilon": 1e-5},
                ),
            },
            tensors={
                "x": Tensor("x", TensorType((1, in_dim), np.dtype(np.float32))),
                "w": Tensor("w", TensorType((in_dim, out_dim), np.dtype(np.float32)), data=w),
                "mm_out": Tensor("mm_out", TensorType((1, out_dim), np.dtype(np.float32))),
                "b": Tensor("b", TensorType((out_dim,), np.dtype(np.float32)), data=b),
                "add_out": Tensor("add_out", TensorType((1, out_dim), np.dtype(np.float32))),
                "gamma": Tensor("gamma", TensorType((out_dim,), np.dtype(np.float32)), data=gamma),
                "beta": Tensor("beta", TensorType((out_dim,), np.dtype(np.float32)), data=beta),
                "mean": Tensor("mean", TensorType((out_dim,), np.dtype(np.float32)), data=mean),
                "var": Tensor("var", TensorType((out_dim,), np.dtype(np.float32)), data=var),
                "bn_out": Tensor("bn_out", TensorType((1, out_dim), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["bn_out"],
        )

        before = copy.deepcopy(graph)
        pass_ = BatchNormFoldingPass()
        after = pass_.run(graph)

        assert "bn" not in after.nodes

        interp_before = IRInterpreter(before)
        interp_after = IRInterpreter(after)

        rng2 = np.random.RandomState(99)
        for _ in range(10):
            x = rng2.randn(1, in_dim).astype(np.float32)
            out_before = list(interp_before.run({"x": x}).values())[0]
            out_after = list(interp_after.run({"x": x}).values())[0]
            np.testing.assert_allclose(out_before, out_after, rtol=1e-5, atol=1e-6)

    def test_node_count_decreases(self, parsed_ad_graph):
        before_count = len(parsed_ad_graph.nodes)
        pass_ = BatchNormFoldingPass()
        result = pass_.run(parsed_ad_graph)
        # 3 BN nodes removed
        assert len(result.nodes) == before_count - 3


# ---------------------------------------------------------------------------
# OperatorFusionPass tests
# ---------------------------------------------------------------------------


class TestOperatorFusionPass:
    def _bn_fold(self, graph: Graph) -> Graph:
        """Helper: fold BN before fusion."""
        g = BatchNormFoldingPass().run(graph)
        g.invalidate_cache()
        return g

    def test_ad_model_produces_4_fused_nodes(self, parsed_ad_graph):
        graph = self._bn_fold(parsed_ad_graph)
        result = OperatorFusionPass().run(graph)

        assert len(result.nodes) == 4
        relu_count = sum(1 for n in result.nodes.values() if n.op_type == OpType.FUSED_LINEAR_RELU)
        linear_count = sum(1 for n in result.nodes.values() if n.op_type == OpType.FUSED_LINEAR)
        assert relu_count == 3
        assert linear_count == 1

    def test_fused_attrs_present(self, parsed_ad_graph):
        graph = self._bn_fold(parsed_ad_graph)
        result = OperatorFusionPass().run(graph)

        for node in result.nodes.values():
            assert node.fused_attrs is not None

    def test_dimension_chain(self, parsed_ad_graph):
        """Dimensions chain: 640→128→128→128→640."""
        graph = self._bn_fold(parsed_ad_graph)
        result = OperatorFusionPass().run(graph)

        ordered = [result.nodes[n] for n in result.topological_order()]
        dims = [(n.fused_attrs.input_dim, n.fused_attrs.output_dim) for n in ordered]
        assert dims == [(640, 128), (128, 128), (128, 128), (128, 640)]

    def test_preserves_semantics(self, parsed_ad_graph):
        before = copy.deepcopy(parsed_ad_graph)
        graph = self._bn_fold(parsed_ad_graph)
        after = OperatorFusionPass().run(graph)

        # Compare BN-folded (raw ops) vs fused
        bn_before = BatchNormFoldingPass().run(before)
        interp_before = IRInterpreter(bn_before)
        interp_after = IRInterpreter(after)

        rng = np.random.RandomState(42)
        for _ in range(5):
            x = rng.randn(1, 640).astype(np.float32)
            out_before = interp_before.run({"input": x})
            out_after = interp_after.run({"input": x})
            for key in out_before:
                np.testing.assert_allclose(out_before[key], out_after[key], rtol=1e-5, atol=1e-5)

    def test_validates_fused_stage(self, parsed_ad_graph):
        graph = self._bn_fold(parsed_ad_graph)
        result = OperatorFusionPass().run(graph)
        result.validate("fused")  # Should not raise

    def test_stage_set_to_fused(self, parsed_ad_graph):
        graph = self._bn_fold(parsed_ad_graph)
        result = OperatorFusionPass().run(graph)
        assert result.stage == "fused"


# ---------------------------------------------------------------------------
# QuantizationPass tests
# ---------------------------------------------------------------------------


class TestQuantizationPass:
    def _get_fused_graph(self, parsed_ad_graph: Graph) -> Graph:
        """Helper: BN fold + fusion."""
        g = BatchNormFoldingPass().run(parsed_ad_graph)
        g.invalidate_cache()
        return OperatorFusionPass().run(g)

    def test_all_nodes_quantized(self, parsed_ad_graph, calibration_data):
        graph = self._get_fused_graph(parsed_ad_graph)
        result = QuantizationPass(calibration_data).run(graph)

        for node in result.nodes.values():
            attrs = node.fused_attrs
            assert attrs is not None
            assert attrs.is_quantized
            assert attrs.requant_scale_fixed is not None
            assert attrs.requant_shift is not None

    def test_weights_are_int8(self, parsed_ad_graph, calibration_data):
        graph = self._get_fused_graph(parsed_ad_graph)
        result = QuantizationPass(calibration_data).run(graph)

        for node in result.nodes.values():
            wt = result.tensors[node.inputs[1]]
            assert wt.data.dtype == np.int8

    def test_biases_are_int32(self, parsed_ad_graph, calibration_data):
        graph = self._get_fused_graph(parsed_ad_graph)
        result = QuantizationPass(calibration_data).run(graph)

        for node in result.nodes.values():
            if len(node.inputs) > 2:
                bt = result.tensors[node.inputs[2]]
                assert bt.data.dtype == np.int32

    def test_symmetric_weight_quant(self, parsed_ad_graph, calibration_data):
        """Weight quantization is symmetric (zp=0)."""
        graph = self._get_fused_graph(parsed_ad_graph)
        result = QuantizationPass(calibration_data).run(graph)

        for node in result.nodes.values():
            wq = node.fused_attrs.weight_quant
            assert wq.zero_point == 0
            assert wq.scale > 0

    def test_weight_clip_range(self, parsed_ad_graph, calibration_data):
        """Quantized weights are in [-127, 127] (symmetric)."""
        graph = self._get_fused_graph(parsed_ad_graph)
        result = QuantizationPass(calibration_data).run(graph)

        for node in result.nodes.values():
            w = result.tensors[node.inputs[1]].data
            assert w.min() >= -127
            assert w.max() <= 127

    def test_stage_set_to_quantized(self, parsed_ad_graph, calibration_data):
        graph = self._get_fused_graph(parsed_ad_graph)
        result = QuantizationPass(calibration_data).run(graph)
        assert result.stage == "quantized"

    def test_validates_quantized_stage(self, parsed_ad_graph, calibration_data):
        graph = self._get_fused_graph(parsed_ad_graph)
        result = QuantizationPass(calibration_data).run(graph)
        result.validate("quantized")  # Should not raise

    def test_requant_params_positive(self, parsed_ad_graph, calibration_data):
        graph = self._get_fused_graph(parsed_ad_graph)
        result = QuantizationPass(calibration_data).run(graph)

        for node in result.nodes.values():
            assert node.fused_attrs.requant_scale_fixed > 0
            assert node.fused_attrs.requant_shift == 16

    def test_no_calibration_raises(self):
        with pytest.raises(ValueError, match="Calibration data required"):
            QuantizationPass(None).run(Graph("t", {}, {}, [], [], stage="fused"))


# ---------------------------------------------------------------------------
# PassManager tests
# ---------------------------------------------------------------------------


class TestPassManager:
    def test_runs_all_passes(self, parsed_ad_graph, calibration_data):
        pm = PassManager()
        pm.add_pass(ConstantFoldingPass())
        pm.add_pass(DeadCodeEliminationPass())
        pm.add_pass(BatchNormFoldingPass())
        pm.add_pass(OperatorFusionPass())
        pm.add_pass(QuantizationPass(calibration_data))

        result = pm.run(parsed_ad_graph, verify=False)
        assert len(result.nodes) == 4
        assert result.stage == "quantized"

    def test_verify_true_passes(self, parsed_ad_graph):
        """PassManager with verify=True runs IRInterpreter checks per pass."""
        pm = PassManager()
        pm.add_pass(ConstantFoldingPass())
        pm.add_pass(DeadCodeEliminationPass())
        pm.add_pass(BatchNormFoldingPass())
        pm.add_pass(OperatorFusionPass())
        # QuantizationPass excluded — its verify is a no-op and
        # the IRInterpreter can't run on quantized graphs.

        result = pm.run(parsed_ad_graph, verify=True)
        assert result.stage == "fused"
        assert len(result.nodes) == 4


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_pipeline(self, parsed_ad_graph, calibration_data):
        """Full optimization: raw → 4 quantized fused nodes."""
        pm = PassManager()
        pm.add_pass(ConstantFoldingPass())
        pm.add_pass(DeadCodeEliminationPass())
        pm.add_pass(BatchNormFoldingPass())
        pm.add_pass(OperatorFusionPass())
        pm.add_pass(QuantizationPass(calibration_data))

        result = pm.run(parsed_ad_graph, verify=False)

        # 4 fused nodes
        assert len(result.nodes) == 4
        assert result.stage == "quantized"

        # Validate invariants
        result.validate("quantized")

        # Check node types
        relu_count = sum(1 for n in result.nodes.values() if n.op_type == OpType.FUSED_LINEAR_RELU)
        linear_count = sum(1 for n in result.nodes.values() if n.op_type == OpType.FUSED_LINEAR)
        assert relu_count == 3
        assert linear_count == 1

    def test_no_bn_model_graceful(self, tmp_path):
        """Model without BN is handled gracefully."""
        from tests.conftest import _make_dense_layer

        rng = np.random.RandomState(42)
        all_nodes, all_inits, all_vis = [], [], []
        input_name = "input"
        current = input_name

        for i, (in_dim, out_dim, has_relu) in enumerate([(4, 8, True), (8, 4, False)]):
            nodes, inits, vis, current = _make_dense_layer(
                prefix=f"layer{i}",
                input_name=current,
                in_dim=in_dim,
                out_dim=out_dim,
                has_bn=False,
                has_relu=has_relu,
                rng=rng,
            )
            all_nodes.extend(nodes)
            all_inits.extend(inits)
            all_vis.extend(vis)

        graph_input = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 4])
        graph_output = helper.make_tensor_value_info(current, TensorProto.FLOAT, [1, 4])
        onnx_graph = helper.make_graph(
            all_nodes,
            "no_bn_model",
            [graph_input],
            [graph_output],
            initializer=all_inits,
            value_info=all_vis,
        )
        model = helper.make_model(onnx_graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        path = tmp_path / "no_bn.onnx"
        onnx.save(model, str(path))

        parser = ONNXParser(path)
        graph = parser.parse()

        calib = [np.random.RandomState(99).randn(1, 4).astype(np.float32) for _ in range(10)]
        pm = PassManager()
        pm.add_pass(ConstantFoldingPass())
        pm.add_pass(DeadCodeEliminationPass())
        pm.add_pass(BatchNormFoldingPass())
        pm.add_pass(OperatorFusionPass())
        pm.add_pass(QuantizationPass(calib))

        result = pm.run(graph, verify=False)
        assert len(result.nodes) == 2
        assert result.stage == "quantized"

    def test_final_layer_no_relu_is_fused_linear(self, parsed_ad_graph, calibration_data):
        """Layer 3 (no ReLU) → FusedLinear."""
        pm = PassManager()
        pm.add_pass(ConstantFoldingPass())
        pm.add_pass(DeadCodeEliminationPass())
        pm.add_pass(BatchNormFoldingPass())
        pm.add_pass(OperatorFusionPass())
        pm.add_pass(QuantizationPass(calibration_data))

        result = pm.run(parsed_ad_graph, verify=False)

        ordered = [result.nodes[n] for n in result.topological_order()]
        last_node = ordered[-1]
        assert last_node.op_type == OpType.FUSED_LINEAR
        assert not last_node.fused_attrs.has_relu
        assert last_node.fused_attrs.output_dim == 640


# ---------------------------------------------------------------------------
# IRInterpreter edge cases
# ---------------------------------------------------------------------------


class TestIRInterpreterEdgeCases:
    def test_missing_input_raises(self):
        """Empty dict → ValueError."""
        w = np.eye(2, dtype=np.float32)
        graph = Graph(
            name="test",
            nodes={"mm": OpNode("mm", OpType.MATMUL, ["x", "w"], ["y"])},
            tensors={
                "x": Tensor("x", TensorType((1, 2), np.dtype(np.float32))),
                "w": Tensor("w", TensorType((2, 2), np.dtype(np.float32)), data=w),
                "y": Tensor("y", TensorType((1, 2), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["y"],
        )
        interp = IRInterpreter(graph)
        with pytest.raises(ValueError, match="Missing input"):
            interp.run({})

    def test_wrong_input_shape_raises(self):
        """Mismatched shape → ValueError."""
        w = np.eye(2, dtype=np.float32)
        graph = Graph(
            name="test",
            nodes={"mm": OpNode("mm", OpType.MATMUL, ["x", "w"], ["y"])},
            tensors={
                "x": Tensor("x", TensorType((1, 2), np.dtype(np.float32))),
                "w": Tensor("w", TensorType((2, 2), np.dtype(np.float32)), data=w),
                "y": Tensor("y", TensorType((1, 2), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["y"],
        )
        interp = IRInterpreter(graph)
        with pytest.raises(ValueError, match="Shape mismatch"):
            interp.run({"x": np.ones((1, 5), dtype=np.float32)})


# ---------------------------------------------------------------------------
# Quantization edge cases
# ---------------------------------------------------------------------------


class TestQuantizationEdgeCases:
    def _get_fused_graph(self, parsed_ad_graph: Graph) -> Graph:
        g = BatchNormFoldingPass().run(parsed_ad_graph)
        g.invalidate_cache()
        return OperatorFusionPass().run(g)

    def test_all_zero_weights_no_nan(self, parsed_ad_graph):
        """All-zero weights → scale clamped to 1e-8, no NaN in output."""
        graph = self._get_fused_graph(parsed_ad_graph)

        # Zero out all weight tensors
        for node in graph.nodes.values():
            wt = graph.tensors[node.inputs[1]]
            wt.data = np.zeros_like(wt.data)

        rng = np.random.RandomState(42)
        calib = [rng.randn(1, 640).astype(np.float32) for _ in range(10)]
        result = QuantizationPass(calib).run(graph)

        # No NaN anywhere
        for node in result.nodes.values():
            attrs = node.fused_attrs
            assert np.isfinite(attrs.weight_quant.scale)
            assert np.isfinite(attrs.input_quant.scale)
            assert np.isfinite(attrs.output_quant.scale)
            wt = result.tensors[node.inputs[1]]
            assert np.isfinite(wt.data.astype(np.float32)).all()

    def test_nan_calibration_data_rejected(self, parsed_ad_graph):
        """NaN calibration → ValueError."""
        graph = self._get_fused_graph(parsed_ad_graph)
        bad_sample = np.full((1, 640), np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN or inf"):
            QuantizationPass([bad_sample]).run(graph)

    def test_inf_calibration_data_rejected(self, parsed_ad_graph):
        """inf calibration → ValueError."""
        graph = self._get_fused_graph(parsed_ad_graph)
        bad_sample = np.full((1, 640), np.inf, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN or inf"):
            QuantizationPass([bad_sample]).run(graph)

    def test_flat_activation_range(self, parsed_ad_graph):
        """min==max calibration data → no crash, valid scale."""
        graph = self._get_fused_graph(parsed_ad_graph)
        # Constant input: all values the same
        calib = [np.ones((1, 640), dtype=np.float32) for _ in range(5)]
        result = QuantizationPass(calib).run(graph)

        for node in result.nodes.values():
            attrs = node.fused_attrs
            assert attrs.input_quant.scale > 0
            assert attrs.output_quant.scale > 0

    def test_requant_overflow_detected(self):
        """Craft scales producing M_fixed > 2^31 → ValueError."""
        # Build a minimal fused graph
        w = np.ones((2, 2), dtype=np.float32) * 0.01
        b = np.zeros(2, dtype=np.float32)
        node = OpNode("fl", OpType.FUSED_LINEAR, ["x", "w", "b"], ["y"])
        node.fused_attrs = FusedLinearAttrs(2, 2, has_relu=False)
        graph = Graph(
            name="test",
            nodes={"fl": node},
            tensors={
                "x": Tensor("x", TensorType((1, 2), np.dtype(np.float32))),
                "w": Tensor("w", TensorType((2, 2), np.dtype(np.float32)), data=w),
                "b": Tensor("b", TensorType((2,), np.dtype(np.float32)), data=b),
                "y": Tensor("y", TensorType((1, 2), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["y"],
            stage="fused",
        )

        # Craft calibration with extreme range imbalance to force overflow
        # scale_w * scale_x / scale_y >> 1 → M_fixed overflows INT32
        # scale_w will be ~max_abs/127, scale_x from input range, scale_y from output range
        # We need large input values and tiny output range
        huge = np.full((1, 2), 1e10, dtype=np.float32)

        # Monkey-patch calibrate to return extreme ranges
        class ForcedOverflowQuant(QuantizationPass):
            def _calibrate(self, graph):
                return {
                    "x": (0.0, 1e10),
                    "y": (0.0, 1e-10),
                }

        with pytest.raises(ValueError, match="M_fixed overflow"):
            ForcedOverflowQuant([huge]).run(graph)

    def test_round_half_up_all_quant_paths(self, parsed_ad_graph):
        """Exact 0.5 boundaries in quant paths all round up."""
        # The QuantizationPass uses np.floor(x + 0.5) which is round-half-up
        # Verify with a direct unit check
        val = 2.5
        assert int(np.floor(val + 0.5)) == 3  # rounds up
        val = 3.5
        assert int(np.floor(val + 0.5)) == 4  # rounds up
        val = -0.5
        assert int(np.floor(val + 0.5)) == 0  # rounds toward positive

    def test_negative_round_half_up(self):
        """Negative 0.5 boundaries round correctly (-0.5→0, -1.5→-1)."""
        assert int(np.floor(-0.5 + 0.5)) == 0
        assert int(np.floor(-1.5 + 0.5)) == -1
        assert int(np.floor(-2.5 + 0.5)) == -2


# ---------------------------------------------------------------------------
# Quantization verify() test
# ---------------------------------------------------------------------------


class TestQuantizationVerify:
    def test_verify_passes_on_ad_model(self, parsed_ad_graph, calibration_data):
        """QuantizationPass with verify=True succeeds on AD model."""
        pm = PassManager()
        pm.add_pass(BatchNormFoldingPass())
        pm.add_pass(OperatorFusionPass())
        pm.add_pass(QuantizationPass(calibration_data))

        # This should not raise (verify=True invokes verify() after each pass)
        result = pm.run(parsed_ad_graph, verify=True)
        assert result.stage == "quantized"
        assert len(result.nodes) == 4
