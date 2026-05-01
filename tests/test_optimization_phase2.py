"""Tests for Phase 2 optimization passes (CNN/Transformer fusion + DAG support)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from mlasic.ingestion import ONNXParser
from mlasic.interpreter import IRInterpreter
from mlasic.ir import (
    FusedActivationAttrs,
    FusedConvAttrs,
    FusedLayerNormAttrs,
    Graph,
    OpNode,
    OpType,
    Tensor,
    TensorType,
)
from mlasic.optimization import (
    ActivationFusionPass,
    BatchNormFoldingPass,
    ConstantFoldingPass,
    ConvBatchNormFoldingPass,
    ConvFusionPass,
    ConvQuantizationPass,
    DeadCodeEliminationPass,
    LayerNormFusionPass,
    PassManager,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def parsed_cnn_graph(cnn_model_path) -> Graph:
    """Parse the CNN model and return the IR graph."""
    parser = ONNXParser(cnn_model_path)
    return parser.parse()


@pytest.fixture
def parsed_transformer_graph(transformer_model_path) -> Graph:
    """Parse the transformer model and return the IR graph."""
    parser = ONNXParser(transformer_model_path)
    return parser.parse()


@pytest.fixture
def cnn_calibration_data() -> list[np.ndarray]:
    """Generate calibration data for CNN (1, 1, 8, 8)."""
    rng = np.random.RandomState(123)
    return [rng.randn(1, 1, 8, 8).astype(np.float32) for _ in range(10)]


# ---------------------------------------------------------------------------
# Conv+BN Folding Tests
# ---------------------------------------------------------------------------


class TestConvBatchNormFoldingPass:
    def test_removes_bn_after_conv(self, parsed_cnn_graph):
        """BN nodes after Conv are removed."""
        bn_before = sum(
            1 for n in parsed_cnn_graph.nodes.values() if n.op_type == OpType.BATCH_NORM
        )
        assert bn_before == 2

        pass_ = ConvBatchNormFoldingPass()
        result = pass_.run(parsed_cnn_graph)

        bn_after = sum(1 for n in result.nodes.values() if n.op_type == OpType.BATCH_NORM)
        assert bn_after == 0

    def test_preserves_semantics_3x3(self, parsed_cnn_graph):
        """Conv+BN folding preserves output within FP32 tolerance."""
        before = copy.deepcopy(parsed_cnn_graph)
        pass_ = ConvBatchNormFoldingPass()
        after = pass_.run(parsed_cnn_graph)

        interp_before = IRInterpreter(before)
        interp_after = IRInterpreter(after)

        rng = np.random.RandomState(42)
        for _ in range(5):
            x = rng.randn(1, 1, 8, 8).astype(np.float32)
            out_before = interp_before.run({"input": x})
            out_after = interp_after.run({"input": x})
            for key in out_before:
                np.testing.assert_allclose(out_before[key], out_after[key], rtol=1e-5, atol=1e-4)

    def test_conv_1x1_bn_fold(self):
        """1x1 conv followed by BN folds correctly."""
        rng = np.random.RandomState(77)
        w = rng.randn(4, 2, 1, 1).astype(np.float32) * 0.1
        b = rng.randn(4).astype(np.float32) * 0.01
        gamma = np.array([1.5, 0.8, 2.0, 1.0], dtype=np.float32)
        beta = np.array([0.1, -0.2, 0.0, 0.5], dtype=np.float32)
        mean = np.array([0.5, -0.3, 0.1, 0.0], dtype=np.float32)
        var = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float32)

        graph = Graph(
            name="test",
            nodes={
                "conv": OpNode(
                    "conv",
                    OpType.CONV,
                    ["x", "w", "b"],
                    ["conv_out"],
                    attributes={
                        "kernel_shape": [1, 1],
                        "strides": [1, 1],
                        "pads": [0, 0, 0, 0],
                        "dilations": [1, 1],
                        "group": 1,
                    },
                ),
                "bn": OpNode(
                    "bn",
                    OpType.BATCH_NORM,
                    ["conv_out", "gamma", "beta", "mean", "var"],
                    ["bn_out"],
                    attributes={"epsilon": 1e-5},
                ),
            },
            tensors={
                "x": Tensor("x", TensorType((1, 2, 4, 4), np.dtype(np.float32))),
                "w": Tensor("w", TensorType((4, 2, 1, 1), np.dtype(np.float32)), data=w),
                "b": Tensor("b", TensorType((4,), np.dtype(np.float32)), data=b),
                "conv_out": Tensor("conv_out", TensorType((1, 4, 4, 4), np.dtype(np.float32))),
                "gamma": Tensor("gamma", TensorType((4,), np.dtype(np.float32)), data=gamma),
                "beta": Tensor("beta", TensorType((4,), np.dtype(np.float32)), data=beta),
                "mean": Tensor("mean", TensorType((4,), np.dtype(np.float32)), data=mean),
                "var": Tensor("var", TensorType((4,), np.dtype(np.float32)), data=var),
                "bn_out": Tensor("bn_out", TensorType((1, 4, 4, 4), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["bn_out"],
        )

        before = copy.deepcopy(graph)
        after = ConvBatchNormFoldingPass().run(graph)
        assert "bn" not in after.nodes

        interp_before = IRInterpreter(before)
        interp_after = IRInterpreter(after)

        x = rng.randn(1, 2, 4, 4).astype(np.float32)
        out_b = list(interp_before.run({"x": x}).values())[0]
        out_a = list(interp_after.run({"x": x}).values())[0]
        np.testing.assert_allclose(out_b, out_a, rtol=1e-5, atol=1e-5)

    def test_depthwise_conv_bn_fold(self):
        """Depthwise conv (group=C) with BN folds correctly."""
        rng = np.random.RandomState(88)
        c = 4
        w = rng.randn(c, 1, 3, 3).astype(np.float32) * 0.1
        gamma = np.ones(c, dtype=np.float32)
        beta = np.zeros(c, dtype=np.float32)
        mean = np.zeros(c, dtype=np.float32)
        var = np.ones(c, dtype=np.float32)

        graph = Graph(
            name="test",
            nodes={
                "conv": OpNode(
                    "conv",
                    OpType.CONV,
                    ["x", "w"],
                    ["conv_out"],
                    attributes={
                        "kernel_shape": [3, 3],
                        "strides": [1, 1],
                        "pads": [1, 1, 1, 1],
                        "dilations": [1, 1],
                        "group": c,
                    },
                ),
                "bn": OpNode(
                    "bn",
                    OpType.BATCH_NORM,
                    ["conv_out", "gamma", "beta", "mean", "var"],
                    ["bn_out"],
                    attributes={"epsilon": 1e-5},
                ),
            },
            tensors={
                "x": Tensor("x", TensorType((1, c, 8, 8), np.dtype(np.float32))),
                "w": Tensor("w", TensorType((c, 1, 3, 3), np.dtype(np.float32)), data=w),
                "conv_out": Tensor("conv_out", TensorType((1, c, 8, 8), np.dtype(np.float32))),
                "gamma": Tensor("gamma", TensorType((c,), np.dtype(np.float32)), data=gamma),
                "beta": Tensor("beta", TensorType((c,), np.dtype(np.float32)), data=beta),
                "mean": Tensor("mean", TensorType((c,), np.dtype(np.float32)), data=mean),
                "var": Tensor("var", TensorType((c,), np.dtype(np.float32)), data=var),
                "bn_out": Tensor("bn_out", TensorType((1, c, 8, 8), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["bn_out"],
        )

        before = copy.deepcopy(graph)
        after = ConvBatchNormFoldingPass().run(graph)
        assert "bn" not in after.nodes

        # Conv should now have a bias (was created)
        conv_node = after.nodes["conv"]
        assert len(conv_node.inputs) == 3

        interp_before = IRInterpreter(before)
        interp_after = IRInterpreter(after)
        x = rng.randn(1, c, 8, 8).astype(np.float32)
        out_b = list(interp_before.run({"x": x}).values())[0]
        out_a = list(interp_after.run({"x": x}).values())[0]
        np.testing.assert_allclose(out_b, out_a, rtol=1e-5, atol=1e-5)

    def test_no_bias_conv_bn_fold(self):
        """Conv without bias + BN: bias is created during fold."""
        rng = np.random.RandomState(99)
        w = rng.randn(2, 1, 3, 3).astype(np.float32) * 0.1
        gamma = np.array([2.0, 0.5], dtype=np.float32)
        beta = np.array([1.0, -1.0], dtype=np.float32)
        mean = np.array([0.1, -0.1], dtype=np.float32)
        var = np.array([0.5, 2.0], dtype=np.float32)

        graph = Graph(
            name="test",
            nodes={
                "conv": OpNode(
                    "conv",
                    OpType.CONV,
                    ["x", "w"],
                    ["conv_out"],
                    attributes={
                        "kernel_shape": [3, 3],
                        "strides": [1, 1],
                        "pads": [1, 1, 1, 1],
                        "dilations": [1, 1],
                        "group": 1,
                    },
                ),
                "bn": OpNode(
                    "bn",
                    OpType.BATCH_NORM,
                    ["conv_out", "gamma", "beta", "mean", "var"],
                    ["bn_out"],
                    attributes={"epsilon": 1e-5},
                ),
            },
            tensors={
                "x": Tensor("x", TensorType((1, 1, 4, 4), np.dtype(np.float32))),
                "w": Tensor("w", TensorType((2, 1, 3, 3), np.dtype(np.float32)), data=w),
                "conv_out": Tensor("conv_out", TensorType((1, 2, 4, 4), np.dtype(np.float32))),
                "gamma": Tensor("gamma", TensorType((2,), np.dtype(np.float32)), data=gamma),
                "beta": Tensor("beta", TensorType((2,), np.dtype(np.float32)), data=beta),
                "mean": Tensor("mean", TensorType((2,), np.dtype(np.float32)), data=mean),
                "var": Tensor("var", TensorType((2,), np.dtype(np.float32)), data=var),
                "bn_out": Tensor("bn_out", TensorType((1, 2, 4, 4), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["bn_out"],
        )

        before = copy.deepcopy(graph)
        after = ConvBatchNormFoldingPass().run(graph)
        assert "bn" not in after.nodes
        # Bias was created
        conv_node = after.nodes["conv"]
        assert len(conv_node.inputs) == 3

        interp_before = IRInterpreter(before)
        interp_after = IRInterpreter(after)
        x = rng.randn(1, 1, 4, 4).astype(np.float32)
        out_b = list(interp_before.run({"x": x}).values())[0]
        out_a = list(interp_after.run({"x": x}).values())[0]
        np.testing.assert_allclose(out_b, out_a, rtol=1e-5, atol=1e-5)

    def test_preserves_non_conv_nodes(self, parsed_cnn_graph):
        """Non-conv nodes (MaxPool, Flatten, etc.) are preserved."""
        pass_ = ConvBatchNormFoldingPass()
        result = pass_.run(parsed_cnn_graph)
        after_types = {n.op_type for n in result.nodes.values()}

        # MaxPool, Flatten etc still present
        assert OpType.MAX_POOL in after_types
        assert OpType.FLATTEN in after_types
        assert OpType.BATCH_NORM not in after_types


# ---------------------------------------------------------------------------
# Conv Fusion Tests
# ---------------------------------------------------------------------------


class TestConvFusionPass:
    def _prepare(self, graph: Graph) -> Graph:
        """BN fold + DCE before fusion."""
        g = ConvBatchNormFoldingPass().run(graph)
        g.invalidate_cache()
        g = DeadCodeEliminationPass().run(g)
        g.invalidate_cache()
        return g

    def test_conv_relu_fusion(self, parsed_cnn_graph):
        """Conv+ReLU fuses to FusedConvReLU."""
        graph = self._prepare(parsed_cnn_graph)
        result = ConvFusionPass().run(graph)

        fused_conv_relu = [n for n in result.nodes.values() if n.op_type == OpType.FUSED_CONV_RELU]
        assert len(fused_conv_relu) == 2  # Two conv+relu pairs

    def test_conv_alone_fuses(self):
        """Conv without activation fuses to FusedConv."""
        rng = np.random.RandomState(42)
        w = rng.randn(4, 1, 3, 3).astype(np.float32) * 0.1
        b = rng.randn(4).astype(np.float32) * 0.01

        graph = Graph(
            name="test",
            nodes={
                "conv": OpNode(
                    "conv",
                    OpType.CONV,
                    ["x", "w", "b"],
                    ["y"],
                    attributes={
                        "kernel_shape": [3, 3],
                        "strides": [1, 1],
                        "pads": [1, 1, 1, 1],
                        "dilations": [1, 1],
                        "group": 1,
                    },
                ),
            },
            tensors={
                "x": Tensor("x", TensorType((1, 1, 8, 8), np.dtype(np.float32))),
                "w": Tensor("w", TensorType((4, 1, 3, 3), np.dtype(np.float32)), data=w),
                "b": Tensor("b", TensorType((4,), np.dtype(np.float32)), data=b),
                "y": Tensor("y", TensorType((1, 4, 8, 8), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["y"],
        )

        result = ConvFusionPass().run(graph)
        fused = list(result.nodes.values())
        assert len(fused) == 1
        assert fused[0].op_type == OpType.FUSED_CONV

    def test_conv_relu6_fusion(self):
        """Conv+Clip(0,6) fuses to FusedConvReLU6."""
        rng = np.random.RandomState(42)
        w = rng.randn(4, 1, 3, 3).astype(np.float32) * 0.1
        clip_min = np.array(0.0, dtype=np.float32)
        clip_max = np.array(6.0, dtype=np.float32)

        graph = Graph(
            name="test",
            nodes={
                "conv": OpNode(
                    "conv",
                    OpType.CONV,
                    ["x", "w"],
                    ["conv_out"],
                    attributes={
                        "kernel_shape": [3, 3],
                        "strides": [1, 1],
                        "pads": [1, 1, 1, 1],
                        "dilations": [1, 1],
                        "group": 1,
                    },
                ),
                "clip": OpNode(
                    "clip",
                    OpType.CLIP,
                    ["conv_out", "clip_min", "clip_max"],
                    ["y"],
                ),
            },
            tensors={
                "x": Tensor("x", TensorType((1, 1, 8, 8), np.dtype(np.float32))),
                "w": Tensor("w", TensorType((4, 1, 3, 3), np.dtype(np.float32)), data=w),
                "conv_out": Tensor("conv_out", TensorType((1, 4, 8, 8), np.dtype(np.float32))),
                "clip_min": Tensor("clip_min", TensorType((), np.dtype(np.float32)), data=clip_min),
                "clip_max": Tensor("clip_max", TensorType((), np.dtype(np.float32)), data=clip_max),
                "y": Tensor("y", TensorType((1, 4, 8, 8), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["y"],
        )

        result = ConvFusionPass().run(graph)
        fused = list(result.nodes.values())
        assert len(fused) == 1
        assert fused[0].op_type == OpType.FUSED_CONV_RELU6

    def test_fusion_preserves_semantics(self, parsed_cnn_graph):
        """Fusion preserves numerical output."""
        graph = self._prepare(parsed_cnn_graph)
        before = copy.deepcopy(graph)
        after = ConvFusionPass().run(graph)

        interp_before = IRInterpreter(before)
        interp_after = IRInterpreter(after)

        rng = np.random.RandomState(42)
        for _ in range(5):
            x = rng.randn(1, 1, 8, 8).astype(np.float32)
            out_b = interp_before.run({"input": x})
            out_a = interp_after.run({"input": x})
            for key in out_b:
                np.testing.assert_allclose(out_b[key], out_a[key], rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Conv Quantization Tests
# ---------------------------------------------------------------------------


class TestConvQuantizationPass:
    def _prepare_fused(self, graph: Graph) -> Graph:
        """BN fold + DCE + conv fusion."""
        g = ConvBatchNormFoldingPass().run(graph)
        g.invalidate_cache()
        g = DeadCodeEliminationPass().run(g)
        g.invalidate_cache()
        g = ConvFusionPass().run(g)
        g.invalidate_cache()
        return g

    def test_per_channel_scales(self, parsed_cnn_graph, cnn_calibration_data):
        """Each output channel gets its own scale."""
        graph = self._prepare_fused(parsed_cnn_graph)
        result = ConvQuantizationPass(cnn_calibration_data).run(graph)

        for node in result.nodes.values():
            attrs = node.fused_attrs
            if not isinstance(attrs, FusedConvAttrs):
                continue
            assert attrs.weight_quant is not None
            assert len(attrs.weight_quant) == attrs.out_channels
            for wq in attrs.weight_quant:
                assert wq.zero_point == 0
                assert wq.scale > 0

    def test_weights_are_int8(self, parsed_cnn_graph, cnn_calibration_data):
        """Weight tensors become INT8 after quantization."""
        graph = self._prepare_fused(parsed_cnn_graph)
        result = ConvQuantizationPass(cnn_calibration_data).run(graph)

        for node in result.nodes.values():
            if not isinstance(node.fused_attrs, FusedConvAttrs):
                continue
            wt = result.tensors[node.inputs[1]]
            assert wt.data.dtype == np.int8

    def test_biases_are_int32(self, parsed_cnn_graph, cnn_calibration_data):
        """Bias tensors become INT32 after quantization."""
        graph = self._prepare_fused(parsed_cnn_graph)
        result = ConvQuantizationPass(cnn_calibration_data).run(graph)

        for node in result.nodes.values():
            if not isinstance(node.fused_attrs, FusedConvAttrs):
                continue
            if len(node.inputs) > 2:
                bt = result.tensors[node.inputs[2]]
                assert bt.data.dtype == np.int32

    def test_requant_params_per_channel(self, parsed_cnn_graph, cnn_calibration_data):
        """Per-channel M_fixed values are computed."""
        graph = self._prepare_fused(parsed_cnn_graph)
        result = ConvQuantizationPass(cnn_calibration_data).run(graph)

        for node in result.nodes.values():
            attrs = node.fused_attrs
            if not isinstance(attrs, FusedConvAttrs):
                continue
            assert attrs.requant_scale_fixed is not None
            assert len(attrs.requant_scale_fixed) == attrs.out_channels
            assert attrs.requant_shift == 16
            for m in attrs.requant_scale_fixed:
                assert m > 0

    def test_is_quantized_flag(self, parsed_cnn_graph, cnn_calibration_data):
        """FusedConvAttrs.is_quantized returns True after quantization."""
        graph = self._prepare_fused(parsed_cnn_graph)
        result = ConvQuantizationPass(cnn_calibration_data).run(graph)

        for node in result.nodes.values():
            attrs = node.fused_attrs
            if isinstance(attrs, FusedConvAttrs):
                assert attrs.is_quantized


# ---------------------------------------------------------------------------
# LayerNorm Fusion Tests
# ---------------------------------------------------------------------------


class TestLayerNormFusionPass:
    def test_basic_fusion(self, parsed_transformer_graph):
        """LayerNorm nodes are fused to FusedLayerNorm."""
        ln_before = sum(
            1 for n in parsed_transformer_graph.nodes.values() if n.op_type == OpType.LAYER_NORM
        )
        assert ln_before == 2

        result = LayerNormFusionPass().run(parsed_transformer_graph)
        ln_after = sum(1 for n in result.nodes.values() if n.op_type == OpType.LAYER_NORM)
        fused_ln = sum(1 for n in result.nodes.values() if n.op_type == OpType.FUSED_LAYER_NORM)
        assert ln_after == 0
        assert fused_ln == 2

    def test_attrs_preserved(self, parsed_transformer_graph):
        """FusedLayerNormAttrs has correct axis and epsilon."""
        result = LayerNormFusionPass().run(parsed_transformer_graph)

        for node in result.nodes.values():
            if node.op_type != OpType.FUSED_LAYER_NORM:
                continue
            attrs = node.fused_attrs
            assert isinstance(attrs, FusedLayerNormAttrs)
            assert attrs.axis == -1
            assert abs(attrs.epsilon - 1e-5) < 1e-8
            assert attrs.normalized_shape is not None


# ---------------------------------------------------------------------------
# GELU / SiLU Activation Fusion Tests
# ---------------------------------------------------------------------------


class TestActivationFusionPass:
    def test_gelu_pattern_match(self, parsed_transformer_graph):
        """GELU 5-node pattern is detected and fused."""
        result = ActivationFusionPass().run(parsed_transformer_graph)

        fused_gelu = [n for n in result.nodes.values() if n.op_type == OpType.FUSED_GELU]
        assert len(fused_gelu) == 1
        attrs = fused_gelu[0].fused_attrs
        assert isinstance(attrs, FusedActivationAttrs)
        assert attrs.activation_type == "gelu"

    def test_gelu_semantics(self, parsed_transformer_graph):
        """GELU fusion preserves numerical semantics."""
        before = copy.deepcopy(parsed_transformer_graph)
        after = ActivationFusionPass().run(parsed_transformer_graph)

        interp_before = IRInterpreter(before)
        interp_after = IRInterpreter(after)

        rng = np.random.RandomState(42)
        for _ in range(3):
            x = rng.randint(0, 32, (1, 4)).astype(np.int64)
            out_b = interp_before.run({"input_ids": x})
            out_a = interp_after.run({"input_ids": x})
            for key in out_b:
                np.testing.assert_allclose(out_b[key], out_a[key], rtol=1e-5, atol=1e-5)

    def test_silu_pattern_match(self):
        """Sigmoid(x) * x is fused to FusedSiLU."""
        graph = Graph(
            name="test",
            nodes={
                "sig": OpNode("sig", OpType.SIGMOID, ["x"], ["sig_out"]),
                "mul": OpNode("mul", OpType.MUL, ["x", "sig_out"], ["y"]),
            },
            tensors={
                "x": Tensor("x", TensorType((1, 4), np.dtype(np.float32))),
                "sig_out": Tensor("sig_out", TensorType((1, 4), np.dtype(np.float32))),
                "y": Tensor("y", TensorType((1, 4), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["y"],
        )

        result = ActivationFusionPass().run(graph)
        fused = [n for n in result.nodes.values() if n.op_type == OpType.FUSED_SILU]
        assert len(fused) == 1
        attrs = fused[0].fused_attrs
        assert isinstance(attrs, FusedActivationAttrs)
        assert attrs.activation_type == "silu"

    def test_silu_semantics(self):
        """SiLU fusion preserves numerical semantics."""
        graph = Graph(
            name="test",
            nodes={
                "sig": OpNode("sig", OpType.SIGMOID, ["x"], ["sig_out"]),
                "mul": OpNode("mul", OpType.MUL, ["x", "sig_out"], ["y"]),
            },
            tensors={
                "x": Tensor("x", TensorType((1, 4), np.dtype(np.float32))),
                "sig_out": Tensor("sig_out", TensorType((1, 4), np.dtype(np.float32))),
                "y": Tensor("y", TensorType((1, 4), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["y"],
        )

        before = copy.deepcopy(graph)
        after = ActivationFusionPass().run(graph)

        interp_before = IRInterpreter(before)
        interp_after = IRInterpreter(after)

        x = np.array([[-1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
        out_b = list(interp_before.run({"x": x}).values())[0]
        out_a = list(interp_after.run({"x": x}).values())[0]
        np.testing.assert_allclose(out_b, out_a, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# DAG Validation Tests
# ---------------------------------------------------------------------------


class TestDAGValidation:
    def _build_mixed_dag(self) -> Graph:
        """Build a graph with fused + passthrough ops for validation."""
        rng = np.random.RandomState(42)
        w = rng.randn(4, 1, 3, 3).astype(np.float32) * 0.1
        b = rng.randn(4).astype(np.float32) * 0.01

        fused_node = OpNode(
            "fc",
            OpType.FUSED_CONV_RELU,
            ["x", "w", "b"],
            ["fc_out"],
        )
        fused_node.fused_attrs = FusedConvAttrs(
            in_channels=1,
            out_channels=4,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            has_relu=True,
        )

        return Graph(
            name="test",
            nodes={
                "fc": fused_node,
                "pool": OpNode(
                    "pool",
                    OpType.MAX_POOL,
                    ["fc_out"],
                    ["pool_out"],
                    attributes={"kernel_shape": [2, 2], "strides": [2, 2]},
                ),
                "flat": OpNode(
                    "flat",
                    OpType.FLATTEN,
                    ["pool_out"],
                    ["y"],
                    attributes={"axis": 1},
                ),
            },
            tensors={
                "x": Tensor("x", TensorType((1, 1, 8, 8), np.dtype(np.float32))),
                "w": Tensor("w", TensorType((4, 1, 3, 3), np.dtype(np.float32)), data=w),
                "b": Tensor("b", TensorType((4,), np.dtype(np.float32)), data=b),
                "fc_out": Tensor("fc_out", TensorType((1, 4, 8, 8), np.dtype(np.float32))),
                "pool_out": Tensor("pool_out", TensorType((1, 4, 4, 4), np.dtype(np.float32))),
                "y": Tensor("y", TensorType((1, 64), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["y"],
            stage="fused_dag",
        )

    def test_accepts_mixed_ops(self):
        """fused_dag stage accepts fused + passthrough ops."""
        graph = self._build_mixed_dag()
        graph.validate("fused_dag")  # Should not raise

    def test_rejects_raw_conv(self):
        """fused_dag rejects raw Conv (not fused)."""
        graph = self._build_mixed_dag()
        # Replace fused with raw Conv
        raw = OpNode(
            "raw_conv",
            OpType.CONV,
            ["x", "w", "b"],
            ["fc_out"],
            attributes={
                "kernel_shape": [3, 3],
                "strides": [1, 1],
                "pads": [1, 1, 1, 1],
                "dilations": [1, 1],
                "group": 1,
            },
        )
        del graph.nodes["fc"]
        graph.nodes["raw_conv"] = raw
        graph.invalidate_cache()

        from mlasic.exceptions import IRValidationError

        with pytest.raises(IRValidationError):
            graph.validate("fused_dag")

    def test_rejects_missing_fused_attrs(self):
        """fused_dag rejects fused node without fused_attrs."""
        graph = self._build_mixed_dag()
        # Remove fused_attrs
        graph.nodes["fc"].attributes.pop("fused_attrs")

        from mlasic.exceptions import IRValidationError

        with pytest.raises(IRValidationError):
            graph.validate("fused_dag")

    def test_replace_subgraph_works(self):
        """Graph.replace_subgraph properly rewires."""
        rng = np.random.RandomState(42)
        w = rng.randn(4, 2).astype(np.float32)
        b = rng.randn(4).astype(np.float32)

        graph = Graph(
            name="test",
            nodes={
                "mm": OpNode("mm", OpType.MATMUL, ["x", "w"], ["mm_out"]),
                "add": OpNode("add", OpType.ADD, ["mm_out", "b"], ["y"]),
            },
            tensors={
                "x": Tensor("x", TensorType((1, 2), np.dtype(np.float32))),
                "w": Tensor("w", TensorType((2, 4), np.dtype(np.float32)), data=w),
                "mm_out": Tensor("mm_out", TensorType((1, 4), np.dtype(np.float32))),
                "b": Tensor("b", TensorType((4,), np.dtype(np.float32)), data=b),
                "y": Tensor("y", TensorType((1, 4), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["y"],
        )

        from mlasic.ir import FusedLinearAttrs

        new_node = OpNode("fused", OpType.FUSED_LINEAR, ["x", "w", "b"], ["fused_out"])
        new_node.fused_attrs = FusedLinearAttrs(2, 4, has_relu=False)
        new_tensors = {"fused_out": Tensor("fused_out", TensorType((1, 4), np.dtype(np.float32)))}

        graph.replace_subgraph(["mm", "add"], new_node, new_tensors)

        assert "mm" not in graph.nodes
        assert "add" not in graph.nodes
        assert "fused" in graph.nodes
        assert graph.outputs == ["fused_out"]
        # mm_out should be cleaned up (orphan)
        assert "mm_out" not in graph.tensors

    def test_insert_remove_node(self):
        """Graph insert_node and remove_node work correctly."""
        graph = Graph(
            name="test",
            nodes={
                "relu": OpNode("relu", OpType.RELU, ["x"], ["y"]),
            },
            tensors={
                "x": Tensor("x", TensorType((1, 4), np.dtype(np.float32))),
                "y": Tensor("y", TensorType((1, 4), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["y"],
        )

        # Insert
        new_node = OpNode("relu2", OpType.RELU, ["y"], ["z"])
        new_tensors = {"z": Tensor("z", TensorType((1, 4), np.dtype(np.float32)))}
        graph.insert_node(new_node, new_tensors)
        assert "relu2" in graph.nodes
        assert "z" in graph.tensors

        # Remove
        graph.remove_node("relu2")
        assert "relu2" not in graph.nodes

    def test_find_skip_connections(self):
        """find_skip_connections detects tensor consumed by 2+ including Add."""
        graph = Graph(
            name="test",
            nodes={
                "relu": OpNode("relu", OpType.RELU, ["x"], ["relu_out"]),
                "mm": OpNode("mm", OpType.MATMUL, ["relu_out", "w"], ["mm_out"]),
                "add": OpNode("add", OpType.ADD, ["relu_out", "mm_out"], ["y"]),
            },
            tensors={
                "x": Tensor("x", TensorType((1, 4), np.dtype(np.float32))),
                "relu_out": Tensor("relu_out", TensorType((1, 4), np.dtype(np.float32))),
                "w": Tensor(
                    "w",
                    TensorType((4, 4), np.dtype(np.float32)),
                    data=np.eye(4, dtype=np.float32),
                ),
                "mm_out": Tensor("mm_out", TensorType((1, 4), np.dtype(np.float32))),
                "y": Tensor("y", TensorType((1, 4), np.dtype(np.float32))),
            },
            inputs=["x"],
            outputs=["y"],
        )

        skips = graph.find_skip_connections()
        skip_tensors = [s[0] for s in skips]
        assert "relu_out" in skip_tensors


# ---------------------------------------------------------------------------
# CNN Integration Test
# ---------------------------------------------------------------------------


class TestCNNIntegration:
    def test_full_cnn_pipeline(self, cnn_model_path, cnn_calibration_data):
        """CNN goes through full pipeline: parse → fold → fuse → quantize."""
        parser = ONNXParser(cnn_model_path)
        graph = parser.parse()

        # Run optimization pipeline
        pm = PassManager()
        pm.add_pass(ConstantFoldingPass())
        pm.add_pass(DeadCodeEliminationPass())
        pm.add_pass(BatchNormFoldingPass())
        pm.add_pass(ConvBatchNormFoldingPass())
        pm.add_pass(DeadCodeEliminationPass())
        pm.add_pass(ConvFusionPass())
        pm.add_pass(ConvQuantizationPass(cnn_calibration_data))
        graph = pm.run(graph, verify=False)

        # Should have FusedConvReLU nodes
        conv_fused = [
            n
            for n in graph.nodes.values()
            if n.op_type
            in {
                OpType.FUSED_CONV,
                OpType.FUSED_CONV_RELU,
                OpType.FUSED_CONV_RELU6,
            }
        ]
        assert len(conv_fused) == 2

        # All fused conv nodes should be quantized
        for node in conv_fused:
            attrs = node.fused_attrs
            assert isinstance(attrs, FusedConvAttrs)
            assert attrs.is_quantized

        # Weights should be INT8
        for node in conv_fused:
            wt = graph.tensors[node.inputs[1]]
            assert wt.data.dtype == np.int8


# ---------------------------------------------------------------------------
# Transformer Partial Integration Test
# ---------------------------------------------------------------------------


class TestTransformerPartialIntegration:
    def test_parse_fold_fuse(self, transformer_model_path):
        """Transformer model: parse + BN fold + LN fusion + GELU fusion."""
        parser = ONNXParser(transformer_model_path)
        graph = parser.parse()
        before = copy.deepcopy(graph)

        # Run subset of passes
        pm = PassManager()
        pm.add_pass(ConstantFoldingPass())
        pm.add_pass(DeadCodeEliminationPass())
        pm.add_pass(LayerNormFusionPass())
        pm.add_pass(ActivationFusionPass())
        graph = pm.run(graph, verify=False)

        # Should have FusedLayerNorm and FusedGELU
        fused_ln = sum(1 for n in graph.nodes.values() if n.op_type == OpType.FUSED_LAYER_NORM)
        fused_gelu = sum(1 for n in graph.nodes.values() if n.op_type == OpType.FUSED_GELU)
        assert fused_ln == 2
        assert fused_gelu == 1

        # No raw LayerNorm or Erf nodes
        assert not any(n.op_type == OpType.LAYER_NORM for n in graph.nodes.values())
        assert not any(n.op_type == OpType.ERF for n in graph.nodes.values())

        # Verify semantics preserved
        interp_before = IRInterpreter(before)
        interp_after = IRInterpreter(graph)

        rng = np.random.RandomState(42)
        for _ in range(3):
            x = rng.randint(0, 32, (1, 4)).astype(np.int64)
            out_b = interp_before.run({"input_ids": x})
            out_a = interp_after.run({"input_ids": x})
            # Output tensor names may differ after fusion; compare values
            vals_b = list(out_b.values())
            vals_a = list(out_a.values())
            assert len(vals_b) == len(vals_a)
            for vb, va in zip(vals_b, vals_a):
                np.testing.assert_allclose(vb, va, rtol=1e-5, atol=1e-5)
