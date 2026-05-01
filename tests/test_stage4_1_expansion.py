"""Tests for Stage 4.1 Compiler Expansion — remaining gaps.

Covers:
  - IR: AttentionAttrs, FusedAttentionAttrs, FusedMLPAttrs, Tier 3 OpTypes
  - Parser: Tier 3 ops, ONNX subgraph rejection
  - Optimization: AttentionFusionPass, AttentionQuantizationPass, GraphPartitioner, FusedMLPPass
  - INT8 Interpreter: fused LayerNorm, GELU, SiLU, Attention, MLP
  - Weight Packing: Conv, Attention, INT4, Per-tile
  - DAG Scheduler: WeightStreamScheduler, DRAM bandwidth analysis
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mlasic.ir import (
    AttentionAttrs,
    FusedActivationAttrs,
    FusedAttentionAttrs,
    FusedConvAttrs,
    FusedLayerNormAttrs,
    FusedLinearAttrs,
    FusedMLPAttrs,
    Graph,
    HardwareConstraints,
    OpNode,
    OpType,
    QuantParams,
    Tensor,
    TensorType,
)


# ======================================================================
# IR Data Structure Tests
# ======================================================================


class TestAttentionAttrs:
    """Test AttentionAttrs dataclass."""

    def test_basic_construction(self):
        attrs = AttentionAttrs(num_heads=8, head_dim=64)
        assert attrs.num_heads == 8
        assert attrs.head_dim == 64
        assert attrs.seq_len == 0
        assert attrs.has_mask is False
        assert attrs.qkv_bias is True

    def test_with_all_params(self):
        attrs = AttentionAttrs(
            num_heads=12, head_dim=64, seq_len=128, has_mask=True, qkv_bias=False
        )
        assert attrs.num_heads == 12
        assert attrs.seq_len == 128
        assert attrs.has_mask is True
        assert attrs.qkv_bias is False


class TestFusedAttentionAttrs:
    """Test FusedAttentionAttrs dataclass."""

    def test_basic_construction(self):
        attrs = FusedAttentionAttrs(num_heads=8, head_dim=64, seq_len=128)
        assert attrs.num_heads == 8
        assert attrs.head_dim == 64
        assert attrs.seq_len == 128
        assert attrs.embed_dim == 512

    def test_is_quantized_false_initially(self):
        attrs = FusedAttentionAttrs(num_heads=4, head_dim=32, seq_len=64)
        assert attrs.is_quantized is False

    def test_is_quantized_true(self):
        qp = QuantParams(scale=0.1, zero_point=0, calibrated=True)
        attrs = FusedAttentionAttrs(
            num_heads=4, head_dim=32, seq_len=64,
            q_weight_quant=qp, k_weight_quant=qp,
            v_weight_quant=qp, output_weight_quant=qp,
            input_quant=qp, output_quant=qp,
        )
        assert attrs.is_quantized is True

    def test_is_quantized_partial(self):
        qp = QuantParams(scale=0.1, zero_point=0, calibrated=True)
        attrs = FusedAttentionAttrs(
            num_heads=4, head_dim=32, seq_len=64,
            q_weight_quant=qp,
            input_quant=qp, output_quant=qp,
        )
        assert attrs.is_quantized is False

    def test_embed_dim(self):
        attrs = FusedAttentionAttrs(num_heads=12, head_dim=64, seq_len=128)
        assert attrs.embed_dim == 768


class TestFusedMLPAttrs:
    """Test FusedMLPAttrs dataclass."""

    def test_basic_construction(self):
        attrs = FusedMLPAttrs(input_dim=512, hidden_dim=2048, output_dim=512, activation_type="gelu")
        assert attrs.input_dim == 512
        assert attrs.hidden_dim == 2048
        assert attrs.output_dim == 512
        assert attrs.activation_type == "gelu"

    def test_is_quantized_false(self):
        attrs = FusedMLPAttrs(input_dim=128, hidden_dim=512, output_dim=128, activation_type="relu")
        assert attrs.is_quantized is False

    def test_is_quantized_true(self):
        qp = QuantParams(scale=0.1, zero_point=0, calibrated=True)
        attrs = FusedMLPAttrs(
            input_dim=128, hidden_dim=512, output_dim=128, activation_type="gelu",
            fc1_weight_quant=qp, fc2_weight_quant=qp,
            input_quant=qp, mid_quant=qp, output_quant=qp,
        )
        assert attrs.is_quantized is True


class TestTier3OpTypes:
    """Test Tier 3 operator type enum values."""

    def test_group_norm(self):
        assert OpType.GROUP_NORM.value == "GroupNormalization"

    def test_instance_norm(self):
        assert OpType.INSTANCE_NORM.value == "InstanceNormalization"

    def test_q_linear_matmul(self):
        assert OpType.Q_LINEAR_MATMUL.value == "QLinearMatMul"

    def test_resize(self):
        assert OpType.RESIZE.value == "Resize"

    def test_fused_mlp(self):
        assert OpType.FUSED_MLP.value == "FusedMLP"

    def test_from_onnx_group_norm(self):
        assert OpType.from_onnx("GroupNormalization") == OpType.GROUP_NORM

    def test_from_onnx_resize(self):
        assert OpType.from_onnx("Resize") == OpType.RESIZE

    def test_fused_attention_is_fused(self):
        assert OpType.FUSED_ATTENTION.is_fused is True

    def test_fused_mlp_is_fused(self):
        assert OpType.FUSED_MLP.is_fused is True


class TestOpNodeFusedAttrs:
    """Test OpNode.fused_attrs with new attr types."""

    def test_fused_attention_attrs(self):
        node = OpNode(name="attn", op_type=OpType.FUSED_ATTENTION, inputs=[], outputs=[])
        attrs = FusedAttentionAttrs(num_heads=8, head_dim=64, seq_len=128)
        node.fused_attrs = attrs
        assert node.fused_attrs is attrs
        assert isinstance(node.fused_attrs, FusedAttentionAttrs)

    def test_fused_mlp_attrs(self):
        node = OpNode(name="mlp", op_type=OpType.FUSED_MLP, inputs=[], outputs=[])
        attrs = FusedMLPAttrs(input_dim=128, hidden_dim=512, output_dim=128, activation_type="gelu")
        node.fused_attrs = attrs
        assert node.fused_attrs is attrs
        assert isinstance(node.fused_attrs, FusedMLPAttrs)


# ======================================================================
# ONNX Subgraph Rejection Tests
# ======================================================================


class TestSubgraphRejection:
    """Test ONNX If/Loop subgraph rejection at parse time."""

    def test_if_node_rejected(self, tmp_path):
        """If op in ONNX model raises UnsupportedOperatorError."""
        import onnx
        from onnx import TensorProto, helper

        from mlasic.exceptions import UnsupportedOperatorError
        from mlasic.ingestion import ONNXParser

        # Build minimal model with If node
        cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])

        # Then/else branches
        then_body = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["then_out"])],
            "then_graph",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [1, 4])],
        )
        else_body = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["else_out"])],
            "else_graph",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [1, 4])],
        )

        if_node = helper.make_node(
            "If", ["cond"], ["y"],
            then_branch=then_body,
            else_branch=else_body,
        )

        graph = helper.make_graph([if_node], "test_if", [cond, x], [y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model_path = tmp_path / "if_model.onnx"
        onnx.save(model, str(model_path))

        parser = ONNXParser(str(model_path))
        with pytest.raises(UnsupportedOperatorError, match="subgraph ops"):
            parser.parse()


# ======================================================================
# Tier 3 Parser Tests
# ======================================================================


class TestTier3Parsing:
    """Test Tier 3 op parsing."""

    def test_group_norm_in_supported_ops(self):
        from mlasic.ingestion import ONNXParser
        assert "GroupNormalization" in ONNXParser.SUPPORTED_OPS

    def test_instance_norm_in_supported_ops(self):
        from mlasic.ingestion import ONNXParser
        assert "InstanceNormalization" in ONNXParser.SUPPORTED_OPS

    def test_q_linear_matmul_in_supported_ops(self):
        from mlasic.ingestion import ONNXParser
        assert "QLinearMatMul" in ONNXParser.SUPPORTED_OPS

    def test_resize_in_supported_ops(self):
        from mlasic.ingestion import ONNXParser
        assert "Resize" in ONNXParser.SUPPORTED_OPS

    def test_loop_in_supported_ops(self):
        from mlasic.ingestion import ONNXParser
        assert "Loop" in ONNXParser.SUPPORTED_OPS


# ======================================================================
# Optimization Pass Tests
# ======================================================================


def _make_attention_graph():
    """Create a minimal attention-like graph for fusion testing.

    Pattern: x → Q_proj(MatMul) → reshape → transpose
             x → K_proj(MatMul) → reshape → transpose
             x → V_proj(MatMul) → reshape → transpose
             → MatMul(Q, K^T) → Softmax → MatMul(attn, V)
             → transpose → reshape → O_proj(MatMul)
    """
    embed_dim = 64
    num_heads = 4
    head_dim = embed_dim // num_heads
    seq_len = 8

    rng = np.random.RandomState(42)
    x_data = rng.randn(seq_len, embed_dim).astype(np.float32)
    wq = rng.randn(embed_dim, embed_dim).astype(np.float32)
    wk = rng.randn(embed_dim, embed_dim).astype(np.float32)
    wv = rng.randn(embed_dim, embed_dim).astype(np.float32)
    wo = rng.randn(embed_dim, embed_dim).astype(np.float32)

    tensors = {
        "x": Tensor("x", TensorType((seq_len, embed_dim), np.float32)),
        "wq": Tensor("wq", TensorType((embed_dim, embed_dim), np.float32), data=wq),
        "wk": Tensor("wk", TensorType((embed_dim, embed_dim), np.float32), data=wk),
        "wv": Tensor("wv", TensorType((embed_dim, embed_dim), np.float32), data=wv),
        "wo": Tensor("wo", TensorType((embed_dim, embed_dim), np.float32), data=wo),
        "q_proj": Tensor("q_proj", TensorType((seq_len, embed_dim), np.float32)),
        "k_proj": Tensor("k_proj", TensorType((seq_len, embed_dim), np.float32)),
        "v_proj": Tensor("v_proj", TensorType((seq_len, embed_dim), np.float32)),
        "q_reshaped": Tensor("q_reshaped", TensorType((seq_len, num_heads, head_dim), np.float32)),
        "k_reshaped": Tensor("k_reshaped", TensorType((seq_len, num_heads, head_dim), np.float32)),
        "v_reshaped": Tensor("v_reshaped", TensorType((seq_len, num_heads, head_dim), np.float32)),
        "q_transposed": Tensor("q_transposed", TensorType((num_heads, seq_len, head_dim), np.float32)),
        "k_transposed": Tensor("k_transposed", TensorType((num_heads, head_dim, seq_len), np.float32)),
        "v_transposed": Tensor("v_transposed", TensorType((num_heads, seq_len, head_dim), np.float32)),
        "qk": Tensor("qk", TensorType((num_heads, seq_len, seq_len), np.float32)),
        "attn_weights": Tensor("attn_weights", TensorType((num_heads, seq_len, seq_len), np.float32)),
        "attn_out": Tensor("attn_out", TensorType((num_heads, seq_len, head_dim), np.float32)),
        "attn_transposed": Tensor("attn_transposed", TensorType((seq_len, num_heads, head_dim), np.float32)),
        "attn_reshaped": Tensor("attn_reshaped", TensorType((seq_len, embed_dim), np.float32)),
        "output": Tensor("output", TensorType((seq_len, embed_dim), np.float32)),
        "reshape_shape_q": Tensor("reshape_shape_q", TensorType((3,), np.int64),
                                   data=np.array([seq_len, num_heads, head_dim], dtype=np.int64)),
        "reshape_shape_k": Tensor("reshape_shape_k", TensorType((3,), np.int64),
                                   data=np.array([seq_len, num_heads, head_dim], dtype=np.int64)),
        "reshape_shape_v": Tensor("reshape_shape_v", TensorType((3,), np.int64),
                                   data=np.array([seq_len, num_heads, head_dim], dtype=np.int64)),
        "reshape_shape_merge": Tensor("reshape_shape_merge", TensorType((2,), np.int64),
                                       data=np.array([seq_len, embed_dim], dtype=np.int64)),
    }

    nodes = {
        "q_matmul": OpNode("q_matmul", OpType.MATMUL, ["x", "wq"], ["q_proj"]),
        "k_matmul": OpNode("k_matmul", OpType.MATMUL, ["x", "wk"], ["k_proj"]),
        "v_matmul": OpNode("v_matmul", OpType.MATMUL, ["x", "wv"], ["v_proj"]),
        "q_reshape": OpNode("q_reshape", OpType.RESHAPE, ["q_proj", "reshape_shape_q"], ["q_reshaped"]),
        "k_reshape": OpNode("k_reshape", OpType.RESHAPE, ["k_proj", "reshape_shape_k"], ["k_reshaped"]),
        "v_reshape": OpNode("v_reshape", OpType.RESHAPE, ["v_proj", "reshape_shape_v"], ["v_reshaped"]),
        "q_transpose": OpNode("q_transpose", OpType.TRANSPOSE, ["q_reshaped"], ["q_transposed"],
                               attributes={"perm": [1, 0, 2]}),
        "k_transpose": OpNode("k_transpose", OpType.TRANSPOSE, ["k_reshaped"], ["k_transposed"],
                               attributes={"perm": [1, 2, 0]}),
        "v_transpose": OpNode("v_transpose", OpType.TRANSPOSE, ["v_reshaped"], ["v_transposed"],
                               attributes={"perm": [1, 0, 2]}),
        "qk_matmul": OpNode("qk_matmul", OpType.MATMUL, ["q_transposed", "k_transposed"], ["qk"]),
        "softmax": OpNode("softmax", OpType.SOFTMAX, ["qk"], ["attn_weights"],
                           attributes={"axis": -1}),
        "attn_v_matmul": OpNode("attn_v_matmul", OpType.MATMUL, ["attn_weights", "v_transposed"], ["attn_out"]),
        "attn_transpose": OpNode("attn_transpose", OpType.TRANSPOSE, ["attn_out"], ["attn_transposed"],
                                  attributes={"perm": [1, 0, 2]}),
        "attn_reshape": OpNode("attn_reshape", OpType.RESHAPE, ["attn_transposed", "reshape_shape_merge"], ["attn_reshaped"]),
        "o_matmul": OpNode("o_matmul", OpType.MATMUL, ["attn_reshaped", "wo"], ["output"]),
    }

    graph = Graph(
        name="attention_test",
        nodes=nodes,
        tensors=tensors,
        inputs=["x"],
        outputs=["output"],
        stage="raw",
    )
    return graph


class TestAttentionFusionPass:
    """Test AttentionFusionPass."""

    def test_attention_pattern_fused(self):
        from mlasic.optimization import AttentionFusionPass

        graph = _make_attention_graph()
        initial_count = len(graph.nodes)
        assert initial_count == 15

        fused_pass = AttentionFusionPass()
        graph = fused_pass.run(graph)

        # Should have exactly 1 FusedAttention node
        fused_nodes = [n for n in graph.nodes.values() if n.op_type == OpType.FUSED_ATTENTION]
        assert len(fused_nodes) == 1

        node = fused_nodes[0]
        attrs = node.fused_attrs
        assert isinstance(attrs, FusedAttentionAttrs)
        # Dimension inference: embed_dim=64, inferred from weight shape
        assert attrs.num_heads * attrs.head_dim == 64
        assert attrs.head_dim > 0
        assert attrs.num_heads > 0

    def test_non_attention_graph_unchanged(self):
        """A graph without attention pattern should not be changed."""
        from mlasic.optimization import AttentionFusionPass

        # Simple linear graph
        tensors = {
            "x": Tensor("x", TensorType((1, 128), np.float32)),
            "w": Tensor("w", TensorType((128, 64), np.float32), data=np.zeros((128, 64), dtype=np.float32)),
            "y": Tensor("y", TensorType((1, 64), np.float32)),
        }
        nodes = {"mm": OpNode("mm", OpType.MATMUL, ["x", "w"], ["y"])}
        graph = Graph("test", nodes, tensors, ["x"], ["y"], stage="raw")

        fused_pass = AttentionFusionPass()
        graph = fused_pass.run(graph)
        assert len(graph.nodes) == 1
        assert "mm" in graph.nodes


class TestFusedMLPPass:
    """Test FusedMLPPass."""

    def _make_mlp_graph(self, activation_type="gelu"):
        """Create FusedLinear → FusedGELU/SiLU → FusedLinear graph."""
        qp = QuantParams(scale=0.1, zero_point=0, calibrated=True)
        qp_act = QuantParams(scale=0.1, zero_point=5, calibrated=True)

        fc1_attrs = FusedLinearAttrs(
            input_dim=128, output_dim=512, has_relu=False,
            weight_quant=qp, input_quant=qp_act, output_quant=qp_act,
            requant_scale_fixed=1000, requant_shift=16,
        )
        fc2_attrs = FusedLinearAttrs(
            input_dim=512, output_dim=128, has_relu=False,
            weight_quant=qp, input_quant=qp_act, output_quant=qp_act,
            requant_scale_fixed=1000, requant_shift=16,
        )

        act_op = OpType.FUSED_GELU if activation_type == "gelu" else OpType.FUSED_SILU
        act_attrs = FusedActivationAttrs(activation_type=activation_type)

        tensors = {
            "x": Tensor("x", TensorType((1, 128), np.float32)),
            "w1": Tensor("w1", TensorType((128, 512), np.float32),
                         data=np.zeros((128, 512), dtype=np.float32)),
            "b1": Tensor("b1", TensorType((512,), np.float32),
                         data=np.zeros(512, dtype=np.float32)),
            "fc1_out": Tensor("fc1_out", TensorType((1, 512), np.float32)),
            "act_out": Tensor("act_out", TensorType((1, 512), np.float32)),
            "w2": Tensor("w2", TensorType((512, 128), np.float32),
                         data=np.zeros((512, 128), dtype=np.float32)),
            "b2": Tensor("b2", TensorType((128,), np.float32),
                         data=np.zeros(128, dtype=np.float32)),
            "y": Tensor("y", TensorType((1, 128), np.float32)),
        }

        fc1 = OpNode("fc1", OpType.FUSED_LINEAR, ["x", "w1", "b1"], ["fc1_out"])
        fc1.fused_attrs = fc1_attrs

        act = OpNode("act", act_op, ["fc1_out"], ["act_out"])
        act.fused_attrs = act_attrs

        fc2 = OpNode("fc2", OpType.FUSED_LINEAR, ["act_out", "w2", "b2"], ["y"])
        fc2.fused_attrs = fc2_attrs

        return Graph("mlp_test", {"fc1": fc1, "act": act, "fc2": fc2},
                      tensors, ["x"], ["y"], stage="fused_dag")

    def test_gelu_mlp_fusion(self):
        from mlasic.optimization import FusedMLPPass

        graph = self._make_mlp_graph("gelu")
        assert len(graph.nodes) == 3

        mlp_pass = FusedMLPPass()
        graph = mlp_pass.run(graph)

        fused = [n for n in graph.nodes.values() if n.op_type == OpType.FUSED_MLP]
        assert len(fused) == 1
        attrs = fused[0].fused_attrs
        assert isinstance(attrs, FusedMLPAttrs)
        assert attrs.activation_type == "gelu"
        assert attrs.input_dim == 128
        assert attrs.hidden_dim == 512
        assert attrs.output_dim == 128

    def test_silu_mlp_fusion(self):
        from mlasic.optimization import FusedMLPPass

        graph = self._make_mlp_graph("silu")
        mlp_pass = FusedMLPPass()
        graph = mlp_pass.run(graph)

        fused = [n for n in graph.nodes.values() if n.op_type == OpType.FUSED_MLP]
        assert len(fused) == 1
        assert fused[0].fused_attrs.activation_type == "silu"


class TestGraphPartitioner:
    """Test GraphPartitioner."""

    def test_partition_linear_graph(self):
        from mlasic.optimization import GraphPartitioner

        qp = QuantParams(scale=0.1, zero_point=0, calibrated=True)
        attrs = FusedLinearAttrs(
            input_dim=128, output_dim=128, has_relu=True,
            weight_quant=qp, input_quant=qp, output_quant=qp,
            requant_scale_fixed=1000, requant_shift=16,
        )

        tensors = {
            "x": Tensor("x", TensorType((1, 128), np.float32)),
            "w": Tensor("w", TensorType((128, 128), np.float32), data=np.zeros((128, 128), dtype=np.float32)),
            "y": Tensor("y", TensorType((1, 128), np.float32)),
        }

        node = OpNode("fc", OpType.FUSED_LINEAR_RELU, ["x", "w"], ["y"])
        node.fused_attrs = attrs

        graph = Graph("test", {"fc": node}, tensors, ["x"], ["y"], stage="fused_dag")

        partitioner = GraphPartitioner()
        parts = partitioner.partition(graph)
        assert len(parts) == 1
        assert parts[0].block_type == "linear"
        assert parts[0].node_names == ["fc"]

    def test_partition_conv_graph(self):
        from mlasic.optimization import GraphPartitioner

        tensors = {
            "x": Tensor("x", TensorType((1, 3, 32, 32), np.float32)),
            "w": Tensor("w", TensorType((16, 3, 3, 3), np.float32), data=np.zeros((16, 3, 3, 3), dtype=np.float32)),
            "y": Tensor("y", TensorType((1, 16, 30, 30), np.float32)),
        }

        node = OpNode("conv", OpType.FUSED_CONV_RELU, ["x", "w"], ["y"])
        qp = QuantParams(scale=0.1, zero_point=0, calibrated=True)
        node.fused_attrs = FusedConvAttrs(
            in_channels=3, out_channels=16, kernel_shape=[3, 3],
            has_relu=True,
            weight_quant=[qp] * 16, input_quant=qp, output_quant=qp,
            requant_scale_fixed=[1000] * 16, requant_shift=16,
        )

        graph = Graph("test", {"conv": node}, tensors, ["x"], ["y"], stage="fused_dag")

        partitioner = GraphPartitioner()
        parts = partitioner.partition(graph)
        assert len(parts) == 1
        assert parts[0].block_type == "conv"

    def test_partition_attention_graph(self):
        from mlasic.optimization import GraphPartitioner

        tensors = {
            "x": Tensor("x", TensorType((1, 64), np.float32)),
            "y": Tensor("y", TensorType((1, 64), np.float32)),
        }
        node = OpNode("attn", OpType.FUSED_ATTENTION, ["x"], ["y"])
        node.fused_attrs = FusedAttentionAttrs(num_heads=4, head_dim=16, seq_len=8)

        graph = Graph("test", {"attn": node}, tensors, ["x"], ["y"], stage="fused_dag")

        partitioner = GraphPartitioner()
        parts = partitioner.partition(graph)
        assert len(parts) == 1
        assert parts[0].block_type == "attention"

    def test_merged_partitions(self):
        from mlasic.optimization import GraphPartitioner

        qp = QuantParams(scale=0.1, zero_point=0, calibrated=True)

        tensors = {
            "x": Tensor("x", TensorType((1, 128), np.float32)),
            "w1": Tensor("w1", TensorType((128, 128), np.float32), data=np.zeros((128, 128), dtype=np.float32)),
            "h": Tensor("h", TensorType((1, 128), np.float32)),
            "w2": Tensor("w2", TensorType((128, 128), np.float32), data=np.zeros((128, 128), dtype=np.float32)),
            "y": Tensor("y", TensorType((1, 128), np.float32)),
        }

        fc1 = OpNode("fc1", OpType.FUSED_LINEAR, ["x", "w1"], ["h"])
        fc1.fused_attrs = FusedLinearAttrs(
            input_dim=128, output_dim=128, has_relu=False,
            weight_quant=qp, input_quant=qp, output_quant=qp,
            requant_scale_fixed=1000, requant_shift=16,
        )
        fc2 = OpNode("fc2", OpType.FUSED_LINEAR, ["h", "w2"], ["y"])
        fc2.fused_attrs = FusedLinearAttrs(
            input_dim=128, output_dim=128, has_relu=False,
            weight_quant=qp, input_quant=qp, output_quant=qp,
            requant_scale_fixed=1000, requant_shift=16,
        )

        graph = Graph("test", {"fc1": fc1, "fc2": fc2}, tensors, ["x"], ["y"], stage="fused_dag")

        partitioner = GraphPartitioner()
        merged = partitioner.partition_merged(graph)
        assert len(merged) == 1
        assert merged[0].block_type == "linear"
        assert len(merged[0].node_names) == 2


# ======================================================================
# INT8 Interpreter Tests
# ======================================================================


class TestINT8FusedLayerNorm:
    """Test INT8 execution of FusedLayerNorm."""

    def test_basic_layer_norm(self):
        from mlasic.int8_interpreter import INT8Interpreter

        qp_in = QuantParams(scale=0.05, zero_point=0, calibrated=True)
        qp_out = QuantParams(scale=0.05, zero_point=0, calibrated=True)

        scale_data = np.ones(64, dtype=np.float32)
        bias_data = np.zeros(64, dtype=np.float32)

        attrs = FusedLayerNormAttrs(
            axis=-1, epsilon=1e-5, normalized_shape=(64,),
            input_quant=qp_in, output_quant=qp_out,
        )

        tensors = {
            "x": Tensor("x", TensorType((4, 64), np.int8)),
            "scale": Tensor("scale", TensorType((64,), np.float32), data=scale_data),
            "bias": Tensor("bias", TensorType((64,), np.float32), data=bias_data),
            "y": Tensor("y", TensorType((4, 64), np.int8)),
        }
        node = OpNode("ln", OpType.FUSED_LAYER_NORM, ["x", "scale", "bias"], ["y"])
        node.fused_attrs = attrs

        graph = Graph("test", {"ln": node}, tensors, ["x"], ["y"], stage="quantized")

        interp = INT8Interpreter(graph)
        x_q = qp_in.quantize(np.random.randn(4, 64).astype(np.float32))
        result = interp.run({"x": x_q})
        assert result["y"].dtype == np.int8
        assert result["y"].shape == (4, 64)


class TestINT8FusedActivation:
    """Test INT8 execution of FusedGELU and FusedSiLU."""

    def _make_graph(self, activation_type: str):
        qp = QuantParams(scale=0.05, zero_point=0, calibrated=True)
        op_type = OpType.FUSED_GELU if activation_type == "gelu" else OpType.FUSED_SILU

        tensors = {
            "x": Tensor("x", TensorType((1, 128), np.int8)),
            "y": Tensor("y", TensorType((1, 128), np.int8)),
        }
        node = OpNode("act", op_type, ["x"], ["y"])
        node.fused_attrs = FusedActivationAttrs(
            activation_type=activation_type,
            input_quant=qp, output_quant=qp,
        )
        return Graph("test", {"act": node}, tensors, ["x"], ["y"], stage="quantized")

    def test_gelu(self):
        from mlasic.int8_interpreter import INT8Interpreter

        graph = self._make_graph("gelu")
        interp = INT8Interpreter(graph)
        x_q = np.random.randint(-128, 127, (1, 128), dtype=np.int8)
        result = interp.run({"x": x_q})
        assert result["y"].dtype == np.int8
        assert result["y"].shape == (1, 128)

    def test_silu(self):
        from mlasic.int8_interpreter import INT8Interpreter

        graph = self._make_graph("silu")
        interp = INT8Interpreter(graph)
        x_q = np.random.randint(-128, 127, (1, 128), dtype=np.int8)
        result = interp.run({"x": x_q})
        assert result["y"].dtype == np.int8
        assert result["y"].shape == (1, 128)


class TestINT8FusedAttention:
    """Test INT8 execution of FusedAttention."""

    def test_basic_attention(self):
        from mlasic.int8_interpreter import INT8Interpreter

        embed_dim = 32
        num_heads = 4
        head_dim = 8
        seq_len = 4

        qp = QuantParams(scale=0.01, zero_point=0, calibrated=True)
        qp_act = QuantParams(scale=0.05, zero_point=5, calibrated=True)

        wq = np.random.randn(embed_dim, embed_dim).astype(np.float32)
        wk = np.random.randn(embed_dim, embed_dim).astype(np.float32)
        wv = np.random.randn(embed_dim, embed_dim).astype(np.float32)
        wo = np.random.randn(embed_dim, embed_dim).astype(np.float32)

        # Quantize weights
        wq_q = np.clip(np.floor(wq / 0.01 + 0.5), -127, 127).astype(np.int8)
        wk_q = np.clip(np.floor(wk / 0.01 + 0.5), -127, 127).astype(np.int8)
        wv_q = np.clip(np.floor(wv / 0.01 + 0.5), -127, 127).astype(np.int8)
        wo_q = np.clip(np.floor(wo / 0.01 + 0.5), -127, 127).astype(np.int8)

        attrs = FusedAttentionAttrs(
            num_heads=num_heads, head_dim=head_dim, seq_len=seq_len,
            q_weight_quant=qp, k_weight_quant=qp, v_weight_quant=qp,
            output_weight_quant=qp,
            input_quant=qp_act, output_quant=qp_act,
        )

        tensors = {
            "x": Tensor("x", TensorType((seq_len, embed_dim), np.int8)),
            "wq": Tensor("wq", TensorType((embed_dim, embed_dim), np.int8), data=wq_q),
            "wk": Tensor("wk", TensorType((embed_dim, embed_dim), np.int8), data=wk_q),
            "wv": Tensor("wv", TensorType((embed_dim, embed_dim), np.int8), data=wv_q),
            "wo": Tensor("wo", TensorType((embed_dim, embed_dim), np.int8), data=wo_q),
            "y": Tensor("y", TensorType((seq_len, embed_dim), np.int8)),
        }
        node = OpNode("attn", OpType.FUSED_ATTENTION, ["x", "wq", "wk", "wv", "wo"], ["y"])
        node.fused_attrs = attrs

        graph = Graph("test", {"attn": node}, tensors, ["x"], ["y"], stage="quantized")

        interp = INT8Interpreter(graph)
        x_q = qp_act.quantize(np.random.randn(seq_len, embed_dim).astype(np.float32))
        result = interp.run({"x": x_q})

        assert result["y"].dtype == np.int8
        assert result["y"].shape == (seq_len, embed_dim)


# ======================================================================
# Conv Weight Packing Tests
# ======================================================================


class TestConvWeightPacking:
    """Test ConvWeightPacker pack/unpack roundtrip."""

    def test_standard_conv_roundtrip(self):
        from mlasic.weight_packer import ConvWeightPacker

        packer = ConvWeightPacker(parallelism=16, row_bytes=128)
        oc, ic, kh, kw = 32, 3, 3, 3
        w_q = np.random.randint(-127, 127, (oc, ic, kh, kw), dtype=np.int8)

        attrs = FusedConvAttrs(
            in_channels=ic, out_channels=oc, kernel_shape=[kh, kw], group=1,
        )

        rows = packer.pack_conv_weights(w_q, attrs)
        assert len(rows) > 0

        # Unpack and verify roundtrip
        w_unpacked = packer.unpack_conv_weights(rows, oc, ic, [kh, kw], group=1)
        np.testing.assert_array_equal(w_q, w_unpacked)

    def test_depthwise_conv_roundtrip(self):
        from mlasic.weight_packer import ConvWeightPacker

        packer = ConvWeightPacker(parallelism=16, row_bytes=128)
        oc = 32
        w_q = np.random.randint(-127, 127, (oc, 1, 3, 3), dtype=np.int8)

        attrs = FusedConvAttrs(
            in_channels=oc, out_channels=oc, kernel_shape=[3, 3], group=oc,
        )

        rows = packer.pack_conv_weights(w_q, attrs)
        w_unpacked = packer.unpack_conv_weights(rows, oc, 1, [3, 3], group=oc)
        np.testing.assert_array_equal(w_q, w_unpacked)

    def test_1d_conv_roundtrip(self):
        from mlasic.weight_packer import ConvWeightPacker

        packer = ConvWeightPacker(parallelism=16, row_bytes=128)
        oc, ic, kw = 16, 8, 5
        w_q = np.random.randint(-127, 127, (oc, ic, kw), dtype=np.int8)

        attrs = FusedConvAttrs(
            in_channels=ic, out_channels=oc, kernel_shape=[kw], group=1,
        )

        rows = packer.pack_conv_weights(w_q, attrs)
        w_unpacked = packer.unpack_conv_weights(rows, oc, ic, [kw], group=1)
        np.testing.assert_array_equal(w_q, w_unpacked)


# ======================================================================
# Attention Weight Packing Tests
# ======================================================================


class TestAttentionWeightPacking:
    """Test AttentionWeightPacker."""

    def test_pack_attention_weights(self, tmp_path):
        from mlasic.weight_packer import AttentionWeightPacker

        packer = AttentionWeightPacker(parallelism=32, row_bytes=128)
        embed_dim = 64
        rng = np.random.RandomState(42)

        wq = rng.randint(-127, 127, (embed_dim, embed_dim), dtype=np.int8)
        wk = rng.randint(-127, 127, (embed_dim, embed_dim), dtype=np.int8)
        wv = rng.randint(-127, 127, (embed_dim, embed_dim), dtype=np.int8)
        wo = rng.randint(-127, 127, (embed_dim, embed_dim), dtype=np.int8)

        biases = {
            "q": np.zeros(embed_dim, dtype=np.int32),
            "k": np.zeros(embed_dim, dtype=np.int32),
        }

        packed = packer.pack_attention_weights(wq, wk, wv, wo, biases)
        assert "q_weight" in packed
        assert "k_weight" in packed
        assert "v_weight" in packed
        assert "o_weight" in packed
        assert "q_bias" in packed
        assert "k_bias" in packed
        assert len(packed["q_weight"]) > 0

        # Write to .mem files
        file_map = packer.write_attention_mem(packed, tmp_path, "layer0_attn")
        assert len(file_map) > 0
        for path_str in file_map.values():
            assert Path(path_str).exists()


# ======================================================================
# INT4 Quantization Tests
# ======================================================================


class TestINT4Quantization:
    """Test INT4Quantizer."""

    def test_quantize_weights(self):
        from mlasic.weight_packer import INT4Quantizer

        w_fp = np.random.randn(16, 32).astype(np.float32)
        w_q, params = INT4Quantizer.quantize_weights(w_fp, per_channel=True)

        assert w_q.dtype == np.int8
        assert w_q.shape == (16, 32)
        assert w_q.min() >= -7
        assert w_q.max() <= 7
        assert len(params) == 16
        assert all(p.bit_width == 4 for p in params)

    def test_quantize_per_tensor(self):
        from mlasic.weight_packer import INT4Quantizer

        w_fp = np.random.randn(8, 16).astype(np.float32) * 0.5
        w_q, params = INT4Quantizer.quantize_weights(w_fp, per_channel=False)

        assert w_q.min() >= -7
        assert w_q.max() <= 7
        # All channels should have same scale (per-tensor)
        assert all(p.scale == params[0].scale for p in params)

    def test_pack_unpack_int4_roundtrip(self):
        from mlasic.weight_packer import INT4Quantizer

        values = np.array([1, -2, 3, -4, 5, -6, 7, -7, 0, 1], dtype=np.int8)
        packed = INT4Quantizer.pack_int4(values)

        # 10 values → 5 bytes
        assert len(packed) == 5

        unpacked = INT4Quantizer.unpack_int4(packed, 10)
        np.testing.assert_array_equal(values, unpacked)

    def test_pack_unpack_odd_length(self):
        from mlasic.weight_packer import INT4Quantizer

        values = np.array([3, -5, 7], dtype=np.int8)
        packed = INT4Quantizer.pack_int4(values)
        assert len(packed) == 2  # padded to 4 values

        unpacked = INT4Quantizer.unpack_int4(packed, 3)
        np.testing.assert_array_equal(values, unpacked)

    def test_int4_accuracy_vs_int8(self):
        """INT4 quantization produces valid results and MSE is bounded."""
        from mlasic.weight_packer import INT4Quantizer

        rng = np.random.RandomState(42)
        w_fp = rng.randn(64, 128).astype(np.float32) * 0.3

        # INT8 quantization
        max_abs_8 = max(abs(w_fp.min()), abs(w_fp.max()))
        scale_8 = max_abs_8 / 127.0
        w_int8 = np.clip(np.floor(w_fp / scale_8 + 0.5), -127, 127).astype(np.int8)
        w_deq8 = w_int8.astype(np.float32) * scale_8

        # INT4 quantization
        w_int4, params = INT4Quantizer.quantize_weights(w_fp, per_channel=False)
        scale_4 = params[0].scale
        w_deq4 = w_int4.astype(np.float32) * scale_4

        # Compare MSE
        mse_8 = np.mean((w_fp - w_deq8) ** 2)
        mse_4 = np.mean((w_fp - w_deq4) ** 2)

        # INT4 MSE is higher (fewer levels), but should be bounded
        # With 7 levels vs 127 levels, ~18x^2 ≈ 324x worse MSE theoretical max
        assert mse_4 > 0  # INT4 has quantization error
        assert mse_4 < mse_8 * 500  # Very generous bound
        # Also check absolute MSE is reasonable (< 10% of signal variance)
        signal_var = np.var(w_fp)
        assert mse_4 < signal_var * 0.1

    def test_int4_rows_packing(self):
        from mlasic.weight_packer import INT4Quantizer

        q = INT4Quantizer()
        w = np.random.randint(-7, 7, (32, 64), dtype=np.int8)
        rows = q.pack_int4_rows(w, parallelism=64, row_bytes=128)
        assert len(rows) > 0
        # Each row should be 128 bytes (256 INT4 values packed)
        for row in rows:
            assert len(row) == 128


# ======================================================================
# Per-Tile Weight File Tests
# ======================================================================


class TestPerTileWeightFiles:
    """Test PerTileWeightWriter."""

    def test_per_tile_files_generated(self, tmp_path):
        from mlasic.weight_packer import PerTileWeightWriter

        writer = PerTileWeightWriter(tmp_path)
        w_q = np.random.randint(-127, 127, (64, 128), dtype=np.int8)

        tile_map = writer.write_per_tile_files("layer0", w_q, parallelism=64)

        assert len(tile_map) == 2  # 128/64 = 2 tiles
        for tile_id, info in tile_map.items():
            assert Path(info["file"]).exists()
            assert info["num_rows"] == 64  # input_dim rows per tile

    def test_per_tile_binary(self, tmp_path):
        from mlasic.weight_packer import PerTileWeightWriter

        writer = PerTileWeightWriter(tmp_path)
        w_q = np.random.randint(-127, 127, (32, 64), dtype=np.int8)

        tile_map = writer.write_per_tile_binary("layer0", w_q, parallelism=64)

        assert len(tile_map) == 1  # 64/64 = 1 tile
        for tile_id, info in tile_map.items():
            assert Path(info["file"]).exists()
            assert info["size_bytes"] == 32 * 128  # 32 rows × 128 bytes

    def test_tile_weight_map_json(self, tmp_path):
        from mlasic.weight_packer import PerTileWeightWriter

        writer = PerTileWeightWriter(tmp_path)
        w_q = np.random.randint(-127, 127, (16, 32), dtype=np.int8)

        writer.write_per_tile_files("test_layer", w_q, parallelism=16)

        import json
        map_path = tmp_path / "test_layer_tile_weight_map.json"
        assert map_path.exists()
        with open(map_path) as f:
            tile_map = json.load(f)
        assert len(tile_map) == 2  # 32/16 = 2 tiles


# ======================================================================
# Weight Streaming Scheduler Tests
# ======================================================================


class TestWeightStreamScheduler:
    """Test WeightStreamScheduler."""

    def _make_dag_schedule(self, layers):
        """Create a mock DAGSchedule from layer specs.

        layers: list of (name, compute_cycles, weight_bytes)
        """
        from mlasic.dag_scheduler import DAGSchedule
        from mlasic.ir import DAGLayerSchedule

        node_schedules = []
        current = 0
        for i, (name, cycles, wb) in enumerate(layers):
            ns = DAGLayerSchedule(
                node_index=i, node_name=name, op_type="FusedLinear",
                total_cycles=cycles, start_cycle=current, end_cycle=current + cycles,
                weight_bytes=wb, bias_bytes=0,
            )
            node_schedules.append(ns)
            current += cycles

        return DAGSchedule(
            total_cycles=current, clock_mhz=100,
            node_schedules=node_schedules, lifetimes=[],
            peak_activation_bytes=0,
            total_weight_bytes=sum(wb for _, _, wb in layers),
            total_bias_bytes=0,
        )

    def test_double_buffer_hides_latency(self):
        """Verify double-buffering overlaps weight loads with compute."""
        from mlasic.dag_scheduler import WeightStreamScheduler

        # 3 layers: each with 1000 compute cycles and 200 bytes of weights
        dag_sched = self._make_dag_schedule([
            ("layer0", 1000, 200),
            ("layer1", 1000, 200),
            ("layer2", 1000, 200),
        ])

        scheduler = WeightStreamScheduler(axi_width_bits=64)
        ws = scheduler.schedule(dag_sched)

        assert ws.total_cycles < ws.total_cycles_no_overlap
        assert ws.overlap_savings_cycles > 0

    def test_compute_bound_layers(self):
        """Layers with small weights should be compute-bound."""
        from mlasic.dag_scheduler import WeightStreamScheduler

        # Large compute, small weights
        dag_sched = self._make_dag_schedule([
            ("layer0", 10000, 64),
            ("layer1", 10000, 64),
        ])

        scheduler = WeightStreamScheduler(axi_width_bits=64)
        ws = scheduler.schedule(dag_sched)

        report = ws.dram_bandwidth_report
        assert report.compute_bound_count >= report.memory_bound_count

    def test_memory_bound_detection(self):
        """Layers with huge weights should be memory-bound."""
        from mlasic.dag_scheduler import WeightStreamScheduler

        # Small compute, large weights
        dag_sched = self._make_dag_schedule([
            ("layer0", 10, 100000),
        ])

        scheduler = WeightStreamScheduler(axi_width_bits=64)
        ws = scheduler.schedule(dag_sched)

        assert ws.plans[0].is_memory_bound is True
        report = ws.dram_bandwidth_report
        assert report.memory_bound_count >= 1

    def test_peak_sram_double_buffer(self):
        """Peak SRAM should be 2x the largest single layer's weights."""
        from mlasic.dag_scheduler import WeightStreamScheduler

        dag_sched = self._make_dag_schedule([
            ("layer0", 1000, 1024),
            ("layer1", 1000, 2048),
            ("layer2", 1000, 512),
        ])

        scheduler = WeightStreamScheduler(axi_width_bits=64)
        ws = scheduler.schedule(dag_sched)

        assert ws.peak_weight_sram_bytes == 2 * 2048

    def test_no_weight_layers_handled(self):
        """Layers with 0 weights should not cause issues."""
        from mlasic.dag_scheduler import WeightStreamScheduler

        dag_sched = self._make_dag_schedule([
            ("layer0", 1000, 0),
            ("layer1", 1000, 256),
        ])

        scheduler = WeightStreamScheduler(axi_width_bits=64)
        ws = scheduler.schedule(dag_sched)
        assert len(ws.plans) == 2
        assert ws.plans[0].weight_bytes == 0


class TestDRAMBandwidthAnalysis:
    """Test DRAM bandwidth analysis report."""

    def test_bandwidth_report_structure(self):
        from mlasic.dag_scheduler import WeightStreamScheduler

        from mlasic.dag_scheduler import DAGSchedule
        from mlasic.ir import DAGLayerSchedule

        ns = DAGLayerSchedule(
            node_index=0, node_name="test", op_type="FusedLinear",
            total_cycles=1000, start_cycle=0, end_cycle=1000,
            weight_bytes=512, bias_bytes=0,
        )
        dag_sched = DAGSchedule(
            total_cycles=1000, clock_mhz=100,
            node_schedules=[ns], lifetimes=[],
            peak_activation_bytes=0, total_weight_bytes=512, total_bias_bytes=0,
        )

        scheduler = WeightStreamScheduler(axi_width_bits=64)
        ws = scheduler.schedule(dag_sched)

        report = ws.dram_bandwidth_report
        assert report.axi_width_bits == 64
        assert report.clock_mhz == 100
        assert report.max_bandwidth_mbps > 0
        assert len(report.layer_analyses) == 1
        assert report.compute_bound_count + report.memory_bound_count == 1
        assert 0.0 <= report.compute_bound_ratio <= 1.0


# ======================================================================
# DAG Validator with new types
# ======================================================================


class TestDAGValidatorNewTypes:
    """Test that the DAG validator accepts FUSED_ATTENTION and FUSED_MLP."""

    def test_fused_attention_allowed(self):
        qp = QuantParams(scale=0.1, zero_point=0, calibrated=True)
        attrs = FusedAttentionAttrs(
            num_heads=4, head_dim=16, seq_len=8,
            q_weight_quant=qp, k_weight_quant=qp,
            v_weight_quant=qp, output_weight_quant=qp,
            input_quant=qp, output_quant=qp,
        )

        tensors = {
            "x": Tensor("x", TensorType((8, 64), np.float32)),
            "y": Tensor("y", TensorType((8, 64), np.float32)),
        }
        node = OpNode("attn", OpType.FUSED_ATTENTION, ["x"], ["y"])
        node.fused_attrs = attrs

        graph = Graph("test", {"attn": node}, tensors, ["x"], ["y"], stage="fused_dag")

        from mlasic.ir import IRValidator
        validator = IRValidator()
        errors = validator.validate_stage2d_dag(graph)
        assert not errors, f"Unexpected validation errors: {errors}"

    def test_fused_mlp_allowed(self):
        qp = QuantParams(scale=0.1, zero_point=0, calibrated=True)
        attrs = FusedMLPAttrs(
            input_dim=128, hidden_dim=512, output_dim=128, activation_type="gelu",
            fc1_weight_quant=qp, fc2_weight_quant=qp,
            input_quant=qp, mid_quant=qp, output_quant=qp,
        )

        tensors = {
            "x": Tensor("x", TensorType((1, 128), np.float32)),
            "y": Tensor("y", TensorType((1, 128), np.float32)),
        }
        node = OpNode("mlp", OpType.FUSED_MLP, ["x"], ["y"])
        node.fused_attrs = attrs

        graph = Graph("test", {"mlp": node}, tensors, ["x"], ["y"], stage="fused_dag")

        from mlasic.ir import IRValidator
        validator = IRValidator()
        errors = validator.validate_stage2d_dag(graph)
        assert not errors, f"Unexpected validation errors: {errors}"
