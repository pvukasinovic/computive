"""Tests for IR data structures: QuantParams, Graph, IRValidator."""

import numpy as np
import pytest

from mlasic.exceptions import IRValidationError
from mlasic.ir import (
    Graph,
    IRValidator,
    OpNode,
    OpType,
    QuantParams,
    Tensor,
    TensorType,
)

# ---------------------------------------------------------------------------
# QuantParams tests
# ---------------------------------------------------------------------------


class TestQuantParams:
    def test_quantize_dequantize_roundtrip(self):
        """Quantize then dequantize should be close to original."""
        qp = QuantParams(scale=0.1, zero_point=0)
        fp = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        q = qp.quantize(fp)
        recovered = qp.dequantize(q)
        np.testing.assert_allclose(recovered, fp, atol=qp.scale)

    def test_symmetric_clip_range(self):
        """Symmetric weights (zp=0) should clip to [-127, 127]."""
        qp = QuantParams(scale=1.0, zero_point=0)
        fp = np.array([-200.0, -127.0, 0.0, 127.0, 200.0], dtype=np.float32)
        q = qp.quantize(fp)
        assert q.min() == -127
        assert q.max() == 127

    def test_asymmetric_clip_range(self):
        """Asymmetric activations (zp!=0) should clip to [-128, 127]."""
        qp = QuantParams(scale=1.0, zero_point=10)
        fp = np.array([-200.0, -138.0, 0.0, 117.0, 200.0], dtype=np.float32)
        q = qp.quantize(fp)
        assert q.min() == -128
        assert q.max() == 127

    def test_round_half_up(self):
        """Exact 0.5 cases should round UP (toward +inf), not banker's rounding."""
        qp = QuantParams(scale=1.0, zero_point=0)
        # 0.5 should round to 1, not 0 (banker's would give 0)
        fp = np.array([0.5, 1.5, 2.5, -0.5, -1.5], dtype=np.float32)
        q = qp.quantize(fp)
        expected = np.array([1, 2, 3, 0, -1], dtype=np.int8)
        np.testing.assert_array_equal(q, expected)

    def test_unsigned_quantization(self):
        """Unsigned quantization clips to [0, 255]."""
        qp = QuantParams(scale=1.0, zero_point=128, signed=False)
        fp = np.array([-200.0, 0.0, 127.0], dtype=np.float32)
        q = qp.quantize(fp)
        assert q.dtype == np.uint8
        assert q.min() >= 0
        assert q.max() <= 255

    def test_dequantize_with_zero_point(self):
        """Dequantize correctly applies zero point."""
        qp = QuantParams(scale=0.5, zero_point=10)
        q = np.array([10, 20, 0], dtype=np.int8)
        fp = qp.dequantize(q)
        expected = np.array([0.0, 5.0, -5.0], dtype=np.float32)
        np.testing.assert_array_equal(fp, expected)


# ---------------------------------------------------------------------------
# TensorType tests
# ---------------------------------------------------------------------------


class TestTensorType:
    def test_shape_known(self):
        tt = TensorType(shape=(1, 640), dtype=np.dtype("float32"))
        assert tt.is_shape_known is True

    def test_shape_unknown(self):
        tt = TensorType(shape=(-1,), dtype=np.dtype("float32"))
        assert tt.is_shape_known is False

    def test_numel(self):
        tt = TensorType(shape=(2, 3, 4), dtype=np.dtype("float32"))
        assert tt.numel == 24

    def test_numel_unknown_raises(self):
        tt = TensorType(shape=(-1, 10), dtype=np.dtype("float32"))
        with pytest.raises(ValueError, match="unknown shape"):
            _ = tt.numel

    def test_size_bytes(self):
        tt = TensorType(shape=(10,), dtype=np.dtype("float32"))
        assert tt.size_bytes == 40  # 10 * 4 bytes

    def test_size_bytes_int8(self):
        tt = TensorType(shape=(100,), dtype=np.dtype("int8"))
        assert tt.size_bytes == 100


# ---------------------------------------------------------------------------
# OpType tests
# ---------------------------------------------------------------------------


class TestOpType:
    def test_from_onnx_valid(self):
        assert OpType.from_onnx("MatMul") == OpType.MATMUL
        assert OpType.from_onnx("Add") == OpType.ADD
        assert OpType.from_onnx("Relu") == OpType.RELU
        assert OpType.from_onnx("BatchNormalization") == OpType.BATCH_NORM
        assert OpType.from_onnx("Reshape") == OpType.RESHAPE
        assert OpType.from_onnx("Transpose") == OpType.TRANSPOSE
        assert OpType.from_onnx("Flatten") == OpType.FLATTEN

    def test_from_onnx_unsupported(self):
        with pytest.raises(ValueError, match="Unsupported"):
            OpType.from_onnx("LpNormalization")

    def test_is_fused(self):
        assert OpType.FUSED_LINEAR.is_fused is True
        assert OpType.FUSED_LINEAR_RELU.is_fused is True
        assert OpType.MATMUL.is_fused is False


# ---------------------------------------------------------------------------
# Graph tests
# ---------------------------------------------------------------------------


def _make_simple_graph() -> Graph:
    """Build a minimal graph: input -> MatMul(input, weight) -> output."""
    return Graph(
        name="test",
        nodes={
            "mm0": OpNode(
                name="mm0",
                op_type=OpType.MATMUL,
                inputs=["input", "weight"],
                outputs=["output"],
            ),
        },
        tensors={
            "input": Tensor(
                name="input",
                type=TensorType(shape=(1, 4), dtype=np.dtype("float32")),
            ),
            "weight": Tensor(
                name="weight",
                type=TensorType(shape=(4, 2), dtype=np.dtype("float32")),
                data=np.ones((4, 2), dtype=np.float32),
            ),
            "output": Tensor(
                name="output",
                type=TensorType(shape=(1, 2), dtype=np.dtype("float32")),
            ),
        },
        inputs=["input"],
        outputs=["output"],
    )


def _make_chain_graph() -> Graph:
    """Build: input -> MatMul -> Add -> Relu -> output."""
    return Graph(
        name="chain",
        nodes={
            "mm0": OpNode(
                name="mm0",
                op_type=OpType.MATMUL,
                inputs=["input", "weight"],
                outputs=["mm_out"],
            ),
            "add0": OpNode(
                name="add0",
                op_type=OpType.ADD,
                inputs=["mm_out", "bias"],
                outputs=["add_out"],
            ),
            "relu0": OpNode(
                name="relu0",
                op_type=OpType.RELU,
                inputs=["add_out"],
                outputs=["output"],
            ),
        },
        tensors={
            "input": Tensor(
                name="input",
                type=TensorType(shape=(1, 4), dtype=np.dtype("float32")),
            ),
            "weight": Tensor(
                name="weight",
                type=TensorType(shape=(4, 2), dtype=np.dtype("float32")),
                data=np.ones((4, 2), dtype=np.float32),
            ),
            "bias": Tensor(
                name="bias",
                type=TensorType(shape=(2,), dtype=np.dtype("float32")),
                data=np.zeros(2, dtype=np.float32),
            ),
            "mm_out": Tensor(
                name="mm_out",
                type=TensorType(shape=(1, 2), dtype=np.dtype("float32")),
            ),
            "add_out": Tensor(
                name="add_out",
                type=TensorType(shape=(1, 2), dtype=np.dtype("float32")),
            ),
            "output": Tensor(
                name="output",
                type=TensorType(shape=(1, 2), dtype=np.dtype("float32")),
            ),
        },
        inputs=["input"],
        outputs=["output"],
    )


class TestGraph:
    def test_topological_order_simple(self):
        g = _make_simple_graph()
        order = g.topological_order()
        assert order == ["mm0"]

    def test_topological_order_chain(self):
        g = _make_chain_graph()
        order = g.topological_order()
        assert order.index("mm0") < order.index("add0")
        assert order.index("add0") < order.index("relu0")

    def test_cycle_detection(self):
        """Graph with a cycle should raise ValueError from topological_order()."""
        g = Graph(
            name="cyclic",
            nodes={
                "a": OpNode(name="a", op_type=OpType.RELU, inputs=["t_b"], outputs=["t_a"]),
                "b": OpNode(name="b", op_type=OpType.RELU, inputs=["t_a"], outputs=["t_b"]),
            },
            tensors={
                "t_a": Tensor(
                    name="t_a",
                    type=TensorType(shape=(1,), dtype=np.dtype("float32")),
                ),
                "t_b": Tensor(
                    name="t_b",
                    type=TensorType(shape=(1,), dtype=np.dtype("float32")),
                ),
            },
            inputs=[],
            outputs=["t_a"],
        )
        with pytest.raises(ValueError, match="Cycle detected"):
            g.topological_order()

    def test_get_producer(self):
        g = _make_chain_graph()
        producer = g.get_producer("mm_out")
        assert producer is not None
        assert producer.name == "mm0"

    def test_get_producer_none_for_input(self):
        g = _make_chain_graph()
        assert g.get_producer("input") is None

    def test_get_consumers(self):
        g = _make_chain_graph()
        consumers = g.get_consumers("mm_out")
        assert len(consumers) == 1
        assert consumers[0].name == "add0"

    def test_invalidate_cache(self):
        g = _make_chain_graph()
        _ = g.topological_order()
        assert len(g._topo_order) > 0
        g.invalidate_cache()
        assert len(g._topo_order) == 0

    def test_validate_passes_for_valid_graph(self):
        g = _make_chain_graph()
        g.validate("raw")  # Should not raise

    def test_validate_raises_on_missing_tensor(self):
        g = _make_simple_graph()
        # Remove the weight tensor
        del g.tensors["weight"]
        g.invalidate_cache()
        with pytest.raises(IRValidationError, match="INV-1.3"):
            g.validate("raw")


# ---------------------------------------------------------------------------
# IRValidator invariant tests
# ---------------------------------------------------------------------------


class TestIRValidator:
    def test_inv_1_1_unique_tensor_names(self):
        """INV-1.1: Dict keys enforce uniqueness, so this always passes."""
        g = _make_simple_graph()
        v = IRValidator()
        errors = v.validate_stage1(g)
        assert not any("INV-1.1" in e for e in errors)

    def test_inv_1_3_missing_input_tensor(self):
        g = _make_simple_graph()
        del g.tensors["weight"]
        g.invalidate_cache()
        v = IRValidator()
        errors = v.validate_stage1(g)
        assert any("INV-1.3" in e for e in errors)

    def test_inv_1_4_missing_output_tensor(self):
        g = _make_simple_graph()
        del g.tensors["output"]
        g.invalidate_cache()
        v = IRValidator()
        errors = v.validate_stage1(g)
        assert any("INV-1.4" in e for e in errors)

    def test_inv_1_5_input_has_producer(self):
        """Graph input should have no producer."""
        g = _make_chain_graph()
        # Make mm_out a graph input — but it has a producer (mm0)
        g.inputs.append("mm_out")
        g.invalidate_cache()
        v = IRValidator()
        errors = v.validate_stage1(g)
        assert any("INV-1.5" in e for e in errors)

    def test_inv_1_6_output_has_no_producer(self):
        """Graph output must have exactly one producer."""
        g = _make_simple_graph()
        # Make 'input' a graph output — it has no producer
        g.outputs.append("input")
        g.invalidate_cache()
        v = IRValidator()
        errors = v.validate_stage1(g)
        assert any("INV-1.6" in e for e in errors)

    def test_inv_1_9_unknown_shape(self):
        g = _make_simple_graph()
        g.tensors["input"].type.shape = (-1, 4)
        v = IRValidator()
        errors = v.validate_stage1(g)
        assert any("INV-1.9" in e for e in errors)

    def test_inv_1_10_cycle(self):
        g = Graph(
            name="cyclic",
            nodes={
                "a": OpNode(name="a", op_type=OpType.RELU, inputs=["t_b"], outputs=["t_a"]),
                "b": OpNode(name="b", op_type=OpType.RELU, inputs=["t_a"], outputs=["t_b"]),
            },
            tensors={
                "t_a": Tensor(
                    name="t_a",
                    type=TensorType(shape=(1,), dtype=np.dtype("float32")),
                ),
                "t_b": Tensor(
                    name="t_b",
                    type=TensorType(shape=(1,), dtype=np.dtype("float32")),
                ),
            },
            inputs=[],
            outputs=["t_a"],
        )
        v = IRValidator()
        errors = v.validate_stage1(g)
        assert any("INV-1.10" in e for e in errors)

    def test_all_invariants_pass_on_valid_graph(self):
        g = _make_chain_graph()
        v = IRValidator()
        errors = v.validate_stage1(g)
        assert errors == []
