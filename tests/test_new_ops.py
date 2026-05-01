"""Tests for Phase 1 operator expansion (~80 tests).

Covers:
  - Per-operator interpreter tests (verify against NumPy/ORT on random inputs)
  - Per-operator parser roundtrip tests (ONNX→IR→interpret→match ORT)
  - Shape inference tests for Conv, pooling, Gather, Concat, Slice
  - Constant folding extension tests
  - Integration: parse synthetic CNN model
  - Integration: parse synthetic transformer model
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from mlasic.ingestion import ONNXParser
from mlasic.interpreter import IRInterpreter
from mlasic.ir import (
    ConvAttrs,
    GemmAttrs,
    Graph,
    NormAttrs,
    OpNode,
    OpType,
    PoolAttrs,
    Tensor,
    TensorType,
)
from mlasic.optimization import ConstantFoldingPass

# =========================================================================
# Helpers
# =========================================================================


def _run_ort(model: onnx.ModelProto, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run ONNX model through ORT and return outputs."""
    sess = ort.InferenceSession(model.SerializeToString())
    output_names = [o.name for o in sess.get_outputs()]
    results = sess.run(output_names, feeds)
    return dict(zip(output_names, results))


def _make_single_op_model(
    op_type: str,
    inputs: list[tuple[str, np.ndarray]],
    output_name: str,
    output_shape: list[int],
    output_dtype: int = TensorProto.FLOAT,
    attrs: dict | None = None,
    initializer_names: set[str] | None = None,
    extra_outputs: list[tuple[str, list[int], int]] | None = None,
) -> onnx.ModelProto:
    """Build a single-op ONNX model for testing."""
    attrs = attrs or {}
    initializer_names = initializer_names or set()

    graph_inputs = []
    initializers = []
    for name, data in inputs:
        if name in initializer_names:
            initializers.append(numpy_helper.from_array(data, name=name))
        else:
            elem = helper.np_dtype_to_tensor_dtype(data.dtype)
            graph_inputs.append(helper.make_tensor_value_info(name, elem, list(data.shape)))
            # Also add as initializer if it's meant to be a constant weight
            if data.dtype != np.float32 and name not in initializer_names:
                # int64 shape/index tensors are typically initializers
                pass

    all_input_names = [name for name, _ in inputs]
    all_output_names = [output_name]
    if extra_outputs:
        all_output_names.extend(o[0] for o in extra_outputs)

    node = helper.make_node(op_type, all_input_names, all_output_names, **attrs)
    graph_output = helper.make_tensor_value_info(output_name, output_dtype, output_shape)
    outputs = [graph_output]
    if extra_outputs:
        for eo_name, eo_shape, eo_dtype in extra_outputs:
            outputs.append(helper.make_tensor_value_info(eo_name, eo_dtype, eo_shape))

    graph = helper.make_graph([node], "test", graph_inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _parse_and_run(
    model: onnx.ModelProto,
    feeds: dict[str, np.ndarray],
    tmp_path: Path,
) -> dict[str, np.ndarray]:
    """Parse ONNX model → IR → interpret → return outputs."""
    path = tmp_path / "model.onnx"
    onnx.save(model, str(path))
    parser = ONNXParser(path)
    graph = parser.parse()
    interp = IRInterpreter(graph)
    return interp.run(feeds)


# =========================================================================
# Per-operator interpreter tests
# =========================================================================


class TestElementWiseOps:
    """Test trivial element-wise operators."""

    def test_mul(self):
        g = _simple_binary_graph(OpType.MUL)
        interp = IRInterpreter(g)
        a = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        b = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
        out = interp.run({"a": a, "b": b})
        np.testing.assert_allclose(out["out"], a * b)

    def test_div(self):
        g = _simple_binary_graph(OpType.DIV)
        interp = IRInterpreter(g)
        a = np.array([[6.0, 10.0, 15.0]], dtype=np.float32)
        b = np.array([[2.0, 5.0, 3.0]], dtype=np.float32)
        out = interp.run({"a": a, "b": b})
        np.testing.assert_allclose(out["out"], a / b)

    def test_sub(self):
        g = _simple_binary_graph(OpType.SUB)
        interp = IRInterpreter(g)
        a = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
        b = np.array([[3.0, 7.0, 5.0]], dtype=np.float32)
        out = interp.run({"a": a, "b": b})
        np.testing.assert_allclose(out["out"], a - b)

    def test_sigmoid(self):
        g = _simple_unary_graph(OpType.SIGMOID)
        interp = IRInterpreter(g)
        x = np.array([[-1.0, 0.0, 1.0, 5.0]], dtype=np.float32)
        out = interp.run({"x": x})
        expected = 1.0 / (1.0 + np.exp(-x))
        np.testing.assert_allclose(out["out"], expected, rtol=1e-5)

    def test_tanh(self):
        g = _simple_unary_graph(OpType.TANH)
        interp = IRInterpreter(g)
        x = np.array([[-2.0, -1.0, 0.0, 2.0]], dtype=np.float32)
        out = interp.run({"x": x})
        np.testing.assert_allclose(out["out"], np.tanh(x), rtol=1e-5)

    def test_erf(self):
        from scipy.special import erf as scipy_erf

        g = _simple_unary_graph(OpType.ERF)
        interp = IRInterpreter(g)
        x = np.array([[-1.0, 0.0, 0.5, 1.0]], dtype=np.float32)
        out = interp.run({"x": x})
        np.testing.assert_allclose(out["out"], scipy_erf(x), rtol=1e-5)

    def test_sqrt(self):
        g = _simple_unary_graph(OpType.SQRT)
        interp = IRInterpreter(g)
        x = np.array([[1.0, 4.0, 9.0, 16.0]], dtype=np.float32)
        out = interp.run({"x": x})
        np.testing.assert_allclose(out["out"], np.sqrt(x))

    def test_pow(self):
        g = _simple_binary_graph(OpType.POW)
        interp = IRInterpreter(g)
        a = np.array([[2.0, 3.0, 4.0]], dtype=np.float32)
        b = np.array([[3.0, 2.0, 0.5]], dtype=np.float32)
        out = interp.run({"a": a, "b": b})
        np.testing.assert_allclose(out["out"], np.power(a, b))

    def test_clip(self):
        g = Graph(
            name="test",
            nodes={
                "clip": OpNode(
                    name="clip",
                    op_type=OpType.CLIP,
                    inputs=["x", "lo", "hi"],
                    outputs=["out"],
                )
            },
            tensors={
                "x": Tensor("x", TensorType((1, 4), np.dtype("float32"))),
                "lo": Tensor("lo", TensorType((), np.dtype("float32")), data=np.float32(0.0)),
                "hi": Tensor("hi", TensorType((), np.dtype("float32")), data=np.float32(6.0)),
                "out": Tensor("out", TensorType((1, 4), np.dtype("float32"))),
            },
            inputs=["x"],
            outputs=["out"],
        )
        interp = IRInterpreter(g)
        x = np.array([[-1.0, 3.0, 7.0, 0.5]], dtype=np.float32)
        out = interp.run({"x": x})
        np.testing.assert_allclose(out["out"], np.clip(x, 0, 6))


class TestShapeOps:
    """Test shape manipulation operators."""

    def test_squeeze(self):
        g = Graph(
            name="test",
            nodes={
                "sq": OpNode(
                    name="sq",
                    op_type=OpType.SQUEEZE,
                    inputs=["x", "axes"],
                    outputs=["out"],
                )
            },
            tensors={
                "x": Tensor("x", TensorType((1, 3, 1), np.dtype("float32"))),
                "axes": Tensor(
                    "axes",
                    TensorType((2,), np.dtype("int64")),
                    data=np.array([0, 2], dtype=np.int64),
                ),
                "out": Tensor("out", TensorType((3,), np.dtype("float32"))),
            },
            inputs=["x"],
            outputs=["out"],
        )
        interp = IRInterpreter(g)
        x = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32)
        out = interp.run({"x": x})
        assert out["out"].shape == (3,)
        np.testing.assert_allclose(out["out"], [1.0, 2.0, 3.0])

    def test_unsqueeze(self):
        g = Graph(
            name="test",
            nodes={
                "usq": OpNode(
                    name="usq",
                    op_type=OpType.UNSQUEEZE,
                    inputs=["x", "axes"],
                    outputs=["out"],
                )
            },
            tensors={
                "x": Tensor("x", TensorType((3,), np.dtype("float32"))),
                "axes": Tensor(
                    "axes",
                    TensorType((2,), np.dtype("int64")),
                    data=np.array([0, 2], dtype=np.int64),
                ),
                "out": Tensor("out", TensorType((1, 3, 1), np.dtype("float32"))),
            },
            inputs=["x"],
            outputs=["out"],
        )
        interp = IRInterpreter(g)
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = interp.run({"x": x})
        assert out["out"].shape == (1, 3, 1)

    def test_concat(self):
        g = Graph(
            name="test",
            nodes={
                "cat": OpNode(
                    name="cat",
                    op_type=OpType.CONCAT,
                    inputs=["a", "b"],
                    outputs=["out"],
                    attributes={"axis": 1},
                )
            },
            tensors={
                "a": Tensor("a", TensorType((1, 2), np.dtype("float32"))),
                "b": Tensor("b", TensorType((1, 3), np.dtype("float32"))),
                "out": Tensor("out", TensorType((1, 5), np.dtype("float32"))),
            },
            inputs=["a", "b"],
            outputs=["out"],
        )
        interp = IRInterpreter(g)
        a = np.array([[1.0, 2.0]], dtype=np.float32)
        b = np.array([[3.0, 4.0, 5.0]], dtype=np.float32)
        out = interp.run({"a": a, "b": b})
        np.testing.assert_allclose(out["out"], [[1, 2, 3, 4, 5]])

    def test_slice(self):
        g = Graph(
            name="test",
            nodes={
                "sl": OpNode(
                    name="sl",
                    op_type=OpType.SLICE,
                    inputs=["x", "starts", "ends", "axes"],
                    outputs=["out"],
                )
            },
            tensors={
                "x": Tensor("x", TensorType((1, 6), np.dtype("float32"))),
                "starts": Tensor(
                    "starts",
                    TensorType((1,), np.dtype("int64")),
                    data=np.array([1], dtype=np.int64),
                ),
                "ends": Tensor(
                    "ends",
                    TensorType((1,), np.dtype("int64")),
                    data=np.array([4], dtype=np.int64),
                ),
                "axes": Tensor(
                    "axes",
                    TensorType((1,), np.dtype("int64")),
                    data=np.array([1], dtype=np.int64),
                ),
                "out": Tensor("out", TensorType((1, 3), np.dtype("float32"))),
            },
            inputs=["x"],
            outputs=["out"],
        )
        interp = IRInterpreter(g)
        x = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
        out = interp.run({"x": x})
        np.testing.assert_allclose(out["out"], [[1, 2, 3]])

    def test_split(self):
        g = Graph(
            name="test",
            nodes={
                "sp": OpNode(
                    name="sp",
                    op_type=OpType.SPLIT,
                    inputs=["x", "split_sizes"],
                    outputs=["out1", "out2"],
                    attributes={"axis": 1},
                )
            },
            tensors={
                "x": Tensor("x", TensorType((1, 5), np.dtype("float32"))),
                "split_sizes": Tensor(
                    "split_sizes",
                    TensorType((2,), np.dtype("int64")),
                    data=np.array([2, 3], dtype=np.int64),
                ),
                "out1": Tensor("out1", TensorType((1, 2), np.dtype("float32"))),
                "out2": Tensor("out2", TensorType((1, 3), np.dtype("float32"))),
            },
            inputs=["x"],
            outputs=["out1", "out2"],
        )
        interp = IRInterpreter(g)
        x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
        out = interp.run({"x": x})
        np.testing.assert_allclose(out["out1"], [[1, 2]])
        np.testing.assert_allclose(out["out2"], [[3, 4, 5]])

    def test_gather(self):
        g = Graph(
            name="test",
            nodes={
                "ga": OpNode(
                    name="ga",
                    op_type=OpType.GATHER,
                    inputs=["data", "indices"],
                    outputs=["out"],
                    attributes={"axis": 0},
                )
            },
            tensors={
                "data": Tensor(
                    "data",
                    TensorType((4, 3), np.dtype("float32")),
                    data=np.arange(12, dtype=np.float32).reshape(4, 3),
                ),
                "indices": Tensor("indices", TensorType((2,), np.dtype("int64"))),
                "out": Tensor("out", TensorType((2, 3), np.dtype("float32"))),
            },
            inputs=["indices"],
            outputs=["out"],
        )
        interp = IRInterpreter(g)
        indices = np.array([0, 2], dtype=np.int64)
        out = interp.run({"indices": indices})
        expected = np.take(np.arange(12).reshape(4, 3), [0, 2], axis=0)
        np.testing.assert_allclose(out["out"], expected)

    def test_shape(self):
        g = Graph(
            name="test",
            nodes={
                "sh": OpNode(
                    name="sh",
                    op_type=OpType.SHAPE,
                    inputs=["x"],
                    outputs=["out"],
                )
            },
            tensors={
                "x": Tensor("x", TensorType((2, 3, 4), np.dtype("float32"))),
                "out": Tensor("out", TensorType((3,), np.dtype("int64"))),
            },
            inputs=["x"],
            outputs=["out"],
        )
        interp = IRInterpreter(g)
        x = np.zeros((2, 3, 4), dtype=np.float32)
        out = interp.run({"x": x})
        np.testing.assert_array_equal(out["out"], [2, 3, 4])

    def test_constant_of_shape(self):
        g = Graph(
            name="test",
            nodes={
                "cos": OpNode(
                    name="cos",
                    op_type=OpType.CONSTANT_OF_SHAPE,
                    inputs=["shape_input"],
                    outputs=["out"],
                    attributes={"value": np.array([1.0], dtype=np.float32)},
                )
            },
            tensors={
                "shape_input": Tensor(
                    "shape_input",
                    TensorType((2,), np.dtype("int64")),
                    data=np.array([3, 4], dtype=np.int64),
                ),
                "out": Tensor("out", TensorType((3, 4), np.dtype("float32"))),
            },
            inputs=[],
            outputs=["out"],
        )
        interp = IRInterpreter(g)
        out = interp.run({})
        assert out["out"].shape == (3, 4)
        np.testing.assert_allclose(out["out"], np.ones((3, 4)))

    def test_cast(self):
        g = Graph(
            name="test",
            nodes={
                "c": OpNode(
                    name="c",
                    op_type=OpType.CAST,
                    inputs=["x"],
                    outputs=["out"],
                    attributes={"to": 7},  # INT64
                )
            },
            tensors={
                "x": Tensor("x", TensorType((3,), np.dtype("float32"))),
                "out": Tensor("out", TensorType((3,), np.dtype("int64"))),
            },
            inputs=["x"],
            outputs=["out"],
        )
        interp = IRInterpreter(g)
        x = np.array([1.5, 2.7, 3.0], dtype=np.float32)
        out = interp.run({"x": x})
        assert out["out"].dtype == np.int64
        np.testing.assert_array_equal(out["out"], [1, 2, 3])

    def test_pad(self):
        g = Graph(
            name="test",
            nodes={
                "p": OpNode(
                    name="p",
                    op_type=OpType.PAD,
                    inputs=["x", "pads", "const_val"],
                    outputs=["out"],
                    attributes={"mode": "constant"},
                )
            },
            tensors={
                "x": Tensor("x", TensorType((1, 3), np.dtype("float32"))),
                "pads": Tensor(
                    "pads",
                    TensorType((4,), np.dtype("int64")),
                    data=np.array([0, 1, 0, 2], dtype=np.int64),
                ),
                "const_val": Tensor(
                    "const_val",
                    TensorType((), np.dtype("float32")),
                    data=np.float32(9.0),
                ),
                "out": Tensor("out", TensorType((1, 6), np.dtype("float32"))),
            },
            inputs=["x"],
            outputs=["out"],
        )
        interp = IRInterpreter(g)
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        out = interp.run({"x": x})
        np.testing.assert_allclose(out["out"], [[9, 1, 2, 3, 9, 9]])

    def test_where(self):
        g = Graph(
            name="test",
            nodes={
                "w": OpNode(
                    name="w",
                    op_type=OpType.WHERE,
                    inputs=["cond", "a", "b"],
                    outputs=["out"],
                )
            },
            tensors={
                "cond": Tensor("cond", TensorType((1, 4), np.dtype("float32"))),
                "a": Tensor("a", TensorType((1, 4), np.dtype("float32"))),
                "b": Tensor("b", TensorType((1, 4), np.dtype("float32"))),
                "out": Tensor("out", TensorType((1, 4), np.dtype("float32"))),
            },
            inputs=["cond", "a", "b"],
            outputs=["out"],
        )
        interp = IRInterpreter(g)
        cond = np.array([[1, 0, 1, 0]], dtype=np.float32)
        a = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
        b = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        out = interp.run({"cond": cond, "a": a, "b": b})
        np.testing.assert_allclose(out["out"], [[10, 2, 30, 4]])

    def test_expand(self):
        g = Graph(
            name="test",
            nodes={
                "e": OpNode(
                    name="e",
                    op_type=OpType.EXPAND,
                    inputs=["x", "new_shape"],
                    outputs=["out"],
                )
            },
            tensors={
                "x": Tensor("x", TensorType((1, 3), np.dtype("float32"))),
                "new_shape": Tensor(
                    "new_shape",
                    TensorType((2,), np.dtype("int64")),
                    data=np.array([4, 3], dtype=np.int64),
                ),
                "out": Tensor("out", TensorType((4, 3), np.dtype("float32"))),
            },
            inputs=["x"],
            outputs=["out"],
        )
        interp = IRInterpreter(g)
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        out = interp.run({"x": x})
        assert out["out"].shape == (4, 3)
        np.testing.assert_allclose(out["out"][0], [1, 2, 3])

    def test_tile(self):
        g = Graph(
            name="test",
            nodes={
                "t": OpNode(
                    name="t",
                    op_type=OpType.TILE,
                    inputs=["x", "repeats"],
                    outputs=["out"],
                )
            },
            tensors={
                "x": Tensor("x", TensorType((1, 2), np.dtype("float32"))),
                "repeats": Tensor(
                    "repeats",
                    TensorType((2,), np.dtype("int64")),
                    data=np.array([3, 2], dtype=np.int64),
                ),
                "out": Tensor("out", TensorType((3, 4), np.dtype("float32"))),
            },
            inputs=["x"],
            outputs=["out"],
        )
        interp = IRInterpreter(g)
        x = np.array([[1.0, 2.0]], dtype=np.float32)
        out = interp.run({"x": x})
        expected = np.tile(x, [3, 2])
        np.testing.assert_allclose(out["out"], expected)


class TestComputeOps:
    """Test compute operators: Conv, Gemm, Softmax, LayerNorm, pooling."""

    def test_conv2d_basic(self, tmp_path: Path):
        """Conv2D 1x1→1x1, 3x3 kernel, no padding, vs ORT."""
        rng = np.random.RandomState(42)
        x_data = rng.randn(1, 1, 5, 5).astype(np.float32)
        w_data = rng.randn(1, 1, 3, 3).astype(np.float32)

        model = _make_single_op_model(
            "Conv",
            [("x", x_data), ("w", w_data)],
            "out",
            [1, 1, 3, 3],
            attrs={"kernel_shape": [3, 3]},
            initializer_names={"w"},
        )
        ort_out = _run_ort(model, {"x": x_data})
        ir_out = _parse_and_run(model, {"x": x_data}, tmp_path)
        np.testing.assert_allclose(ir_out["out"], ort_out["out"], rtol=1e-5, atol=1e-5)

    def test_conv2d_with_padding_stride(self, tmp_path: Path):
        """Conv2D with padding and stride."""
        rng = np.random.RandomState(43)
        x_data = rng.randn(1, 1, 8, 8).astype(np.float32)
        w_data = rng.randn(2, 1, 3, 3).astype(np.float32)
        b_data = rng.randn(2).astype(np.float32)

        model = _make_single_op_model(
            "Conv",
            [("x", x_data), ("w", w_data), ("b", b_data)],
            "out",
            [1, 2, 4, 4],
            attrs={"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1]},
            initializer_names={"w", "b"},
        )
        ort_out = _run_ort(model, {"x": x_data})
        ir_out = _parse_and_run(model, {"x": x_data}, tmp_path)
        np.testing.assert_allclose(ir_out["out"], ort_out["out"], rtol=1e-4, atol=1e-4)

    def test_conv2d_grouped(self, tmp_path: Path):
        """Depthwise Conv2D (group=channels)."""
        rng = np.random.RandomState(44)
        x_data = rng.randn(1, 4, 6, 6).astype(np.float32)
        # Depthwise: group=4, C_out=4, C_in/group=1
        w_data = rng.randn(4, 1, 3, 3).astype(np.float32)

        model = _make_single_op_model(
            "Conv",
            [("x", x_data), ("w", w_data)],
            "out",
            [1, 4, 4, 4],
            attrs={"kernel_shape": [3, 3], "group": 4},
            initializer_names={"w"},
        )
        ort_out = _run_ort(model, {"x": x_data})
        ir_out = _parse_and_run(model, {"x": x_data}, tmp_path)
        np.testing.assert_allclose(ir_out["out"], ort_out["out"], rtol=1e-4, atol=1e-4)

    def test_gemm(self, tmp_path: Path):
        """Gemm: alpha*A@B + beta*C, with transB."""
        rng = np.random.RandomState(45)
        a = rng.randn(2, 3).astype(np.float32)
        b = rng.randn(4, 3).astype(np.float32)  # transB=1 → (3, 4)
        c = rng.randn(4).astype(np.float32)

        model = _make_single_op_model(
            "Gemm",
            [("a", a), ("b", b), ("c", c)],
            "out",
            [2, 4],
            attrs={"alpha": 1.0, "beta": 1.0, "transB": 1},
            initializer_names={"b", "c"},
        )
        ort_out = _run_ort(model, {"a": a})
        ir_out = _parse_and_run(model, {"a": a}, tmp_path)
        np.testing.assert_allclose(ir_out["out"], ort_out["out"], rtol=1e-5, atol=1e-5)

    def test_softmax(self, tmp_path: Path):
        """Softmax along last axis."""
        x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        model = _make_single_op_model("Softmax", [("x", x)], "out", [2, 3], attrs={"axis": -1})
        ort_out = _run_ort(model, {"x": x})
        ir_out = _parse_and_run(model, {"x": x}, tmp_path)
        np.testing.assert_allclose(ir_out["out"], ort_out["out"], rtol=1e-5, atol=1e-6)

    def test_layer_norm(self, tmp_path: Path):
        """LayerNormalization with scale and bias."""
        rng = np.random.RandomState(46)
        x = rng.randn(1, 4, 8).astype(np.float32)
        scale = np.ones(8, dtype=np.float32)
        bias = np.zeros(8, dtype=np.float32)

        model = _make_single_op_model(
            "LayerNormalization",
            [("x", x), ("scale", scale), ("bias", bias)],
            "out",
            [1, 4, 8],
            attrs={"axis": -1, "epsilon": 1e-5},
            initializer_names={"scale", "bias"},
        )
        ort_out = _run_ort(model, {"x": x})
        ir_out = _parse_and_run(model, {"x": x}, tmp_path)
        np.testing.assert_allclose(ir_out["out"], ort_out["out"], rtol=1e-4, atol=1e-5)

    def test_maxpool(self, tmp_path: Path):
        """MaxPool 2x2, stride 2."""
        rng = np.random.RandomState(47)
        x = rng.randn(1, 1, 4, 4).astype(np.float32)

        model = _make_single_op_model(
            "MaxPool",
            [("x", x)],
            "out",
            [1, 1, 2, 2],
            attrs={"kernel_shape": [2, 2], "strides": [2, 2]},
        )
        ort_out = _run_ort(model, {"x": x})
        ir_out = _parse_and_run(model, {"x": x}, tmp_path)
        np.testing.assert_allclose(ir_out["out"], ort_out["out"])

    def test_averagepool(self, tmp_path: Path):
        """AveragePool 2x2, stride 2."""
        rng = np.random.RandomState(48)
        x = rng.randn(1, 1, 4, 4).astype(np.float32)

        model = _make_single_op_model(
            "AveragePool",
            [("x", x)],
            "out",
            [1, 1, 2, 2],
            attrs={"kernel_shape": [2, 2], "strides": [2, 2]},
        )
        ort_out = _run_ort(model, {"x": x})
        ir_out = _parse_and_run(model, {"x": x}, tmp_path)
        np.testing.assert_allclose(ir_out["out"], ort_out["out"], rtol=1e-5, atol=1e-6)

    def test_global_average_pool(self, tmp_path: Path):
        """GlobalAveragePool."""
        rng = np.random.RandomState(49)
        x = rng.randn(1, 3, 4, 4).astype(np.float32)

        model = _make_single_op_model(
            "GlobalAveragePool",
            [("x", x)],
            "out",
            [1, 3, 1, 1],
        )
        ort_out = _run_ort(model, {"x": x})
        ir_out = _parse_and_run(model, {"x": x}, tmp_path)
        np.testing.assert_allclose(ir_out["out"], ort_out["out"], rtol=1e-5, atol=1e-6)

    def test_reduce_mean(self, tmp_path: Path):
        """ReduceMean over axis 1."""
        x = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)  # (1, 2, 2)

        # Build manually — ReduceMean opset 18 takes axes as input
        graph_input = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 2])
        graph_output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1, 2])
        axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
        node = helper.make_node("ReduceMean", ["x", "axes"], ["out"], keepdims=1)
        graph = helper.make_graph([node], "test", [graph_input], [graph_output], initializer=[axes])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = 8

        ort_out = _run_ort(model, {"x": x})
        ir_out = _parse_and_run(model, {"x": x}, tmp_path)
        np.testing.assert_allclose(ir_out["out"], ort_out["out"], rtol=1e-5)

    def test_reduce_sum(self, tmp_path: Path):
        """ReduceSum over axis 1."""
        x = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)

        graph_input = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 2])
        graph_output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1, 2])
        axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
        node = helper.make_node("ReduceSum", ["x", "axes"], ["out"], keepdims=1)
        graph = helper.make_graph([node], "test", [graph_input], [graph_output], initializer=[axes])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = 8

        ort_out = _run_ort(model, {"x": x})
        ir_out = _parse_and_run(model, {"x": x}, tmp_path)
        np.testing.assert_allclose(ir_out["out"], ort_out["out"], rtol=1e-5)

    def test_conv_transpose(self, tmp_path: Path):
        """ConvTranspose basic."""
        rng = np.random.RandomState(50)
        x = rng.randn(1, 1, 3, 3).astype(np.float32)
        w = rng.randn(1, 1, 3, 3).astype(np.float32)

        model = _make_single_op_model(
            "ConvTranspose",
            [("x", x), ("w", w)],
            "out",
            [1, 1, 5, 5],
            attrs={"kernel_shape": [3, 3]},
            initializer_names={"w"},
        )
        ort_out = _run_ort(model, {"x": x})
        ir_out = _parse_and_run(model, {"x": x}, tmp_path)
        np.testing.assert_allclose(ir_out["out"], ort_out["out"], rtol=1e-4, atol=1e-4)


# =========================================================================
# Parser roundtrip tests (ONNX → IR → interpret → match ORT)
# =========================================================================


class TestParserRoundtrip:
    """Test that parsing ONNX → IR and running through interpreter matches ORT."""

    def test_mul_roundtrip(self, tmp_path: Path):
        a = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        b = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
        model = _make_single_op_model(
            "Mul", [("a", a), ("b", b)], "out", [1, 3], initializer_names={"b"}
        )
        ort_out = _run_ort(model, {"a": a})
        ir_out = _parse_and_run(model, {"a": a}, tmp_path)
        np.testing.assert_allclose(ir_out["out"], ort_out["out"])

    def test_sigmoid_roundtrip(self, tmp_path: Path):
        x = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
        model = _make_single_op_model("Sigmoid", [("x", x)], "out", [1, 3])
        ort_out = _run_ort(model, {"x": x})
        ir_out = _parse_and_run(model, {"x": x}, tmp_path)
        np.testing.assert_allclose(ir_out["out"], ort_out["out"], rtol=1e-5)

    def test_tanh_roundtrip(self, tmp_path: Path):
        x = np.array([[-2.0, 0.0, 2.0]], dtype=np.float32)
        model = _make_single_op_model("Tanh", [("x", x)], "out", [1, 3])
        ort_out = _run_ort(model, {"x": x})
        ir_out = _parse_and_run(model, {"x": x}, tmp_path)
        np.testing.assert_allclose(ir_out["out"], ort_out["out"], rtol=1e-5)

    def test_erf_roundtrip(self, tmp_path: Path):
        x = np.array([[-1.0, 0.0, 0.5, 1.0]], dtype=np.float32)
        model = _make_single_op_model("Erf", [("x", x)], "out", [1, 4])
        ort_out = _run_ort(model, {"x": x})
        ir_out = _parse_and_run(model, {"x": x}, tmp_path)
        np.testing.assert_allclose(ir_out["out"], ort_out["out"], rtol=1e-5)

    def test_sqrt_roundtrip(self, tmp_path: Path):
        x = np.array([[1.0, 4.0, 9.0]], dtype=np.float32)
        model = _make_single_op_model("Sqrt", [("x", x)], "out", [1, 3])
        ort_out = _run_ort(model, {"x": x})
        ir_out = _parse_and_run(model, {"x": x}, tmp_path)
        np.testing.assert_allclose(ir_out["out"], ort_out["out"])

    def test_gather_roundtrip(self, tmp_path: Path):
        """Gather with axis=0 (embedding lookup)."""
        data = np.arange(12, dtype=np.float32).reshape(4, 3)
        indices = np.array([0, 2], dtype=np.int64)

        graph_input_idx = helper.make_tensor_value_info("indices", TensorProto.INT64, [2])
        graph_output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])
        data_init = numpy_helper.from_array(data, name="data")
        node = helper.make_node("Gather", ["data", "indices"], ["out"], axis=0)
        graph = helper.make_graph(
            [node], "test", [graph_input_idx], [graph_output], initializer=[data_init]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8

        ort_out = _run_ort(model, {"indices": indices})
        ir_out = _parse_and_run(model, {"indices": indices}, tmp_path)
        np.testing.assert_allclose(ir_out["out"], ort_out["out"])


# =========================================================================
# Shape inference tests
# =========================================================================


class TestShapeInference:
    """Test that ONNX shape inference resolves shapes for new ops."""

    def test_conv_shape_inference(self, tmp_path: Path):
        """Conv output shape is correctly inferred."""
        rng = np.random.RandomState(42)
        x = rng.randn(1, 1, 8, 8).astype(np.float32)
        w = rng.randn(4, 1, 3, 3).astype(np.float32)
        model = _make_single_op_model(
            "Conv",
            [("x", x), ("w", w)],
            "out",
            [1, 4, 6, 6],
            attrs={"kernel_shape": [3, 3]},
            initializer_names={"w"},
        )
        path = tmp_path / "model.onnx"
        onnx.save(model, str(path))
        graph = ONNXParser(path).parse()
        out_shape = graph.tensors[graph.outputs[0]].type.shape
        assert out_shape == (1, 4, 6, 6)

    def test_maxpool_shape_inference(self, tmp_path: Path):
        """MaxPool output shape is correctly inferred."""
        x = np.zeros((1, 1, 4, 4), dtype=np.float32)
        model = _make_single_op_model(
            "MaxPool",
            [("x", x)],
            "out",
            [1, 1, 2, 2],
            attrs={"kernel_shape": [2, 2], "strides": [2, 2]},
        )
        path = tmp_path / "model.onnx"
        onnx.save(model, str(path))
        graph = ONNXParser(path).parse()
        out_shape = graph.tensors[graph.outputs[0]].type.shape
        assert out_shape == (1, 1, 2, 2)

    def test_concat_shape_inference(self, tmp_path: Path):
        """Concat output shape is correctly inferred."""
        graph_in_a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 2])
        graph_in_b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 3])
        graph_out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 5])
        node = helper.make_node("Concat", ["a", "b"], ["out"], axis=1)
        graph = helper.make_graph([node], "test", [graph_in_a, graph_in_b], [graph_out])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8

        path = tmp_path / "model.onnx"
        onnx.save(model, str(path))
        ir_graph = ONNXParser(path).parse()
        out_shape = ir_graph.tensors[ir_graph.outputs[0]].type.shape
        assert out_shape == (1, 5)

    def test_gather_shape_inference(self, tmp_path: Path):
        """Gather output shape is correctly inferred."""
        data = np.arange(12, dtype=np.float32).reshape(4, 3)
        data_init = numpy_helper.from_array(data, name="data")
        idx_input = helper.make_tensor_value_info("indices", TensorProto.INT64, [2])
        graph_out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])
        node = helper.make_node("Gather", ["data", "indices"], ["out"], axis=0)
        graph = helper.make_graph([node], "test", [idx_input], [graph_out], initializer=[data_init])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8

        path = tmp_path / "model.onnx"
        onnx.save(model, str(path))
        ir_graph = ONNXParser(path).parse()
        out_shape = ir_graph.tensors[ir_graph.outputs[0]].type.shape
        assert out_shape == (2, 3)


# =========================================================================
# Constant folding tests
# =========================================================================


class TestConstantFoldingExtension:
    """Test that constant folding handles new ops."""

    def _make_graph_with_constant_op(
        self, op_type: OpType, const_inputs: dict[str, np.ndarray], attrs: dict | None = None
    ) -> Graph:
        """Build a graph: [constant inputs] → op → output."""
        attrs = attrs or {}
        tensors = {}
        input_names = []
        for name, data in const_inputs.items():
            tensors[name] = Tensor(name, TensorType(tuple(data.shape), data.dtype), data=data)
            input_names.append(name)
        out_name = "out"
        tensors[out_name] = Tensor(out_name, TensorType((1,), np.dtype("float32")))
        node = OpNode(
            name="op",
            op_type=op_type,
            inputs=input_names,
            outputs=[out_name],
            attributes=attrs,
        )
        return Graph(
            name="test",
            nodes={"op": node},
            tensors=tensors,
            inputs=[],
            outputs=[out_name],
        )

    def test_fold_mul(self):
        a = np.array([2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0], dtype=np.float32)
        g = self._make_graph_with_constant_op(OpType.MUL, {"a": a, "b": b})
        g = ConstantFoldingPass().run(g)
        assert len(g.nodes) == 0
        np.testing.assert_allclose(g.tensors["out"].data, [8, 15])

    def test_fold_div(self):
        a = np.array([10.0, 6.0], dtype=np.float32)
        b = np.array([2.0, 3.0], dtype=np.float32)
        g = self._make_graph_with_constant_op(OpType.DIV, {"a": a, "b": b})
        g = ConstantFoldingPass().run(g)
        assert len(g.nodes) == 0
        np.testing.assert_allclose(g.tensors["out"].data, [5, 2])

    def test_fold_sub(self):
        a = np.array([10.0, 5.0], dtype=np.float32)
        b = np.array([3.0, 2.0], dtype=np.float32)
        g = self._make_graph_with_constant_op(OpType.SUB, {"a": a, "b": b})
        g = ConstantFoldingPass().run(g)
        assert len(g.nodes) == 0
        np.testing.assert_allclose(g.tensors["out"].data, [7, 3])

    def test_fold_concat(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        g = self._make_graph_with_constant_op(OpType.CONCAT, {"a": a, "b": b}, {"axis": 0})
        g = ConstantFoldingPass().run(g)
        assert len(g.nodes) == 0
        np.testing.assert_allclose(g.tensors["out"].data, [1, 2, 3, 4])

    def test_fold_squeeze(self):
        x = np.array([[[1.0], [2.0]]], dtype=np.float32)
        axes = np.array([0, 2], dtype=np.int64)
        g = self._make_graph_with_constant_op(OpType.SQUEEZE, {"x": x, "axes": axes})
        g = ConstantFoldingPass().run(g)
        assert len(g.nodes) == 0
        assert g.tensors["out"].data.shape == (2,)

    def test_fold_unsqueeze(self):
        x = np.array([1.0, 2.0], dtype=np.float32)
        axes = np.array([0, 2], dtype=np.int64)
        g = self._make_graph_with_constant_op(OpType.UNSQUEEZE, {"x": x, "axes": axes})
        g = ConstantFoldingPass().run(g)
        assert len(g.nodes) == 0
        assert g.tensors["out"].data.shape == (1, 2, 1)

    def test_fold_cast(self):
        x = np.array([1.5, 2.7], dtype=np.float32)
        g = self._make_graph_with_constant_op(OpType.CAST, {"x": x}, {"to": 7})
        g = ConstantFoldingPass().run(g)
        assert len(g.nodes) == 0
        assert g.tensors["out"].data.dtype == np.int64

    def test_fold_gather(self):
        data = np.arange(12, dtype=np.float32).reshape(4, 3)
        indices = np.array([0, 2], dtype=np.int64)
        g = self._make_graph_with_constant_op(
            OpType.GATHER, {"data": data, "indices": indices}, {"axis": 0}
        )
        g = ConstantFoldingPass().run(g)
        assert len(g.nodes) == 0
        np.testing.assert_allclose(g.tensors["out"].data, data[[0, 2]])

    def test_fold_shape(self):
        x = np.zeros((2, 3, 4), dtype=np.float32)
        g = self._make_graph_with_constant_op(OpType.SHAPE, {"x": x})
        g = ConstantFoldingPass().run(g)
        assert len(g.nodes) == 0
        np.testing.assert_array_equal(g.tensors["out"].data, [2, 3, 4])

    def test_fold_constant_of_shape(self):
        shape_tensor = np.array([3, 4], dtype=np.int64)
        g = self._make_graph_with_constant_op(
            OpType.CONSTANT_OF_SHAPE,
            {"shape": shape_tensor},
            {"value": np.array([1.0], dtype=np.float32)},
        )
        g = ConstantFoldingPass().run(g)
        assert len(g.nodes) == 0
        assert g.tensors["out"].data.shape == (3, 4)

    def test_fold_expand(self):
        x = np.array([[1.0, 2.0]], dtype=np.float32)
        new_shape = np.array([3, 2], dtype=np.int64)
        g = self._make_graph_with_constant_op(OpType.EXPAND, {"x": x, "new_shape": new_shape})
        g = ConstantFoldingPass().run(g)
        assert len(g.nodes) == 0
        assert g.tensors["out"].data.shape == (3, 2)

    def test_fold_tile(self):
        x = np.array([[1.0, 2.0]], dtype=np.float32)
        repeats = np.array([2, 3], dtype=np.int64)
        g = self._make_graph_with_constant_op(OpType.TILE, {"x": x, "repeats": repeats})
        g = ConstantFoldingPass().run(g)
        assert len(g.nodes) == 0
        assert g.tensors["out"].data.shape == (2, 6)

    def test_fold_slice(self):
        x = np.arange(6, dtype=np.float32).reshape(1, 6)
        starts = np.array([1], dtype=np.int64)
        ends = np.array([4], dtype=np.int64)
        axes = np.array([1], dtype=np.int64)
        g = self._make_graph_with_constant_op(
            OpType.SLICE, {"x": x, "starts": starts, "ends": ends, "axes": axes}
        )
        g = ConstantFoldingPass().run(g)
        assert len(g.nodes) == 0
        np.testing.assert_allclose(g.tensors["out"].data, [[1, 2, 3]])


# =========================================================================
# Typed attributes tests
# =========================================================================


class TestTypedAttributes:
    """Test that parser attaches typed attributes to nodes."""

    def test_conv_attrs(self, tmp_path: Path):
        """Conv nodes get ConvAttrs."""
        rng = np.random.RandomState(42)
        x = rng.randn(1, 1, 8, 8).astype(np.float32)
        w = rng.randn(4, 1, 3, 3).astype(np.float32)
        model = _make_single_op_model(
            "Conv",
            [("x", x), ("w", w)],
            "out",
            [1, 4, 6, 6],
            attrs={"kernel_shape": [3, 3], "strides": [1, 1], "pads": [0, 0, 0, 0]},
            initializer_names={"w"},
        )
        path = tmp_path / "model.onnx"
        onnx.save(model, str(path))
        graph = ONNXParser(path).parse()
        conv_node = [n for n in graph.nodes.values() if n.op_type == OpType.CONV][0]
        assert isinstance(conv_node.op_attrs, ConvAttrs)
        assert conv_node.op_attrs.kernel_shape == [3, 3]

    def test_gemm_attrs(self, tmp_path: Path):
        """Gemm nodes get GemmAttrs."""
        a = np.zeros((2, 3), dtype=np.float32)
        b = np.zeros((4, 3), dtype=np.float32)
        model = _make_single_op_model(
            "Gemm",
            [("a", a), ("b", b)],
            "out",
            [2, 4],
            attrs={"transB": 1},
            initializer_names={"b"},
        )
        path = tmp_path / "model.onnx"
        onnx.save(model, str(path))
        graph = ONNXParser(path).parse()
        gemm_node = [n for n in graph.nodes.values() if n.op_type == OpType.GEMM][0]
        assert isinstance(gemm_node.op_attrs, GemmAttrs)
        assert gemm_node.op_attrs.transB == 1

    def test_maxpool_attrs(self, tmp_path: Path):
        """MaxPool nodes get PoolAttrs."""
        x = np.zeros((1, 1, 4, 4), dtype=np.float32)
        model = _make_single_op_model(
            "MaxPool",
            [("x", x)],
            "out",
            [1, 1, 2, 2],
            attrs={"kernel_shape": [2, 2], "strides": [2, 2]},
        )
        path = tmp_path / "model.onnx"
        onnx.save(model, str(path))
        graph = ONNXParser(path).parse()
        pool_node = [n for n in graph.nodes.values() if n.op_type == OpType.MAX_POOL][0]
        assert isinstance(pool_node.op_attrs, PoolAttrs)
        assert pool_node.op_attrs.kernel_shape == [2, 2]

    def test_layer_norm_attrs(self, tmp_path: Path):
        """LayerNormalization nodes get NormAttrs."""
        x = np.zeros((1, 4, 8), dtype=np.float32)
        scale = np.ones(8, dtype=np.float32)
        bias = np.zeros(8, dtype=np.float32)
        model = _make_single_op_model(
            "LayerNormalization",
            [("x", x), ("scale", scale), ("bias", bias)],
            "out",
            [1, 4, 8],
            attrs={"axis": -1, "epsilon": 1e-5},
            initializer_names={"scale", "bias"},
        )
        path = tmp_path / "model.onnx"
        onnx.save(model, str(path))
        graph = ONNXParser(path).parse()
        ln_node = [n for n in graph.nodes.values() if n.op_type == OpType.LAYER_NORM][0]
        assert isinstance(ln_node.op_attrs, NormAttrs)
        assert ln_node.op_attrs.axis == -1


# =========================================================================
# TensorType helper tests
# =========================================================================


class TestTensorTypeHelpers:
    """Test TensorType ndim, batch_size, channels, spatial_shape."""

    def test_ndim(self):
        t = TensorType(shape=(1, 3, 8, 8), dtype=np.dtype("float32"))
        assert t.ndim == 4

    def test_batch_size(self):
        t = TensorType(shape=(2, 3, 8, 8), dtype=np.dtype("float32"))
        assert t.batch_size == 2

    def test_channels(self):
        t = TensorType(shape=(1, 3, 8, 8), dtype=np.dtype("float32"))
        assert t.channels == 3

    def test_spatial_shape(self):
        t = TensorType(shape=(1, 3, 8, 8), dtype=np.dtype("float32"))
        assert t.spatial_shape == (8, 8)

    def test_spatial_shape_3d(self):
        t = TensorType(shape=(1, 3, 16), dtype=np.dtype("float32"))
        assert t.spatial_shape == (16,)

    def test_channels_raises_for_1d(self):
        t = TensorType(shape=(10,), dtype=np.dtype("float32"))
        with pytest.raises(ValueError, match="Cannot get channels"):
            _ = t.channels

    def test_spatial_shape_raises_for_2d(self):
        t = TensorType(shape=(1, 3), dtype=np.dtype("float32"))
        with pytest.raises(ValueError, match="Cannot get spatial_shape"):
            _ = t.spatial_shape


# =========================================================================
# OpType tests
# =========================================================================


class TestOpTypeExpansion:
    """Test new OpType members and is_fused behavior."""

    def test_new_raw_ops_exist(self):
        """All new raw ops are accessible."""
        expected = [
            "CONV",
            "GEMM",
            "SOFTMAX",
            "LAYER_NORM",
            "GATHER",
            "MUL",
            "DIV",
            "SUB",
            "SIGMOID",
            "TANH",
            "CONCAT",
            "SQUEEZE",
            "UNSQUEEZE",
            "CAST",
            "ERF",
            "POW",
            "SQRT",
            "REDUCE_MEAN",
            "REDUCE_SUM",
            "SLICE",
            "SPLIT",
            "WHERE",
            "MAX_POOL",
            "AVERAGE_POOL",
            "GLOBAL_AVERAGE_POOL",
            "PAD",
            "EXPAND",
            "TILE",
            "CONV_TRANSPOSE",
            "CLIP",
            "CONSTANT_OF_SHAPE",
            "SHAPE",
        ]
        for name in expected:
            assert hasattr(OpType, name), f"Missing OpType.{name}"

    def test_new_fused_ops_exist(self):
        """Phase 2 fused placeholder ops exist."""
        for name in [
            "FUSED_CONV",
            "FUSED_CONV_RELU",
            "FUSED_CONV_RELU6",
            "FUSED_ATTENTION",
            "FUSED_LAYER_NORM",
            "FUSED_GELU",
            "FUSED_SILU",
        ]:
            assert hasattr(OpType, name)

    def test_is_fused_for_original(self):
        """Existing fused ops still detected correctly."""
        assert OpType.FUSED_LINEAR.is_fused
        assert OpType.FUSED_LINEAR_RELU.is_fused

    def test_is_fused_for_new(self):
        """New fused ops detected correctly."""
        assert OpType.FUSED_CONV.is_fused
        assert OpType.FUSED_ATTENTION.is_fused

    def test_is_not_fused_for_raw(self):
        """Raw ops not detected as fused."""
        assert not OpType.CONV.is_fused
        assert not OpType.SOFTMAX.is_fused
        assert not OpType.MATMUL.is_fused

    def test_from_onnx_new_ops(self):
        """from_onnx works for all new ops."""
        assert OpType.from_onnx("Conv") == OpType.CONV
        assert OpType.from_onnx("Softmax") == OpType.SOFTMAX
        assert OpType.from_onnx("LayerNormalization") == OpType.LAYER_NORM
        assert OpType.from_onnx("MaxPool") == OpType.MAX_POOL
        assert OpType.from_onnx("GlobalAveragePool") == OpType.GLOBAL_AVERAGE_POOL
        assert OpType.from_onnx("Erf") == OpType.ERF


# =========================================================================
# Integration tests: CNN and Transformer models
# =========================================================================


class TestIntegrationCNN:
    """Integration test: parse and interpret synthetic CNN model."""

    def test_parse_cnn_model(self, cnn_model_path: Path):
        """CNN model parses successfully with all shapes resolved."""
        parser = ONNXParser(cnn_model_path)
        graph = parser.parse()
        assert graph.name == "simple_cnn"
        for name, tensor in graph.tensors.items():
            assert tensor.type.is_shape_known, f"Tensor '{name}' has unknown shape"

    def test_cnn_has_conv_and_pool(self, cnn_model_path: Path):
        """CNN model has Conv and MaxPool nodes."""
        graph = ONNXParser(cnn_model_path).parse()
        op_types = {n.op_type for n in graph.nodes.values()}
        assert OpType.CONV in op_types
        assert OpType.MAX_POOL in op_types
        assert OpType.GLOBAL_AVERAGE_POOL in op_types

    def test_cnn_interpreter_matches_ort(self, cnn_model_path: Path):
        """CNN interpreter output matches ORT."""
        from tests.conftest import build_simple_cnn

        model = build_simple_cnn()
        rng = np.random.RandomState(42)
        x = rng.randn(1, 1, 8, 8).astype(np.float32)

        ort_out = _run_ort(model, {"input": x})

        graph = ONNXParser(cnn_model_path).parse()
        interp = IRInterpreter(graph)
        ir_out = interp.run({"input": x})

        np.testing.assert_allclose(
            ir_out[graph.outputs[0]],
            ort_out["output"],
            rtol=1e-4,
            atol=1e-4,
        )


class TestIntegrationTransformer:
    """Integration test: parse and interpret synthetic transformer model."""

    def test_parse_transformer_model(self, transformer_model_path: Path):
        """Transformer model parses successfully with all shapes resolved."""
        parser = ONNXParser(transformer_model_path)
        graph = parser.parse()
        assert graph.name == "simple_transformer"
        for name, tensor in graph.tensors.items():
            assert tensor.type.is_shape_known, f"Tensor '{name}' has unknown shape"

    def test_transformer_has_expected_ops(self, transformer_model_path: Path):
        """Transformer model has Gather, Softmax, LayerNorm, Erf, Mul nodes."""
        graph = ONNXParser(transformer_model_path).parse()
        op_types = {n.op_type for n in graph.nodes.values()}
        assert OpType.GATHER in op_types
        assert OpType.SOFTMAX in op_types
        assert OpType.LAYER_NORM in op_types
        assert OpType.ERF in op_types
        assert OpType.MUL in op_types

    def test_transformer_interpreter_matches_ort(self, transformer_model_path: Path):
        """Transformer interpreter output matches ORT."""
        from tests.conftest import build_simple_transformer_encoder

        model = build_simple_transformer_encoder()
        input_ids = np.array([[0, 5, 10, 15]], dtype=np.int64)

        ort_out = _run_ort(model, {"input_ids": input_ids})

        graph = ONNXParser(transformer_model_path).parse()
        interp = IRInterpreter(graph)
        ir_out = interp.run({"input_ids": input_ids})

        np.testing.assert_allclose(
            ir_out[graph.outputs[0]],
            ort_out["output"],
            rtol=1e-4,
            atol=1e-4,
        )


# =========================================================================
# Helper graph builders for isolated interpreter tests
# =========================================================================


def _simple_binary_graph(op_type: OpType) -> Graph:
    """Build a trivial a OP b → out graph for testing binary ops."""
    return Graph(
        name="test",
        nodes={"op": OpNode(name="op", op_type=op_type, inputs=["a", "b"], outputs=["out"])},
        tensors={
            "a": Tensor("a", TensorType((1, 3), np.dtype("float32"))),
            "b": Tensor("b", TensorType((1, 3), np.dtype("float32"))),
            "out": Tensor("out", TensorType((1, 3), np.dtype("float32"))),
        },
        inputs=["a", "b"],
        outputs=["out"],
    )


def _simple_unary_graph(op_type: OpType) -> Graph:
    """Build a trivial x → OP → out graph for testing unary ops."""
    return Graph(
        name="test",
        nodes={"op": OpNode(name="op", op_type=op_type, inputs=["x"], outputs=["out"])},
        tensors={
            "x": Tensor("x", TensorType((1, 4), np.dtype("float32"))),
            "out": Tensor("out", TensorType((1, 4), np.dtype("float32"))),
        },
        inputs=["x"],
        outputs=["out"],
    )
