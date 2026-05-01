"""Tests for ONNX ingestion (parser)."""

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from mlasic.exceptions import ParseError, UnsupportedOperatorError
from mlasic.ingestion import ONNXParser
from mlasic.ir import OpType


class TestONNXParser:
    def test_parse_ad_model_structure(self, ad_model_path: Path):
        """Parse synthetic AD model and verify structure."""
        parser = ONNXParser(ad_model_path)
        graph = parser.parse()

        assert graph.name == "ad_model"
        assert graph.stage == "raw"
        assert len(graph.inputs) == 1
        assert len(graph.outputs) == 1

    def test_parse_ad_model_node_count(self, ad_model_path: Path):
        """AD model should have MatMul, Add, BN, ReLU nodes."""
        parser = ONNXParser(ad_model_path)
        graph = parser.parse()

        op_counts: dict[OpType, int] = {}
        for node in graph.nodes.values():
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

        # 4 layers: each has MatMul + Add
        assert op_counts[OpType.MATMUL] == 4
        assert op_counts[OpType.ADD] == 4
        # First 3 layers have BN + ReLU
        assert op_counts[OpType.BATCH_NORM] == 3
        assert op_counts[OpType.RELU] == 3

    def test_parse_ad_model_shapes(self, ad_model_path: Path):
        """All tensor shapes should be resolved after parsing."""
        parser = ONNXParser(ad_model_path)
        graph = parser.parse()

        for name, tensor in graph.tensors.items():
            assert tensor.type.is_shape_known, f"Tensor '{name}' has unknown shape"

    def test_parse_ad_model_input_output_shapes(self, ad_model_path: Path):
        """Input should be [1,640], output should be [1,640]."""
        parser = ONNXParser(ad_model_path)
        graph = parser.parse()

        inp = graph.tensors[graph.inputs[0]]
        assert inp.type.shape == (1, 640)

        out = graph.tensors[graph.outputs[0]]
        assert out.type.shape == (1, 640)

    def test_parse_ad_model_weight_shapes(self, ad_model_path: Path):
        """Verify weight tensor shapes match MLP architecture."""
        parser = ONNXParser(ad_model_path)
        graph = parser.parse()

        expected_weight_shapes = {
            (640, 128),  # Layer 0
            (128, 128),  # Layer 1 and 2
            (128, 640),  # Layer 3
        }

        weight_shapes = set()
        for tensor in graph.tensors.values():
            if tensor.is_constant and len(tensor.type.shape) == 2:
                weight_shapes.add(tensor.type.shape)

        assert expected_weight_shapes.issubset(weight_shapes)

    def test_parse_ad_model_constants_have_data(self, ad_model_path: Path):
        """All constant tensors (weights, biases) should have numpy data."""
        parser = ONNXParser(ad_model_path)
        graph = parser.parse()

        for tensor in graph.tensors.values():
            if tensor.is_constant:
                assert tensor.data is not None
                assert isinstance(tensor.data, np.ndarray)

    def test_unsupported_op_rejection(self, unsupported_model_path: Path):
        """Model with LpNormalization should raise UnsupportedOperatorError."""
        parser = ONNXParser(unsupported_model_path)
        with pytest.raises(UnsupportedOperatorError, match="LpNormalization"):
            parser.parse()

    def test_unsupported_op_lists_all(self, tmp_path: Path):
        """Multiple unsupported ops should all be reported in one error."""
        import onnx
        from onnx import TensorProto, helper

        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])

        nodes = [
            helper.make_node("LpNormalization", ["input"], ["lp_out"], name="lp_0", p=2),
            helper.make_node("Hardmax", ["lp_out"], ["output"], name="hm_0"),
        ]

        graph = helper.make_graph(nodes, "multi_unsupported", [inp], [out])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8

        path = tmp_path / "multi_unsupported.onnx"
        onnx.save(model, str(path))

        parser = ONNXParser(path)
        with pytest.raises(UnsupportedOperatorError) as exc_info:
            parser.parse()

        # Both ops should be reported
        assert len(exc_info.value.unsupported_ops) == 2

    def test_parse_validates_stage1(self, ad_model_path: Path):
        """Parsed graph should pass all stage-1 invariants."""
        parser = ONNXParser(ad_model_path)
        graph = parser.parse()
        # Should not raise
        graph.validate("raw")

    def test_node_naming_sequential(self, tmp_path: Path):
        """Unnamed ONNX nodes should get sequential names like MatMul_0."""
        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])

        w = numpy_helper.from_array(np.ones((4, 2), dtype=np.float32), name="w")

        # Node with empty name
        mm = helper.make_node("MatMul", ["input", "w"], ["output"])
        mm.name = ""  # Force empty name

        graph = helper.make_graph([mm], "unnamed_test", [inp], [out], initializer=[w])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8

        path = tmp_path / "unnamed.onnx"
        onnx.save(model, str(path))

        parser = ONNXParser(path)
        g = parser.parse()

        node_names = list(g.nodes.keys())
        assert len(node_names) == 1
        assert node_names[0] == "MatMul_0"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestIngestionEdgeCases:
    def test_empty_model_rejected(self, tmp_path: Path):
        """ONNX with 0 nodes raises ParseError."""
        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
        out = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])

        graph = helper.make_graph([], "empty", [inp], [out])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8

        path = tmp_path / "empty.onnx"
        onnx.save(model, str(path))

        parser = ONNXParser(path)
        with pytest.raises(ParseError, match="0 nodes"):
            parser.parse()

    def test_node_name_collision_resolved(self, tmp_path: Path):
        """Auto-name conflicts with existing ONNX-named node → both unique."""
        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])

        w1 = numpy_helper.from_array(np.eye(4, dtype=np.float32), name="w1")
        w2 = numpy_helper.from_array(np.eye(4, dtype=np.float32), name="w2")

        # First node: explicitly named "MatMul_0" (collides with auto-name)
        mm1 = helper.make_node("MatMul", ["input", "w1"], ["mid"], name="MatMul_0")
        # Second node: unnamed → would auto-name to "MatMul_0" without fix
        mm2 = helper.make_node("MatMul", ["mid", "w2"], ["output"])
        mm2.name = ""

        vi_mid = helper.make_tensor_value_info("mid", TensorProto.FLOAT, [1, 4])
        graph = helper.make_graph(
            [mm1, mm2],
            "collision_test",
            [inp],
            [out],
            initializer=[w1, w2],
            value_info=[vi_mid],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8

        path = tmp_path / "collision.onnx"
        onnx.save(model, str(path))

        parser = ONNXParser(path)
        g = parser.parse()

        names = list(g.nodes.keys())
        assert len(names) == 2
        assert len(set(names)) == 2  # All unique
        assert "MatMul_0" in names  # The explicitly named one
        # The auto-named one should get "MatMul_1" instead of "MatMul_0"
        assert "MatMul_1" in names

    def test_unsupported_dtype_rejected(self, tmp_path: Path):
        """COMPLEX64 value_info raises ParseError."""
        inp = helper.make_tensor_value_info("input", TensorProto.COMPLEX64, [1, 4])
        out = helper.make_tensor_value_info("output", TensorProto.COMPLEX64, [1, 4])

        w = numpy_helper.from_array(np.eye(4, dtype=np.float32), name="w")
        mm = helper.make_node("MatMul", ["input", "w"], ["output"], name="mm0")

        graph = helper.make_graph([mm], "complex_test", [inp], [out], initializer=[w])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8

        path = tmp_path / "complex.onnx"
        onnx.save(model, str(path))

        parser = ONNXParser(path)
        with pytest.raises(ParseError, match="Unsupported dtype"):
            parser.parse()

    def test_unsupported_initializer_dtype_rejected(self, tmp_path: Path):
        """COMPLEX64 initializer raises ParseError."""
        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])

        # Create COMPLEX64 initializer directly via TensorProto
        w_tensor = helper.make_tensor(
            "w",
            TensorProto.COMPLEX64,
            [4, 4],
            np.eye(4, dtype=np.complex64).flatten().tolist(),
        )
        mm = helper.make_node("MatMul", ["input", "w"], ["output"], name="mm0")

        graph = helper.make_graph(
            [mm],
            "complex_init_test",
            [inp],
            [out],
            initializer=[w_tensor],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8

        path = tmp_path / "complex_init.onnx"
        onnx.save(model, str(path))

        parser = ONNXParser(path)
        with pytest.raises(ParseError, match="Unsupported dtype"):
            parser.parse()
