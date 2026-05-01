"""Tests for IR-to-ONNX export."""

from pathlib import Path

import numpy as np
import onnx
import pytest

from mlasic.export import IRExporter
from mlasic.ingestion import ONNXParser
from mlasic.ir import FusedLinearAttrs, Graph, OpNode, OpType, Tensor, TensorType


class TestIRExporter:
    def test_export_produces_valid_onnx(self, ad_model_path: Path):
        """Exported model should pass onnx.checker.check_model()."""
        parser = ONNXParser(ad_model_path)
        graph = parser.parse()

        exporter = IRExporter()
        model = exporter.export(graph)

        # Should not raise
        onnx.checker.check_model(model)

    def test_export_preserves_graph_name(self, ad_model_path: Path):
        parser = ONNXParser(ad_model_path)
        graph = parser.parse()

        exporter = IRExporter()
        model = exporter.export(graph)

        assert model.graph.name == graph.name

    def test_export_preserves_input_output_count(self, ad_model_path: Path):
        parser = ONNXParser(ad_model_path)
        graph = parser.parse()

        exporter = IRExporter()
        model = exporter.export(graph)

        # Count non-initializer inputs (graph activation inputs)
        init_names = {init.name for init in model.graph.initializer}
        model_inputs = [inp for inp in model.graph.input if inp.name not in init_names]
        assert len(model_inputs) == len(graph.inputs)
        assert len(model.graph.output) == len(graph.outputs)

    def test_export_preserves_node_count(self, ad_model_path: Path):
        parser = ONNXParser(ad_model_path)
        graph = parser.parse()

        exporter = IRExporter()
        model = exporter.export(graph)

        assert len(model.graph.node) == len(graph.nodes)

    def test_export_preserves_weight_data(self, ad_model_path: Path):
        """Weight initializer data should be preserved exactly."""
        parser = ONNXParser(ad_model_path)
        graph = parser.parse()

        exporter = IRExporter()
        model = exporter.export(graph)

        exported_inits = {
            init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer
        }

        for tensor in graph.tensors.values():
            if tensor.is_constant and tensor.data is not None:
                assert tensor.name in exported_inits, f"Missing initializer: {tensor.name}"
                np.testing.assert_array_equal(tensor.data, exported_inits[tensor.name])

    def test_export_rejects_fused_ops(self):
        """Fused operators cannot be exported to ONNX."""
        graph = Graph(
            name="fused",
            nodes={
                "fused0": OpNode(
                    name="fused0",
                    op_type=OpType.FUSED_LINEAR_RELU,
                    inputs=["input", "weight", "bias"],
                    outputs=["output"],
                    attributes={
                        "fused_attrs": FusedLinearAttrs(input_dim=4, output_dim=2, has_relu=True)
                    },
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
                "output": Tensor(
                    name="output",
                    type=TensorType(shape=(1, 2), dtype=np.dtype("float32")),
                ),
            },
            inputs=["input"],
            outputs=["output"],
        )

        exporter = IRExporter()
        with pytest.raises(ValueError, match="fused"):
            exporter.export(graph)
