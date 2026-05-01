"""IR-to-ONNX exporter for round-trip testing.

Converts an IR Graph back to an ONNX ModelProto so we can verify
that parsing preserves model semantics via ONNX Runtime comparison.
"""

from __future__ import annotations

import onnx
from onnx import TensorProto, helper, numpy_helper

from mlasic.ir import Graph, OpType, Tensor

# Reverse mapping: OpType -> ONNX op_type string
_OPTYPE_TO_ONNX: dict[OpType, str] = {
    # Original (MLP)
    OpType.MATMUL: "MatMul",
    OpType.ADD: "Add",
    OpType.RELU: "Relu",
    OpType.BATCH_NORM: "BatchNormalization",
    OpType.RESHAPE: "Reshape",
    OpType.TRANSPOSE: "Transpose",
    OpType.FLATTEN: "Flatten",
    # Tier 1 (CNN/Transformer core)
    OpType.CONV: "Conv",
    OpType.GEMM: "Gemm",
    OpType.SOFTMAX: "Softmax",
    OpType.LAYER_NORM: "LayerNormalization",
    OpType.GATHER: "Gather",
    OpType.MUL: "Mul",
    OpType.DIV: "Div",
    OpType.SUB: "Sub",
    OpType.SIGMOID: "Sigmoid",
    OpType.TANH: "Tanh",
    OpType.CONCAT: "Concat",
    OpType.SQUEEZE: "Squeeze",
    OpType.UNSQUEEZE: "Unsqueeze",
    OpType.CAST: "Cast",
    # Tier 2 (model completeness)
    OpType.ERF: "Erf",
    OpType.POW: "Pow",
    OpType.SQRT: "Sqrt",
    OpType.REDUCE_MEAN: "ReduceMean",
    OpType.REDUCE_SUM: "ReduceSum",
    OpType.SLICE: "Slice",
    OpType.SPLIT: "Split",
    OpType.WHERE: "Where",
    OpType.MAX_POOL: "MaxPool",
    OpType.AVERAGE_POOL: "AveragePool",
    OpType.GLOBAL_AVERAGE_POOL: "GlobalAveragePool",
    OpType.PAD: "Pad",
    OpType.EXPAND: "Expand",
    OpType.TILE: "Tile",
    OpType.CONV_TRANSPOSE: "ConvTranspose",
    OpType.CLIP: "Clip",
    OpType.CONSTANT_OF_SHAPE: "ConstantOfShape",
    OpType.SHAPE: "Shape",
}


class IRExporter:
    """Export IR Graph to ONNX ModelProto."""

    def export(self, graph: Graph, opset_version: int = 17) -> onnx.ModelProto:
        """Convert IR Graph back to ONNX protobuf.

        Only raw (unfused) graphs can be exported. Fused operators have no
        ONNX equivalent.

        Args:
            graph: IR Graph to export.
            opset_version: ONNX opset version.

        Returns:
            Valid ONNX ModelProto.

        Raises:
            ValueError: If graph contains fused operators.
        """
        # Reject fused ops
        for node in graph.nodes.values():
            if node.op_type.is_fused:
                raise ValueError(
                    f"Cannot export fused operator '{node.op_type.value}' "
                    f"at node '{node.name}' to ONNX. Only raw graphs can be exported."
                )

        # Build ONNX components
        initializers = self._build_initializers(graph)
        nodes = self._build_nodes(graph)
        inputs = self._build_inputs(graph)
        outputs = self._build_outputs(graph)

        # Create ONNX graph
        onnx_graph = helper.make_graph(
            nodes=nodes,
            name=graph.name,
            inputs=inputs,
            outputs=outputs,
            initializer=initializers,
        )

        # Create model
        opset = helper.make_opsetid("", opset_version)
        model = helper.make_model(onnx_graph, opset_imports=[opset])
        model.ir_version = 8

        # Validate
        onnx.checker.check_model(model)

        return model

    def _build_initializers(self, graph: Graph) -> list[TensorProto]:
        """Convert constant tensors to ONNX initializers."""
        initializers = []
        for tensor in graph.tensors.values():
            if tensor.is_constant and tensor.data is not None:
                tp = numpy_helper.from_array(tensor.data, name=tensor.name)
                initializers.append(tp)
        return initializers

    def _build_nodes(self, graph: Graph) -> list[onnx.NodeProto]:
        """Convert IR OpNodes to ONNX NodeProtos in topological order."""
        nodes = []
        for node_name in graph.topological_order():
            node = graph.nodes[node_name]
            onnx_op = _OPTYPE_TO_ONNX[node.op_type]

            # Build ONNX attributes (exclude internal attributes like 'fused_attrs')
            attrs: dict = {}
            for key, val in node.attributes.items():
                if key in ("fused_attrs", "op_attrs"):
                    continue
                if isinstance(val, list) and val and isinstance(val[0], int):
                    attrs[key] = val
                elif isinstance(val, list) and val and isinstance(val[0], float):
                    attrs[key] = val
                elif isinstance(val, (int, float, str)):
                    attrs[key] = val

            onnx_node = helper.make_node(
                onnx_op,
                inputs=node.inputs,
                outputs=node.outputs,
                name=node.name,
                **attrs,
            )
            nodes.append(onnx_node)
        return nodes

    def _build_inputs(self, graph: Graph) -> list[onnx.ValueInfoProto]:
        """Build ONNX ValueInfoProto for graph inputs.

        In modern ONNX (opset >= 13), initializers should NOT appear in
        graph inputs — only activation inputs belong here. Including
        initializers in inputs causes ONNX Runtime to treat them as
        overridable feed values, preventing constant folding.
        """
        inputs = []
        for inp_name in graph.inputs:
            tensor = graph.tensors[inp_name]
            vi = self._tensor_to_value_info(tensor)
            inputs.append(vi)
        return inputs

    def _build_outputs(self, graph: Graph) -> list[onnx.ValueInfoProto]:
        """Build ONNX ValueInfoProto for graph outputs."""
        outputs = []
        for out_name in graph.outputs:
            tensor = graph.tensors[out_name]
            vi = self._tensor_to_value_info(tensor)
            outputs.append(vi)
        return outputs

    def _tensor_to_value_info(self, tensor: Tensor) -> onnx.ValueInfoProto:
        """Convert IR Tensor to ONNX ValueInfoProto."""
        elem_type = helper.np_dtype_to_tensor_dtype(tensor.type.dtype)
        return helper.make_tensor_value_info(tensor.name, elem_type, list(tensor.type.shape))
