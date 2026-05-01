"""ONNX model ingestion — parse ONNX protobuf into IR Graph.

Reference: docs/compiler-ir-spec.md Section 4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper, shape_inference
from onnx import helper as onnx_helper

from mlasic.exceptions import ParseError, ShapeInferenceError, UnsupportedOperatorError
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


class ONNXParser:
    """Parse ONNX protobuf into IR Graph."""

    SUPPORTED_OPS = {
        # Original (MLP)
        "MatMul",
        "Add",
        "Relu",
        "BatchNormalization",
        "Reshape",
        "Transpose",
        "Flatten",
        # Tier 1 (CNN/Transformer core)
        "Conv",
        "Gemm",
        "Softmax",
        "LayerNormalization",
        "Gather",
        "Mul",
        "Div",
        "Sub",
        "Sigmoid",
        "Tanh",
        "Concat",
        "Squeeze",
        "Unsqueeze",
        "Cast",
        # Tier 2 (model completeness)
        "Erf",
        "Pow",
        "Sqrt",
        "ReduceMean",
        "ReduceSum",
        "Slice",
        "Split",
        "Where",
        "MaxPool",
        "AveragePool",
        "GlobalAveragePool",
        "Pad",
        "Expand",
        "Tile",
        "ConvTranspose",
        "Clip",
        "ConstantOfShape",
        "Shape",
        "Identity",
        "Constant",
        "Range",
        "Equal",
        "Less",
        "Greater",
        "Not",
        "Neg",
        "Sin",
        "Cos",
        "Trilu",
        "DequantizeLinear",
        "DynamicQuantizeLinear",
        "MatMulInteger",
        "ScatterND",
        "If",
        "Loop",
        # Tier 3 (extended model support)
        "GroupNormalization",
        "InstanceNormalization",
        "QLinearMatMul",
        "Resize",
    }

    SUPPORTED_DTYPES = {
        TensorProto.FLOAT,
        TensorProto.DOUBLE,
        TensorProto.INT32,
        TensorProto.INT64,
        TensorProto.INT8,
        TensorProto.UINT8,
        TensorProto.FLOAT16,
        TensorProto.BOOL,
    }

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise ParseError(f"Model file not found: {self.model_path}")
        self.model = onnx.load(str(self.model_path))
        onnx.checker.check_model(self.model)
        self._node_counter: dict[str, int] = {}
        self._used_names: set[str] = set()

    def parse(self) -> Graph:
        """Convert ONNX model to IR Graph."""
        # Check for unsupported ops first (scan all, report all at once)
        self._check_supported_ops()

        # Reject empty models (0 nodes)
        if len(self.model.graph.node) == 0:
            raise ParseError("Model has 0 nodes — empty models are not supported")

        # Collect ONNX-defined node names for collision avoidance
        for node in self.model.graph.node:
            if node.name:
                self._used_names.add(node.name)

        graph = Graph(
            name=self.model.graph.name or "model",
            nodes={},
            tensors={},
            inputs=[],
            outputs=[],
            stage="raw",
        )

        # Parse initializers (weights/constants)
        for init in self.model.graph.initializer:
            tensor = self._parse_initializer(init)
            graph.tensors[tensor.name] = tensor

        # Parse graph inputs (skip initializers — they appear in both)
        for inp in self.model.graph.input:
            if inp.name not in graph.tensors:
                tensor = self._parse_value_info(inp)
                graph.tensors[tensor.name] = tensor
                graph.inputs.append(tensor.name)

        # Parse graph outputs
        for out in self.model.graph.output:
            tensor = self._parse_value_info(out)
            if tensor.name not in graph.tensors:
                graph.tensors[tensor.name] = tensor
            graph.outputs.append(tensor.name)

        # Parse nodes
        for node in self.model.graph.node:
            op_node = self._parse_node(node)
            self._attach_typed_attrs(op_node)
            graph.nodes[op_node.name] = op_node

            # Create placeholder tensors for intermediate outputs
            for out_name in op_node.outputs:
                if out_name not in graph.tensors:
                    graph.tensors[out_name] = Tensor(
                        name=out_name,
                        type=TensorType(shape=(-1,), dtype=np.float32),
                    )

        # Run shape inference to resolve all tensor shapes
        self._infer_shapes(graph)

        # Build adjacency caches
        graph.invalidate_cache()

        return graph

    def _check_supported_ops(self) -> None:
        """Scan all nodes for unsupported operators, raise single error listing all."""
        unsupported: list[tuple[str, str]] = []
        for node in self.model.graph.node:
            if node.op_type not in self.SUPPORTED_OPS:
                name = node.name or f"unnamed_{node.op_type}"
                unsupported.append((node.op_type, name))
        if unsupported:
            raise UnsupportedOperatorError(unsupported)

    def _parse_initializer(self, init: TensorProto) -> Tensor:
        """Parse ONNX initializer to IR Tensor with numpy data."""
        if init.data_type not in self.SUPPORTED_DTYPES:
            dtype_name = TensorProto.DataType.Name(init.data_type)
            raise ParseError(f"Unsupported dtype '{dtype_name}' for initializer '{init.name}'")
        data = numpy_helper.to_array(init)
        return Tensor(
            name=init.name,
            type=TensorType(shape=tuple(data.shape), dtype=data.dtype),
            data=data,
        )

    def _parse_value_info(self, vi: onnx.ValueInfoProto) -> Tensor:
        """Parse ONNX ValueInfo to IR Tensor (graph inputs/outputs)."""
        elem_type = vi.type.tensor_type.elem_type
        if elem_type not in self.SUPPORTED_DTYPES:
            dtype_name = TensorProto.DataType.Name(elem_type)
            raise ParseError(f"Unsupported dtype '{dtype_name}' for value_info '{vi.name}'")
        shape = tuple(
            dim.dim_value if dim.HasField("dim_value") else -1
            for dim in vi.type.tensor_type.shape.dim
        )
        dtype = onnx_helper.tensor_dtype_to_np_dtype(elem_type)
        return Tensor(name=vi.name, type=TensorType(shape=shape, dtype=dtype))

    def _parse_node(self, node: onnx.NodeProto) -> OpNode:
        """Parse ONNX NodeProto to IR OpNode."""
        # Reject ONNX subgraph ops (If/Loop) — not supported in v0.2
        if node.op_type in ("If", "Loop"):
            raise UnsupportedOperatorError(
                [(node.op_type, node.name or f"unnamed_{node.op_type}")],
                message=(
                    f"ONNX subgraph ops ({node.op_type}) are not supported in v0.2. "
                    "Export model with fixed shapes."
                ),
            )
        op_type = OpType.from_onnx(node.op_type)
        attributes = self._parse_attributes(node)

        # Generate sequential name if ONNX node is unnamed
        name = node.name
        if not name:
            op_str = node.op_type
            count = self._node_counter.get(op_str, 0)
            name = f"{op_str}_{count}"
            self._node_counter[op_str] = count + 1
            # Avoid collisions with existing ONNX-named nodes
            while name in self._used_names:
                count += 1
                name = f"{op_str}_{count}"
                self._node_counter[op_str] = count + 1

        self._used_names.add(name)

        # Filter empty string inputs (BatchNorm optional inputs)
        inputs = [inp for inp in node.input if inp]
        outputs = [out for out in node.output if out]

        return OpNode(
            name=name,
            op_type=op_type,
            inputs=inputs,
            outputs=outputs,
            attributes=attributes,
        )

    def _parse_attributes(self, node: onnx.NodeProto) -> dict:
        """Extract ONNX attributes into Python dict."""
        attributes: dict = {}
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.FLOAT:
                attributes[attr.name] = attr.f
            elif attr.type == onnx.AttributeProto.INT:
                attributes[attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.STRING:
                attributes[attr.name] = attr.s.decode("utf-8")
            elif attr.type == onnx.AttributeProto.INTS:
                attributes[attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOATS:
                attributes[attr.name] = list(attr.floats)
            elif attr.type == onnx.AttributeProto.TENSOR:
                attributes[attr.name] = numpy_helper.to_array(attr.t)
            elif attr.type == onnx.AttributeProto.GRAPH:
                # Store subgraph as raw ONNX GraphProto (used by If op)
                attributes[attr.name] = attr.g
        return attributes

    def _attach_typed_attrs(self, node: OpNode) -> None:
        """Create typed attribute dataclasses from raw ONNX attributes."""
        a = node.attributes
        if node.op_type in (OpType.CONV, OpType.CONV_TRANSPOSE):
            node.op_attrs = ConvAttrs(
                kernel_shape=a.get("kernel_shape", [1, 1]),
                strides=a.get("strides", [1, 1]),
                pads=a.get("pads", [0, 0, 0, 0]),
                dilations=a.get("dilations", [1, 1]),
                group=a.get("group", 1),
            )
        elif node.op_type == OpType.GEMM:
            node.op_attrs = GemmAttrs(
                alpha=a.get("alpha", 1.0),
                beta=a.get("beta", 1.0),
                transA=a.get("transA", 0),
                transB=a.get("transB", 0),
            )
        elif node.op_type in (OpType.MAX_POOL, OpType.AVERAGE_POOL):
            node.op_attrs = PoolAttrs(
                kernel_shape=a.get("kernel_shape", [1, 1]),
                strides=a.get("strides", [1, 1]),
                pads=a.get("pads", [0, 0, 0, 0]),
                ceil_mode=a.get("ceil_mode", 0),
            )
        elif node.op_type == OpType.LAYER_NORM:
            node.op_attrs = NormAttrs(
                axis=a.get("axis", -1),
                epsilon=a.get("epsilon", 1e-5),
            )
        elif node.op_type == OpType.GROUP_NORM:
            node.op_attrs = NormAttrs(
                axis=-1,
                epsilon=a.get("epsilon", 1e-5),
                num_groups=a.get("num_groups", 1),
            )
        elif node.op_type == OpType.INSTANCE_NORM:
            # InstanceNorm = GroupNorm with num_groups = number of channels
            # Channels inferred from scale input at shape inference time
            node.op_attrs = NormAttrs(
                axis=-1,
                epsilon=a.get("epsilon", 1e-5),
                num_groups=0,  # 0 = placeholder, means num_groups = channels
            )

    def _infer_shapes(self, graph: Graph) -> None:
        """Infer shapes for all intermediate tensors using ONNX shape inference."""
        inferred = shape_inference.infer_shapes(self.model)

        # Track which tensors ONNX shape inference touched
        inferred_names: set[str] = set()

        # Propagate inferred shapes to IR tensors
        for vi in inferred.graph.value_info:
            if vi.name in graph.tensors:
                shape = tuple(
                    dim.dim_value if dim.HasField("dim_value") else -1
                    for dim in vi.type.tensor_type.shape.dim
                )
                dtype = onnx_helper.tensor_dtype_to_np_dtype(vi.type.tensor_type.elem_type)
                graph.tensors[vi.name].type.shape = shape
                graph.tensors[vi.name].type.dtype = dtype
                inferred_names.add(vi.name)

        # Also propagate from graph inputs/outputs (may have dynamic dims)
        for inp_vi in inferred.graph.input:
            if inp_vi.name in graph.tensors:
                inferred_names.add(inp_vi.name)
        for out_vi in inferred.graph.output:
            if out_vi.name in graph.tensors:
                shape = tuple(
                    dim.dim_value if dim.HasField("dim_value") else -1
                    for dim in out_vi.type.tensor_type.shape.dim
                )
                dtype = onnx_helper.tensor_dtype_to_np_dtype(out_vi.type.tensor_type.elem_type)
                graph.tensors[out_vi.name].type.shape = shape
                graph.tensors[out_vi.name].type.dtype = dtype
                inferred_names.add(out_vi.name)

        # Initializers are already shape-known from parse
        for init in self.model.graph.initializer:
            inferred_names.add(init.name)

        # Only flag tensors that ONNX shape inference didn't touch at all
        unresolved = []
        for name, t in graph.tensors.items():
            if t.type.is_shape_known:
                continue
            if name in inferred_names:
                # ONNX inference resolved this but with dynamic dims — acceptable
                continue
            unresolved.append(name)
        if unresolved:
            raise ShapeInferenceError(unresolved)
