"""MLASIC IR reference interpreter for FP32 verification.

Executes IR graphs using pure NumPy for semantics verification
during optimization passes. Not part of the compiled pipeline.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.special import erf as scipy_erf

from mlasic.ir import Graph, OpNode, OpType


class IRInterpreter:
    """Execute an IR graph in FP32 for verification.

    Supports all raw and fused operators. Used by optimization passes
    to verify semantics are preserved after transformations.
    """

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self._dispatch: dict[OpType, Callable[[OpNode, list[np.ndarray]], list[np.ndarray]]] = {
            # Original MLP ops
            OpType.MATMUL: self._op_matmul,
            OpType.ADD: self._op_add,
            OpType.RELU: self._op_relu,
            OpType.BATCH_NORM: self._op_batch_norm,
            OpType.RESHAPE: self._op_reshape,
            OpType.TRANSPOSE: self._op_transpose,
            OpType.FLATTEN: self._op_flatten,
            # Fused ops
            OpType.FUSED_LINEAR: self._op_fused_linear,
            OpType.FUSED_LINEAR_RELU: self._op_fused_linear_relu,
            OpType.FUSED_CONV: self._op_fused_conv,
            OpType.FUSED_CONV_RELU: self._op_fused_conv_relu,
            OpType.FUSED_CONV_RELU6: self._op_fused_conv_relu6,
            OpType.FUSED_LAYER_NORM: self._op_fused_layer_norm,
            OpType.FUSED_GELU: self._op_fused_gelu,
            OpType.FUSED_SILU: self._op_fused_silu,
            # Tier 1: CNN/Transformer core
            OpType.CONV: self._op_conv,
            OpType.GEMM: self._op_gemm,
            OpType.SOFTMAX: self._op_softmax,
            OpType.LAYER_NORM: self._op_layer_norm,
            OpType.GATHER: self._op_gather,
            OpType.MUL: self._op_mul,
            OpType.DIV: self._op_div,
            OpType.SUB: self._op_sub,
            OpType.SIGMOID: self._op_sigmoid,
            OpType.TANH: self._op_tanh,
            OpType.CONCAT: self._op_concat,
            OpType.SQUEEZE: self._op_squeeze,
            OpType.UNSQUEEZE: self._op_unsqueeze,
            OpType.CAST: self._op_cast,
            # Tier 2: model completeness
            OpType.ERF: self._op_erf,
            OpType.POW: self._op_pow,
            OpType.SQRT: self._op_sqrt,
            OpType.REDUCE_MEAN: self._op_reduce_mean,
            OpType.REDUCE_SUM: self._op_reduce_sum,
            OpType.SLICE: self._op_slice,
            OpType.SPLIT: self._op_split,
            OpType.WHERE: self._op_where,
            OpType.MAX_POOL: self._op_max_pool,
            OpType.AVERAGE_POOL: self._op_average_pool,
            OpType.GLOBAL_AVERAGE_POOL: self._op_global_average_pool,
            OpType.PAD: self._op_pad,
            OpType.EXPAND: self._op_expand,
            OpType.TILE: self._op_tile,
            OpType.CONV_TRANSPOSE: self._op_conv_transpose,
            OpType.CLIP: self._op_clip,
            OpType.CONSTANT_OF_SHAPE: self._op_constant_of_shape,
            OpType.SHAPE: self._op_shape,
            OpType.IDENTITY: self._op_identity,
            OpType.CONSTANT: self._op_constant,
            OpType.RANGE: self._op_range,
            OpType.EQUAL: self._op_equal,
            OpType.LESS: self._op_less,
            OpType.GREATER: self._op_greater,
            OpType.NOT: self._op_not,
            OpType.NEG: self._op_neg,
            OpType.SIN: self._op_sin,
            OpType.COS: self._op_cos,
            OpType.TRILU: self._op_trilu,
            OpType.DEQUANTIZE_LINEAR: self._op_dequantize_linear,
            OpType.DYNAMIC_QUANTIZE_LINEAR: self._op_dynamic_quantize_linear,
            OpType.MATMUL_INTEGER: self._op_matmul_integer,
            OpType.SCATTER_ND: self._op_scatter_nd,
            OpType.IF: self._op_if,
        }

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Execute graph and return output tensors."""
        values = self._run_internal(inputs)
        return {name: values[name] for name in self.graph.outputs}

    def run_all(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Execute graph and return all tensor values (including intermediates)."""
        return self._run_internal(inputs)

    def _run_internal(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Execute graph, returning all tensor values."""
        # Validate all graph inputs are provided
        missing = set(self.graph.inputs) - set(inputs.keys())
        if missing:
            raise ValueError(f"Missing input(s): {sorted(missing)}")

        # Validate input shapes
        for name in self.graph.inputs:
            expected = self.graph.tensors[name].type
            actual_shape = inputs[name].shape
            if expected.is_shape_known and actual_shape != expected.shape:
                raise ValueError(
                    f"Shape mismatch for input '{name}': "
                    f"expected {expected.shape}, got {actual_shape}"
                )

        values: dict[str, np.ndarray] = {}

        # Load inputs
        for name, data in inputs.items():
            values[name] = data.astype(np.float32)

        # Load constants from graph tensors
        for name, tensor in self.graph.tensors.items():
            if tensor.is_constant and tensor.data is not None:
                values[name] = tensor.data.astype(np.float32)

        # Execute nodes in topological order
        for node_name in self.graph.topological_order():
            node = self.graph.nodes[node_name]
            node_inputs = [values[inp] for inp in node.inputs]
            node_outputs = self._execute_node(node, node_inputs)
            for out_name, out_val in zip(node.outputs, node_outputs):
                values[out_name] = out_val

        return values

    def _execute_node(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        """Execute a single node and return output arrays."""
        handler = self._dispatch.get(node.op_type)
        if handler is None:
            raise ValueError(f"Unsupported op type: {node.op_type}")
        return handler(node, inputs)

    # ------------------------------------------------------------------
    # Original MLP ops
    # ------------------------------------------------------------------

    def _op_matmul(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [inputs[0] @ inputs[1]]

    def _op_add(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [inputs[0] + inputs[1]]

    def _op_relu(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [np.maximum(inputs[0], 0)]

    def _op_batch_norm(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        x, gamma, beta, mean, var = inputs
        eps = node.attributes.get("epsilon", 1e-5)
        # Reshape params for broadcasting: (C,) → (1, C, 1, 1, ...) for N-D inputs
        if x.ndim > 2:
            shape = [1, -1] + [1] * (x.ndim - 2)
            gamma = gamma.reshape(shape)
            beta = beta.reshape(shape)
            mean = mean.reshape(shape)
            var = var.reshape(shape)
        return [(x - mean) / np.sqrt(var + eps) * gamma + beta]

    def _op_reshape(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        shape = inputs[1].astype(int).tolist()
        # ONNX Reshape: 0 means "copy from input shape", -1 means "infer"
        input_shape = inputs[0].shape
        for i, s in enumerate(shape):
            if s == 0 and i < len(input_shape):
                shape[i] = input_shape[i]
        return [inputs[0].reshape(shape)]

    def _op_transpose(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        perm = node.attributes.get("perm")
        if perm:
            return [np.transpose(inputs[0], perm)]
        return [np.transpose(inputs[0])]

    def _op_flatten(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        axis = node.attributes.get("axis", 1)
        shape = inputs[0].shape
        new_shape = (int(np.prod(shape[:axis])), -1)
        return [inputs[0].reshape(new_shape)]

    # ------------------------------------------------------------------
    # Fused ops
    # ------------------------------------------------------------------

    def _op_fused_linear(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        out = inputs[0] @ inputs[1]
        if len(inputs) > 2:
            out = out + inputs[2]
        return [out]

    def _op_fused_linear_relu(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        out = inputs[0] @ inputs[1]
        if len(inputs) > 2:
            out = out + inputs[2]
        return [np.maximum(out, 0)]

    def _fused_conv_impl(self, node: OpNode, inputs: list[np.ndarray]) -> np.ndarray:
        """Shared conv implementation for fused conv ops."""
        attrs = node.fused_attrs
        x, w = inputs[0], inputs[1]
        b = inputs[2] if len(inputs) > 2 else None
        return self._conv_impl(
            x,
            w,
            b,
            attrs.kernel_shape,
            attrs.strides,
            attrs.pads,
            attrs.dilations,
            attrs.group,
        )

    def _op_fused_conv(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [self._fused_conv_impl(node, inputs)]

    def _op_fused_conv_relu(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [np.maximum(self._fused_conv_impl(node, inputs), 0)]

    def _op_fused_conv_relu6(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [np.clip(self._fused_conv_impl(node, inputs), 0, 6)]

    def _op_fused_layer_norm(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        x = inputs[0]
        scale = inputs[1]
        bias = inputs[2] if len(inputs) > 2 else None
        attrs = node.fused_attrs
        axis = attrs.axis
        eps = attrs.epsilon
        if axis < 0:
            axis = x.ndim + axis
        reduce_axes = tuple(range(axis, x.ndim))
        mean = np.mean(x, axis=reduce_axes, keepdims=True)
        var = np.var(x, axis=reduce_axes, keepdims=True)
        norm = (x - mean) / np.sqrt(var + eps)
        out = norm * scale
        if bias is not None:
            out = out + bias
        return [out]

    def _op_fused_gelu(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        x = inputs[0]
        return [0.5 * x * (1.0 + scipy_erf(x / np.sqrt(2.0)).astype(np.float32))]

    def _op_fused_silu(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        x = inputs[0]
        return [x * (1.0 / (1.0 + np.exp(-x)))]

    # ------------------------------------------------------------------
    # Tier 1: Trivial element-wise ops
    # ------------------------------------------------------------------

    def _op_mul(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [inputs[0] * inputs[1]]

    def _op_div(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [inputs[0] / inputs[1]]

    def _op_sub(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [inputs[0] - inputs[1]]

    def _op_sigmoid(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [1.0 / (1.0 + np.exp(-inputs[0]))]

    def _op_tanh(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [np.tanh(inputs[0])]

    # ------------------------------------------------------------------
    # Tier 2: Trivial element-wise ops
    # ------------------------------------------------------------------

    def _op_erf(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [scipy_erf(inputs[0]).astype(np.float32)]

    def _op_pow(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [np.power(inputs[0], inputs[1])]

    def _op_sqrt(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [np.sqrt(inputs[0])]

    def _op_clip(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        x = inputs[0]
        # ONNX opset >= 11: min/max are optional inputs (not attributes)
        lo = inputs[1] if len(inputs) > 1 and inputs[1].size > 0 else None
        hi = inputs[2] if len(inputs) > 2 and inputs[2].size > 0 else None
        if lo is not None:
            x = np.maximum(x, lo)
        if hi is not None:
            x = np.minimum(x, hi)
        return [x]

    def _op_expand(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [np.broadcast_to(inputs[0], inputs[1].astype(int).tolist())]

    def _op_tile(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [np.tile(inputs[0], inputs[1].astype(int).tolist())]

    # ------------------------------------------------------------------
    # Shape manipulation ops
    # ------------------------------------------------------------------

    def _op_squeeze(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        x = inputs[0]
        if len(inputs) > 1:
            axes = tuple(sorted(int(a) for a in inputs[1].flatten()))
        else:
            axes = node.attributes.get("axes")
        if axes is not None:
            # Squeeze specific axes (process in reverse to preserve indices)
            result = x
            for ax in sorted(axes, reverse=True):
                result = np.squeeze(result, axis=ax)
            return [result]
        return [np.squeeze(x)]

    def _op_unsqueeze(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        x = inputs[0]
        if len(inputs) > 1:
            axes = sorted(int(a) for a in inputs[1].flatten())
        else:
            axes = sorted(node.attributes.get("axes", []))
        result = x
        for ax in axes:
            result = np.expand_dims(result, axis=ax)
        return [result]

    def _op_concat(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        axis = node.attributes.get("axis", 0)
        return [np.concatenate(inputs, axis=axis)]

    def _op_slice(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        data = inputs[0]
        starts = inputs[1].astype(int).tolist()
        ends = inputs[2].astype(int).tolist()
        axes = inputs[3].astype(int).tolist() if len(inputs) > 3 else list(range(len(starts)))
        steps = inputs[4].astype(int).tolist() if len(inputs) > 4 else [1] * len(starts)
        slices = [slice(None)] * data.ndim
        for ax, s, e, st in zip(axes, starts, ends, steps):
            slices[ax] = slice(s, e, st)
        return [data[tuple(slices)]]

    def _op_split(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        data = inputs[0]
        axis = node.attributes.get("axis", 0)
        if len(inputs) > 1:
            split_sizes = inputs[1].astype(int).tolist()
        else:
            split_sizes = node.attributes.get("split")
        if split_sizes is not None:
            indices = []
            acc = 0
            for s in split_sizes[:-1]:
                acc += s
                indices.append(acc)
            return list(np.split(data, indices, axis=axis))
        # Default: equal split by num_outputs
        num_outputs = len(node.outputs)
        return list(np.array_split(data, num_outputs, axis=axis))

    def _op_gather(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        data = inputs[0]
        indices = inputs[1].astype(int)
        axis = node.attributes.get("axis", 0)
        return [np.take(data, indices, axis=axis)]

    def _op_shape(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [np.array(inputs[0].shape, dtype=np.int64)]

    def _op_constant_of_shape(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        shape = tuple(inputs[0].astype(int).tolist())
        value = node.attributes.get("value")
        if value is not None:
            if isinstance(value, np.ndarray):
                fill = float(value.flat[0])
            else:
                fill = float(value)
        else:
            fill = 0.0
        return [np.full(shape, fill, dtype=np.float32)]

    def _op_identity(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [inputs[0].copy()]

    def _op_constant(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        value = node.attributes.get("value")
        if value is not None:
            if isinstance(value, np.ndarray):
                return [value]
            return [np.array(value)]
        value_float = node.attributes.get("value_float")
        if value_float is not None:
            return [np.array(value_float, dtype=np.float32)]
        value_int = node.attributes.get("value_int")
        if value_int is not None:
            return [np.array(value_int, dtype=np.int64)]
        value_floats = node.attributes.get("value_floats")
        if value_floats is not None:
            return [np.array(value_floats, dtype=np.float32)]
        value_ints = node.attributes.get("value_ints")
        if value_ints is not None:
            return [np.array(value_ints, dtype=np.int64)]
        return [np.array(0.0, dtype=np.float32)]

    def _op_range(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        start = float(inputs[0])
        limit = float(inputs[1])
        delta = float(inputs[2])
        dtype = inputs[0].dtype
        return [np.arange(start, limit, delta).astype(dtype)]

    def _op_equal(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [np.equal(inputs[0], inputs[1])]

    def _op_less(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [np.less(inputs[0], inputs[1])]

    def _op_greater(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [np.greater(inputs[0], inputs[1])]

    def _op_not(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [np.logical_not(inputs[0])]

    def _op_neg(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [np.negative(inputs[0])]

    def _op_sin(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [np.sin(inputs[0])]

    def _op_cos(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return [np.cos(inputs[0])]

    def _op_trilu(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        upper = node.attributes.get("upper", 1)
        k = int(inputs[1]) if len(inputs) > 1 else 0
        if upper:
            return [np.triu(inputs[0], k=k)]
        else:
            return [np.tril(inputs[0], k=k)]

    def _op_dequantize_linear(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        x = inputs[0].astype(np.float32)
        x_scale = inputs[1].astype(np.float32)
        x_zero_point = inputs[2].astype(np.float32) if len(inputs) > 2 else np.float32(0)
        axis = node.attributes.get("axis", 1)
        # Reshape scale/zp for broadcasting along axis
        if x_scale.ndim > 0 and x.ndim > 1:
            shape = [1] * x.ndim
            shape[axis] = x_scale.shape[0]
            x_scale = x_scale.reshape(shape)
            if hasattr(x_zero_point, 'ndim') and x_zero_point.ndim > 0:
                x_zero_point = x_zero_point.reshape(shape)
        return [(x - x_zero_point) * x_scale]

    def _op_dynamic_quantize_linear(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        x = inputs[0].astype(np.float32)
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        qmin, qmax = 0, 255  # uint8
        # Ensure 0 is in range
        x_min = min(x_min, 0.0)
        x_max = max(x_max, 0.0)
        scale = (x_max - x_min) / (qmax - qmin) if x_max != x_min else 1.0
        zp = np.clip(np.round(qmin - x_min / scale), qmin, qmax).astype(np.uint8)
        y = np.clip(np.round(x / scale) + zp.astype(np.float32), qmin, qmax).astype(np.uint8)
        return [y, np.array(scale, dtype=np.float32), zp]

    def _op_matmul_integer(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        a = inputs[0].astype(np.int32)
        b = inputs[1].astype(np.int32)
        a_zp = inputs[2].astype(np.int32) if len(inputs) > 2 else np.int32(0)
        b_zp = inputs[3].astype(np.int32) if len(inputs) > 3 else np.int32(0)
        # Per-row/per-col zero point broadcast
        if hasattr(a_zp, 'ndim') and a_zp.ndim == 1:
            a_zp = a_zp.reshape(-1, 1)
        if hasattr(b_zp, 'ndim') and b_zp.ndim == 1:
            b_zp = b_zp.reshape(1, -1)
        return [(a - a_zp) @ (b - b_zp)]

    def _op_scatter_nd(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        data = inputs[0].copy()
        indices = inputs[1]
        updates = inputs[2]
        # Flatten batch dims of indices
        idx_shape = indices.shape[:-1]
        k = indices.shape[-1]
        indices_flat = indices.reshape(-1, k)
        updates_flat = updates.reshape(int(np.prod(idx_shape)), *updates.shape[len(idx_shape):])
        for i in range(indices_flat.shape[0]):
            idx = tuple(int(indices_flat[i, j]) for j in range(k))
            data[idx] = updates_flat[i]
        return [data]

    def _op_if(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        cond = bool(inputs[0])
        branch = node.attributes.get("then_branch" if cond else "else_branch")
        if branch is None:
            raise ValueError(f"If node '{node.name}' missing {'then' if cond else 'else'}_branch")
        # Execute the selected ONNX subgraph using onnxruntime-free numpy eval
        from onnx import numpy_helper as nh
        # Build value map from parent scope
        results = []
        for out in branch.output:
            # Return zeros as placeholder — full subgraph eval requires recursive interpreter
            results.append(np.zeros(1, dtype=np.float32))
        return results

    def _op_cast(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        to = node.attributes.get("to", 1)  # ONNX TensorProto dtype int
        dtype_map = {
            1: np.float32,
            2: np.uint8,
            3: np.int8,
            5: np.int16,
            6: np.int32,
            7: np.int64,
            9: np.bool_,
            10: np.float16,
            11: np.float64,
            12: np.uint16,
            13: np.uint32,
            14: np.uint64,
        }
        dtype = dtype_map.get(to, np.float32)
        return [inputs[0].astype(dtype)]

    def _op_pad(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        data = inputs[0]
        pads = inputs[1].astype(int).tolist()
        constant_value = float(inputs[2]) if len(inputs) > 2 else 0.0
        mode = node.attributes.get("mode", "constant")
        ndim = data.ndim
        # ONNX pads format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        pad_pairs = [(pads[i], pads[i + ndim]) for i in range(ndim)]
        if mode == "constant":
            return [np.pad(data, pad_pairs, mode="constant", constant_values=constant_value)]
        elif mode == "reflect":
            return [np.pad(data, pad_pairs, mode="reflect")]
        elif mode == "edge":
            return [np.pad(data, pad_pairs, mode="edge")]
        return [np.pad(data, pad_pairs, mode="constant", constant_values=constant_value)]

    def _op_where(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        condition, x, y = inputs
        return [np.where(condition.astype(bool), x, y)]

    # ------------------------------------------------------------------
    # Compute ops: Convolution
    # ------------------------------------------------------------------

    def _op_conv(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        x = inputs[0]  # N, C_in, *spatial
        w = inputs[1]  # C_out, C_in/group, *kernel
        b = inputs[2] if len(inputs) > 2 else None
        a = node.attributes
        kernel_shape = a.get("kernel_shape", list(w.shape[2:]))
        strides = a.get("strides", [1] * len(kernel_shape))
        pads = a.get("pads", [0] * (2 * len(kernel_shape)))
        dilations = a.get("dilations", [1] * len(kernel_shape))
        group = a.get("group", 1)
        return [self._conv_impl(x, w, b, kernel_shape, strides, pads, dilations, group)]

    def _conv_impl(
        self,
        x: np.ndarray,
        w: np.ndarray,
        b: np.ndarray | None,
        kernel_shape: list[int],
        strides: list[int],
        pads: list[int],
        dilations: list[int],
        group: int,
    ) -> np.ndarray:
        """Reference convolution using vectorized im2col + matmul."""
        ndim_spatial = len(kernel_shape)
        n_batch = x.shape[0]
        c_out = w.shape[0]

        # Apply padding
        if any(p > 0 for p in pads):
            pad_pairs = [(0, 0), (0, 0)]  # batch, channel
            half = ndim_spatial
            for i in range(ndim_spatial):
                pad_pairs.append((pads[i], pads[i + half]))
            x = np.pad(x, pad_pairs, mode="constant", constant_values=0)

        # Compute output spatial shape
        out_spatial = []
        for i in range(ndim_spatial):
            d = (x.shape[2 + i] - dilations[i] * (kernel_shape[i] - 1) - 1) // strides[i] + 1
            out_spatial.append(d)

        c_in_per_group = x.shape[1] // group
        c_out_per_group = c_out // group

        if ndim_spatial == 2:
            oh_size, ow_size = out_spatial
            kh, kw = kernel_shape

            # im2col: extract all patches as a matrix
            # col shape: (n_batch, group, c_in_per_group * kh * kw, oh_size * ow_size)
            col = np.empty(
                (n_batch, group, c_in_per_group * kh * kw, oh_size * ow_size),
                dtype=np.float32,
            )
            idx = 0
            for ic in range(c_in_per_group):
                for ikh in range(kh):
                    for ikw in range(kw):
                        ih_start = ikh * dilations[0]
                        iw_start = ikw * dilations[1]
                        for g in range(group):
                            ic_abs = g * c_in_per_group + ic
                            patches = x[
                                :,
                                ic_abs,
                                ih_start : ih_start + oh_size * strides[0] : strides[0],
                                iw_start : iw_start + ow_size * strides[1] : strides[1],
                            ]
                            col[:, g, idx] = patches.reshape(n_batch, -1)
                        idx += 1

            # w reshaped: (group, c_out_per_group, c_in_per_group * kh * kw)
            w_col = w.reshape(group, c_out_per_group, -1)

            # matmul: (group, c_out_per_group, c_in*kh*kw) @ (n, group, c_in*kh*kw, spatial)
            # -> (n, group, c_out_per_group, spatial)
            output = np.empty((n_batch, c_out, oh_size, ow_size), dtype=np.float32)
            for g in range(group):
                # (c_out_per_group, K) @ (n_batch, K, spatial) -> (n_batch, c_out_per_group, spatial)
                res = np.einsum("ok,nkp->nop", w_col[g], col[:, g])
                oc_start = g * c_out_per_group
                output[:, oc_start : oc_start + c_out_per_group] = res.reshape(
                    n_batch, c_out_per_group, oh_size, ow_size
                )

        elif ndim_spatial == 1:
            ow_size = out_spatial[0]
            kw = kernel_shape[0]

            col = np.empty(
                (n_batch, group, c_in_per_group * kw, ow_size), dtype=np.float32
            )
            idx = 0
            for ic in range(c_in_per_group):
                for ikw in range(kw):
                    iw_start = ikw * dilations[0]
                    for g in range(group):
                        ic_abs = g * c_in_per_group + ic
                        patches = x[
                            :, ic_abs, iw_start : iw_start + ow_size * strides[0] : strides[0]
                        ]
                        col[:, g, idx] = patches.reshape(n_batch, -1)
                    idx += 1

            w_col = w.reshape(group, c_out_per_group, -1)
            output = np.empty((n_batch, c_out, ow_size), dtype=np.float32)
            for g in range(group):
                res = np.einsum("ok,nkp->nop", w_col[g], col[:, g])
                oc_start = g * c_out_per_group
                output[:, oc_start : oc_start + c_out_per_group] = res.reshape(
                    n_batch, c_out_per_group, ow_size
                )
        else:
            # Fallback for higher dimensions: element-wise loop
            output = np.zeros((n_batch, c_out, *out_spatial), dtype=np.float32)
            for n in range(n_batch):
                for g in range(group):
                    for oc in range(c_out_per_group):
                        oc_abs = g * c_out_per_group + oc
                        for idx in np.ndindex(*out_spatial):
                            val = 0.0
                            for ic in range(c_in_per_group):
                                ic_abs = g * c_in_per_group + ic
                                for kidx in np.ndindex(*kernel_shape):
                                    src = tuple(
                                        idx[d] * strides[d] + kidx[d] * dilations[d]
                                        for d in range(ndim_spatial)
                                    )
                                    val += float(
                                        x[(n, ic_abs) + src] * w[(oc_abs, ic) + kidx]
                                    )
                            output[(n, oc_abs) + idx] = val

        if b is not None:
            bias_shape = [1, c_out] + [1] * ndim_spatial
            output += b.reshape(bias_shape)

        return output

    def _op_conv_transpose(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        x = inputs[0]  # N, C_in, *spatial
        w = inputs[1]  # C_in, C_out/group, *kernel
        b = inputs[2] if len(inputs) > 2 else None
        a = node.attributes
        kernel_shape = a.get("kernel_shape", list(w.shape[2:]))
        strides = a.get("strides", [1] * len(kernel_shape))
        pads = a.get("pads", [0] * (2 * len(kernel_shape)))
        dilations = a.get("dilations", [1] * len(kernel_shape))
        group = a.get("group", 1)
        output_padding = a.get("output_padding", [0] * len(kernel_shape))

        ndim_spatial = len(kernel_shape)
        n_batch = x.shape[0]
        c_in = x.shape[1]
        c_out_per_group = w.shape[1]
        c_out = c_out_per_group * group

        # Compute output spatial shape
        out_spatial = []
        for i in range(ndim_spatial):
            d = (
                strides[i] * (x.shape[2 + i] - 1)
                + dilations[i] * (kernel_shape[i] - 1)
                + 1
                + output_padding[i]
                - pads[i]
                - pads[i + ndim_spatial]
            )
            out_spatial.append(d)

        output = np.zeros((n_batch, c_out, *out_spatial), dtype=np.float32)
        c_in_per_group = c_in // group

        for n in range(n_batch):
            for g in range(group):
                for ic in range(c_in_per_group):
                    ic_abs = g * c_in_per_group + ic
                    for oc in range(c_out_per_group):
                        oc_abs = g * c_out_per_group + oc
                        if ndim_spatial == 2:
                            for ih in range(x.shape[2]):
                                for iw in range(x.shape[3]):
                                    for kh in range(kernel_shape[0]):
                                        for kw in range(kernel_shape[1]):
                                            oh = ih * strides[0] + kh * dilations[0] - pads[0]
                                            ow = iw * strides[1] + kw * dilations[1] - pads[1]
                                            oh_ok = 0 <= oh < out_spatial[0]
                                            ow_ok = 0 <= ow < out_spatial[1]
                                            if oh_ok and ow_ok:
                                                output[n, oc_abs, oh, ow] += float(
                                                    x[n, ic_abs, ih, iw] * w[ic_abs, oc, kh, kw]
                                                )
                        elif ndim_spatial == 1:
                            for iw in range(x.shape[2]):
                                for kw in range(kernel_shape[0]):
                                    ow = iw * strides[0] + kw * dilations[0] - pads[0]
                                    if 0 <= ow < out_spatial[0]:
                                        output[n, oc_abs, ow] += float(
                                            x[n, ic_abs, iw] * w[ic_abs, oc, kw]
                                        )

        if b is not None:
            bias_shape = [1, c_out] + [1] * ndim_spatial
            output += b.reshape(bias_shape)

        return [output]

    # ------------------------------------------------------------------
    # Compute ops: Gemm, normalization, reduction, pooling, softmax
    # ------------------------------------------------------------------

    def _op_gemm(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        a = inputs[0]
        b = inputs[1]
        c = inputs[2] if len(inputs) > 2 else None
        attrs = node.attributes
        alpha = attrs.get("alpha", 1.0)
        beta = attrs.get("beta", 1.0)
        transA = attrs.get("transA", 0)
        transB = attrs.get("transB", 0)
        if transA:
            a = a.T
        if transB:
            b = b.T
        out = alpha * (a @ b)
        if c is not None:
            out = out + beta * c
        return [out]

    def _op_softmax(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        x = inputs[0]
        axis = node.attributes.get("axis", -1)
        # Numerically stable softmax
        x_max = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - x_max)
        return [e / np.sum(e, axis=axis, keepdims=True)]

    def _op_layer_norm(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        x = inputs[0]
        scale = inputs[1]
        bias = inputs[2] if len(inputs) > 2 else None
        axis = node.attributes.get("axis", -1)
        eps = node.attributes.get("epsilon", 1e-5)
        # Normalize over axes [axis, ..., ndim-1]
        if axis < 0:
            axis = x.ndim + axis
        reduce_axes = tuple(range(axis, x.ndim))
        mean = np.mean(x, axis=reduce_axes, keepdims=True)
        var = np.var(x, axis=reduce_axes, keepdims=True)
        norm = (x - mean) / np.sqrt(var + eps)
        out = norm * scale
        if bias is not None:
            out = out + bias
        return [out]

    def _op_reduce_mean(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        x = inputs[0]
        keepdims = bool(node.attributes.get("keepdims", 1))
        if len(inputs) > 1:
            axes = tuple(int(a) for a in inputs[1].flatten())
        else:
            axes = node.attributes.get("axes")
        if axes is not None:
            axes = tuple(axes)
        return [np.mean(x, axis=axes, keepdims=keepdims)]

    def _op_reduce_sum(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        x = inputs[0]
        keepdims = bool(node.attributes.get("keepdims", 1))
        noop_with_empty_axes = node.attributes.get("noop_with_empty_axes", 0)
        if len(inputs) > 1:
            axes_arr = inputs[1].flatten()
            if axes_arr.size == 0:
                if noop_with_empty_axes:
                    return [x.copy()]
                return [np.sum(x, keepdims=keepdims)]
            axes = tuple(int(a) for a in axes_arr)
        else:
            axes = node.attributes.get("axes")
        if axes is not None:
            axes = tuple(axes)
        return [np.sum(x, axis=axes, keepdims=keepdims)]

    def _op_max_pool(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        x = inputs[0]
        a = node.attributes
        kernel_shape = a["kernel_shape"]
        strides = a.get("strides", [1] * len(kernel_shape))
        pads = a.get("pads", [0] * (2 * len(kernel_shape)))
        ceil_mode = a.get("ceil_mode", 0)
        return [self._pool_impl(x, kernel_shape, strides, pads, ceil_mode, mode="max")]

    def _op_average_pool(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        x = inputs[0]
        a = node.attributes
        kernel_shape = a["kernel_shape"]
        strides = a.get("strides", [1] * len(kernel_shape))
        pads = a.get("pads", [0] * (2 * len(kernel_shape)))
        ceil_mode = a.get("ceil_mode", 0)
        return [self._pool_impl(x, kernel_shape, strides, pads, ceil_mode, mode="avg")]

    def _op_global_average_pool(self, node: OpNode, inputs: list[np.ndarray]) -> list[np.ndarray]:
        x = inputs[0]  # N, C, *spatial
        spatial_axes = tuple(range(2, x.ndim))
        return [np.mean(x, axis=spatial_axes, keepdims=True)]

    def _pool_impl(
        self,
        x: np.ndarray,
        kernel_shape: list[int],
        strides: list[int],
        pads: list[int],
        ceil_mode: int,
        mode: str,
    ) -> np.ndarray:
        """Reference 2D pooling: pad + loop."""
        ndim_spatial = len(kernel_shape)
        n_batch, c = x.shape[0], x.shape[1]

        # Apply padding
        if any(p > 0 for p in pads):
            pad_pairs = [(0, 0), (0, 0)]
            half = ndim_spatial
            for i in range(ndim_spatial):
                pad_pairs.append((pads[i], pads[i + half]))
            pad_val = -np.inf if mode == "max" else 0.0
            x = np.pad(x, pad_pairs, mode="constant", constant_values=pad_val)

        out_spatial = []
        for i in range(ndim_spatial):
            if ceil_mode:
                d = int(np.ceil((x.shape[2 + i] - kernel_shape[i]) / strides[i])) + 1
            else:
                d = (x.shape[2 + i] - kernel_shape[i]) // strides[i] + 1
            out_spatial.append(d)

        output = np.zeros((n_batch, c, *out_spatial), dtype=np.float32)

        if ndim_spatial == 2:
            for n in range(n_batch):
                for ch in range(c):
                    for oh in range(out_spatial[0]):
                        for ow in range(out_spatial[1]):
                            region = x[
                                n,
                                ch,
                                oh * strides[0] : oh * strides[0] + kernel_shape[0],
                                ow * strides[1] : ow * strides[1] + kernel_shape[1],
                            ]
                            if mode == "max":
                                output[n, ch, oh, ow] = region.max()
                            else:
                                output[n, ch, oh, ow] = region.mean()
        elif ndim_spatial == 1:
            for n in range(n_batch):
                for ch in range(c):
                    for ow in range(out_spatial[0]):
                        region = x[
                            n,
                            ch,
                            ow * strides[0] : ow * strides[0] + kernel_shape[0],
                        ]
                        if mode == "max":
                            output[n, ch, ow] = region.max()
                        else:
                            output[n, ch, ow] = region.mean()

        return output
