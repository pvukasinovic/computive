"""MLASIC INT8 reference interpreter.

Performs exact hardware-matching INT8 math per docs/quantization-spec.md §3.2.
Used for QuantizationPass.verify() and end-to-end correctness testing.

Pipeline per fused layer:
  1. INT8 × INT8 → INT32 matmul (using np.int32 promotion)
  2. Add INT32 pre-scaled bias
  3. acc (INT32) × M_fixed (INT32) → INT64
  4. Round-half-up: (scaled + (1 << 15)) >> 16
  5. Add output zero_point
  6. Clamp to [-128, 127] → INT8
  7. If has_relu: max(result, 0)  [NOT max(result, zp_out)]
"""

from __future__ import annotations

import numpy as np

from mlasic.ir import (
    FusedActivationAttrs,
    FusedAttentionAttrs,
    FusedConvAttrs,
    FusedLayerNormAttrs,
    FusedLinearAttrs,
    FusedMLPAttrs,
    Graph,
    OpType,
)


class INT8Interpreter:
    """Execute a quantized/scheduled IR graph in exact INT8 arithmetic."""

    def __init__(self, graph: Graph) -> None:
        if graph.stage not in ("quantized", "scheduled", "scheduled_dag"):
            raise ValueError(
                f"INT8Interpreter requires 'quantized', 'scheduled', or "
                f"'scheduled_dag' stage, got '{graph.stage}'"
            )
        self.graph = graph

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Execute graph and return output tensors as INT8."""
        values = self._run_internal(inputs)
        return {name: values[name] for name in self.graph.outputs}

    def run_all(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Execute graph and return all intermediate values."""
        return self._run_internal(inputs)

    def _run_internal(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Execute graph, returning all tensor values."""
        # Validate inputs
        missing = set(self.graph.inputs) - set(inputs.keys())
        if missing:
            raise ValueError(f"Missing input(s): {sorted(missing)}")

        values: dict[str, np.ndarray] = {}

        # Load and auto-quantize inputs
        for name in self.graph.inputs:
            data = inputs[name]
            if np.issubdtype(data.dtype, np.floating):
                # Auto-quantize: find first node consuming this input
                data = self._auto_quantize_input(name, data)
            values[name] = data

        # Load constant tensors (already quantized INT8/INT32)
        for name, tensor in self.graph.tensors.items():
            if tensor.is_constant and tensor.data is not None:
                values[name] = tensor.data

        # Execute nodes in topological order
        for node_name in self.graph.topological_order():
            node = self.graph.nodes[node_name]
            output = self._execute_fused_layer(node, values)
            values[node.outputs[0]] = output

        return values

    def _auto_quantize_input(self, input_name: str, data: np.ndarray) -> np.ndarray:
        """Quantize float32 input using the first consuming node's input_quant."""
        for node in self.graph.nodes.values():
            if input_name in node.inputs:
                attrs = node.fused_attrs
                if attrs is not None and attrs.input_quant is not None:
                    return attrs.input_quant.quantize(data.astype(np.float32))
        raise ValueError(
            f"Cannot auto-quantize input '{input_name}': no consumer with quant params"
        )

    def _execute_fused_layer(self, node, values: dict[str, np.ndarray]) -> np.ndarray:
        """Execute one fused layer in exact INT8 arithmetic."""
        conv_fused = {OpType.FUSED_CONV, OpType.FUSED_CONV_RELU, OpType.FUSED_CONV_RELU6}
        linear_fused = {OpType.FUSED_LINEAR, OpType.FUSED_LINEAR_RELU}

        if node.op_type in conv_fused:
            return self._execute_fused_conv(node, values)
        elif node.op_type in linear_fused:
            return self._execute_fused_linear(node, values)
        elif node.op_type == OpType.FUSED_LAYER_NORM:
            return self._execute_fused_layer_norm(node, values)
        elif node.op_type in (OpType.FUSED_GELU, OpType.FUSED_SILU):
            return self._execute_fused_activation(node, values)
        elif node.op_type == OpType.FUSED_ATTENTION:
            return self._execute_fused_attention(node, values)
        elif node.op_type == OpType.FUSED_MLP:
            return self._execute_fused_mlp(node, values)
        else:
            raise ValueError(f"INT8Interpreter only supports fused ops, got {node.op_type}")

    def _execute_fused_linear(self, node, values: dict[str, np.ndarray]) -> np.ndarray:
        """Execute one fused linear layer in exact INT8 arithmetic."""
        attrs = node.fused_attrs

        # Step 1: INT8 × INT8 → INT32 matmul
        x = values[node.inputs[0]].astype(np.int32)
        w = values[node.inputs[1]].astype(np.int32)
        acc = x @ w  # INT32 accumulator

        # Step 2: Add INT32 pre-scaled bias
        if len(node.inputs) > 2:
            bias = values[node.inputs[2]].astype(np.int32)
            acc = acc + bias

        # Step 3-4: Requantize — acc × M_fixed with round-half-up shift
        m_fixed = np.int64(attrs.requant_scale_fixed)
        shift = attrs.requant_shift

        scaled = acc.astype(np.int64) * m_fixed
        # Round-half-up: (scaled + (1 << (shift-1))) >> shift
        rounded = (scaled + (np.int64(1) << (shift - 1))) >> shift

        # Step 5: Add output zero_point
        zp_out = np.int64(attrs.output_quant.zero_point)
        result = rounded + zp_out

        # Step 6: Clamp to [-128, 127]
        result = np.clip(result, -128, 127).astype(np.int8)

        # Step 7: ReLU (threshold is 0, NOT zp_out)
        if attrs.has_relu:
            result = np.maximum(result, np.int8(0))

        return result

    def _execute_fused_conv(self, node, values: dict[str, np.ndarray]) -> np.ndarray:
        """Execute one fused conv layer in exact INT8 arithmetic with per-channel requant."""
        attrs = node.fused_attrs
        assert isinstance(attrs, FusedConvAttrs)

        x = values[node.inputs[0]]  # INT8: (N, C_in, *spatial)
        w = values[node.inputs[1]]  # INT8: (C_out, C_in/group, *kernel)
        b = values[node.inputs[2]] if len(node.inputs) > 2 else None  # INT32: (C_out,)

        # Step 1: INT8 conv → INT32 accumulator (reference loop)
        acc = self._int8_conv_impl(
            x.astype(np.int32),
            w.astype(np.int32),
            attrs.kernel_shape,
            attrs.strides,
            attrs.pads,
            attrs.dilations,
            attrs.group,
        )

        # Step 2: Add INT32 bias
        c_out = w.shape[0]
        if b is not None:
            ndim_spatial = len(attrs.kernel_shape)
            bias_shape = [1, c_out] + [1] * ndim_spatial
            acc = acc + b.astype(np.int32).reshape(bias_shape)

        # Step 3-4: Per-channel requantization
        shift = attrs.requant_shift
        ndim_spatial = len(attrs.kernel_shape)
        m_fixed_arr = np.array(attrs.requant_scale_fixed, dtype=np.int64)
        # Shape: (1, C_out, 1, 1, ...)
        m_shape = [1, c_out] + [1] * ndim_spatial
        m_fixed_bc = m_fixed_arr.reshape(m_shape)

        scaled = acc.astype(np.int64) * m_fixed_bc
        rounded = (scaled + (np.int64(1) << (shift - 1))) >> shift

        # Step 5: Add output zero_point
        zp_out = np.int64(attrs.output_quant.zero_point)
        result = rounded + zp_out

        # Step 6: Clamp
        result = np.clip(result, -128, 127).astype(np.int8)

        # Step 7: Activation
        if attrs.has_relu:
            result = np.maximum(result, np.int8(0))
        elif attrs.has_relu6:
            result = np.maximum(result, np.int8(0))
            # Clip to quantized 6.0: floor(6.0 / scale + 0.5) + zp
            scale_y = attrs.output_quant.scale
            zp_y = attrs.output_quant.zero_point
            q6 = int(np.floor(6.0 / scale_y + 0.5)) + zp_y
            q6 = min(q6, 127)
            result = np.minimum(result, np.int8(q6))

        return result

    @staticmethod
    def _int8_conv_impl(
        x: np.ndarray,
        w: np.ndarray,
        kernel_shape: list[int],
        strides: list[int],
        pads: list[int],
        dilations: list[int],
        group: int,
    ) -> np.ndarray:
        """Reference INT32 convolution (inputs already promoted to int32)."""
        ndim_spatial = len(kernel_shape)
        n_batch = x.shape[0]
        c_out = w.shape[0]

        if any(p > 0 for p in pads):
            pad_pairs = [(0, 0), (0, 0)]
            half = ndim_spatial
            for i in range(ndim_spatial):
                pad_pairs.append((pads[i], pads[i + half]))
            x = np.pad(x, pad_pairs, mode="constant", constant_values=0)

        out_spatial = []
        for i in range(ndim_spatial):
            d = (x.shape[2 + i] - dilations[i] * (kernel_shape[i] - 1) - 1) // strides[i] + 1
            out_spatial.append(d)

        output = np.zeros((n_batch, c_out, *out_spatial), dtype=np.int32)
        c_in_per_group = x.shape[1] // group
        c_out_per_group = c_out // group

        for n in range(n_batch):
            for g in range(group):
                for oc in range(c_out_per_group):
                    oc_abs = g * c_out_per_group + oc
                    if ndim_spatial == 2:
                        for oh in range(out_spatial[0]):
                            for ow in range(out_spatial[1]):
                                val = np.int32(0)
                                for ic in range(c_in_per_group):
                                    ic_abs = g * c_in_per_group + ic
                                    for kh in range(kernel_shape[0]):
                                        for kw in range(kernel_shape[1]):
                                            ih = oh * strides[0] + kh * dilations[0]
                                            iw = ow * strides[1] + kw * dilations[1]
                                            val += x[n, ic_abs, ih, iw] * w[oc_abs, ic, kh, kw]
                                output[n, oc_abs, oh, ow] = val
                    elif ndim_spatial == 1:
                        for ow in range(out_spatial[0]):
                            val = np.int32(0)
                            for ic in range(c_in_per_group):
                                ic_abs = g * c_in_per_group + ic
                                for kw in range(kernel_shape[0]):
                                    iw = ow * strides[0] + kw * dilations[0]
                                    val += x[n, ic_abs, iw] * w[oc_abs, ic, kw]
                            output[n, oc_abs, ow] = val

        return output

    def _execute_fused_layer_norm(self, node, values: dict[str, np.ndarray]) -> np.ndarray:
        """Execute FusedLayerNorm: INT8 → dequantize → FP32 LN → requantize → INT8.

        LayerNorm requires FP32 for mean/variance computation.
        """
        attrs = node.fused_attrs
        assert isinstance(attrs, FusedLayerNormAttrs)

        x_q = values[node.inputs[0]]

        # Dequantize input to FP32
        if attrs.input_quant is not None:
            x_fp = attrs.input_quant.dequantize(x_q)
        else:
            x_fp = x_q.astype(np.float32)

        # Load scale and bias (FP32 constants)
        scale = values[node.inputs[1]].astype(np.float32) if len(node.inputs) > 1 else None
        bias = values[node.inputs[2]].astype(np.float32) if len(node.inputs) > 2 else None

        # FP32 LayerNorm
        axis = attrs.axis
        eps = attrs.epsilon
        mean = np.mean(x_fp, axis=axis, keepdims=True)
        var = np.var(x_fp, axis=axis, keepdims=True)
        x_norm = (x_fp - mean) / np.sqrt(var + eps)

        if scale is not None:
            x_norm = x_norm * scale
        if bias is not None:
            x_norm = x_norm + bias

        # Requantize output to INT8
        if attrs.output_quant is not None:
            return attrs.output_quant.quantize(x_norm)
        return np.clip(np.floor(x_norm + 0.5), -128, 127).astype(np.int8)

    def _execute_fused_activation(self, node, values: dict[str, np.ndarray]) -> np.ndarray:
        """Execute FusedGELU / FusedSiLU: INT8 → dequantize → FP32 activation → requantize.

        Uses direct FP32 computation (LUT is a hardware optimization, not needed for reference).
        """
        attrs = node.fused_attrs
        assert isinstance(attrs, FusedActivationAttrs)

        x_q = values[node.inputs[0]]

        # Dequantize to FP32
        if attrs.input_quant is not None:
            x_fp = attrs.input_quant.dequantize(x_q)
        else:
            x_fp = x_q.astype(np.float32)

        # Apply activation in FP32
        if attrs.activation_type == "gelu":
            # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
            result = 0.5 * x_fp * (1.0 + _erf_approx(x_fp / np.sqrt(2.0)))
        elif attrs.activation_type == "silu":
            # SiLU(x) = x * sigmoid(x)
            result = x_fp * _sigmoid(x_fp)
        else:
            raise ValueError(f"Unknown activation type: {attrs.activation_type}")

        # Requantize to INT8
        if attrs.output_quant is not None:
            return attrs.output_quant.quantize(result)
        return np.clip(np.floor(result + 0.5), -128, 127).astype(np.int8)

    def _execute_fused_attention(self, node, values: dict[str, np.ndarray]) -> np.ndarray:
        """Execute FusedAttention in mixed INT8/FP32.

        INT8 QKV projections → FP32 softmax → INT8 output projection.
        """
        attrs = node.fused_attrs
        assert isinstance(attrs, FusedAttentionAttrs)

        x_q = values[node.inputs[0]]  # INT8 input

        # Dequantize input for FP32 attention computation
        if attrs.input_quant is not None:
            x_fp = attrs.input_quant.dequantize(x_q)
        else:
            x_fp = x_q.astype(np.float32)

        # Collect projection weights and biases from inputs[1:]
        proj_tensors = []
        for i in range(1, len(node.inputs)):
            proj_tensors.append(values[node.inputs[i]])

        # QKV projections in FP32 (weights may be INT8 — dequantize if needed)
        idx = 0
        proj_quants = [attrs.q_weight_quant, attrs.k_weight_quant,
                       attrs.v_weight_quant, attrs.output_weight_quant]
        projections = []  # [Q, K, V] then O later

        for proj_i in range(3):  # Q, K, V
            if idx >= len(proj_tensors):
                break
            w = proj_tensors[idx].astype(np.float32)
            idx += 1
            # Dequantize weight if INT8
            pq = proj_quants[proj_i]
            if pq is not None:
                w = (w - pq.zero_point) * pq.scale

            proj = x_fp @ w
            # Add bias if present
            if idx < len(proj_tensors) and proj_tensors[idx].ndim == 1:
                b = proj_tensors[idx].astype(np.float32)
                proj = proj + b
                idx += 1
            projections.append(proj)

        if len(projections) < 3:
            # Fallback: not enough projections, just pass through
            if attrs.output_quant is not None:
                return attrs.output_quant.quantize(x_fp)
            return np.clip(np.floor(x_fp + 0.5), -128, 127).astype(np.int8)

        Q, K, V = projections

        # Multi-head reshape: (batch, seq, embed) → (batch, heads, seq, head_dim)
        nh = attrs.num_heads
        hd = attrs.head_dim
        orig_shape = Q.shape

        if Q.ndim == 2:
            # (seq, embed) → (1, seq, embed)
            Q = Q.reshape(1, -1, nh * hd)
            K = K.reshape(1, -1, nh * hd)
            V = V.reshape(1, -1, nh * hd)

        batch = Q.shape[0]
        seq = Q.shape[1]

        Q = Q.reshape(batch, seq, nh, hd).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq, nh, hd).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq, nh, hd).transpose(0, 2, 1, 3)

        # Attention scores: Q @ K^T / sqrt(head_dim) — FP32
        scale = 1.0 / np.sqrt(float(hd))
        attn_scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale

        # Softmax in FP32
        attn_max = np.max(attn_scores, axis=-1, keepdims=True)
        attn_exp = np.exp(attn_scores - attn_max)
        attn_weights = attn_exp / np.sum(attn_exp, axis=-1, keepdims=True)

        # Weighted sum: attn_weights @ V
        attn_out = np.matmul(attn_weights, V)

        # Merge heads: (batch, heads, seq, head_dim) → (batch, seq, embed)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq, nh * hd)

        # Squeeze batch if original was 2D
        if len(orig_shape) == 2:
            attn_out = attn_out.squeeze(0)

        # Output projection
        if idx < len(proj_tensors):
            w_o = proj_tensors[idx].astype(np.float32)
            pq_o = proj_quants[3]
            if pq_o is not None:
                w_o = (w_o - pq_o.zero_point) * pq_o.scale
            idx += 1
            attn_out = attn_out @ w_o
            # Add bias if present
            if idx < len(proj_tensors) and proj_tensors[idx].ndim == 1:
                b_o = proj_tensors[idx].astype(np.float32)
                attn_out = attn_out + b_o

        # Requantize to INT8
        if attrs.output_quant is not None:
            return attrs.output_quant.quantize(attn_out)
        return np.clip(np.floor(attn_out + 0.5), -128, 127).astype(np.int8)

    def _execute_fused_mlp(self, node, values: dict[str, np.ndarray]) -> np.ndarray:
        """Execute FusedMLP: fc1 → activation → fc2, all in INT8/FP32 mixed.

        fc1 and fc2 use INT8 matmul when quantized, activation is FP32.
        """
        attrs = node.fused_attrs
        assert isinstance(attrs, FusedMLPAttrs)

        x_q = values[node.inputs[0]]

        # Dequantize input
        if attrs.input_quant is not None:
            x_fp = attrs.input_quant.dequantize(x_q)
        else:
            x_fp = x_q.astype(np.float32)

        # fc1: inputs[1] = weight, inputs[2] = bias (optional)
        w1 = values[node.inputs[1]].astype(np.float32)
        if attrs.fc1_weight_quant is not None:
            w1 = (w1 - attrs.fc1_weight_quant.zero_point) * attrs.fc1_weight_quant.scale
        h = x_fp @ w1
        if len(node.inputs) > 2:
            t = values[node.inputs[2]]
            if t.ndim == 1:
                h = h + t.astype(np.float32)
                bias1_present = True
            else:
                bias1_present = False
        else:
            bias1_present = False

        # Activation in FP32
        if attrs.activation_type == "gelu":
            h = 0.5 * h * (1.0 + _erf_approx(h / np.sqrt(2.0)))
        elif attrs.activation_type == "silu":
            h = h * _sigmoid(h)
        elif attrs.activation_type == "relu":
            h = np.maximum(h, 0.0)
        # "none" — no activation

        # fc2: next weight and optional bias
        fc2_w_idx = 3 if bias1_present else 2
        if fc2_w_idx >= len(node.inputs):
            # Fallback
            if attrs.output_quant is not None:
                return attrs.output_quant.quantize(h)
            return np.clip(np.floor(h + 0.5), -128, 127).astype(np.int8)

        w2 = values[node.inputs[fc2_w_idx]].astype(np.float32)
        if attrs.fc2_weight_quant is not None:
            w2 = (w2 - attrs.fc2_weight_quant.zero_point) * attrs.fc2_weight_quant.scale
        result = h @ w2

        fc2_b_idx = fc2_w_idx + 1
        if fc2_b_idx < len(node.inputs):
            b2 = values[node.inputs[fc2_b_idx]]
            if b2.ndim == 1:
                result = result + b2.astype(np.float32)

        # Requantize to INT8
        if attrs.output_quant is not None:
            return attrs.output_quant.quantize(result)
        return np.clip(np.floor(result + 0.5), -128, 127).astype(np.int8)

    @staticmethod
    def quantize_input(data: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
        """Quantize FP32 input to INT8 using asymmetric params."""
        q = np.floor(data / scale + 0.5) + zero_point
        return np.clip(q, -128, 127).astype(np.int8)

    @staticmethod
    def dequantize_output(data: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
        """Dequantize INT8 output back to FP32."""
        return (data.astype(np.float32) - zero_point) * scale


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _erf_approx(x: np.ndarray) -> np.ndarray:
    """Error function using scipy if available, else numpy approximation."""
    try:
        from scipy.special import erf
        return erf(x)
    except ImportError:
        # Abramowitz and Stegun approximation
        a = np.abs(x)
        t = 1.0 / (1.0 + 0.3275911 * a)
        p = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        return np.sign(x) * (1.0 - p * np.exp(-a * a))
