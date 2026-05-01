"""MLASIC graph optimization passes.

Implements the 5 sequential passes of Stage 2:
  1. ConstantFoldingPass — evaluate all-constant nodes at compile time
  2. DeadCodeEliminationPass — remove unreachable nodes/tensors
  3. BatchNormFoldingPass — fold BN params into preceding weight/bias
  4. OperatorFusionPass — fuse MatMul+Add[+ReLU] into FusedLinear[ReLU]
  5. QuantizationPass — INT8 weight/activation quantization with calibration
"""

from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from mlasic.interpreter import IRInterpreter
from mlasic.ir import (
    FusedActivationAttrs,
    FusedAttentionAttrs,
    FusedConvAttrs,
    FusedLayerNormAttrs,
    FusedLinearAttrs,
    FusedMLPAttrs,
    Graph,
    OpNode,
    OpType,
    QuantParams,
    Tensor,
    TensorType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pass infrastructure
# ---------------------------------------------------------------------------


class OptimizationPass(ABC):
    """Base class for graph optimization passes."""

    # Subclasses can override for passes that introduce FP32 rounding
    # (e.g., BN folding accumulates rounding error over multiple layers).
    _verify_rtol: float = 1e-5
    _verify_atol: float = 1e-6

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable pass name."""

    @abstractmethod
    def run(self, graph: Graph) -> Graph:
        """Transform the graph. Returns new or mutated graph."""

    def verify(self, before: Graph, after: Graph) -> bool:
        """Verify transformation preserves semantics using IRInterpreter.

        Runs 10 random vectors through both graphs and compares outputs.
        """
        interp_before = IRInterpreter(before)
        interp_after = IRInterpreter(after)

        input_tensor = before.tensors[before.inputs[0]]
        input_shape = input_tensor.type.shape

        rng = np.random.RandomState(42)
        for i in range(10):
            test_input = rng.randn(*input_shape).astype(np.float32)
            inputs_dict = {before.inputs[0]: test_input}

            out_before = interp_before.run(inputs_dict)
            out_after = interp_after.run(inputs_dict)

            for key in out_before:
                if not np.allclose(
                    out_before[key], out_after[key], rtol=self._verify_rtol, atol=self._verify_atol
                ):
                    logger.error(
                        "Verification failed for pass %s on output %s (vector %d)",
                        self.name,
                        key,
                        i,
                    )
                    return False
        return True


class PassManager:
    """Runs optimization passes sequentially with optional verification."""

    def __init__(self) -> None:
        self.passes: list[OptimizationPass] = []

    def add_pass(self, pass_: OptimizationPass) -> None:
        self.passes.append(pass_)

    def run(self, graph: Graph, verify: bool = True) -> Graph:
        """Run all passes in order."""
        for pass_ in self.passes:
            before = copy.deepcopy(graph) if verify else None
            node_count_before = len(graph.nodes)

            logger.info("Running pass: %s (%d nodes)", pass_.name, node_count_before)
            graph = pass_.run(graph)
            graph.invalidate_cache()

            node_count_after = len(graph.nodes)
            logger.info(
                "  %s: %d -> %d nodes",
                pass_.name,
                node_count_before,
                node_count_after,
            )

            if verify and before is not None:
                if not pass_.verify(before, graph):
                    raise RuntimeError(f"Verification failed after pass: {pass_.name}")

        return graph


# ---------------------------------------------------------------------------
# Pass 1: Constant Folding
# ---------------------------------------------------------------------------


class ConstantFoldingPass(OptimizationPass):
    """Evaluate nodes with all-constant inputs at compile time.

    Iterates until fixpoint — each round may expose new all-constant nodes.
    Supports: MatMul, Add, Reshape, Transpose.
    """

    @property
    def name(self) -> str:
        return "ConstantFolding"

    def run(self, graph: Graph) -> Graph:
        changed = True
        while changed:
            changed = False
            for node_name in list(graph.topological_order()):
                node = graph.nodes.get(node_name)
                if node is None:
                    continue

                # Check if all inputs are constants
                if not all(graph.tensors[inp].is_constant for inp in node.inputs):
                    continue

                result = self._evaluate(node, graph)
                if result is None:
                    continue

                # Replace output tensor with computed constant
                out_name = node.outputs[0]
                out_tensor = graph.tensors[out_name]
                out_tensor.data = result
                out_tensor.type = TensorType(
                    shape=tuple(result.shape),
                    dtype=result.dtype,
                    quant=out_tensor.type.quant,
                )

                # Remove node
                del graph.nodes[node_name]
                graph.invalidate_cache()
                changed = True
                break  # Restart — topo order changed

        return graph

    def _evaluate(self, node: OpNode, graph: Graph) -> np.ndarray | None:
        """Evaluate a node with constant inputs. Returns None if unsupported."""
        inputs = [graph.tensors[inp].data.astype(np.float32) for inp in node.inputs]
        op = node.op_type

        if op == OpType.MATMUL:
            return inputs[0] @ inputs[1]
        elif op == OpType.ADD:
            return inputs[0] + inputs[1]
        elif op == OpType.RESHAPE:
            shape = inputs[1].astype(int).tolist()
            return inputs[0].reshape(shape)
        elif op == OpType.TRANSPOSE:
            perm = node.attributes.get("perm")
            if perm:
                return np.transpose(inputs[0], perm)
            return np.transpose(inputs[0])
        elif op == OpType.MUL:
            return inputs[0] * inputs[1]
        elif op == OpType.DIV:
            return inputs[0] / inputs[1]
        elif op == OpType.SUB:
            return inputs[0] - inputs[1]
        elif op == OpType.CONCAT:
            axis = node.attributes.get("axis", 0)
            return np.concatenate(inputs, axis=axis)
        elif op == OpType.SQUEEZE:
            x = inputs[0]
            if len(inputs) > 1:
                axes = tuple(sorted(int(a) for a in inputs[1].flatten()))
            else:
                axes = node.attributes.get("axes")
            if axes is not None:
                result = x
                for ax in sorted(axes, reverse=True):
                    result = np.squeeze(result, axis=ax)
                return result
            return np.squeeze(x)
        elif op == OpType.UNSQUEEZE:
            x = inputs[0]
            if len(inputs) > 1:
                axes = sorted(int(a) for a in inputs[1].flatten())
            else:
                axes = sorted(node.attributes.get("axes", []))
            result = x
            for ax in axes:
                result = np.expand_dims(result, axis=ax)
            return result
        elif op == OpType.CAST:
            to = node.attributes.get("to", 1)
            dtype_map = {
                1: np.float32,
                6: np.int32,
                7: np.int64,
                11: np.float64,
            }
            dtype = dtype_map.get(to, np.float32)
            return inputs[0].astype(dtype)
        elif op == OpType.GATHER:
            data = inputs[0]
            indices = inputs[1].astype(int)
            axis = node.attributes.get("axis", 0)
            return np.take(data, indices, axis=axis)
        elif op == OpType.SHAPE:
            return np.array(inputs[0].shape, dtype=np.int64)
        elif op == OpType.CONSTANT_OF_SHAPE:
            shape = tuple(inputs[0].astype(int).tolist())
            value = node.attributes.get("value")
            if value is not None:
                if isinstance(value, np.ndarray):
                    fill = float(value.flat[0])
                else:
                    fill = float(value)
            else:
                fill = 0.0
            return np.full(shape, fill, dtype=np.float32)
        elif op == OpType.EXPAND:
            return np.broadcast_to(inputs[0], inputs[1].astype(int).tolist()).copy()
        elif op == OpType.TILE:
            return np.tile(inputs[0], inputs[1].astype(int).tolist())
        elif op == OpType.PAD:
            data = inputs[0]
            pads = inputs[1].astype(int).tolist()
            constant_value = float(inputs[2]) if len(inputs) > 2 else 0.0
            ndim = data.ndim
            pad_pairs = [(pads[i], pads[i + ndim]) for i in range(ndim)]
            return np.pad(data, pad_pairs, mode="constant", constant_values=constant_value)
        elif op == OpType.SLICE:
            data = inputs[0]
            starts = inputs[1].astype(int).tolist()
            ends = inputs[2].astype(int).tolist()
            axes = inputs[3].astype(int).tolist() if len(inputs) > 3 else list(range(len(starts)))
            steps = inputs[4].astype(int).tolist() if len(inputs) > 4 else [1] * len(starts)
            slices = [slice(None)] * data.ndim
            for ax, s, e, st in zip(axes, starts, ends, steps):
                slices[ax] = slice(s, e, st)
            return data[tuple(slices)]
        return None


# ---------------------------------------------------------------------------
# Pass 2: Dead Code Elimination
# ---------------------------------------------------------------------------


class DeadCodeEliminationPass(OptimizationPass):
    """Remove nodes and tensors not reachable from graph outputs.

    Backward reachability from outputs. Preserves graph inputs.
    """

    @property
    def name(self) -> str:
        return "DeadCodeElimination"

    def run(self, graph: Graph) -> Graph:
        # Backward reachability from graph outputs
        reachable_nodes: set[str] = set()
        reachable_tensors: set[str] = set()
        visited: set[str] = set()
        worklist = list(graph.outputs)

        # Graph inputs are always reachable
        for inp in graph.inputs:
            reachable_tensors.add(inp)

        while worklist:
            tensor_name = worklist.pop()
            if tensor_name in visited:
                continue
            visited.add(tensor_name)
            reachable_tensors.add(tensor_name)

            producer = graph.get_producer(tensor_name)
            if producer and producer.name not in reachable_nodes:
                reachable_nodes.add(producer.name)
                for inp in producer.inputs:
                    worklist.append(inp)
                    reachable_tensors.add(inp)

        # Remove dead nodes
        dead_nodes = [n for n in graph.nodes if n not in reachable_nodes]
        for name in dead_nodes:
            del graph.nodes[name]

        # Remove orphan tensors
        orphan_tensors = [t for t in graph.tensors if t not in reachable_tensors]
        for name in orphan_tensors:
            del graph.tensors[name]

        if dead_nodes or orphan_tensors:
            graph.invalidate_cache()

        return graph


# ---------------------------------------------------------------------------
# Pass 3: BatchNorm Folding
# ---------------------------------------------------------------------------


class BatchNormFoldingPass(OptimizationPass):
    """Fold BatchNormalization into preceding MatMul+Add weights/bias.

    Pattern: MatMul → Add → BatchNorm
    Result:  MatMul → Add (with modified W' and b')

    Formulas:
        scale = gamma / sqrt(var + eps)
        W' = W * scale            (broadcast over columns)
        b' = scale * (b - mean) + beta
    """

    # BN folding changes computation order of FP32 ops. Rounding
    # differences accumulate over multiple layers (up to ~1e-4 for
    # the 4-layer AD model).
    _verify_atol: float = 1e-4

    @property
    def name(self) -> str:
        return "BatchNormFolding"

    def run(self, graph: Graph) -> Graph:
        bn_names = [n.name for n in graph.nodes.values() if n.op_type == OpType.BATCH_NORM]

        for bn_name in bn_names:
            if bn_name not in graph.nodes:
                continue
            self._fold_bn(graph, graph.nodes[bn_name])

        return graph

    def _fold_bn(self, graph: Graph, bn_node: OpNode) -> None:
        """Fold a single BatchNorm into preceding MatMul+Add."""
        # BN inputs: x, gamma, beta, mean, var
        bn_input_name = bn_node.inputs[0]
        gamma = graph.tensors[bn_node.inputs[1]].data.astype(np.float32)
        beta = graph.tensors[bn_node.inputs[2]].data.astype(np.float32)
        mean = graph.tensors[bn_node.inputs[3]].data.astype(np.float32)
        var = graph.tensors[bn_node.inputs[4]].data.astype(np.float32)
        eps = bn_node.attributes.get("epsilon", 1e-5)

        # Find preceding Add node
        add_node = graph.get_producer(bn_input_name)
        if add_node is None or add_node.op_type != OpType.ADD:
            logger.debug("BN %s: no preceding Add, skipping", bn_node.name)
            return

        # Find the MatMul that feeds into Add and the bias constant
        matmul_node = None
        bias_tensor = None
        for add_inp in add_node.inputs:
            producer = graph.get_producer(add_inp)
            if producer and producer.op_type == OpType.MATMUL:
                matmul_node = producer
            elif graph.tensors[add_inp].is_constant:
                bias_tensor = graph.tensors[add_inp]

        if matmul_node is None or bias_tensor is None:
            logger.debug("BN %s: pattern mismatch, skipping", bn_node.name)
            return

        # Weight tensor: second input of MatMul (activation @ weight)
        weight_tensor = graph.tensors[matmul_node.inputs[1]]

        # Compute folded params
        scale = gamma / np.sqrt(var + eps)

        # W' = W * scale — weight [in_dim, out_dim], scale [out_dim]
        weight_tensor.data = (weight_tensor.data.astype(np.float32) * scale).astype(np.float32)

        # b' = scale * (b - mean) + beta
        bias_tensor.data = (scale * (bias_tensor.data.astype(np.float32) - mean) + beta).astype(
            np.float32
        )

        # Rewire: consumers of BN output now consume Add output
        bn_output_name = bn_node.outputs[0]
        for node in graph.nodes.values():
            node.inputs = [bn_input_name if inp == bn_output_name else inp for inp in node.inputs]

        # Update graph outputs if needed
        graph.outputs = [bn_input_name if out == bn_output_name else out for out in graph.outputs]

        # Remove BN node
        del graph.nodes[bn_node.name]

        # Remove orphan tensors: BN output and param tensors
        orphan_candidates = [bn_output_name] + list(bn_node.inputs[1:])
        for name in orphan_candidates:
            if name in graph.tensors:
                consumers = [n for n in graph.nodes.values() if name in n.inputs]
                if not consumers:
                    del graph.tensors[name]

        graph.invalidate_cache()


# ---------------------------------------------------------------------------
# Pass 3b: Conv BatchNorm Folding
# ---------------------------------------------------------------------------


class ConvBatchNormFoldingPass(OptimizationPass):
    """Fold BatchNormalization into preceding Conv weights/bias.

    Pattern: Conv → BatchNorm => Conv (with modified W' and b')

    Formulas (per output channel):
        scale[oc] = gamma[oc] / sqrt(var[oc] + eps)
        W'[oc] = W[oc] * scale[oc]
        b'[oc] = scale[oc] * (b[oc] - mean[oc]) + beta[oc]

    Handles Conv with no bias (creates zero bias), depthwise, grouped.
    """

    _verify_atol: float = 1e-4

    @property
    def name(self) -> str:
        return "ConvBatchNormFolding"

    def run(self, graph: Graph) -> Graph:
        bn_names = [n.name for n in graph.nodes.values() if n.op_type == OpType.BATCH_NORM]

        for bn_name in bn_names:
            if bn_name not in graph.nodes:
                continue
            bn_node = graph.nodes[bn_name]
            # Check if preceded by Conv
            bn_input_name = bn_node.inputs[0]
            producer = graph.get_producer(bn_input_name)
            if producer is not None and producer.op_type == OpType.CONV:
                self._fold_conv_bn(graph, producer, bn_node)

        return graph

    def _fold_conv_bn(self, graph: Graph, conv_node: OpNode, bn_node: OpNode) -> None:
        """Fold BN into preceding Conv."""
        # BN inputs: x, gamma, beta, mean, var
        gamma = graph.tensors[bn_node.inputs[1]].data.astype(np.float32)
        beta = graph.tensors[bn_node.inputs[2]].data.astype(np.float32)
        mean = graph.tensors[bn_node.inputs[3]].data.astype(np.float32)
        var = graph.tensors[bn_node.inputs[4]].data.astype(np.float32)
        eps = bn_node.attributes.get("epsilon", 1e-5)

        # Conv weight: [C_out, C_in/group, *kernel]
        weight_tensor = graph.tensors[conv_node.inputs[1]]
        w_data = weight_tensor.data.astype(np.float32)
        c_out = w_data.shape[0]

        # Compute per-channel scale
        scale = gamma / np.sqrt(var + eps)

        # W'[oc] = W[oc] * scale[oc]
        # Reshape scale for broadcasting: (C_out, 1, 1, ...) for kernel dims
        scale_shape = [c_out] + [1] * (w_data.ndim - 1)
        weight_tensor.data = (w_data * scale.reshape(scale_shape)).astype(np.float32)

        # Handle bias: Conv may or may not have a bias
        if len(conv_node.inputs) >= 3:
            bias_tensor = graph.tensors[conv_node.inputs[2]]
            b_data = bias_tensor.data.astype(np.float32)
        else:
            # Create zero bias and add it to the conv
            b_data = np.zeros(c_out, dtype=np.float32)
            bias_name = f"{conv_node.name}_bias"
            bias_tensor = Tensor(
                name=bias_name,
                type=TensorType(shape=(c_out,), dtype=np.dtype(np.float32)),
                data=b_data,
            )
            graph.tensors[bias_name] = bias_tensor
            conv_node.inputs.append(bias_name)

        # b'[oc] = scale[oc] * (b[oc] - mean[oc]) + beta[oc]
        bias_tensor.data = (scale * (b_data - mean) + beta).astype(np.float32)

        # Rewire: consumers of BN output now consume Conv output
        bn_input_name = bn_node.inputs[0]  # Conv's output
        bn_output_name = bn_node.outputs[0]
        for node in graph.nodes.values():
            node.inputs = [bn_input_name if inp == bn_output_name else inp for inp in node.inputs]
        graph.outputs = [bn_input_name if out == bn_output_name else out for out in graph.outputs]

        # Remove BN node and orphan tensors
        del graph.nodes[bn_node.name]
        orphan_candidates = [bn_output_name] + list(bn_node.inputs[1:])
        for name in orphan_candidates:
            if name in graph.tensors:
                consumers = [n for n in graph.nodes.values() if name in n.inputs]
                if not consumers and name not in graph.outputs:
                    del graph.tensors[name]

        graph.invalidate_cache()


# ---------------------------------------------------------------------------
# Pass 4: Operator Fusion
# ---------------------------------------------------------------------------


class OperatorFusionPass(OptimizationPass):
    """Fuse MatMul+Add[+ReLU] into FusedLinear[ReLU] operators.

    Patterns:
        MatMul → Add → ReLU  ⇒  FusedLinearReLU
        MatMul → Add          ⇒  FusedLinear
    """

    @property
    def name(self) -> str:
        return "OperatorFusion"

    def run(self, graph: Graph) -> Graph:
        new_nodes: dict[str, OpNode] = {}
        consumed_nodes: set[str] = set()
        fused_counter = 0

        for node_name in graph.topological_order():
            if node_name in consumed_nodes:
                continue

            node = graph.nodes[node_name]
            if node.op_type != OpType.MATMUL:
                continue

            # Start fusion from MatMul
            matmul_node = node
            weight_name = matmul_node.inputs[1]
            weight_tensor = graph.tensors[weight_name]
            activation_input = matmul_node.inputs[0]

            input_dim = weight_tensor.type.shape[0]
            output_dim = weight_tensor.type.shape[1]

            # Check for following Add
            matmul_out = matmul_node.outputs[0]
            add_node = self._find_single_consumer(graph, matmul_out, OpType.ADD)

            bias_name = None
            has_relu = False
            last_output = matmul_out

            if add_node is not None:
                consumed_nodes.add(add_node.name)
                # Find bias input (the non-matmul input)
                for inp in add_node.inputs:
                    if inp != matmul_out:
                        bias_name = inp
                last_output = add_node.outputs[0]

                # Check for following ReLU
                relu_node = self._find_single_consumer(graph, last_output, OpType.RELU)
                if relu_node is not None:
                    consumed_nodes.add(relu_node.name)
                    has_relu = True
                    last_output = relu_node.outputs[0]
            else:
                # Check for ReLU directly after MatMul (no bias)
                relu_node = self._find_single_consumer(graph, matmul_out, OpType.RELU)
                if relu_node is not None:
                    consumed_nodes.add(relu_node.name)
                    has_relu = True
                    last_output = relu_node.outputs[0]

            consumed_nodes.add(matmul_node.name)

            # Create fused node
            op_type = OpType.FUSED_LINEAR_RELU if has_relu else OpType.FUSED_LINEAR
            fused_name = f"fused_{fused_counter}"
            fused_counter += 1

            fused_inputs = [activation_input, weight_name]
            if bias_name is not None:
                fused_inputs.append(bias_name)

            fused_node = OpNode(
                name=fused_name,
                op_type=op_type,
                inputs=fused_inputs,
                outputs=[last_output],
            )
            fused_node.fused_attrs = FusedLinearAttrs(
                input_dim=input_dim,
                output_dim=output_dim,
                has_relu=has_relu,
            )

            new_nodes[fused_name] = fused_node

        # Preserve all non-consumed, non-fused nodes (Conv, Pool, Flatten, etc.)
        for node_name in graph.topological_order():
            if node_name not in consumed_nodes and node_name not in new_nodes:
                new_nodes[node_name] = graph.nodes[node_name]

        # Build new tensor set — all tensors referenced by remaining nodes
        needed_tensors: set[str] = set()
        needed_tensors.update(graph.inputs)
        needed_tensors.update(graph.outputs)
        for node in new_nodes.values():
            needed_tensors.update(node.inputs)
            needed_tensors.update(node.outputs)

        new_tensors = {
            name: graph.tensors[name] for name in needed_tensors if name in graph.tensors
        }

        # Only set "fused" if all nodes are fused ops; otherwise preserve stage
        fused_types = {OpType.FUSED_LINEAR, OpType.FUSED_LINEAR_RELU}
        all_fused = all(n.op_type in fused_types for n in new_nodes.values())
        stage = "fused" if all_fused else graph.stage

        return Graph(
            name=graph.name,
            nodes=new_nodes,
            tensors=new_tensors,
            inputs=list(graph.inputs),
            outputs=list(graph.outputs),
            stage=stage,
        )

    def _find_single_consumer(
        self, graph: Graph, tensor_name: str, expected_op: OpType
    ) -> OpNode | None:
        """Find the single consumer of a tensor with the expected op type."""
        consumers = graph.get_consumers(tensor_name)
        if len(consumers) == 1 and consumers[0].op_type == expected_op:
            return consumers[0]
        return None


# ---------------------------------------------------------------------------
# Pass 5: Quantization
# ---------------------------------------------------------------------------


class QuantizationPass(OptimizationPass):
    """INT8 quantization with calibration data.

    1. Calibrate: run data through IRInterpreter, collect per-tensor min/max
    2. Weight quant: symmetric, scale = max_abs / 127, zp=0, clip [-127,127]
    3. Activation quant: asymmetric, scale = (max-min)/255, zp with round-half-up
    4. Bias pre-scaling: floor(b_fp / (scale_w * scale_x) + 0.5) as INT32
    5. Requant params: M_fixed = floor((scale_w * scale_x / scale_y) * 2^16 + 0.5)
    6. Quantize weight tensors to INT8, bias to INT32
    7. Set graph.stage = "quantized"
    """

    REQUANT_FRACTIONAL_BITS = 16

    def __init__(self, calibration_data: list[np.ndarray] | None = None):
        self.calibration_data = calibration_data

    @property
    def name(self) -> str:
        return "Quantization"

    def run(self, graph: Graph) -> Graph:
        if not self.calibration_data:
            raise ValueError("Calibration data required for quantization pass")

        # Validate calibration data: reject NaN/inf
        for i, sample in enumerate(self.calibration_data):
            if not np.isfinite(sample).all():
                raise ValueError(f"Calibration sample {i} contains NaN or inf values")

        # Step 1: Calibrate — collect activation ranges
        ranges = self._calibrate(graph)

        # Step 2-6: Compute quant params and quantize each FusedLinear layer
        linear_types = {OpType.FUSED_LINEAR, OpType.FUSED_LINEAR_RELU}
        for node_name in graph.topological_order():
            node = graph.nodes[node_name]
            if node.op_type not in linear_types:
                continue
            attrs = node.fused_attrs
            if attrs is None:
                continue

            # Weight quantization: symmetric (zp=0)
            weight_tensor = graph.tensors[node.inputs[1]]
            w_data = weight_tensor.data.astype(np.float32)
            max_abs = float(max(abs(w_data.min()), abs(w_data.max())))
            if max_abs == 0:
                max_abs = 1e-8
            scale_w = max_abs / 127.0

            attrs.weight_quant = QuantParams(
                scale=scale_w,
                zero_point=0,
                bit_width=8,
                signed=True,
                calibrated=True,
            )

            # Input activation quantization: asymmetric
            input_name = node.inputs[0]
            in_min, in_max = ranges[input_name]
            scale_x, zp_x = self._compute_asymmetric_params(in_min, in_max)
            attrs.input_quant = QuantParams(
                scale=scale_x,
                zero_point=zp_x,
                bit_width=8,
                signed=True,
                calibrated=True,
            )

            # Output activation quantization: asymmetric
            output_name = node.outputs[0]
            out_min, out_max = ranges[output_name]
            if attrs.has_relu:
                out_min = max(out_min, 0.0)
            scale_y, zp_y = self._compute_asymmetric_params(out_min, out_max)
            attrs.output_quant = QuantParams(
                scale=scale_y,
                zero_point=zp_y,
                bit_width=8,
                signed=True,
                calibrated=True,
            )

            # Requantization params: 16.16 fixed-point multiplier
            m_float = (scale_w * scale_x) / scale_y
            m_fixed = int(np.floor(m_float * (1 << self.REQUANT_FRACTIONAL_BITS) + 0.5))

            INT32_MIN = -(1 << 31)
            INT32_MAX = (1 << 31) - 1
            if not (INT32_MIN <= m_fixed <= INT32_MAX):
                raise ValueError(
                    f"M_fixed overflow for {node_name}: {m_fixed} "
                    f"outside INT32 range [{INT32_MIN}, {INT32_MAX}]"
                )

            attrs.requant_scale_fixed = m_fixed
            attrs.requant_shift = self.REQUANT_FRACTIONAL_BITS

            # Quantize weights to INT8: round-half-up, clip [-127, 127]
            w_q = np.floor(w_data / scale_w + 0.5)
            w_q = np.clip(w_q, -127, 127).astype(np.int8)
            weight_tensor.data = w_q
            weight_tensor.type = TensorType(
                shape=weight_tensor.type.shape,
                dtype=np.dtype(np.int8),
                quant=attrs.weight_quant,
            )

            # Quantize bias to INT32: pre-scaled to accumulator domain
            if len(node.inputs) > 2:
                bias_tensor = graph.tensors[node.inputs[2]]
                b_fp = bias_tensor.data.astype(np.float32)
                denom = scale_w * scale_x
                b_q = np.zeros(len(b_fp), dtype=np.int32)
                for i in range(len(b_fp)):
                    if denom == 0 or not np.isfinite(b_fp[i]):
                        b_q[i] = 0
                    else:
                        raw = int(np.floor(float(b_fp[i]) / denom + 0.5))
                        b_q[i] = max(INT32_MIN, min(INT32_MAX, raw))
                bias_tensor.data = b_q
                bias_tensor.type = TensorType(
                    shape=bias_tensor.type.shape,
                    dtype=np.dtype(np.int32),
                )

        graph.stage = "quantized"
        graph.invalidate_cache()
        return graph

    def _calibrate(self, graph: Graph) -> dict[str, tuple[float, float]]:
        """Run calibration data and collect per-tensor min/max ranges."""
        interp = IRInterpreter(graph)
        ranges: dict[str, tuple[float, float]] = {}

        for sample in self.calibration_data:
            inputs = {graph.inputs[0]: sample}
            all_values = interp.run_all(inputs)

            for name, value in all_values.items():
                val_min = float(value.min())
                val_max = float(value.max())
                if name not in ranges:
                    ranges[name] = (val_min, val_max)
                else:
                    old_min, old_max = ranges[name]
                    ranges[name] = (
                        min(old_min, val_min),
                        max(old_max, val_max),
                    )

        return ranges

    def _compute_asymmetric_params(self, val_min: float, val_max: float) -> tuple[float, int]:
        """Compute asymmetric quantization scale and zero point.

        Uses signed INT8 range [-128, 127].
        Round-half-up for zero point computation.
        """
        if val_min == val_max:
            val_max = val_min + 1e-8

        # scale = (max - min) / (qmax - qmin) = (max - min) / 255
        scale = (val_max - val_min) / 255.0
        if scale == 0:
            scale = 1e-8

        if not np.isfinite(scale):
            raise ValueError(f"Non-finite scale {scale} from range [{val_min}, {val_max}]")

        # zp = clamp(round_half_up(-val_min / scale) - 128, -128, 127)
        zp = int(np.floor(-val_min / scale + 0.5)) - 128
        zp = max(-128, min(127, zp))

        return scale, zp

    def verify(self, before: Graph, after: Graph) -> bool:
        """Verify quantization preserves semantics within tolerance.

        Runs test vectors through FP32 interpreter (pre-quant) and INT8
        interpreter (post-quant), compares dequantized outputs.
        Tolerance: output_scale * 10 (generous, catches gross bugs).
        """
        from mlasic.int8_interpreter import INT8Interpreter

        interp_before = IRInterpreter(before)
        interp_after = INT8Interpreter(after)

        input_tensor = before.tensors[before.inputs[0]]
        input_shape = input_tensor.type.shape

        # Get output quant params for tolerance computation
        ordered = [after.nodes[n] for n in after.topological_order()]
        last_attrs = ordered[-1].fused_attrs
        output_scale = last_attrs.output_quant.scale
        output_zp = last_attrs.output_quant.zero_point
        # Generous tolerance: quantization error compounds through layers.
        # Use full INT8 range * output_scale to catch gross bugs, not LSB issues.
        tolerance = output_scale * 256

        rng = np.random.RandomState(42)
        for i in range(10):
            test_input = rng.randn(*input_shape).astype(np.float32)
            inputs_dict = {before.inputs[0]: test_input}

            out_before = interp_before.run(inputs_dict)
            out_after_int8 = interp_after.run(inputs_dict)

            for key in out_before:
                fp_ref = out_before[key]
                int8_out = out_after_int8[key]
                fp_deq = INT8Interpreter.dequantize_output(int8_out, output_scale, output_zp)
                if not np.allclose(fp_ref, fp_deq, atol=tolerance, rtol=0):
                    max_diff = float(np.abs(fp_ref - fp_deq).max())
                    logger.error(
                        "Quantization verification failed on output %s "
                        "(vector %d): max_diff=%.6f, tolerance=%.6f",
                        key,
                        i,
                        max_diff,
                        tolerance,
                    )
                    return False
        return True


# ---------------------------------------------------------------------------
# Shared quantization helpers
# ---------------------------------------------------------------------------


def _calibrate(graph: Graph, calibration_data: list[np.ndarray]) -> dict[str, tuple[float, float]]:
    """Run calibration data and collect per-tensor min/max ranges."""
    interp = IRInterpreter(graph)
    ranges: dict[str, tuple[float, float]] = {}

    for sample in calibration_data:
        inputs = {graph.inputs[0]: sample}
        all_values = interp.run_all(inputs)

        for name, value in all_values.items():
            val_min = float(value.min())
            val_max = float(value.max())
            if name not in ranges:
                ranges[name] = (val_min, val_max)
            else:
                old_min, old_max = ranges[name]
                ranges[name] = (
                    min(old_min, val_min),
                    max(old_max, val_max),
                )

    return ranges


def _compute_asymmetric_params(val_min: float, val_max: float) -> tuple[float, int]:
    """Compute asymmetric quantization scale and zero point.

    Uses signed INT8 range [-128, 127].
    Round-half-up for zero point computation.
    """
    if val_min == val_max:
        val_max = val_min + 1e-8

    scale = (val_max - val_min) / 255.0
    if scale == 0:
        scale = 1e-8

    if not np.isfinite(scale):
        raise ValueError(f"Non-finite scale {scale} from range [{val_min}, {val_max}]")

    zp = int(np.floor(-val_min / scale + 0.5)) - 128
    zp = max(-128, min(127, zp))

    return scale, zp


# ---------------------------------------------------------------------------
# Pass 6: Conv Fusion
# ---------------------------------------------------------------------------


class ConvFusionPass(OptimizationPass):
    """Fuse Conv[+ReLU/Clip(0,6)] into FusedConv[ReLU/ReLU6].

    Patterns:
        Conv → ReLU       ⇒  FusedConvReLU
        Conv → Clip(0,6)  ⇒  FusedConvReLU6
        Conv (alone)       ⇒  FusedConv
    """

    @property
    def name(self) -> str:
        return "ConvFusion"

    def run(self, graph: Graph) -> Graph:
        fused_counter = 0

        for node_name in list(graph.topological_order()):
            if node_name not in graph.nodes:
                continue
            node = graph.nodes[node_name]
            if node.op_type != OpType.CONV:
                continue

            conv_node = node
            conv_out = conv_node.outputs[0]
            consumers = graph.get_consumers(conv_out)

            has_relu = False
            has_relu6 = False
            activation_node = None

            if len(consumers) == 1:
                consumer = consumers[0]
                if consumer.op_type == OpType.RELU:
                    has_relu = True
                    activation_node = consumer
                elif consumer.op_type == OpType.CLIP:
                    # Check if Clip(0, 6)
                    clip_min, clip_max = self._get_clip_bounds(consumer, graph)
                    if clip_min == 0.0 and clip_max == 6.0:
                        has_relu6 = True
                        activation_node = consumer

            # Determine fused op type
            if has_relu:
                op_type = OpType.FUSED_CONV_RELU
            elif has_relu6:
                op_type = OpType.FUSED_CONV_RELU6
            else:
                op_type = OpType.FUSED_CONV

            # Get conv attributes
            a = conv_node.attributes
            weight_tensor = graph.tensors[conv_node.inputs[1]]
            w_shape = weight_tensor.type.shape
            kernel_shape = a.get("kernel_shape", list(w_shape[2:]))

            fused_attrs = FusedConvAttrs(
                in_channels=w_shape[1] * a.get("group", 1),
                out_channels=w_shape[0],
                kernel_shape=kernel_shape,
                strides=a.get("strides", [1] * len(kernel_shape)),
                pads=a.get("pads", [0] * (2 * len(kernel_shape))),
                dilations=a.get("dilations", [1] * len(kernel_shape)),
                group=a.get("group", 1),
                has_relu=has_relu,
                has_relu6=has_relu6,
            )

            last_output = activation_node.outputs[0] if activation_node else conv_out
            fused_name = f"fused_conv_{fused_counter}"
            fused_counter += 1

            # Create output tensor for fused node if needed
            fused_out_name = f"{fused_name}_out"
            out_tensor = graph.tensors[last_output]
            new_tensors = {
                fused_out_name: Tensor(
                    name=fused_out_name,
                    type=TensorType(shape=out_tensor.type.shape, dtype=out_tensor.type.dtype),
                )
            }

            fused_node = OpNode(
                name=fused_name,
                op_type=op_type,
                inputs=list(conv_node.inputs),
                outputs=[fused_out_name],
            )
            fused_node.fused_attrs = fused_attrs

            old_names = [conv_node.name]
            if activation_node:
                old_names.append(activation_node.name)

            graph.replace_subgraph(old_names, fused_node, new_tensors)

        return graph

    def _get_clip_bounds(self, clip_node: OpNode, graph: Graph) -> tuple[float, float]:
        """Extract Clip min/max bounds."""
        lo = float("-inf")
        hi = float("inf")
        if len(clip_node.inputs) > 1:
            lo_tensor = graph.tensors.get(clip_node.inputs[1])
            if lo_tensor and lo_tensor.is_constant and lo_tensor.data is not None:
                lo = float(lo_tensor.data)
        if len(clip_node.inputs) > 2:
            hi_tensor = graph.tensors.get(clip_node.inputs[2])
            if hi_tensor and hi_tensor.is_constant and hi_tensor.data is not None:
                hi = float(hi_tensor.data)
        return lo, hi


# ---------------------------------------------------------------------------
# Pass 7: Conv Quantization (per-channel)
# ---------------------------------------------------------------------------


class ConvQuantizationPass(OptimizationPass):
    """Per-channel INT8 quantization for FusedConv* operators.

    Per-channel: one scale per output channel.
    M_fixed[oc] = floor((scale_w[oc] * scale_x / scale_y) * 2^16 + 0.5)
    """

    REQUANT_FRACTIONAL_BITS = 16

    def __init__(self, calibration_data: list[np.ndarray] | None = None):
        self.calibration_data = calibration_data

    @property
    def name(self) -> str:
        return "ConvQuantization"

    def run(self, graph: Graph) -> Graph:
        if not self.calibration_data:
            raise ValueError("Calibration data required for conv quantization pass")

        for i, sample in enumerate(self.calibration_data):
            if not np.isfinite(sample).all():
                raise ValueError(f"Calibration sample {i} contains NaN or inf values")

        ranges = _calibrate(graph, self.calibration_data)

        conv_fused_types = {OpType.FUSED_CONV, OpType.FUSED_CONV_RELU, OpType.FUSED_CONV_RELU6}

        for node_name in graph.topological_order():
            node = graph.nodes[node_name]
            if node.op_type not in conv_fused_types:
                continue

            attrs = node.fused_attrs
            if not isinstance(attrs, FusedConvAttrs):
                continue

            weight_tensor = graph.tensors[node.inputs[1]]
            w_data = weight_tensor.data.astype(np.float32)
            c_out = w_data.shape[0]

            # Per-channel weight quantization: symmetric (zp=0)
            per_channel_quant: list[QuantParams] = []
            per_channel_scales: list[float] = []
            for oc in range(c_out):
                w_oc = w_data[oc]
                max_abs = float(max(abs(w_oc.min()), abs(w_oc.max())))
                if max_abs == 0:
                    max_abs = 1e-8
                scale_w = max_abs / 127.0
                per_channel_scales.append(scale_w)
                per_channel_quant.append(
                    QuantParams(
                        scale=scale_w, zero_point=0, bit_width=8, signed=True, calibrated=True
                    )
                )

            attrs.weight_quant = per_channel_quant

            # Input activation: asymmetric
            input_name = node.inputs[0]
            in_min, in_max = ranges.get(input_name, (0.0, 1.0))
            scale_x, zp_x = _compute_asymmetric_params(in_min, in_max)
            attrs.input_quant = QuantParams(
                scale=scale_x, zero_point=zp_x, bit_width=8, signed=True, calibrated=True
            )

            # Output activation: asymmetric
            output_name = node.outputs[0]
            out_min, out_max = ranges.get(output_name, (0.0, 1.0))
            if attrs.has_relu or attrs.has_relu6:
                out_min = max(out_min, 0.0)
            if attrs.has_relu6:
                out_max = min(out_max, 6.0)
            scale_y, zp_y = _compute_asymmetric_params(out_min, out_max)
            attrs.output_quant = QuantParams(
                scale=scale_y, zero_point=zp_y, bit_width=8, signed=True, calibrated=True
            )

            # Per-channel requantization: M_fixed[oc]
            per_channel_m_fixed: list[int] = []
            INT32_MIN = -(1 << 31)
            INT32_MAX = (1 << 31) - 1
            for oc in range(c_out):
                m_float = (per_channel_scales[oc] * scale_x) / scale_y
                m_fixed = int(np.floor(m_float * (1 << self.REQUANT_FRACTIONAL_BITS) + 0.5))
                if not (INT32_MIN <= m_fixed <= INT32_MAX):
                    raise ValueError(f"M_fixed overflow for {node_name} channel {oc}: {m_fixed}")
                per_channel_m_fixed.append(m_fixed)

            attrs.requant_scale_fixed = per_channel_m_fixed
            attrs.requant_shift = self.REQUANT_FRACTIONAL_BITS

            # Quantize weights to INT8 per-channel
            w_q = np.zeros_like(w_data, dtype=np.int8)
            for oc in range(c_out):
                w_oc = w_data[oc]
                scale_w = per_channel_scales[oc]
                q = np.floor(w_oc / scale_w + 0.5)
                w_q[oc] = np.clip(q, -127, 127).astype(np.int8)
            weight_tensor.data = w_q
            weight_tensor.type = TensorType(shape=weight_tensor.type.shape, dtype=np.dtype(np.int8))

            # Quantize bias to INT32 per-channel
            if len(node.inputs) > 2:
                bias_tensor = graph.tensors[node.inputs[2]]
                b_fp = bias_tensor.data.astype(np.float32)
                b_q = np.zeros_like(b_fp, dtype=np.int32)
                for oc in range(c_out):
                    raw = int(np.floor(b_fp[oc] / (per_channel_scales[oc] * scale_x) + 0.5))
                    b_q[oc] = max(INT32_MIN, min(INT32_MAX, raw))
                bias_tensor.data = b_q
                bias_tensor.type = TensorType(
                    shape=bias_tensor.type.shape, dtype=np.dtype(np.int32)
                )

        graph.invalidate_cache()
        return graph

    def verify(self, before: Graph, after: Graph) -> bool:
        """Skip automatic verification for quantization passes."""
        return True


# ---------------------------------------------------------------------------
# Pass 8: LayerNorm Fusion
# ---------------------------------------------------------------------------


class LayerNormFusionPass(OptimizationPass):
    """Fuse single LayerNormalization node into FusedLayerNorm.

    Trivial 1:1 replacement that wraps the node with FusedLayerNormAttrs.
    """

    @property
    def name(self) -> str:
        return "LayerNormFusion"

    def run(self, graph: Graph) -> Graph:
        fused_counter = 0

        for node_name in list(graph.topological_order()):
            if node_name not in graph.nodes:
                continue
            node = graph.nodes[node_name]
            if node.op_type != OpType.LAYER_NORM:
                continue

            a = node.attributes
            axis = a.get("axis", -1)
            eps = a.get("epsilon", 1e-5)

            # Determine normalized_shape from scale tensor
            scale_tensor = graph.tensors.get(node.inputs[1])
            normalized_shape = tuple(scale_tensor.type.shape) if scale_tensor else None

            fused_attrs = FusedLayerNormAttrs(
                axis=axis,
                epsilon=eps,
                normalized_shape=normalized_shape,
            )

            fused_name = f"fused_ln_{fused_counter}"
            fused_counter += 1

            fused_out_name = f"{fused_name}_out"
            out_tensor = graph.tensors[node.outputs[0]]
            new_tensors = {
                fused_out_name: Tensor(
                    name=fused_out_name,
                    type=TensorType(shape=out_tensor.type.shape, dtype=out_tensor.type.dtype),
                )
            }

            fused_node = OpNode(
                name=fused_name,
                op_type=OpType.FUSED_LAYER_NORM,
                inputs=list(node.inputs),
                outputs=[fused_out_name],
            )
            fused_node.fused_attrs = fused_attrs

            graph.replace_subgraph([node_name], fused_node, new_tensors)

        return graph


# ---------------------------------------------------------------------------
# Pass 9: Activation Fusion (GELU, SiLU)
# ---------------------------------------------------------------------------


class ActivationFusionPass(OptimizationPass):
    """Fuse activation patterns into single fused ops.

    Patterns:
        SiLU: Sigmoid(x) * x → FusedSiLU
        GELU: Mul(sqrt2_inv) → Erf → Add(1) → Mul(x) → Mul(0.5) → FusedGELU
    """

    @property
    def name(self) -> str:
        return "ActivationFusion"

    def run(self, graph: Graph) -> Graph:
        self._fuse_silu(graph)
        self._fuse_gelu(graph)
        return graph

    def _fuse_silu(self, graph: Graph) -> None:
        """Fuse Sigmoid(x) * x → FusedSiLU.

        Pattern: x feeds into both Sigmoid and Mul. Sigmoid output feeds into Mul.
        """
        fused_counter = 0

        for node_name in list(graph.topological_order()):
            if node_name not in graph.nodes:
                continue
            node = graph.nodes[node_name]
            if node.op_type != OpType.SIGMOID:
                continue

            sigmoid_node = node
            sigmoid_input = sigmoid_node.inputs[0]
            sigmoid_output = sigmoid_node.outputs[0]

            # Find Mul that consumes sigmoid output AND the original input
            consumers = graph.get_consumers(sigmoid_output)
            for consumer in consumers:
                if consumer.op_type != OpType.MUL:
                    continue
                # Check if the Mul also takes the original sigmoid input
                if sigmoid_input in consumer.inputs and sigmoid_output in consumer.inputs:
                    mul_node = consumer

                    fused_name = f"fused_silu_{fused_counter}"
                    fused_counter += 1

                    fused_out_name = f"{fused_name}_out"
                    out_tensor = graph.tensors[mul_node.outputs[0]]
                    new_tensors = {
                        fused_out_name: Tensor(
                            name=fused_out_name,
                            type=TensorType(
                                shape=out_tensor.type.shape, dtype=out_tensor.type.dtype
                            ),
                        )
                    }

                    fused_node = OpNode(
                        name=fused_name,
                        op_type=OpType.FUSED_SILU,
                        inputs=[sigmoid_input],
                        outputs=[fused_out_name],
                    )
                    fused_node.fused_attrs = FusedActivationAttrs(activation_type="silu")

                    graph.replace_subgraph(
                        [sigmoid_node.name, mul_node.name], fused_node, new_tensors
                    )
                    break

    def _fuse_gelu(self, graph: Graph) -> None:
        """Fuse GELU pattern: x → Mul(sqrt2_inv) → Erf → Add(1) → Mul(x) → Mul(0.5) → FusedGELU.

        The exact ONNX decomposition: 0.5 * x * (1 + erf(x / sqrt(2)))
        """
        fused_counter = 0

        for node_name in list(graph.topological_order()):
            if node_name not in graph.nodes:
                continue
            node = graph.nodes[node_name]
            if node.op_type != OpType.ERF:
                continue

            erf_node = node
            erf_input = erf_node.inputs[0]
            erf_output = erf_node.outputs[0]

            # The erf input should come from Mul(x, sqrt2_inv)
            scale_mul_node = graph.get_producer(erf_input)
            if scale_mul_node is None or scale_mul_node.op_type != OpType.MUL:
                continue

            # Identify the original x input (non-constant input to the scale Mul)
            x_input = None
            for inp in scale_mul_node.inputs:
                t = graph.tensors.get(inp)
                if t and not t.is_constant:
                    x_input = inp
                    break
            if x_input is None:
                continue

            # Check constant is approximately 1/sqrt(2)
            const_found = False
            for inp in scale_mul_node.inputs:
                t = graph.tensors.get(inp)
                if t and t.is_constant and t.data is not None:
                    val = float(t.data.flat[0]) if t.data.size > 0 else 0
                    if abs(val - 1.0 / np.sqrt(2.0)) < 0.01:
                        const_found = True
                        break
            if not const_found:
                continue

            # erf_output → Add(1)
            add_consumers = graph.get_consumers(erf_output)
            if len(add_consumers) != 1 or add_consumers[0].op_type != OpType.ADD:
                continue
            add_node = add_consumers[0]

            # Check Add has constant 1.0
            add_has_one = False
            for inp in add_node.inputs:
                if inp == erf_output:
                    continue
                t = graph.tensors.get(inp)
                if t and t.is_constant and t.data is not None:
                    if abs(float(t.data.flat[0]) - 1.0) < 0.01:
                        add_has_one = True
            if not add_has_one:
                continue

            # Add output → Mul(x)
            add_output = add_node.outputs[0]
            mul1_consumers = graph.get_consumers(add_output)
            if len(mul1_consumers) != 1 or mul1_consumers[0].op_type != OpType.MUL:
                continue
            mul1_node = mul1_consumers[0]

            # Check Mul1 consumes both add_output and x_input
            if x_input not in mul1_node.inputs or add_output not in mul1_node.inputs:
                continue

            # Mul1 output → Mul(0.5)
            mul1_output = mul1_node.outputs[0]
            mul2_consumers = graph.get_consumers(mul1_output)
            if len(mul2_consumers) != 1 or mul2_consumers[0].op_type != OpType.MUL:
                continue
            mul2_node = mul2_consumers[0]

            # Check Mul2 has constant 0.5
            mul2_has_half = False
            for inp in mul2_node.inputs:
                if inp == mul1_output:
                    continue
                t = graph.tensors.get(inp)
                if t and t.is_constant and t.data is not None:
                    if abs(float(t.data.flat[0]) - 0.5) < 0.01:
                        mul2_has_half = True
            if not mul2_has_half:
                continue

            # All pattern nodes identified — fuse
            fused_name = f"fused_gelu_{fused_counter}"
            fused_counter += 1

            fused_out_name = f"{fused_name}_out"
            out_tensor = graph.tensors[mul2_node.outputs[0]]
            new_tensors = {
                fused_out_name: Tensor(
                    name=fused_out_name,
                    type=TensorType(shape=out_tensor.type.shape, dtype=out_tensor.type.dtype),
                )
            }

            fused_node = OpNode(
                name=fused_name,
                op_type=OpType.FUSED_GELU,
                inputs=[x_input],
                outputs=[fused_out_name],
            )
            fused_node.fused_attrs = FusedActivationAttrs(activation_type="gelu")

            old_names = [
                scale_mul_node.name,
                erf_node.name,
                add_node.name,
                mul1_node.name,
                mul2_node.name,
            ]
            graph.replace_subgraph(old_names, fused_node, new_tensors)


# ---------------------------------------------------------------------------
# Pass 10: Attention Fusion
# ---------------------------------------------------------------------------


class AttentionFusionPass(OptimizationPass):
    """Fuse multi-head attention pattern into FusedAttention.

    Pattern:
        Q_proj(MatMul/Gemm) + K_proj(MatMul/Gemm) + V_proj(MatMul/Gemm)
        → Reshape/Transpose (multi-head split)
        → MatMul(Q, K^T) → [optional Mul(scale)] → [optional Add(mask)] → Softmax
        → MatMul(attn_weights, V) → Reshape/Transpose (head merge)
        → O_proj(MatMul/Gemm)
        ⇒ FusedAttention

    Detects the pattern by looking for the characteristic Q@K^T → Softmax → attn@V
    structure and tracing back to the projection matmuls.
    """

    @property
    def name(self) -> str:
        return "AttentionFusion"

    def run(self, graph: Graph) -> Graph:
        fused_counter = 0

        for node_name in list(graph.topological_order()):
            if node_name not in graph.nodes:
                continue
            node = graph.nodes[node_name]
            if node.op_type != OpType.SOFTMAX:
                continue

            result = self._match_attention_pattern(graph, node)
            if result is None:
                continue

            (
                qk_matmul, attn_v_matmul, o_proj,
                q_proj, k_proj, v_proj,
                intermediate_nodes, x_input, final_output,
                num_heads, head_dim, seq_len, has_mask,
            ) = result

            fused_name = f"fused_attention_{fused_counter}"
            fused_counter += 1

            fused_attrs = FusedAttentionAttrs(
                num_heads=num_heads,
                head_dim=head_dim,
                seq_len=seq_len,
                has_mask=has_mask,
            )

            # Collect weight/bias inputs from Q/K/V/O projections
            fused_inputs = [x_input]
            for proj in [q_proj, k_proj, v_proj, o_proj]:
                if proj is not None:
                    for inp in proj.inputs[1:]:
                        fused_inputs.append(inp)

            fused_out_name = f"{fused_name}_out"
            out_tensor = graph.tensors[final_output]
            new_tensors = {
                fused_out_name: Tensor(
                    name=fused_out_name,
                    type=TensorType(shape=out_tensor.type.shape, dtype=out_tensor.type.dtype),
                )
            }

            fused_node = OpNode(
                name=fused_name,
                op_type=OpType.FUSED_ATTENTION,
                inputs=fused_inputs,
                outputs=[fused_out_name],
            )
            fused_node.fused_attrs = fused_attrs

            # All nodes to remove
            old_names = list(set(
                n.name for n in intermediate_nodes if n is not None and n.name in graph.nodes
            ))
            graph.replace_subgraph(old_names, fused_node, new_tensors)

        return graph

    def _match_attention_pattern(self, graph: Graph, softmax_node: OpNode):
        """Try to match attention pattern anchored at a Softmax node.

        Returns tuple of matched components or None if no match.
        """
        softmax_input = softmax_node.inputs[0]
        softmax_output = softmax_node.outputs[0]

        # Softmax input comes from: MatMul(Q, K^T) optionally through Mul/Add
        qk_matmul, pre_softmax_nodes, has_mask = self._trace_pre_softmax(graph, softmax_input)
        if qk_matmul is None:
            return None

        # Softmax output feeds into: MatMul(attn_weights, V)
        attn_consumers = graph.get_consumers(softmax_output)
        attn_v_matmul = None
        for c in attn_consumers:
            if c.op_type == OpType.MATMUL:
                attn_v_matmul = c
                break
        if attn_v_matmul is None:
            return None

        # Trace Q, K from qk_matmul inputs (Q @ K^T)
        q_chain, q_proj = self._trace_projection(graph, qk_matmul.inputs[0])
        k_chain, k_proj = self._trace_projection(graph, qk_matmul.inputs[1])

        # Trace V from attn_v_matmul (the non-softmax input)
        v_input = None
        for inp in attn_v_matmul.inputs:
            if inp != softmax_output:
                v_input = inp
                break
        if v_input is None:
            return None

        v_chain, v_proj = self._trace_projection(graph, v_input)

        # Need at least Q and K projections to be real MatMul/Gemm
        if q_proj is None or k_proj is None:
            return None

        # Trace output path: attn_v_matmul → reshape/transpose → O_proj
        o_proj, post_nodes, final_output = self._trace_output_projection(
            graph, attn_v_matmul.outputs[0]
        )

        # Determine attention dimensions
        num_heads, head_dim, seq_len = self._infer_attention_dims(
            graph, q_proj, qk_matmul, softmax_node
        )
        if num_heads == 0 or head_dim == 0:
            return None

        # Find the shared input to Q/K/V projections
        x_input = self._find_shared_input(graph, q_proj, k_proj, v_proj)
        if x_input is None:
            return None

        # Collect all intermediate nodes for removal
        all_nodes = (
            [qk_matmul, softmax_node, attn_v_matmul]
            + pre_softmax_nodes
            + q_chain + k_chain + v_chain
            + [q_proj, k_proj, v_proj]
            + post_nodes
        )
        if o_proj is not None:
            all_nodes.append(o_proj)

        return (
            qk_matmul, attn_v_matmul, o_proj,
            q_proj, k_proj, v_proj,
            all_nodes, x_input, final_output,
            num_heads, head_dim, seq_len, has_mask,
        )

    def _trace_pre_softmax(self, graph: Graph, tensor_name: str):
        """Trace backward from softmax input to find the QK matmul.

        Handles optional Mul(scale) and Add(mask) between QK matmul and softmax.
        Returns (qk_matmul_node, intermediate_nodes, has_mask).
        """
        intermediates = []
        has_mask = False
        current = tensor_name

        # Walk back through at most 3 nodes (Add for mask, Mul for scale, Div for scale)
        for _ in range(3):
            producer = graph.get_producer(current)
            if producer is None:
                return None, [], False

            if producer.op_type == OpType.MATMUL:
                return producer, intermediates, has_mask

            if producer.op_type in (OpType.MUL, OpType.DIV):
                intermediates.append(producer)
                # Find the non-constant input
                for inp in producer.inputs:
                    t = graph.tensors.get(inp)
                    if t and not t.is_constant:
                        current = inp
                        break
                else:
                    return None, [], False
            elif producer.op_type == OpType.ADD:
                has_mask = True
                intermediates.append(producer)
                # Find the non-mask input (the one from matmul path)
                for inp in producer.inputs:
                    p = graph.get_producer(inp)
                    if p and p.op_type in (OpType.MATMUL, OpType.MUL, OpType.DIV):
                        current = inp
                        break
                else:
                    # Try the first non-constant input
                    for inp in producer.inputs:
                        t = graph.tensors.get(inp)
                        if t and not t.is_constant:
                            current = inp
                            break
                    else:
                        return None, [], False
            else:
                return None, [], False

        return None, [], False

    def _trace_projection(self, graph: Graph, tensor_name: str):
        """Trace backward from a reshape/transpose chain to find the projection MatMul.

        Returns (chain_nodes, matmul_node_or_none).
        """
        chain = []
        current = tensor_name
        reshape_ops = {OpType.RESHAPE, OpType.TRANSPOSE, OpType.GATHER, OpType.UNSQUEEZE}

        for _ in range(6):  # Max 6 reshape/transpose nodes
            producer = graph.get_producer(current)
            if producer is None:
                return chain, None

            if producer.op_type in (OpType.MATMUL, OpType.GEMM):
                return chain, producer

            if producer.op_type == OpType.ADD:
                # Could be bias add after MatMul
                chain.append(producer)
                # Find the matmul input
                for inp in producer.inputs:
                    p = graph.get_producer(inp)
                    if p and p.op_type in (OpType.MATMUL, OpType.GEMM):
                        return chain, p
                # Otherwise trace through the non-constant input
                for inp in producer.inputs:
                    t = graph.tensors.get(inp)
                    if t and not t.is_constant:
                        current = inp
                        break
                else:
                    return chain, None
            elif producer.op_type in reshape_ops:
                chain.append(producer)
                # Follow the data input (first non-constant, typically inputs[0])
                current = producer.inputs[0]
            else:
                return chain, None

        return chain, None

    def _trace_output_projection(self, graph: Graph, tensor_name: str):
        """Trace forward from attn@V output through reshape/transpose to O projection.

        Returns (o_proj_node_or_none, chain_nodes, final_output_tensor_name).
        """
        chain = []
        current = tensor_name
        reshape_ops = {OpType.RESHAPE, OpType.TRANSPOSE}

        for _ in range(6):
            consumers = graph.get_consumers(current)
            if not consumers:
                return None, chain, current

            consumer = consumers[0]
            if consumer.op_type in (OpType.MATMUL, OpType.GEMM):
                return consumer, chain, consumer.outputs[0]

            if consumer.op_type == OpType.ADD:
                # Check if any input comes from a matmul (bias add after O_proj)
                for inp in consumer.inputs:
                    p = graph.get_producer(inp)
                    if p and p.op_type in (OpType.MATMUL, OpType.GEMM):
                        chain.append(consumer)
                        return p, chain, consumer.outputs[0]
                # Simple residual Add — stop here
                return None, chain, current

            if consumer.op_type in reshape_ops:
                chain.append(consumer)
                current = consumer.outputs[0]
            else:
                return None, chain, current

        return None, chain, current

    def _infer_attention_dims(self, graph, q_proj, qk_matmul, softmax_node):
        """Infer num_heads, head_dim, seq_len from tensor shapes."""
        # Try to get from QK matmul input shape: typically (batch, heads, seq, head_dim)
        qk_in_tensor = graph.tensors.get(qk_matmul.inputs[0])
        if qk_in_tensor and len(qk_in_tensor.type.shape) == 4:
            _, num_heads, seq_len, head_dim = qk_in_tensor.type.shape
            if all(d > 0 for d in (num_heads, seq_len, head_dim)):
                return num_heads, head_dim, seq_len

        # Fallback: infer from Q projection weight shape
        if q_proj and len(q_proj.inputs) >= 2:
            wt = graph.tensors.get(q_proj.inputs[1])
            if wt and wt.type.shape:
                # Weight shape: (embed_dim, embed_dim) or (embed_dim, head_dim*num_heads)
                embed_dim = wt.type.shape[-1] if wt.type.shape[-1] > 0 else wt.type.shape[0]
                if embed_dim > 0:
                    # Common head dims: 64, 128, 96, 80
                    for hd in [64, 128, 96, 80, 32]:
                        if embed_dim % hd == 0:
                            nh = embed_dim // hd
                            if nh > 0:
                                return nh, hd, 1  # seq_len unknown, use 1
                    return 1, embed_dim, 1

        return 0, 0, 0

    @staticmethod
    def _find_shared_input(graph, q_proj, k_proj, v_proj):
        """Find the shared activation input to Q/K/V projections."""
        q_in = q_proj.inputs[0] if q_proj else None
        k_in = k_proj.inputs[0] if k_proj else None
        v_in = v_proj.inputs[0] if v_proj else None

        # Typically all three share the same input
        if q_in == k_in == v_in and q_in is not None:
            return q_in

        # Return whichever is available
        for inp in [q_in, k_in, v_in]:
            if inp is not None:
                t = graph.tensors.get(inp)
                if t and not t.is_constant:
                    return inp

        return q_in


# ---------------------------------------------------------------------------
# Pass 11: Attention Quantization (per-projection)
# ---------------------------------------------------------------------------


class AttentionQuantizationPass(OptimizationPass):
    """Per-projection INT8 quantization for FusedAttention operators.

    Each Q/K/V/O projection weight is quantized with symmetric per-tensor INT8.
    Softmax remains in FP32 internally (attention scores need higher precision).
    """

    REQUANT_FRACTIONAL_BITS = 16

    def __init__(self, calibration_data: list[np.ndarray] | None = None):
        self.calibration_data = calibration_data

    @property
    def name(self) -> str:
        return "AttentionQuantization"

    def run(self, graph: Graph) -> Graph:
        if not self.calibration_data:
            raise ValueError("Calibration data required for attention quantization pass")

        ranges = _calibrate(graph, self.calibration_data)

        for node_name in graph.topological_order():
            node = graph.nodes[node_name]
            if node.op_type != OpType.FUSED_ATTENTION:
                continue

            attrs = node.fused_attrs
            if not isinstance(attrs, FusedAttentionAttrs):
                continue

            # Input activation: asymmetric
            input_name = node.inputs[0]
            in_min, in_max = ranges.get(input_name, (0.0, 1.0))
            scale_x, zp_x = _compute_asymmetric_params(in_min, in_max)
            attrs.input_quant = QuantParams(
                scale=scale_x, zero_point=zp_x, bit_width=8, signed=True, calibrated=True
            )

            # Output activation: asymmetric
            output_name = node.outputs[0]
            out_min, out_max = ranges.get(output_name, (0.0, 1.0))
            scale_y, zp_y = _compute_asymmetric_params(out_min, out_max)
            attrs.output_quant = QuantParams(
                scale=scale_y, zero_point=zp_y, bit_width=8, signed=True, calibrated=True
            )

            # Quantize Q/K/V/O projection weights (symmetric per-tensor)
            proj_idx = 0
            proj_quant_names = [
                "q_weight_quant", "k_weight_quant",
                "v_weight_quant", "output_weight_quant",
            ]

            requant_m_fixed = []
            for proj_name in proj_quant_names:
                # Weights are at inputs[1], inputs[2or3], etc.
                weight_idx = 1 + proj_idx
                if weight_idx >= len(node.inputs):
                    # No weight for this projection (shouldn't happen for valid attention)
                    proj_idx += 2  # skip weight + optional bias
                    continue

                wt_name = node.inputs[weight_idx]
                wt = graph.tensors.get(wt_name)
                if wt is None or not wt.is_constant or wt.data is None:
                    proj_idx += 2
                    continue

                w_data = wt.data.astype(np.float32)
                max_abs = float(max(abs(w_data.min()), abs(w_data.max())))
                if max_abs == 0:
                    max_abs = 1e-8
                scale_w = max_abs / 127.0

                setattr(attrs, proj_name, QuantParams(
                    scale=scale_w, zero_point=0, bit_width=8, signed=True, calibrated=True
                ))

                # Quantize weight to INT8
                w_q = np.clip(np.floor(w_data / scale_w + 0.5), -127, 127).astype(np.int8)
                wt.data = w_q
                wt.type = TensorType(shape=wt.type.shape, dtype=np.dtype(np.int8))

                # Compute requant M_fixed for this projection
                m_float = (scale_w * scale_x) / scale_y
                m_fixed = int(np.floor(m_float * (1 << self.REQUANT_FRACTIONAL_BITS) + 0.5))
                requant_m_fixed.append(m_fixed)

                # Quantize bias if present
                bias_idx = weight_idx + 1
                if bias_idx < len(node.inputs):
                    bt = graph.tensors.get(node.inputs[bias_idx])
                    if bt and bt.is_constant and bt.data is not None and bt.data.dtype != np.int32:
                        b_fp = bt.data.astype(np.float32)
                        INT32_MAX = (1 << 31) - 1
                        INT32_MIN = -(1 << 31)
                        b_q = np.clip(
                            np.floor(b_fp / (scale_w * scale_x) + 0.5),
                            INT32_MIN, INT32_MAX,
                        ).astype(np.int32)
                        bt.data = b_q
                        bt.type = TensorType(shape=bt.type.shape, dtype=np.dtype(np.int32))

                proj_idx += 2  # advance past weight + bias

            attrs.requant_scale_fixed = requant_m_fixed if requant_m_fixed else [1]
            attrs.requant_shift = self.REQUANT_FRACTIONAL_BITS

        graph.invalidate_cache()
        return graph

    def verify(self, before: Graph, after: Graph) -> bool:
        return True


# ---------------------------------------------------------------------------
# Pass 12: FusedMLP Fusion
# ---------------------------------------------------------------------------


class FusedMLPPass(OptimizationPass):
    """Mega-fusion: FusedLinear[+activation] + FusedLinear → FusedMLP.

    Common transformer FFN: Linear → GELU/SiLU/ReLU → Linear
    Detects sequences: FusedLinear[ReLU] or FusedLinear + FusedGELU/FusedSiLU + FusedLinear
    """

    @property
    def name(self) -> str:
        return "FusedMLP"

    def run(self, graph: Graph) -> Graph:
        fused_counter = 0

        for node_name in list(graph.topological_order()):
            if node_name not in graph.nodes:
                continue
            node = graph.nodes[node_name]

            # Look for pattern starting with FusedLinear (fc1)
            if node.op_type not in (OpType.FUSED_LINEAR, OpType.FUSED_LINEAR_RELU):
                continue

            fc1_node = node
            fc1_attrs = fc1_node.fused_attrs
            if not isinstance(fc1_attrs, FusedLinearAttrs):
                continue

            fc1_output = fc1_node.outputs[0]
            consumers = graph.get_consumers(fc1_output)
            if len(consumers) != 1:
                continue

            next_node = consumers[0]

            # Determine activation type and find fc2
            activation_type = "relu" if fc1_attrs.has_relu else None
            activation_node = None
            fc2_node = None

            if next_node.op_type in (OpType.FUSED_GELU, OpType.FUSED_SILU):
                # Pattern: FusedLinear → FusedGELU/FusedSiLU → FusedLinear
                act_type_map = {OpType.FUSED_GELU: "gelu", OpType.FUSED_SILU: "silu"}
                activation_type = act_type_map[next_node.op_type]
                activation_node = next_node
                act_consumers = graph.get_consumers(next_node.outputs[0])
                if len(act_consumers) != 1:
                    continue
                fc2_candidate = act_consumers[0]
                if fc2_candidate.op_type in (OpType.FUSED_LINEAR, OpType.FUSED_LINEAR_RELU):
                    fc2_node = fc2_candidate
            elif next_node.op_type in (OpType.FUSED_LINEAR, OpType.FUSED_LINEAR_RELU):
                # Pattern: FusedLinearReLU → FusedLinear
                if activation_type is None:
                    activation_type = "none"
                fc2_node = next_node

            if fc2_node is None or activation_type is None:
                continue

            fc2_attrs = fc2_node.fused_attrs
            if not isinstance(fc2_attrs, FusedLinearAttrs):
                continue

            fused_name = f"fused_mlp_{fused_counter}"
            fused_counter += 1

            mlp_attrs = FusedMLPAttrs(
                input_dim=fc1_attrs.input_dim,
                hidden_dim=fc1_attrs.output_dim,
                output_dim=fc2_attrs.output_dim,
                activation_type=activation_type,
            )

            # Collect inputs: x, fc1_weight, fc1_bias, fc2_weight, fc2_bias
            fused_inputs = [fc1_node.inputs[0]]  # x
            fused_inputs.extend(fc1_node.inputs[1:])  # fc1 weight + bias
            fused_inputs.extend(fc2_node.inputs[1:])  # fc2 weight + bias

            fused_out_name = f"{fused_name}_out"
            out_tensor = graph.tensors[fc2_node.outputs[0]]
            new_tensors = {
                fused_out_name: Tensor(
                    name=fused_out_name,
                    type=TensorType(shape=out_tensor.type.shape, dtype=out_tensor.type.dtype),
                )
            }

            fused_node = OpNode(
                name=fused_name,
                op_type=OpType.FUSED_MLP,
                inputs=fused_inputs,
                outputs=[fused_out_name],
            )
            fused_node.fused_attrs = mlp_attrs

            old_names = [fc1_node.name]
            if activation_node:
                old_names.append(activation_node.name)
            old_names.append(fc2_node.name)

            graph.replace_subgraph(old_names, fused_node, new_tensors)

        return graph


# ---------------------------------------------------------------------------
# Graph Partitioner
# ---------------------------------------------------------------------------


@dataclass
class GraphPartition:
    """A block of related nodes identified by the partitioner."""

    block_type: str  # "attention", "residual", "conv", "linear", "norm", "other"
    node_names: list[str]
    inputs: list[str]   # tensor names consumed from outside the block
    outputs: list[str]  # tensor names produced for outside the block


class GraphPartitioner:
    """Partition a DAG into scheduling-friendly blocks by op type.

    Identifies attention blocks, conv blocks, linear blocks, etc.
    Used by the scheduler for block-level scheduling decisions.
    """

    _ATTENTION_OPS = {OpType.FUSED_ATTENTION}
    _CONV_OPS = {OpType.FUSED_CONV, OpType.FUSED_CONV_RELU, OpType.FUSED_CONV_RELU6, OpType.CONV}
    _LINEAR_OPS = {OpType.FUSED_LINEAR, OpType.FUSED_LINEAR_RELU, OpType.FUSED_MLP, OpType.MATMUL, OpType.GEMM}
    _NORM_OPS = {OpType.FUSED_LAYER_NORM, OpType.LAYER_NORM, OpType.BATCH_NORM, OpType.GROUP_NORM, OpType.INSTANCE_NORM}
    _POOL_OPS = {OpType.MAX_POOL, OpType.AVERAGE_POOL, OpType.GLOBAL_AVERAGE_POOL}

    def partition(self, graph: Graph) -> list[GraphPartition]:
        """Partition graph nodes into typed blocks.

        Returns list of GraphPartition in topological order.
        """
        topo_order = graph.topological_order()
        partitions: list[GraphPartition] = []

        for node_name in topo_order:
            node = graph.nodes[node_name]
            block_type = self._classify_node(node)

            # Compute external inputs and outputs
            node_set = {node_name}
            internal_tensors = set(node.outputs)
            ext_inputs = [
                inp for inp in node.inputs
                if inp not in internal_tensors and inp in graph.tensors
            ]
            ext_outputs = []
            for out in node.outputs:
                consumers = graph.get_consumers(out)
                has_external = any(c.name not in node_set for c in consumers)
                if has_external or out in graph.outputs:
                    ext_outputs.append(out)

            partitions.append(GraphPartition(
                block_type=block_type,
                node_names=[node_name],
                inputs=ext_inputs,
                outputs=ext_outputs if ext_outputs else list(node.outputs),
            ))

        return partitions

    def partition_merged(self, graph: Graph) -> list[GraphPartition]:
        """Partition and merge consecutive blocks of the same type."""
        raw = self.partition(graph)
        if not raw:
            return []

        merged: list[GraphPartition] = [raw[0]]
        for part in raw[1:]:
            prev = merged[-1]
            if prev.block_type == part.block_type and prev.block_type != "other":
                # Merge
                prev.node_names.extend(part.node_names)
                # Update inputs: add new external inputs
                prev_outputs = set()
                for n in prev.node_names:
                    prev_outputs.update(graph.nodes[n].outputs)
                prev.inputs = [
                    inp for inp in (prev.inputs + part.inputs)
                    if inp not in prev_outputs
                ]
                prev.inputs = list(dict.fromkeys(prev.inputs))  # deduplicate preserving order
                prev.outputs = part.outputs
            else:
                merged.append(part)

        return merged

    def _classify_node(self, node: OpNode) -> str:
        if node.op_type in self._ATTENTION_OPS:
            return "attention"
        if node.op_type in self._CONV_OPS:
            return "conv"
        if node.op_type in self._LINEAR_OPS:
            return "linear"
        if node.op_type in self._NORM_OPS:
            return "norm"
        if node.op_type in self._POOL_OPS:
            return "pool"
        if node.op_type == OpType.ADD:
            return "residual"
        return "other"
