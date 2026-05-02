"""MLASIC DAG Scheduler — scheduling for arbitrary graph topologies.

Extends the linear Scheduler (Stage 3) to handle residual connections,
attention blocks, parallel branches, and other DAG structures found in
CNNs and Transformers.

Key differences from linear Scheduler:
  - Per-operator cycle models instead of single FusedLinear formula
  - Activation lifetime analysis for skip connections
  - SRAM budget planning with multiple live activations
  - No SRAM row address assignment (weights go to per-tile ROM)
  - No ping-pong banking assumption
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from mlasic.ir import (
    DAGLayerSchedule,
    FusedConvAttrs,
    FusedLinearAttrs,
    Graph,
    HardwareConstraints,
    OpType,
    PoolAttrs,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cycle Models (Step 2)
# ---------------------------------------------------------------------------


class CycleModel(ABC):
    """Base class for per-operator cycle estimation."""

    @abstractmethod
    def estimate_cycles(
        self,
        node_name: str,
        graph: Graph,
        constraints: HardwareConstraints,
    ) -> int:
        """Estimate execution cycles for a node."""


class LinearCycleModel(CycleModel):
    """Cycle model for FusedLinear / FusedLinearReLU.

    Formula: num_tiles * (input_dim + 149)
    Wraps the existing CycleBreakdown logic.
    """

    OVERHEAD = 149  # bias_load(128) + pipeline_fill(1) + requant(3) + write_tile(16) + done(1)

    def estimate_cycles(
        self,
        node_name: str,
        graph: Graph,
        constraints: HardwareConstraints,
    ) -> int:
        node = graph.nodes[node_name]
        attrs = node.fused_attrs
        if not isinstance(attrs, FusedLinearAttrs):
            raise ValueError(f"LinearCycleModel requires FusedLinearAttrs on {node_name}")
        parallelism = _compute_parallelism(attrs.output_dim, constraints.max_parallelism)
        num_tiles = attrs.output_dim // parallelism
        return num_tiles * (attrs.input_dim + self.OVERHEAD)


class ConvCycleModel(CycleModel):
    """Cycle model for Conv / FusedConv / FusedConvReLU / FusedConvReLU6.

    Formula: num_oc_tiles * OH * OW * (KH * KW * IC_per_group + overhead)
    where overhead = 10 (bias/requant/write per output pixel).
    """

    PER_PIXEL_OVERHEAD = 10

    def estimate_cycles(
        self,
        node_name: str,
        graph: Graph,
        constraints: HardwareConstraints,
    ) -> int:
        node = graph.nodes[node_name]

        # Get output spatial dimensions from the output tensor
        out_tensor = graph.tensors[node.outputs[0]]
        out_shape = out_tensor.type.shape  # (N, OC, OH, OW)
        oh = _safe_dim(out_shape[2]) if len(out_shape) == 4 else 1
        ow = _safe_dim(out_shape[3]) if len(out_shape) == 4 else 1

        # Support both FusedConvAttrs and unfused Conv (via op_attrs/attributes)
        attrs = node.fused_attrs
        if isinstance(attrs, FusedConvAttrs):
            oc = attrs.out_channels
            ic_per_group = attrs.in_channels // attrs.group
            kh, kw = attrs.kernel_shape[0], attrs.kernel_shape[1]
        elif node.op_attrs is not None and hasattr(node.op_attrs, "kernel_shape"):
            # Unfused Conv with ConvAttrs
            kh, kw = node.op_attrs.kernel_shape[0], node.op_attrs.kernel_shape[1]
            group = node.op_attrs.group
            # Infer OC and IC from weight tensor shape [OC, IC/group, KH, KW]
            if len(node.inputs) >= 2:
                wt = graph.tensors.get(node.inputs[1])
                if wt and wt.type.shape and len(wt.type.shape) == 4:
                    oc = wt.type.shape[0]
                    ic_per_group = wt.type.shape[1]
                else:
                    oc = out_shape[1] if len(out_shape) >= 2 else 1
                    ic_per_group = 1
            else:
                oc = out_shape[1] if len(out_shape) >= 2 else 1
                ic_per_group = 1
        else:
            # Fallback: estimate from tensor shapes
            oc = out_shape[1] if len(out_shape) >= 2 else 1
            kh, kw = 3, 3  # default kernel
            ic_per_group = 1

        parallelism = _compute_parallelism(oc, constraints.max_parallelism)
        num_oc_tiles = math.ceil(oc / parallelism)

        cycles_per_pixel = kh * kw * ic_per_group + self.PER_PIXEL_OVERHEAD
        return num_oc_tiles * oh * ow * cycles_per_pixel


class SoftmaxCycleModel(CycleModel):
    """Cycle model for Softmax: 3 passes over the sequence dimension."""

    def estimate_cycles(
        self,
        node_name: str,
        graph: Graph,
        constraints: HardwareConstraints,
    ) -> int:
        node = graph.nodes[node_name]
        in_tensor = graph.tensors[node.inputs[0]]
        if not in_tensor.type.shape:
            return 3  # scalar
        seq_len = _safe_dim(in_tensor.type.shape[-1])
        return 3 * seq_len


class LayerNormCycleModel(CycleModel):
    """Cycle model for LayerNormalization / FusedLayerNorm.

    Formula: 2 * feature_dim + overhead (2 passes: stats, normalize+affine).
    """

    OVERHEAD = 10

    def estimate_cycles(
        self,
        node_name: str,
        graph: Graph,
        constraints: HardwareConstraints,
    ) -> int:
        node = graph.nodes[node_name]
        in_tensor = graph.tensors[node.inputs[0]]
        if not in_tensor.type.shape:
            return self.OVERHEAD
        feature_dim = _safe_dim(in_tensor.type.shape[-1])
        return 2 * feature_dim + self.OVERHEAD


class ElementwiseCycleModel(CycleModel):
    """Cycle model for elementwise ops (Add, Mul, Sub, Div, Sigmoid, Tanh, etc.).

    Formula: ceil(numel / parallelism).
    """

    def estimate_cycles(
        self,
        node_name: str,
        graph: Graph,
        constraints: HardwareConstraints,
    ) -> int:
        node = graph.nodes[node_name]
        out_tensor = graph.tensors[node.outputs[0]]
        numel = _safe_numel(out_tensor.type.shape)
        return math.ceil(numel / constraints.max_parallelism)


class PoolCycleModel(CycleModel):
    """Cycle model for MaxPool, AveragePool, GlobalAveragePool.

    Formula: output_pixels * kernel_area.
    """

    def estimate_cycles(
        self,
        node_name: str,
        graph: Graph,
        constraints: HardwareConstraints,
    ) -> int:
        node = graph.nodes[node_name]
        out_tensor = graph.tensors[node.outputs[0]]
        out_shape = out_tensor.type.shape

        # Calculate output pixels (all dims except batch)
        output_pixels = 1
        for d in out_shape[1:]:
            output_pixels *= _safe_dim(d)

        # Determine kernel area
        if node.op_type == OpType.GLOBAL_AVERAGE_POOL:
            in_tensor = graph.tensors[node.inputs[0]]
            in_shape = in_tensor.type.shape
            kernel_area = 1
            for d in in_shape[2:]:  # spatial dims
                kernel_area *= _safe_dim(d)
        else:
            op_a = node.op_attrs
            if isinstance(op_a, PoolAttrs):
                kernel_area = 1
                for k in op_a.kernel_shape:
                    kernel_area *= k
            else:
                # Fallback: check attributes dict
                ks = node.attributes.get("kernel_shape", [2, 2])
                kernel_area = 1
                for k in ks:
                    kernel_area *= k

        return output_pixels * kernel_area


class ReshapeCycleModel(CycleModel):
    """Cycle model for Reshape, Flatten, Transpose, Squeeze, Unsqueeze, Gather, Concat, Cast.

    Zero-cost metadata change (or near-zero copy).
    """

    def estimate_cycles(
        self,
        node_name: str,
        graph: Graph,
        constraints: HardwareConstraints,
    ) -> int:
        return 0


class ActivationCycleModel(CycleModel):
    """Cycle model for LUT-based activations (FusedGELU, FusedSiLU, Erf).

    Formula: ceil(numel / parallelism).
    """

    def estimate_cycles(
        self,
        node_name: str,
        graph: Graph,
        constraints: HardwareConstraints,
    ) -> int:
        node = graph.nodes[node_name]
        out_tensor = graph.tensors[node.outputs[0]]
        numel = _safe_numel(out_tensor.type.shape)
        return math.ceil(numel / constraints.max_parallelism)


class MatMulCycleModel(CycleModel):
    """Cycle model for unfused MatMul (3D batched or 2D).

    Uses linear-like formula: tiles * (K + overhead) for each batch element.
    """

    OVERHEAD = 149

    def estimate_cycles(
        self,
        node_name: str,
        graph: Graph,
        constraints: HardwareConstraints,
    ) -> int:
        node = graph.nodes[node_name]
        out_tensor = graph.tensors[node.outputs[0]]
        in_a = graph.tensors[node.inputs[0]]

        out_shape = out_tensor.type.shape
        in_a_shape = in_a.type.shape

        # Handle scalar/empty shapes
        if not in_a_shape or not out_shape:
            return self.OVERHEAD

        # K = last dim of first input
        k_dim = _safe_dim(in_a_shape[-1])
        # N = last dim of output
        n_dim = _safe_dim(out_shape[-1])

        parallelism = _compute_parallelism(n_dim, constraints.max_parallelism)
        num_tiles = math.ceil(n_dim / parallelism)

        # Batch size = product of all dims except last two
        batch = 1
        for d in out_shape[:-2]:
            batch *= _safe_dim(d)
        if len(out_shape) <= 2:
            batch = 1

        return batch * num_tiles * (k_dim + self.OVERHEAD)


class GemmCycleModel(CycleModel):
    """Cycle model for Gemm (alpha*A*B + beta*C).

    Similar to MatMul but always 2D. Formula: tiles * (K + overhead).
    """

    OVERHEAD = 149

    def estimate_cycles(
        self,
        node_name: str,
        graph: Graph,
        constraints: HardwareConstraints,
    ) -> int:
        node = graph.nodes[node_name]
        out_tensor = graph.tensors[node.outputs[0]]
        in_a = graph.tensors[node.inputs[0]]

        # Gemm: A is (M, K) or transposed, B is (K, N) or transposed
        transA = node.attributes.get("transA", 0)
        transB = node.attributes.get("transB", 0)

        a_shape = in_a.type.shape
        out_shape = out_tensor.type.shape
        if not a_shape or not out_shape:
            return self.OVERHEAD
        k_dim = _safe_dim(a_shape[-1] if not transA else a_shape[-2] if len(a_shape) >= 2 else a_shape[-1])
        n_dim = _safe_dim(out_shape[-1])

        parallelism = _compute_parallelism(n_dim, constraints.max_parallelism)
        num_tiles = math.ceil(n_dim / parallelism)
        return num_tiles * (k_dim + self.OVERHEAD)


class ReductionCycleModel(CycleModel):
    """Cycle model for ReduceMean, ReduceSum.

    Formula: numel of input tensor / parallelism.
    """

    def estimate_cycles(
        self,
        node_name: str,
        graph: Graph,
        constraints: HardwareConstraints,
    ) -> int:
        node = graph.nodes[node_name]
        in_tensor = graph.tensors[node.inputs[0]]
        numel = _safe_numel(in_tensor.type.shape)
        return math.ceil(numel / constraints.max_parallelism)


class ControlFlowCycleModel(CycleModel):
    """Cycle model for If and other control flow ops.

    Estimates as sum of both branch costs (conservative upper bound).
    Falls back to 1 cycle if no branch info available.
    """

    def estimate_cycles(
        self,
        node_name: str,
        graph: Graph,
        constraints: HardwareConstraints,
    ) -> int:
        # Conservative: 1 cycle placeholder (branches not analyzed)
        return 1


# ---------------------------------------------------------------------------
# Cycle model registry
# ---------------------------------------------------------------------------

CYCLE_MODEL_REGISTRY: dict[OpType, CycleModel] = {
    # Fused linear
    OpType.FUSED_LINEAR: LinearCycleModel(),
    OpType.FUSED_LINEAR_RELU: LinearCycleModel(),
    # Fused conv
    OpType.FUSED_CONV: ConvCycleModel(),
    OpType.FUSED_CONV_RELU: ConvCycleModel(),
    OpType.FUSED_CONV_RELU6: ConvCycleModel(),
    # Attention / norm
    OpType.SOFTMAX: SoftmaxCycleModel(),
    OpType.LAYER_NORM: LayerNormCycleModel(),
    OpType.FUSED_LAYER_NORM: LayerNormCycleModel(),
    OpType.BATCH_NORM: LayerNormCycleModel(),
    # Elementwise
    OpType.ADD: ElementwiseCycleModel(),
    OpType.MUL: ElementwiseCycleModel(),
    OpType.SUB: ElementwiseCycleModel(),
    OpType.DIV: ElementwiseCycleModel(),
    OpType.POW: ElementwiseCycleModel(),
    OpType.SQRT: ElementwiseCycleModel(),
    OpType.NEG: ElementwiseCycleModel(),
    OpType.CLIP: ElementwiseCycleModel(),
    OpType.WHERE: ElementwiseCycleModel(),
    OpType.SIGMOID: ActivationCycleModel(),
    OpType.TANH: ActivationCycleModel(),
    OpType.RELU: ElementwiseCycleModel(),
    OpType.ERF: ActivationCycleModel(),
    OpType.SIN: ActivationCycleModel(),
    OpType.COS: ActivationCycleModel(),
    # Comparison / logical
    OpType.EQUAL: ElementwiseCycleModel(),
    OpType.LESS: ElementwiseCycleModel(),
    OpType.GREATER: ElementwiseCycleModel(),
    OpType.NOT: ElementwiseCycleModel(),
    # Fused activations
    OpType.FUSED_GELU: ActivationCycleModel(),
    OpType.FUSED_SILU: ActivationCycleModel(),
    # Pooling
    OpType.MAX_POOL: PoolCycleModel(),
    OpType.AVERAGE_POOL: PoolCycleModel(),
    OpType.GLOBAL_AVERAGE_POOL: PoolCycleModel(),
    # Reshape / metadata (zero-cost)
    OpType.RESHAPE: ReshapeCycleModel(),
    OpType.FLATTEN: ReshapeCycleModel(),
    OpType.TRANSPOSE: ReshapeCycleModel(),
    OpType.SQUEEZE: ReshapeCycleModel(),
    OpType.UNSQUEEZE: ReshapeCycleModel(),
    OpType.GATHER: ReshapeCycleModel(),
    OpType.CONCAT: ReshapeCycleModel(),
    OpType.CAST: ReshapeCycleModel(),
    OpType.SLICE: ReshapeCycleModel(),
    OpType.SPLIT: ReshapeCycleModel(),
    OpType.SHAPE: ReshapeCycleModel(),
    OpType.IDENTITY: ReshapeCycleModel(),
    OpType.CONSTANT: ReshapeCycleModel(),
    OpType.CONSTANT_OF_SHAPE: ReshapeCycleModel(),
    OpType.RANGE: ReshapeCycleModel(),
    OpType.EXPAND: ReshapeCycleModel(),
    OpType.PAD: ReshapeCycleModel(),
    OpType.TILE: ReshapeCycleModel(),
    # Quantization ops (elementwise throughput)
    OpType.DEQUANTIZE_LINEAR: ElementwiseCycleModel(),
    OpType.DYNAMIC_QUANTIZE_LINEAR: ElementwiseCycleModel(),
    OpType.TRILU: ElementwiseCycleModel(),
    OpType.SCATTER_ND: ElementwiseCycleModel(),
    # MatMul / Gemm (unfused)
    OpType.MATMUL: MatMulCycleModel(),
    OpType.GEMM: GemmCycleModel(),
    OpType.MATMUL_INTEGER: MatMulCycleModel(),
    # Conv (unfused)
    OpType.CONV: ConvCycleModel(),
    OpType.CONV_TRANSPOSE: ConvCycleModel(),
    # Reduction
    OpType.REDUCE_MEAN: ReductionCycleModel(),
    OpType.REDUCE_SUM: ReductionCycleModel(),
    # Control flow
    OpType.IF: ControlFlowCycleModel(),
    # Tier 3 ops
    OpType.GROUP_NORM: LayerNormCycleModel(),
    OpType.INSTANCE_NORM: LayerNormCycleModel(),
    OpType.Q_LINEAR_MATMUL: MatMulCycleModel(),
    OpType.RESIZE: ElementwiseCycleModel(),
    # Fused attention / MLP
    OpType.FUSED_ATTENTION: MatMulCycleModel(),  # dominated by QKV matmuls
    OpType.FUSED_MLP: LinearCycleModel(),
}


def _compute_parallelism(output_dim: int, max_par: int) -> int:
    """Compute the largest factor of output_dim <= max_par."""
    par = min(output_dim, max_par)
    while par > 1:
        if output_dim % par == 0:
            return par
        par -= 1
    return 1


def _safe_dim(dim: int, default: int = 1) -> int:
    """Return dim if positive, otherwise default (for dynamic dimensions)."""
    return dim if dim > 0 else default


def _safe_numel(shape: tuple, default_dim: int = 1) -> int:
    """Compute numel with dynamic dims replaced by default_dim."""
    result = 1
    for d in shape:
        result *= _safe_dim(d, default_dim)
    return result


# ---------------------------------------------------------------------------
# Activation Lifetime Analysis (Step 3)
# ---------------------------------------------------------------------------


@dataclass
class ActivationLifetime:
    """Tracks when a tensor is produced and when it can be freed."""

    tensor_name: str
    producer_node: str
    last_consumer_node: str
    produce_cycle: int
    free_cycle: int
    size_bytes: int


class ActivationLifetimeAnalyzer:
    """Analyzes activation tensor lifetimes across DAG execution."""

    def analyze(
        self,
        graph: Graph,
        execution_order: list[str],
        node_end_cycles: dict[str, int],
    ) -> list[ActivationLifetime]:
        """Compute lifetime for each non-constant activation tensor.

        Args:
            graph: The IR graph.
            execution_order: Node names in execution (topo) order.
            node_end_cycles: Mapping node_name -> end_cycle.

        Returns:
            List of ActivationLifetime, one per non-constant produced tensor.
        """
        graph._build_adjacency()
        lifetimes: list[ActivationLifetime] = []

        for node_name in execution_order:
            node = graph.nodes[node_name]
            for out_name in node.outputs:
                tensor = graph.tensors.get(out_name)
                if tensor is None or tensor.is_constant:
                    continue

                # Find all consumer nodes
                consumers = graph.get_consumers(out_name)
                if not consumers:
                    # Graph output with no further consumers — free at producer end
                    free_cycle = node_end_cycles[node_name]
                    last_consumer = node_name
                else:
                    # Free when last consumer finishes
                    consumer_ends = []
                    for c in consumers:
                        if c.name in node_end_cycles:
                            consumer_ends.append((node_end_cycles[c.name], c.name))
                    if consumer_ends:
                        free_cycle, last_consumer = max(consumer_ends, key=lambda x: x[0])
                    else:
                        free_cycle = node_end_cycles[node_name]
                        last_consumer = node_name

                # Compute size_bytes safely (0 for dynamic shapes)
                if tensor.type.is_shape_known:
                    sz_bytes = tensor.type.size_bytes
                else:
                    sz_bytes = _safe_numel(tensor.type.shape) * tensor.type.dtype.itemsize

                lifetimes.append(
                    ActivationLifetime(
                        tensor_name=out_name,
                        producer_node=node_name,
                        last_consumer_node=last_consumer,
                        produce_cycle=node_end_cycles[node_name],
                        free_cycle=free_cycle,
                        size_bytes=sz_bytes,
                    )
                )

        return lifetimes


# ---------------------------------------------------------------------------
# SRAM Budget Planner (Step 4)
# ---------------------------------------------------------------------------


@dataclass
class SRAMSnapshot:
    """Snapshot of live activations at a point in time."""

    cycle: int
    live_bytes: int
    live_tensors: list[str]


class SRAMBudgetPlanner:
    """Check whether live activations fit in SRAM budget."""

    def __init__(self, budget_bytes: Optional[int] = None):
        self.budget_bytes = budget_bytes

    def check_budget(
        self,
        lifetimes: list[ActivationLifetime],
        node_schedules: dict[str, DAGLayerSchedule],
    ) -> tuple[bool, int, list[SRAMSnapshot]]:
        """Check if peak activation memory fits budget.

        Returns:
            (fits, peak_bytes, snapshots) where fits is True if under budget
            (or no budget set), peak_bytes is maximum simultaneous live bytes,
            and snapshots captures state at each event.
        """
        # Build time events: (cycle, +/- bytes, tensor_name)
        events: list[tuple[int, int, str]] = []
        for lt in lifetimes:
            events.append((lt.produce_cycle, lt.size_bytes, lt.tensor_name))
            events.append((lt.free_cycle, -lt.size_bytes, lt.tensor_name))

        # Sort by cycle, with frees (negative) before produces at same cycle
        events.sort(key=lambda e: (e[0], e[1]))

        live_bytes = 0
        peak_bytes = 0
        live_set: set[str] = set()
        snapshots: list[SRAMSnapshot] = []

        for cycle, delta, tensor_name in events:
            if delta > 0:
                live_set.add(tensor_name)
            else:
                live_set.discard(tensor_name)
            live_bytes += delta
            if live_bytes > peak_bytes:
                peak_bytes = live_bytes
                snapshots.append(
                    SRAMSnapshot(
                        cycle=cycle,
                        live_bytes=live_bytes,
                        live_tensors=sorted(live_set),
                    )
                )

        fits = True
        if self.budget_bytes is not None and peak_bytes > self.budget_bytes:
            fits = False

        return fits, peak_bytes, snapshots


# ---------------------------------------------------------------------------
# DAGSchedule result (Step 5)
# ---------------------------------------------------------------------------


@dataclass
class DAGSchedule:
    """Complete schedule for a DAG graph."""

    total_cycles: int
    clock_mhz: int
    node_schedules: list[DAGLayerSchedule]
    lifetimes: list[ActivationLifetime]
    peak_activation_bytes: int
    total_weight_bytes: int
    total_bias_bytes: int

    @property
    def latency_us(self) -> float:
        return self.total_cycles / self.clock_mhz

    @property
    def latency_ms(self) -> float:
        return self.latency_us / 1000.0

    @property
    def throughput_inferences_per_sec(self) -> float:
        if self.latency_us == 0:
            return 0.0
        return 1_000_000.0 / self.latency_us

    def to_json(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "clock_mhz": self.clock_mhz,
            "total_cycles": self.total_cycles,
            "latency_us": self.latency_us,
            "latency_ms": self.latency_ms,
            "throughput_inferences_per_sec": self.throughput_inferences_per_sec,
            "peak_activation_bytes": self.peak_activation_bytes,
            "total_weight_bytes": self.total_weight_bytes,
            "total_bias_bytes": self.total_bias_bytes,
            "nodes": [
                {
                    "node_index": ns.node_index,
                    "node_name": ns.node_name,
                    "op_type": ns.op_type,
                    "total_cycles": ns.total_cycles,
                    "start_cycle": ns.start_cycle,
                    "end_cycle": ns.end_cycle,
                    "weight_bytes": ns.weight_bytes,
                    "bias_bytes": ns.bias_bytes,
                    "input_tensors": ns.input_tensors,
                    "output_tensors": ns.output_tensors,
                    "input_bytes": ns.input_bytes,
                    "output_bytes": ns.output_bytes,
                }
                for ns in self.node_schedules
            ],
            "lifetimes": [
                {
                    "tensor_name": lt.tensor_name,
                    "producer_node": lt.producer_node,
                    "last_consumer_node": lt.last_consumer_node,
                    "produce_cycle": lt.produce_cycle,
                    "free_cycle": lt.free_cycle,
                    "size_bytes": lt.size_bytes,
                }
                for lt in self.lifetimes
            ],
        }


# ---------------------------------------------------------------------------
# Peak-SRAM-aware topological reorder
# ---------------------------------------------------------------------------


def _tensor_live_bytes(graph: Graph, tensor_name: str) -> int:
    """Live activation bytes for a tensor; 0 for constants/unknown shapes."""
    t = graph.tensors.get(tensor_name)
    if t is None or t.is_constant:
        return 0
    if t.type.is_shape_known:
        return t.type.size_bytes
    return _safe_numel(t.type.shape) * t.type.dtype.itemsize


def peak_aware_topological_order(graph: Graph) -> list[str]:
    """Topological order that greedily minimises peak activation bytes.

    Standard topo order is correctness-only. For a branchy DAG, two
    distinct topo orders can produce wildly different peak SRAM. This
    function uses a Sethi–Ullman-style greedy: among nodes whose
    in-degree is zero, pick the one whose execution would *increase*
    live activation bytes the least (output bytes minus bytes freed by
    consuming the last use of an input).

    For a strict linear chain this is identical to the canonical topo
    order — only one node is ready at each step. The reorder kicks in
    for residual blocks, parallel branches, and concat patterns where
    deep branches should run after shallow ones to keep peak SRAM low.
    """
    graph._build_adjacency()

    canonical_order = graph.topological_order()
    canonical_idx = {n: i for i, n in enumerate(canonical_order)}

    in_deg: dict[str, int] = {n: 0 for n in graph.nodes}
    for node_name, node in graph.nodes.items():
        for inp in node.inputs:
            prod = graph._tensor_to_producer.get(inp)
            if prod and prod in graph.nodes:
                in_deg[node_name] += 1

    consumers_remaining: dict[str, int] = {}
    for tname in graph.tensors:
        consumers_remaining[tname] = len(graph._tensor_to_consumers.get(tname, []))

    ready: list[str] = [n for n, d in in_deg.items() if d == 0]
    order: list[str] = []

    def cost(name: str) -> tuple:
        node = graph.nodes[name]
        out_bytes = sum(_tensor_live_bytes(graph, t) for t in node.outputs)
        freed = 0
        for inp in node.inputs:
            if consumers_remaining.get(inp, 0) == 1:
                freed += _tensor_live_bytes(graph, inp)
        # delta = bytes added; tiebreak by canonical topo position
        # for determinism across runs.
        return (out_bytes - freed, canonical_idx.get(name, 0))

    while ready:
        ready.sort(key=cost)
        chosen = ready.pop(0)
        order.append(chosen)
        node = graph.nodes[chosen]
        for inp in node.inputs:
            if inp in consumers_remaining:
                consumers_remaining[inp] = max(0, consumers_remaining[inp] - 1)
        for out in node.outputs:
            for cname in graph._tensor_to_consumers.get(out, []):
                if cname not in graph.nodes:
                    continue
                in_deg[cname] -= 1
                if in_deg[cname] == 0:
                    ready.append(cname)

    if len(order) != len(graph.nodes):
        raise ValueError(
            f"Cycle detected during peak-aware reorder: "
            f"{len(order)}/{len(graph.nodes)} nodes scheduled"
        )
    return order


def estimate_peak_bytes(graph: Graph, order: list[str]) -> int:
    """Simulate live-byte usage along ``order``; returns peak."""
    graph._build_adjacency()
    consumers_remaining: dict[str, int] = {}
    for tname in graph.tensors:
        consumers_remaining[tname] = len(graph._tensor_to_consumers.get(tname, []))

    live: set[str] = set()
    live_bytes = 0
    peak = 0
    for name in order:
        node = graph.nodes[name]
        for out in node.outputs:
            sz = _tensor_live_bytes(graph, out)
            if sz > 0:
                live.add(out)
                live_bytes += sz
        peak = max(peak, live_bytes)
        for inp in node.inputs:
            if inp not in consumers_remaining:
                continue
            consumers_remaining[inp] -= 1
            if consumers_remaining[inp] <= 0 and inp in live:
                live.remove(inp)
                live_bytes -= _tensor_live_bytes(graph, inp)
    return peak


# ---------------------------------------------------------------------------
# DAGScheduler (Step 5)
# ---------------------------------------------------------------------------


class DAGScheduler:
    """Schedule arbitrary DAG graphs for ASIC execution.

    Unlike the linear Scheduler, this:
      - Uses per-operator cycle models
      - Tracks activation lifetimes for skip connections
      - Reports peak activation SRAM usage
      - Does NOT assign SRAM row addresses (tile mapper does that)
    """

    def __init__(
        self,
        constraints: Optional[HardwareConstraints] = None,
        sram_budget_bytes: Optional[int] = None,
        peak_aware_order: bool = True,
    ):
        self.constraints = constraints or HardwareConstraints()
        self.sram_budget_bytes = sram_budget_bytes
        self.peak_aware_order = peak_aware_order

    def schedule(self, graph: Graph) -> DAGSchedule:
        """Schedule the graph. Returns DAGSchedule and mutates graph in place.

        Preconditions:
          - graph.stage in ("quantized_dag", "quantized")
          - All fused nodes have quantized fused_attrs

        Raises:
          ValueError: If preconditions not met or unknown op type.
        """
        self._validate_preconditions(graph)

        if self.peak_aware_order:
            execution_order = peak_aware_topological_order(graph)
        else:
            execution_order = graph.topological_order()
        node_schedules: list[DAGLayerSchedule] = []
        node_end_cycles: dict[str, int] = {}
        current_cycle = 0

        for idx, node_name in enumerate(execution_order):
            node = graph.nodes[node_name]

            # Look up cycle model
            cycle_model = CYCLE_MODEL_REGISTRY.get(node.op_type)
            if cycle_model is None:
                raise ValueError(
                    f"No cycle model for op type {node.op_type.value} at node {node_name}"
                )

            cycles = cycle_model.estimate_cycles(node_name, graph, self.constraints)

            # Compute weight/bias bytes
            weight_bytes, bias_bytes = self._compute_weight_bias_bytes(node, graph)

            # Compute input/output bytes (skip dynamic-shape tensors)
            input_tensors = [inp for inp in node.inputs if inp in graph.tensors]
            output_tensors = [out for out in node.outputs if out in graph.tensors]
            input_bytes = 0
            for t in input_tensors:
                tensor = graph.tensors[t]
                if not tensor.is_constant and tensor.type.is_shape_known:
                    input_bytes += tensor.type.size_bytes
            output_bytes = 0
            for t in output_tensors:
                tensor = graph.tensors[t]
                if not tensor.is_constant and tensor.type.is_shape_known:
                    output_bytes += tensor.type.size_bytes

            ds = DAGLayerSchedule(
                node_index=idx,
                node_name=node_name,
                op_type=node.op_type.value,
                total_cycles=cycles,
                start_cycle=current_cycle,
                end_cycle=current_cycle + cycles,
                weight_bytes=weight_bytes,
                bias_bytes=bias_bytes,
                input_tensors=input_tensors,
                output_tensors=output_tensors,
                input_bytes=input_bytes,
                output_bytes=output_bytes,
            )

            node.dag_schedule = ds
            node_schedules.append(ds)
            node_end_cycles[node_name] = current_cycle + cycles
            current_cycle += cycles

        # Activation lifetime analysis
        analyzer = ActivationLifetimeAnalyzer()
        lifetimes = analyzer.analyze(graph, execution_order, node_end_cycles)

        # SRAM budget check
        planner = SRAMBudgetPlanner(self.sram_budget_bytes)
        node_sched_map = {ns.node_name: ns for ns in node_schedules}
        fits, peak_bytes, _ = planner.check_budget(lifetimes, node_sched_map)

        if not fits:
            logger.warning(
                "Peak activation SRAM %d bytes exceeds budget %d bytes",
                peak_bytes,
                self.sram_budget_bytes,
            )

        total_weight_bytes = sum(ns.weight_bytes for ns in node_schedules)
        total_bias_bytes = sum(ns.bias_bytes for ns in node_schedules)

        graph.stage = "scheduled_dag"

        schedule = DAGSchedule(
            total_cycles=current_cycle,
            clock_mhz=self.constraints.clock_mhz,
            node_schedules=node_schedules,
            lifetimes=lifetimes,
            peak_activation_bytes=peak_bytes,
            total_weight_bytes=total_weight_bytes,
            total_bias_bytes=total_bias_bytes,
        )

        self._log_summary(schedule)
        return schedule

    @staticmethod
    def _compute_weight_bias_bytes(node, graph: Graph) -> tuple[int, int]:
        """Compute weight and bias bytes for a node."""
        weight_bytes = 0
        bias_bytes = 0

        attrs = node.fused_attrs
        if isinstance(attrs, (FusedLinearAttrs, FusedConvAttrs)):
            # Weight tensor is inputs[1], bias is inputs[2]
            if len(node.inputs) >= 2:
                wt = graph.tensors.get(node.inputs[1])
                if wt and wt.is_constant and wt.data is not None:
                    weight_bytes = wt.data.nbytes
            if len(node.inputs) >= 3:
                bt = graph.tensors.get(node.inputs[2])
                if bt and bt.is_constant and bt.data is not None:
                    bias_bytes = bt.data.nbytes
        elif node.op_type == OpType.MATMUL:
            # Second input may be a constant weight
            if len(node.inputs) >= 2:
                wt = graph.tensors.get(node.inputs[1])
                if wt and wt.is_constant and wt.data is not None:
                    weight_bytes = wt.data.nbytes

        return weight_bytes, bias_bytes

    @staticmethod
    def _validate_preconditions(graph: Graph) -> None:
        """Validate that the graph is ready for DAG scheduling."""
        if graph.stage not in ("quantized_dag", "quantized"):
            raise ValueError(
                f"Graph must be in 'quantized_dag' or 'quantized' stage, got '{graph.stage}'"
            )

        if len(graph.nodes) == 0:
            raise ValueError("Cannot schedule an empty graph (0 nodes)")

        # All fused nodes must be quantized
        for node in graph.nodes.values():
            if node.op_type.is_fused:
                attrs = node.fused_attrs
                if attrs is None:
                    raise ValueError(f"Fused node {node.name} missing fused_attrs")
                if hasattr(attrs, "is_quantized") and not attrs.is_quantized:
                    raise ValueError(f"Fused node {node.name} is not fully quantized")

    @staticmethod
    def _log_summary(schedule: DAGSchedule) -> None:
        """Log a human-readable schedule summary."""
        logger.info("=" * 80)
        logger.info("DAG Schedule Summary")
        logger.info("=" * 80)
        logger.info(
            "%-4s %-25s %-20s %8s %8s %8s",
            "Idx",
            "Name",
            "OpType",
            "Cycles",
            "Start",
            "End",
        )
        logger.info("-" * 80)
        for ns in schedule.node_schedules:
            logger.info(
                "%-4d %-25s %-20s %8d %8d %8d",
                ns.node_index,
                ns.node_name[:25],
                ns.op_type[:20],
                ns.total_cycles,
                ns.start_cycle,
                ns.end_cycle,
            )
        logger.info("-" * 80)
        logger.info("Total cycles:           %d", schedule.total_cycles)
        logger.info("Latency:                %.2f µs", schedule.latency_us)
        logger.info("Peak activation SRAM:   %d bytes", schedule.peak_activation_bytes)
        logger.info("Total weight bytes:     %d", schedule.total_weight_bytes)
        logger.info("Total bias bytes:       %d", schedule.total_bias_bytes)
        logger.info("=" * 80)


# ---------------------------------------------------------------------------
# Weight Stream Scheduler (Phase 3 gap)
# ---------------------------------------------------------------------------


@dataclass
class WeightStreamPlan:
    """Per-layer weight streaming plan."""

    layer_name: str
    weight_bytes: int
    load_start_cycle: int
    load_end_cycle: int
    compute_start_cycle: int
    compute_end_cycle: int
    is_memory_bound: bool  # True if weight load takes longer than compute


@dataclass
class WeightStreamSchedule:
    """Complete weight streaming schedule with double-buffering."""

    plans: list[WeightStreamPlan]
    total_cycles: int  # Total cycles including weight streaming overhead
    total_cycles_no_overlap: int  # If no double-buffering (serial load+compute)
    overlap_savings_cycles: int  # Cycles saved by overlapping
    peak_weight_sram_bytes: int  # Min SRAM needed for double-buffering
    dram_bandwidth_report: "DRAMBandwidthReport"


@dataclass
class DRAMBandwidthReport:
    """DRAM bandwidth analysis report."""

    axi_width_bits: int
    clock_mhz: int
    max_bandwidth_mbps: float  # MB/s
    layer_analyses: list["LayerBandwidthAnalysis"]
    compute_bound_count: int
    memory_bound_count: int
    compute_bound_ratio: float


@dataclass
class LayerBandwidthAnalysis:
    """Per-layer DRAM bandwidth analysis."""

    layer_name: str
    weight_bytes: int
    load_cycles: int
    compute_cycles: int
    is_memory_bound: bool
    bandwidth_utilization: float  # fraction of max bandwidth actually needed


class WeightStreamScheduler:
    """Schedule weight loads from DDR via AXI to overlap with compute.

    Double-buffering strategy: while one layer computes with weights in SRAM,
    prefetch the next layer's weights into the other SRAM buffer.
    """

    def __init__(
        self,
        constraints: Optional[HardwareConstraints] = None,
        axi_width_bits: int = 64,
    ):
        self.constraints = constraints or HardwareConstraints()
        self.axi_width_bits = axi_width_bits
        self.axi_width_bytes = axi_width_bits // 8

    def schedule(self, dag_schedule: "DAGSchedule") -> WeightStreamSchedule:
        """Create a weight streaming schedule with double-buffering.

        Args:
            dag_schedule: The compute schedule from DAGScheduler.

        Returns:
            WeightStreamSchedule with overlapped load/compute timing.
        """
        node_schedules = dag_schedule.node_schedules

        # Compute load cycles for each layer's weights
        plans: list[WeightStreamPlan] = []
        current_cycle = 0
        total_no_overlap = 0
        max_single_weight = 0

        for ns in node_schedules:
            if ns.weight_bytes == 0:
                # No weights to load — compute-only node
                plans.append(WeightStreamPlan(
                    layer_name=ns.node_name,
                    weight_bytes=0,
                    load_start_cycle=current_cycle,
                    load_end_cycle=current_cycle,
                    compute_start_cycle=current_cycle,
                    compute_end_cycle=current_cycle + ns.total_cycles,
                    is_memory_bound=False,
                ))
                current_cycle += ns.total_cycles
                total_no_overlap += ns.total_cycles
                continue

            load_cycles = self._compute_load_cycles(ns.weight_bytes)
            compute_cycles = ns.total_cycles
            is_mem_bound = load_cycles > compute_cycles

            max_single_weight = max(max_single_weight, ns.weight_bytes)

            # Double-buffer: start loading next layer's weights during current compute
            # For the first layer, must load before compute
            if not plans:
                # First layer: load, then compute
                load_start = current_cycle
                load_end = current_cycle + load_cycles
                compute_start = load_end
                compute_end = compute_start + compute_cycles
                current_cycle = compute_end
                total_no_overlap += load_cycles + compute_cycles
            else:
                # Subsequent layers: overlap load with previous compute
                prev_plan = plans[-1]
                load_start = prev_plan.compute_start_cycle  # start loading during prev compute
                load_end = load_start + load_cycles
                # Compute can start when both load is done and previous compute is done
                compute_start = max(load_end, prev_plan.compute_end_cycle)
                compute_end = compute_start + compute_cycles
                current_cycle = compute_end
                total_no_overlap += load_cycles + compute_cycles

            plans.append(WeightStreamPlan(
                layer_name=ns.node_name,
                weight_bytes=ns.weight_bytes,
                load_start_cycle=load_start,
                load_end_cycle=load_end,
                compute_start_cycle=compute_start,
                compute_end_cycle=compute_end,
                is_memory_bound=is_mem_bound,
            ))

        total_cycles = current_cycle
        overlap_savings = total_no_overlap - total_cycles

        # Peak SRAM for double-buffering: need 2 × max single-layer weight
        peak_sram = 2 * max_single_weight

        # DRAM bandwidth analysis
        bw_report = self._analyze_bandwidth(node_schedules)

        return WeightStreamSchedule(
            plans=plans,
            total_cycles=total_cycles,
            total_cycles_no_overlap=total_no_overlap,
            overlap_savings_cycles=overlap_savings,
            peak_weight_sram_bytes=peak_sram,
            dram_bandwidth_report=bw_report,
        )

    def _compute_load_cycles(self, weight_bytes: int) -> int:
        """Compute cycles to load weight_bytes via AXI bus.

        At clock_mhz with axi_width_bytes per cycle.
        """
        return math.ceil(weight_bytes / self.axi_width_bytes)

    def _analyze_bandwidth(
        self, node_schedules: list[DAGLayerSchedule],
    ) -> DRAMBandwidthReport:
        """Analyze DRAM bandwidth utilization per layer."""
        max_bw = self.axi_width_bytes * self.constraints.clock_mhz  # bytes/sec (in MHz units)
        max_bw_mbps = max_bw / 1_000_000.0 * 1_000_000.0  # MB/s

        analyses: list[LayerBandwidthAnalysis] = []
        compute_bound = 0
        memory_bound = 0

        for ns in node_schedules:
            if ns.weight_bytes == 0:
                analyses.append(LayerBandwidthAnalysis(
                    layer_name=ns.node_name,
                    weight_bytes=0,
                    load_cycles=0,
                    compute_cycles=ns.total_cycles,
                    is_memory_bound=False,
                    bandwidth_utilization=0.0,
                ))
                compute_bound += 1
                continue

            load_cycles = self._compute_load_cycles(ns.weight_bytes)
            is_mem_bound = load_cycles > ns.total_cycles

            if is_mem_bound:
                memory_bound += 1
            else:
                compute_bound += 1

            # Bandwidth utilization: fraction of max BW needed to match compute time
            if ns.total_cycles > 0:
                needed_bw = ns.weight_bytes / ns.total_cycles  # bytes per cycle
                max_bw_per_cycle = self.axi_width_bytes
                bw_util = needed_bw / max_bw_per_cycle if max_bw_per_cycle > 0 else 0.0
            else:
                bw_util = 1.0

            analyses.append(LayerBandwidthAnalysis(
                layer_name=ns.node_name,
                weight_bytes=ns.weight_bytes,
                load_cycles=load_cycles,
                compute_cycles=ns.total_cycles,
                is_memory_bound=is_mem_bound,
                bandwidth_utilization=min(bw_util, 1.0),
            ))

        total = compute_bound + memory_bound
        ratio = compute_bound / total if total > 0 else 1.0

        return DRAMBandwidthReport(
            axi_width_bits=self.axi_width_bits,
            clock_mhz=self.constraints.clock_mhz,
            max_bandwidth_mbps=max_bw_mbps,
            layer_analyses=analyses,
            compute_bound_count=compute_bound,
            memory_bound_count=memory_bound,
            compute_bound_ratio=ratio,
        )
