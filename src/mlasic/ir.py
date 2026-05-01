"""MLASIC Compiler IR data structures.

Transcribed from docs/compiler-ir-spec.md Section 2.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from mlasic.exceptions import IRValidationError

# ---------------------------------------------------------------------------
# QuantParams
# ---------------------------------------------------------------------------


@dataclass
class QuantParams:
    """Quantization parameters for a tensor."""

    scale: float
    zero_point: int
    bit_width: int = 8
    signed: bool = True
    calibrated: bool = False

    def quantize(self, fp_value: np.ndarray) -> np.ndarray:
        """Convert floating-point values to quantized integers.

        Uses round-half-up (np.floor(x + 0.5)) to match hardware rounding.
        Symmetric weights (zero_point=0) clip to [-127, 127].
        Asymmetric activations clip to [-128, 127].
        """
        scaled = np.floor(fp_value / self.scale + 0.5) + self.zero_point
        if self.signed:
            if self.zero_point == 0:
                qmin = -(1 << (self.bit_width - 1)) + 1  # -127
                qmax = (1 << (self.bit_width - 1)) - 1  # 127
            else:
                qmin = -(1 << (self.bit_width - 1))  # -128
                qmax = (1 << (self.bit_width - 1)) - 1  # 127
            return np.clip(scaled, qmin, qmax).astype(np.int8)
        else:
            qmin = 0
            qmax = (1 << self.bit_width) - 1  # 255
            return np.clip(scaled, qmin, qmax).astype(np.uint8)

    def dequantize(self, q_value: np.ndarray) -> np.ndarray:
        """Convert quantized integers back to floating-point."""
        return (q_value.astype(np.float32) - self.zero_point) * self.scale


# ---------------------------------------------------------------------------
# TensorType
# ---------------------------------------------------------------------------


@dataclass
class TensorType:
    """Type information for a tensor edge in the graph."""

    shape: tuple[int, ...]
    dtype: np.dtype
    quant: Optional[QuantParams] = None

    @property
    def is_shape_known(self) -> bool:
        """Whether all dimensions are resolved (no -1 or 0).

        Scalar tensors (shape=()) are considered known.
        Placeholder tensors (shape=(-1,)) are not.
        """
        return all(d > 0 for d in self.shape)

    @property
    def numel(self) -> int:
        """Total number of elements. Raises if shape is unknown."""
        if not self.is_shape_known:
            raise ValueError(f"Cannot compute numel for unknown shape: {self.shape}")
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    @property
    def size_bytes(self) -> int:
        """Size in bytes."""
        return self.numel * self.dtype.itemsize

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    @property
    def batch_size(self) -> int:
        """First dimension (batch). Raises if shape is empty."""
        if not self.shape:
            raise ValueError("Cannot get batch_size of scalar tensor")
        return self.shape[0]

    @property
    def channels(self) -> int:
        """Second dimension (channels for NCHW). Raises if ndim < 2."""
        if self.ndim < 2:
            raise ValueError(f"Cannot get channels for {self.ndim}D tensor")
        return self.shape[1]

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        """Spatial dimensions (H, W, ... for NCHW). Raises if ndim < 3."""
        if self.ndim < 3:
            raise ValueError(f"Cannot get spatial_shape for {self.ndim}D tensor")
        return self.shape[2:]


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------


@dataclass
class Tensor:
    """A tensor in the IR graph (either intermediate activation or constant weight)."""

    name: str
    type: TensorType
    data: Optional[np.ndarray] = None

    @property
    def is_constant(self) -> bool:
        return self.data is not None


# ---------------------------------------------------------------------------
# OpType
# ---------------------------------------------------------------------------


class OpType(Enum):
    """Supported operator types."""

    # Raw ONNX operators — original (MLP)
    MATMUL = "MatMul"
    ADD = "Add"
    RELU = "Relu"
    BATCH_NORM = "BatchNormalization"
    RESHAPE = "Reshape"
    TRANSPOSE = "Transpose"
    FLATTEN = "Flatten"

    # Raw ONNX operators — Tier 1 (CNN/Transformer core)
    CONV = "Conv"
    GEMM = "Gemm"
    SOFTMAX = "Softmax"
    LAYER_NORM = "LayerNormalization"
    GATHER = "Gather"
    MUL = "Mul"
    DIV = "Div"
    SUB = "Sub"
    SIGMOID = "Sigmoid"
    TANH = "Tanh"
    CONCAT = "Concat"
    SQUEEZE = "Squeeze"
    UNSQUEEZE = "Unsqueeze"
    CAST = "Cast"

    # Raw ONNX operators — Tier 2 (model completeness)
    ERF = "Erf"
    POW = "Pow"
    SQRT = "Sqrt"
    REDUCE_MEAN = "ReduceMean"
    REDUCE_SUM = "ReduceSum"
    SLICE = "Slice"
    SPLIT = "Split"
    WHERE = "Where"
    MAX_POOL = "MaxPool"
    AVERAGE_POOL = "AveragePool"
    GLOBAL_AVERAGE_POOL = "GlobalAveragePool"
    PAD = "Pad"
    EXPAND = "Expand"
    TILE = "Tile"
    CONV_TRANSPOSE = "ConvTranspose"
    CLIP = "Clip"
    CONSTANT_OF_SHAPE = "ConstantOfShape"
    SHAPE = "Shape"
    IDENTITY = "Identity"
    CONSTANT = "Constant"
    RANGE = "Range"
    EQUAL = "Equal"
    LESS = "Less"
    GREATER = "Greater"
    NOT = "Not"
    NEG = "Neg"
    SIN = "Sin"
    COS = "Cos"
    TRILU = "Trilu"
    DEQUANTIZE_LINEAR = "DequantizeLinear"
    DYNAMIC_QUANTIZE_LINEAR = "DynamicQuantizeLinear"
    MATMUL_INTEGER = "MatMulInteger"
    SCATTER_ND = "ScatterND"
    IF = "If"

    # Fused operators (after optimization) — existing
    FUSED_LINEAR = "FusedLinear"
    FUSED_LINEAR_RELU = "FusedLinearReLU"

    # Raw ONNX operators — Tier 3 (extended model support)
    GROUP_NORM = "GroupNormalization"
    INSTANCE_NORM = "InstanceNormalization"
    Q_LINEAR_MATMUL = "QLinearMatMul"
    RESIZE = "Resize"

    # Fused operators — Phase 2 placeholders
    FUSED_CONV = "FusedConv"
    FUSED_CONV_RELU = "FusedConvReLU"
    FUSED_CONV_RELU6 = "FusedConvReLU6"
    FUSED_ATTENTION = "FusedAttention"
    FUSED_LAYER_NORM = "FusedLayerNorm"
    FUSED_GELU = "FusedGELU"
    FUSED_SILU = "FusedSiLU"
    FUSED_MLP = "FusedMLP"

    @classmethod
    def from_onnx(cls, op_type_str: str) -> OpType:
        """Convert ONNX op_type string to OpType enum.

        Raises ValueError if the operator is not supported.
        """
        for member in cls:
            if member.value == op_type_str:
                return member
        raise ValueError(f"Unsupported ONNX operator: {op_type_str}")

    @property
    def is_fused(self) -> bool:
        return self.name.startswith("FUSED_")


# ---------------------------------------------------------------------------
# FusedLinearAttrs
# ---------------------------------------------------------------------------


@dataclass
class FusedLinearAttrs:
    """Attributes for FusedLinear and FusedLinearReLU operators."""

    input_dim: int
    output_dim: int
    has_relu: bool

    weight_quant: Optional[QuantParams] = None
    input_quant: Optional[QuantParams] = None
    output_quant: Optional[QuantParams] = None

    requant_scale_fixed: Optional[int] = None
    requant_shift: Optional[int] = None

    @property
    def is_quantized(self) -> bool:
        """Whether quantization params have been assigned."""
        return (
            self.weight_quant is not None
            and self.input_quant is not None
            and self.output_quant is not None
            and self.weight_quant.calibrated
            and self.input_quant.calibrated
            and self.output_quant.calibrated
        )


@dataclass
class FusedConvAttrs:
    """Attributes for FusedConv, FusedConvReLU, FusedConvReLU6 operators."""

    in_channels: int
    out_channels: int
    kernel_shape: list[int]
    strides: list[int] = field(default_factory=lambda: [1, 1])
    pads: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    dilations: list[int] = field(default_factory=lambda: [1, 1])
    group: int = 1
    has_relu: bool = False
    has_relu6: bool = False

    # Per-channel weight quantization (one QuantParams per output channel)
    weight_quant: Optional[list[QuantParams]] = None
    input_quant: Optional[QuantParams] = None
    output_quant: Optional[QuantParams] = None

    # Per-channel requantization: M_fixed[oc] = floor((scale_w[oc]*scale_x/scale_y)*2^16 + 0.5)
    requant_scale_fixed: Optional[list[int]] = None
    requant_shift: Optional[int] = None

    @property
    def is_quantized(self) -> bool:
        """Whether quantization params have been assigned."""
        return (
            self.weight_quant is not None
            and self.input_quant is not None
            and self.output_quant is not None
            and all(wq.calibrated for wq in self.weight_quant)
            and self.input_quant.calibrated
            and self.output_quant.calibrated
        )


@dataclass
class FusedLayerNormAttrs:
    """Attributes for FusedLayerNorm operator."""

    axis: int = -1
    epsilon: float = 1e-5
    normalized_shape: Optional[tuple[int, ...]] = None

    input_quant: Optional[QuantParams] = None
    output_quant: Optional[QuantParams] = None


@dataclass
class FusedActivationAttrs:
    """Attributes for FusedGELU and FusedSiLU operators."""

    activation_type: str  # "gelu" or "silu"

    input_quant: Optional[QuantParams] = None
    output_quant: Optional[QuantParams] = None

    # Optional LUT table for hardware implementation
    lut_table: Optional[np.ndarray] = None


@dataclass
class AttentionAttrs:
    """Attributes for unfused multi-head attention patterns."""

    num_heads: int
    head_dim: int
    seq_len: int = 0  # 0 = dynamic (rejected at validation)
    has_mask: bool = False
    qkv_bias: bool = True


@dataclass
class FusedAttentionAttrs:
    """Attributes for FusedAttention operator (Q/K/V projections + attention + O projection)."""

    num_heads: int
    head_dim: int
    seq_len: int
    has_mask: bool = False
    qkv_bias: bool = True

    # Per-projection quantization
    q_weight_quant: Optional[QuantParams] = None
    k_weight_quant: Optional[QuantParams] = None
    v_weight_quant: Optional[QuantParams] = None
    output_weight_quant: Optional[QuantParams] = None
    input_quant: Optional[QuantParams] = None
    output_quant: Optional[QuantParams] = None
    requant_scale_fixed: Optional[list[int]] = None
    requant_shift: Optional[int] = None

    @property
    def embed_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def is_quantized(self) -> bool:
        return (
            self.q_weight_quant is not None
            and self.k_weight_quant is not None
            and self.v_weight_quant is not None
            and self.output_weight_quant is not None
            and self.input_quant is not None
            and self.output_quant is not None
            and self.q_weight_quant.calibrated
            and self.k_weight_quant.calibrated
            and self.v_weight_quant.calibrated
            and self.output_weight_quant.calibrated
            and self.input_quant.calibrated
            and self.output_quant.calibrated
        )


@dataclass
class FusedMLPAttrs:
    """Attributes for FusedMLP (Linear + Activation + Linear, transformer FFN)."""

    input_dim: int
    hidden_dim: int
    output_dim: int
    activation_type: str  # "gelu", "silu", or "relu"

    # First linear (input_dim -> hidden_dim)
    fc1_weight_quant: Optional[QuantParams] = None
    # Second linear (hidden_dim -> output_dim)
    fc2_weight_quant: Optional[QuantParams] = None
    input_quant: Optional[QuantParams] = None
    mid_quant: Optional[QuantParams] = None  # after activation, before fc2
    output_quant: Optional[QuantParams] = None

    requant_scale_fixed_fc1: Optional[int] = None
    requant_shift_fc1: Optional[int] = None
    requant_scale_fixed_fc2: Optional[int] = None
    requant_shift_fc2: Optional[int] = None

    @property
    def is_quantized(self) -> bool:
        return (
            self.fc1_weight_quant is not None
            and self.fc2_weight_quant is not None
            and self.input_quant is not None
            and self.mid_quant is not None
            and self.output_quant is not None
            and self.fc1_weight_quant.calibrated
            and self.fc2_weight_quant.calibrated
            and self.input_quant.calibrated
            and self.mid_quant.calibrated
            and self.output_quant.calibrated
        )


# ---------------------------------------------------------------------------
# Typed operator attributes
# ---------------------------------------------------------------------------


@dataclass
class ConvAttrs:
    """Attributes for Conv and ConvTranspose operators."""

    kernel_shape: list[int]
    strides: list[int] = field(default_factory=lambda: [1, 1])
    pads: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    dilations: list[int] = field(default_factory=lambda: [1, 1])
    group: int = 1


@dataclass
class GemmAttrs:
    """Attributes for Gemm operator."""

    alpha: float = 1.0
    beta: float = 1.0
    transA: int = 0
    transB: int = 0


@dataclass
class NormAttrs:
    """Attributes for LayerNormalization (and GroupNorm)."""

    axis: int = -1
    epsilon: float = 1e-5
    num_groups: int = 1


@dataclass
class PoolAttrs:
    """Attributes for MaxPool, AveragePool operators."""

    kernel_shape: list[int]
    strides: list[int] = field(default_factory=lambda: [1, 1])
    pads: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    ceil_mode: int = 0


# ---------------------------------------------------------------------------
# HardwareConstraints
# ---------------------------------------------------------------------------


@dataclass
class HardwareConstraints:
    """Target FPGA resource constraints."""

    max_parallelism: int = 128
    clock_mhz: int = 100

    # Weight SRAM: unified bank, 1024-bit wide rows (128 bytes/row)
    weight_bank_depth: int = 1536  # rows
    weight_row_bytes: int = 128

    # Bias SRAM: separate bank, 1024-bit wide rows (32 INT32 biases/row)
    bias_bank_depth: int = 32  # rows
    biases_per_row: int = 32

    # Activation buffer
    act_buffer_bytes: int = 640

    # AXI overhead
    axi_overhead_cycles: int = 160

    target_device: str = "xck26-sfvc784-2LV"

    @property
    def weight_bank_bytes(self) -> int:
        return self.weight_bank_depth * self.weight_row_bytes

    @property
    def bias_bank_bytes(self) -> int:
        return self.bias_bank_depth * self.biases_per_row * 4  # INT32 = 4 bytes


# ---------------------------------------------------------------------------
# LayerSchedule
# ---------------------------------------------------------------------------


@dataclass
class LayerSchedule:
    """Per-layer scheduling info, attached to OpNode.schedule_info."""

    layer_index: int
    layer_name: str
    input_dim: int
    output_dim: int
    parallelism: int
    num_tiles: int
    has_relu: bool

    # Timing
    cycles_per_tile: int
    total_cycles: int
    start_cycle: int
    end_cycle: int

    # Weight SRAM (row-addressed, unified bank)
    weight_start_row: int
    weight_rows: int
    weight_bytes: int

    # Bias SRAM (separate bank)
    bias_start_row: int
    bias_rows: int
    bias_bytes: int

    # Activation ping-pong
    act_in_bank: str  # "A" or "B"
    act_out_bank: str  # "A" or "B"


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


@dataclass
class Schedule:
    """Complete schedule for all layers."""

    layers: list[LayerSchedule]
    total_compute_cycles: int
    total_cycles: int  # compute + AXI overhead
    total_weight_bytes: int
    total_bias_bytes: int
    total_act_bytes: int
    clock_mhz: int
    axi_overhead_cycles: int

    @property
    def latency_us(self) -> float:
        """Total latency in microseconds."""
        return self.total_cycles / self.clock_mhz

    @property
    def latency_ms(self) -> float:
        """Total latency in milliseconds."""
        return self.latency_us / 1000.0

    @property
    def throughput_inferences_per_sec(self) -> float:
        """Maximum throughput in inferences per second."""
        if self.latency_us == 0:
            return 0.0
        return 1_000_000.0 / self.latency_us

    def to_json(self) -> dict:
        """Serialize schedule to JSON-compatible dict."""
        return {
            "clock_mhz": self.clock_mhz,
            "axi_overhead_cycles": self.axi_overhead_cycles,
            "total_compute_cycles": self.total_compute_cycles,
            "total_cycles": self.total_cycles,
            "latency_us": self.latency_us,
            "latency_ms": self.latency_ms,
            "throughput_inferences_per_sec": self.throughput_inferences_per_sec,
            "total_weight_bytes": self.total_weight_bytes,
            "total_bias_bytes": self.total_bias_bytes,
            "total_act_bytes": self.total_act_bytes,
            "layers": [
                {
                    "layer_index": ls.layer_index,
                    "layer_name": ls.layer_name,
                    "input_dim": ls.input_dim,
                    "output_dim": ls.output_dim,
                    "parallelism": ls.parallelism,
                    "num_tiles": ls.num_tiles,
                    "has_relu": ls.has_relu,
                    "cycles_per_tile": ls.cycles_per_tile,
                    "total_cycles": ls.total_cycles,
                    "start_cycle": ls.start_cycle,
                    "end_cycle": ls.end_cycle,
                    "weight_start_row": ls.weight_start_row,
                    "weight_rows": ls.weight_rows,
                    "weight_bytes": ls.weight_bytes,
                    "bias_start_row": ls.bias_start_row,
                    "bias_rows": ls.bias_rows,
                    "bias_bytes": ls.bias_bytes,
                    "act_in_bank": ls.act_in_bank,
                    "act_out_bank": ls.act_out_bank,
                }
                for ls in self.layers
            ],
        }


# ---------------------------------------------------------------------------
# DAGLayerSchedule
# ---------------------------------------------------------------------------


@dataclass
class DAGLayerSchedule:
    """Per-node scheduling info for DAG scheduling (arbitrary topologies)."""

    node_index: int
    node_name: str
    op_type: str  # OpType.value string
    total_cycles: int
    start_cycle: int
    end_cycle: int

    weight_bytes: int = 0
    bias_bytes: int = 0

    input_tensors: list[str] = field(default_factory=list)
    output_tensors: list[str] = field(default_factory=list)
    input_bytes: int = 0
    output_bytes: int = 0


# ---------------------------------------------------------------------------
# OpNode
# ---------------------------------------------------------------------------


@dataclass
class OpNode:
    """An operator node in the IR graph."""

    name: str
    op_type: OpType
    inputs: list[str]
    outputs: list[str]
    attributes: dict[str, Any] = field(default_factory=dict)

    schedule_info: Optional[LayerSchedule] = None

    @property
    def fused_attrs(
        self,
    ) -> (
        FusedLinearAttrs
        | FusedConvAttrs
        | FusedLayerNormAttrs
        | FusedActivationAttrs
        | FusedAttentionAttrs
        | FusedMLPAttrs
        | None
    ):
        """Convenience accessor for fused operator attributes."""
        val = self.attributes.get("fused_attrs")
        if isinstance(
            val,
            (
                FusedLinearAttrs,
                FusedConvAttrs,
                FusedLayerNormAttrs,
                FusedActivationAttrs,
                FusedAttentionAttrs,
                FusedMLPAttrs,
            ),
        ):
            return val
        return None

    @fused_attrs.setter
    def fused_attrs(
        self,
        attrs: (
            FusedLinearAttrs
            | FusedConvAttrs
            | FusedLayerNormAttrs
            | FusedActivationAttrs
            | FusedAttentionAttrs
            | FusedMLPAttrs
        ),
    ) -> None:
        self.attributes["fused_attrs"] = attrs

    @property
    def dag_schedule(self) -> DAGLayerSchedule | None:
        """Convenience accessor for DAG scheduling info."""
        val = self.attributes.get("dag_schedule_info")
        if isinstance(val, DAGLayerSchedule):
            return val
        return None

    @dag_schedule.setter
    def dag_schedule(self, info: DAGLayerSchedule) -> None:
        self.attributes["dag_schedule_info"] = info

    @property
    def op_attrs(self) -> ConvAttrs | GemmAttrs | NormAttrs | PoolAttrs | None:
        """Convenience accessor for typed operator attributes."""
        val = self.attributes.get("op_attrs")
        if isinstance(val, (ConvAttrs, GemmAttrs, NormAttrs, PoolAttrs)):
            return val
        return None

    @op_attrs.setter
    def op_attrs(self, attrs: ConvAttrs | GemmAttrs | NormAttrs | PoolAttrs) -> None:
        self.attributes["op_attrs"] = attrs


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


@dataclass
class Graph:
    """The complete IR graph."""

    name: str
    nodes: dict[str, OpNode]
    tensors: dict[str, Tensor]
    inputs: list[str]
    outputs: list[str]
    stage: str = "raw"

    # Cached derived data (invalidated on mutation)
    _topo_order: list[str] = field(default_factory=list, repr=False)
    _tensor_to_producer: dict[str, str] = field(default_factory=dict, repr=False)
    _tensor_to_consumers: dict[str, list[str]] = field(default_factory=dict, repr=False)

    def invalidate_cache(self) -> None:
        """Invalidate cached derived data. Call after graph mutations."""
        self._topo_order = []
        self._tensor_to_producer = {}
        self._tensor_to_consumers = {}

    def _build_adjacency(self) -> None:
        """Build tensor -> producer/consumer maps."""
        if self._tensor_to_producer or self._tensor_to_consumers:
            return
        self._tensor_to_producer = {}
        self._tensor_to_consumers = {t: [] for t in self.tensors}
        for node_name, node in self.nodes.items():
            for out in node.outputs:
                self._tensor_to_producer[out] = node_name
            for inp in node.inputs:
                if inp in self._tensor_to_consumers:
                    self._tensor_to_consumers[inp].append(node_name)

    def topological_order(self) -> list[str]:
        """Return node names in topological order (Kahn's algorithm)."""
        if self._topo_order:
            return self._topo_order

        self._build_adjacency()

        in_degree: dict[str, int] = {name: 0 for name in self.nodes}
        adj: dict[str, list[str]] = {name: [] for name in self.nodes}

        for node_name, node in self.nodes.items():
            for inp in node.inputs:
                producer = self._tensor_to_producer.get(inp)
                if producer and producer in self.nodes:
                    adj[producer].append(node_name)
                    in_degree[node_name] += 1

        queue = deque(n for n, d in in_degree.items() if d == 0)
        result: list[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.nodes):
            raise ValueError(
                f"Cycle detected in graph: processed {len(result)}/{len(self.nodes)} nodes"
            )

        self._topo_order = result
        return result

    def get_node(self, name: str) -> OpNode:
        return self.nodes[name]

    def get_tensor(self, name: str) -> Tensor:
        return self.tensors[name]

    def get_producer(self, tensor_name: str) -> Optional[OpNode]:
        """Get the single node that produces this tensor, or None."""
        self._build_adjacency()
        producer_name = self._tensor_to_producer.get(tensor_name)
        if producer_name and producer_name in self.nodes:
            return self.nodes[producer_name]
        return None

    def get_producers(self, tensor_name: str) -> list[OpNode]:
        """Get nodes that produce this tensor (0 or 1 for valid graphs)."""
        p = self.get_producer(tensor_name)
        return [p] if p else []

    def get_consumers(self, tensor_name: str) -> list[OpNode]:
        """Get nodes that consume this tensor."""
        self._build_adjacency()
        return [
            self.nodes[n] for n in self._tensor_to_consumers.get(tensor_name, []) if n in self.nodes
        ]

    # ------------------------------------------------------------------
    # Graph mutation helpers (for surgical pass replacement)
    # ------------------------------------------------------------------

    def remove_node(self, name: str) -> None:
        """Remove a node from the graph by name. Invalidates cache."""
        if name in self.nodes:
            del self.nodes[name]
            self.invalidate_cache()

    def insert_node(self, node: OpNode, tensors: Optional[dict[str, "Tensor"]] = None) -> None:
        """Add a node and optional tensors to the graph. Invalidates cache."""
        self.nodes[node.name] = node
        if tensors:
            self.tensors.update(tensors)
        self.invalidate_cache()

    def replace_subgraph(
        self,
        old_names: list[str],
        new_node: OpNode,
        new_tensors: Optional[dict[str, "Tensor"]] = None,
    ) -> None:
        """Replace a set of nodes with a single fused node.

        - Removes old_names nodes
        - Adds new_node (with optional new_tensors)
        - Rewires downstream consumers: any node consuming an output of an old
          node that is NOT an input to another old node gets rewired to consume
          new_node's output instead.
        - Cleans up orphaned intermediate tensors.
        """
        old_set = set(old_names)

        # Collect all outputs of old subgraph
        old_outputs: set[str] = set()
        old_inputs: set[str] = set()
        for name in old_names:
            if name in self.nodes:
                old_outputs.update(self.nodes[name].outputs)
                old_inputs.update(self.nodes[name].inputs)

        # The "external" outputs are old outputs not consumed by another old node
        internal_tensors = old_outputs & old_inputs
        external_outputs = old_outputs - internal_tensors

        # The new node's output replaces external outputs
        new_output = new_node.outputs[0] if new_node.outputs else None

        # Rewire downstream consumers
        if new_output and external_outputs:
            for node in self.nodes.values():
                if node.name in old_set:
                    continue
                node.inputs = [
                    new_output if inp in external_outputs else inp for inp in node.inputs
                ]

            # Rewire graph outputs
            self.outputs = [new_output if out in external_outputs else out for out in self.outputs]

        # Remove old nodes
        for name in old_names:
            if name in self.nodes:
                del self.nodes[name]

        # Add new node and tensors
        self.nodes[new_node.name] = new_node
        if new_tensors:
            self.tensors.update(new_tensors)

        # Cleanup orphaned intermediate tensors
        referenced: set[str] = set()
        referenced.update(self.inputs)
        referenced.update(self.outputs)
        for node in self.nodes.values():
            referenced.update(node.inputs)
            referenced.update(node.outputs)

        orphans = [t for t in list(self.tensors.keys()) if t not in referenced]
        for t in orphans:
            del self.tensors[t]

        self.invalidate_cache()

    def get_subgraph(self, op_types: set[OpType]) -> list[OpNode]:
        """Return nodes of given op_types in topological order."""
        return [
            self.nodes[name]
            for name in self.topological_order()
            if self.nodes[name].op_type in op_types
        ]

    def find_skip_connections(self) -> list[tuple[str, list[str]]]:
        """Detect skip connections: tensors consumed by 2+ nodes where one is Add.

        Returns list of (tensor_name, [consumer_names]) tuples.
        """
        self._build_adjacency()
        result = []
        for tensor_name, consumer_names in self._tensor_to_consumers.items():
            if len(consumer_names) < 2:
                continue
            consumers = [self.nodes[n] for n in consumer_names if n in self.nodes]
            if len(consumers) >= 2 and any(c.op_type == OpType.ADD for c in consumers):
                result.append((tensor_name, [c.name for c in consumers]))
        return result

    def validate(self, stage: Optional[str] = None) -> None:
        """Validate graph invariants for the given stage. Raises IRValidationError."""
        stage = stage or self.stage
        validator = IRValidator()
        dispatch = {
            "raw": validator.validate_stage1,
            "optimized": validator.validate_stage2,
            "fused": validator.validate_stage2d,
            "quantized": validator.validate_stage2e,
            "scheduled": validator.validate_stage3,
            "fused_dag": validator.validate_stage2d_dag,
            "quantized_dag": validator.validate_stage2e_dag,
            "scheduled_dag": validator.validate_stage3_dag,
        }
        validate_fn = dispatch.get(stage, validator.validate_stage1)
        errors = validate_fn(self)
        if errors:
            raise IRValidationError(errors)


# ---------------------------------------------------------------------------
# IRValidator
# ---------------------------------------------------------------------------


class IRValidator:
    """Validate IR invariants by stage."""

    def validate_stage1(self, graph: Graph) -> list[str]:
        """Validate post-parse invariants (INV-1.1 through INV-1.10)."""
        errors: list[str] = []

        # INV-1.1: All tensor names are unique (guaranteed by dict keys, but verify)
        tensor_names = list(graph.tensors.keys())
        if len(tensor_names) != len(set(tensor_names)):
            errors.append("INV-1.1: Duplicate tensor names found")

        # INV-1.2: All node names are unique
        node_names = list(graph.nodes.keys())
        if len(node_names) != len(set(node_names)):
            errors.append("INV-1.2: Duplicate node names found")

        # INV-1.3: All node inputs reference existing tensors
        for node in graph.nodes.values():
            for inp in node.inputs:
                if inp not in graph.tensors:
                    errors.append(f"INV-1.3: Node '{node.name}' references unknown input '{inp}'")

        # INV-1.4: All node outputs reference existing tensors
        for node in graph.nodes.values():
            for out in node.outputs:
                if out not in graph.tensors:
                    errors.append(f"INV-1.4: Node '{node.name}' references unknown output '{out}'")

        # INV-1.5: Graph inputs are tensors with no producer
        for inp in graph.inputs:
            producers = graph.get_producers(inp)
            if producers:
                errors.append(f"INV-1.5: Graph input '{inp}' has producer: {producers[0].name}")

        # INV-1.6: Graph outputs are tensors with a producer
        for out in graph.outputs:
            producers = graph.get_producers(out)
            if len(producers) != 1:
                errors.append(
                    f"INV-1.6: Graph output '{out}' has {len(producers)} producers (expected 1)"
                )

        # INV-1.7: All operators are supported
        for node in graph.nodes.values():
            if not isinstance(node.op_type, OpType):
                errors.append(f"INV-1.7: Unsupported op type at node '{node.name}'")

        # INV-1.8: All constant tensors have data
        for t in graph.tensors.values():
            if t.is_constant and t.data is None:
                errors.append(f"INV-1.8: Constant tensor '{t.name}' has no data")

        # INV-1.9: All tensors have known shape (post shape-inference)
        for t in graph.tensors.values():
            if not t.type.is_shape_known:
                errors.append(f"INV-1.9: Tensor '{t.name}' has unknown shape: {t.type.shape}")

        # INV-1.10: Graph is acyclic
        try:
            graph.topological_order()
        except ValueError as e:
            errors.append(f"INV-1.10: {e}")

        return errors

    def validate_stage2(self, graph: Graph) -> list[str]:
        """Validate post-optimization invariants."""
        errors = self.validate_stage1(graph)

        # INV-2.1: No BatchNormalization nodes
        for node in graph.nodes.values():
            if node.op_type == OpType.BATCH_NORM:
                errors.append(f"INV-2.1: BatchNormalization still present: {node.name}")

        # INV-2.2: No dead nodes
        reachable = self._find_reachable_nodes(graph)
        for name in graph.nodes:
            if name not in reachable:
                errors.append(f"INV-2.2: Dead node found: {name}")

        return errors

    def validate_stage2d(self, graph: Graph) -> list[str]:
        """Validate post-fusion invariants."""
        errors: list[str] = []

        # INV-3.1: Only fused operators
        for node in graph.nodes.values():
            if node.op_type not in {OpType.FUSED_LINEAR, OpType.FUSED_LINEAR_RELU}:
                errors.append(f"INV-3.1: Non-fused operator: {node.op_type} at {node.name}")

        # INV-3.4: FusedLinearAttrs present
        for node in graph.nodes.values():
            if node.fused_attrs is None:
                errors.append(f"INV-3.4: Missing FusedLinearAttrs on {node.name}")

        # INV-3.5: Dimensions chain correctly
        ordered_nodes = [graph.nodes[n] for n in graph.topological_order()]
        for i in range(len(ordered_nodes) - 1):
            curr = ordered_nodes[i].fused_attrs
            nxt = ordered_nodes[i + 1].fused_attrs
            if curr and nxt and curr.output_dim != nxt.input_dim:
                errors.append(
                    f"INV-3.5: Dimension mismatch: {ordered_nodes[i].name} "
                    f"output_dim={curr.output_dim} != "
                    f"{ordered_nodes[i + 1].name} input_dim={nxt.input_dim}"
                )

        return errors

    def validate_stage2e(self, graph: Graph) -> list[str]:
        """Validate post-quantization invariants."""
        errors = self.validate_stage2d(graph)

        for node in graph.nodes.values():
            attrs = node.fused_attrs
            if attrs is None:
                continue
            if not attrs.is_quantized:
                errors.append(f"INV-Q.1: Node {node.name} not fully quantized")
            if attrs.requant_scale_fixed is None or attrs.requant_shift is None:
                errors.append(f"INV-Q.2: Node {node.name} missing requantization params")

        # INV-Q.3: Weight tensors are INT8
        for node in graph.nodes.values():
            if len(node.inputs) >= 2:
                wt = graph.tensors.get(node.inputs[1])
                if wt and wt.is_constant and wt.data is not None and wt.data.dtype != np.int8:
                    errors.append(
                        f"INV-Q.3: Weight tensor '{wt.name}' is {wt.data.dtype}, expected int8"
                    )

        return errors

    # Allowed passthrough ops in DAG-mode graphs (CNN/Transformer pipelines)
    _DAG_PASSTHROUGH_OPS: set[OpType] = {
        OpType.MAX_POOL,
        OpType.AVERAGE_POOL,
        OpType.GLOBAL_AVERAGE_POOL,
        OpType.FLATTEN,
        OpType.RESHAPE,
        OpType.CONCAT,
        OpType.ADD,
        OpType.TRANSPOSE,
        OpType.GATHER,
        OpType.SOFTMAX,
        OpType.MATMUL,
        OpType.MUL,
        OpType.DIV,
        OpType.SUB,
    }

    _DAG_FUSED_OPS: set[OpType] = {
        OpType.FUSED_LINEAR,
        OpType.FUSED_LINEAR_RELU,
        OpType.FUSED_CONV,
        OpType.FUSED_CONV_RELU,
        OpType.FUSED_CONV_RELU6,
        OpType.FUSED_LAYER_NORM,
        OpType.FUSED_GELU,
        OpType.FUSED_SILU,
        OpType.FUSED_ATTENTION,
        OpType.FUSED_MLP,
    }

    def validate_stage2d_dag(self, graph: Graph) -> list[str]:
        """Validate fused DAG invariants (mixed fused + passthrough ops)."""
        errors: list[str] = []

        allowed = self._DAG_FUSED_OPS | self._DAG_PASSTHROUGH_OPS
        for node in graph.nodes.values():
            if node.op_type not in allowed:
                errors.append(f"INV-DAG.1: Disallowed operator {node.op_type.value} at {node.name}")

        # All fused nodes must have fused_attrs
        for node in graph.nodes.values():
            if node.op_type.is_fused and node.fused_attrs is None:
                errors.append(f"INV-DAG.2: Missing fused_attrs on fused node {node.name}")

        # No BatchNormalization remaining
        for node in graph.nodes.values():
            if node.op_type == OpType.BATCH_NORM:
                errors.append(f"INV-DAG.3: BatchNormalization still present: {node.name}")

        # Graph must be acyclic
        try:
            graph.topological_order()
        except ValueError as e:
            errors.append(f"INV-DAG.4: {e}")

        # Tensor references valid
        for node in graph.nodes.values():
            for inp in node.inputs:
                if inp not in graph.tensors:
                    errors.append(f"INV-DAG.5: Node '{node.name}' references unknown input '{inp}'")
            for out in node.outputs:
                if out not in graph.tensors:
                    errors.append(
                        f"INV-DAG.5: Node '{node.name}' references unknown output '{out}'"
                    )

        return errors

    def validate_stage2e_dag(self, graph: Graph) -> list[str]:
        """Validate quantized DAG invariants (fused_dag + quantization checks)."""
        errors = self.validate_stage2d_dag(graph)

        # All fused conv nodes must be quantized
        conv_fused = {OpType.FUSED_CONV, OpType.FUSED_CONV_RELU, OpType.FUSED_CONV_RELU6}
        for node in graph.nodes.values():
            if node.op_type in conv_fused:
                attrs = node.fused_attrs
                if attrs is None or not isinstance(attrs, FusedConvAttrs):
                    errors.append(f"INV-QDAG.1: Conv node {node.name} missing FusedConvAttrs")
                elif not attrs.is_quantized:
                    errors.append(f"INV-QDAG.2: Conv node {node.name} not fully quantized")

        # Weight tensors for fused conv nodes must be INT8
        for node in graph.nodes.values():
            if node.op_type in conv_fused and len(node.inputs) >= 2:
                wt = graph.tensors.get(node.inputs[1])
                if wt and wt.is_constant and wt.data is not None and wt.data.dtype != np.int8:
                    errors.append(
                        f"INV-QDAG.3: Weight tensor '{wt.name}' is {wt.data.dtype}, expected int8"
                    )

        return errors

    def validate_stage3_dag(self, graph: Graph) -> list[str]:
        """Validate post-DAG-scheduling invariants (INV-DAG-S.1 through INV-DAG-S.4)."""
        errors = self.validate_stage2e_dag(graph)

        ordered_names = graph.topological_order()

        # INV-DAG-S.1: All nodes have a DAGLayerSchedule
        for name in ordered_names:
            node = graph.nodes[name]
            if node.dag_schedule is None:
                errors.append(f"INV-DAG-S.1: Node {name} missing DAGLayerSchedule")

        scheduled_nodes = [
            graph.nodes[n] for n in ordered_names if graph.nodes[n].dag_schedule is not None
        ]
        if not scheduled_nodes:
            return errors

        # INV-DAG-S.2: Topo order respected (consumer starts after producers)
        node_end: dict[str, int] = {}
        for node in scheduled_nodes:
            ds = node.dag_schedule
            node_end[node.name] = ds.end_cycle

        for node in scheduled_nodes:
            ds = node.dag_schedule
            for inp in node.inputs:
                producer = graph.get_producer(inp)
                if producer and producer.name in node_end:
                    if ds.start_cycle < node_end[producer.name]:
                        errors.append(
                            f"INV-DAG-S.2: Node {node.name} starts at cycle {ds.start_cycle} "
                            f"before producer {producer.name} ends at {node_end[producer.name]}"
                        )

        # INV-DAG-S.3: No timing overlaps (layer-sequential execution)
        schedules_by_start = sorted(scheduled_nodes, key=lambda n: n.dag_schedule.start_cycle)
        for i in range(len(schedules_by_start) - 1):
            curr = schedules_by_start[i].dag_schedule
            nxt = schedules_by_start[i + 1].dag_schedule
            if curr.end_cycle > nxt.start_cycle:
                errors.append(
                    f"INV-DAG-S.3: Timing overlap: {curr.node_name} ends at {curr.end_cycle}, "
                    f"{nxt.node_name} starts at {nxt.start_cycle}"
                )

        # INV-DAG-S.4: All fused nodes must be quantized
        conv_fused = {OpType.FUSED_CONV, OpType.FUSED_CONV_RELU, OpType.FUSED_CONV_RELU6}
        linear_fused = {OpType.FUSED_LINEAR, OpType.FUSED_LINEAR_RELU}
        for node in scheduled_nodes:
            if node.op_type in conv_fused | linear_fused:
                attrs = node.fused_attrs
                if attrs is None or not attrs.is_quantized:
                    errors.append(f"INV-DAG-S.4: Fused node {node.name} not quantized")

        return errors

    def validate_stage3(self, graph: Graph) -> list[str]:
        """Validate post-scheduling invariants (INV-4.1 through INV-4.6)."""
        errors = self.validate_stage2e(graph)

        ordered_names = graph.topological_order()
        ordered_nodes = [graph.nodes[n] for n in ordered_names]

        # INV-4.1: Every fused node has a LayerSchedule
        for node in ordered_nodes:
            if node.schedule_info is None:
                errors.append(f"INV-4.1: Node {node.name} missing LayerSchedule")

        scheduled_nodes = [n for n in ordered_nodes if n.schedule_info is not None]
        if not scheduled_nodes:
            return errors

        # INV-4.2: Layer indices are sequential starting from 0
        indices = [n.schedule_info.layer_index for n in scheduled_nodes]
        if indices != list(range(len(indices))):
            errors.append(f"INV-4.2: Layer indices not sequential: {indices}")

        # INV-4.3: Activation banks alternate (ping-pong)
        for i, node in enumerate(scheduled_nodes):
            ls = node.schedule_info
            expected_in = "A" if i % 2 == 0 else "B"
            expected_out = "B" if i % 2 == 0 else "A"
            if ls.act_in_bank != expected_in or ls.act_out_bank != expected_out:
                errors.append(
                    f"INV-4.3: Layer {i} activation banks "
                    f"{ls.act_in_bank}->{ls.act_out_bank}, "
                    f"expected {expected_in}->{expected_out}"
                )

        # INV-4.4: Weight SRAM rows are sequential (no gaps/overlap)
        # Check sequentiality: each layer's start_row == previous end
        for i in range(1, len(scheduled_nodes)):
            prev = scheduled_nodes[i - 1].schedule_info
            curr = scheduled_nodes[i].schedule_info
            expected_start = prev.weight_start_row + prev.weight_rows
            if curr.weight_start_row != expected_start:
                errors.append(
                    f"INV-4.4: Weight row gap at layer {i}: "
                    f"expected start {expected_start}, got {curr.weight_start_row}"
                )

        # INV-4.5: Parallelism divides output_dim for each layer
        for node in scheduled_nodes:
            ls = node.schedule_info
            if ls.output_dim % ls.parallelism != 0:
                errors.append(
                    f"INV-4.5: Layer {ls.layer_index} parallelism {ls.parallelism} "
                    f"does not divide output_dim {ls.output_dim}"
                )

        # INV-4.6: All fused attrs are quantized
        for node in scheduled_nodes:
            attrs = node.fused_attrs
            if attrs and not attrs.is_quantized:
                errors.append(f"INV-4.6: Node {node.name} not quantized before scheduling")

        return errors

    def _find_reachable_nodes(self, graph: Graph) -> set[str]:
        """Find all nodes reachable backward from graph outputs."""
        reachable: set[str] = set()
        worklist = list(graph.outputs)
        visited_tensors: set[str] = set()

        while worklist:
            tensor_name = worklist.pop()
            if tensor_name in visited_tensors:
                continue
            visited_tensors.add(tensor_name)

            producer = graph.get_producer(tensor_name)
            if producer and producer.name not in reachable:
                reachable.add(producer.name)
                for inp in producer.inputs:
                    worklist.append(inp)

        return reachable
