"""MLASIC Tile Mapper — map scheduled DAG nodes to ASIC tiles.

Assigns each node to one or more hardware tiles, partitions large weights
across tiles, and routes data between tiles. ROM/ASIC path: weights are
hardcoded in per-tile ROM, no DRAM streaming.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from mlasic.ir import (
    FusedConvAttrs,
    FusedLinearAttrs,
    Graph,
    OpType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TileType
# ---------------------------------------------------------------------------


class TileType(Enum):
    """Hardware tile types in the ASIC fabric."""

    MAC = "mac"  # Matrix multiply-accumulate (linear, conv)
    ALU = "alu"  # Elementwise operations (add, mul, etc.)
    NORM = "norm"  # Normalization (LayerNorm, BatchNorm)
    SOFTMAX = "softmax"
    ACTIVATION = "activation"  # LUT-based activations (GELU, SiLU, ReLU)
    POOL = "pool"  # Pooling operations
    RESHAPE = "reshape"  # Zero-cost reshape/transpose


# ---------------------------------------------------------------------------
# Op → TileType registry
# ---------------------------------------------------------------------------

OP_TO_TILE: dict[OpType, TileType] = {
    # MAC tiles
    OpType.FUSED_LINEAR: TileType.MAC,
    OpType.FUSED_LINEAR_RELU: TileType.MAC,
    OpType.FUSED_CONV: TileType.MAC,
    OpType.FUSED_CONV_RELU: TileType.MAC,
    OpType.FUSED_CONV_RELU6: TileType.MAC,
    OpType.MATMUL: TileType.MAC,
    OpType.GEMM: TileType.MAC,
    OpType.MATMUL_INTEGER: TileType.MAC,
    OpType.CONV: TileType.MAC,
    OpType.CONV_TRANSPOSE: TileType.MAC,
    # ALU tiles
    OpType.ADD: TileType.ALU,
    OpType.MUL: TileType.ALU,
    OpType.SUB: TileType.ALU,
    OpType.DIV: TileType.ALU,
    OpType.POW: TileType.ALU,
    OpType.SQRT: TileType.ALU,
    OpType.NEG: TileType.ALU,
    OpType.CLIP: TileType.ALU,
    OpType.WHERE: TileType.ALU,
    OpType.EQUAL: TileType.ALU,
    OpType.LESS: TileType.ALU,
    OpType.GREATER: TileType.ALU,
    OpType.NOT: TileType.ALU,
    OpType.DEQUANTIZE_LINEAR: TileType.ALU,
    OpType.DYNAMIC_QUANTIZE_LINEAR: TileType.ALU,
    OpType.SCATTER_ND: TileType.ALU,
    OpType.TRILU: TileType.ALU,
    # Norm tiles
    OpType.LAYER_NORM: TileType.NORM,
    OpType.FUSED_LAYER_NORM: TileType.NORM,
    OpType.BATCH_NORM: TileType.NORM,
    OpType.REDUCE_MEAN: TileType.NORM,
    OpType.REDUCE_SUM: TileType.NORM,
    # Softmax tile
    OpType.SOFTMAX: TileType.SOFTMAX,
    # Activation tiles
    OpType.RELU: TileType.ACTIVATION,
    OpType.SIGMOID: TileType.ACTIVATION,
    OpType.TANH: TileType.ACTIVATION,
    OpType.ERF: TileType.ACTIVATION,
    OpType.SIN: TileType.ACTIVATION,
    OpType.COS: TileType.ACTIVATION,
    OpType.FUSED_GELU: TileType.ACTIVATION,
    OpType.FUSED_SILU: TileType.ACTIVATION,
    # Pool tiles
    OpType.MAX_POOL: TileType.POOL,
    OpType.AVERAGE_POOL: TileType.POOL,
    OpType.GLOBAL_AVERAGE_POOL: TileType.POOL,
    # Reshape tiles (zero-cost)
    OpType.RESHAPE: TileType.RESHAPE,
    OpType.FLATTEN: TileType.RESHAPE,
    OpType.TRANSPOSE: TileType.RESHAPE,
    OpType.SQUEEZE: TileType.RESHAPE,
    OpType.UNSQUEEZE: TileType.RESHAPE,
    OpType.GATHER: TileType.RESHAPE,
    OpType.CONCAT: TileType.RESHAPE,
    OpType.CAST: TileType.RESHAPE,
    OpType.SLICE: TileType.RESHAPE,
    OpType.SPLIT: TileType.RESHAPE,
    OpType.SHAPE: TileType.RESHAPE,
    OpType.IDENTITY: TileType.RESHAPE,
    OpType.CONSTANT: TileType.RESHAPE,
    OpType.CONSTANT_OF_SHAPE: TileType.RESHAPE,
    OpType.RANGE: TileType.RESHAPE,
    OpType.EXPAND: TileType.RESHAPE,
    OpType.PAD: TileType.RESHAPE,
    OpType.TILE: TileType.RESHAPE,
    # Control flow
    OpType.IF: TileType.ALU,
}


# ---------------------------------------------------------------------------
# TileConstraints
# ---------------------------------------------------------------------------


@dataclass
class TileConstraints:
    """Physical constraints for tile assignment."""

    max_weight_rom_bytes: int = 256 * 1024  # 256 KB per tile
    max_bias_rom_bytes: int = 16 * 1024  # 16 KB per tile
    mac_parallelism: int = 128


# ---------------------------------------------------------------------------
# TileConfig
# ---------------------------------------------------------------------------


@dataclass
class TileConfig:
    """Configuration for a single hardware tile."""

    tile_id: int
    tile_type: TileType
    operator_assignment: str  # node name
    weight_bytes: int = 0
    bias_bytes: int = 0
    input_routes: list[int] = field(default_factory=list)  # source tile IDs
    output_routes: list[int] = field(default_factory=list)  # destination tile IDs
    partition_info: Optional[dict] = None  # for weight-partitioned tiles

    def to_json(self) -> dict:
        return {
            "tile_id": self.tile_id,
            "tile_type": self.tile_type.value,
            "operator_assignment": self.operator_assignment,
            "weight_bytes": self.weight_bytes,
            "bias_bytes": self.bias_bytes,
            "input_routes": self.input_routes,
            "output_routes": self.output_routes,
            "partition_info": self.partition_info,
        }


# ---------------------------------------------------------------------------
# RouteSegment
# ---------------------------------------------------------------------------


@dataclass
class RouteSegment:
    """A data route between two tiles."""

    src_tile: int
    dst_tile: int
    tensor_name: str
    latency_cycles: int  # 0 for adjacent, 1 per hop for distant


# ---------------------------------------------------------------------------
# FabricConfig
# ---------------------------------------------------------------------------


@dataclass
class FabricConfig:
    """Complete tile fabric configuration."""

    tiles: list[TileConfig]
    routes: list[RouteSegment]
    total_weight_bytes: int
    total_bias_bytes: int
    total_tiles: int

    def to_json(self) -> dict:
        return {
            "total_tiles": self.total_tiles,
            "total_weight_bytes": self.total_weight_bytes,
            "total_bias_bytes": self.total_bias_bytes,
            "tiles": [t.to_json() for t in self.tiles],
            "routes": [
                {
                    "src_tile": r.src_tile,
                    "dst_tile": r.dst_tile,
                    "tensor_name": r.tensor_name,
                    "latency_cycles": r.latency_cycles,
                }
                for r in self.routes
            ],
        }


# ---------------------------------------------------------------------------
# WeightPartitioner
# ---------------------------------------------------------------------------


class WeightPartitioner:
    """Split large weights across multiple tiles along output dimension.

    Each partition tile computes independent output neurons/channels.
    No cross-tile reduction needed.
    """

    def __init__(self, constraints: TileConstraints):
        self.constraints = constraints

    def partition(
        self,
        node_name: str,
        weight_bytes: int,
        bias_bytes: int,
        graph: Graph,
    ) -> list[dict]:
        """Determine how to partition weights across tiles.

        Returns a list of partition dicts:
          [{"output_range": (start, end), "weight_bytes": ..., "bias_bytes": ...}, ...]
        """
        node = graph.nodes[node_name]
        attrs = node.fused_attrs

        # Only include bias if the node actually has bias data
        has_bias = bias_bytes > 0

        if isinstance(attrs, FusedLinearAttrs):
            output_dim = attrs.output_dim
            input_dim = attrs.input_dim
            bytes_per_output = input_dim  # INT8 weight per input
            bias_bytes_per_output = 4 if has_bias else 0  # INT32
        elif isinstance(attrs, FusedConvAttrs):
            output_dim = attrs.out_channels
            # weight per output channel = IC * KH * KW
            ic = attrs.in_channels // attrs.group
            kh, kw = attrs.kernel_shape
            bytes_per_output = ic * kh * kw  # INT8
            bias_bytes_per_output = 4 if has_bias else 0  # INT32
        else:
            return [
                {
                    "output_range": (0, 0),
                    "weight_bytes": weight_bytes,
                    "bias_bytes": bias_bytes,
                }
            ]

        # How many output neurons fit in one tile ROM?
        max_outputs_per_tile = self.constraints.max_weight_rom_bytes // max(bytes_per_output, 1)
        if max_outputs_per_tile <= 0:
            max_outputs_per_tile = 1

        num_partitions = math.ceil(output_dim / max_outputs_per_tile)

        partitions = []
        for i in range(num_partitions):
            start = i * max_outputs_per_tile
            end = min((i + 1) * max_outputs_per_tile, output_dim)
            count = end - start
            partitions.append(
                {
                    "output_range": (start, end),
                    "weight_bytes": count * bytes_per_output,
                    "bias_bytes": count * bias_bytes_per_output,
                }
            )

        return partitions


# ---------------------------------------------------------------------------
# StaticRouter
# ---------------------------------------------------------------------------


class StaticRouter:
    """Route data between tiles. Adjacent = 0 latency, distant = 1 cycle/hop."""

    def route(
        self,
        tiles: list[TileConfig],
        graph: Graph,
    ) -> list[RouteSegment]:
        """Compute routes between tiles based on graph tensor edges."""
        # Build mapping: node_name -> list of tile_ids
        node_to_tiles: dict[str, list[int]] = {}
        for tile in tiles:
            node_to_tiles.setdefault(tile.operator_assignment, []).append(tile.tile_id)

        # Build mapping: tensor -> producer node, consumer nodes
        graph._build_adjacency()

        routes: list[RouteSegment] = []
        seen: set[tuple[int, int, str]] = set()

        for node_name, node in graph.nodes.items():
            for out_name in node.outputs:
                consumers = graph.get_consumers(out_name)
                src_tiles = node_to_tiles.get(node_name, [])
                for consumer in consumers:
                    dst_tiles = node_to_tiles.get(consumer.name, [])
                    for src_id in src_tiles:
                        for dst_id in dst_tiles:
                            key = (src_id, dst_id, out_name)
                            if key in seen:
                                continue
                            seen.add(key)
                            # Adjacent tiles (sequential IDs) = 0 latency
                            distance = abs(dst_id - src_id)
                            latency = 0 if distance <= 1 else distance - 1
                            routes.append(
                                RouteSegment(
                                    src_tile=src_id,
                                    dst_tile=dst_id,
                                    tensor_name=out_name,
                                    latency_cycles=latency,
                                )
                            )

        return routes


# ---------------------------------------------------------------------------
# TileMapper
# ---------------------------------------------------------------------------


class TileMapper:
    """Map scheduled DAG nodes to hardware tiles.

    For each node:
      1. Look up TileType from OP_TO_TILE
      2. If weight fits single tile ROM, assign 1 tile
      3. Otherwise WeightPartitioner splits along output dim
      4. StaticRouter traces tensor edges for routing
    """

    def __init__(self, constraints: Optional[TileConstraints] = None):
        self.constraints = constraints or TileConstraints()
        self.partitioner = WeightPartitioner(self.constraints)
        self.router = StaticRouter()

    def map(self, graph: Graph, schedule=None) -> FabricConfig:
        """Map graph nodes to tiles.

        Args:
            graph: The scheduled DAG graph.
            schedule: Optional DAGSchedule (used for weight/bias bytes).

        Returns:
            FabricConfig with tile assignments and routes.
        """
        tiles: list[TileConfig] = []
        tile_id = 0

        execution_order = graph.topological_order()

        for node_name in execution_order:
            node = graph.nodes[node_name]

            tile_type = OP_TO_TILE.get(node.op_type)
            if tile_type is None:
                logger.warning("No tile mapping for %s, skipping", node.op_type.value)
                continue

            # Get weight/bias bytes from dag_schedule or fused_attrs
            weight_bytes = 0
            bias_bytes = 0
            if node.dag_schedule is not None:
                weight_bytes = node.dag_schedule.weight_bytes
                bias_bytes = node.dag_schedule.bias_bytes

            # Check if weight needs partitioning
            needs_partition = (
                tile_type == TileType.MAC and weight_bytes > self.constraints.max_weight_rom_bytes
            )

            if needs_partition:
                partitions = self.partitioner.partition(node_name, weight_bytes, bias_bytes, graph)
                for part in partitions:
                    tiles.append(
                        TileConfig(
                            tile_id=tile_id,
                            tile_type=tile_type,
                            operator_assignment=node_name,
                            weight_bytes=part["weight_bytes"],
                            bias_bytes=part["bias_bytes"],
                            partition_info=part,
                        )
                    )
                    tile_id += 1
            else:
                tiles.append(
                    TileConfig(
                        tile_id=tile_id,
                        tile_type=tile_type,
                        operator_assignment=node_name,
                        weight_bytes=weight_bytes,
                        bias_bytes=bias_bytes,
                    )
                )
                tile_id += 1

        # Route data between tiles
        routes = self.router.route(tiles, graph)

        # Update tile input/output routes
        tile_map = {t.tile_id: t for t in tiles}
        for route in routes:
            if route.src_tile in tile_map:
                tile_map[route.src_tile].output_routes.append(route.dst_tile)
            if route.dst_tile in tile_map:
                tile_map[route.dst_tile].input_routes.append(route.src_tile)

        total_weight = sum(t.weight_bytes for t in tiles)
        total_bias = sum(t.bias_bytes for t in tiles)

        config = FabricConfig(
            tiles=tiles,
            routes=routes,
            total_weight_bytes=total_weight,
            total_bias_bytes=total_bias,
            total_tiles=len(tiles),
        )

        logger.info("Tile mapping: %d tiles, %d routes", len(tiles), len(routes))
        return config
