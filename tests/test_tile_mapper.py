"""Tests for Tile Mapper (Phase 3 tile mapping)."""

from __future__ import annotations

import numpy as np

from mlasic.dag_scheduler import DAGScheduler
from mlasic.ir import (
    FusedConvAttrs,
    FusedLinearAttrs,
    Graph,
    OpNode,
    OpType,
    QuantParams,
    Tensor,
    TensorType,
)
from mlasic.tile_mapper import (
    OP_TO_TILE,
    StaticRouter,
    TileConfig,
    TileConstraints,
    TileMapper,
    TileType,
    WeightPartitioner,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_quantized_linear_node(
    name, input_name, output_name, input_dim, output_dim, has_relu=True
):
    qp = QuantParams(scale=0.1, zero_point=0, calibrated=True)
    attrs = FusedLinearAttrs(
        input_dim=input_dim,
        output_dim=output_dim,
        has_relu=has_relu,
        weight_quant=qp,
        input_quant=qp,
        output_quant=qp,
        requant_scale_fixed=65536,
        requant_shift=16,
    )
    op = OpType.FUSED_LINEAR_RELU if has_relu else OpType.FUSED_LINEAR
    w_name = f"{name}_w"
    b_name = f"{name}_b"
    node = OpNode(name, op, [input_name, w_name, b_name], [output_name])
    node.fused_attrs = attrs
    tensors = {
        input_name: Tensor(input_name, TensorType((1, input_dim), np.dtype(np.int8))),
        w_name: Tensor(
            w_name,
            TensorType((input_dim, output_dim), np.dtype(np.int8)),
            data=np.zeros((input_dim, output_dim), dtype=np.int8),
        ),
        b_name: Tensor(
            b_name,
            TensorType((output_dim,), np.dtype(np.int32)),
            data=np.zeros(output_dim, dtype=np.int32),
        ),
        output_name: Tensor(output_name, TensorType((1, output_dim), np.dtype(np.int8))),
    }
    return node, tensors


def _make_scheduled_linear_chain():
    """Build and schedule a 2-layer linear chain."""
    n0, t0 = _make_quantized_linear_node("n0", "x", "h0", 128, 64)
    n1, t1 = _make_quantized_linear_node("n1", "h0", "y", 64, 32, has_relu=False)
    all_tensors = {**t0, **t1}
    graph = Graph(
        "test_chain",
        {"n0": n0, "n1": n1},
        all_tensors,
        ["x"],
        ["y"],
        stage="quantized_dag",
    )
    sched = DAGScheduler().schedule(graph)
    return graph, sched


# ---------------------------------------------------------------------------
# TestTileType
# ---------------------------------------------------------------------------


class TestTileType:
    def test_op_to_tile_coverage(self):
        """All expected ops have tile mappings."""
        assert OpType.FUSED_LINEAR in OP_TO_TILE
        assert OpType.FUSED_CONV_RELU in OP_TO_TILE
        assert OpType.ADD in OP_TO_TILE
        assert OpType.FLATTEN in OP_TO_TILE

    def test_mac_tile_types(self):
        assert OP_TO_TILE[OpType.FUSED_LINEAR] == TileType.MAC
        assert OP_TO_TILE[OpType.FUSED_CONV] == TileType.MAC
        assert OP_TO_TILE[OpType.MATMUL] == TileType.MAC

    def test_alu_tile_types(self):
        assert OP_TO_TILE[OpType.ADD] == TileType.ALU
        assert OP_TO_TILE[OpType.MUL] == TileType.ALU


# ---------------------------------------------------------------------------
# TestTileMapper
# ---------------------------------------------------------------------------


class TestTileMapper:
    def test_linear_chain_mapping(self):
        """Two-layer linear chain maps to 2 MAC tiles."""
        graph, sched = _make_scheduled_linear_chain()
        mapper = TileMapper()
        fabric = mapper.map(graph, sched)

        assert fabric.total_tiles == 2
        assert all(t.tile_type == TileType.MAC for t in fabric.tiles)
        assert fabric.tiles[0].tile_id == 0
        assert fabric.tiles[1].tile_id == 1

    def test_tile_ids_sequential(self):
        """Tile IDs are assigned sequentially."""
        graph, sched = _make_scheduled_linear_chain()
        mapper = TileMapper()
        fabric = mapper.map(graph, sched)

        ids = [t.tile_id for t in fabric.tiles]
        assert ids == list(range(len(ids)))

    def test_weight_bytes_from_schedule(self):
        """Tile weight/bias bytes come from dag_schedule."""
        graph, sched = _make_scheduled_linear_chain()
        mapper = TileMapper()
        fabric = mapper.map(graph, sched)

        # n0: 128*64 = 8192 weights
        assert fabric.tiles[0].weight_bytes == 8192
        # n1: 64*32 = 2048 weights
        assert fabric.tiles[1].weight_bytes == 2048

    def test_mixed_ops_mapping(self):
        """Graph with conv + add maps to MAC + ALU tiles."""
        qp_list = [QuantParams(scale=0.1, zero_point=0, calibrated=True)] * 16
        qp = QuantParams(scale=0.1, zero_point=0, calibrated=True)
        attrs = FusedConvAttrs(
            in_channels=16,
            out_channels=16,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            has_relu=True,
            weight_quant=qp_list,
            input_quant=qp,
            output_quant=qp,
            requant_scale_fixed=[65536] * 16,
            requant_shift=16,
        )
        n_conv = OpNode("conv", OpType.FUSED_CONV_RELU, ["x", "w", "b"], ["conv_out"])
        n_conv.fused_attrs = attrs

        n_add = OpNode("add", OpType.ADD, ["x", "conv_out"], ["y"])

        tensors = {
            "x": Tensor("x", TensorType((1, 16, 8, 8), np.dtype(np.int8))),
            "w": Tensor(
                "w",
                TensorType((16, 16, 3, 3), np.dtype(np.int8)),
                data=np.zeros((16, 16, 3, 3), dtype=np.int8),
            ),
            "b": Tensor(
                "b",
                TensorType((16,), np.dtype(np.int32)),
                data=np.zeros(16, dtype=np.int32),
            ),
            "conv_out": Tensor("conv_out", TensorType((1, 16, 8, 8), np.dtype(np.int8))),
            "y": Tensor("y", TensorType((1, 16, 8, 8), np.dtype(np.int8))),
        }
        graph = Graph(
            "test",
            {"conv": n_conv, "add": n_add},
            tensors,
            ["x"],
            ["y"],
            stage="quantized_dag",
        )
        DAGScheduler().schedule(graph)
        fabric = TileMapper().map(graph)

        types = [t.tile_type for t in fabric.tiles]
        assert TileType.MAC in types
        assert TileType.ALU in types

    def test_fabric_config_json(self):
        """FabricConfig.to_json() returns valid structure."""
        graph, sched = _make_scheduled_linear_chain()
        fabric = TileMapper().map(graph, sched)
        data = fabric.to_json()

        assert "tiles" in data
        assert "routes" in data
        assert "total_tiles" in data
        assert data["total_tiles"] == 2

    def test_routes_exist_between_connected_tiles(self):
        """Connected nodes have routes between their tiles."""
        graph, sched = _make_scheduled_linear_chain()
        fabric = TileMapper().map(graph, sched)

        # n0 -> n1 via tensor h0
        route_tensors = [r.tensor_name for r in fabric.routes]
        assert "h0" in route_tensors


# ---------------------------------------------------------------------------
# TestWeightPartitioner
# ---------------------------------------------------------------------------


class TestWeightPartitioner:
    def test_no_partition_needed(self):
        """Small weight fits in single tile — 1 partition."""
        n, t = _make_quantized_linear_node("n0", "x", "y", 64, 32)
        graph = Graph("test", {"n0": n}, t, ["x"], ["y"], stage="quantized_dag")
        constraints = TileConstraints(max_weight_rom_bytes=256 * 1024)
        partitioner = WeightPartitioner(constraints)
        parts = partitioner.partition("n0", 2048, 128, graph)
        assert len(parts) == 1
        assert parts[0]["output_range"] == (0, 32)

    def test_partition_large_weight(self):
        """Large weight split across multiple tiles."""
        n, t = _make_quantized_linear_node("n0", "x", "y", 1024, 1024)
        graph = Graph("test", {"n0": n}, t, ["x"], ["y"], stage="quantized_dag")
        # Force small ROM: 128 KB can hold 128K / 1024 bytes_per_output = 128 outputs
        constraints = TileConstraints(max_weight_rom_bytes=128 * 1024)
        partitioner = WeightPartitioner(constraints)
        parts = partitioner.partition("n0", 1024 * 1024, 4096, graph)
        # 1024 outputs / 128 per tile = 8 partitions
        assert len(parts) == 8
        # Verify ranges cover all outputs
        ranges = [p["output_range"] for p in parts]
        assert ranges[0][0] == 0
        assert ranges[-1][1] == 1024

    def test_partition_preserves_total(self):
        """Total bytes across partitions equals original."""
        n, t = _make_quantized_linear_node("n0", "x", "y", 256, 512)
        graph = Graph("test", {"n0": n}, t, ["x"], ["y"], stage="quantized_dag")
        constraints = TileConstraints(max_weight_rom_bytes=64 * 1024)
        partitioner = WeightPartitioner(constraints)
        parts = partitioner.partition("n0", 256 * 512, 512 * 4, graph)
        total_w = sum(p["weight_bytes"] for p in parts)
        assert total_w == 256 * 512


# ---------------------------------------------------------------------------
# TestStaticRouter
# ---------------------------------------------------------------------------


class TestStaticRouter:
    def test_adjacent_tiles_zero_latency(self):
        """Adjacent tiles (sequential IDs) have 0 latency."""
        graph, _ = _make_scheduled_linear_chain()
        tiles = [
            TileConfig(0, TileType.MAC, "n0"),
            TileConfig(1, TileType.MAC, "n1"),
        ]
        router = StaticRouter()
        routes = router.route(tiles, graph)

        for r in routes:
            if r.src_tile == 0 and r.dst_tile == 1:
                assert r.latency_cycles == 0

    def test_distant_tiles_have_latency(self):
        """Non-adjacent tiles have hop latency."""
        # Create a graph where n0 feeds n1 but tiles are far apart
        n0, t0 = _make_quantized_linear_node("n0", "x", "h0", 64, 32)
        n1, t1 = _make_quantized_linear_node("n1", "h0", "y", 32, 16, has_relu=False)
        all_t = {**t0, **t1}
        graph = Graph("test", {"n0": n0, "n1": n1}, all_t, ["x"], ["y"], stage="quantized_dag")

        tiles = [
            TileConfig(0, TileType.MAC, "n0"),
            TileConfig(5, TileType.MAC, "n1"),  # 5 hops away
        ]
        router = StaticRouter()
        routes = router.route(tiles, graph)

        h0_routes = [r for r in routes if r.tensor_name == "h0"]
        assert len(h0_routes) == 1
        assert h0_routes[0].latency_cycles == 4  # |5-0| - 1 = 4


# ---------------------------------------------------------------------------
# TestTileMapperWithPartitioning
# ---------------------------------------------------------------------------


class TestTileMapperWithPartitioning:
    def test_large_weight_creates_multiple_tiles(self):
        """Weight larger than tile ROM creates multiple tiles for same node."""
        n, t = _make_quantized_linear_node("n0", "x", "y", 1024, 1024)
        graph = Graph("test", {"n0": n}, t, ["x"], ["y"], stage="quantized_dag")
        DAGScheduler().schedule(graph)

        # Very small ROM to force partitioning
        constraints = TileConstraints(max_weight_rom_bytes=128 * 1024)
        mapper = TileMapper(constraints)
        fabric = mapper.map(graph)

        # Should have multiple tiles for n0
        n0_tiles = [t for t in fabric.tiles if t.operator_assignment == "n0"]
        assert len(n0_tiles) > 1
