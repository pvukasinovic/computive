"""Tests for Stage 3 dataflow scheduling."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mlasic.ingestion import ONNXParser
from mlasic.ir import (
    FusedLinearAttrs,
    Graph,
    HardwareConstraints,
    OpNode,
    OpType,
    QuantParams,
    Schedule,
    Tensor,
    TensorType,
)
from mlasic.optimization import (
    BatchNormFoldingPass,
    OperatorFusionPass,
    PassManager,
    QuantizationPass,
)
from mlasic.scheduler import CycleBreakdown, Scheduler, schedule_to_json

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def quantized_ad_graph(ad_model_path) -> Graph:
    """Full pipeline: parse → BN fold → fusion → quantization."""
    parser = ONNXParser(ad_model_path)
    graph = parser.parse()
    rng = np.random.RandomState(123)
    calib = [rng.randn(1, 640).astype(np.float32) for _ in range(20)]

    pm = PassManager()
    pm.add_pass(BatchNormFoldingPass())
    pm.add_pass(OperatorFusionPass())
    pm.add_pass(QuantizationPass(calib))
    return pm.run(graph, verify=False)


@pytest.fixture
def ad_schedule(quantized_ad_graph) -> tuple[Schedule, Graph]:
    """Schedule the AD model and return (schedule, graph)."""
    scheduler = Scheduler()
    sched = scheduler.schedule(quantized_ad_graph)
    return sched, quantized_ad_graph


# ---------------------------------------------------------------------------
# CycleBreakdown tests
# ---------------------------------------------------------------------------


class TestCycleBreakdown:
    def test_default_overhead(self):
        cb = CycleBreakdown()
        # 128 + 1 + 3 + 16 + 1 = 149
        assert cb.overhead == 149

    def test_cycles_per_tile_640(self):
        cb = CycleBreakdown()
        assert cb.cycles_per_tile(640) == 789

    def test_cycles_per_tile_128(self):
        cb = CycleBreakdown()
        assert cb.cycles_per_tile(128) == 277

    def test_custom_breakdown(self):
        cb = CycleBreakdown(bias_load=64, pipeline_fill=2, requant=4, write_tile=8, next_or_done=2)
        assert cb.overhead == 80
        assert cb.cycles_per_tile(100) == 180


# ---------------------------------------------------------------------------
# Parallelism tests
# ---------------------------------------------------------------------------


class TestParallelism:
    def test_128_output_128_max(self):
        assert Scheduler._compute_parallelism(128, 128) == 128

    def test_640_output_128_max(self):
        assert Scheduler._compute_parallelism(640, 128) == 128

    def test_64_output_128_max(self):
        assert Scheduler._compute_parallelism(64, 128) == 64

    def test_prime_output_dim(self):
        # 127 is prime — only divisible by 1 and 127
        assert Scheduler._compute_parallelism(127, 128) == 127

    def test_small_prime(self):
        # 17 is prime, max_par=128 → parallelism=17
        assert Scheduler._compute_parallelism(17, 128) == 17

    def test_output_larger_than_max(self):
        # 256 with max=128 → 128 (divides 256)
        assert Scheduler._compute_parallelism(256, 128) == 128

    def test_non_power_of_two(self):
        # 192 with max=128 → 128 doesn't divide 192, try 127...96 → 96 divides
        assert Scheduler._compute_parallelism(192, 128) == 96


# ---------------------------------------------------------------------------
# Cycle count tests (task 3.4.1)
# ---------------------------------------------------------------------------


class TestCycleCounts:
    def test_layer0_cycles(self, ad_schedule):
        sched, _ = ad_schedule
        ls = sched.layers[0]
        assert ls.input_dim == 640
        assert ls.output_dim == 128
        assert ls.num_tiles == 1
        assert ls.cycles_per_tile == 789
        assert ls.total_cycles == 789

    def test_layer1_cycles(self, ad_schedule):
        sched, _ = ad_schedule
        ls = sched.layers[1]
        assert ls.input_dim == 128
        assert ls.output_dim == 128
        assert ls.num_tiles == 1
        assert ls.cycles_per_tile == 277
        assert ls.total_cycles == 277

    def test_layer2_cycles(self, ad_schedule):
        sched, _ = ad_schedule
        ls = sched.layers[2]
        assert ls.input_dim == 128
        assert ls.output_dim == 128
        assert ls.num_tiles == 1
        assert ls.cycles_per_tile == 277
        assert ls.total_cycles == 277

    def test_layer3_cycles(self, ad_schedule):
        sched, _ = ad_schedule
        ls = sched.layers[3]
        assert ls.input_dim == 128
        assert ls.output_dim == 640
        assert ls.num_tiles == 5
        assert ls.cycles_per_tile == 277
        assert ls.total_cycles == 1385

    def test_total_compute_cycles(self, ad_schedule):
        sched, _ = ad_schedule
        assert sched.total_compute_cycles == 789 + 277 + 277 + 1385
        assert sched.total_compute_cycles == 2728

    def test_total_cycles_with_axi(self, ad_schedule):
        sched, _ = ad_schedule
        assert sched.total_cycles == 2728 + 160
        assert sched.total_cycles == 2888


# ---------------------------------------------------------------------------
# Memory layout tests (tasks 3.4.3, 3.4.4)
# ---------------------------------------------------------------------------


class TestMemoryLayout:
    def test_weight_rows_per_layer(self, ad_schedule):
        sched, _ = ad_schedule
        # Layer 0: 1 tile * 640 = 640 rows
        assert sched.layers[0].weight_rows == 640
        # Layer 1: 1 * 128 = 128
        assert sched.layers[1].weight_rows == 128
        # Layer 2: 1 * 128 = 128
        assert sched.layers[2].weight_rows == 128
        # Layer 3: 5 * 128 = 640
        assert sched.layers[3].weight_rows == 640

    def test_weight_start_rows_sequential(self, ad_schedule):
        sched, _ = ad_schedule
        assert sched.layers[0].weight_start_row == 0
        assert sched.layers[1].weight_start_row == 640
        assert sched.layers[2].weight_start_row == 768
        assert sched.layers[3].weight_start_row == 896

    def test_weight_rows_no_overlap(self, ad_schedule):
        sched, _ = ad_schedule
        for i in range(len(sched.layers) - 1):
            curr = sched.layers[i]
            nxt = sched.layers[i + 1]
            assert curr.weight_start_row + curr.weight_rows == nxt.weight_start_row

    def test_weight_rows_fit_sram(self, ad_schedule):
        sched, _ = ad_schedule
        total_rows = sum(ls.weight_rows for ls in sched.layers)
        assert total_rows == 1536
        assert total_rows <= HardwareConstraints().weight_bank_depth

    def test_bias_rows_per_layer(self, ad_schedule):
        sched, _ = ad_schedule
        assert sched.layers[0].bias_rows == 4  # ceil(128/32)
        assert sched.layers[1].bias_rows == 4
        assert sched.layers[2].bias_rows == 4
        assert sched.layers[3].bias_rows == 20  # ceil(640/32)

    def test_bias_start_rows_sequential(self, ad_schedule):
        sched, _ = ad_schedule
        assert sched.layers[0].bias_start_row == 0
        assert sched.layers[1].bias_start_row == 4
        assert sched.layers[2].bias_start_row == 8
        assert sched.layers[3].bias_start_row == 12

    def test_bias_rows_fit_sram(self, ad_schedule):
        sched, _ = ad_schedule
        total_rows = sum(ls.bias_rows for ls in sched.layers)
        assert total_rows == 32
        assert total_rows <= HardwareConstraints().bias_bank_depth


# ---------------------------------------------------------------------------
# Activation bank tests (task 3.4.2)
# ---------------------------------------------------------------------------


class TestActivationBanks:
    def test_ping_pong_pattern(self, ad_schedule):
        sched, _ = ad_schedule
        assert sched.layers[0].act_in_bank == "A"
        assert sched.layers[0].act_out_bank == "B"
        assert sched.layers[1].act_in_bank == "B"
        assert sched.layers[1].act_out_bank == "A"
        assert sched.layers[2].act_in_bank == "A"
        assert sched.layers[2].act_out_bank == "B"
        assert sched.layers[3].act_in_bank == "B"
        assert sched.layers[3].act_out_bank == "A"

    def test_no_same_bank_read_write(self, ad_schedule):
        sched, _ = ad_schedule
        for ls in sched.layers:
            assert ls.act_in_bank != ls.act_out_bank

    def test_consecutive_layers_chain(self, ad_schedule):
        """Output bank of layer N == input bank of layer N+1."""
        sched, _ = ad_schedule
        for i in range(len(sched.layers) - 1):
            assert sched.layers[i].act_out_bank == sched.layers[i + 1].act_in_bank


# ---------------------------------------------------------------------------
# Invariant tests (task 3.4.5)
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_validate_scheduled_passes(self, ad_schedule):
        _, graph = ad_schedule
        graph.validate("scheduled")  # Should not raise

    def test_inv_4_1_all_nodes_have_schedule(self, ad_schedule):
        _, graph = ad_schedule
        for node in graph.nodes.values():
            assert node.schedule_info is not None

    def test_inv_4_2_sequential_indices(self, ad_schedule):
        sched, _ = ad_schedule
        indices = [ls.layer_index for ls in sched.layers]
        assert indices == [0, 1, 2, 3]

    def test_inv_4_5_parallelism_divides_output(self, ad_schedule):
        sched, _ = ad_schedule
        for ls in sched.layers:
            assert ls.output_dim % ls.parallelism == 0

    def test_inv_4_6_all_quantized(self, ad_schedule):
        _, graph = ad_schedule
        for node in graph.nodes.values():
            assert node.fused_attrs.is_quantized


# ---------------------------------------------------------------------------
# Latency tests (task 3.4.6)
# ---------------------------------------------------------------------------


class TestLatency:
    def test_latency_under_30us(self, ad_schedule):
        sched, _ = ad_schedule
        assert sched.latency_us < 30.0

    def test_exact_latency(self, ad_schedule):
        sched, _ = ad_schedule
        assert sched.latency_us == pytest.approx(28.88)

    def test_throughput(self, ad_schedule):
        sched, _ = ad_schedule
        assert sched.throughput_inferences_per_sec > 20_000


# ---------------------------------------------------------------------------
# JSON export tests (task 3.3.x)
# ---------------------------------------------------------------------------


class TestJSONExport:
    def test_to_json_structure(self, ad_schedule):
        sched, _ = ad_schedule
        data = sched.to_json()
        assert "layers" in data
        assert "total_cycles" in data
        assert "clock_mhz" in data
        assert "latency_us" in data

    def test_to_json_layer_count(self, ad_schedule):
        sched, _ = ad_schedule
        data = sched.to_json()
        assert len(data["layers"]) == 4

    def test_schedule_to_json_writes_file(self, ad_schedule, tmp_path):
        sched, _ = ad_schedule
        path = tmp_path / "schedule.json"
        data = schedule_to_json(sched, path)

        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["total_cycles"] == data["total_cycles"]
        assert len(loaded["layers"]) == 4

    def test_schedule_to_json_no_file(self, ad_schedule):
        sched, _ = ad_schedule
        data = schedule_to_json(sched)
        assert isinstance(data, dict)
        assert data["total_cycles"] == 2888


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_rejects_unquantized_graph(self):
        graph = Graph("test", {}, {}, [], [], stage="fused")
        scheduler = Scheduler()
        with pytest.raises(ValueError, match="quantized"):
            scheduler.schedule(graph)

    def test_rejects_missing_fused_attrs(self):
        node = OpNode("n0", OpType.FUSED_LINEAR, ["x", "w", "b"], ["y"])
        graph = Graph(
            "test",
            {"n0": node},
            {
                "x": Tensor("x", TensorType((1, 4), np.dtype(np.float32))),
                "w": Tensor("w", TensorType((4, 4), np.dtype(np.float32))),
                "b": Tensor("b", TensorType((4,), np.dtype(np.float32))),
                "y": Tensor("y", TensorType((1, 4), np.dtype(np.float32))),
            },
            ["x"],
            ["y"],
            stage="quantized",
        )
        scheduler = Scheduler()
        with pytest.raises(ValueError, match="fused_attrs"):
            scheduler.schedule(graph)

    def test_weight_overflow_raises(self):
        """Scheduler raises RuntimeError if weights exceed SRAM budget."""
        qp = QuantParams(scale=0.1, zero_point=0, calibrated=True)
        attrs = FusedLinearAttrs(
            input_dim=2000,
            output_dim=128,
            has_relu=True,
            weight_quant=qp,
            input_quant=qp,
            output_quant=qp,
            requant_scale_fixed=65536,
            requant_shift=16,
        )
        node = OpNode("n0", OpType.FUSED_LINEAR_RELU, ["x", "w", "b"], ["y"])
        node.fused_attrs = attrs
        graph = Graph(
            "test",
            {"n0": node},
            {
                "x": Tensor("x", TensorType((1, 2000), np.dtype(np.int8))),
                "w": Tensor(
                    "w",
                    TensorType((2000, 128), np.dtype(np.int8)),
                    data=np.zeros((2000, 128), dtype=np.int8),
                ),
                "b": Tensor(
                    "b",
                    TensorType((128,), np.dtype(np.int32)),
                    data=np.zeros(128, dtype=np.int32),
                ),
                "y": Tensor("y", TensorType((1, 128), np.dtype(np.int8))),
            },
            ["x"],
            ["y"],
            stage="quantized",
        )
        # Constraints with very small SRAM budget
        constraints = HardwareConstraints(weight_bank_depth=100)
        scheduler = Scheduler(constraints=constraints)
        with pytest.raises(RuntimeError, match="Weight SRAM overflow"):
            scheduler.schedule(graph)

    def test_bias_overflow_raises(self):
        """Scheduler raises RuntimeError if biases exceed SRAM budget."""
        qp = QuantParams(scale=0.1, zero_point=0, calibrated=True)
        attrs = FusedLinearAttrs(
            input_dim=128,
            output_dim=640,
            has_relu=False,
            weight_quant=qp,
            input_quant=qp,
            output_quant=qp,
            requant_scale_fixed=65536,
            requant_shift=16,
        )
        node = OpNode("n0", OpType.FUSED_LINEAR, ["x", "w", "b"], ["y"])
        node.fused_attrs = attrs
        graph = Graph(
            "test",
            {"n0": node},
            {
                "x": Tensor("x", TensorType((1, 128), np.dtype(np.int8))),
                "w": Tensor(
                    "w",
                    TensorType((128, 640), np.dtype(np.int8)),
                    data=np.zeros((128, 640), dtype=np.int8),
                ),
                "b": Tensor(
                    "b",
                    TensorType((640,), np.dtype(np.int32)),
                    data=np.zeros(640, dtype=np.int32),
                ),
                "y": Tensor("y", TensorType((1, 640), np.dtype(np.int8))),
            },
            ["x"],
            ["y"],
            stage="quantized",
        )
        # bias_rows needed: ceil(640/32)=20, set max to 5
        constraints = HardwareConstraints(bias_bank_depth=5)
        scheduler = Scheduler(constraints=constraints)
        with pytest.raises(RuntimeError, match="Bias SRAM overflow"):
            scheduler.schedule(graph)

    def test_graph_stage_set_to_scheduled(self, quantized_ad_graph):
        scheduler = Scheduler()
        scheduler.schedule(quantized_ad_graph)
        assert quantized_ad_graph.stage == "scheduled"

    def test_has_relu_flag(self, ad_schedule):
        sched, _ = ad_schedule
        # Layers 0-2 have ReLU, layer 3 does not
        assert sched.layers[0].has_relu is True
        assert sched.layers[1].has_relu is True
        assert sched.layers[2].has_relu is True
        assert sched.layers[3].has_relu is False


# ---------------------------------------------------------------------------
# Scheduler edge case tests
# ---------------------------------------------------------------------------


class TestSchedulerEdgeCases:
    def _make_single_layer_graph(
        self, input_dim: int, output_dim: int, has_relu: bool = True
    ) -> Graph:
        """Build a single-node quantized graph for scheduler testing."""
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
        node = OpNode("n0", op, ["x", "w", "b"], ["y"])
        node.fused_attrs = attrs
        return Graph(
            "test",
            {"n0": node},
            {
                "x": Tensor("x", TensorType((1, input_dim), np.dtype(np.int8))),
                "w": Tensor(
                    "w",
                    TensorType((input_dim, output_dim), np.dtype(np.int8)),
                    data=np.zeros((input_dim, output_dim), dtype=np.int8),
                ),
                "b": Tensor(
                    "b",
                    TensorType((output_dim,), np.dtype(np.int32)),
                    data=np.zeros(output_dim, dtype=np.int32),
                ),
                "y": Tensor("y", TensorType((1, output_dim), np.dtype(np.int8))),
            },
            ["x"],
            ["y"],
            stage="quantized",
        )

    def test_prime_output_dim_small(self):
        """output_dim=127 (prime, <=128) → parallelism=127, 1 tile."""
        graph = self._make_single_layer_graph(64, 127)
        sched = Scheduler().schedule(graph)
        assert sched.layers[0].parallelism == 127
        assert sched.layers[0].num_tiles == 1

    def test_prime_output_dim_large(self):
        """output_dim=131 (prime, >128) → parallelism=1, 131 tiles."""
        graph = self._make_single_layer_graph(64, 131)
        # 131 tiles * 64 input_dim = 8384 weight rows — needs big SRAM budget
        constraints = HardwareConstraints(weight_bank_depth=10000, bias_bank_depth=100)
        sched = Scheduler(constraints=constraints).schedule(graph)
        assert sched.layers[0].parallelism == 1
        assert sched.layers[0].num_tiles == 131

    def test_output_dim_1(self):
        """output_dim=1 → parallelism=1, 1 tile."""
        graph = self._make_single_layer_graph(64, 1)
        sched = Scheduler().schedule(graph)
        assert sched.layers[0].parallelism == 1
        assert sched.layers[0].num_tiles == 1

    def test_single_layer_graph(self):
        """Single fused node schedules correctly."""
        graph = self._make_single_layer_graph(128, 128)
        sched = Scheduler().schedule(graph)
        assert len(sched.layers) == 1
        assert sched.layers[0].layer_index == 0
        assert sched.layers[0].act_in_bank == "A"
        assert sched.layers[0].act_out_bank == "B"

    def test_empty_graph_rejected(self):
        """0 nodes → ValueError."""
        graph = Graph("test", {}, {}, [], [], stage="quantized")
        with pytest.raises(ValueError, match="empty graph"):
            Scheduler().schedule(graph)
