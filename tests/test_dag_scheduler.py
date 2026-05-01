"""Tests for DAG Scheduler (Phase 3 scheduling expansion)."""

from __future__ import annotations

import numpy as np
import pytest

from mlasic.dag_scheduler import (
    ActivationLifetime,
    ActivationLifetimeAnalyzer,
    ConvCycleModel,
    DAGScheduler,
    ElementwiseCycleModel,
    LayerNormCycleModel,
    LinearCycleModel,
    MatMulCycleModel,
    ReshapeCycleModel,
    SoftmaxCycleModel,
    SRAMBudgetPlanner,
)
from mlasic.ingestion import ONNXParser
from mlasic.ir import (
    DAGLayerSchedule,
    FusedConvAttrs,
    FusedLinearAttrs,
    Graph,
    HardwareConstraints,
    OpNode,
    OpType,
    QuantParams,
    Tensor,
    TensorType,
)
from mlasic.optimization import (
    ActivationFusionPass,
    ConstantFoldingPass,
    ConvBatchNormFoldingPass,
    ConvFusionPass,
    ConvQuantizationPass,
    DeadCodeEliminationPass,
    LayerNormFusionPass,
    PassManager,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_quantized_linear_node(
    name: str,
    input_name: str,
    output_name: str,
    input_dim: int,
    output_dim: int,
    has_relu: bool = True,
) -> tuple[OpNode, dict[str, Tensor]]:
    """Create a quantized FusedLinear node with tensors for testing."""
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


def _make_quantized_conv_node(
    name: str,
    input_name: str,
    output_name: str,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    spatial: int = 8,
    has_relu: bool = True,
) -> tuple[OpNode, dict[str, Tensor]]:
    """Create a quantized FusedConv node with tensors for testing."""
    qp_list = [QuantParams(scale=0.1, zero_point=0, calibrated=True)] * out_channels
    qp = QuantParams(scale=0.1, zero_point=0, calibrated=True)
    attrs = FusedConvAttrs(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_shape=[kernel_size, kernel_size],
        pads=[kernel_size // 2] * 4,
        has_relu=has_relu,
        weight_quant=qp_list,
        input_quant=qp,
        output_quant=qp,
        requant_scale_fixed=[65536] * out_channels,
        requant_shift=16,
    )
    op = OpType.FUSED_CONV_RELU if has_relu else OpType.FUSED_CONV
    w_name = f"{name}_w"
    b_name = f"{name}_b"
    node = OpNode(name, op, [input_name, w_name, b_name], [output_name])
    node.fused_attrs = attrs
    tensors = {
        input_name: Tensor(
            input_name,
            TensorType((1, in_channels, spatial, spatial), np.dtype(np.int8)),
        ),
        w_name: Tensor(
            w_name,
            TensorType((out_channels, in_channels, kernel_size, kernel_size), np.dtype(np.int8)),
            data=np.zeros((out_channels, in_channels, kernel_size, kernel_size), dtype=np.int8),
        ),
        b_name: Tensor(
            b_name,
            TensorType((out_channels,), np.dtype(np.int32)),
            data=np.zeros(out_channels, dtype=np.int32),
        ),
        output_name: Tensor(
            output_name,
            TensorType((1, out_channels, spatial, spatial), np.dtype(np.int8)),
        ),
    }
    return node, tensors


def _make_quantized_dag_linear_chain() -> Graph:
    """Build a 2-layer quantized linear chain as DAG for testing."""
    n0, t0 = _make_quantized_linear_node("n0", "x", "h0", 128, 64)
    n1, t1 = _make_quantized_linear_node("n1", "h0", "y", 64, 32, has_relu=False)
    all_tensors = {**t0, **t1}
    return Graph(
        "test_chain",
        {"n0": n0, "n1": n1},
        all_tensors,
        ["x"],
        ["y"],
        stage="quantized_dag",
    )


# ---------------------------------------------------------------------------
# TestDAGCycleModels
# ---------------------------------------------------------------------------


class TestDAGCycleModels:
    def test_linear_cycle_model(self):
        """FusedLinearReLU: num_tiles * (input_dim + 149)."""
        n0, t0 = _make_quantized_linear_node("n0", "x", "y", 640, 128)
        graph = Graph("test", {"n0": n0}, t0, ["x"], ["y"], stage="quantized_dag")
        constraints = HardwareConstraints()
        model = LinearCycleModel()
        cycles = model.estimate_cycles("n0", graph, constraints)
        # 128 / 128 = 1 tile, 640 + 149 = 789
        assert cycles == 789

    def test_linear_tiled(self):
        """FusedLinear with output_dim > parallelism."""
        n0, t0 = _make_quantized_linear_node("n0", "x", "y", 128, 640, has_relu=False)
        graph = Graph("test", {"n0": n0}, t0, ["x"], ["y"], stage="quantized_dag")
        constraints = HardwareConstraints()
        model = LinearCycleModel()
        cycles = model.estimate_cycles("n0", graph, constraints)
        # 640 / 128 = 5 tiles, 128 + 149 = 277 per tile
        assert cycles == 5 * 277

    def test_conv_cycle_model(self):
        """FusedConvReLU cycle estimation."""
        n0, t0 = _make_quantized_conv_node("n0", "x", "y", 16, 16, kernel_size=3, spatial=8)
        graph = Graph("test", {"n0": n0}, t0, ["x"], ["y"], stage="quantized_dag")
        constraints = HardwareConstraints()
        model = ConvCycleModel()
        cycles = model.estimate_cycles("n0", graph, constraints)
        # OC=16, par=16, tiles=1, OH=OW=8, KH=KW=3, IC=16
        # cycles_per_pixel = 3*3*16 + 10 = 154
        # total = 1 * 8 * 8 * 154 = 9856
        assert cycles == 9856

    def test_softmax_cycle_model(self):
        """Softmax: 3 * seq_len."""
        node = OpNode("soft", OpType.SOFTMAX, ["x"], ["y"])
        tensors = {
            "x": Tensor("x", TensorType((1, 4, 4), np.dtype(np.float32))),
            "y": Tensor("y", TensorType((1, 4, 4), np.dtype(np.float32))),
        }
        graph = Graph("test", {"soft": node}, tensors, ["x"], ["y"], stage="quantized_dag")
        model = SoftmaxCycleModel()
        # Last dim = 4, cycles = 3 * 4 = 12
        assert model.estimate_cycles("soft", graph, HardwareConstraints()) == 12

    def test_reshape_zero_cycles(self):
        """Reshape/Flatten: 0 cycles."""
        node = OpNode("flat", OpType.FLATTEN, ["x"], ["y"])
        tensors = {
            "x": Tensor("x", TensorType((1, 4, 2, 2), np.dtype(np.float32))),
            "y": Tensor("y", TensorType((1, 16), np.dtype(np.float32))),
        }
        graph = Graph("test", {"flat": node}, tensors, ["x"], ["y"], stage="quantized_dag")
        model = ReshapeCycleModel()
        assert model.estimate_cycles("flat", graph, HardwareConstraints()) == 0

    def test_layernorm_cycle_model(self):
        """LayerNorm: 2 * feature_dim + overhead."""
        node = OpNode("ln", OpType.LAYER_NORM, ["x"], ["y"])
        tensors = {
            "x": Tensor("x", TensorType((1, 4, 16), np.dtype(np.float32))),
            "y": Tensor("y", TensorType((1, 4, 16), np.dtype(np.float32))),
        }
        graph = Graph("test", {"ln": node}, tensors, ["x"], ["y"], stage="quantized_dag")
        model = LayerNormCycleModel()
        # feature_dim=16, cycles = 2*16 + 10 = 42
        assert model.estimate_cycles("ln", graph, HardwareConstraints()) == 42

    def test_elementwise_cycle_model(self):
        """Add: ceil(numel / parallelism)."""
        node = OpNode("add", OpType.ADD, ["x", "y"], ["z"])
        tensors = {
            "x": Tensor("x", TensorType((1, 256), np.dtype(np.float32))),
            "y": Tensor("y", TensorType((1, 256), np.dtype(np.float32))),
            "z": Tensor("z", TensorType((1, 256), np.dtype(np.float32))),
        }
        graph = Graph("test", {"add": node}, tensors, ["x", "y"], ["z"], stage="quantized_dag")
        model = ElementwiseCycleModel()
        # numel=256, par=128, cycles = 2
        assert model.estimate_cycles("add", graph, HardwareConstraints()) == 2

    def test_matmul_cycle_model(self):
        """Unfused MatMul cycle estimation."""
        node = OpNode("mm", OpType.MATMUL, ["x", "w"], ["y"])
        tensors = {
            "x": Tensor("x", TensorType((1, 4, 16), np.dtype(np.float32))),
            "w": Tensor(
                "w",
                TensorType((16, 16), np.dtype(np.float32)),
                data=np.zeros((16, 16), dtype=np.float32),
            ),
            "y": Tensor("y", TensorType((1, 4, 16), np.dtype(np.float32))),
        }
        graph = Graph("test", {"mm": node}, tensors, ["x"], ["y"], stage="quantized_dag")
        model = MatMulCycleModel()
        cycles = model.estimate_cycles("mm", graph, HardwareConstraints())
        # K=16, N=16, par=16, tiles=1, batch=1
        # 1 * 1 * (16 + 149) = 165
        assert cycles == 165


# ---------------------------------------------------------------------------
# TestActivationLifetime
# ---------------------------------------------------------------------------


class TestActivationLifetime:
    def test_linear_chain_lifetimes(self):
        """In a linear chain, each activation lives until the next node finishes."""
        graph = _make_quantized_dag_linear_chain()
        order = graph.topological_order()
        # Manually set end cycles
        end_cycles = {"n0": 100, "n1": 200}

        analyzer = ActivationLifetimeAnalyzer()
        lifetimes = analyzer.analyze(graph, order, end_cycles)

        # Non-constant tensors produced by nodes: h0 (by n0), y (by n1)
        lt_map = {lt.tensor_name: lt for lt in lifetimes}
        assert "h0" in lt_map
        assert lt_map["h0"].produce_cycle == 100
        assert lt_map["h0"].free_cycle == 200  # consumed by n1
        assert lt_map["h0"].last_consumer_node == "n1"

    def test_skip_connection_extends_lifetime(self):
        """Skip connection keeps activation alive longer."""
        # Build: conv1 -> conv2 -> add(input, conv2_out)
        n_conv1, t_conv1 = _make_quantized_conv_node("conv1", "x", "conv1_out", 16, 16, spatial=8)
        # conv2 takes conv1's output
        n_conv2, t_conv2 = _make_quantized_conv_node(
            "conv2", "conv1_out", "conv2_out", 16, 16, spatial=8
        )
        # Add: skip connection from x + conv2_out
        n_add = OpNode("add", OpType.ADD, ["x", "conv2_out"], ["add_out"])
        add_out_t = Tensor("add_out", TensorType((1, 16, 8, 8), np.dtype(np.int8)))

        all_tensors = {**t_conv1, **t_conv2, "add_out": add_out_t}
        graph = Graph(
            "test_skip",
            {"conv1": n_conv1, "conv2": n_conv2, "add": n_add},
            all_tensors,
            ["x"],
            ["add_out"],
            stage="quantized_dag",
        )

        order = graph.topological_order()
        end_cycles = {"conv1": 100, "conv2": 200, "add": 210}

        analyzer = ActivationLifetimeAnalyzer()
        lifetimes = analyzer.analyze(graph, order, end_cycles)
        lt_map = {lt.tensor_name: lt for lt in lifetimes}

        # conv1_out consumed only by conv2, freed at 200
        assert lt_map["conv1_out"].free_cycle == 200

        # conv2_out consumed by add, freed at 210
        assert lt_map["conv2_out"].free_cycle == 210

    def test_multi_consumer_tensor(self):
        """Tensor consumed by multiple nodes freed when last one finishes."""
        # x -> n0 -> h0, x -> n1 -> h1
        n0, t0 = _make_quantized_linear_node("n0", "x", "h0", 64, 32)
        n1, t1 = _make_quantized_linear_node("n1", "x", "h1", 64, 32)
        # Merge tensors — x is shared
        all_tensors = {**t0, **t1}
        graph = Graph(
            "test_multi",
            {"n0": n0, "n1": n1},
            all_tensors,
            ["x"],
            ["h0", "h1"],
            stage="quantized_dag",
        )

        order = graph.topological_order()
        end_cycles = {"n0": 100, "n1": 200}

        analyzer = ActivationLifetimeAnalyzer()
        lifetimes = analyzer.analyze(graph, order, end_cycles)

        # h0 and h1 are outputs with no further consumers
        lt_map = {lt.tensor_name: lt for lt in lifetimes}
        assert "h0" in lt_map
        assert "h1" in lt_map

    def test_graph_inputs_excluded(self):
        """Graph inputs (no producer) are not in lifetimes."""
        graph = _make_quantized_dag_linear_chain()
        order = graph.topological_order()
        end_cycles = {"n0": 100, "n1": 200}

        analyzer = ActivationLifetimeAnalyzer()
        lifetimes = analyzer.analyze(graph, order, end_cycles)

        tensor_names = {lt.tensor_name for lt in lifetimes}
        # 'x' is a graph input — no producer node, should not appear
        assert "x" not in tensor_names


# ---------------------------------------------------------------------------
# TestSRAMBudget
# ---------------------------------------------------------------------------


class TestSRAMBudget:
    def test_linear_chain_peak(self):
        """Linear chain: peak = max single activation size."""
        lifetimes = [
            ActivationLifetime("h0", "n0", "n1", 100, 200, 256),
            ActivationLifetime("y", "n1", "n1", 200, 200, 128),
        ]
        planner = SRAMBudgetPlanner(budget_bytes=1024)
        fits, peak, snapshots = planner.check_budget(lifetimes, {})
        assert fits is True
        assert peak == 256

    def test_residual_peak(self):
        """Residual block: skip connection keeps 2 activations alive."""
        lifetimes = [
            ActivationLifetime("conv1_out", "conv1", "conv2", 100, 200, 1024),
            ActivationLifetime("conv2_out", "conv2", "add", 200, 210, 1024),
        ]
        planner = SRAMBudgetPlanner(budget_bytes=4096)
        fits, peak, _ = planner.check_budget(lifetimes, {})
        assert fits is True
        # At cycle 200: conv1_out freed (-1024) and conv2_out produced (+1024)
        # Since frees happen before produces at same cycle, peak = 1024
        assert peak == 1024

    def test_overflow_detected(self):
        """Budget exceeded returns fits=False."""
        lifetimes = [
            ActivationLifetime("a", "n0", "n2", 0, 300, 2048),
            ActivationLifetime("b", "n1", "n2", 100, 300, 2048),
        ]
        planner = SRAMBudgetPlanner(budget_bytes=3000)
        fits, peak, _ = planner.check_budget(lifetimes, {})
        assert fits is False
        assert peak == 4096  # both live simultaneously


# ---------------------------------------------------------------------------
# TestDAGScheduler
# ---------------------------------------------------------------------------


class TestDAGScheduler:
    def test_linear_chain_compat(self):
        """DAG scheduler handles a simple linear chain."""
        graph = _make_quantized_dag_linear_chain()
        scheduler = DAGScheduler()
        sched = scheduler.schedule(graph)

        assert len(sched.node_schedules) == 2
        assert sched.node_schedules[0].start_cycle == 0
        assert sched.node_schedules[1].start_cycle == sched.node_schedules[0].end_cycle
        assert sched.total_cycles == sched.node_schedules[1].end_cycle
        assert graph.stage == "scheduled_dag"

    def test_residual_block_scheduling(self):
        """Schedule a graph with a skip connection."""
        # conv1 -> conv2 -> add(x, conv2_out) -> relu
        n_conv1, t_conv1 = _make_quantized_conv_node("conv1", "x", "conv1_out", 16, 16, spatial=8)
        n_conv2, t_conv2 = _make_quantized_conv_node(
            "conv2", "conv1_out", "conv2_out", 16, 16, spatial=8
        )
        n_add = OpNode("add", OpType.ADD, ["x", "conv2_out"], ["add_out"])
        add_out_t = Tensor("add_out", TensorType((1, 16, 8, 8), np.dtype(np.int8)))
        n_relu = OpNode("relu", OpType.RELU, ["add_out"], ["y"])
        y_t = Tensor("y", TensorType((1, 16, 8, 8), np.dtype(np.int8)))

        all_tensors = {**t_conv1, **t_conv2, "add_out": add_out_t, "y": y_t}
        graph = Graph(
            "resblock",
            {"conv1": n_conv1, "conv2": n_conv2, "add": n_add, "relu": n_relu},
            all_tensors,
            ["x"],
            ["y"],
            stage="quantized_dag",
        )

        scheduler = DAGScheduler()
        sched = scheduler.schedule(graph)

        assert len(sched.node_schedules) == 4
        # Verify sequentiality
        for i in range(len(sched.node_schedules) - 1):
            assert sched.node_schedules[i].end_cycle == sched.node_schedules[i + 1].start_cycle

    def test_dependency_order(self):
        """Verify nodes execute in dependency order."""
        graph = _make_quantized_dag_linear_chain()
        scheduler = DAGScheduler()
        sched = scheduler.schedule(graph)

        # n0 must come before n1
        names = [ns.node_name for ns in sched.node_schedules]
        assert names.index("n0") < names.index("n1")

    def test_reject_unquantized_fused(self):
        """Rejects fused nodes without quantization."""
        attrs = FusedLinearAttrs(input_dim=64, output_dim=32, has_relu=True)
        node = OpNode("n0", OpType.FUSED_LINEAR_RELU, ["x", "w", "b"], ["y"])
        node.fused_attrs = attrs
        tensors = {
            "x": Tensor("x", TensorType((1, 64), np.dtype(np.float32))),
            "w": Tensor("w", TensorType((64, 32), np.dtype(np.float32))),
            "b": Tensor("b", TensorType((32,), np.dtype(np.float32))),
            "y": Tensor("y", TensorType((1, 32), np.dtype(np.float32))),
        }
        graph = Graph("test", {"n0": node}, tensors, ["x"], ["y"], stage="quantized_dag")
        with pytest.raises(ValueError, match="not fully quantized"):
            DAGScheduler().schedule(graph)

    def test_reject_unknown_op(self):
        """Rejects fused ops missing fused_attrs."""
        node = OpNode("n0", OpType.FUSED_ATTENTION, ["x"], ["y"])
        tensors = {
            "x": Tensor("x", TensorType((3,), np.dtype(np.int64))),
            "y": Tensor("y", TensorType((2, 3, 4), np.dtype(np.float32))),
        }
        graph = Graph("test", {"n0": node}, tensors, ["x"], ["y"], stage="quantized_dag")
        with pytest.raises(ValueError, match="missing fused_attrs"):
            DAGScheduler().schedule(graph)

    def test_reject_wrong_stage(self):
        """Rejects graphs not in quantized_dag or quantized stage."""
        graph = Graph("test", {}, {}, [], [], stage="fused_dag")
        with pytest.raises(ValueError, match="quantized_dag"):
            DAGScheduler().schedule(graph)

    def test_weight_bytes_tracked(self):
        """Weight and bias bytes are tracked correctly."""
        graph = _make_quantized_dag_linear_chain()
        scheduler = DAGScheduler()
        sched = scheduler.schedule(graph)

        # n0: weight = 128*64 = 8192 bytes, bias = 64*4 = 256 bytes
        # n1: weight = 64*32 = 2048 bytes, bias = 32*4 = 128 bytes
        assert sched.total_weight_bytes == 8192 + 2048
        assert sched.total_bias_bytes == 256 + 128


# ---------------------------------------------------------------------------
# TestDAGSchedulerEdgeCases
# ---------------------------------------------------------------------------


class TestDAGSchedulerEdgeCases:
    def test_single_node(self):
        """Single node graph schedules correctly."""
        n0, t0 = _make_quantized_linear_node("n0", "x", "y", 64, 32)
        graph = Graph("test", {"n0": n0}, t0, ["x"], ["y"], stage="quantized_dag")
        sched = DAGScheduler().schedule(graph)
        assert len(sched.node_schedules) == 1
        assert sched.node_schedules[0].start_cycle == 0
        assert sched.total_cycles > 0

    def test_parallel_branches(self):
        """Two independent branches: both consume graph input."""
        n0, t0 = _make_quantized_linear_node("n0", "x", "h0", 64, 32)
        n1, t1 = _make_quantized_linear_node("n1", "x", "h1", 64, 32)
        all_tensors = {**t0, **t1}
        graph = Graph(
            "test_par",
            {"n0": n0, "n1": n1},
            all_tensors,
            ["x"],
            ["h0", "h1"],
            stage="quantized_dag",
        )
        sched = DAGScheduler().schedule(graph)

        assert len(sched.node_schedules) == 2
        # Both should execute sequentially (layer-sequential)
        assert sched.node_schedules[0].end_cycle == sched.node_schedules[1].start_cycle

    def test_empty_graph_rejected(self):
        """Empty graph raises ValueError."""
        graph = Graph("test", {}, {}, [], [], stage="quantized_dag")
        with pytest.raises(ValueError, match="empty graph"):
            DAGScheduler().schedule(graph)

    def test_dag_schedule_json_export(self):
        """DAGSchedule.to_json() produces valid structure."""
        graph = _make_quantized_dag_linear_chain()
        sched = DAGScheduler().schedule(graph)
        data = sched.to_json()
        assert "nodes" in data
        assert "lifetimes" in data
        assert "total_cycles" in data
        assert "peak_activation_bytes" in data
        assert len(data["nodes"]) == 2

    def test_latency_properties(self):
        """Latency and throughput properties work correctly."""
        graph = _make_quantized_dag_linear_chain()
        sched = DAGScheduler().schedule(graph)
        assert sched.latency_us > 0
        assert sched.latency_ms == sched.latency_us / 1000.0
        assert sched.throughput_inferences_per_sec > 0

    def test_dag_schedule_attached_to_nodes(self):
        """Each node has dag_schedule attached after scheduling."""
        graph = _make_quantized_dag_linear_chain()
        DAGScheduler().schedule(graph)
        for node in graph.nodes.values():
            assert node.dag_schedule is not None
            assert isinstance(node.dag_schedule, DAGLayerSchedule)


# ---------------------------------------------------------------------------
# TestDAGIntegration — full pipeline tests
# ---------------------------------------------------------------------------


class TestDAGIntegration:
    def test_cnn_pipeline(self, cnn_model_path):
        """CNN model through full pipeline including DAG scheduling."""
        parser = ONNXParser(cnn_model_path)
        graph = parser.parse()
        rng = np.random.RandomState(123)
        calib = [rng.randn(1, 1, 8, 8).astype(np.float32) for _ in range(10)]

        pm = PassManager()
        pm.add_pass(ConstantFoldingPass())
        pm.add_pass(ConvBatchNormFoldingPass())
        pm.add_pass(DeadCodeEliminationPass())
        pm.add_pass(ConvFusionPass())
        pm.add_pass(ActivationFusionPass())
        pm.add_pass(LayerNormFusionPass())
        pm.add_pass(ConvQuantizationPass(calib))
        graph = pm.run(graph, verify=False)

        # CNN passes don't auto-set stage; set it for DAG scheduling
        graph.stage = "quantized_dag"

        scheduler = DAGScheduler()
        sched = scheduler.schedule(graph)

        assert sched.total_cycles > 0
        assert len(sched.node_schedules) > 0
        assert graph.stage == "scheduled_dag"

    def test_transformer_pipeline(self, transformer_model_path):
        """Transformer model through full pipeline including DAG scheduling."""
        parser = ONNXParser(transformer_model_path)
        graph = parser.parse()

        pm = PassManager()
        pm.add_pass(ConstantFoldingPass())
        pm.add_pass(DeadCodeEliminationPass())
        pm.add_pass(LayerNormFusionPass())
        pm.add_pass(ActivationFusionPass())
        graph = pm.run(graph, verify=False)

        # Transformer may be in fused_dag — we need to get it to quantized_dag
        # For transformer without conv, just set stage directly for scheduling test
        graph.stage = "quantized_dag"

        scheduler = DAGScheduler()
        sched = scheduler.schedule(graph)

        assert sched.total_cycles > 0
        assert len(sched.node_schedules) > 0

    def test_resnet_block_pipeline(self, resnet_block_model_path):
        """ResNet block with skip connection through full pipeline."""
        parser = ONNXParser(resnet_block_model_path)
        graph = parser.parse()
        rng = np.random.RandomState(123)
        calib = [rng.randn(1, 16, 8, 8).astype(np.float32) for _ in range(10)]

        pm = PassManager()
        pm.add_pass(ConstantFoldingPass())
        pm.add_pass(ConvBatchNormFoldingPass())
        pm.add_pass(DeadCodeEliminationPass())
        pm.add_pass(ConvFusionPass())
        pm.add_pass(ActivationFusionPass())
        pm.add_pass(ConvQuantizationPass(calib))
        graph = pm.run(graph, verify=False)

        # CNN passes don't auto-set stage; set it for DAG scheduling
        graph.stage = "quantized_dag"

        scheduler = DAGScheduler()
        sched = scheduler.schedule(graph)

        assert sched.total_cycles > 0
        assert sched.peak_activation_bytes > 0
        assert graph.stage == "scheduled_dag"

        # Verify skip connection creates non-trivial lifetimes
        assert len(sched.lifetimes) > 0

    def test_ir_validation_scheduled_dag(self):
        """Validate scheduled_dag stage passes IR validation."""
        graph = _make_quantized_dag_linear_chain()
        DAGScheduler().schedule(graph)
        # Should not raise
        graph.validate("scheduled_dag")
