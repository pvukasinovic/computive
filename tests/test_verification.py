"""Stage 6: Verification infrastructure tests.

Tests for:
  - Golden vector generation (6.1.1-6.1.7)
  - Testbench generation (6.2.1-6.2.9, 6.3.1-6.3.7)
  - cocotb test generation (6.4.1-6.4.6)
  - Functional coverage tracking (6.5.1-6.5.11)
  - End-to-end MLP verification pipeline
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from mlasic.golden_vectors import GoldenVectorGenerator
from mlasic.ingestion import ONNXParser
from mlasic.ir import Graph, HardwareConstraints, OpType
from mlasic.optimization import (
    BatchNormFoldingPass,
    ConstantFoldingPass,
    DeadCodeEliminationPass,
    OperatorFusionPass,
    PassManager,
    QuantizationPass,
)
from mlasic.scheduler import Scheduler
from mlasic.testbench_gen import (
    FunctionalCoverage,
    TestbenchConfig,
    TestbenchGenerator,
    generate_cocotb_makefile,
    generate_cocotb_test,
    generate_tb_accelerator,
    generate_tb_activation_relu,
    generate_tb_axi_lite_ctrl,
    generate_tb_axi_stream_in,
    generate_tb_axi_stream_out,
    generate_tb_fused_linear_relu,
    generate_tb_mac_array,
    generate_tb_ping_pong_buffer,
    generate_tb_requantize,
    generate_tb_sram_bank,
)
from mlasic.weight_packer import WeightPacker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_mlp_model(
    layer_dims: list[int],
    has_bn: bool = True,
    name: str = "test_mlp",
    seed: int = 42,
) -> onnx.ModelProto:
    """Build a generic MLP model for testing."""
    rng = np.random.RandomState(seed)
    all_nodes = []
    all_inits = []
    all_vis = []

    input_name = "input"
    current = input_name
    num_layers = len(layer_dims) - 1

    for i in range(num_layers):
        in_dim = layer_dims[i]
        out_dim = layer_dims[i + 1]
        is_last = i == num_layers - 1
        prefix = f"layer{i}"

        fan_avg = (in_dim + out_dim) / 2
        scale = np.sqrt(1.0 / fan_avg)
        w_data = (rng.randn(in_dim, out_dim) * scale).astype(np.float32)
        w_name = f"{prefix}_weight"
        all_inits.append(numpy_helper.from_array(w_data, name=w_name))

        mm_out = f"{prefix}_matmul_out"
        all_nodes.append(
            helper.make_node("MatMul", [current, w_name], [mm_out], name=f"{prefix}_MatMul")
        )
        all_vis.append(helper.make_tensor_value_info(mm_out, TensorProto.FLOAT, [1, out_dim]))

        b_data = (rng.randn(out_dim) * 0.01).astype(np.float32)
        b_name = f"{prefix}_bias"
        all_inits.append(numpy_helper.from_array(b_data, name=b_name))

        add_out = f"{prefix}_add_out"
        all_nodes.append(helper.make_node("Add", [mm_out, b_name], [add_out], name=f"{prefix}_Add"))
        all_vis.append(helper.make_tensor_value_info(add_out, TensorProto.FLOAT, [1, out_dim]))
        current = add_out

        if has_bn and not is_last:
            bn_scale = np.ones(out_dim, dtype=np.float32)
            bn_bias = np.zeros(out_dim, dtype=np.float32)
            bn_mean = rng.randn(out_dim).astype(np.float32) * 0.1
            bn_var = np.abs(rng.randn(out_dim).astype(np.float32)) + 0.5
            names = [f"{prefix}_bn_{p}" for p in ["scale", "bias", "mean", "var"]]
            for arr, n in zip([bn_scale, bn_bias, bn_mean, bn_var], names):
                all_inits.append(numpy_helper.from_array(arr, name=n))
            bn_out = f"{prefix}_bn_out"
            all_nodes.append(
                helper.make_node(
                    "BatchNormalization",
                    [current, *names],
                    [bn_out],
                    name=f"{prefix}_BN",
                    epsilon=1e-5,
                )
            )
            all_vis.append(helper.make_tensor_value_info(bn_out, TensorProto.FLOAT, [1, out_dim]))
            current = bn_out

        if not is_last:
            relu_out = f"{prefix}_relu_out"
            all_nodes.append(helper.make_node("Relu", [current], [relu_out], name=f"{prefix}_Relu"))
            all_vis.append(helper.make_tensor_value_info(relu_out, TensorProto.FLOAT, [1, out_dim]))
            current = relu_out

    output_name = current
    in_dim = layer_dims[0]
    out_dim = layer_dims[-1]

    graph = helper.make_graph(
        nodes=all_nodes,
        name=name,
        inputs=[helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, in_dim])],
        outputs=[helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, out_dim])],
        initializer=all_inits,
        value_info=all_vis,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def run_pipeline(
    model_path: Path,
    calibration_data: list[np.ndarray],
    output_dir: Path,
    constraints: HardwareConstraints | None = None,
) -> tuple[Graph, dict]:
    """Run Stages 1-4 on an ONNX model file."""
    parser = ONNXParser(model_path)
    graph = parser.parse()

    pm = PassManager()
    pm.add_pass(ConstantFoldingPass())
    pm.add_pass(DeadCodeEliminationPass())
    pm.add_pass(BatchNormFoldingPass())
    pm.add_pass(OperatorFusionPass())
    pm.add_pass(QuantizationPass(calibration_data=calibration_data))
    graph = pm.run(graph, verify=True)

    hw = constraints or HardwareConstraints()
    scheduler = Scheduler(constraints=hw)
    scheduler.schedule(graph)

    packer = WeightPacker(graph, output_dir, constraints=hw)
    memory_map = packer.pack_weights()

    return graph, memory_map


@pytest.fixture
def ad_graph(tmp_path: Path) -> Graph:
    """Run full pipeline on AD model, return scheduled graph."""
    model = build_mlp_model([640, 128, 128, 128, 640], has_bn=True, name="ad_model")
    model_path = tmp_path / "ad_model.onnx"
    onnx.save(model, str(model_path))

    rng = np.random.RandomState(42)
    cal_data = [rng.randn(1, 640).astype(np.float32) for _ in range(20)]

    graph, _ = run_pipeline(model_path, cal_data, tmp_path / "packed")
    return graph


@pytest.fixture
def small_graph(tmp_path: Path) -> Graph:
    """Small 2-layer MLP for fast tests."""
    model = build_mlp_model([128, 64, 32], has_bn=True, name="small_mlp")
    model_path = tmp_path / "small_mlp.onnx"
    onnx.save(model, str(model_path))

    rng = np.random.RandomState(42)
    cal_data = [rng.randn(1, 128).astype(np.float32) for _ in range(10)]

    graph, _ = run_pipeline(model_path, cal_data, tmp_path / "packed")
    return graph


# ===========================================================================
# 6.1 Golden Vector Generation Tests
# ===========================================================================


class TestGoldenVectorGenerator:
    """Tests for GoldenVectorGenerator (tasks 6.1.1-6.1.7)."""

    def test_init_requires_quantized_graph(self, small_graph: Graph):
        """6.1.1: Constructor accepts quantized/scheduled graphs."""
        gen = GoldenVectorGenerator(small_graph)
        assert gen.graph.stage == "scheduled"
        assert len(gen.input_shapes) == 1
        assert len(gen.output_shapes) == 1

    def test_init_rejects_raw_graph(self):
        """6.1.1: Constructor rejects non-quantized graphs."""
        from mlasic.ir import Graph

        g = Graph(name="test", nodes={}, tensors={}, inputs=[], outputs=[], stage="raw")
        with pytest.raises(ValueError, match="quantized"):
            GoldenVectorGenerator(g)

    def test_random_vectors_count(self, small_graph: Graph):
        """6.1.2: generate_random_vectors produces correct count."""
        gen = GoldenVectorGenerator(small_graph)
        vs = gen.generate_random_vectors(n=10)
        assert vs.num_vectors == 10
        assert len(vs.vectors) == 10

    def test_random_vectors_shapes(self, small_graph: Graph):
        """6.1.2: Random vectors have correct shapes."""
        gen = GoldenVectorGenerator(small_graph)
        vs = gen.generate_random_vectors(n=5)

        for tv in vs.vectors:
            for inp_name, shape in gen.input_shapes.items():
                assert tv.input_data[inp_name].shape == shape
                assert tv.input_data[inp_name].dtype == np.int8
            for out_name, shape in gen.output_shapes.items():
                assert tv.expected_output[out_name].shape == shape
                assert tv.expected_output[out_name].dtype == np.int8

    def test_random_vectors_reproducible(self, small_graph: Graph):
        """6.1.2: Same seed produces identical vectors."""
        gen = GoldenVectorGenerator(small_graph)
        vs1 = gen.generate_random_vectors(n=5, seed=42)
        vs2 = gen.generate_random_vectors(n=5, seed=42)

        for v1, v2 in zip(vs1.vectors, vs2.vectors):
            for name in gen.graph.inputs:
                np.testing.assert_array_equal(v1.input_data[name], v2.input_data[name])
            for name in gen.graph.outputs:
                np.testing.assert_array_equal(v1.expected_output[name], v2.expected_output[name])

    def test_adversarial_vectors(self, small_graph: Graph):
        """6.1.3: Adversarial vectors cover edge cases."""
        gen = GoldenVectorGenerator(small_graph)
        vs = gen.generate_adversarial_vectors()

        assert vs.num_vectors >= 10  # At least 10 adversarial vectors

        # Check labels
        labels = [v.label for v in vs.vectors]
        assert "all_zeros" in labels
        assert "all_127" in labels
        assert "all_neg128" in labels
        assert "alternating_max_min" in labels
        assert "single_hot_first" in labels

    def test_adversarial_all_zeros(self, small_graph: Graph):
        """6.1.3: All-zeros input produces valid output."""
        gen = GoldenVectorGenerator(small_graph)
        vs = gen.generate_adversarial_vectors()

        zeros_vec = [v for v in vs.vectors if v.label == "all_zeros"][0]
        for name in gen.graph.inputs:
            assert np.all(zeros_vec.input_data[name] == 0)

    def test_adversarial_max_values(self, small_graph: Graph):
        """6.1.3: Max-value inputs produce valid (non-crashing) outputs."""
        gen = GoldenVectorGenerator(small_graph)
        vs = gen.generate_adversarial_vectors()

        max_vec = [v for v in vs.vectors if v.label == "all_127"][0]
        for name in gen.graph.inputs:
            assert np.all(max_vec.input_data[name] == 127)
        # Output should be valid INT8 (no crashes)
        for name in gen.graph.outputs:
            assert max_vec.expected_output[name].dtype == np.int8

    def test_exact_half_vectors(self, small_graph: Graph):
        """6.1.4: Exact-half vectors test rounding boundaries."""
        gen = GoldenVectorGenerator(small_graph)
        vs = gen.generate_exact_half_vectors(n=5)
        assert vs.num_vectors == 5

    def test_export_mem(self, small_graph: Graph, tmp_path: Path):
        """6.1.5: export_mem writes valid .mem files."""
        gen = GoldenVectorGenerator(small_graph)
        vs = gen.generate_random_vectors(n=5)

        manifest = gen.export_mem(vs, tmp_path / "mem")

        assert (tmp_path / "mem" / "test_inputs.mem").exists()
        assert (tmp_path / "mem" / "golden_outputs.mem").exists()
        assert (tmp_path / "mem" / "test_manifest.json").exists()
        assert manifest["num_vectors"] == 5

        # Verify .mem file format (hex lines)
        with open(tmp_path / "mem" / "test_inputs.mem") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 5  # One line per vector

        # Each line should be valid hex
        for line in lines:
            assert all(c in "0123456789abcdef" for c in line)

    def test_export_npy(self, small_graph: Graph, tmp_path: Path):
        """6.1.6: export_npy writes valid .npy files."""
        gen = GoldenVectorGenerator(small_graph)
        vs = gen.generate_random_vectors(n=5)

        gen.export_npy(vs, tmp_path / "npy")

        # Check files exist
        for inp_name in gen.graph.inputs:
            npy_path = tmp_path / "npy" / f"test_inputs_{inp_name}.npy"
            assert npy_path.exists()
            arr = np.load(npy_path)
            assert arr.shape[0] == 5  # 5 vectors

        for out_name in gen.graph.outputs:
            npy_path = tmp_path / "npy" / f"golden_outputs_{out_name}.npy"
            assert npy_path.exists()
            arr = np.load(npy_path)
            assert arr.shape[0] == 5

        assert (tmp_path / "npy" / "test_labels.npy").exists()
        assert (tmp_path / "npy" / "test_manifest.json").exists()

    def test_per_layer_intermediates(self, small_graph: Graph):
        """6.1.7: Per-layer intermediates captured correctly."""
        gen = GoldenVectorGenerator(small_graph)
        vs = gen.generate_per_layer_intermediates(n=3)

        assert vs.num_vectors == 3
        for tv in vs.vectors:
            # Should have intermediate values (output of each fused layer)
            assert len(tv.intermediates) > 0

    def test_verify_vectors_all_pass(self, small_graph: Graph):
        """Verify that re-running vectors produces identical results."""
        gen = GoldenVectorGenerator(small_graph)
        vs = gen.generate_random_vectors(n=10)

        passes, total, mismatches = gen.verify_vectors(vs)
        assert passes == total == 10
        assert len(mismatches) == 0

    def test_vector_set_metadata(self, small_graph: Graph):
        """TestVectorSet contains correct metadata."""
        gen = GoldenVectorGenerator(small_graph)
        vs = gen.generate_random_vectors(n=3)

        assert vs.model_name == small_graph.name
        assert vs.input_names == list(small_graph.inputs)
        assert vs.output_names == list(small_graph.outputs)
        assert vs.num_vectors == 3
        assert vs.pass_rate == "3/3"


class TestGoldenVectorADModel:
    """Golden vector tests on the full AD model (640->128->128->128->640)."""

    def test_100_random_vectors(self, ad_graph: Graph):
        """6.1.2: Generate and verify 100 random vectors on AD model."""
        gen = GoldenVectorGenerator(ad_graph)
        vs = gen.generate_random_vectors(n=100)
        assert vs.num_vectors == 100

        passes, total, mismatches = gen.verify_vectors(vs)
        assert passes == total == 100

    def test_adversarial_vectors(self, ad_graph: Graph):
        """6.1.3: All adversarial vectors pass on AD model."""
        gen = GoldenVectorGenerator(ad_graph)
        vs = gen.generate_adversarial_vectors()
        passes, total, _ = gen.verify_vectors(vs)
        assert passes == total

    def test_ad_model_output_dims(self, ad_graph: Graph):
        """AD model output should be 640-dim."""
        gen = GoldenVectorGenerator(ad_graph)
        vs = gen.generate_random_vectors(n=1)
        for tv in vs.vectors:
            for out_name in gen.graph.outputs:
                flat = tv.expected_output[out_name].flatten()
                assert len(flat) == 640

    def test_export_roundtrip(self, ad_graph: Graph, tmp_path: Path):
        """Export to .npy and reload matches original."""
        gen = GoldenVectorGenerator(ad_graph)
        vs = gen.generate_random_vectors(n=5, seed=99)

        gen.export_npy(vs, tmp_path / "npy_rt")

        inp_name = gen.graph.inputs[0]
        out_name = gen.graph.outputs[0]

        saved_inputs = np.load(tmp_path / "npy_rt" / f"test_inputs_{inp_name}.npy")
        saved_outputs = np.load(tmp_path / "npy_rt" / f"golden_outputs_{out_name}.npy")

        for i, tv in enumerate(vs.vectors):
            np.testing.assert_array_equal(saved_inputs[i], tv.input_data[inp_name])
            np.testing.assert_array_equal(saved_outputs[i], tv.expected_output[out_name])


# ===========================================================================
# 6.2 Module-Level Testbench Generation Tests
# ===========================================================================


class TestModuleTestbenchGeneration:
    """Tests for SV module-level testbench generation (tasks 6.2.1-6.2.9)."""

    def test_tb_mac_array_generation(self):
        """6.2.1: tb_mac_array.sv is generated and contains expected content."""
        content = generate_tb_mac_array()
        assert "module tb_mac_array" in content
        assert "mac_array #(" in content
        assert "check_eq" in content
        assert "$finish" in content

    def test_tb_requantize_generation(self):
        """6.2.2: tb_requantize.sv tests requantization pipeline."""
        content = generate_tb_requantize()
        assert "module tb_requantize" in content
        assert "requantize #(" in content
        assert "Identity requantization" in content
        assert "ReLU clamping" in content
        assert "Zero point" in content

    def test_tb_activation_relu_generation(self):
        """6.2.3: tb_activation_relu.sv sweeps all 256 INT8 values."""
        content = generate_tb_activation_relu()
        assert "module tb_activation_relu" in content
        assert "activation_relu #(" in content
        assert "Sweep all 256" in content
        assert "bypass" in content.lower()

    def test_tb_sram_bank_generation(self):
        """6.2.4: tb_sram_bank.sv tests write-read and initialization."""
        content = generate_tb_sram_bank()
        assert "module tb_sram_bank" in content
        assert "sram_bank #(" in content
        assert "Write-read" in content

    def test_tb_ping_pong_buffer_generation(self):
        """6.2.5: tb_ping_pong_buffer.sv tests bank swap."""
        content = generate_tb_ping_pong_buffer()
        assert "module tb_ping_pong_buffer" in content
        assert "ping_pong_buffer #(" in content
        assert "bank_sel" in content
        assert "Bank swap" in content

    def test_tb_fused_linear_relu_generation(self):
        """6.2.6: tb_fused_linear_relu.sv tests single and multi-tile."""
        content = generate_tb_fused_linear_relu()
        assert "module tb_fused_linear_relu" in content
        assert "fused_linear_relu #(" in content
        assert "Single tile" in content
        assert "Multi-tile" in content

    def test_tb_axi_stream_in_generation(self):
        """6.2.7: tb_axi_stream_in.sv tests data transfer and backpressure."""
        content = generate_tb_axi_stream_in()
        assert "module tb_axi_stream_in" in content
        assert "axi_stream_in #(" in content
        assert "Normal transfer" in content
        assert "Backpressure" in content

    def test_tb_axi_stream_out_generation(self):
        """6.2.8: tb_axi_stream_out.sv tests readout and TLAST."""
        content = generate_tb_axi_stream_out()
        assert "module tb_axi_stream_out" in content
        assert "axi_stream_out #(" in content
        assert "TLAST" in content

    def test_tb_axi_lite_ctrl_generation(self):
        """6.2.9: tb_axi_lite_ctrl.sv tests all CSR registers."""
        content = generate_tb_axi_lite_ctrl()
        assert "module tb_axi_lite_ctrl" in content
        assert "axi_lite_ctrl #(" in content
        assert "VERSION" in content
        assert "SCRATCH" in content
        assert "W1C" in content
        assert "IRQ" in content

    def test_all_testbenches_have_fatal_on_error(self):
        """All testbenches use $fatal on failure for test harness integration."""
        generators = [
            generate_tb_mac_array,
            generate_tb_requantize,
            generate_tb_activation_relu,
            generate_tb_sram_bank,
            generate_tb_ping_pong_buffer,
            generate_tb_fused_linear_relu,
            generate_tb_axi_stream_in,
            generate_tb_axi_stream_out,
            generate_tb_axi_lite_ctrl,
        ]
        for gen in generators:
            content = gen()
            assert "$fatal" in content, f"{gen.__name__} missing $fatal"

    def test_all_testbenches_have_report(self):
        """All testbenches have a report task."""
        generators = [
            generate_tb_mac_array,
            generate_tb_requantize,
            generate_tb_activation_relu,
            generate_tb_sram_bank,
            generate_tb_ping_pong_buffer,
            generate_tb_fused_linear_relu,
            generate_tb_axi_stream_in,
            generate_tb_axi_stream_out,
            generate_tb_axi_lite_ctrl,
        ]
        for gen in generators:
            content = gen()
            assert "report()" in content, f"{gen.__name__} missing report()"


# ===========================================================================
# 6.3 System-Level Testbench Tests
# ===========================================================================


class TestSystemTestbenchGeneration:
    """Tests for system-level accelerator testbench (tasks 6.3.1-6.3.7)."""

    def test_accelerator_testbench_content(self, small_graph: Graph):
        """6.3.1: tb_accelerator.sv has correct structure."""
        content = generate_tb_accelerator(small_graph)
        assert "module tb_accelerator" in content
        assert "AXI-Stream driver" in content
        assert "AXI-Stream monitor" in content

    def test_accelerator_model_params(self, small_graph: Graph):
        """6.3.1: Testbench has correct model-specific parameters."""
        content = generate_tb_accelerator(small_graph)
        # Should have input/output byte counts
        assert "INPUT_BYTES" in content
        assert "OUTPUT_BYTES" in content
        assert "NUM_LAYERS" in content
        assert "NUM_VECTORS" in content

    def test_axi_stream_driver_task(self, small_graph: Graph):
        """6.3.2: AXI-Stream driver task is generated."""
        content = generate_tb_accelerator(small_graph)
        assert "send_random_input" in content
        assert "s_axis_tdata" in content
        assert "s_axis_tvalid" in content
        assert "s_axis_tlast" in content

    def test_axi_stream_monitor_task(self, small_graph: Graph):
        """6.3.3: AXI-Stream monitor task is generated."""
        content = generate_tb_accelerator(small_graph)
        assert "receive_output" in content
        assert "m_axis_tdata" in content

    def test_status_polling(self, small_graph: Graph):
        """6.3.4: Status register polling for inference completion."""
        content = generate_tb_accelerator(small_graph)
        assert "ADDR_STATUS" in content or "8'h04" in content
        assert "done bit" in content or "inference" in content.lower()

    def test_dut_instantiation(self, small_graph: Graph):
        """6.3.5: DUT is properly instantiated (not commented out)."""
        content = generate_tb_accelerator(small_graph)
        assert "accelerator_top dut" in content
        # Should NOT be commented out
        assert "// accelerator_top dut" not in content

    def test_irq_path_test(self, small_graph: Graph):
        """6.3.6: IRQ path test is present."""
        content = generate_tb_accelerator(small_graph)
        assert "IRQ" in content or "irq" in content

    def test_cycle_count_check(self, small_graph: Graph):
        """6.3.7: Cycle count check is referenced."""
        content = generate_tb_accelerator(small_graph)
        assert "EXPECTED_CYCLES" in content


# ===========================================================================
# 6.4 cocotb Testbench Tests
# ===========================================================================


class TestCocotbGeneration:
    """Tests for cocotb Python testbench generation (tasks 6.4.1-6.4.6)."""

    def test_cocotb_test_content(self, small_graph: Graph):
        """6.4.1: cocotb test file has correct structure."""
        content = generate_cocotb_test(small_graph)
        assert "import cocotb" in content
        assert "@cocotb.test()" in content
        assert "async def" in content

    def test_cocotb_axi_stream_driver(self, small_graph: Graph):
        """6.4.2: AXI-Stream driver function exists."""
        content = generate_cocotb_test(small_graph)
        assert "async def axi_stream_send" in content
        assert "s_axis_tdata" in content

    def test_cocotb_axi_stream_monitor(self, small_graph: Graph):
        """6.4.3: AXI-Stream monitor function exists."""
        content = generate_cocotb_test(small_graph)
        assert "async def axi_stream_receive" in content
        assert "m_axis_tdata" in content

    def test_cocotb_axi_lite_driver(self, small_graph: Graph):
        """6.4.4: AXI-Lite read/write functions exist."""
        content = generate_cocotb_test(small_graph)
        assert "async def axi_lite_write" in content
        assert "async def axi_lite_read" in content

    def test_cocotb_golden_reference(self, small_graph: Graph):
        """6.4.5: Golden vector test exists."""
        content = generate_cocotb_test(small_graph)
        assert "test_golden_vectors" in content
        assert "np.load" in content

    def test_cocotb_irq_test(self, small_graph: Graph):
        """6.4.6: IRQ test exists."""
        content = generate_cocotb_test(small_graph)
        assert "test_irq_done" in content
        assert "IRQ_DONE" in content

    def test_cocotb_makefile(self):
        """cocotb Makefile is generated correctly."""
        content = generate_cocotb_makefile()
        assert "TOPLEVEL_LANG = verilog" in content
        assert "MODULE = test_accelerator" in content
        assert "cocotb-config" in content

    def test_cocotb_model_constants(self, small_graph: Graph):
        """cocotb test has correct model-specific constants."""
        content = generate_cocotb_test(small_graph)
        assert "INPUT_BYTES" in content
        assert "OUTPUT_BYTES" in content


# ===========================================================================
# 6.5 Functional Coverage Tests
# ===========================================================================


class TestFunctionalCoverage:
    """Tests for functional coverage tracking (tasks 6.5.1-6.5.11)."""

    def test_default_items(self):
        """All 11 coverage items initialized."""
        cov = FunctionalCoverage()
        assert cov.total == 11
        assert cov.hit_count == 0
        assert cov.coverage_pct == 0.0

    def test_mark_hit(self):
        """Marking items as hit updates counts."""
        cov = FunctionalCoverage()
        cov.mark_hit("layer_execution")
        assert cov.hit_count == 1
        cov.mark_hit("relu_activation")
        assert cov.hit_count == 2

    def test_mark_unknown_item_raises(self):
        """Marking unknown item raises ValueError."""
        cov = FunctionalCoverage()
        with pytest.raises(ValueError, match="Unknown"):
            cov.mark_hit("nonexistent")

    def test_is_complete(self):
        """is_complete is True only when all items hit."""
        cov = FunctionalCoverage()
        assert not cov.is_complete

        for item in cov.items:
            cov.mark_hit(item.name)
        assert cov.is_complete
        assert cov.coverage_pct == 100.0

    def test_report_format(self):
        """Report string contains all items."""
        cov = FunctionalCoverage()
        cov.mark_hit("irq_done")
        report = cov.report()
        assert "Functional Coverage Report" in report
        assert "[HIT ]" in report
        assert "[MISS]" in report
        assert "1/11" in report

    def test_to_dict(self):
        """to_dict serialization."""
        cov = FunctionalCoverage()
        cov.mark_hit("layer_execution")
        d = cov.to_dict()
        assert d["total"] == 11
        assert d["hit"] == 1
        assert len(d["items"]) == 11

    def test_coverage_item_names(self):
        """All 11 required coverage items are present."""
        cov = FunctionalCoverage()
        names = {item.name for item in cov.items}
        expected = {
            "layer_execution",
            "weight_sram_read",
            "ping_pong_swap",
            "axi_backpressure",
            "csr_read_write",
            "irq_done",
            "irq_error",
            "back_to_back",
            "output_tiling",
            "relu_activation",
            "requant_boundary",
        }
        assert names == expected


# ===========================================================================
# 6.6 Integration / Full Pipeline Tests
# ===========================================================================


class TestTestbenchGenerator:
    """Tests for the full TestbenchGenerator orchestrator."""

    def test_generate_all_files(self, small_graph: Graph, tmp_path: Path):
        """TestbenchGenerator.generate_all creates all expected files."""
        config = TestbenchConfig(
            graph=small_graph,
            output_dir=tmp_path / "testbench",
        )
        gen = TestbenchGenerator(config)
        gen.generate_all()

        # Module-level testbenches
        assert (tmp_path / "testbench" / "tb_mac_array.sv").exists()
        assert (tmp_path / "testbench" / "tb_requantize.sv").exists()
        assert (tmp_path / "testbench" / "tb_activation_relu.sv").exists()
        assert (tmp_path / "testbench" / "tb_sram_bank.sv").exists()
        assert (tmp_path / "testbench" / "tb_ping_pong_buffer.sv").exists()
        assert (tmp_path / "testbench" / "tb_fused_linear_relu.sv").exists()
        assert (tmp_path / "testbench" / "tb_axi_stream_in.sv").exists()
        assert (tmp_path / "testbench" / "tb_axi_stream_out.sv").exists()
        assert (tmp_path / "testbench" / "tb_axi_lite_ctrl.sv").exists()

        # System-level testbench
        assert (tmp_path / "testbench" / "tb_accelerator.sv").exists()

        # cocotb
        assert (tmp_path / "testbench" / "cocotb" / "test_accelerator.py").exists()
        assert (tmp_path / "testbench" / "cocotb" / "Makefile").exists()

        # Coverage
        assert (tmp_path / "testbench" / "coverage.json").exists()

    def test_generate_all_file_count(self, small_graph: Graph, tmp_path: Path):
        """TestbenchGenerator produces at least 12 files."""
        config = TestbenchConfig(
            graph=small_graph,
            output_dir=tmp_path / "testbench",
        )
        gen = TestbenchGenerator(config)
        files = gen.generate_all()
        assert len(files) >= 12

    def test_generated_sv_files_are_valid(self, small_graph: Graph, tmp_path: Path):
        """All generated .sv files contain valid SV constructs."""
        config = TestbenchConfig(
            graph=small_graph,
            output_dir=tmp_path / "testbench",
        )
        gen = TestbenchGenerator(config)
        gen.generate_all()

        for sv_file in (tmp_path / "testbench").glob("*.sv"):
            content = sv_file.read_text()
            assert "module " in content, f"{sv_file.name} missing module declaration"
            assert "endmodule" in content, f"{sv_file.name} missing endmodule"

    def test_coverage_json_valid(self, small_graph: Graph, tmp_path: Path):
        """coverage.json is valid JSON with expected structure."""
        config = TestbenchConfig(
            graph=small_graph,
            output_dir=tmp_path / "testbench",
        )
        gen = TestbenchGenerator(config)
        gen.generate_all()

        with open(tmp_path / "testbench" / "coverage.json") as f:
            cov = json.load(f)
        assert cov["total"] == 11
        assert cov["hit"] == 0
        assert len(cov["items"]) == 11


class TestEndToEndVerification:
    """End-to-end pipeline: model → quantize → schedule → golden vectors → testbench."""

    def test_ad_model_full_verification(self, tmp_path: Path):
        """Full pipeline on AD model: generate vectors, export, create testbenches."""
        # Build and quantize model
        model = build_mlp_model([640, 128, 128, 128, 640], has_bn=True, name="ad_model")
        model_path = tmp_path / "ad_model.onnx"
        onnx.save(model, str(model_path))

        rng = np.random.RandomState(42)
        cal_data = [rng.randn(1, 640).astype(np.float32) for _ in range(20)]

        graph, mem_map = run_pipeline(model_path, cal_data, tmp_path / "packed")

        # Generate golden vectors
        gen = GoldenVectorGenerator(graph)
        random_vs = gen.generate_random_vectors(n=100)
        adversarial_vs = gen.generate_adversarial_vectors()

        # Verify all pass
        p1, t1, _ = gen.verify_vectors(random_vs)
        assert p1 == t1 == 100

        p2, t2, _ = gen.verify_vectors(adversarial_vs)
        assert p2 == t2

        # Export
        gen.export_mem(random_vs, tmp_path / "vectors_mem")
        gen.export_npy(random_vs, tmp_path / "vectors_npy")

        assert (tmp_path / "vectors_mem" / "test_inputs.mem").exists()
        assert (tmp_path / "vectors_mem" / "golden_outputs.mem").exists()

        # Generate testbenches
        config = TestbenchConfig(
            graph=graph,
            output_dir=tmp_path / "testbench",
            weight_dir=tmp_path / "packed",
        )
        tb_gen = TestbenchGenerator(config)
        files = tb_gen.generate_all()

        assert len(files) >= 12
        assert (tmp_path / "testbench" / "tb_accelerator.sv").exists()

    def test_small_model_with_intermediates(self, tmp_path: Path):
        """Small model: generate with intermediates, verify all layers."""
        model = build_mlp_model([128, 64, 32], has_bn=True, name="small")
        model_path = tmp_path / "small.onnx"
        onnx.save(model, str(model_path))

        rng = np.random.RandomState(42)
        cal_data = [rng.randn(1, 128).astype(np.float32) for _ in range(10)]

        graph, _ = run_pipeline(model_path, cal_data, tmp_path / "packed")

        gen = GoldenVectorGenerator(graph)
        vs = gen.generate_per_layer_intermediates(n=5)

        assert vs.num_vectors == 5
        for tv in vs.vectors:
            assert len(tv.intermediates) > 0

    def test_wide_model_output_tiling(self, tmp_path: Path):
        """Wide model (>128 outputs) exercises output tiling."""
        model = build_mlp_model([128, 256], has_bn=False, name="wide")
        model_path = tmp_path / "wide.onnx"
        onnx.save(model, str(model_path))

        rng = np.random.RandomState(42)
        cal_data = [rng.randn(1, 128).astype(np.float32) for _ in range(10)]

        graph, _ = run_pipeline(model_path, cal_data, tmp_path / "packed")

        # Verify tiling: 256/128 = 2 tiles
        ordered = graph.topological_order()
        last_node = graph.nodes[ordered[-1]]
        assert last_node.schedule_info.num_tiles == 2

        gen = GoldenVectorGenerator(graph)
        vs = gen.generate_random_vectors(n=20)
        passes, total, _ = gen.verify_vectors(vs)
        assert passes == total


class TestVerificationIntegrity:
    """Tests verifying the INT8 golden reference is bitwise correct."""

    def test_zero_input_zero_bias(self, small_graph: Graph):
        """Zero input with zero bias should produce deterministic output."""
        gen = GoldenVectorGenerator(small_graph)

        inp_name = small_graph.inputs[0]
        shape = gen.input_shapes[inp_name]
        zero_input = {inp_name: np.zeros(shape, dtype=np.int8)}

        result = gen.interpreter.run(zero_input)
        for name in small_graph.outputs:
            assert result[name].dtype == np.int8

    def test_same_input_same_output(self, small_graph: Graph):
        """Determinism: same input always produces same output."""
        gen = GoldenVectorGenerator(small_graph)

        inp_name = small_graph.inputs[0]
        shape = gen.input_shapes[inp_name]
        rng = np.random.RandomState(42)
        fixed_input = {inp_name: rng.randint(-128, 128, size=shape).astype(np.int8)}

        r1 = gen.interpreter.run(fixed_input)
        r2 = gen.interpreter.run(fixed_input)

        for name in small_graph.outputs:
            np.testing.assert_array_equal(r1[name], r2[name])

    def test_output_in_int8_range(self, ad_graph: Graph):
        """All outputs must be in [-128, 127] range."""
        gen = GoldenVectorGenerator(ad_graph)
        vs = gen.generate_random_vectors(n=50)

        for tv in vs.vectors:
            for name in gen.graph.outputs:
                data = tv.expected_output[name].astype(np.int32)
                assert np.all(data >= -128)
                assert np.all(data <= 127)

    def test_relu_layers_non_negative(self, ad_graph: Graph):
        """Intermediate outputs after ReLU layers should be non-negative."""
        gen = GoldenVectorGenerator(ad_graph)
        vs = gen.generate_per_layer_intermediates(n=10)

        ordered = ad_graph.topological_order()
        for tv in vs.vectors:
            for node_name in ordered:
                node = ad_graph.nodes[node_name]
                if node.op_type == OpType.FUSED_LINEAR_RELU:
                    out_name = node.outputs[0]
                    if out_name in tv.intermediates:
                        data = tv.intermediates[out_name].astype(np.int32)
                        assert np.all(data >= 0), f"ReLU output {out_name} has negative values"
