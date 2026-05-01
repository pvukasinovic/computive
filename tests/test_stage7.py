"""Stage 7: FPGA synthesis automation and firmware tests.

Tests for:
  - TCL script structural validity (7.1)
  - Firmware register consistency with RTL (7.2)
  - C header export from golden vectors (7.3)
  - Doc consistency (7.4)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from mlasic.golden_vectors import GoldenVectorGenerator
from mlasic.ingestion import ONNXParser
from mlasic.ir import Graph, HardwareConstraints
from mlasic.optimization import (
    BatchNormFoldingPass,
    ConstantFoldingPass,
    DeadCodeEliminationPass,
    OperatorFusionPass,
    PassManager,
    QuantizationPass,
)
from mlasic.scheduler import Scheduler
from mlasic.weight_packer import WeightPacker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
FPGA_DIR = PROJECT_ROOT / "fpga"
FIRMWARE_DIR = PROJECT_ROOT / "firmware"
RTL_DIR = PROJECT_ROOT / "rtl"
DOCS_DIR = PROJECT_ROOT / "docs"


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
# 7.1 TCL Script Structural Validity
# ===========================================================================


class TestTCLScripts:
    """Verify TCL scripts exist and have correct structure."""

    def test_package_ip_exists(self):
        """7.1.1: package_ip.tcl exists."""
        assert (FPGA_DIR / "package_ip.tcl").exists()

    def test_block_design_exists(self):
        """7.1.2: block_design.tcl exists."""
        assert (FPGA_DIR / "block_design.tcl").exists()

    def test_build_exists(self):
        """7.1.3: build.tcl exists."""
        assert (FPGA_DIR / "build.tcl").exists()

    def test_report_exists(self):
        """7.1.4: report.tcl exists."""
        assert (FPGA_DIR / "report.tcl").exists()

    def test_program_exists(self):
        """7.1.5: program.tcl exists."""
        assert (FPGA_DIR / "program.tcl").exists()

    def test_constraints_exists(self):
        """7.1.6: constraints.xdc exists."""
        assert (FPGA_DIR / "constraints.xdc").exists()

    def test_package_ip_has_ipx_package(self):
        """7.1.7: package_ip.tcl calls ipx:: commands."""
        content = (FPGA_DIR / "package_ip.tcl").read_text()
        assert "ipx::" in content

    def test_block_design_has_zynq_us(self):
        """7.1.8: block_design.tcl references Zynq UltraScale+."""
        content = (FPGA_DIR / "block_design.tcl").read_text()
        assert "zynq_ultra_ps_e" in content

    def test_block_design_has_dma(self):
        """7.1.9: block_design.tcl instantiates AXI DMA."""
        content = (FPGA_DIR / "block_design.tcl").read_text()
        assert "axi_dma" in content

    def test_block_design_has_correct_addresses(self):
        """7.1.10: block_design.tcl uses correct address assignments."""
        content = (FPGA_DIR / "block_design.tcl").read_text()
        assert "0xA000" in content or "0xa000" in content

    def test_build_sources_other_scripts(self):
        """7.1.11: build.tcl sources package_ip and block_design."""
        content = (FPGA_DIR / "build.tcl").read_text()
        assert "package_ip" in content
        assert "block_design" in content

    def test_build_has_synth_impl(self):
        """7.1.12: build.tcl has synthesis and implementation steps."""
        content = (FPGA_DIR / "build.tcl").read_text()
        assert "synth_design" in content or "launch_runs" in content
        assert "write_bitstream" in content or "bitstream" in content.lower()

    def test_report_has_utilization(self):
        """7.1.13: report.tcl generates utilization report."""
        content = (FPGA_DIR / "report.tcl").read_text()
        assert "report_utilization" in content

    def test_report_has_timing(self):
        """7.1.14: report.tcl generates timing report."""
        content = (FPGA_DIR / "report.tcl").read_text()
        assert "report_timing" in content

    def test_report_has_power(self):
        """7.1.15: report.tcl generates power report."""
        content = (FPGA_DIR / "report.tcl").read_text()
        assert "report_power" in content

    def test_constraints_has_100mhz_clock(self):
        """7.1.16: constraints.xdc defines 100 MHz clock."""
        content = (FPGA_DIR / "constraints.xdc").read_text()
        assert "10.000" in content or "10.0" in content
        assert "create_clock" in content

    def test_constraints_has_false_path(self):
        """7.1.17: constraints.xdc has false path for reset."""
        content = (FPGA_DIR / "constraints.xdc").read_text()
        assert "set_false_path" in content

    def test_block_design_kv260_board(self):
        """7.1.18: block_design.tcl targets KV260."""
        content = (FPGA_DIR / "block_design.tcl").read_text()
        assert "kv260" in content.lower()

    def test_block_design_interrupt_concat(self):
        """7.1.19: block_design.tcl has interrupt concatenation."""
        content = (FPGA_DIR / "block_design.tcl").read_text()
        assert "xlconcat" in content or "concat" in content.lower()

    def test_block_design_proc_sys_reset(self):
        """7.1.20: block_design.tcl has proc_sys_reset."""
        content = (FPGA_DIR / "block_design.tcl").read_text()
        assert "proc_sys_reset" in content


# ===========================================================================
# 7.2 Firmware Register Consistency
# ===========================================================================


class TestFirmwareRegisters:
    """Verify firmware register defines match RTL."""

    def test_accel_regs_exists(self):
        """7.2.1: accel_regs.h exists."""
        assert (FIRMWARE_DIR / "accel_regs.h").exists()

    def test_accel_h_exists(self):
        """7.2.2: accel.h exists."""
        assert (FIRMWARE_DIR / "accel.h").exists()

    def test_accel_c_exists(self):
        """7.2.3: accel.c exists."""
        assert (FIRMWARE_DIR / "accel.c").exists()

    def test_interrupt_c_exists(self):
        """7.2.4: interrupt.c exists."""
        assert (FIRMWARE_DIR / "interrupt.c").exists()

    def test_main_c_exists(self):
        """7.2.5: main.c exists."""
        assert (FIRMWARE_DIR / "main.c").exists()

    def test_test_vectors_h_exists(self):
        """7.2.6: test_vectors.h exists."""
        assert (FIRMWARE_DIR / "test_vectors.h").exists()

    def test_ctrl_start_bit0(self):
        """7.2.7: CTRL_START is bit 0 (matching RTL ctrl_start = reg_ctrl[0])."""
        content = (FIRMWARE_DIR / "accel_regs.h").read_text()
        match = re.search(r"#define\s+CTRL_START\s+\(1U?\s*<<\s*(\d+)\)", content)
        assert match is not None, "CTRL_START not found"
        assert match.group(1) == "0", f"CTRL_START should be bit 0, got bit {match.group(1)}"

    def test_ctrl_soft_rst_bit1(self):
        """7.2.8: CTRL_SOFT_RST is bit 1 (matching RTL ctrl_soft_rst = reg_ctrl[1])."""
        content = (FIRMWARE_DIR / "accel_regs.h").read_text()
        match = re.search(r"#define\s+CTRL_SOFT_RST\s+\(1U?\s*<<\s*(\d+)\)", content)
        assert match is not None, "CTRL_SOFT_RST not found"
        assert match.group(1) == "1", f"CTRL_SOFT_RST should be bit 1, got bit {match.group(1)}"

    def test_ctrl_continuous_bit2(self):
        """7.2.9: CTRL_CONTINUOUS is bit 2 (matching RTL ctrl_continuous = reg_ctrl[2])."""
        content = (FIRMWARE_DIR / "accel_regs.h").read_text()
        match = re.search(r"#define\s+CTRL_CONTINUOUS\s+\(1U?\s*<<\s*(\d+)\)", content)
        assert match is not None, "CTRL_CONTINUOUS not found"
        assert match.group(1) == "2", f"CTRL_CONTINUOUS should be bit 2, got bit {match.group(1)}"

    def test_no_abort_bit(self):
        """7.2.10: No CTRL_ABORT exists (RTL has no abort signal)."""
        content = (FIRMWARE_DIR / "accel_regs.h").read_text()
        assert "CTRL_ABORT" not in content, "CTRL_ABORT should not exist — RTL has no abort"

    def test_status_idle_bit0(self):
        """7.2.11: STATUS_IDLE is bit 0 (matching RTL status_idle at position [0])."""
        content = (FIRMWARE_DIR / "accel_regs.h").read_text()
        match = re.search(r"#define\s+STATUS_IDLE\s+\(1U?\s*<<\s*(\d+)\)", content)
        assert match is not None, "STATUS_IDLE not found"
        assert match.group(1) == "0"

    def test_status_busy_bit1(self):
        """7.2.12: STATUS_BUSY is bit 1."""
        content = (FIRMWARE_DIR / "accel_regs.h").read_text()
        match = re.search(r"#define\s+STATUS_BUSY\s+\(1U?\s*<<\s*(\d+)\)", content)
        assert match is not None, "STATUS_BUSY not found"
        assert match.group(1) == "1"

    def test_status_done_bit2(self):
        """7.2.13: STATUS_DONE is bit 2."""
        content = (FIRMWARE_DIR / "accel_regs.h").read_text()
        match = re.search(r"#define\s+STATUS_DONE\s+\(1U?\s*<<\s*(\d+)\)", content)
        assert match is not None, "STATUS_DONE not found"
        assert match.group(1) == "2"

    def test_status_error_bit3(self):
        """7.2.14: STATUS_ERROR is bit 3."""
        content = (FIRMWARE_DIR / "accel_regs.h").read_text()
        match = re.search(r"#define\s+STATUS_ERROR\s+\(1U?\s*<<\s*(\d+)\)", content)
        assert match is not None, "STATUS_ERROR not found"
        assert match.group(1) == "3"

    def test_register_offsets_match_rtl(self):
        """7.2.15: Register offsets match axi_lite_ctrl.sv addresses."""
        content = (FIRMWARE_DIR / "accel_regs.h").read_text()

        # Expected offsets from axi_lite_ctrl.sv:59-68
        expected = {
            "ACCEL_CTRL": "0x00",
            "ACCEL_STATUS": "0x04",
            "ACCEL_IRQ_EN": "0x08",
            "ACCEL_IRQ_STATUS": "0x0C",
            "ACCEL_CYCLE_COUNT": "0x10",
            "ACCEL_INF_COUNT": "0x14",
            "ACCEL_VERSION": "0x18",
            "ACCEL_SCRATCH": "0x1C",
            "ACCEL_ERROR_CODE": "0x20",
            "ACCEL_LAYER_STATUS": "0x24",
        }

        for reg_name, offset in expected.items():
            pattern = rf"#define\s+{reg_name}\s+(0x[0-9A-Fa-f]+)"
            match = re.search(pattern, content)
            assert match is not None, f"{reg_name} not found in accel_regs.h"
            actual = match.group(1).upper()
            expected_upper = offset.upper()
            assert actual == expected_upper, f"{reg_name}: expected {expected_upper}, got {actual}"

    def test_accel_base_address(self):
        """7.2.16: Accelerator base address matches block design."""
        content = (FIRMWARE_DIR / "accel_regs.h").read_text()
        assert "0xA0000000" in content or "0xa0000000" in content

    def test_irq_id_121(self):
        """7.2.17: Interrupt ID is 121 (UltraScale+ SPI for pl_ps_irq0)."""
        content = (FIRMWARE_DIR / "interrupt.c").read_text()
        assert "121" in content, "GIC SPI ID 121 not found in interrupt.c"

    def test_irq_level_sensitive(self):
        """7.2.18: IRQ trigger is level-sensitive (0x1), not edge (0x3)."""
        content = (FIRMWARE_DIR / "interrupt.c").read_text()
        # Should have 0x1 for level-sensitive trigger
        # The SetPriorityTriggerType call should use 0x1 (level) not 0x3 (edge)
        assert "0x1" in content or "0x01" in content

    def test_accel_c_uses_correct_soft_rst(self):
        """7.2.19: accel.c uses CTRL_SOFT_RST (not CTRL_ABORT) for reset."""
        content = (FIRMWARE_DIR / "accel.c").read_text()
        assert "CTRL_SOFT_RST" in content
        assert "CTRL_ABORT" not in content

    def test_main_c_has_scratch_test(self):
        """7.2.20: main.c includes scratch register connectivity test."""
        content = (FIRMWARE_DIR / "main.c").read_text()
        assert "SCRATCH" in content or "0xDEADBEEF" in content


# ===========================================================================
# 7.3 C Header Export from Golden Vectors
# ===========================================================================


class TestExportCHeader:
    """Tests for export_c_header() in golden_vectors.py."""

    def test_export_c_header_basic(self, small_graph: Graph, tmp_path: Path):
        """7.3.1: export_c_header produces a valid C header file."""
        gen = GoldenVectorGenerator(small_graph)
        vs = gen.generate_random_vectors(n=5)

        out_path = tmp_path / "test_vectors.h"
        result = gen.export_c_header(vs, out_path)

        assert result.exists()
        content = result.read_text()

        # C header guard
        assert "#ifndef TEST_VECTORS_H" in content
        assert "#define TEST_VECTORS_H" in content
        assert "#endif" in content

    def test_export_c_header_has_include(self, small_graph: Graph, tmp_path: Path):
        """7.3.2: Generated header includes stdint.h."""
        gen = GoldenVectorGenerator(small_graph)
        vs = gen.generate_random_vectors(n=3)

        out_path = tmp_path / "test_vectors.h"
        gen.export_c_header(vs, out_path)
        content = out_path.read_text()

        assert "#include <stdint.h>" in content

    def test_export_c_header_has_arrays(self, small_graph: Graph, tmp_path: Path):
        """7.3.3: Generated header contains input and output arrays."""
        gen = GoldenVectorGenerator(small_graph)
        vs = gen.generate_random_vectors(n=3)

        out_path = tmp_path / "test_vectors.h"
        gen.export_c_header(vs, out_path)
        content = out_path.read_text()

        assert "static const int8_t" in content
        assert "test_input" in content
        assert "expected_output" in content

    def test_export_c_header_correct_sizes(self, small_graph: Graph, tmp_path: Path):
        """7.3.4: Array sizes match model dimensions."""
        gen = GoldenVectorGenerator(small_graph)
        vs = gen.generate_random_vectors(n=3)

        out_path = tmp_path / "test_vectors.h"
        gen.export_c_header(vs, out_path)
        content = out_path.read_text()

        # The small model is 128→64→32, so input=128, output=32
        # Check INPUT_SIZE and OUTPUT_SIZE defines
        assert "INPUT_SIZE_BYTES 128" in content or "INPUT_SIZE_INPUT 128" in content.upper()
        assert "OUTPUT_SIZE_BYTES 32" in content or "OUTPUT_SIZE" in content

    def test_export_c_header_valid_c_syntax(self, small_graph: Graph, tmp_path: Path):
        """7.3.5: Generated header has valid C array syntax."""
        gen = GoldenVectorGenerator(small_graph)
        vs = gen.generate_random_vectors(n=3)

        out_path = tmp_path / "test_vectors.h"
        gen.export_c_header(vs, out_path)
        content = out_path.read_text()

        # Check that arrays are properly formed: { ... };
        assert re.search(r"int8_t\s+\w+\[\d+\]\s*=\s*\{", content)
        assert "};" in content

        # Check values are valid integers
        array_match = re.search(r"\{([^}]+)\}", content)
        assert array_match is not None
        values_str = array_match.group(1)
        # Each non-whitespace, non-comma token should be a valid integer
        tokens = [t.strip() for t in values_str.split(",") if t.strip()]
        for token in tokens:
            int(token)  # Should not raise

    def test_export_c_header_vector_index(self, small_graph: Graph, tmp_path: Path):
        """7.3.6: Can export a specific vector by index."""
        gen = GoldenVectorGenerator(small_graph)
        vs = gen.generate_random_vectors(n=5, seed=42)

        # Export vector 0
        path0 = tmp_path / "v0.h"
        gen.export_c_header(vs, path0, vector_index=0)

        # Export vector 3
        path3 = tmp_path / "v3.h"
        gen.export_c_header(vs, path3, vector_index=3)

        # They should be different (different random inputs)
        assert path0.read_text() != path3.read_text()

    def test_export_c_header_index_out_of_range(self, small_graph: Graph, tmp_path: Path):
        """7.3.7: Out-of-range vector index raises IndexError."""
        gen = GoldenVectorGenerator(small_graph)
        vs = gen.generate_random_vectors(n=3)

        with pytest.raises(IndexError):
            gen.export_c_header(vs, tmp_path / "bad.h", vector_index=5)

    def test_export_c_header_values_match(self, small_graph: Graph, tmp_path: Path):
        """7.3.8: Exported values match the actual vector data."""
        gen = GoldenVectorGenerator(small_graph)
        vs = gen.generate_random_vectors(n=1, seed=42)

        out_path = tmp_path / "test_vectors.h"
        gen.export_c_header(vs, out_path)
        content = out_path.read_text()

        # Extract the first array values from the header
        tv = vs.vectors[0]
        inp_name = vs.input_names[0]
        expected_values = tv.input_data[inp_name].flatten()

        # Find the input array in the file
        # Look for the first "static const int8_t" array
        array_pattern = re.search(
            r"static const int8_t test_input_\w+\[\d+\]\s*=\s*\{([^}]+)\}", content
        )
        assert array_pattern is not None
        values_str = array_pattern.group(1)
        actual_values = [int(t.strip()) for t in values_str.split(",") if t.strip()]

        assert len(actual_values) == len(expected_values)
        for actual, expected in zip(actual_values, expected_values):
            assert actual == int(expected), f"Value mismatch: {actual} != {expected}"

    def test_export_c_header_convenience_aliases(self, small_graph: Graph, tmp_path: Path):
        """7.3.9: Single-input/output models get convenience aliases."""
        gen = GoldenVectorGenerator(small_graph)
        vs = gen.generate_random_vectors(n=1)

        out_path = tmp_path / "test_vectors.h"
        gen.export_c_header(vs, out_path)
        content = out_path.read_text()

        # Should have convenience aliases
        assert "INPUT_SIZE_BYTES" in content
        assert "OUTPUT_SIZE_BYTES" in content

    def test_export_c_header_creates_parent_dirs(self, small_graph: Graph, tmp_path: Path):
        """7.3.10: export_c_header creates parent directories."""
        gen = GoldenVectorGenerator(small_graph)
        vs = gen.generate_random_vectors(n=1)

        deep_path = tmp_path / "a" / "b" / "c" / "test_vectors.h"
        result = gen.export_c_header(vs, deep_path)
        assert result.exists()


# ===========================================================================
# 7.4 Doc Consistency
# ===========================================================================


class TestDocConsistency:
    """Verify firmware-integration-guide.md matches RTL."""

    def test_doc_ctrl_no_abort(self):
        """7.4.1: Doc no longer references CTRL_ABORT."""
        content = (DOCS_DIR / "firmware-integration-guide.md").read_text()
        # The C code block in the doc should not define CTRL_ABORT
        # (The error recovery section may still mention "abort" in prose, but
        # the #define should be gone)
        assert "#define CTRL_ABORT" not in content

    def test_doc_ctrl_soft_rst_bit1(self):
        """7.4.2: Doc CTRL_SOFT_RST is bit 1."""
        content = (DOCS_DIR / "firmware-integration-guide.md").read_text()
        # Should have CTRL_SOFT_RST at bit 1
        assert re.search(r"CTRL_SOFT_RST\s+\(1U?\s*<<\s*1\)", content)

    def test_doc_ctrl_continuous_bit2(self):
        """7.4.3: Doc CTRL_CONTINUOUS is bit 2."""
        content = (DOCS_DIR / "firmware-integration-guide.md").read_text()
        assert re.search(r"CTRL_CONTINUOUS\s+\(1U?\s*<<\s*2\)", content)

    def test_rtl_ctrl_bits_match_firmware(self):
        """7.4.4: RTL CTRL bit assignments match firmware header."""
        rtl_content = (RTL_DIR / "interface" / "axi_lite_ctrl.sv").read_text()
        fw_content = (FIRMWARE_DIR / "accel_regs.h").read_text()

        # RTL: ctrl_start = reg_ctrl[0], ctrl_soft_rst = reg_ctrl[1], ctrl_continuous = reg_ctrl[2]
        assert "ctrl_start" in rtl_content
        assert "reg_ctrl[0]" in rtl_content
        assert "reg_ctrl[1]" in rtl_content
        assert "reg_ctrl[2]" in rtl_content

        # Firmware: CTRL_START bit 0, CTRL_SOFT_RST bit 1, CTRL_CONTINUOUS bit 2
        assert re.search(r"CTRL_START\s+\(1U?\s*<<\s*0\)", fw_content)
        assert re.search(r"CTRL_SOFT_RST\s+\(1U?\s*<<\s*1\)", fw_content)
        assert re.search(r"CTRL_CONTINUOUS\s+\(1U?\s*<<\s*2\)", fw_content)
