"""Tests for MLASIC Stage 5: RTL Generation.

Tests structural correctness of generated RTL:
  - parameters.svh values match Schedule fields
  - accelerator_top.sv has correct instantiations and FSM states
  - All expected files present in output directory
  - Weight files copied correctly
  - Verilator lint (if available)
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mlasic.ingestion import ONNXParser
from mlasic.ir import Graph
from mlasic.optimization import (
    BatchNormFoldingPass,
    ConstantFoldingPass,
    DeadCodeEliminationPass,
    OperatorFusionPass,
    PassManager,
    QuantizationPass,
)
from mlasic.rtl_gen import RTLGenerator
from mlasic.scheduler import Scheduler
from mlasic.weight_packer import WeightPacker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scheduled_ad_graph(ad_model_path: Path, test_vectors: np.ndarray) -> Graph:
    """Run Stages 1-3 on the AD model, return scheduled graph."""
    parser = ONNXParser(ad_model_path)
    graph = parser.parse()

    pm = PassManager()
    pm.add_pass(ConstantFoldingPass())
    pm.add_pass(DeadCodeEliminationPass())
    pm.add_pass(BatchNormFoldingPass())
    pm.add_pass(OperatorFusionPass())
    pm.add_pass(QuantizationPass(calibration_data=[v for v in test_vectors[:20]]))
    graph = pm.run(graph, verify=True)

    scheduler = Scheduler()
    scheduler.schedule(graph)

    return graph


@pytest.fixture
def packed_output(scheduled_ad_graph: Graph, tmp_path: Path) -> tuple[dict, Path, Graph]:
    """Run Stage 4 (weight packing), return (memory_map, weight_dir, graph)."""
    weight_dir = tmp_path / "weights"
    packer = WeightPacker(scheduled_ad_graph, weight_dir)
    memory_map = packer.pack_weights()
    return memory_map, weight_dir, scheduled_ad_graph


@pytest.fixture
def generated_rtl(packed_output: tuple[dict, Path, Graph], tmp_path: Path) -> tuple[Path, Graph]:
    """Run Stage 5 (RTL generation), return (output_dir, graph)."""
    memory_map, weight_dir, graph = packed_output
    output_dir = tmp_path / "rtl_output"
    gen = RTLGenerator(
        graph=graph,
        weight_dir=weight_dir,
        output_dir=output_dir,
    )
    gen.generate_all()
    return output_dir, graph


# ---------------------------------------------------------------------------
# Test: RTLGenerator construction
# ---------------------------------------------------------------------------


class TestRTLGeneratorConstruction:
    """Test RTLGenerator initialization and preconditions."""

    def test_requires_scheduled_stage(self, tmp_path: Path) -> None:
        """RTLGenerator rejects graphs not in 'scheduled' stage."""
        from mlasic.ir import Graph

        graph = Graph(name="test", nodes={}, tensors={}, inputs=[], outputs=[], stage="raw")
        with pytest.raises(ValueError, match="scheduled"):
            RTLGenerator(graph=graph, weight_dir=tmp_path, output_dir=tmp_path)

    def test_accepts_scheduled_graph(
        self, packed_output: tuple[dict, Path, Graph], tmp_path: Path
    ) -> None:
        """RTLGenerator accepts a properly scheduled graph."""
        _, weight_dir, graph = packed_output
        gen = RTLGenerator(graph=graph, weight_dir=weight_dir, output_dir=tmp_path / "out")
        assert gen.graph.stage == "scheduled"


# ---------------------------------------------------------------------------
# Test: File presence
# ---------------------------------------------------------------------------


class TestFilePresence:
    """Test that all expected files are generated."""

    def test_all_sv_files_present(self, generated_rtl: tuple[Path, Graph]) -> None:
        """All module library .sv files are copied to output."""
        output_dir, _ = generated_rtl
        expected = [
            "compute/mac_array.sv",
            "compute/requantize.sv",
            "compute/activation_relu.sv",
            "memory/sram_bank.sv",
            "memory/ping_pong_buffer.sv",
            "layer/fused_linear_relu.sv",
            "layer/byte_select.sv",
            "layer/bias_unpack.sv",
            "interface/axi_stream_in.sv",
            "interface/axi_stream_out.sv",
            "interface/axi_lite_ctrl.sv",
        ]
        for f in expected:
            assert (output_dir / f).is_file(), f"Missing: {f}"

    def test_constraints_xdc_present(self, generated_rtl: tuple[Path, Graph]) -> None:
        """constraints.xdc is copied."""
        output_dir, _ = generated_rtl
        assert (output_dir / "constraints" / "constraints.xdc").is_file()

    def test_generated_files_present(self, generated_rtl: tuple[Path, Graph]) -> None:
        """parameters.svh and accelerator_top.sv are generated."""
        output_dir, _ = generated_rtl
        assert (output_dir / "parameters.svh").is_file()
        assert (output_dir / "accelerator_top.sv").is_file()

    def test_weight_files_present(self, generated_rtl: tuple[Path, Graph]) -> None:
        """weight_bank.mem and bias_bank.mem are copied to output."""
        output_dir, _ = generated_rtl
        assert (output_dir / "weight_bank.mem").is_file()
        assert (output_dir / "bias_bank.mem").is_file()

    def test_per_layer_weight_files_present(self, generated_rtl: tuple[Path, Graph]) -> None:
        """Per-layer weight/bias .mem files are copied."""
        output_dir, graph = generated_rtl
        num_layers = len(graph.topological_order())
        for i in range(num_layers):
            assert (output_dir / f"weights_layer{i}.mem").is_file()
            assert (output_dir / f"biases_layer{i}.mem").is_file()


# ---------------------------------------------------------------------------
# Test: parameters.svh correctness
# ---------------------------------------------------------------------------


class TestParametersSvh:
    """Test that parameters.svh matches the Schedule."""

    def test_num_layers(self, generated_rtl: tuple[Path, Graph]) -> None:
        """NUM_LAYERS matches number of nodes in graph."""
        output_dir, graph = generated_rtl
        content = (output_dir / "parameters.svh").read_text()
        num_layers = len(graph.topological_order())
        assert f"NUM_LAYERS   = {num_layers};" in content

    def test_max_dim(self, generated_rtl: tuple[Path, Graph]) -> None:
        """MAX_DIM is the maximum of all input/output dimensions."""
        output_dir, graph = generated_rtl
        content = (output_dir / "parameters.svh").read_text()
        # AD model: max dim is 640
        assert "MAX_DIM      = 640;" in content

    def test_parallelism(self, generated_rtl: tuple[Path, Graph]) -> None:
        """PARALLELISM matches constraints."""
        output_dir, _ = generated_rtl
        content = (output_dir / "parameters.svh").read_text()
        assert "PARALLELISM  = 128;" in content

    def test_layer_in_dims(self, generated_rtl: tuple[Path, Graph]) -> None:
        """LAYER_IN_DIM array matches schedule."""
        output_dir, graph = generated_rtl
        content = (output_dir / "parameters.svh").read_text()
        ordered = [graph.nodes[n] for n in graph.topological_order()]
        for node in ordered:
            ls = node.schedule_info
            assert f"16'd{ls.input_dim}" in content

    def test_layer_out_dims(self, generated_rtl: tuple[Path, Graph]) -> None:
        """LAYER_OUT_DIM array matches schedule."""
        output_dir, graph = generated_rtl
        content = (output_dir / "parameters.svh").read_text()
        ordered = [graph.nodes[n] for n in graph.topological_order()]
        for node in ordered:
            ls = node.schedule_info
            assert f"16'd{ls.output_dim}" in content

    def test_weight_bases(self, generated_rtl: tuple[Path, Graph]) -> None:
        """WEIGHT_BASE array matches schedule."""
        output_dir, graph = generated_rtl
        content = (output_dir / "parameters.svh").read_text()
        ordered = [graph.nodes[n] for n in graph.topological_order()]
        weight_bases = [str(node.schedule_info.weight_start_row) for node in ordered]
        # Check each base appears in the WEIGHT_BASE line
        wb_line = [line for line in content.split("\n") if "WEIGHT_BASE" in line][0]
        for wb in weight_bases:
            assert f"16'd{wb}" in wb_line

    def test_bias_bases(self, generated_rtl: tuple[Path, Graph]) -> None:
        """BIAS_BASE array matches schedule."""
        output_dir, graph = generated_rtl
        content = (output_dir / "parameters.svh").read_text()
        ordered = [graph.nodes[n] for n in graph.topological_order()]
        bias_bases = [str(node.schedule_info.bias_start_row) for node in ordered]
        bb_line = [line for line in content.split("\n") if "BIAS_BASE" in line][0]
        for bb in bias_bases:
            assert f"16'd{bb}" in bb_line

    def test_relu_flags(self, generated_rtl: tuple[Path, Graph]) -> None:
        """LAYER_RELU array matches schedule."""
        output_dir, graph = generated_rtl
        content = (output_dir / "parameters.svh").read_text()
        relu_line = [line for line in content.split("\n") if "LAYER_RELU" in line][0]
        ordered = [graph.nodes[n] for n in graph.topological_order()]
        for node in ordered:
            ls = node.schedule_info
            expected_val = "1'b1" if ls.has_relu else "1'b0"
            assert expected_val in relu_line

    def test_requant_params_present(self, generated_rtl: tuple[Path, Graph]) -> None:
        """Requantization parameters are present in parameters.svh."""
        output_dir, _ = generated_rtl
        content = (output_dir / "parameters.svh").read_text()
        assert "REQUANT_M" in content
        assert "REQUANT_SHIFT" in content
        assert "REQUANT_ZP" in content

    def test_sram_depths(self, generated_rtl: tuple[Path, Graph]) -> None:
        """SRAM depth parameters are correct."""
        output_dir, _ = generated_rtl
        content = (output_dir / "parameters.svh").read_text()
        assert "WEIGHT_DEPTH = 1536;" in content
        assert "BIAS_DEPTH   = 32;" in content
        assert "ACT_DEPTH    = 80;" in content  # 640/8 = 80


# ---------------------------------------------------------------------------
# Test: accelerator_top.sv correctness
# ---------------------------------------------------------------------------


class TestAcceleratorTop:
    """Test structural correctness of generated accelerator_top.sv."""

    def test_module_declaration(self, generated_rtl: tuple[Path, Graph]) -> None:
        """accelerator_top.sv has correct module declaration."""
        output_dir, _ = generated_rtl
        content = (output_dir / "accelerator_top.sv").read_text()
        assert "module accelerator_top" in content

    def test_includes_parameters(self, generated_rtl: tuple[Path, Graph]) -> None:
        """accelerator_top.sv includes parameters.svh."""
        output_dir, _ = generated_rtl
        content = (output_dir / "accelerator_top.sv").read_text()
        assert '`include "parameters.svh"' in content

    def test_fsm_states(self, generated_rtl: tuple[Path, Graph]) -> None:
        """accelerator_top.sv has all required FSM states."""
        output_dir, _ = generated_rtl
        content = (output_dir / "accelerator_top.sv").read_text()
        for state in ["S_IDLE", "S_RECV_IN", "S_RUN_LAYER", "S_NEXT_LAYER", "S_SEND_OUT", "S_DONE"]:
            assert state in content, f"Missing FSM state: {state}"

    def test_weight_sram_instantiation(self, generated_rtl: tuple[Path, Graph]) -> None:
        """accelerator_top.sv instantiates weight_bank SRAM."""
        output_dir, _ = generated_rtl
        content = (output_dir / "accelerator_top.sv").read_text()
        assert "weight_bank" in content
        assert "weight_bank.mem" in content

    def test_bias_sram_instantiation(self, generated_rtl: tuple[Path, Graph]) -> None:
        """accelerator_top.sv instantiates bias_bank SRAM."""
        output_dir, _ = generated_rtl
        content = (output_dir / "accelerator_top.sv").read_text()
        assert "bias_bank" in content
        assert "bias_bank.mem" in content

    def test_mac_array_instantiation(self, generated_rtl: tuple[Path, Graph]) -> None:
        """accelerator_top.sv instantiates one MAC array."""
        output_dir, _ = generated_rtl
        content = (output_dir / "accelerator_top.sv").read_text()
        assert "mac_array" in content
        assert "u_mac" in content

    def test_layer_controller_instantiation(self, generated_rtl: tuple[Path, Graph]) -> None:
        """accelerator_top.sv instantiates fused_linear_relu layer controller."""
        output_dir, _ = generated_rtl
        content = (output_dir / "accelerator_top.sv").read_text()
        assert "fused_linear_relu" in content
        assert "u_layer_ctrl" in content

    def test_axi_lite_ctrl_instantiation(self, generated_rtl: tuple[Path, Graph]) -> None:
        """accelerator_top.sv instantiates AXI-Lite control."""
        output_dir, _ = generated_rtl
        content = (output_dir / "accelerator_top.sv").read_text()
        assert "axi_lite_ctrl" in content
        assert "u_ctrl" in content

    def test_axi_stream_in_instantiation(self, generated_rtl: tuple[Path, Graph]) -> None:
        """accelerator_top.sv instantiates AXI-Stream input."""
        output_dir, _ = generated_rtl
        content = (output_dir / "accelerator_top.sv").read_text()
        assert "axi_stream_in" in content
        assert "u_axis_in" in content

    def test_axi_stream_out_instantiation(self, generated_rtl: tuple[Path, Graph]) -> None:
        """accelerator_top.sv instantiates AXI-Stream output."""
        output_dir, _ = generated_rtl
        content = (output_dir / "accelerator_top.sv").read_text()
        assert "axi_stream_out" in content
        assert "u_axis_out" in content

    def test_act_bank_instantiations(self, generated_rtl: tuple[Path, Graph]) -> None:
        """accelerator_top.sv instantiates two activation banks (A and B)."""
        output_dir, _ = generated_rtl
        content = (output_dir / "accelerator_top.sv").read_text()
        assert "act_a" in content
        assert "act_b" in content

    def test_interrupt_output(self, generated_rtl: tuple[Path, Graph]) -> None:
        """accelerator_top.sv has interrupt output."""
        output_dir, _ = generated_rtl
        content = (output_dir / "accelerator_top.sv").read_text()
        assert "output logic irq" in content

    def test_layer_idx_comparison(self, generated_rtl: tuple[Path, Graph]) -> None:
        """accelerator_top.sv compares layer_idx to NUM_LAYERS - 1."""
        output_dir, _ = generated_rtl
        content = (output_dir / "accelerator_top.sv").read_text()
        assert "NUM_LAYERS - 1" in content

    def test_model_name_in_comment(self, generated_rtl: tuple[Path, Graph]) -> None:
        """accelerator_top.sv has the model name in a comment."""
        output_dir, graph = generated_rtl
        content = (output_dir / "accelerator_top.sv").read_text()
        assert graph.name in content


# ---------------------------------------------------------------------------
# Test: Weight file integrity
# ---------------------------------------------------------------------------


class TestWeightFileIntegrity:
    """Test that copied weight files are bit-identical to originals."""

    def test_weight_bank_identical(
        self,
        packed_output: tuple[dict, Path, Graph],
        generated_rtl: tuple[Path, Graph],
    ) -> None:
        """weight_bank.mem in output is identical to Stage 4 output."""
        _, weight_dir, _ = packed_output
        output_dir, _ = generated_rtl
        original = (weight_dir / "weight_bank.mem").read_text()
        copied = (output_dir / "weight_bank.mem").read_text()
        assert original == copied

    def test_bias_bank_identical(
        self,
        packed_output: tuple[dict, Path, Graph],
        generated_rtl: tuple[Path, Graph],
    ) -> None:
        """bias_bank.mem in output is identical to Stage 4 output."""
        _, weight_dir, _ = packed_output
        output_dir, _ = generated_rtl
        original = (weight_dir / "bias_bank.mem").read_text()
        copied = (output_dir / "bias_bank.mem").read_text()
        assert original == copied


# ---------------------------------------------------------------------------
# Test: Verilator lint (optional — skipped if verilator not installed)
# ---------------------------------------------------------------------------


def _verilator_available() -> bool:
    """Check if verilator is available on PATH."""
    return shutil.which("verilator") is not None


@pytest.mark.skipif(not _verilator_available(), reason="verilator not installed")
class TestVerilatorLint:
    """Run verilator --lint-only on generated RTL."""

    def test_lint_leaf_modules(self, generated_rtl: tuple[Path, Graph]) -> None:
        """Leaf modules (no sub-instantiations) pass verilator lint."""
        output_dir, _ = generated_rtl
        # Leaf modules that don't instantiate other modules
        leaf_files = [
            "compute/activation_relu.sv",
            "compute/requantize.sv",
            "memory/sram_bank.sv",
            "layer/byte_select.sv",
            "layer/bias_unpack.sv",
            "interface/axi_lite_ctrl.sv",
            "interface/axi_stream_in.sv",
            "interface/axi_stream_out.sv",
        ]

        for f in leaf_files:
            sv_file = output_dir / f
            result = subprocess.run(
                [
                    "verilator",
                    "--lint-only",
                    "--language",
                    "1800-2017",
                    "-Wall",
                    str(sv_file),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, (
                f"Verilator lint failed for {f}:\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

    def test_lint_hierarchical_modules(self, generated_rtl: tuple[Path, Graph]) -> None:
        """Hierarchical modules pass verilator lint with dependencies."""
        output_dir, _ = generated_rtl
        # Map each hierarchical module to its specific dependencies
        hier_modules = {
            "compute/mac_array.sv": [
                "compute/requantize.sv",
            ],
            "memory/ping_pong_buffer.sv": [
                "memory/sram_bank.sv",
            ],
            "layer/fused_linear_relu.sv": [],  # control-only, no sub-instantiations
        }

        for f, dep_list in hier_modules.items():
            sv_file = output_dir / f
            dep_files = [str(output_dir / d) for d in dep_list]
            # Extract module name from filename (e.g. mac_array.sv -> mac_array)
            top_module = sv_file.stem
            result = subprocess.run(
                [
                    "verilator",
                    "--lint-only",
                    "--language",
                    "1800-2017",
                    "-Wall",
                    "--top-module",
                    top_module,
                    str(sv_file),
                    *dep_files,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, (
                f"Verilator lint failed for {f}:\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )
