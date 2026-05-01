#!/usr/bin/env python3
"""MLASIC Compiler Validation: MLPerf Tiny Anomaly Detection Model.

Builds the 640→128→128→128→640 MLP autoencoder, runs it through the full
6-stage compiler pipeline, and validates every generated artifact with
explicit PASS/FAIL checks.

Usage:
    python validate_compiler.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import onnx

# ── Reuse project model builder ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "tests"))
from conftest import build_ad_model

from mlasic.golden_vectors import GoldenVectorGenerator
from mlasic.ingestion import ONNXParser
from mlasic.int8_interpreter import INT8Interpreter
from mlasic.ir import HardwareConstraints, OpType
from mlasic.optimization import (
    BatchNormFoldingPass,
    ConstantFoldingPass,
    DeadCodeEliminationPass,
    OperatorFusionPass,
    PassManager,
    QuantizationPass,
)
from mlasic.rtl_gen import RTLGenerator
from mlasic.scheduler import Scheduler, schedule_to_json
from mlasic.testbench_gen import VerifConfig, VerifGenerator
from mlasic.weight_packer import WeightPacker, load_bias_mem, load_weight_mem

# ── Test infrastructure ──────────────────────────────────────────────────

_pass_count = 0
_fail_count = 0
_checks: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record a PASS/FAIL check."""
    global _pass_count, _fail_count
    if condition:
        _pass_count += 1
        tag = "\033[32mPASS\033[0m"
    else:
        _fail_count += 1
        tag = "\033[31mFAIL\033[0m"
    suffix = f" -- {detail}" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    _checks.append((name, condition, detail))


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")


# ── Expected constants for AD model ─────────────────────────────────────

AD_LAYERS = [
    {"input_dim": 640, "output_dim": 128, "has_relu": True},
    {"input_dim": 128, "output_dim": 128, "has_relu": True},
    {"input_dim": 128, "output_dim": 128, "has_relu": True},
    {"input_dim": 128, "output_dim": 640, "has_relu": False},
]
NUM_LAYERS = 4
PARALLELISM = 128
CLOCK_MHZ = 100
INPUT_DIM = 640
OUTPUT_DIM = 640

# Cycle formula: tiles_per_layer * (input_dim + 149)
EXPECTED_TILES = [
    640 // 128,   # Layer 0: 640→128, 1 tile (output fits in 128)
    # Actually: output_dim / parallelism, Layer 0: 128/128 = 1
    128 // 128,   # Layer 1: 128→128, 1 tile
    128 // 128,   # Layer 2: 128→128, 1 tile
    640 // 128,   # Layer 3: 128→640, 5 tiles
]
# Recompute: tiles = ceil(output_dim / PARALLELISM)
EXPECTED_TILES = [1, 1, 1, 5]

# Weight sizes
EXPECTED_WEIGHT_ROWS = [
    640 * 128 // 128,   # 640  rows (640 input * 128 output / 128 per row)
    128 * 128 // 128,   # 128  rows
    128 * 128 // 128,   # 128  rows
    128 * 640 // 128,   # 640  rows
]
TOTAL_WEIGHT_ROWS = sum(EXPECTED_WEIGHT_ROWS)  # 1536
TOTAL_WEIGHT_BYTES = TOTAL_WEIGHT_ROWS * 128   # 196,608

# ── Main validation ─────────────────────────────────────────────────────


def main() -> None:
    t0 = time.perf_counter()
    output_dir = Path("output/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  MLASIC Compiler Validation")
    print("  Model: MLPerf Tiny Anomaly Detection (640→128→128→128→640)")
    print("=" * 70)

    # ════════════════════════════════════════════════════════════════════
    # STAGE 0: Build and save ONNX model
    # ════════════════════════════════════════════════════════════════════
    section("Stage 0: Build ONNX Model")

    model = build_ad_model()
    model_path = output_dir / "ad_model.onnx"
    onnx.save(model, str(model_path))

    check("ONNX model built", model is not None)
    check("ONNX model saved", model_path.exists())
    check("ONNX node count", len(model.graph.node) == 14,
          f"expected 14 (4×MatMul+Add + 3×BN + 3×ReLU), got {len(model.graph.node)}")
    check("ONNX input shape", list(model.graph.input[0].type.tensor_type.shape.dim[1].dim_value for _ in [0]) == [640])
    check("ONNX opset version", model.opset_import[0].version == 17,
          f"got {model.opset_import[0].version}")

    # ════════════════════════════════════════════════════════════════════
    # STAGE 1: ONNX Ingestion
    # ════════════════════════════════════════════════════════════════════
    section("Stage 1: ONNX Ingestion")
    t1 = time.perf_counter()

    graph = ONNXParser(model_path).parse()
    dt1 = time.perf_counter() - t1

    check("Parse succeeds", graph is not None)
    check("Stage is 'raw'", graph.stage == "raw", f"got '{graph.stage}'")
    check("Node count = 14", len(graph.nodes) == 14, f"got {len(graph.nodes)}")
    check("Has single input", len(graph.inputs) == 1)
    check("Has single output", len(graph.outputs) == 1)

    # Check all expected op types present
    op_types = {n.op_type for n in graph.nodes.values()}
    check("Contains MatMul ops", OpType.MATMUL in op_types)
    check("Contains Add ops", OpType.ADD in op_types)
    check("Contains BN ops", OpType.BATCH_NORM in op_types)
    check("Contains ReLU ops", OpType.RELU in op_types)

    # Validate IR invariants
    try:
        graph.validate("raw")
        check("IR invariants pass (raw)", True)
    except Exception as e:
        check("IR invariants pass (raw)", False, str(e))

    print(f"  Time: {dt1:.3f}s")

    # ════════════════════════════════════════════════════════════════════
    # STAGE 2: Graph Optimization
    # ════════════════════════════════════════════════════════════════════
    section("Stage 2: Graph Optimization")
    t2 = time.perf_counter()

    rng = np.random.RandomState(42)
    cal_data = [rng.randn(1, 640).astype(np.float32) for _ in range(20)]

    pm = PassManager()
    pm.add_pass(ConstantFoldingPass())
    pm.add_pass(DeadCodeEliminationPass())
    pm.add_pass(BatchNormFoldingPass())
    pm.add_pass(OperatorFusionPass())
    pm.add_pass(QuantizationPass(calibration_data=cal_data))
    graph = pm.run(graph, verify=True)
    dt2 = time.perf_counter() - t2

    check("Optimization succeeds", graph is not None)
    check("Stage is 'quantized'", graph.stage == "quantized", f"got '{graph.stage}'")
    check("4 fused nodes", len(graph.nodes) == 4, f"got {len(graph.nodes)}")

    # Verify each fused node
    topo = graph.topological_order()
    check("4 nodes in topological order", len(topo) == 4)

    for i, name in enumerate(topo):
        node = graph.nodes[name]
        expected = AD_LAYERS[i]
        attrs = node.fused_attrs

        if i < 3:
            check(f"Layer {i} is FusedLinearReLU",
                  node.op_type == OpType.FUSED_LINEAR_RELU,
                  f"got {node.op_type.name}")
        else:
            check(f"Layer {i} is FusedLinear",
                  node.op_type == OpType.FUSED_LINEAR,
                  f"got {node.op_type.name}")

        check(f"Layer {i} input_dim={expected['input_dim']}",
              attrs.input_dim == expected["input_dim"],
              f"got {attrs.input_dim}")
        check(f"Layer {i} output_dim={expected['output_dim']}",
              attrs.output_dim == expected["output_dim"],
              f"got {attrs.output_dim}")
        check(f"Layer {i} has_relu={expected['has_relu']}",
              attrs.has_relu == expected["has_relu"],
              f"got {attrs.has_relu}")
        check(f"Layer {i} is quantized",
              attrs.is_quantized,
              "missing quantization params")
        check(f"Layer {i} requant_scale_fixed is int",
              isinstance(attrs.requant_scale_fixed, int),
              f"type={type(attrs.requant_scale_fixed).__name__}")
        check(f"Layer {i} requant_shift is int",
              isinstance(attrs.requant_shift, int),
              f"type={type(attrs.requant_shift).__name__}")
        check(f"Layer {i} requant_scale_fixed fits INT32",
              0 < attrs.requant_scale_fixed < 2**31,
              f"value={attrs.requant_scale_fixed}")

    # Validate quantized IR invariants
    try:
        graph.validate("quantized")
        check("IR invariants pass (quantized)", True)
    except Exception as e:
        check("IR invariants pass (quantized)", False, str(e))

    print(f"  Time: {dt2:.3f}s")

    # ════════════════════════════════════════════════════════════════════
    # STAGE 3: Dataflow Scheduling
    # ════════════════════════════════════════════════════════════════════
    section("Stage 3: Dataflow Scheduling")
    t3 = time.perf_counter()

    hw = HardwareConstraints()
    schedule = Scheduler(constraints=hw).schedule(graph)
    dt3 = time.perf_counter() - t3

    check("Scheduling succeeds", schedule is not None)
    check("Stage is 'scheduled'", graph.stage == "scheduled", f"got '{graph.stage}'")
    check("4 layer schedules", len(schedule.layers) == 4,
          f"got {len(schedule.layers)}")
    check("Clock = 100 MHz", schedule.clock_mhz == 100)

    # Verify cycle counts per layer
    for i, layer in enumerate(schedule.layers):
        expected_tiles = EXPECTED_TILES[i]
        expected_in_dim = AD_LAYERS[i]["input_dim"]
        expected_cycles = expected_tiles * (expected_in_dim + 149)
        check(f"Layer {i} cycles = {expected_cycles}",
              layer.total_cycles == expected_cycles,
              f"got {layer.total_cycles} (tiles={expected_tiles}, in_dim={expected_in_dim})")

    # Total cycle check
    expected_total_compute = sum(
        EXPECTED_TILES[i] * (AD_LAYERS[i]["input_dim"] + 149)
        for i in range(4)
    )
    check(f"Total compute cycles = {expected_total_compute}",
          schedule.total_compute_cycles == expected_total_compute,
          f"got {schedule.total_compute_cycles}")

    check("Latency < 50 µs", schedule.latency_us < 50,
          f"{schedule.latency_us:.2f} µs")
    check("Throughput > 20K inf/sec",
          schedule.throughput_inferences_per_sec > 20000,
          f"{schedule.throughput_inferences_per_sec:,.0f} inf/sec")

    # Save schedule JSON
    schedule_to_json(schedule, output_dir / "schedule.json")
    check("schedule.json written", (output_dir / "schedule.json").exists())

    # Validate scheduled IR invariants
    try:
        graph.validate("scheduled")
        check("IR invariants pass (scheduled)", True)
    except Exception as e:
        check("IR invariants pass (scheduled)", False, str(e))

    print(f"  Time: {dt3:.3f}s")

    # ════════════════════════════════════════════════════════════════════
    # STAGE 4: Weight Packing
    # ════════════════════════════════════════════════════════════════════
    section("Stage 4: Weight Packing")
    t4 = time.perf_counter()

    weight_dir = output_dir / "weights"
    packer = WeightPacker(graph, weight_dir, constraints=hw)
    memory_map = packer.pack_weights()
    dt4 = time.perf_counter() - t4

    check("Weight packing succeeds", memory_map is not None)

    # Check output files exist
    for i in range(4):
        wf = weight_dir / f"weights_layer{i}.mem"
        bf = weight_dir / f"biases_layer{i}.mem"
        check(f"weights_layer{i}.mem exists", wf.exists())
        check(f"biases_layer{i}.mem exists", bf.exists())

    check("weight_bank.mem exists", (weight_dir / "weight_bank.mem").exists())
    check("bias_bank.mem exists", (weight_dir / "bias_bank.mem").exists())
    check("memory_map.json exists", (weight_dir / "memory_map.json").exists())

    # Validate memory map
    w_bytes = memory_map["weight_bank"]["total_bytes"]
    b_bytes = memory_map["bias_bank"]["total_bytes"]
    total_bytes = w_bytes + b_bytes

    check(f"Weight bank = {TOTAL_WEIGHT_BYTES} bytes",
          w_bytes == TOTAL_WEIGHT_BYTES,
          f"got {w_bytes}")
    check("Total fits KV260 SRAM (<200 KB)",
          total_bytes < 200 * 1024,
          f"{total_bytes:,} bytes = {total_bytes // 1024} KB")

    # Verify per-layer weight dimensions
    for i, layer_info in enumerate(memory_map["layers"]):
        expected = AD_LAYERS[i]
        check(f"Layer {i} weight in map: {expected['input_dim']}×{expected['output_dim']}",
              layer_info["input_dim"] == expected["input_dim"] and
              layer_info["output_dim"] == expected["output_dim"])

    # Round-trip: load .mem files and verify shapes
    for i in range(4):
        expected = AD_LAYERS[i]
        # load_weight_mem returns list[np.ndarray] (one INT8 row per SRAM line)
        w_rows = load_weight_mem(weight_dir / f"weights_layer{i}.mem")
        expected_rows = expected["input_dim"] * expected["output_dim"] // PARALLELISM
        check(f"Layer {i} weight row count = {expected_rows}",
              len(w_rows) == expected_rows,
              f"got {len(w_rows)}")
        if w_rows:
            check(f"Layer {i} weight row width = {PARALLELISM}",
                  len(w_rows[0]) == PARALLELISM,
                  f"got {len(w_rows[0])}")
            all_w = np.concatenate(w_rows)
            check(f"Layer {i} weights are INT8",
                  all_w.dtype == np.int8, f"got {all_w.dtype}")
            check(f"Layer {i} weights in [-127,127]",
                  np.all(all_w >= -127) and np.all(all_w <= 127))

        b_rows = load_bias_mem(weight_dir / f"biases_layer{i}.mem")
        if b_rows:
            all_b = np.concatenate(b_rows)
            check(f"Layer {i} biases are INT32",
                  all_b.dtype == np.int32, f"got {all_b.dtype}")

    # Verify .mem file format (hex lines, correct width)
    with open(weight_dir / "weight_bank.mem") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("//")]
    check(f"weight_bank.mem has {TOTAL_WEIGHT_ROWS} lines",
          len(lines) == TOTAL_WEIGHT_ROWS,
          f"got {len(lines)}")
    check("weight_bank.mem lines are 256 hex digits",
          all(len(l) == 256 for l in lines),
          f"first line len={len(lines[0]) if lines else 0}")
    check("weight_bank.mem lines are valid hex",
          all(re.match(r'^[0-9a-fA-F]+$', l) for l in lines))

    print(f"  Time: {dt4:.3f}s")

    # ════════════════════════════════════════════════════════════════════
    # STAGE 5: RTL Generation
    # ════════════════════════════════════════════════════════════════════
    section("Stage 5: RTL Generation")
    t5 = time.perf_counter()

    rtl_dir = output_dir / "rtl"
    rtl_gen = RTLGenerator(
        graph=graph,
        weight_dir=weight_dir,
        output_dir=rtl_dir,
        constraints=hw,
    )
    rtl_gen.generate_all()
    dt5 = time.perf_counter() - t5

    check("RTL generation succeeds", rtl_dir.exists())

    # Check all expected files
    expected_sv_files = [
        "accelerator_top.sv",
        "parameters.svh",
        "compute/mac_array.sv",
        "compute/requantize.sv",
        "compute/activation_relu.sv",
        "memory/sram_bank.sv",
        "memory/ping_pong_buffer.sv",
        "layer/fused_linear_relu.sv",
        "layer/bias_unpack.sv",
        "layer/byte_select.sv",
        "interface/axi_lite_ctrl.sv",
        "interface/axi_stream_in.sv",
        "interface/axi_stream_out.sv",
        "constraints/constraints.xdc",
    ]
    for f in expected_sv_files:
        check(f"RTL file: {f}", (rtl_dir / f).exists())

    # Check weight .mem files copied to RTL dir
    for i in range(4):
        check(f"RTL weights_layer{i}.mem",
              (rtl_dir / f"weights_layer{i}.mem").exists())
        check(f"RTL biases_layer{i}.mem",
              (rtl_dir / f"biases_layer{i}.mem").exists())
    check("RTL weight_bank.mem", (rtl_dir / "weight_bank.mem").exists())
    check("RTL bias_bank.mem", (rtl_dir / "bias_bank.mem").exists())

    # ── Validate parameters.svh content ──────────────────────────────
    params_content = (rtl_dir / "parameters.svh").read_text()

    check("parameters.svh defines NUM_LAYERS",
          "NUM_LAYERS" in params_content)
    check("parameters.svh NUM_LAYERS = 4",
          re.search(r'NUM_LAYERS\s*=\s*4', params_content) is not None)
    check("parameters.svh defines PARALLELISM = 128",
          re.search(r'PARALLELISM\s*=\s*128', params_content) is not None)

    # Check per-layer dimension arrays (actual names: LAYER_IN_DIM, LAYER_OUT_DIM, etc.)
    check("parameters.svh has LAYER_IN_DIM array", "LAYER_IN_DIM" in params_content)
    check("parameters.svh has LAYER_OUT_DIM array", "LAYER_OUT_DIM" in params_content)
    check("parameters.svh has LAYER_RELU array", "LAYER_RELU" in params_content)
    check("parameters.svh has REQUANT_M array", "REQUANT_M" in params_content)
    check("parameters.svh has REQUANT_SHIFT array", "REQUANT_SHIFT" in params_content)
    check("parameters.svh has REQUANT_ZP array", "REQUANT_ZP" in params_content)
    check("parameters.svh has WEIGHT_BASE array", "WEIGHT_BASE" in params_content)
    check("parameters.svh has BIAS_BASE array", "BIAS_BASE" in params_content)
    check("parameters.svh has LAYER_NUM_TILES array", "LAYER_NUM_TILES" in params_content)

    # Verify the dimension values appear in the file
    check("parameters.svh contains 640 (input/output dim)",
          "640" in params_content)
    check("parameters.svh contains 128 (hidden dim)",
          "128" in params_content)

    # ── Validate accelerator_top.sv content ──────────────────────────
    top_content = (rtl_dir / "accelerator_top.sv").read_text()

    check("accelerator_top.sv defines module",
          "module accelerator_top" in top_content)
    check("accelerator_top.sv includes parameters.svh",
          "parameters.svh" in top_content)
    check("accelerator_top.sv has clk port",
          "clk" in top_content)
    check("accelerator_top.sv has rst_n port",
          "rst_n" in top_content)
    check("accelerator_top.sv has AXI-Stream ports",
          "s_axis_tvalid" in top_content or "S_AXIS" in top_content.upper())
    check("accelerator_top.sv has AXI-Lite ports",
          "s_axi" in top_content.lower() or "axi_lite" in top_content.lower())
    check("accelerator_top.sv instantiates mac_array",
          "mac_array" in top_content)
    check("accelerator_top.sv instantiates fused_linear_relu",
          "fused_linear_relu" in top_content)
    check("accelerator_top.sv instantiates activation SRAM banks",
          "act_a" in top_content or "ping_pong_buffer" in top_content)
    check("accelerator_top.sv instantiates axi_lite_ctrl",
          "axi_lite_ctrl" in top_content)
    check("accelerator_top.sv instantiates axi_stream_in",
          "axi_stream_in" in top_content)
    check("accelerator_top.sv instantiates axi_stream_out",
          "axi_stream_out" in top_content)
    check("accelerator_top.sv has FSM states",
          "IDLE" in top_content and "DONE" in top_content)
    check("accelerator_top.sv has endmodule",
          "endmodule" in top_content)

    print(f"  Time: {dt5:.3f}s")

    # ════════════════════════════════════════════════════════════════════
    # STAGE 6: Verification (Golden Vectors + Testbenches)
    # ════════════════════════════════════════════════════════════════════
    section("Stage 6a: Golden Vector Generation")
    t6 = time.perf_counter()

    gvg = GoldenVectorGenerator(graph)

    # Random vectors
    random_vecs = gvg.generate_random_vectors(n=100, seed=42)
    check("100 random vectors generated", len(random_vecs.vectors) == 100)

    # Adversarial vectors
    adv_vecs = gvg.generate_adversarial_vectors()
    check("Adversarial vectors generated", len(adv_vecs.vectors) >= 10,
          f"got {len(adv_vecs.vectors)}")

    # Exact-half vectors (rounding boundary tests)
    half_vecs = gvg.generate_exact_half_vectors(n=10, seed=99)
    check("Exact-half vectors generated", len(half_vecs.vectors) >= 10,
          f"got {len(half_vecs.vectors)}")

    # Per-layer intermediates
    intermediate_vecs = gvg.generate_per_layer_intermediates(n=5, seed=123)
    check("Per-layer intermediates generated", len(intermediate_vecs.vectors) >= 5)
    if intermediate_vecs.vectors:
        v0 = intermediate_vecs.vectors[0]
        check("Intermediates have per-layer data",
              len(v0.intermediates) > 0,
              f"{len(v0.intermediates)} intermediate tensors")

    # Bitwise verification
    pass_count, total, mismatched = gvg.verify_vectors(random_vecs)
    check(f"Random vectors: {pass_count}/{total} bitwise match",
          pass_count == total,
          f"mismatched indices: {mismatched}" if mismatched else "")

    adv_pass, adv_total, adv_mm = gvg.verify_vectors(adv_vecs)
    check(f"Adversarial vectors: {adv_pass}/{adv_total} bitwise match",
          adv_pass == adv_total)

    half_pass, half_total, half_mm = gvg.verify_vectors(half_vecs)
    check(f"Exact-half vectors: {half_pass}/{half_total} bitwise match",
          half_pass == half_total)

    # Export
    vec_dir = output_dir / "vectors"
    mem_manifest = gvg.export_mem(random_vecs, vec_dir / "mem")
    npy_manifest = gvg.export_npy(random_vecs, vec_dir / "npy")

    check("test_inputs.mem exported", (vec_dir / "mem" / "test_inputs.mem").exists())
    check("golden_outputs.mem exported", (vec_dir / "mem" / "golden_outputs.mem").exists())
    check("test_manifest.json (mem) exported", (vec_dir / "mem" / "test_manifest.json").exists())
    check("NPY manifest exported", (vec_dir / "npy" / "test_manifest.json").exists())

    # Verify .mem format for test vectors
    input_mem = (vec_dir / "mem" / "test_inputs.mem").read_text()
    input_lines = [l.strip() for l in input_mem.splitlines()
                   if l.strip() and not l.startswith("//")]
    check("test_inputs.mem has 100 lines", len(input_lines) == 100,
          f"got {len(input_lines)}")

    # ── INT8 interpreter direct verification ─────────────────────────
    section("Stage 6: INT8 Interpreter Cross-check")

    interp = INT8Interpreter(graph)
    input_name = graph.inputs[0]
    output_name = graph.outputs[0]
    test_input = rng.randn(1, 640).astype(np.float32)
    result_dict = interp.run({input_name: test_input})
    check("INT8 interpreter runs", result_dict is not None)
    result = result_dict[output_name]
    check("INT8 output is INT8 dtype", result.dtype == np.int8,
          f"got {result.dtype}")
    check("INT8 output shape = (1, 640)", result.shape == (1, 640),
          f"got {result.shape}")
    check("INT8 output in [-128, 127]",
          np.all(result >= -128) and np.all(result <= 127))

    # ── Testbench generation ─────────────────────────────────────────
    section("Stage 6b: Testbench Generation")

    tb_dir = output_dir / "testbench"
    config = VerifConfig(
        graph=graph,
        output_dir=tb_dir,
        weight_dir=weight_dir,
        num_random_vectors=100,
        constraints=hw,
        is_mlp=True,
    )
    vg = VerifGenerator(config)
    generated_files = vg.generate_all()

    check("Testbench generation succeeds", len(generated_files) > 0,
          f"{len(generated_files)} files")

    # Expected module testbenches
    expected_tbs = [
        "tb_mac_array.sv",
        "tb_requantize.sv",
        "tb_activation_relu.sv",
        "tb_sram_bank.sv",
        "tb_ping_pong_buffer.sv",
        "tb_fused_linear_relu.sv",
        "tb_axi_stream_in.sv",
        "tb_axi_stream_out.sv",
        "tb_axi_lite_ctrl.sv",
        "tb_accelerator.sv",
    ]
    for tb_name in expected_tbs:
        check(f"Testbench: {tb_name}",
              tb_name in generated_files,
              f"present={'yes' if tb_name in generated_files else 'no'}")

    # cocotb
    check("cocotb test_accelerator.py generated",
          "cocotb/test_accelerator.py" in generated_files)
    check("cocotb Makefile generated",
          "cocotb/Makefile" in generated_files)

    # Verify system testbench content
    if "tb_accelerator.sv" in generated_files:
        tb_accel_path = generated_files["tb_accelerator.sv"]
        tb_accel = Path(tb_accel_path).read_text()
        check("tb_accelerator.sv references parameters or model constants",
              "parameters" in tb_accel.lower() or "NUM_LAYERS" in tb_accel
              or "localparam" in tb_accel)
        check("tb_accelerator.sv has golden comparison",
              "golden" in tb_accel.lower() or "expected" in tb_accel.lower())
        check("tb_accelerator.sv has AXI driver",
              "axi" in tb_accel.lower())

    # ── Functional coverage ──────────────────────────────────────────
    section("Stage 6c: Functional Coverage")

    cov = vg.coverage
    cov.mark_hit("csr_read_write")
    cov.mark_hit("relu_activation")
    cov.mark_hit("requant_boundary")
    cov.mark_hit("ping_pong_swap")
    cov.mark_hit("axi_backpressure")
    cov.mark_hit("irq_done")

    check("Coverage tracker has 11 items", cov.total == 11,
          f"got {cov.total}")
    check("6/11 coverage items hit", cov.hit_count == 6,
          f"got {cov.hit_count}")

    cov_path = output_dir / "coverage.json"
    with open(cov_path, "w") as f:
        json.dump(cov.to_dict(), f, indent=2)
    check("coverage.json written", cov_path.exists())

    dt6 = time.perf_counter() - t6
    print(f"  Time: {dt6:.3f}s")

    # ════════════════════════════════════════════════════════════════════
    # CROSS-CUTTING: End-to-end data integrity
    # ════════════════════════════════════════════════════════════════════
    section("Cross-cutting: End-to-end Data Integrity")

    # Verify round-trip: the quantized weights in .mem should match
    # what the INT8 interpreter uses
    for i, name in enumerate(topo):
        node = graph.nodes[name]
        w_tensor = None
        for inp in node.inputs:
            if inp in graph.tensors and graph.tensors[inp].data is not None:
                t = graph.tensors[inp]
                if t.data.ndim == 2:  # weight matrix
                    w_tensor = t
                    break

        if w_tensor is not None:
            expected = AD_LAYERS[i]
            w_rows = load_weight_mem(weight_dir / f"weights_layer{i}.mem")
            w_from_mem = np.concatenate(w_rows) if w_rows else np.array([], dtype=np.int8)
            w_graph = w_tensor.data.astype(np.int8)

            # Reconstruct the RTL SRAM packing order for comparison:
            # tile_idx × input_dim rows, each 128-wide slice of output dim
            num_tiles = EXPECTED_TILES[i]
            reconstructed = []
            for tile in range(num_tiles):
                col_start = tile * PARALLELISM
                col_end = col_start + PARALLELISM
                for row_idx in range(expected["input_dim"]):
                    reconstructed.append(w_graph[row_idx, col_start:col_end])
            w_expected = np.concatenate(reconstructed)

            check(f"Layer {i} weight .mem roundtrip matches graph (SRAM layout)",
                  np.array_equal(w_from_mem, w_expected),
                  f"max diff={np.max(np.abs(w_from_mem.astype(int) - w_expected.astype(int)))}"
                  if not np.array_equal(w_from_mem, w_expected) else "exact match")

    # Verify the schedule JSON matches in-memory schedule
    with open(output_dir / "schedule.json") as f:
        sched_json = json.load(f)
    check("Schedule JSON layers count = 4",
          len(sched_json["layers"]) == 4)
    check("Schedule JSON total_cycles matches",
          sched_json["total_cycles"] == schedule.total_cycles)

    # ════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════════════
    total_time = time.perf_counter() - t0

    print(f"\n{'='*70}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total checks: {_pass_count + _fail_count}")
    print(f"  \033[32mPASSED: {_pass_count}\033[0m")
    if _fail_count:
        print(f"  \033[31mFAILED: {_fail_count}\033[0m")
        print(f"\n  Failed checks:")
        for name, passed, detail in _checks:
            if not passed:
                print(f"    - {name}: {detail}")
    else:
        print(f"  FAILED: 0")
    print(f"\n  Pipeline timing:")
    print(f"    Stage 1 (Ingestion):     {dt1:.3f}s")
    print(f"    Stage 2 (Optimization):  {dt2:.3f}s")
    print(f"    Stage 3 (Scheduling):    {dt3:.3f}s")
    print(f"    Stage 4 (Weight Pack):   {dt4:.3f}s")
    print(f"    Stage 5 (RTL Gen):       {dt5:.3f}s")
    print(f"    Stage 6 (Verification):  {dt6:.3f}s")
    print(f"    Total:                   {total_time:.3f}s")

    # Output artifact summary
    sv_count = len(list(rtl_dir.rglob("*.sv")))
    mem_count = len(list(rtl_dir.rglob("*.mem")))
    tb_count = len([f for f in generated_files if f.endswith(".sv")])
    print(f"\n  Generated artifacts:")
    print(f"    RTL modules:     {sv_count} .sv files")
    print(f"    Weight files:    {mem_count} .mem files")
    print(f"    Testbenches:     {tb_count} .sv + cocotb")
    print(f"    Golden vectors:  {len(random_vecs.vectors)} random + "
          f"{len(adv_vecs.vectors)} adversarial + {len(half_vecs.vectors)} exact-half")
    print(f"    Output dir:      {output_dir}")
    print(f"{'='*70}")

    sys.exit(1 if _fail_count > 0 else 0)


if __name__ == "__main__":
    main()
