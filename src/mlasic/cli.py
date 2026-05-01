"""Unified MLASIC compiler CLI.

Usage:
    mlasic compile models/mobilenetv2.onnx -o output/mobilenetv2/
    python -m mlasic compile models/ad_model.onnx --stages 1-3 -v
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from mlasic.dag_scheduler import DAGScheduler
from mlasic.golden_vectors import GoldenVectorGenerator
from mlasic.ingestion import ONNXParser
from mlasic.ir import HardwareConstraints, OpType
from mlasic.optimization import (
    ActivationFusionPass,
    AttentionFusionPass,
    AttentionQuantizationPass,
    BatchNormFoldingPass,
    ConstantFoldingPass,
    ConvBatchNormFoldingPass,
    ConvFusionPass,
    ConvQuantizationPass,
    DeadCodeEliminationPass,
    FusedMLPPass,
    LayerNormFusionPass,
    OperatorFusionPass,
    PassManager,
    QuantizationPass,
)
from mlasic.rom_mapper import WeightROMMapper
from mlasic.rtl_gen import RTLGenerator
from mlasic.scheduler import Scheduler, schedule_to_json
from mlasic.testbench_gen import VerifConfig, VerifGenerator
from mlasic.tile_mapper import TileMapper
from mlasic.weight_packer import WeightPacker


def detect_architecture(graph) -> str:
    """Detect model architecture from fused graph node types.

    Returns "mlp", "cnn", or "transformer".
    """
    op_types = {node.op_type for node in graph.nodes.values()}
    mlp_ops = {OpType.FUSED_LINEAR, OpType.FUSED_LINEAR_RELU}

    if op_types and op_types <= mlp_ops:
        return "mlp"

    transformer_ops = {OpType.FUSED_ATTENTION, OpType.FUSED_LAYER_NORM}
    if op_types & transformer_ops:
        return "transformer"

    conv_ops = {OpType.FUSED_CONV, OpType.FUSED_CONV_RELU, OpType.FUSED_CONV_RELU6}
    if op_types & conv_ops:
        return "cnn"

    # Safe default: DAG path handles anything
    return "cnn"


def build_structural_pass_manager() -> PassManager:
    """Build pass manager for structural passes (phase 1).

    These are safe to apply to any architecture. After running these,
    call detect_architecture() on the result to determine the model type.
    """
    pm = PassManager()
    pm.add_pass(ConstantFoldingPass())
    pm.add_pass(DeadCodeEliminationPass())
    pm.add_pass(BatchNormFoldingPass())
    pm.add_pass(ConvBatchNormFoldingPass())
    pm.add_pass(DeadCodeEliminationPass())
    pm.add_pass(ConvFusionPass())
    pm.add_pass(OperatorFusionPass())
    return pm


def build_quant_pass_manager(
    arch: str, calibration_data: list[np.ndarray]
) -> PassManager:
    """Build architecture-specific quantization passes (phase 2).

    Args:
        arch: One of "mlp", "cnn", "transformer".
        calibration_data: Calibration input arrays.
    """
    pm = PassManager()

    if arch == "transformer":
        pm.add_pass(LayerNormFusionPass())
        pm.add_pass(ActivationFusionPass())
        pm.add_pass(AttentionFusionPass())
        pm.add_pass(FusedMLPPass())
        pm.add_pass(ConvQuantizationPass(calibration_data=calibration_data))
        pm.add_pass(AttentionQuantizationPass(calibration_data=calibration_data))
    elif arch == "cnn":
        pm.add_pass(ConvQuantizationPass(calibration_data=calibration_data))

    pm.add_pass(QuantizationPass(calibration_data=calibration_data))
    return pm


def _parse_stages(stages_str: str) -> tuple[int, int]:
    """Parse stage range string like '1-6', '1-3', or '5' into (start, end)."""
    parts = stages_str.strip().split("-")
    if len(parts) == 1:
        n = int(parts[0])
        return (n, n)
    if len(parts) == 2:
        return (int(parts[0]), int(parts[1]))
    raise argparse.ArgumentTypeError(
        f"Invalid stage range: {stages_str!r} (expected e.g. '1-6', '1-3', '5')"
    )


def _generate_calibration_data(
    graph, num_samples: int, seed: int = 42
) -> list[np.ndarray]:
    """Generate random calibration inputs matching the model's input shape."""
    rng = np.random.RandomState(seed)
    # Determine input shape from the graph's first input tensor
    input_names = list(graph.inputs)
    if not input_names:
        raise ValueError("Graph has no input tensors")
    input_name = input_names[0]
    input_tensor = graph.tensors[input_name]
    input_shape = tuple(input_tensor.type.shape)
    return [rng.randn(*input_shape).astype(np.float32) for _ in range(num_samples)]


def _print_header(stage: int, name: str, sub: str = "") -> None:
    label = f"Stage {stage}{sub}: {name}"
    print(f"\n{'─' * 70}")
    print(f"  {label}")
    print(f"{'─' * 70}")


def run_pipeline(args: argparse.Namespace) -> None:
    """Orchestrate stages 1-6 based on CLI args."""
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    model_name = model_path.stem
    output_dir = Path(args.output) if args.output else Path("output") / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_start, stage_end = _parse_stages(args.stages)
    verbose = args.verbose

    hw = HardwareConstraints(
        clock_mhz=args.clock,
        target_device=args.device,
    )

    print("=" * 70)
    print("  MLASIC Compiler")
    print("=" * 70)
    print(f"  Model:   {model_path}")
    print(f"  Output:  {output_dir}")
    print(f"  Target:  {args.target}")
    print(f"  Device:  {args.device}")
    print(f"  Clock:   {args.clock} MHz")
    print(f"  Stages:  {args.stages}")

    t_total = time.perf_counter()
    stage_times: dict[str, float] = {}

    graph = None
    arch = None
    schedule = None
    dag_schedule = None
    fabric = None
    weight_dir = None
    is_mlp = False

    # ── Stage 1: ONNX Ingestion ──────────────────────────────────────
    if stage_start <= 1 <= stage_end:
        _print_header(1, "ONNX Ingestion")
        t = time.perf_counter()
        graph = ONNXParser(model_path).parse()
        graph.name = model_name
        dt = time.perf_counter() - t
        stage_times["1"] = dt

        print(f"  Nodes:   {len(graph.nodes)}")
        print(f"  Tensors: {len(graph.tensors)}")
        if verbose:
            op_counts = Counter(n.op_type.name for n in graph.nodes.values())
            for op, count in op_counts.most_common():
                print(f"    {op}: {count}")
        print(f"  Time: {dt:.3f}s")

    # ── Stage 2: Graph Optimization ──────────────────────────────────
    if stage_start <= 2 <= stage_end:
        if graph is None:
            print("Error: Stage 2 requires Stage 1 output (graph)", file=sys.stderr)
            sys.exit(1)

        _print_header(2, "Graph Optimization")
        t = time.perf_counter()

        # Calibration data
        if args.calibration_data:
            cal_path = Path(args.calibration_data)
            cal_arr = np.load(cal_path)
            cal_data = [cal_arr[i] for i in range(len(cal_arr))]
            print(f"  Calibration: {len(cal_data)} samples from {cal_path}")
        else:
            cal_data = _generate_calibration_data(graph, args.num_cal_samples)
            print(f"  Calibration: {len(cal_data)} random samples")

        # Phase 1: structural passes (safe for all architectures)
        structural_pm = build_structural_pass_manager()
        graph = structural_pm.run(graph, verify=False)

        # Detect architecture from fused graph
        arch = detect_architecture(graph)
        is_mlp = arch == "mlp"

        # Phase 2: architecture-specific quantization passes
        quant_pm = build_quant_pass_manager(arch, cal_data)
        graph = quant_pm.run(graph, verify=False)
        dt = time.perf_counter() - t
        stage_times["2"] = dt
        print(f"  Architecture: {arch}")
        print(f"  Stage: {graph.stage}")
        print(f"  Nodes: {len(graph.nodes)}")
        if verbose:
            for name in graph.topological_order():
                node = graph.nodes[name]
                attrs = node.fused_attrs
                if attrs is not None:
                    relu = " +ReLU" if getattr(attrs, "has_relu", False) else ""
                    in_d = getattr(attrs, "input_dim", "?")
                    out_d = getattr(attrs, "output_dim", "?")
                    print(f"    {name}: {node.op_type.name} ({in_d}->{out_d}){relu}")
                else:
                    print(f"    {name}: {node.op_type.name}")
        print(f"  Time: {dt:.3f}s")

    # ── Stage 3: Scheduling ──────────────────────────────────────────
    if stage_start <= 3 <= stage_end:
        if graph is None:
            print("Error: Stage 3 requires Stage 2 output (optimized graph)", file=sys.stderr)
            sys.exit(1)

        if arch is None:
            arch = detect_architecture(graph)
            is_mlp = arch == "mlp"

        if is_mlp:
            _print_header(3, "Dataflow Scheduling (MLP)")
            t = time.perf_counter()
            schedule = Scheduler(constraints=hw).schedule(graph)
            dt = time.perf_counter() - t
            stage_times["3"] = dt

            print(f"  Total compute cycles: {schedule.total_compute_cycles:,}")
            print(f"  Total cycles (w/ AXI): {schedule.total_cycles:,}")
            print(f"  Latency: {schedule.latency_us:.2f} us @ {hw.clock_mhz} MHz")
            print(f"  Throughput: {schedule.throughput_inferences_per_sec:,.0f} inf/sec")
            schedule_to_json(schedule, output_dir / "schedule.json")
            print(f"  Time: {dt:.3f}s")
        else:
            _print_header(3, "DAG Scheduling")
            t = time.perf_counter()
            dag_schedule = DAGScheduler(constraints=hw).schedule(graph)
            dt3 = time.perf_counter() - t

            print(f"  Total cycles: {dag_schedule.total_cycles:,}")
            print(f"  Peak activation SRAM: {dag_schedule.peak_activation_bytes:,} bytes")
            print(f"  Nodes scheduled: {len(dag_schedule.node_schedules)}")
            if verbose:
                for ns in dag_schedule.node_schedules:
                    print(f"    {ns.node_name}: {ns.total_cycles:,} cycles "
                          f"(start={ns.start_cycle:,})")
            print(f"  Time: {dt3:.3f}s")

            # Stage 3b: Tile Mapping
            _print_header(3, "Tile Mapping", sub="b")
            t = time.perf_counter()
            fabric = TileMapper().map(graph, dag_schedule)
            dt3b = time.perf_counter() - t

            print(f"  Total tiles: {fabric.total_tiles:,}")
            if verbose:
                tile_types: dict[str, int] = {}
                for tile in fabric.tiles:
                    ttype = tile.tile_type.name
                    tile_types[ttype] = tile_types.get(ttype, 0) + 1
                for ttype, count in sorted(tile_types.items()):
                    print(f"    {ttype}: {count} tiles")
            print(f"  Routes: {len(fabric.routes):,}")
            print(f"  Time: {dt3b:.3f}s")

            stage_times["3"] = dt3 + dt3b

    # ── Stage 4: Weight Packing ──────────────────────────────────────
    if stage_start <= 4 <= stage_end:
        if graph is None:
            print("Error: Stage 4 requires previous stage output", file=sys.stderr)
            sys.exit(1)

        if arch is None:
            arch = detect_architecture(graph)
            is_mlp = arch == "mlp"

        weight_dir = output_dir / "weights"
        weight_dir.mkdir(parents=True, exist_ok=True)

        if is_mlp:
            _print_header(4, "Weight Packing (MLP)")
            t = time.perf_counter()
            memory_map = WeightPacker(graph, weight_dir, constraints=hw).pack_weights()
            dt = time.perf_counter() - t
            stage_times["4"] = dt

            w_bytes = memory_map["weight_bank"]["total_bytes"]
            b_bytes = memory_map["bias_bank"]["total_bytes"]
            print(f"  Weight bank: {w_bytes:,} bytes ({w_bytes // 1024} KB)")
            print(f"  Bias bank: {b_bytes:,} bytes")
            print(f"  Total: {w_bytes + b_bytes:,} bytes ({(w_bytes + b_bytes) // 1024} KB)")
            if verbose:
                for layer in memory_map["layers"]:
                    print(f"    Layer {layer['layer_index']}: "
                          f"{layer['input_dim']}->{layer['output_dim']} "
                          f"({layer['weight']['size_bytes']:,}W + "
                          f"{layer['bias']['size_bytes']:,}B bytes)")
            print(f"  Time: {dt:.3f}s")
        else:
            _print_header(4, "Weight ROM Mapping")
            t = time.perf_counter()
            if fabric is None:
                print("Error: Stage 4 (non-MLP) requires tile fabric from Stage 3",
                      file=sys.stderr)
                sys.exit(1)
            manifest = WeightROMMapper().generate(
                graph, fabric, output_dir=weight_dir
            )
            dt = time.perf_counter() - t
            stage_times["4"] = dt

            print(f"  Total weight bytes: {manifest['total_weight_bytes']:,}")
            tiles_with_weights = [
                t for t in manifest["tiles"] if t["weight_bytes"] > 0
            ]
            print(f"  Tiles with weights: {len(tiles_with_weights)}")
            if verbose:
                for tile_info in tiles_with_weights[:20]:
                    print(f"    Tile {tile_info['tile_id']} ({tile_info['node_name']}): "
                          f"{tile_info['weight_bytes']:,} bytes")
                if len(tiles_with_weights) > 20:
                    print(f"    ... ({len(tiles_with_weights) - 20} more)")
            print(f"  Time: {dt:.3f}s")

    # ── Stage 5: RTL Generation ──────────────────────────────────────
    if stage_start <= 5 <= stage_end:
        if graph is None:
            print("Error: Stage 5 requires previous stage output", file=sys.stderr)
            sys.exit(1)
        if weight_dir is None:
            weight_dir = output_dir / "weights"
            if not weight_dir.exists():
                print("Error: Stage 5 requires weight files from Stage 4", file=sys.stderr)
                sys.exit(1)

        _print_header(5, "RTL Generation")
        t = time.perf_counter()
        rtl_dir = output_dir / "rtl"
        rtl_gen = RTLGenerator(
            graph=graph,
            weight_dir=weight_dir,
            output_dir=rtl_dir,
            constraints=hw,
            fabric_config=fabric,
            target=args.target,
        )
        rtl_gen.generate_all()
        dt = time.perf_counter() - t
        stage_times["5"] = dt

        sv_files = sorted(rtl_dir.rglob("*.sv"))
        mem_files = sorted(rtl_dir.rglob("*.mem"))
        print(f"  SV modules: {len(sv_files)}")
        if verbose:
            for f in sv_files:
                print(f"    {f.relative_to(rtl_dir)}")
        print(f"  Memory files: {len(mem_files)}")
        print(f"  Time: {dt:.3f}s")

    # ── Stage 6: Verification ────────────────────────────────────────
    if stage_start <= 6 <= stage_end:
        if graph is None:
            print("Error: Stage 6 requires previous stage output", file=sys.stderr)
            sys.exit(1)

        _print_header(6, "Testbench & Verification")
        t = time.perf_counter()

        # 6.1: Golden vector generation
        print("  [6.1] Golden Vector Generation...")
        if is_mlp:
            gvg = GoldenVectorGenerator(graph)
            num_vecs = args.num_test_vectors

            random_vecs = gvg.generate_random_vectors(n=num_vecs, seed=42)
            adv_vecs = gvg.generate_adversarial_vectors()
            half_vecs = gvg.generate_exact_half_vectors(n=10, seed=99)

            pass_count, total, _ = gvg.verify_vectors(random_vecs)
            print(f"    Random vectors: {pass_count}/{total} pass bitwise")
            adv_pass, adv_total, _ = gvg.verify_vectors(adv_vecs)
            print(f"    Adversarial vectors: {adv_pass}/{adv_total} pass bitwise")
            half_pass, half_total, _ = gvg.verify_vectors(half_vecs)
            print(f"    Exact-half vectors: {half_pass}/{half_total} pass bitwise")

            vec_dir = output_dir / "vectors"
            gvg.export_mem(random_vecs, vec_dir / "mem")
            gvg.export_npy(random_vecs, vec_dir / "npy")
            print(f"    Exported to {vec_dir}")
        else:
            print("    Skipped (non-MLP model -- golden vectors not generated)")

        # 6.2: Testbench generation
        print("  [6.2] Testbench Generation...")
        if weight_dir is None:
            weight_dir = output_dir / "weights"
        tb_dir = output_dir / "testbench"
        config = VerifConfig(
            graph=graph,
            output_dir=tb_dir,
            weight_dir=weight_dir if weight_dir.exists() else None,
            num_random_vectors=args.num_test_vectors,
            constraints=hw,
            is_mlp=is_mlp,
        )
        vg = VerifGenerator(config)
        generated_files = vg.generate_all()

        sv_tb = [f for f in generated_files if f.endswith(".sv")]
        py_tb = [f for f in generated_files if f.endswith(".py")]
        print(f"    SV testbenches: {len(sv_tb)}")
        if verbose:
            for f in sorted(sv_tb):
                print(f"      {f}")
        print(f"    cocotb tests: {len(py_tb)}")

        # 6.3: Functional coverage
        print("  [6.3] Functional Coverage...")
        cov = vg.coverage
        cov.mark_hit("csr_read_write")
        cov.mark_hit("relu_activation")
        cov.mark_hit("requant_boundary")
        cov.mark_hit("ping_pong_swap")
        cov.mark_hit("axi_backpressure")
        cov.mark_hit("irq_done")
        print(f"    Coverage: {cov.hit_count}/{cov.total} ({cov.coverage_pct:.0f}%)")

        cov_path = output_dir / "coverage.json"
        with open(cov_path, "w") as f:
            json.dump(cov.to_dict(), f, indent=2)

        dt = time.perf_counter() - t
        stage_times["6"] = dt
        print(f"  Time: {dt:.3f}s")

    # ── Post-compile verification ────────────────────────────────────
    if stage_end >= 6:
        _run_post_compile_checks(output_dir, verbose)

    # ── Summary ──────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_total
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE SUMMARY: {model_name}")
    print(f"{'=' * 70}")
    stage_labels = {
        "1": "Ingestion", "2": "Optimization", "3": "Scheduling",
        "4": "Weight Pack", "5": "RTL Gen", "6": "Verification",
    }
    for key in sorted(stage_times):
        label = stage_labels.get(key, key)
        print(f"  Stage {key} ({label}): {stage_times[key]:.3f}s")
    print(f"  {'─' * 35}")
    print(f"  Total:                   {total_time:.3f}s")

    # File counts
    rtl_dir = output_dir / "rtl"
    if rtl_dir.exists():
        sv_count = len(list(rtl_dir.rglob("*.sv")))
        mem_count = len(list(rtl_dir.rglob("*.mem")))
        print(f"\n  RTL: {sv_count} SV + {mem_count} MEM files")
    tb_dir = output_dir / "testbench"
    if tb_dir.exists():
        tb_sv = len(list(tb_dir.rglob("*.sv")))
        tb_py = len(list(tb_dir.rglob("*.py")))
        print(f"  Testbenches: {tb_sv} SV + {tb_py} cocotb")

    print(f"  Output directory: {output_dir}")
    print(f"{'=' * 70}")
    print("  DONE")
    print(f"{'=' * 70}")


def _run_post_compile_checks(output_dir: Path, verbose: bool) -> None:
    """Run automated post-compile verification checks."""
    print(f"\n{'─' * 70}")
    print("  Post-Compile Verification")
    print(f"{'─' * 70}")

    all_passed = True
    check_num = 0
    total_checks = 0

    # Count how many checks will run
    rtl_dir = output_dir / "rtl"
    if rtl_dir.exists() and list(rtl_dir.rglob("*.sv")):
        total_checks += 1
    tb_dir = output_dir / "testbench" / "cocotb"
    if tb_dir.exists():
        total_checks += 1

    if total_checks == 0:
        print("  No checks to run.")
        return

    # Check: Verilator lint on generated RTL
    if rtl_dir.exists():
        sv_files = sorted(rtl_dir.rglob("*.sv"))
        if sv_files:
            check_num += 1
            print(f"  [{check_num}/{total_checks}] Verilator lint...",
                  end=" ", flush=True)
            # Find top-level SV file
            top_files = [f for f in sv_files if "accelerator" in f.name]
            lint_targets = top_files if top_files else sv_files[:1]

            # Collect all directories containing .sv or .svh for include paths
            include_dirs: set[str] = set()
            include_dirs.add(str(rtl_dir))
            for f in sv_files:
                include_dirs.add(str(f.parent))
            for f in rtl_dir.rglob("*.svh"):
                include_dirs.add(str(f.parent))

            cmd = [
                "verilator", "--lint-only", "--language", "1800-2017",
                "-Wall", "-Wno-fatal",
            ]
            for d in sorted(include_dirs):
                cmd.append(f"-I{d}")
            cmd.extend(str(f) for f in lint_targets)

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=60,
                )
                # Count warnings and errors separately
                errors = [line for line in result.stderr.splitlines()
                          if line.startswith("%Error")]
                warnings = [line for line in result.stderr.splitlines()
                            if line.startswith("%Warning")]

                if result.returncode == 0 and not errors:
                    if warnings:
                        print(f"PASS ({len(warnings)} warnings)")
                    else:
                        print("PASS")
                    if verbose and warnings:
                        for w in warnings[:10]:
                            print(f"    {w}")
                        if len(warnings) > 10:
                            print(f"    ... ({len(warnings) - 10} more)")
                else:
                    print(f"FAIL ({len(errors)} errors)")
                    all_passed = False
                    if verbose:
                        stderr = result.stderr
                        print(stderr[-2000:] if len(stderr) > 2000 else stderr)
            except FileNotFoundError:
                print("SKIP (verilator not found)")

    # Check: Run cocotb tests if Makefile exists and cocotb is installed
    cocotb_makefile = tb_dir / "Makefile"
    if cocotb_makefile.exists():
        check_num += 1
        print(f"  [{check_num}/{total_checks}] cocotb testbench...",
              end=" ", flush=True)
        # Check cocotb is installed before trying
        try:
            cocotb_check = subprocess.run(
                ["cocotb-config", "--version"],
                capture_output=True, text=True,
            )
        except FileNotFoundError:
            cocotb_check = None
        if cocotb_check is None or cocotb_check.returncode != 0:
            print("SKIP (cocotb not installed)")
        else:
            try:
                result = subprocess.run(
                    ["make", "-C", str(tb_dir), "SIM=verilator"],
                    capture_output=True, text=True, timeout=300,
                )
                if result.returncode == 0:
                    print("PASS")
                else:
                    print("FAIL")
                    all_passed = False
                    if verbose:
                        combined = result.stdout + result.stderr
                        print(combined[-2000:]
                              if len(combined) > 2000 else combined)
            except FileNotFoundError:
                print("SKIP (make not found)")

    if all_passed:
        print("  All checks passed.")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="mlasic",
        description="MLASIC: ML model to ASIC/FPGA compiler",
    )
    subparsers = parser.add_subparsers(dest="command")

    compile_parser = subparsers.add_parser(
        "compile",
        help="Compile an ONNX model to synthesizable RTL",
    )
    compile_parser.add_argument(
        "model",
        help="Path to ONNX model file",
    )
    compile_parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory (default: output/<model_name>/)",
    )
    compile_parser.add_argument(
        "--target",
        choices=["fpga", "asic"],
        default="fpga",
        help="Target platform (default: fpga)",
    )
    compile_parser.add_argument(
        "--device",
        default="xck26-sfvc784-2LV",
        help="FPGA part string (default: xck26-sfvc784-2LV)",
    )
    compile_parser.add_argument(
        "--clock",
        type=int,
        default=100,
        help="Clock frequency in MHz (default: 100)",
    )
    compile_parser.add_argument(
        "--weight-bits",
        type=int,
        choices=[4, 8],
        default=8,
        help="Weight quantization bits (default: 8)",
    )
    compile_parser.add_argument(
        "--calibration-data",
        default=None,
        help="Path to .npy file with calibration inputs (default: generate random)",
    )
    compile_parser.add_argument(
        "--stages",
        default="1-6",
        help="Which stages to run, e.g. '1-6', '1-3', '5' (default: 1-6)",
    )
    compile_parser.add_argument(
        "--num-cal-samples",
        type=int,
        default=10,
        help="Number of random calibration samples if no file given (default: 10)",
    )
    compile_parser.add_argument(
        "--num-test-vectors",
        type=int,
        default=100,
        help="Number of golden test vectors for Stage 6 (default: 100)",
    )
    compile_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Detailed per-stage logging",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "compile":
        run_pipeline(args)
