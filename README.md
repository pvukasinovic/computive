# Computive

> https://github.com/pvukasinovic/computive

Compile a frozen ONNX model into a model-specific, synthesizable INT8
accelerator: SystemVerilog RTL, weight images, golden vectors, and
testbenches. Target is Xilinx Zynq UltraScale+ (Kria KV260) for the v0.1
flow; an ASIC ROM target is also supported.

Computive is a **real compiler**, not a model-specific code generator:

- **Frontend** parses ONNX into a typed IR (`computive.ingestion`,
  `computive.ir`).
- **Mid-end** runs analysis and transformation passes — constant
  folding, DCE, BatchNorm fold, operator fusion, calibration,
  quantization (`computive.optimization`).
- **Backend** lowers the IR into an RTL IR (`RTLModule`, `RTLInstance`,
  `RTLContinuousAssign`, …) and a dumb emitter walks that IR to print
  SystemVerilog (`computive.rtl_ir`, `computive.rtl_lowering*`,
  `computive.rtl_emit`). No templated `.sv` blob with `{var}` slots —
  instances are built from graph attributes.
- **Placement** is liveness-driven: activation banks are assigned by an
  analysis pass over the dataflow graph, and the DAG schedule reorders
  ready nodes to minimise peak SRAM (`BankAllocator` in
  `computive.scheduler`, `peak_aware_topological_order` in
  `computive.dag_scheduler`).

> The Python package on disk is still imported as `mlasic` for now — the
> codebase is being renamed incrementally.

---

## Quick start

```bash
git clone https://github.com/pvukasinovic/computive.git
cd computive
python3 -m venv .venv && source .venv/bin/activate
make install      # editable install + dev deps (onnx, onnxruntime, pytest, ruff)
make test         # run the test suite
```

Compile a model:

```bash
python -m mlasic compile path/to/model.onnx -o output/my_model
```

Programmatic equivalent (handy when scripting calibration or
custom passes):

```python
from pathlib import Path
import numpy as np
from mlasic.ingestion import ONNXParser
from mlasic.optimization import (
    PassManager, ConstantFoldingPass, DeadCodeEliminationPass,
    BatchNormFoldingPass, OperatorFusionPass, QuantizationPass,
)
from mlasic.scheduler import Scheduler
from mlasic.weight_packer import WeightPacker
from mlasic.rtl_gen import RTLGenerator
from mlasic.ir import HardwareConstraints

graph = ONNXParser(Path("model.onnx")).parse()

calib = [np.random.randn(1, 640).astype(np.float32) for _ in range(20)]
pm = PassManager()
for p in [ConstantFoldingPass(), DeadCodeEliminationPass(),
          BatchNormFoldingPass(), OperatorFusionPass(),
          QuantizationPass(calibration_data=calib)]:
    pm.add_pass(p)
graph = pm.run(graph, verify=True)

hw = HardwareConstraints()
schedule = Scheduler(constraints=hw).schedule(graph)

WeightPacker(graph, Path("out/weights"), constraints=hw).pack_weights()
RTLGenerator(graph=graph, weight_dir=Path("out/weights"),
             output_dir=Path("out/rtl"), constraints=hw).generate_all()
```

CNN/Transformer flow (`DAGScheduler` + `TileMapper` instead of the
linear `Scheduler`) is in `validate_compiler.py` and the demo scripts.

---

## Repo layout

```
src/mlasic/                         (package directory; will be renamed to computive)
  ingestion.py          Stage 1 — ONNX → IR (parser, shape inference)
  ir.py                 IR types: Graph, OpNode, Tensor, OpType, QuantParams,
                        HardwareConstraints, LayerSchedule, …
  optimization.py       Stage 2 — every analysis/transform pass + PassManager
  interpreter.py        FP32 reference interpreter (correctness oracle)
  int8_interpreter.py   Bitwise-faithful INT8 interpreter (golden oracle)
  scheduler.py          Stage 3 — linear MLP schedule + BankAllocator
  dag_scheduler.py      Stage 3 — DAG schedule, lifetime analysis,
                        peak-aware reorder, weight streaming
  tile_mapper.py        Stage 3b — operator → tile placement (CNN/ASIC)
  rom_mapper.py         Stage 4 — per-tile weight ROM extraction
  weight_packer.py      Stage 4 — INT8 weight/bias .mem packing
  rtl_ir.py             RTL IR (RTLModule/Instance/Signal/RawBlock)
  rtl_emit.py           Dumb structural emitter (zero logic)
  rtl_lowering.py       Linear-MLP IR → RTL IR lowering
  rtl_lowering_tile.py  Tile-fabric IR → RTL IR lowering
  rtl_gen.py            Stage 5 — orchestration, model-path selection
  golden_vectors.py     Stage 6 — generates input/output vectors
  testbench_gen.py      Stage 6 — SV + cocotb testbench generation
  cli.py                CLI entry point

rtl/                    Parameterized SystemVerilog cell library
  compute/  memory/  layer/  interface/  tile/  top/  constraints/

tests/                  ~520 unit + integration tests
docs/                   PRD, IR spec, quantization spec, RTL spec, firmware guide
```

A node added in `optimization.py` should not need changes anywhere
else in the compiler — every pass walks `graph.topological_order()` and
dispatches on `node.op_type`. Same goes for adding an op: register it
in `ir.OpType`, give it an entry in any pass that should see it, and
add a cycle model in `dag_scheduler.CYCLE_MODEL_REGISTRY`.

---

## Pipeline at a glance

```
ONNX ─► Ingestion ─► Optimization ─► Scheduling ─► Weight pack ─► RTL gen ─► Verification
        Stage 1      Stage 2          Stage 3       Stage 4         Stage 5    Stage 6
        (parse,      (fold, fuse,    (cycle budget, (INT8 .mem      (RTL IR    (golden
         shape       quantize)        bank assign,   files,          → SV       vectors,
         infer)                       tile map)      memory map)     emit)      testbenches)
```

Each stage attaches its results to the IR (`node.fused_attrs`,
`node.schedule_info`, `node.dag_schedule`) and bumps `graph.stage` so
downstream stages can validate their preconditions.

---

## Compiler architecture (the “real compiler” bits)

### Mid-end passes

`computive.optimization` is a sequence of passes registered with
`PassManager`. Every pass iterates `graph.topological_order()` and
dispatches by `node.op_type`; nothing assumes a specific layer count or
shape. Examples:

| Pass | What it does |
|---|---|
| `ConstantFoldingPass` | Fixed-point folding over 17+ ops (MatMul, Reshape, Slice, Concat, …) |
| `DeadCodeEliminationPass` | Backward reachability from `graph.outputs` |
| `BatchNormFoldingPass` | Pattern-match MatMul→Add→BN, fold params; same generically for Conv→BN |
| `OperatorFusionPass` | MatMul→Add[→ReLU] → `FusedLinear[ReLU]` via single-consumer check |
| `QuantizationPass` | Calibration over node activations; per-channel weight scales |

Adding a new pass: subclass `Pass`, implement `run(graph) -> Graph`,
register with `PassManager`. Run the `validate_ir` skill (or
`computive.ir.verify_invariants`) after each pass during development.

### RTL backend

`computive.rtl_ir` defines a structural IR for SystemVerilog:
`RTLModule`, `RTLInstance`, `RTLSignal`, `RTLContinuousAssign`,
`RTLRawBlock`. The emitter (`rtl_emit.py`) is deliberately dumb — it
only formats. All lowering decisions live in `rtl_lowering*.py`, which
walks the compute graph, extracts attributes, and builds `RTLInstance`s
dynamically.

Behavioural blobs (top-level FSM, AXI handshake bodies) are wrapped in
`RTLRawBlock` with explicit `defines`/`uses` so liveness analysis still
sees them — analogous to LLVM inline-asm. They are the boundary, not
the rule.

To add a new RTL primitive:

1. Drop the parameterized `.sv` cell into `rtl/<group>/`.
2. In the lowering pass, build an `RTLInstance(module_type="my_cell", …)`
   from the IR node’s attributes.
3. Reference its ports/params in surrounding `add_signal` /
   `add_continuous_assign` calls.

Nothing about the new cell touches the emitter or other modules.

### RTL formatting conventions

- **Inputs use the implicit wire type** — `input clk`, not
  `input logic clk`. Outputs/inouts keep `logic` because the body
  drives them procedurally.
- **Single-space port decls** — no column alignment with multiple
  spaces.
- **Sized fill literals** — `{ACC_W{1'b0}}` or `7'b0`, never `'0`. The
  width is always explicit so the synthesiser cannot silently extend
  or truncate.

The emitter follows these rules in code; running the compiler is the
only sanctioned way to regenerate `accelerator_top.sv` and friends.

### Placement & scheduling

The scheduler is **not** a fixed “layer 0 = bank A, layer 1 = bank B”
rule. Two model-driven decisions:

- **`BankAllocator`** (`scheduler.py`). Two physical activation banks
  (A, B). For each layer in topo order:
  1. Find the bank holding the layer’s activation input.
  2. Decrement that tensor’s remaining-use counter; free the bank if
     it falls to zero.
  3. Pick a free bank for the output, preferring the bank not used by
     the input (so reads and writes use disjoint SRAMs). If both banks
     hold live tensors, raise `BankConflict` rather than silently
     overwriting one.

  For a strict linear MLP this collapses to A/B/A/B alternation
  (preserving the hardware ping-pong invariant). For a graph that
  would clobber a still-live tensor, the compiler refuses instead of
  producing wrong RTL. See `tests/test_placement.py`.

- **`peak_aware_topological_order`** (`dag_scheduler.py`). Among nodes
  whose in-degree just hit zero, pick the one with the smallest
  `output_bytes − freed_input_bytes`. This is a Sethi–Ullman-style
  greedy and is the default for `DAGScheduler`. On a strict chain it
  is identical to Kahn’s order; on a branchy graph it lowers peak
  activation SRAM. Disable with `DAGScheduler(peak_aware_order=False)`.

`ActivationLifetimeAnalyzer` and `SRAMBudgetPlanner` in
`dag_scheduler.py` consume the resulting order to produce the
budget report.

---

## Tests

```bash
make test                                              # full suite
python -m pytest tests/test_placement.py -v            # bank allocator + reorder
python -m pytest tests/test_dag_scheduler.py -v        # DAG path
python -m pytest tests/test_rtl_gen.py -v              # RTL lowering
make rtl-lint                                          # Verilator lint on cell library
```

Conftest fixtures (`tests/conftest.py`) build the AD model, a CNN, a
transformer, and a residual block on demand — most stage tests reuse
those.

The full suite includes a few real-model integration tests
(`test_real_models.py`, `test_tinyml_models.py`, `test_large_models.py`)
that load larger ONNX files and can take a minute or more. Skip them
locally with `-k "not real_models and not tinyml and not large_models"`
if you only care about unit-level changes.

---

## Generated output (per model)

```
output/<name>/
  schedule.json                     timing + SRAM layout
  weights/                          INT8 .mem files + memory_map.json
  rtl/
    parameters.svh                  per-model localparams
    accelerator_top.sv              top module
    compute/ memory/ layer/ interface/ tile/ constraints/
  vectors/
    mem/  (for SV)                  $readmemh inputs/expected outputs
    npy/  (for cocotb)              .npy + manifest
  testbench/
    tb_*.sv                         per-module SV testbenches
    test_accelerator.py             cocotb system test
    Makefile                        cocotb runner
  coverage.json                     functional coverage report
```

---

## Hardware target (v0.1)

| | |
|---|---|
| Device | Xilinx Zynq UltraScale+ `xck26-sfvc784-2LV` (Kria KV260) |
| Clock | 100 MHz |
| MAC array | 128 INT8 MACs, INT32 accumulators |
| Activation SRAM | 2 banks × 640 B (ping-pong) |
| Weight SRAM | 1536 rows × 128 B |
| Bias SRAM | 32 rows × 32×INT32 |
| Control / data | AXI-Lite @ 0x4000_0000, AXI-Stream 64-bit |
| Quantization | Symmetric INT8 weights, asymmetric INT8 acts, fixed-point 16.16 requant, round-half-up |
| PPA targets | <0.05 ms latency, >20K inf/sec, <15 % LUT, <40 % BRAM |

Round-half-up (`floor(x + 0.5)`) is mandatory in any quantization path
— `np.round()` uses banker’s rounding and produces bitwise mismatches
against the hardware. `int8_interpreter.py` is the canonical reference;
golden vectors are derived from it.

---

## Specs

| File | Contents |
|---|---|
| [`docs/PRD.md`](docs/PRD.md) | Product requirements, milestones, PPA targets |
| [`docs/compiler-ir-spec.md`](docs/compiler-ir-spec.md) | IR data structures, supported ops, invariants |
| [`docs/quantization-spec.md`](docs/quantization-spec.md) | INT8 math, requantization, rounding |
| [`docs/rtl-interface-spec.md`](docs/rtl-interface-spec.md) | SystemVerilog modules, AXI, memory map, FSM timing |
| [`docs/firmware-integration-guide.md`](docs/firmware-integration-guide.md) | Zynq driver API, DMA, register map |

---

## License

MIT
