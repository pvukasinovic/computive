# MLASIC

**ML model to ASIC compiler** — transforms frozen ONNX models into synthesizable, model-specific hardware accelerators (SystemVerilog RTL) optimized for power, performance, and area.

```
ONNX Model  ──>  Ingestion  ──>  Optimization  ──>  Scheduling  ──>  Weight Packing  ──>  RTL Gen  ──>  Verification
 (.onnx)        (Stage 1)       (Stage 2)          (Stage 3)        (Stage 4)           (Stage 5)     (Stage 6)
```

## Overview

MLASIC takes a trained ONNX neural network and compiles it into a complete, synthesizable hardware accelerator. The generated RTL includes:

- **Compute modules**: 128-wide INT8 MAC array, requantization, activation functions
- **Memory system**: SRAM banks with ping-pong buffering for activations, weight/bias storage
- **Control logic**: Layer FSMs with output tiling for large dimensions
- **Interfaces**: AXI-Lite control/status registers, AXI-Stream data input/output
- **Verification**: Golden test vectors, SystemVerilog testbenches, cocotb tests, functional coverage

The v0.1 target is the [MLPerf Tiny Anomaly Detection](https://github.com/mlcommons/tiny) model (4-layer dense autoencoder, ~100K parameters, INT8 quantized) on a Xilinx Zynq UltraScale+ (Kria KV260).

### Supported Model Types

| Path | Model Types | Architecture |
|------|-------------|--------------|
| **MLP** | Dense/linear autoencoders, classifiers | Shared MAC array, time-multiplexed layers, ping-pong SRAM |
| **CNN** | Convolutional networks (Conv, Pool, etc.) | Tile fabric with per-tile compute units, static routing |
| **ASIC** | Large models (100M+ params) | Tile fabric with per-tile hardcoded weight ROM |

The compiler auto-selects the appropriate path based on the model's operator types.

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone <repo-url>
cd mlasic

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode
make install
```

This installs the `mlasic` package and its dependencies (`onnx`, `onnxruntime`, `numpy`, `scipy`) plus dev tools (`pytest`, `ruff`).

### Run the Demo Pipeline

The fastest way to see MLASIC in action is the included demo script that compiles 4 models end-to-end:

```bash
python run_two_models.py
```

This compiles:

1. **MLPerf Tiny AD** — 640->128->128->128->640 MLP autoencoder (MLP path)
2. **Narrow Bottleneck MLP** — 640->128->64->128->640 autoencoder (MLP path)
3. **Simple CNN** — Conv->BN->ReLU->MaxPool->GlobalAvgPool->Dense (tile fabric path)
4. **Large MLP** — 369M parameter, 21-layer MLP (ASIC tile fabric path)

Output goes to `output/<model_name>/` with RTL, weights, testbenches, and golden vectors.

### Run Tests

```bash
make test          # Run all 514+ tests
make lint          # Check code style (ruff)
make format        # Auto-fix code style
make rtl-lint      # Verilator lint on SystemVerilog modules (requires verilator)
```

## Compiling Your Own Model

### Step 1: Prepare an ONNX Model

Export your trained model to ONNX format. For PyTorch:

```python
import torch
import torch.onnx

model = YourModel()
model.eval()
dummy_input = torch.randn(1, 640)  # Match your input shape
torch.onnx.export(model, dummy_input, "my_model.onnx", opset_version=17)
```

Place the `.onnx` file anywhere accessible. The `models/` directory contains pre-downloaded models for reference (BERT, GPT2, MobileNetV2, ResNet50, etc.).

### Step 2: Run the Compiler Pipeline

```python
#!/usr/bin/env python3
from pathlib import Path
import numpy as np

from mlasic.ingestion import ONNXParser
from mlasic.optimization import (
    PassManager,
    ConstantFoldingPass,
    DeadCodeEliminationPass,
    BatchNormFoldingPass,
    OperatorFusionPass,
    QuantizationPass,
)
from mlasic.scheduler import Scheduler, schedule_to_json
from mlasic.weight_packer import WeightPacker
from mlasic.rtl_gen import RTLGenerator
from mlasic.golden_vectors import GoldenVectorGenerator
from mlasic.testbench_gen import VerifConfig, VerifGenerator
from mlasic.ir import HardwareConstraints

# ── Configuration ──
model_path = Path("my_model.onnx")
output_dir = Path("output/my_model")
input_shape = (1, 640)  # Your model's input shape

# ── Stage 1: Parse ONNX ──
graph = ONNXParser(model_path).parse()
print(f"Parsed {len(graph.nodes)} nodes, {len(graph.tensors)} tensors")

# ── Stage 2: Optimize & Quantize ──
# Generate calibration data (representative inputs for quantization)
rng = np.random.RandomState(42)
calibration_data = [rng.randn(*input_shape).astype(np.float32) for _ in range(20)]

pm = PassManager()
pm.add_pass(ConstantFoldingPass())
pm.add_pass(DeadCodeEliminationPass())
pm.add_pass(BatchNormFoldingPass())
pm.add_pass(OperatorFusionPass())
pm.add_pass(QuantizationPass(calibration_data=calibration_data))
graph = pm.run(graph, verify=True)
print(f"Optimized to {len(graph.nodes)} fused nodes")

# ── Stage 3: Schedule ──
hw = HardwareConstraints()
schedule = Scheduler(constraints=hw).schedule(graph)
schedule_to_json(schedule, output_dir / "schedule.json")
print(f"Scheduled: {schedule.total_cycles} cycles, {schedule.latency_us:.2f} us")

# ── Stage 4: Pack Weights ──
weight_dir = output_dir / "weights"
memory_map = WeightPacker(graph, weight_dir, constraints=hw).pack_weights()
print(f"Packed weights: {memory_map['weight_bank']['total_bytes']:,} bytes")

# ── Stage 5: Generate RTL ──
rtl_dir = output_dir / "rtl"
RTLGenerator(
    graph=graph,
    weight_dir=weight_dir,
    output_dir=rtl_dir,
    constraints=hw,
).generate_all()
print(f"Generated {len(list(rtl_dir.rglob('*.sv')))} SystemVerilog files")

# ── Stage 6: Generate Verification ──
gvg = GoldenVectorGenerator(graph)
vectors = gvg.generate_random_vectors(n=100, seed=42)
vec_dir = output_dir / "vectors"
gvg.export_mem(vectors, vec_dir / "mem")
gvg.export_npy(vectors, vec_dir / "npy")

tb_dir = output_dir / "testbench"
config = VerifConfig(graph=graph, output_dir=tb_dir, weight_dir=weight_dir, constraints=hw)
VerifGenerator(config).generate_all()
print(f"Generated testbenches in {tb_dir}")
```

### Step 3: CNN/Transformer Models

For models with Conv, Softmax, LayerNorm, or other non-linear operators, use the DAG pipeline:

```python
from mlasic.dag_scheduler import DAGScheduler
from mlasic.tile_mapper import TileMapper
from mlasic.optimization import (
    ConvBatchNormFoldingPass,
    ConvFusionPass,
    ConvQuantizationPass,
)

# Stage 2: Use CNN-specific passes
pm = PassManager()
pm.add_pass(ConstantFoldingPass())
pm.add_pass(DeadCodeEliminationPass())
pm.add_pass(BatchNormFoldingPass())
pm.add_pass(ConvBatchNormFoldingPass())
pm.add_pass(DeadCodeEliminationPass())
pm.add_pass(ConvFusionPass())
pm.add_pass(OperatorFusionPass())
pm.add_pass(ConvQuantizationPass(calibration_data=calibration_data))
pm.add_pass(QuantizationPass(calibration_data=calibration_data))
graph = pm.run(graph, verify=False)

# Stage 3: DAG scheduling + tile mapping
dag_schedule = DAGScheduler().schedule(graph)
fabric = TileMapper().map(graph, dag_schedule)

# Stage 5: RTL with tile fabric
RTLGenerator(
    graph=graph,
    weight_dir=weight_dir,
    output_dir=rtl_dir,
    constraints=HardwareConstraints(),
    fabric_config=fabric,      # Enables tile fabric path
    target="fpga",             # or "asic" for hardcoded weight ROM
).generate_all()
```

## Output Structure

After compilation, each model produces:

```
output/<model_name>/
├── <model_name>.onnx           # Original ONNX model
├── schedule.json               # Timing & SRAM layout
├── coverage.json               # Functional coverage report
│
├── weights/                    # INT8 weights in $readmemh format
│   ├── weights_layer0.mem      # Per-layer weight memory
│   ├── biases_layer0.mem       # Per-layer bias memory
│   ├── weight_bank.mem         # Combined weight SRAM image
│   ├── bias_bank.mem           # Combined bias SRAM image
│   └── memory_map.json         # SRAM address map for RTL
│
├── rtl/                        # Complete synthesizable RTL
│   ├── parameters.svh          # Model-specific parameters
│   ├── accelerator_top.sv      # Top-level module
│   ├── compute/                # MAC array, requantize, activations
│   ├── memory/                 # SRAM banks, ping-pong buffer
│   ├── layer/                  # Layer control FSMs
│   ├── interface/              # AXI-Lite/Stream interfaces
│   ├── tile/                   # Tile wrapper & fabric (CNN/ASIC)
│   └── constraints/            # Xilinx .xdc timing constraints
│
├── vectors/                    # Golden test vectors
│   ├── mem/                    # $readmemh format (for SV testbenches)
│   │   ├── input_vectors.mem
│   │   ├── expected_outputs.mem
│   │   └── test_manifest.json
│   └── npy/                    # NumPy format (for cocotb)
│       ├── inputs.npy
│       ├── outputs.npy
│       └── manifest.json
│
└── testbench/                  # Verification testbenches
    ├── tb_mac_array.sv         # MAC array unit test
    ├── tb_requantize.sv        # Requantization unit test
    ├── tb_activation_relu.sv   # ReLU unit test
    ├── tb_sram_bank.sv         # SRAM unit test
    ├── tb_ping_pong_buffer.sv  # Ping-pong unit test
    ├── tb_fused_linear_relu.sv # Layer FSM unit test
    ├── tb_axi_stream_in.sv     # AXI-Stream input test
    ├── tb_axi_stream_out.sv    # AXI-Stream output test
    ├── tb_axi_lite_ctrl.sv     # AXI-Lite CSR test
    ├── tb_accelerator.sv       # System-level integration test
    ├── test_accelerator.py     # cocotb Python test
    └── Makefile                # cocotb simulation runner
```

## Pipeline Stages

### Stage 1: ONNX Ingestion

Parses ONNX protobuf into an internal IR (Intermediate Representation) graph. Supports 40+ ONNX operators across two tiers:

**Tier 1** (MLP): MatMul, Add, ReLU, BatchNormalization, Reshape, Transpose, Flatten
**Tier 2** (CNN/Transformer): Conv, Gemm, Softmax, LayerNorm, MaxPool, AveragePool, GlobalAveragePool, Sigmoid, Tanh, Concat, Gather, and more

The parser performs shape inference, extracts operator attributes, and builds a typed tensor graph.

### Stage 2: Graph Optimization

Runs a sequence of optimization and quantization passes:

| Pass | Effect |
|------|--------|
| `ConstantFoldingPass` | Evaluates constant subexpressions at compile time |
| `DeadCodeEliminationPass` | Removes unreachable nodes |
| `BatchNormFoldingPass` | Folds BatchNorm parameters into preceding MatMul+Add |
| `ConvBatchNormFoldingPass` | Folds BatchNorm into Conv weights (CNN models) |
| `OperatorFusionPass` | Fuses MatMul+Add[+ReLU] into FusedLinear[ReLU] |
| `ConvFusionPass` | Fuses Conv+ReLU/Clip into FusedConvReLU |
| `ActivationFusionPass` | Fuses GELU and SiLU patterns |
| `QuantizationPass` | Calibration-based INT8 quantization with requantization parameters |
| `ConvQuantizationPass` | Per-channel INT8 quantization for Conv operators |

Each pass can be verified against a floating-point reference interpreter to ensure correctness.

### Stage 3: Dataflow Scheduling

Two scheduling modes:

- **Linear Scheduler** (`Scheduler`): For pure MLP models. Assigns cycle budgets, SRAM addresses, and ping-pong buffer banks per layer. Formula: `cycles_per_tile = input_dim + 149`.
- **DAG Scheduler** (`DAGScheduler`): For arbitrary graphs (CNN, Transformer). Per-operator cycle models, activation lifetime analysis, multi-activation SRAM budget planning. Plus **tile mapping** (`TileMapper`) for spatial assignment.

### Stage 4: Weight Packing

Quantized INT8 weights and INT32 biases are packed into `$readmemh`-compatible `.mem` files matching the RTL SRAM access pattern:

- 1024-bit wide rows (128 bytes per row for weights, 32 INT32 biases per row)
- Per-layer files + combined bank files
- `memory_map.json` with SRAM addresses for RTL parameterization

For CNN/ASIC targets, `WeightROMMapper` generates per-tile ROM binary images.

### Stage 5: RTL Generation

Generates a complete, synthesizable SystemVerilog design:

- **`parameters.svh`**: Model-specific localparam arrays (dimensions, SRAM addresses, requantization parameters)
- **`accelerator_top.sv`**: Top-level module instantiating all submodules
- **Module library**: MAC array, requantize, activation, SRAM, ping-pong buffer, layer FSM, AXI interfaces
- **Tile fabric** (CNN/ASIC): Per-tile compute units with static inter-tile routing

Supports dual targets:
- **FPGA**: `sram_bank.sv` with `$readmemh` weight loading
- **ASIC**: `rom_tile.sv` with hardcoded `initial` blocks for weight ROM

### Stage 6: Verification

Generates everything needed to verify the hardware:

- **Golden vectors**: 100+ random + adversarial + rounding-boundary test vectors generated by an INT8 interpreter that matches hardware arithmetic bitwise
- **9 module-level SystemVerilog testbenches**: One per RTL module
- **System-level testbench**: Full accelerator with AXI drivers and per-vector comparison
- **cocotb Python test**: AXI-Lite/Stream drivers with golden vector checking
- **Functional coverage**: 11 coverage items tracking layer execution, SRAM access, tiling, interrupts, etc.

## Architecture Details

### INT8 Quantization

| Component | Strategy |
|-----------|----------|
| Weights | Symmetric: zero_point=0, range [-127, 127] |
| Activations | Asymmetric: variable zero_point, range [-128, 127] |
| Accumulation | INT32 (prevents overflow) |
| Requantization | Fixed-point 16.16 multiplier, round-half-up (`floor(x + 0.5)`) |

The round-half-up rounding mode is critical — it matches hardware behavior. Python's `np.round()` uses banker's rounding, which would cause bitwise mismatches.

### Hardware Target (v0.1)

| Parameter | Value |
|-----------|-------|
| FPGA | Xilinx Zynq UltraScale+ (xck26-sfvc784-2LV) |
| Clock | 100 MHz |
| MAC Parallelism | 128 INT8 MACs |
| SRAM Budget | ~197 KB (fits KV260 BRAM) |
| Control Interface | AXI-Lite (base 0x4000_0000) |
| Data Interface | AXI-Stream, 64-bit, 80 beats per input |
| PPA Targets | <0.05 ms latency, >20K inf/sec, <15% LUT, <40% BRAM |

### SystemVerilog Module Library

```
rtl/
├── compute/
│   ├── mac_array.sv           # 128 parallel MACs + FSM
│   ├── requantize.sv          # INT32->INT8 pipeline (3 stages)
│   ├── activation_relu.sv     # Combinational INT8 ReLU
│   ├── conv_engine.sv         # im2col + MAC array reuse
│   ├── softmax_unit.sv        # 3-stage pipeline with exp LUT
│   ├── layer_norm_unit.sv     # 2-pass normalize with rsqrt LUT
│   ├── activation_unit.sv     # ReLU/ReLU6/GELU/SiLU (parameterized)
│   └── pool_unit.sv           # Max/Avg/GlobalAvg pooling
├── memory/
│   ├── sram_bank.sv           # Behavioral SRAM, $readmemh init
│   ├── ping_pong_buffer.sv    # Double-buffered activation memory
│   └── rom_tile.sv            # ASIC ROM template
├── layer/
│   ├── fused_linear_relu.sv   # Layer control FSM with output tiling
│   ├── byte_select.sv         # 64-bit -> 8-bit byte mux
│   └── bias_unpack.sv         # 1024-bit -> 32-bit unpacker
├── interface/
│   ├── axi_stream_in.sv       # AXI-Stream slave
│   ├── axi_stream_out.sv      # AXI-Stream master
│   └── axi_lite_ctrl.sv       # CSR register file (10 registers)
├── tile/
│   ├── tile.sv                # Generic tile wrapper
│   └── tile_fabric.sv         # Generated fabric template
├── top/
│   └── accelerator_top.sv     # Model-specific top module
└── constraints/
    └── constraints.xdc        # Xilinx timing constraints
```

## Project Structure

```
mlasic/
├── src/mlasic/                 # Python compiler package
│   ├── __init__.py             # Public API (155 exports)
│   ├── ir.py                   # IR data structures (Graph, OpNode, Tensor, QuantParams, ...)
│   ├── exceptions.py           # Custom exceptions
│   ├── ingestion.py            # Stage 1: ONNX parser
│   ├── optimization.py         # Stage 2: Optimization passes
│   ├── interpreter.py          # FP32 reference interpreter
│   ├── int8_interpreter.py     # Hardware-exact INT8 interpreter
│   ├── scheduler.py            # Stage 3: Linear MLP scheduler
│   ├── dag_scheduler.py        # Stage 3: DAG scheduler (CNN/Transformer)
│   ├── tile_mapper.py          # Stage 3b: Spatial tile mapping
│   ├── rom_mapper.py           # Stage 4: Per-tile ROM generation
│   ├── weight_packer.py        # Stage 4: Weight/bias .mem packing
│   ├── rtl_gen.py              # Stage 5: SystemVerilog generation
│   ├── golden_vectors.py       # Stage 6: Golden test vector generation
│   ├── testbench_gen.py        # Stage 6: Testbench generation
│   └── export.py               # IR-to-ONNX exporter (round-trip testing)
│
├── rtl/                        # SystemVerilog module library
├── tests/                      # 514+ tests across 19 test files
├── models/                     # Pre-downloaded ONNX models
├── output/                     # Generated output (per-model directories)
├── docs/                       # Specifications
│   ├── PRD.md                  # Product requirements
│   ├── compiler-ir-spec.md     # IR specification & invariants
│   ├── quantization-spec.md    # INT8 quantization specification
│   ├── rtl-interface-spec.md   # SystemVerilog module specifications
│   └── firmware-integration-guide.md  # Zynq driver & DMA guide
│
├── pyproject.toml              # Package metadata & dependencies
├── Makefile                    # Build/test/lint commands
├── run_two_models.py           # End-to-end demo (4 models)
└── STATUS.md                   # Development progress tracking
```

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make install` | Install package in editable mode with dev dependencies |
| `make test` | Run all tests with pytest |
| `make lint` | Check code style with ruff |
| `make format` | Auto-fix code style |
| `make rtl-lint` | Verilator lint on SystemVerilog modules |
| `make clean` | Remove build artifacts |

## Specifications

Detailed design specifications live in `docs/`:

| Document | Contents |
|----------|----------|
| [PRD](docs/PRD.md) | Product requirements, milestones, success criteria, PPA targets |
| [Compiler IR Spec](docs/compiler-ir-spec.md) | IR data structures, supported operators, invariants (INV-1.1 through INV-4.6) |
| [Quantization Spec](docs/quantization-spec.md) | INT8 symmetric/asymmetric quantization, INT32 accumulation, fixed-point requantization |
| [RTL Interface Spec](docs/rtl-interface-spec.md) | SystemVerilog module specs, AXI interfaces, memory map, FSM timing |
| [Firmware Guide](docs/firmware-integration-guide.md) | Zynq ARM driver API, DMA setup, register map, interrupt handling |

## License

MIT
