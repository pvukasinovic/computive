// parameters.svh — Template for model-specific constants
// This file is a TEMPLATE. The compiler generates a model-specific
// version in the output directory with actual values filled in.
//
// The generated version contains:
//   - NUM_LAYERS, MAX_DIM, PARALLELISM, etc.
//   - Per-layer arrays: LAYER_IN_DIM, LAYER_OUT_DIM, LAYER_RELU
//   - Weight/bias base addresses: WEIGHT_BASE, BIAS_BASE
//   - Requantization parameters: REQUANT_M, REQUANT_SHIFT, REQUANT_ZP
//   - SRAM depths: WEIGHT_DEPTH, BIAS_DEPTH, ACT_DEPTH
//   - Memory init file paths

// Example (filled by RTLGenerator for AD model):
// localparam int NUM_LAYERS  = 4;
// localparam int MAX_DIM     = 640;
// localparam int PARALLELISM = 128;
// localparam int AXI_DATA_W  = 64;
// localparam int WEIGHT_DEPTH = 1536;
// localparam int BIAS_DEPTH   = 32;
// localparam int ACT_DEPTH    = 80;
//
// localparam logic [15:0] LAYER_IN_DIM  [0:3] = '{16'd640, 16'd128, 16'd128, 16'd128};
// localparam logic [15:0] LAYER_OUT_DIM [0:3] = '{16'd128, 16'd128, 16'd128, 16'd640};
// ...
