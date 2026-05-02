// tile_fabric.sv — Template for generated tile fabric
//
// This file is a TEMPLATE only. The RTL generator produces a
// model-specific version with:
//   - N tile instances (from FabricConfig.tiles)
//   - Static wiring (assign statements from FabricConfig.routes)
//   - Activation buffers (from lifetime analysis)
//   - Fabric FSM (sequential tile execution)
//
// The generated module follows this structure:

// module tile_fabric #(
//     parameter int NUM_TILES    = N,
//     parameter int PARALLELISM  = 128,
//     parameter int DATA_W       = 8,
//     parameter int ACC_W        = 32,
//     parameter int AXI_DATA_W   = 64
// ) (
//     input  logic        clk,
//     input  logic        rst_n,
//
//     // Fabric control
//     input  logic        start,
//     output logic        done,
//     output logic        busy,
//
//     // Activation input (from AXI-Stream)
//     input  logic [AXI_DATA_W-1:0] act_in_data,
//     output logic [15:0] act_in_addr,
//     input  logic        act_in_valid,
//
//     // Activation output (to AXI-Stream)
//     output logic [AXI_DATA_W-1:0] act_out_data,
//     output logic [15:0] act_out_addr,
//     output logic        act_out_valid
// );
//
//     // ============================================================
//     // Fabric FSM: sequential tile execution
//     // ============================================================
//
//     typedef enum logic [2:0] {
//         S_IDLE,
//         S_RUN_TILE,
//         S_NEXT_TILE,
//         S_DONE
//     } fabric_state_t;
//
//     fabric_state_t state;
//     logic [$clog2(NUM_TILES)-1:0] tile_idx;
//     logic [NUM_TILES-1:0] tile_start;
//     logic [NUM_TILES-1:0] tile_done;
//
//     // Per-tile instances generated here by RTL generator
//     // tile #(...) u_tile_0 (...);
//     // tile #(...) u_tile_1 (...);
//     // ...
//
//     // Static wiring (assign statements)
//     // assign tile_0_act_in = fabric_act_in;
//     // assign tile_1_act_in = tile_0_act_out;
//     // ...
//
//     // Fabric FSM
//     // always_ff @(posedge clk) ...
//
// endmodule
