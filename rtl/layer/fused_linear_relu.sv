// fused_linear_relu.sv — Layer control FSM with output tiling
// From rtl-interface-spec.md §6
// This is a CONTROL-ONLY module: it generates addresses and MAC control
// signals. The single shared MAC array is instantiated in accelerator_top.
// FSM: IDLE -> LOAD_BIAS -> COMPUTE -> REQUANT -> WRITE_TILE -> NEXT_TILE/DONE

/* verilator lint_off VARHIDDEN */
module fused_linear_relu #(
    parameter int PARALLELISM = 128,
    parameter int DATA_W      = 8,
    parameter int ACC_W       = 32,
    parameter int ACT_DEPTH   = 80,
    parameter int ACT_WIDTH   = 64
) (
    input  logic                               clk,
    input  logic                               rst_n,

    // Layer configuration (from parameter ROM in accelerator_top)
    input  logic                               start,
    input  logic [15:0]                        input_dim,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [15:0]                        output_dim,  // Used by accelerator_top for addressing
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [15:0]                        num_tiles,
    input  logic                               has_relu,
    input  logic [15:0]                        weight_base,
    input  logic [15:0]                        bias_base,
    input  logic [ACC_W-1:0]                   requant_scale,
    input  logic [5:0]                         requant_shift,
    input  logic signed [DATA_W-1:0]           requant_zp,

    // Weight SRAM interface
    output logic [15:0]                        weight_addr,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [PARALLELISM*DATA_W-1:0]      weight_rdata,  // Routed to MAC array in accelerator_top
    /* verilator lint_on UNUSEDSIGNAL */

    // Bias SRAM interface
    output logic [15:0]                        bias_addr,
    input  logic [PARALLELISM*ACC_W-1:0]       bias_rdata,

    // Activation read interface (from ping-pong input bank)
    output logic [$clog2(ACT_DEPTH)-1:0]       act_rd_addr,
    input  logic [ACT_WIDTH-1:0]               act_rd_data,

    // Activation write interface (to ping-pong output bank)
    output logic [$clog2(ACT_DEPTH)-1:0]       act_wr_addr,
    output logic                               act_wr_en,
    output logic [ACT_WIDTH-1:0]               act_wr_data,

    // MAC array control interface
    output logic                               mac_start,
    output logic [15:0]                        mac_input_dim,
    output logic                               mac_has_relu,
    output logic [ACC_W-1:0]                   mac_scale,
    output logic [5:0]                         mac_shift,
    output logic signed [DATA_W-1:0]           mac_zp,
    output logic signed [DATA_W-1:0]           mac_input_data,
    output logic                               mac_input_valid,
    output logic                               mac_bias_valid,
    output logic [$clog2(PARALLELISM)-1:0]     mac_bias_idx,
    output logic signed [ACC_W-1:0]            mac_bias_data,

    // MAC array output interface
    input  logic signed [DATA_W-1:0]           mac_output_data [0:PARALLELISM-1],
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic                               mac_output_valid,  // FSM uses mac_done instead
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic                               mac_done,

    // Layer done
    output logic                               layer_done
);

// FSM states
typedef enum logic [2:0] {
    S_IDLE,
    S_LOAD_BIAS,
    S_COMPUTE,
    S_WAIT_MAC,
    S_WRITE_TILE,
    S_NEXT_TILE,
    S_DONE
} layer_state_t;

layer_state_t state, state_next;

// Tile and counters
logic [15:0] tile_idx;
logic [6:0]  bias_cnt;       // 0..127 (for bias loading)
logic [4:0]  write_cnt;      // 0..15 (16 cycles to write 128 bytes via 64-bit port)

// Byte select for activation input
logic [2:0] byte_sel;

// Weight address: weight_base + tile_idx * input_dim + input_idx
logic [15:0] input_cnt;

// FSM next-state
always_comb begin
    state_next = state;
    case (state)
        S_IDLE:       if (start) state_next = S_LOAD_BIAS;
        S_LOAD_BIAS:  if (bias_cnt == 7'(PARALLELISM - 1)) state_next = S_COMPUTE;
        S_COMPUTE:    if (input_cnt == input_dim - 16'd1) state_next = S_WAIT_MAC;
        S_WAIT_MAC:   if (mac_done) state_next = S_WRITE_TILE;
        S_WRITE_TILE: if (write_cnt == 5'd15) state_next = S_NEXT_TILE;
        S_NEXT_TILE:  begin
            if (tile_idx == num_tiles - 16'd1)
                state_next = S_DONE;
            else
                state_next = S_LOAD_BIAS;
        end
        S_DONE: state_next = S_IDLE;
        default: state_next = S_IDLE;
    endcase
end

// State register
always_ff @(posedge clk) begin
    if (!rst_n)
        state <= S_IDLE;
    else
        state <= state_next;
end

// Tile counter
always_ff @(posedge clk) begin
    if (!rst_n || state == S_IDLE)
        tile_idx <= '0;
    else if (state == S_NEXT_TILE)
        tile_idx <= tile_idx + 1;
end

// Bias load counter and SRAM addressing
// 128 biases loaded over 128 cycles: 4 SRAM reads of 32 biases each
always_ff @(posedge clk) begin
    if (!rst_n || state == S_IDLE || state == S_NEXT_TILE)
        bias_cnt <= '0;
    else if (state == S_LOAD_BIAS)
        bias_cnt <= bias_cnt + 1;
end

// Bias SRAM row: bias_base + tile_idx * 4 + floor(bias_cnt / 32)
always_comb begin
    bias_addr = bias_base + tile_idx * 16'd4 + {14'b0, bias_cnt[6:5]};
end

// Bias data to MAC: extract from 1024-bit row using bias_cnt[4:0]
always_comb begin
    mac_bias_data = $signed(bias_rdata[bias_cnt[4:0]*ACC_W +: ACC_W]);
    mac_bias_idx  = bias_cnt[$clog2(PARALLELISM)-1:0];
    mac_bias_valid = (state == S_LOAD_BIAS);
end

// Compute: iterate over input_dim
always_ff @(posedge clk) begin
    if (!rst_n || state != S_COMPUTE)
        input_cnt <= '0;
    else
        input_cnt <= input_cnt + 1;
end

// Weight address: weight_base + tile_idx * input_dim + input_cnt
always_comb begin
    weight_addr = weight_base + tile_idx * input_dim + input_cnt;
end

// Activation read: 64-bit words, 8 bytes each
// act_rd_addr = floor(input_cnt / 8), byte_sel = input_cnt % 8
always_comb begin
    act_rd_addr = input_cnt[$clog2(ACT_DEPTH)+2:3];  // input_cnt / 8
    byte_sel    = input_cnt[2:0];    // input_cnt % 8
end

// Extract byte from 64-bit activation word
always_comb begin
    mac_input_data  = $signed(act_rd_data[byte_sel*8 +: 8]);
    mac_input_valid = (state == S_COMPUTE);
end

// MAC control signals
assign mac_start     = (state == S_IDLE && start);
assign mac_input_dim = input_dim;
assign mac_has_relu  = has_relu;
assign mac_scale     = requant_scale;
assign mac_shift     = requant_shift;
assign mac_zp        = requant_zp;

// Write tile: 128 bytes of output over 16 cycles (8 bytes per cycle via 64-bit port)
// Base write address: tile_idx * (PARALLELISM / 8)
always_ff @(posedge clk) begin
    if (!rst_n || state != S_WRITE_TILE)
        write_cnt <= '0;
    else
        write_cnt <= write_cnt + 1;
end

// Write address calculation
/* verilator lint_off WIDTHTRUNC */
logic [$clog2(ACT_DEPTH)-1:0] wr_addr_full;
assign wr_addr_full = tile_idx * 16'(PARALLELISM / 8) + {11'b0, write_cnt};
/* verilator lint_on WIDTHTRUNC */

always_comb begin
    act_wr_addr = wr_addr_full;
    act_wr_en   = (state == S_WRITE_TILE);

    // Pack 8 INT8 outputs into 64-bit word
    for (int b = 0; b < 8; b++) begin
        act_wr_data[b*8 +: 8] = mac_output_data[write_cnt*8 + b];
    end
end

// Layer done
assign layer_done = (state == S_DONE);

endmodule
