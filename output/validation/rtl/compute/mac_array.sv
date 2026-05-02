// mac_array.sv — Parallel MAC array with internal FSM
// From rtl-interface-spec.md §5
// 128 parallel MACs, INT8 inputs/weights, INT32 accumulators
// FSM: IDLE -> BIAS_LOAD -> COMPUTE -> REQUANT -> OUTPUT -> DONE
// Instantiates requantize module for INT32->INT8 conversion

/* verilator lint_off VARHIDDEN */
module mac_array #(
    parameter int PARALLELISM  = 128,
    parameter int DATA_W       = 8,
    parameter int ACC_W        = 32,
    parameter int PER_CHANNEL  = 0    // 0=scalar requant, 1=per-lane buses
) (
    input clk,
    input rst_n,

    // Control
    input start,
    input [15:0] input_dim,
    input has_relu,

    // Weight interface (1024-bit row = 128 x INT8)
    input [PARALLELISM*DATA_W-1:0] weight_data,
    /* verilator lint_off UNUSEDSIGNAL */
    input weight_valid,  // AXI-spec port, timing driven by FSM
    /* verilator lint_on UNUSEDSIGNAL */

    // Input activation (one byte per cycle from byte_select)
    input signed [DATA_W-1:0] input_data,
    input input_valid,

    // Bias interface (from bias_unpack: 32-bit bias, loaded over 128 cycles)
    input signed [ACC_W-1:0] bias_data,
    input bias_valid,
    input [$clog2(PARALLELISM)-1:0] bias_idx,

    // Requantization parameters (scalar — used when PER_CHANNEL=0)
    input [ACC_W-1:0] scale,
    input [5:0] shift,
    input signed [DATA_W-1:0] zero_point,

    // Per-channel requant buses (used when PER_CHANNEL=1)
    /* verilator lint_off UNUSEDSIGNAL */
    input [PARALLELISM*ACC_W-1:0] scale_bus,
    input [PARALLELISM*6-1:0] shift_bus,
    input [PARALLELISM*DATA_W-1:0] zp_bus,
    /* verilator lint_on UNUSEDSIGNAL */

    // Output (128 x INT8 after requantization)
    output logic signed [DATA_W-1:0] output_data [0:PARALLELISM-1],
    output logic output_valid,
    output logic done
);

// Internal state
typedef enum logic [2:0] {
    S_IDLE,
    S_BIAS_LOAD,
    S_COMPUTE,
    S_REQUANT,
    S_OUTPUT,
    S_DONE
} mac_state_t;

mac_state_t state, state_next;

// Accumulators
logic signed [ACC_W-1:0] acc [0:PARALLELISM-1];

// Bias load counter
logic [6:0] bias_cnt;  // 0..127

// Compute counter
logic [15:0] compute_cnt;

// Requant pipeline counter (3 stages)
logic [1:0] requant_cnt;

// Requantize module signals
logic                       rq_valid_in;
logic signed [DATA_W-1:0]  rq_data_out [0:PARALLELISM-1];
logic                       rq_valid_out;

requantize #(
    .PARALLELISM(PARALLELISM),
    .ACC_W(ACC_W),
    .DATA_W(DATA_W),
    .PER_CHANNEL(PER_CHANNEL)
) u_requantize (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(rq_valid_in),
    .acc(acc),
    .scale(scale),
    .shift(shift),
    .zero_point(zero_point),
    .scale_bus(scale_bus),
    .shift_bus(shift_bus),
    .zp_bus(zp_bus),
    .has_relu(has_relu),
    .data_out(rq_data_out),
    .valid_out(rq_valid_out)
);

// FSM next-state logic
always_comb begin
    state_next = state;
    case (state)
        S_IDLE:      if (start) state_next = S_BIAS_LOAD;
        S_BIAS_LOAD: if (bias_cnt == 7'(PARALLELISM - 1)) state_next = S_COMPUTE;
        S_COMPUTE:   if (compute_cnt == input_dim) state_next = S_REQUANT;
        S_REQUANT:   if (requant_cnt == 2'd3) state_next = S_OUTPUT;
        S_OUTPUT:    state_next = S_DONE;
        S_DONE:      state_next = S_IDLE;
        default:     state_next = S_IDLE;
    endcase
end

// FSM state register
always_ff @(posedge clk) begin
    if (!rst_n)
        state <= S_IDLE;
    else
        state <= state_next;
end

// Bias load counter
always_ff @(posedge clk) begin
    if (!rst_n || state == S_IDLE)
        bias_cnt <= 7'b0;
    else if (state == S_BIAS_LOAD && bias_valid)
        bias_cnt <= bias_cnt + 1;
end

// Compute counter
always_ff @(posedge clk) begin
    if (!rst_n || state != S_COMPUTE)
        compute_cnt <= 16'b0;
    else if (input_valid)
        compute_cnt <= compute_cnt + 1;
end

// Requant pipeline counter
always_ff @(posedge clk) begin
    if (!rst_n || state != S_REQUANT)
        requant_cnt <= 2'b0;
    else
        requant_cnt <= requant_cnt + 1;
end

// Accumulator logic
always_ff @(posedge clk) begin
    if (!rst_n) begin
        for (int i = 0; i < PARALLELISM; i++)
            acc[i] <= {ACC_W{1'b0}};
    end else if (state == S_BIAS_LOAD && bias_valid) begin
        // Load bias into accumulator slot addressed by bias_idx
        acc[bias_idx] <= bias_data;
    end else if (state == S_COMPUTE && input_valid) begin
        // MAC: acc[i] += input_data * weight[i]
        for (int i = 0; i < PARALLELISM; i++) begin
            automatic logic signed [DATA_W-1:0] w = $signed(weight_data[i*DATA_W +: DATA_W]);
            acc[i] <= acc[i] + ($signed(input_data) * w);
        end
    end
end

// Requantize trigger
assign rq_valid_in = (state == S_REQUANT && requant_cnt == 0);

// Output stage
always_ff @(posedge clk) begin
    if (!rst_n) begin
        output_valid <= 1'b0;
        for (int i = 0; i < PARALLELISM; i++)
            output_data[i] <= {DATA_W{1'b0}};
    end else if (rq_valid_out) begin
        output_valid <= 1'b1;
        for (int i = 0; i < PARALLELISM; i++)
            output_data[i] <= rq_data_out[i];
    end else begin
        output_valid <= 1'b0;
    end
end

// Done signal
assign done = (state == S_DONE);

endmodule
