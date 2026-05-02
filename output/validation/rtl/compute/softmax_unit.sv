// softmax_unit.sv — INT8 softmax with LUT-based exp
// 3-stage pipeline: FIND_MAX → EXP_SUM → NORMALIZE
//
// Input: SEQ_LEN × INT8 values, Output: SEQ_LEN × INT8 probabilities
// Uses 256-entry exp LUT (INT8→INT16) and fixed-point reciprocal multiply.
// Cycle count: 3 × SEQ_LEN

module softmax_unit #(
    parameter int DATA_W      = 8,
    parameter int MAX_SEQ_LEN = 1024,
    parameter int EXP_W       = 16,     // exp LUT output width
    parameter int ACC_W       = 32      // accumulator for sum of exp
) (
    input clk,
    input rst_n,

    // Control
    input start,
    output logic done,
    output logic busy,

    // Configuration
    input [15:0] seq_len,

    // Activation input (read)
    output logic [$clog2(MAX_SEQ_LEN)-1:0] act_rd_addr,
    input signed [DATA_W-1:0] act_rd_data,

    // Activation output (write)
    output logic [$clog2(MAX_SEQ_LEN)-1:0] act_wr_addr,
    output logic act_wr_en,
    output logic signed [DATA_W-1:0] act_wr_data,

    // Exp LUT ROM interface (256 entries)
    output logic [DATA_W-1:0] lut_addr,
    input [EXP_W-1:0] lut_data
);

    // ======================================================================
    // FSM
    // ======================================================================

    typedef enum logic [2:0] {
        S_IDLE,
        S_FIND_MAX,
        S_EXP_SUM,
        S_NORMALIZE,
        S_DONE
    } sm_state_t;

    sm_state_t state, state_next;

    // ======================================================================
    // Internal registers
    // ======================================================================

    logic [15:0] idx_cnt;
    logic signed [DATA_W-1:0] max_val;
    logic [ACC_W-1:0] exp_sum;

    // Register file for input values (reused in normalize pass)
    logic signed [DATA_W-1:0] input_reg [0:MAX_SEQ_LEN-1];

    // Register for per-element exp values
    logic [EXP_W-1:0] exp_reg [0:MAX_SEQ_LEN-1];

    // Reciprocal of exp_sum (fixed-point 0.16)
    logic [31:0] reciprocal;

    // ======================================================================
    // FSM next-state
    // ======================================================================

    always_comb begin
        state_next = state;
        case (state)
            S_IDLE:      if (start) state_next = S_FIND_MAX;
            S_FIND_MAX:  if (idx_cnt == seq_len - 1) state_next = S_EXP_SUM;
            S_EXP_SUM:   if (idx_cnt == seq_len - 1) state_next = S_NORMALIZE;
            S_NORMALIZE: if (idx_cnt == seq_len - 1) state_next = S_DONE;
            S_DONE:      state_next = S_IDLE;
            default:     state_next = S_IDLE;
        endcase
    end

    // ======================================================================
    // State register
    // ======================================================================

    always_ff @(posedge clk) begin
        if (!rst_n)
            state <= S_IDLE;
        else
            state <= state_next;
    end

    // ======================================================================
    // Counter
    // ======================================================================

    always_ff @(posedge clk) begin
        if (!rst_n || state == S_IDLE) begin
            idx_cnt <= 16'b0;
        end else if (state != state_next && state_next != S_DONE) begin
            // Reset counter on state transition (except to DONE)
            idx_cnt <= 16'b0;
        end else begin
            idx_cnt <= idx_cnt + 1;
        end
    end

    // ======================================================================
    // Pass 1: Find max
    // ======================================================================

    always_ff @(posedge clk) begin
        if (!rst_n || state == S_IDLE) begin
            max_val <= {1'b1, {(DATA_W-1){1'b0}}}; // -128
        end else if (state == S_FIND_MAX) begin
            input_reg[idx_cnt] <= act_rd_data;
            if (act_rd_data > max_val)
                max_val <= act_rd_data;
        end
    end

    assign act_rd_addr = idx_cnt[$clog2(MAX_SEQ_LEN)-1:0];

    // ======================================================================
    // Pass 2: Exp(x - max) sum
    // ======================================================================

    // LUT address: (input - max) offset, shifted to unsigned [0, 255]
    logic signed [DATA_W:0] shifted_val;
    assign shifted_val = $signed({1'b0, input_reg[idx_cnt]}) - $signed({1'b0, max_val});

    // Map to LUT index: unsigned byte
    assign lut_addr = shifted_val[DATA_W-1:0];

    always_ff @(posedge clk) begin
        if (!rst_n || state == S_IDLE) begin
            exp_sum <= {ACC_W{1'b0}};
        end else if (state == S_EXP_SUM) begin
            exp_reg[idx_cnt] <= lut_data;
            exp_sum <= exp_sum + {{(ACC_W-EXP_W){1'b0}}, lut_data};
        end
    end

    // ======================================================================
    // Reciprocal computation (1 / exp_sum in fixed-point)
    // ======================================================================

    // Simple approximation: reciprocal = (1 << 24) / exp_sum
    // Computed at transition to NORMALIZE
    always_ff @(posedge clk) begin
        if (state == S_EXP_SUM && state_next == S_NORMALIZE) begin
            if (exp_sum != 0)
                reciprocal <= (32'h0100_0000) / exp_sum; // Q8.16 approx
            else
                reciprocal <= 32'b0;
        end
    end

    // ======================================================================
    // Pass 3: Normalize (exp[i] * reciprocal >> shift → INT8)
    // ======================================================================

    logic [47:0] norm_product;
    assign norm_product = {16'b0, exp_reg[idx_cnt]} * reciprocal;

    // Scale to INT8 range [0, 127] — softmax outputs are non-negative
    logic signed [DATA_W-1:0] norm_result;
    assign norm_result = (norm_product[39:32] > 8'd127) ? 8'sd127 : norm_product[39:32];

    assign act_wr_addr = idx_cnt[$clog2(MAX_SEQ_LEN)-1:0];
    assign act_wr_en   = (state == S_NORMALIZE);
    assign act_wr_data = norm_result;

    // ======================================================================
    // Status
    // ======================================================================

    assign done = (state == S_DONE);
    assign busy = (state != S_IDLE && state != S_DONE);

endmodule
