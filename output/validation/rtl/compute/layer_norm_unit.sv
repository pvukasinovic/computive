// layer_norm_unit.sv — INT8 LayerNorm with 2-pass computation
// Pass 1: COMPUTE_STATS (mean/variance via INT32 accumulators)
// Pass 2: NORMALIZE (reciprocal-sqrt LUT, scale/bias application)
//
// Scale/bias from per-tile ROM. Cycle count: 2 × FEATURE_DIM + 10

module layer_norm_unit #(
    parameter int DATA_W        = 8,
    parameter int ACC_W         = 32,
    parameter int MAX_FEAT_DIM  = 1024,
    parameter int RSQRT_LUT_W   = 16    // reciprocal-sqrt LUT output width
) (
    input clk,
    input rst_n,

    // Control
    input start,
    output logic done,
    output logic busy,

    // Configuration
    input [15:0] feature_dim,
    input [31:0] epsilon_fixed,  // epsilon in fixed-point (unused in LUT path)

    // Activation input (read)
    output logic [$clog2(MAX_FEAT_DIM)-1:0] act_rd_addr,
    input signed [DATA_W-1:0] act_rd_data,

    // Activation output (write)
    output logic [$clog2(MAX_FEAT_DIM)-1:0] act_wr_addr,
    output logic act_wr_en,
    output logic signed [DATA_W-1:0] act_wr_data,

    // Scale/bias ROM interface
    output logic [$clog2(MAX_FEAT_DIM)-1:0] scale_addr,
    input signed [DATA_W-1:0] scale_data,
    output logic [$clog2(MAX_FEAT_DIM)-1:0] bias_addr,
    input signed [DATA_W-1:0] bias_data,

    // Reciprocal-sqrt LUT ROM interface
    output logic [DATA_W-1:0] rsqrt_lut_addr,
    input [RSQRT_LUT_W-1:0] rsqrt_lut_data
);

    // ======================================================================
    // FSM
    // ======================================================================

    typedef enum logic [2:0] {
        S_IDLE,
        S_COMPUTE_MEAN,
        S_COMPUTE_VAR,
        S_LOOKUP_RSQRT,
        S_NORMALIZE,
        S_DONE
    } ln_state_t;

    ln_state_t state, state_next;

    // ======================================================================
    // Internal registers
    // ======================================================================

    logic [15:0] idx_cnt;
    logic signed [ACC_W-1:0] sum_acc;       // sum of x for mean
    logic signed [ACC_W-1:0] var_acc;       // sum of (x - mean)^2
    logic signed [ACC_W-1:0] mean_val;      // mean = sum / N
    logic [RSQRT_LUT_W-1:0] rsqrt_val;     // 1/sqrt(var + eps)

    // Input register file for 2-pass
    logic signed [DATA_W-1:0] input_reg [0:MAX_FEAT_DIM-1];

    // ======================================================================
    // FSM next-state
    // ======================================================================

    always_comb begin
        state_next = state;
        case (state)
            S_IDLE:          if (start) state_next = S_COMPUTE_MEAN;
            S_COMPUTE_MEAN:  if (idx_cnt == feature_dim - 1) state_next = S_COMPUTE_VAR;
            S_COMPUTE_VAR:   if (idx_cnt == feature_dim - 1) state_next = S_LOOKUP_RSQRT;
            S_LOOKUP_RSQRT:  state_next = S_NORMALIZE;  // 1-cycle LUT
            S_NORMALIZE:     if (idx_cnt == feature_dim - 1) state_next = S_DONE;
            S_DONE:          state_next = S_IDLE;
            default:         state_next = S_IDLE;
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
        end else if (state != state_next && state_next != S_DONE && state_next != S_LOOKUP_RSQRT) begin
            idx_cnt <= 16'b0;
        end else if (state != S_LOOKUP_RSQRT) begin
            idx_cnt <= idx_cnt + 1;
        end
    end

    // ======================================================================
    // Pass 1a: Compute mean
    // ======================================================================

    always_ff @(posedge clk) begin
        if (!rst_n || state == S_IDLE) begin
            sum_acc <= {ACC_W{1'b0}};
        end else if (state == S_COMPUTE_MEAN) begin
            input_reg[idx_cnt] <= act_rd_data;
            sum_acc <= sum_acc + {{(ACC_W-DATA_W){act_rd_data[DATA_W-1]}}, act_rd_data};
        end
    end

    assign act_rd_addr = idx_cnt[$clog2(MAX_FEAT_DIM)-1:0];

    // Mean computation (at transition to variance pass)
    always_ff @(posedge clk) begin
        if (state == S_COMPUTE_MEAN && state_next == S_COMPUTE_VAR) begin
            mean_val <= sum_acc / $signed({1'b0, feature_dim});
        end
    end

    // ======================================================================
    // Pass 1b: Compute variance
    // ======================================================================

    logic signed [ACC_W-1:0] diff;
    assign diff = {{(ACC_W-DATA_W){input_reg[idx_cnt][DATA_W-1]}}, input_reg[idx_cnt]} - mean_val;

    logic signed [2*ACC_W-1:0] diff_sq;
    assign diff_sq = diff * diff;

    always_ff @(posedge clk) begin
        if (!rst_n || state == S_COMPUTE_MEAN) begin
            var_acc <= {ACC_W{1'b0}};
        end else if (state == S_COMPUTE_VAR) begin
            var_acc <= var_acc + diff_sq[ACC_W-1:0];
        end
    end

    // ======================================================================
    // Reciprocal-sqrt LUT lookup
    // ======================================================================

    // Quantize variance to 8-bit LUT index
    logic [ACC_W-1:0] variance;
    assign variance = var_acc / {16'b0, feature_dim};

    // Map variance to LUT index (saturating)
    assign rsqrt_lut_addr = (variance[ACC_W-1:8] != '0) ? 8'hFF : variance[7:0];

    always_ff @(posedge clk) begin
        if (state == S_LOOKUP_RSQRT) begin
            rsqrt_val <= rsqrt_lut_data;
        end
    end

    // ======================================================================
    // Pass 2: Normalize
    // ======================================================================

    // normalized = (x - mean) * rsqrt * scale + bias
    logic signed [ACC_W-1:0] norm_diff;
    assign norm_diff = {{(ACC_W-DATA_W){input_reg[idx_cnt][DATA_W-1]}}, input_reg[idx_cnt]} - mean_val;

    logic signed [2*ACC_W-1:0] norm_product;
    assign norm_product = norm_diff * $signed({1'b0, rsqrt_val});

    // Apply scale (INT8) and bias (INT8)
    logic signed [ACC_W-1:0] scaled;
    assign scaled = (norm_product[ACC_W+RSQRT_LUT_W-2:RSQRT_LUT_W-1]) *
                    {{(ACC_W-DATA_W){scale_data[DATA_W-1]}}, scale_data};

    logic signed [ACC_W-1:0] biased;
    assign biased = scaled[ACC_W-1:0] + {{(ACC_W-DATA_W){bias_data[DATA_W-1]}}, bias_data};

    // Clamp to INT8
    logic signed [DATA_W-1:0] clamped;
    assign clamped = (biased > 127) ? 8'sd127 :
                     (biased < -128) ? -8'sd128 :
                     biased[DATA_W-1:0];

    assign scale_addr = idx_cnt[$clog2(MAX_FEAT_DIM)-1:0];
    assign bias_addr  = idx_cnt[$clog2(MAX_FEAT_DIM)-1:0];
    assign act_wr_addr = idx_cnt[$clog2(MAX_FEAT_DIM)-1:0];
    assign act_wr_en   = (state == S_NORMALIZE);
    assign act_wr_data = clamped;

    // ======================================================================
    // Status
    // ======================================================================

    assign done = (state == S_DONE);
    assign busy = (state != S_IDLE && state != S_DONE);

endmodule
