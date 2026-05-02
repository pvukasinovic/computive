// pool_unit.sv — Pooling unit (MaxPool, AvgPool, GlobalAvgPool)
// POOL_TYPE selects: 0=MaxPool (comparator), 1=AvgPool (accum+divide), 2=GlobalAvgPool
// No weights, purely datapath. Sliding window address generation.

/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off WIDTHEXPAND */
module pool_unit #(
    parameter int DATA_W       = 8,
    parameter int ACC_W        = 32,
    parameter int MAX_SPATIAL  = 64,
    parameter int MAX_CHANNELS = 512,
    // POOL_TYPE: 0=MaxPool, 1=AvgPool, 2=GlobalAvgPool
    parameter int POOL_TYPE    = 0
) (
    input clk,
    input rst_n,

    // Control
    input start,
    output logic done,
    output logic busy,

    // Configuration
    input [15:0] channels,
    input [7:0] in_height,
    input [7:0] in_width,
    input [3:0] kernel_h,
    input [3:0] kernel_w,
    input [3:0] stride_h,
    input [3:0] stride_w,
    input [3:0] pad_h,
    input [3:0] pad_w,

    // Activation input (read)
    output logic [$clog2(MAX_CHANNELS*MAX_SPATIAL*MAX_SPATIAL)-1:0] act_rd_addr,
    input signed [DATA_W-1:0] act_rd_data,

    // Activation output (write)
    output logic [$clog2(MAX_CHANNELS*MAX_SPATIAL*MAX_SPATIAL)-1:0] act_wr_addr,
    output logic act_wr_en,
    output logic signed [DATA_W-1:0] act_wr_data
);

    // ======================================================================
    // FSM
    // ======================================================================

    typedef enum logic [2:0] {
        S_IDLE,
        S_POOL_WINDOW,
        S_WRITE_OUTPUT,
        S_NEXT_POS,
        S_DONE
    } pool_state_t;

    pool_state_t state, state_next;

    // ======================================================================
    // Output geometry
    // ======================================================================

    logic [7:0] out_height, out_width;

    generate
        if (POOL_TYPE == 2) begin : gen_global
            assign out_height = 8'd1;
            assign out_width  = 8'd1;
        end else begin : gen_local
            assign out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
            assign out_width  = (in_width  + 2 * pad_w - kernel_w) / stride_w + 1;
        end
    endgenerate

    // For GlobalAvgPool, kernel = input spatial dims
    logic [3:0] eff_kh, eff_kw;

    generate
        if (POOL_TYPE == 2) begin : gen_global_k
            assign eff_kh = in_height[3:0];
            assign eff_kw = in_width[3:0];
        end else begin : gen_local_k
            assign eff_kh = kernel_h;
            assign eff_kw = kernel_w;
        end
    endgenerate

    // ======================================================================
    // Counters
    // ======================================================================

    logic [15:0] ch_cnt;
    logic [7:0]  oh_cnt, ow_cnt;
    logic [3:0]  kh_cnt, kw_cnt;

    // Pool accumulator
    logic signed [ACC_W-1:0] pool_acc;
    logic signed [DATA_W-1:0] pool_max;
    logic [15:0] pool_count; // number of valid elements in window

    // Window position in input
    logic signed [8:0] ih_pos, iw_pos;
    assign ih_pos = $signed({1'b0, oh_cnt}) * $signed({1'b0, stride_h}) - $signed({1'b0, pad_h}) + $signed({1'b0, kh_cnt});
    assign iw_pos = $signed({1'b0, ow_cnt}) * $signed({1'b0, stride_w}) - $signed({1'b0, pad_w}) + $signed({1'b0, kw_cnt});

    logic is_padded;
    assign is_padded = (ih_pos < 0) || (ih_pos >= $signed({1'b0, in_height})) ||
                       (iw_pos < 0) || (iw_pos >= $signed({1'b0, in_width}));

    // ======================================================================
    // FSM next-state
    // ======================================================================

    always_comb begin
        state_next = state;
        case (state)
            S_IDLE: if (start) state_next = S_POOL_WINDOW;
            S_POOL_WINDOW: begin
                if (kh_cnt == eff_kh - 1 && kw_cnt == eff_kw - 1)
                    state_next = S_WRITE_OUTPUT;
            end
            S_WRITE_OUTPUT: state_next = S_NEXT_POS;
            S_NEXT_POS: begin
                if (oh_cnt == out_height - 1 && ow_cnt == out_width - 1 &&
                    ch_cnt == channels - 1)
                    state_next = S_DONE;
                else
                    state_next = S_POOL_WINDOW;
            end
            S_DONE: state_next = S_IDLE;
            default: state_next = S_IDLE;
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
    // Counter logic
    // ======================================================================

    always_ff @(posedge clk) begin
        if (!rst_n || state == S_IDLE) begin
            ch_cnt    <= 16'b0;
            oh_cnt    <= 8'b0;
            ow_cnt    <= 8'b0;
            kh_cnt    <= 4'b0;
            kw_cnt    <= 4'b0;
        end else if (state == S_POOL_WINDOW) begin
            if (kw_cnt == eff_kw - 1) begin
                kw_cnt <= 4'b0;
                kh_cnt <= kh_cnt + 1;
            end else begin
                kw_cnt <= kw_cnt + 1;
            end
        end else if (state == S_NEXT_POS) begin
            kh_cnt <= 4'b0;
            kw_cnt <= 4'b0;
            if (ow_cnt == out_width - 1) begin
                ow_cnt <= 8'b0;
                if (oh_cnt == out_height - 1) begin
                    oh_cnt <= 8'b0;
                    ch_cnt <= ch_cnt + 1;
                end else begin
                    oh_cnt <= oh_cnt + 1;
                end
            end else begin
                ow_cnt <= ow_cnt + 1;
            end
        end
    end

    // ======================================================================
    // Pool computation
    // ======================================================================

    always_ff @(posedge clk) begin
        if (!rst_n || state == S_NEXT_POS || state == S_IDLE) begin
            pool_acc   <= {ACC_W{1'b0}};
            pool_max   <= {1'b1, {(DATA_W-1){1'b0}}}; // -128
            pool_count <= 16'b0;
        end else if (state == S_POOL_WINDOW && !is_padded) begin
            // MaxPool: track maximum
            if (act_rd_data > pool_max)
                pool_max <= act_rd_data;
            // AvgPool: accumulate
            pool_acc   <= pool_acc + {{(ACC_W-DATA_W){act_rd_data[DATA_W-1]}}, act_rd_data};
            pool_count <= pool_count + 1;
        end
    end

    // ======================================================================
    // Read address: NCHW linearized
    // ======================================================================

    logic [31:0] rd_linear;
    assign rd_linear = {16'b0, ch_cnt} * {24'b0, in_height} * {24'b0, in_width} +
                       {24'b0, ih_pos[7:0]} * {24'b0, in_width} +
                       {24'b0, iw_pos[7:0]};
    assign act_rd_addr = rd_linear[$clog2(MAX_CHANNELS*MAX_SPATIAL*MAX_SPATIAL)-1:0];

    // ======================================================================
    // Write output
    // ======================================================================

    logic [31:0] wr_linear;
    assign wr_linear = {16'b0, ch_cnt} * {24'b0, out_height} * {24'b0, out_width} +
                       {24'b0, oh_cnt} * {24'b0, out_width} +
                       {24'b0, ow_cnt};
    assign act_wr_addr = wr_linear[$clog2(MAX_CHANNELS*MAX_SPATIAL*MAX_SPATIAL)-1:0];
    assign act_wr_en   = (state == S_WRITE_OUTPUT);

    // Output value based on pool type
    logic signed [ACC_W-1:0] avg_result;
    assign avg_result = (pool_count != 0) ? pool_acc / $signed({1'b0, pool_count}) : '0;

    generate
        if (POOL_TYPE == 0) begin : gen_max_out
            assign act_wr_data = pool_max;
        end else begin : gen_avg_out
            // Clamp avg to INT8
            assign act_wr_data = (avg_result > 127) ? 8'sd127 :
                                 (avg_result < -128) ? -8'sd128 :
                                 avg_result[DATA_W-1:0];
        end
    endgenerate

    // ======================================================================
    // Status
    // ======================================================================

    assign done = (state == S_DONE);
    assign busy = (state != S_IDLE && state != S_DONE);

endmodule
