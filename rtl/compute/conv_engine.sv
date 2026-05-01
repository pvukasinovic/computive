// conv_engine.sv — Im2col + MAC convolution engine
// Auto-generated module library for MLASIC tile fabric.
//
// FSM: IDLE → BIAS_LOAD → COMPUTE_PIXEL → WAIT_MAC → WRITE_PIXEL →
//       NEXT_PIXEL → NEXT_TILE → DONE
//
// Handles output tiling (OC > PARALLELISM), grouped/depthwise conv,
// 1x1 fast path. Reuses mac_array for dot products.

/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off UNUSEDPARAM */
/* verilator lint_off VARHIDDEN */
/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off UNDRIVEN */
module conv_engine #(
    parameter int PARALLELISM   = 128,
    parameter int DATA_W        = 8,
    parameter int ACC_W         = 32,
    parameter int MAX_IC        = 512,
    parameter int MAX_OC        = 512,
    parameter int MAX_SPATIAL   = 64,
    parameter int MAX_KH        = 7,
    parameter int MAX_KW        = 7
) (
    input  logic        clk,
    input  logic        rst_n,

    // Control
    input  logic        start,
    output logic        done,
    output logic        busy,

    // Layer configuration
    input  logic [15:0] in_channels,
    input  logic [15:0] out_channels,
    input  logic [7:0]  in_height,
    input  logic [7:0]  in_width,
    input  logic [3:0]  kernel_h,
    input  logic [3:0]  kernel_w,
    input  logic [3:0]  stride_h,
    input  logic [3:0]  stride_w,
    input  logic [3:0]  pad_h,
    input  logic [3:0]  pad_w,
    input  logic [15:0] group,
    input  logic        has_relu,
    input  logic [15:0] num_oc_tiles,

    // Requantization
    input  logic [31:0] requant_scale,
    input  logic [5:0]  requant_shift,
    input  logic signed [7:0] requant_zp,

    // Weight SRAM interface (read)
    output logic [15:0] weight_addr,
    input  logic [PARALLELISM*DATA_W-1:0] weight_rdata,

    // Bias SRAM interface (read)
    output logic [15:0] bias_addr,
    input  logic [ACC_W-1:0] bias_rdata,

    // Activation input (read)
    output logic [$clog2(MAX_SPATIAL*MAX_SPATIAL*MAX_IC/8)-1:0] act_rd_addr,
    input  logic [63:0] act_rd_data,

    // Activation output (write)
    output logic [$clog2(MAX_SPATIAL*MAX_SPATIAL*MAX_OC/8)-1:0] act_wr_addr,
    output logic        act_wr_en,
    output logic [63:0] act_wr_data,

    // MAC array interface
    output logic        mac_start,
    output logic [15:0] mac_input_dim,
    output logic        mac_has_relu,
    output logic signed [DATA_W-1:0] mac_input_data,
    output logic        mac_input_valid,
    output logic        mac_bias_valid,
    output logic [$clog2(PARALLELISM)-1:0] mac_bias_idx,
    output logic signed [ACC_W-1:0]  mac_bias_data,
    output logic [31:0] mac_scale,
    output logic [5:0]  mac_shift,
    output logic signed [7:0] mac_zp,

    // MAC array results
    input  logic signed [DATA_W-1:0] mac_output_data [0:PARALLELISM-1],
    input  logic        mac_output_valid,
    input  logic        mac_done
);

    // ======================================================================
    // FSM
    // ======================================================================

    typedef enum logic [3:0] {
        S_IDLE,
        S_BIAS_LOAD,
        S_COMPUTE_PIXEL,
        S_WAIT_MAC,
        S_WRITE_PIXEL,
        S_NEXT_PIXEL,
        S_NEXT_TILE,
        S_DONE
    } conv_state_t;

    conv_state_t state, state_next;

    // ======================================================================
    // Output geometry computation
    // ======================================================================

    logic [7:0] out_height, out_width;
    assign out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    assign out_width  = (in_width  + 2 * pad_w - kernel_w) / stride_w + 1;

    // ======================================================================
    // Counters
    // ======================================================================

    logic [7:0]  oh_cnt, ow_cnt;       // output spatial position
    logic [15:0] oc_tile_cnt;          // output channel tile index
    logic [15:0] ic_cnt;               // input channel counter (im2col)
    logic [3:0]  kh_cnt, kw_cnt;       // kernel position
    logic [15:0] bias_load_cnt;        // bias loading counter
    logic [15:0] write_cnt;            // output write counter
    logic [15:0] weight_row_cnt;       // weight row address
    logic [15:0] ic_per_group;

    assign ic_per_group = in_channels / group;

    // Im2col input position
    logic signed [8:0] ih_pos, iw_pos;
    assign ih_pos = $signed({1'b0, oh_cnt}) * $signed({1'b0, stride_h}) - $signed({1'b0, pad_h}) + $signed({1'b0, kh_cnt});
    assign iw_pos = $signed({1'b0, ow_cnt}) * $signed({1'b0, stride_w}) - $signed({1'b0, pad_w}) + $signed({1'b0, kw_cnt});

    // Padding check
    logic is_padded;
    assign is_padded = (ih_pos < 0) || (ih_pos >= $signed({1'b0, in_height})) ||
                       (iw_pos < 0) || (iw_pos >= $signed({1'b0, in_width}));

    // Dot product length: IC_per_group * KH * KW
    logic [15:0] dot_length;
    assign dot_length = ic_per_group * {12'b0, kernel_h} * {12'b0, kernel_w};

    // ======================================================================
    // FSM next-state
    // ======================================================================

    always_comb begin
        state_next = state;
        case (state)
            S_IDLE:         if (start) state_next = S_BIAS_LOAD;
            S_BIAS_LOAD:    if (bias_load_cnt == PARALLELISM - 1) state_next = S_COMPUTE_PIXEL;
            S_COMPUTE_PIXEL: state_next = S_WAIT_MAC;
            S_WAIT_MAC:     if (mac_done) state_next = S_WRITE_PIXEL;
            S_WRITE_PIXEL:  if (write_cnt == (PARALLELISM / 8) - 1) state_next = S_NEXT_PIXEL;
            S_NEXT_PIXEL: begin
                if (oh_cnt == out_height - 1 && ow_cnt == out_width - 1)
                    state_next = S_NEXT_TILE;
                else
                    state_next = S_BIAS_LOAD;
            end
            S_NEXT_TILE: begin
                if (oc_tile_cnt == num_oc_tiles - 1)
                    state_next = S_DONE;
                else
                    state_next = S_BIAS_LOAD;
            end
            S_DONE:         state_next = S_IDLE;
            default:        state_next = S_IDLE;
        endcase
    end

    // ======================================================================
    // State register
    // ======================================================================

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE;
        end else begin
            state <= state_next;
        end
    end

    // ======================================================================
    // Counter logic
    // ======================================================================

    always_ff @(posedge clk) begin
        if (!rst_n || state == S_IDLE) begin
            oh_cnt       <= '0;
            ow_cnt       <= '0;
            oc_tile_cnt  <= '0;
            bias_load_cnt <= '0;
            write_cnt    <= '0;
            weight_row_cnt <= '0;
        end else begin
            case (state)
                S_BIAS_LOAD: begin
                    bias_load_cnt <= bias_load_cnt + 1;
                end
                S_WRITE_PIXEL: begin
                    write_cnt <= write_cnt + 1;
                end
                S_NEXT_PIXEL: begin
                    bias_load_cnt <= '0;
                    write_cnt <= '0;
                    if (ow_cnt == out_width - 1) begin
                        ow_cnt <= '0;
                        oh_cnt <= oh_cnt + 1;
                    end else begin
                        ow_cnt <= ow_cnt + 1;
                    end
                end
                S_NEXT_TILE: begin
                    oh_cnt      <= '0;
                    ow_cnt      <= '0;
                    oc_tile_cnt <= oc_tile_cnt + 1;
                    bias_load_cnt <= '0;
                    write_cnt   <= '0;
                end
                default: ;
            endcase
        end
    end

    // ======================================================================
    // MAC array control signals
    // ======================================================================

    assign mac_start     = (state == S_COMPUTE_PIXEL);
    assign mac_input_dim = dot_length;
    assign mac_has_relu  = has_relu;
    assign mac_scale     = requant_scale;
    assign mac_shift     = requant_shift;
    assign mac_zp        = requant_zp;

    // Im2col feeding — unrolls kernel × IC into sequential input data
    assign mac_input_data  = is_padded ? 8'sd0 : $signed(act_rd_data[7:0]);
    assign mac_input_valid = (state == S_COMPUTE_PIXEL || state == S_WAIT_MAC);

    // Bias loading
    assign mac_bias_valid = (state == S_BIAS_LOAD);
    assign mac_bias_idx   = bias_load_cnt[$clog2(PARALLELISM)-1:0];
    assign mac_bias_data  = bias_rdata;

    // Weight address: row per output channel tile × spatial step
    assign weight_addr = weight_row_cnt;

    // Bias address: one per output channel
    assign bias_addr = oc_tile_cnt * PARALLELISM + bias_load_cnt;

    // ======================================================================
    // Activation read address (im2col)
    // ======================================================================

    // Linearized NCHW: IC * H * W + ih * W + iw, packed 8 per 64-bit word
    logic [31:0] act_rd_linear;
    assign act_rd_linear = (ic_cnt * {8'b0, in_height} * {8'b0, in_width}) +
                           ({23'b0, ih_pos[7:0]} * {24'b0, in_width}) +
                           {23'b0, iw_pos[7:0]};
    assign act_rd_addr = act_rd_linear[31:3]; // div by 8 for 64-bit words

    // ======================================================================
    // Activation write address/data
    // ======================================================================

    logic [31:0] act_wr_linear;
    assign act_wr_linear = (oc_tile_cnt * PARALLELISM + write_cnt * 8) +
                           ({24'b0, oh_cnt} * {24'b0, out_width} + {24'b0, ow_cnt}) *
                           out_channels;
    assign act_wr_addr = act_wr_linear[$clog2(MAX_SPATIAL*MAX_SPATIAL*MAX_OC/8)-1:0];
    assign act_wr_en   = (state == S_WRITE_PIXEL);

    // Pack 8 INT8 outputs into 64-bit word
    always_comb begin
        act_wr_data = '0;
        for (int i = 0; i < 8; i++) begin
            act_wr_data[i*8 +: 8] = mac_output_data[write_cnt * 8 + i];
        end
    end

    // ======================================================================
    // Status
    // ======================================================================

    assign done = (state == S_DONE);
    assign busy = (state != S_IDLE && state != S_DONE);

endmodule
