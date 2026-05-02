// tile.sv — Generic tile wrapper for MLASIC tile fabric
//
// Wraps a compute unit + local weight ROM + requantize + I/O interface.
// TILE_TYPE selects which compute unit to instantiate:
//   0 = MAC (fused_linear_relu + mac_array)
//   1 = ALU (elementwise passthrough)
//   2 = NORM (layer_norm_unit + scale/bias ROM)
//   3 = SOFTMAX (softmax_unit + exp LUT)
//   4 = ACTIVATION (activation_unit + optional LUT)
//   5 = POOL (pool_unit)
//   6 = RESHAPE (zero-cost passthrough)

// Multi-type tile — some parameters/signals are only used for specific TILE_TYPE
// configurations. Suppress warnings from unused paths in the generate blocks.
/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off UNUSEDPARAM */
/* verilator lint_off VARHIDDEN */
/* verilator lint_off PINCONNECTEMPTY */
/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off SELRANGE */
/* verilator lint_off UNDRIVEN */
module tile #(
    parameter int TILE_ID       = 0,
    parameter int TILE_TYPE     = 0,    // see encoding above
    parameter int PARALLELISM   = 128,
    parameter int DATA_W        = 8,
    parameter int ACC_W         = 32,
    parameter int WEIGHT_DEPTH  = 1024,
    parameter int BIAS_DEPTH    = 128,
    parameter int ACT_DEPTH     = 1024,
    parameter int ACT_WIDTH     = 64,
    // Compute parameters
    parameter int INPUT_DIM     = 128,
    parameter int OUTPUT_DIM    = 128,
    parameter int NUM_TILES     = 1,
    parameter int HAS_RELU      = 0,
    // Requantization
    parameter int REQUANT_M     = 65536,
    parameter int REQUANT_SHIFT = 16,
    parameter int REQUANT_ZP    = 0,
    // Weight ROM init file (FPGA path)
    parameter string WEIGHT_INIT_FILE = "",
    parameter string BIAS_INIT_FILE   = "",
    // Activation LUT (for GELU/SiLU/softmax)
    parameter string LUT_INIT_FILE    = "",
    // Per-channel requant ROM init file (for conv tiles)
    parameter string REQUANT_INIT_FILE = "",
    // Conv-specific (TILE_TYPE=0 with conv mode)
    parameter int IS_CONV       = 0,
    parameter int IN_CHANNELS   = 1,
    parameter int OUT_CHANNELS  = 1,
    parameter int IN_HEIGHT     = 1,
    parameter int IN_WIDTH      = 1,
    parameter int KERNEL_H      = 1,
    parameter int KERNEL_W      = 1,
    parameter int STRIDE_H      = 1,
    parameter int STRIDE_W      = 1,
    parameter int PAD_H         = 0,
    parameter int PAD_W         = 0,
    parameter int GROUP         = 1,
    // Pool-specific (TILE_TYPE=5)
    parameter int POOL_TYPE     = 0,
    // Activation-specific (TILE_TYPE=4)
    parameter int ACT_TYPE      = 0
) (
    input clk,
    input rst_n,

    // Tile control
    input start,
    output logic done,
    output logic busy,

    // Activation input (read port)
    input [ACT_WIDTH-1:0] act_in_data,
    output logic [$clog2(ACT_DEPTH)-1:0] act_in_addr,

    // Activation output (write port)
    output logic [ACT_WIDTH-1:0] act_out_data,
    output logic [$clog2(ACT_DEPTH)-1:0] act_out_addr,
    output logic act_out_we,

    // Second activation input (for residual add — ALU tile only)
    input [ACT_WIDTH-1:0] act_skip_data,
    output logic [$clog2(ACT_DEPTH)-1:0] act_skip_addr
);

    // ======================================================================
    // Internal signals
    // ======================================================================

    logic [15:0] weight_addr;
    logic [PARALLELISM*DATA_W-1:0] weight_rdata;
    logic [15:0] bias_addr;
    logic [ACC_W-1:0] bias_rdata;

    // ======================================================================
    // Weight ROM (FPGA path — uses sram_bank with $readmemh)
    // Only instantiated for tiles that have weights
    // ======================================================================

    generate
        if (TILE_TYPE == 0 && WEIGHT_DEPTH > 0) begin : gen_weight_rom
            sram_bank #(
                .DEPTH(WEIGHT_DEPTH),
                .WIDTH(PARALLELISM * DATA_W),
                .INIT_FILE(WEIGHT_INIT_FILE)
            ) u_weight_rom (
                .clk(clk),
                .addr(weight_addr[$clog2(WEIGHT_DEPTH)-1:0]),
                .we(1'b0),
                .wdata({(PARALLELISM*DATA_W){1'b0}}),
                .rdata(weight_rdata)
            );

            if (BIAS_DEPTH > 0) begin : gen_bias_rom
                sram_bank #(
                    .DEPTH(BIAS_DEPTH),
                    .WIDTH(ACC_W),
                    .INIT_FILE(BIAS_INIT_FILE)
                ) u_bias_rom (
                    .clk(clk),
                    .addr(bias_addr[$clog2(BIAS_DEPTH)-1:0]),
                    .we(1'b0),
                    .wdata({ACC_W{1'b0}}),
                    .rdata(bias_rdata)
                );
            end else begin : gen_no_bias
                assign bias_rdata = {ACC_W{1'b0}};
            end
        end
    endgenerate

    // ======================================================================
    // RESHAPE tile (zero-cost passthrough)
    // ======================================================================

    generate
        if (TILE_TYPE == 6) begin : gen_reshape
            // Zero-cost: wire input directly to output
            assign act_out_data = act_in_data;
            assign act_out_addr = act_in_addr;
            assign act_out_we   = 1'b0;
            assign done = start; // Instant completion
            assign busy = 1'b0;
            assign act_in_addr = {$clog2(ACT_DEPTH){1'b0}};
        end
    endgenerate

    // ======================================================================
    // MAC tile — IS_CONV=0: fused_linear_relu (MLP), IS_CONV=1: conv_engine (CNN)
    // ======================================================================

    generate
        if (TILE_TYPE == 0 && IS_CONV == 0) begin : gen_mac_linear
            // ---- MLP path (backward compatible) ----
            logic mac_start;
            logic [15:0] mac_input_dim;
            logic mac_has_relu;
            logic signed [DATA_W-1:0] mac_input_data;
            logic mac_input_valid;
            logic mac_bias_valid;
            logic [$clog2(PARALLELISM)-1:0] mac_bias_idx;
            logic signed [ACC_W-1:0] mac_bias_data;
            logic [31:0] mac_scale;
            logic [5:0] mac_shift;
            logic signed [7:0] mac_zp;
            logic signed [DATA_W-1:0] mac_output_data [0:PARALLELISM-1];
            logic mac_output_valid, mac_done;

            mac_array #(
                .PARALLELISM(PARALLELISM),
                .DATA_W(DATA_W),
                .ACC_W(ACC_W),
                .PER_CHANNEL(0)
            ) u_mac (
                .clk(clk), .rst_n(rst_n),
                .start(mac_start),
                .input_dim(mac_input_dim),
                .has_relu(mac_has_relu),
                .weight_data(weight_rdata),
                .weight_valid(1'b1),
                .input_data(mac_input_data),
                .input_valid(mac_input_valid),
                .bias_data(mac_bias_data),
                .bias_valid(mac_bias_valid),
                .bias_idx(mac_bias_idx),
                .scale(mac_scale),
                .shift(mac_shift),
                .zero_point(mac_zp),
                .scale_bus({(PARALLELISM*ACC_W){1'b0}}),
                .shift_bus({(PARALLELISM*6){1'b0}}),
                .zp_bus({(PARALLELISM*DATA_W){1'b0}}),
                .output_data(mac_output_data),
                .output_valid(mac_output_valid),
                .done(mac_done)
            );

            fused_linear_relu #(
                .PARALLELISM(PARALLELISM),
                .DATA_W(DATA_W),
                .ACC_W(ACC_W),
                .ACT_DEPTH(ACT_DEPTH),
                .ACT_WIDTH(ACT_WIDTH)
            ) u_layer_ctrl (
                .clk(clk), .rst_n(rst_n),
                .start(start),
                .input_dim(INPUT_DIM[15:0]),
                .output_dim(OUTPUT_DIM[15:0]),
                .num_tiles(NUM_TILES[15:0]),
                .has_relu(HAS_RELU[0]),
                .weight_base(16'd0),
                .bias_base(16'd0),
                .requant_scale(REQUANT_M[31:0]),
                .requant_shift(REQUANT_SHIFT[5:0]),
                .requant_zp(REQUANT_ZP[7:0]),
                .weight_addr(weight_addr),
                .weight_rdata(weight_rdata),
                .bias_addr(bias_addr),
                .bias_rdata({PARALLELISM{bias_rdata}}),
                .act_rd_addr(act_in_addr[$clog2(ACT_DEPTH)-1:0]),
                .act_rd_data(act_in_data),
                .act_wr_addr(act_out_addr[$clog2(ACT_DEPTH)-1:0]),
                .act_wr_en(act_out_we),
                .act_wr_data(act_out_data),
                .mac_start(mac_start),
                .mac_input_dim(mac_input_dim),
                .mac_has_relu(mac_has_relu),
                .mac_scale(mac_scale),
                .mac_shift(mac_shift),
                .mac_zp(mac_zp),
                .mac_input_data(mac_input_data),
                .mac_input_valid(mac_input_valid),
                .mac_bias_valid(mac_bias_valid),
                .mac_bias_idx(mac_bias_idx),
                .mac_bias_data(mac_bias_data),
                .mac_output_data(mac_output_data),
                .mac_output_valid(mac_output_valid),
                .mac_done(mac_done),
                .layer_done(done)
            );

            logic mac_busy_r;
            always_ff @(posedge clk) begin
                if (!rst_n)
                    mac_busy_r <= 1'b0;
                else if (start)
                    mac_busy_r <= 1'b1;
                else if (done)
                    mac_busy_r <= 1'b0;
            end
            assign busy = mac_busy_r;
        end

        if (TILE_TYPE == 0 && IS_CONV == 1) begin : gen_mac_conv
            // ---- Conv path: conv_engine + mac_array(PER_CHANNEL=1) ----

            // Per-channel requant ROM: one 32-bit M_fixed per output channel
            localparam int REQUANT_ROM_DEPTH = (OUT_CHANNELS > 0) ? OUT_CHANNELS : 1;
            logic [ACC_W-1:0] requant_rom_rdata;
            logic [15:0] requant_rom_addr;

            sram_bank #(
                .DEPTH(REQUANT_ROM_DEPTH),
                .WIDTH(ACC_W),
                .INIT_FILE(REQUANT_INIT_FILE)
            ) u_requant_rom (
                .clk(clk),
                .addr(requant_rom_addr[$clog2(REQUANT_ROM_DEPTH)-1:0]),
                .we(1'b0),
                .wdata({ACC_W{1'b0}}),
                .rdata(requant_rom_rdata)
            );

            // Per-channel requant bus construction
            // For conv, the conv_engine feeds scalar requant_scale per tile,
            // but we extend to per-channel via the ROM.
            // The requant ROM stores per-OC M_fixed. During requant, all 128
            // lanes within a tile share the same shift/zp but have per-lane scale.
            logic [PARALLELISM*ACC_W-1:0] pc_scale_bus;
            logic [PARALLELISM*6-1:0]     pc_shift_bus;
            logic [PARALLELISM*DATA_W-1:0] pc_zp_bus;

            // For now, per-channel scale is loaded from ROM by conv_engine,
            // broadcast to all lanes (per-tile, not per-lane within tile).
            // Full per-lane support requires tile-level output channel tracking.
            // In v0.2 we replicate the scalar per-tile values to all lanes.
            logic [31:0] conv_mac_scale;
            logic [5:0]  conv_mac_shift;
            logic signed [7:0] conv_mac_zp;

            // Replicate scalar to bus (conv_engine sets per-OC-tile scalar)
            always_comb begin
                for (int i = 0; i < PARALLELISM; i++) begin
                    pc_scale_bus[i*ACC_W +: ACC_W] = conv_mac_scale;
                    pc_shift_bus[i*6 +: 6]         = conv_mac_shift;
                    pc_zp_bus[i*DATA_W +: DATA_W]  = conv_mac_zp;
                end
            end

            // MAC array signals
            logic mac_start;
            logic [15:0] mac_input_dim;
            logic mac_has_relu;
            logic signed [DATA_W-1:0] mac_input_data;
            logic mac_input_valid;
            logic mac_bias_valid;
            logic [$clog2(PARALLELISM)-1:0] mac_bias_idx;
            logic signed [ACC_W-1:0] mac_bias_data;
            logic signed [DATA_W-1:0] mac_output_data [0:PARALLELISM-1];
            logic mac_output_valid, mac_done;

            // OC tiles = ceil(OUT_CHANNELS / PARALLELISM)
            localparam int NUM_OC_TILES = (OUT_CHANNELS + PARALLELISM - 1) / PARALLELISM;

            mac_array #(
                .PARALLELISM(PARALLELISM),
                .DATA_W(DATA_W),
                .ACC_W(ACC_W),
                .PER_CHANNEL(1)
            ) u_mac (
                .clk(clk), .rst_n(rst_n),
                .start(mac_start),
                .input_dim(mac_input_dim),
                .has_relu(mac_has_relu),
                .weight_data(weight_rdata),
                .weight_valid(1'b1),
                .input_data(mac_input_data),
                .input_valid(mac_input_valid),
                .bias_data(mac_bias_data),
                .bias_valid(mac_bias_valid),
                .bias_idx(mac_bias_idx),
                .scale(32'd0),
                .shift(6'd0),
                .zero_point(8'sd0),
                .scale_bus(pc_scale_bus),
                .shift_bus(pc_shift_bus),
                .zp_bus(pc_zp_bus),
                .output_data(mac_output_data),
                .output_valid(mac_output_valid),
                .done(mac_done)
            );

            conv_engine #(
                .PARALLELISM(PARALLELISM),
                .DATA_W(DATA_W),
                .ACC_W(ACC_W)
            ) u_conv_ctrl (
                .clk(clk), .rst_n(rst_n),
                .start(start),
                .done(done),
                .busy(busy),
                .in_channels(IN_CHANNELS[15:0]),
                .out_channels(OUT_CHANNELS[15:0]),
                .in_height(IN_HEIGHT[7:0]),
                .in_width(IN_WIDTH[7:0]),
                .kernel_h(KERNEL_H[3:0]),
                .kernel_w(KERNEL_W[3:0]),
                .stride_h(STRIDE_H[3:0]),
                .stride_w(STRIDE_W[3:0]),
                .pad_h(PAD_H[3:0]),
                .pad_w(PAD_W[3:0]),
                .group(GROUP[15:0]),
                .has_relu(HAS_RELU[0]),
                .num_oc_tiles(NUM_OC_TILES[15:0]),
                .requant_scale(requant_rom_rdata),
                .requant_shift(REQUANT_SHIFT[5:0]),
                .requant_zp(REQUANT_ZP[7:0]),
                .weight_addr(weight_addr),
                .weight_rdata(weight_rdata),
                .bias_addr(bias_addr),
                .bias_rdata(bias_rdata),
                .act_rd_addr(act_in_addr[$clog2(ACT_DEPTH)-1:0]),
                .act_rd_data(act_in_data),
                .act_wr_addr(act_out_addr[$clog2(ACT_DEPTH)-1:0]),
                .act_wr_en(act_out_we),
                .act_wr_data(act_out_data),
                .mac_start(mac_start),
                .mac_input_dim(mac_input_dim),
                .mac_has_relu(mac_has_relu),
                .mac_input_data(mac_input_data),
                .mac_input_valid(mac_input_valid),
                .mac_bias_valid(mac_bias_valid),
                .mac_bias_idx(mac_bias_idx),
                .mac_bias_data(mac_bias_data),
                .mac_scale(conv_mac_scale),
                .mac_shift(conv_mac_shift),
                .mac_zp(conv_mac_zp),
                .mac_output_data(mac_output_data),
                .mac_output_valid(mac_output_valid),
                .mac_done(mac_done)
            );

            // Requant ROM address — driven by conv_engine's OC tile counter
            assign requant_rom_addr = u_conv_ctrl.oc_tile_cnt;
        end
    endgenerate

    // ======================================================================
    // ALU tile (elementwise — signed saturating add for residual connections)
    // ======================================================================

    generate
        if (TILE_TYPE == 1) begin : gen_alu
            // Reads act_in (main path) and act_skip (skip/residual path),
            // performs signed INT8 saturating add per byte, writes to act_out.
            // Processes 8 bytes per cycle (one 64-bit word).
            logic [15:0] elem_cnt;
            logic alu_active;
            // 1-cycle read latency pipeline
            logic alu_pipe_valid;
            logic [15:0] elem_cnt_d;

            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    elem_cnt <= 16'b0;
                    alu_active <= 1'b0;
                    alu_pipe_valid <= 1'b0;
                    elem_cnt_d <= 16'b0;
                end else if (start && !alu_active) begin
                    alu_active <= 1'b1;
                    elem_cnt <= 16'b0;
                    alu_pipe_valid <= 1'b0;
                end else if (alu_active) begin
                    alu_pipe_valid <= 1'b1;
                    elem_cnt_d <= elem_cnt;
                    if (elem_cnt == INPUT_DIM / 8 - 1) begin
                        alu_active <= 1'b0;
                    end else begin
                        elem_cnt <= elem_cnt + 1;
                    end
                end else begin
                    alu_pipe_valid <= 1'b0;
                end
            end

            // Address generation — both ports read same address
            assign act_in_addr   = elem_cnt[$clog2(ACT_DEPTH)-1:0];
            assign act_skip_addr = elem_cnt[$clog2(ACT_DEPTH)-1:0];
            assign act_out_addr  = elem_cnt_d[$clog2(ACT_DEPTH)-1:0];
            assign act_out_we    = alu_pipe_valid;

            // Signed saturating add: 8 parallel INT8 additions
            logic [ACT_WIDTH-1:0] add_result;
            always_comb begin
                add_result = {ACT_WIDTH{1'b0}};
                for (int i = 0; i < ACT_WIDTH / DATA_W; i++) begin
                    automatic logic signed [DATA_W:0] sum_wide;
                    automatic logic signed [DATA_W-1:0] a_byte = $signed(act_in_data[i*DATA_W +: DATA_W]);
                    automatic logic signed [DATA_W-1:0] b_byte = $signed(act_skip_data[i*DATA_W +: DATA_W]);
                    sum_wide = $signed({a_byte[DATA_W-1], a_byte}) + $signed({b_byte[DATA_W-1], b_byte});
                    // Saturating clamp to [-128, 127]
                    if (sum_wide > 9'sd127)
                        add_result[i*DATA_W +: DATA_W] = 8'sd127;
                    else if (sum_wide < -9'sd128)
                        add_result[i*DATA_W +: DATA_W] = -8'sd128;
                    else
                        add_result[i*DATA_W +: DATA_W] = sum_wide[DATA_W-1:0];
                end
            end

            assign act_out_data = add_result;
            assign done = (alu_pipe_valid && elem_cnt_d == INPUT_DIM / 8 - 1);
            assign busy = alu_active || alu_pipe_valid;
        end
    endgenerate

    // ======================================================================
    // NORM tile (TILE_TYPE=2) — stub for layer_norm_unit
    // ======================================================================

    generate
        if (TILE_TYPE == 2) begin : gen_norm
            // LayerNorm delegated to layer_norm_unit
            // Simplified control wrapper
            logic norm_done, norm_busy;
            logic signed [DATA_W-1:0] norm_rd_data;
            logic signed [DATA_W-1:0] norm_wr_data;
            logic norm_wr_en;

            assign norm_rd_data = act_in_data[DATA_W-1:0];

            layer_norm_unit #(
                .DATA_W(DATA_W),
                .ACC_W(ACC_W),
                .MAX_FEAT_DIM(ACT_DEPTH)
            ) u_layer_norm (
                .clk(clk), .rst_n(rst_n),
                .start(start),
                .done(norm_done),
                .busy(norm_busy),
                .feature_dim(INPUT_DIM[15:0]),
                .epsilon_fixed(32'd0),
                .act_rd_addr(act_in_addr[$clog2(ACT_DEPTH)-1:0]),
                .act_rd_data(norm_rd_data),
                .act_wr_addr(act_out_addr[$clog2(ACT_DEPTH)-1:0]),
                .act_wr_en(norm_wr_en),
                .act_wr_data(norm_wr_data),
                .scale_addr(),
                .scale_data(8'sd1),
                .bias_addr(),
                .bias_data(8'sd0),
                .rsqrt_lut_addr(),
                .rsqrt_lut_data(16'd256)
            );

            assign act_out_data = {{(ACT_WIDTH-DATA_W){norm_wr_data[DATA_W-1]}}, norm_wr_data};
            assign act_out_we = norm_wr_en;
            assign done = norm_done;
            assign busy = norm_busy;
        end
    endgenerate

    // ======================================================================
    // SOFTMAX tile (TILE_TYPE=3) — stub for softmax_unit
    // ======================================================================

    generate
        if (TILE_TYPE == 3) begin : gen_softmax
            logic sm_done, sm_busy;
            logic signed [DATA_W-1:0] sm_rd_data, sm_wr_data;
            logic sm_wr_en;

            assign sm_rd_data = act_in_data[DATA_W-1:0];

            softmax_unit #(
                .DATA_W(DATA_W),
                .MAX_SEQ_LEN(ACT_DEPTH)
            ) u_softmax (
                .clk(clk), .rst_n(rst_n),
                .start(start),
                .done(sm_done),
                .busy(sm_busy),
                .seq_len(INPUT_DIM[15:0]),
                .act_rd_addr(act_in_addr[$clog2(ACT_DEPTH)-1:0]),
                .act_rd_data(sm_rd_data),
                .act_wr_addr(act_out_addr[$clog2(ACT_DEPTH)-1:0]),
                .act_wr_en(sm_wr_en),
                .act_wr_data(sm_wr_data),
                .lut_addr(),
                .lut_data(16'd1)
            );

            assign act_out_data = {{(ACT_WIDTH-DATA_W){sm_wr_data[DATA_W-1]}}, sm_wr_data};
            assign act_out_we = sm_wr_en;
            assign done = sm_done;
            assign busy = sm_busy;
        end
    endgenerate

    // ======================================================================
    // ACTIVATION tile (TILE_TYPE=4)
    // ======================================================================

    generate
        if (TILE_TYPE == 4) begin : gen_activation
            // Stream data through activation_unit element-by-element
            logic [15:0] act_cnt;
            logic act_active;
            logic signed [DATA_W-1:0] au_data_out;
            logic au_valid_out;

            activation_unit #(
                .DATA_W(DATA_W),
                .ACT_TYPE(ACT_TYPE)
            ) u_act (
                .clk(clk), .rst_n(rst_n),
                .data_in(act_in_data[DATA_W-1:0]),
                .valid_in(act_active),
                .data_out(au_data_out),
                .valid_out(au_valid_out),
                .lut_addr(),
                .lut_data({DATA_W{1'b0}})
            );

            always_ff @(posedge clk) begin
                if (!rst_n || !act_active) begin
                    act_cnt <= 16'b0;
                    act_active <= 1'b0;
                end else if (start && !act_active) begin
                    act_active <= 1'b1;
                    act_cnt <= 16'b0;
                end else if (act_active) begin
                    if (act_cnt == INPUT_DIM - 1)
                        act_active <= 1'b0;
                    else
                        act_cnt <= act_cnt + 1;
                end
            end

            assign act_in_addr  = act_cnt[$clog2(ACT_DEPTH)-1:0];
            assign act_out_addr = act_cnt[$clog2(ACT_DEPTH)-1:0];
            assign act_out_data = {{(ACT_WIDTH-DATA_W){au_data_out[DATA_W-1]}}, au_data_out};
            assign act_out_we   = au_valid_out;
            assign done = (act_active && act_cnt == INPUT_DIM - 1);
            assign busy = act_active;
        end
    endgenerate

    // ======================================================================
    // POOL tile (TILE_TYPE=5)
    // ======================================================================

    generate
        if (TILE_TYPE == 5) begin : gen_pool
            logic pool_done, pool_busy;
            logic signed [DATA_W-1:0] pool_wr_data;
            logic pool_wr_en;

            pool_unit #(
                .DATA_W(DATA_W),
                .POOL_TYPE(POOL_TYPE)
            ) u_pool (
                .clk(clk), .rst_n(rst_n),
                .start(start),
                .done(pool_done),
                .busy(pool_busy),
                .channels(IN_CHANNELS[15:0]),
                .in_height(IN_HEIGHT[7:0]),
                .in_width(IN_WIDTH[7:0]),
                .kernel_h(KERNEL_H[3:0]),
                .kernel_w(KERNEL_W[3:0]),
                .stride_h(STRIDE_H[3:0]),
                .stride_w(STRIDE_W[3:0]),
                .pad_h(PAD_H[3:0]),
                .pad_w(PAD_W[3:0]),
                .act_rd_addr(act_in_addr[$clog2(ACT_DEPTH)-1:0]),
                .act_rd_data(act_in_data[DATA_W-1:0]),
                .act_wr_addr(act_out_addr[$clog2(ACT_DEPTH)-1:0]),
                .act_wr_en(pool_wr_en),
                .act_wr_data(pool_wr_data)
            );

            assign act_out_data = {{(ACT_WIDTH-DATA_W){pool_wr_data[DATA_W-1]}}, pool_wr_data};
            assign act_out_we = pool_wr_en;
            assign done = pool_done;
            assign busy = pool_busy;
        end
    endgenerate

    // ======================================================================
    // Default skip port — tie off for non-ALU tiles
    // ======================================================================

    generate
        if (TILE_TYPE != 1) begin : gen_skip_default
            assign act_skip_addr = {$clog2(ACT_DEPTH){1'b0}};
        end
    endgenerate

endmodule
