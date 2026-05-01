// requantize.sv — 3-stage INT32-to-INT8 requantization pipeline
// From rtl-interface-spec.md §5.6
// Pipeline: Stage 1 (multiply) -> Stage 2 (shift+round) -> Stage 3 (clamp+zp+relu)
// 128 parallel units (one per MAC lane)
//
// PER_CHANNEL=0 (default): single scalar scale/shift/zp — MLP path (backward compatible)
// PER_CHANNEL=1: per-lane scale/shift/zp via packed buses — CNN per-channel requant

/* verilator lint_off VARHIDDEN */
module requantize #(
    parameter int PARALLELISM  = 128,
    parameter int ACC_W        = 32,
    parameter int DATA_W       = 8,
    parameter int PER_CHANNEL  = 0    // 0=scalar, 1=per-lane buses
) (
    input  logic                               clk,
    input  logic                               rst_n,
    input  logic                               valid_in,
    input  logic signed [ACC_W-1:0]            acc [0:PARALLELISM-1],
    // Scalar requant inputs (used when PER_CHANNEL=0)
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [ACC_W-1:0]                   scale,       // M (unsigned)
    input  logic [5:0]                         shift,       // Right-shift amount
    input  logic signed [DATA_W-1:0]           zero_point,  // Output zero point
    // Per-channel requant buses (used when PER_CHANNEL=1)
    input  logic [PARALLELISM*ACC_W-1:0]       scale_bus,   // packed per-lane M
    input  logic [PARALLELISM*6-1:0]           shift_bus,   // packed per-lane shift
    input  logic [PARALLELISM*DATA_W-1:0]      zp_bus,      // packed per-lane zp
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic                               has_relu,
    output logic signed [DATA_W-1:0]           data_out [0:PARALLELISM-1],
    output logic                               valid_out
);

// Per-lane parameter extraction
logic [ACC_W-1:0]        lane_scale [0:PARALLELISM-1];
logic [5:0]              lane_shift [0:PARALLELISM-1];
logic signed [DATA_W-1:0] lane_zp   [0:PARALLELISM-1];

generate
    for (genvar g = 0; g < PARALLELISM; g++) begin : gen_lane_params
        if (PER_CHANNEL != 0) begin : per_ch
            assign lane_scale[g] = scale_bus[g*ACC_W +: ACC_W];
            assign lane_shift[g] = shift_bus[g*6 +: 6];
            assign lane_zp[g]    = $signed(zp_bus[g*DATA_W +: DATA_W]);
        end else begin : scalar
            assign lane_scale[g] = scale;
            assign lane_shift[g] = shift;
            assign lane_zp[g]    = zero_point;
        end
    end
endgenerate

// Pipeline stage 1: Multiply acc * M (signed 64-bit product)
logic signed [63:0] product [0:PARALLELISM-1];
logic valid_s1;

always_ff @(posedge clk) begin
    if (!rst_n) begin
        valid_s1 <= 1'b0;
    end else begin
        valid_s1 <= valid_in;
        for (int i = 0; i < PARALLELISM; i++) begin
            product[i] <= $signed(acc[i]) * $signed({1'b0, lane_scale[i]});
        end
    end
end

// Pipeline stage 2: Round-half-away-from-zero + shift
logic signed [63:0] shifted [0:PARALLELISM-1];
logic valid_s2;
logic signed [DATA_W-1:0] zp_s1 [0:PARALLELISM-1];
logic relu_s1;
logic [5:0] shift_s1 [0:PARALLELISM-1];
logic signed [DATA_W-1:0] zp_s2 [0:PARALLELISM-1];
logic relu_s2;

always_ff @(posedge clk) begin
    if (!rst_n) begin
        valid_s2 <= 1'b0;
        relu_s1  <= 1'b0;
        for (int i = 0; i < PARALLELISM; i++) begin
            zp_s1[i]    <= '0;
            shift_s1[i] <= '0;
        end
    end else begin
        // Capture params at S1 entry
        if (valid_in) begin
            relu_s1  <= has_relu;
            for (int i = 0; i < PARALLELISM; i++) begin
                zp_s1[i]    <= lane_zp[i];
                shift_s1[i] <= lane_shift[i];
            end
        end

        valid_s2 <= valid_s1;
        for (int i = 0; i < PARALLELISM; i++) begin
            zp_s2[i] <= zp_s1[i];
        end
        relu_s2  <= relu_s1;

        for (int i = 0; i < PARALLELISM; i++) begin
            // Round-half-away-from-zero: add rounding bias = 1 << (shift - 1)
            shifted[i] <= (product[i] + (64'sd1 <<< (shift_s1[i] - 1))) >>> shift_s1[i];
        end
    end
end

// Pipeline stage 3: Clamp to INT8 range + add zero point + optional ReLU
always_ff @(posedge clk) begin
    if (!rst_n) begin
        valid_out <= 1'b0;
    end else begin
        valid_out <= valid_s2;
        for (int i = 0; i < PARALLELISM; i++) begin
            // Add zero point
            automatic logic signed [63:0] with_zp = shifted[i] + $signed({{56{zp_s2[i][DATA_W-1]}}, zp_s2[i]});
            // Clamp to [-128, 127]
            automatic logic signed [DATA_W-1:0] clamped;
            if (with_zp > 127)
                clamped = 8'sd127;
            else if (with_zp < -128)
                clamped = -8'sd128;
            else
                clamped = with_zp[DATA_W-1:0];
            // Optional ReLU
            if (relu_s2 && clamped[DATA_W-1])
                data_out[i] <= '0;
            else
                data_out[i] <= clamped;
        end
    end
end

endmodule
