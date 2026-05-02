// activation_unit.sv — Parameterized activation function unit
// ACT_TYPE selects: ReLU (combinational), ReLU6 (clamp), GELU (LUT), SiLU (LUT)
// 256-entry LUT for GELU/SiLU (1-cycle lookup)
// Reuses existing activation_relu.sv concept.

/* verilator lint_off UNUSEDPARAM */
/* verilator lint_off UNUSEDSIGNAL */

module activation_unit #(
    parameter int DATA_W    = 8,
    // ACT_TYPE: 0=ReLU, 1=ReLU6, 2=GELU (LUT), 3=SiLU (LUT)
    parameter int ACT_TYPE  = 0,
    parameter int LUT_DEPTH = 256
) (
    input clk,
    input rst_n,

    // Data path
    input signed [DATA_W-1:0] data_in,
    input valid_in,
    output logic signed [DATA_W-1:0] data_out,
    output logic valid_out,

    // LUT ROM interface (for GELU/SiLU)
    output logic [DATA_W-1:0] lut_addr,
    input signed [DATA_W-1:0] lut_data
);

    // ======================================================================
    // ReLU (combinational — 0 cycle latency)
    // ======================================================================

    logic signed [DATA_W-1:0] relu_out;
    assign relu_out = (data_in[DATA_W-1]) ? {DATA_W{1'b0}} : data_in;

    // ======================================================================
    // ReLU6 (combinational — clamp at 6 in quantized scale)
    // ======================================================================

    localparam logic signed [DATA_W-1:0] RELU6_MAX = 8'sd6;

    logic signed [DATA_W-1:0] relu6_out;
    assign relu6_out = (data_in[DATA_W-1]) ? {DATA_W{1'b0}} :
                       (data_in > RELU6_MAX) ? RELU6_MAX :
                       data_in;

    // ======================================================================
    // GELU / SiLU (LUT — 1-cycle latency)
    // ======================================================================

    // LUT address: unsigned interpretation of INT8 input [0, 255]
    assign lut_addr = data_in;

    // Pipeline register for LUT output
    logic signed [DATA_W-1:0] lut_out_reg;
    logic lut_valid_reg;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            lut_out_reg  <= {DATA_W{1'b0}};
            lut_valid_reg <= 1'b0;
        end else begin
            lut_out_reg  <= lut_data;
            lut_valid_reg <= valid_in;
        end
    end

    // ======================================================================
    // Output mux
    // ======================================================================

    generate
        if (ACT_TYPE == 0) begin : gen_relu
            assign data_out  = relu_out;
            assign valid_out = valid_in;
        end else if (ACT_TYPE == 1) begin : gen_relu6
            assign data_out  = relu6_out;
            assign valid_out = valid_in;
        end else begin : gen_lut
            // GELU (ACT_TYPE=2) or SiLU (ACT_TYPE=3) — LUT path
            assign data_out  = lut_out_reg;
            assign valid_out = lut_valid_reg;
        end
    endgenerate

/* verilator lint_on UNUSEDPARAM */
/* verilator lint_on UNUSEDSIGNAL */

endmodule
