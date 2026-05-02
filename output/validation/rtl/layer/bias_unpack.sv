// bias_unpack.sv — 1024-bit to 32-bit sequential unpacker
// Latches a 1024-bit SRAM row and outputs 32 sequential INT32 values.
// Each row contains 32 biases packed as INT32 little-endian.
// The MAC array loads biases over 128 cycles (4 SRAM reads x 32 values).

module bias_unpack #(
    parameter int ROW_W    = 1024,
    parameter int BIAS_W   = 32,
    parameter int BIASES_PER_ROW = 32
) (
    input clk,
    input rst_n,
    input [ROW_W-1:0] row_data,
    input row_valid,
    output logic signed [BIAS_W-1:0] bias_out,
    output logic bias_valid,
    input [$clog2(BIASES_PER_ROW)-1:0] bias_idx
);

// Latch the row when valid
logic [ROW_W-1:0] row_reg;

always_ff @(posedge clk) begin
    if (!rst_n)
        row_reg <= {ROW_W{1'b0}};
    else if (row_valid)
        row_reg <= row_data;
end

// Combinational mux to extract the requested bias
assign bias_out = $signed(row_reg[bias_idx*BIAS_W +: BIAS_W]);
assign bias_valid = 1'b1;  // Always valid once row is latched

endmodule
