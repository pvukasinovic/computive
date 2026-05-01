// byte_select.sv — 64-bit word to 8-bit byte mux (combinational)
// Extracts byte[sel] from a 64-bit word, little-endian

module byte_select #(
    parameter int WORD_W = 64,
    parameter int BYTE_W = 8
) (
    input  logic [WORD_W-1:0]               word_in,
    input  logic [$clog2(WORD_W/BYTE_W)-1:0] sel,
    output logic [BYTE_W-1:0]               byte_out
);

assign byte_out = word_in[sel*BYTE_W +: BYTE_W];

endmodule
