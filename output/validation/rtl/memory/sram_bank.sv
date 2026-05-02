// sram_bank.sv — Behavioral SRAM with $readmemh initialization
// From rtl-interface-spec.md §7.4 (verbatim)
// Read-first mode, inferred as BRAM by Vivado

module sram_bank #(
    parameter int DEPTH     = 1536,
    parameter int WIDTH     = 1024,
    parameter string INIT_FILE = ""
) (
    input clk,
    input [$clog2(DEPTH)-1:0] addr,
    input we,
    input [WIDTH-1:0] wdata,
    output logic [WIDTH-1:0] rdata
);

// Behavioral SRAM (for BRAM inference by Vivado)
logic [WIDTH-1:0] mem [0:DEPTH-1];

initial if (INIT_FILE != "") $readmemh(INIT_FILE, mem);

always_ff @(posedge clk) begin
    if (we)
        mem[addr] <= wdata;
    rdata <= mem[addr];  // Read-first mode
end

endmodule
