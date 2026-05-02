// rom_tile.sv — ASIC ROM template for per-tile weight storage
//
// Read-only logic array with initial block for ASIC synthesis.
// RTL generator fills values per-tile for ASIC target.
// FPGA path uses sram_bank.sv with $readmemh instead.

/* verilator lint_off UNUSEDPARAM */
/* verilator lint_off UNDRIVEN */

module rom_tile #(
    parameter int DEPTH     = 1024,
    parameter int WIDTH     = 1024,
    parameter int TILE_ID   = 0
) (
    input clk,
    input [$clog2(DEPTH)-1:0] addr,
    output logic [WIDTH-1:0] rdata
);

    // ROM storage — synthesized as LUT ROM on ASIC
    (* rom_style = "block" *)
    logic [WIDTH-1:0] rom [0:DEPTH-1];

    // Read port (synchronous, 1-cycle latency)
    always_ff @(posedge clk) begin
        rdata <= rom[addr];
    end

    // Initial block — filled by RTL generator with per-tile weights.
    // For ASIC synthesis, this becomes hardcoded logic.
    // The RTL generator emits a separate rom_tile_<id>.sv with the
    // initial block filled in.

/* verilator lint_on UNUSEDPARAM */
/* verilator lint_on UNDRIVEN */

endmodule
