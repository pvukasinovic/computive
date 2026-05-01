// ping_pong_buffer.sv — Double-buffered activation memory
// Wraps two sram_banks (ACT_A and ACT_B).
// bank_sel swaps read/write roles: 0 = read A/write B, 1 = read B/write A

module ping_pong_buffer #(
    parameter int DEPTH = 80,
    parameter int WIDTH = 64
) (
    input  logic                         clk,

    // Bank select: 0 = input from A, 1 = input from B
    input  logic                         bank_sel,

    // Read port (for input activations)
    input  logic [$clog2(DEPTH)-1:0]     rd_addr,
    output logic [WIDTH-1:0]             rd_data,

    // Write port (for output activations)
    input  logic [$clog2(DEPTH)-1:0]     wr_addr,
    input  logic                         wr_en,
    input  logic [WIDTH-1:0]             wr_data,

    // Direct access to bank A (for AXI-Stream in/out)
    input  logic [$clog2(DEPTH)-1:0]     ext_a_addr,
    input  logic                         ext_a_we,
    input  logic [WIDTH-1:0]             ext_a_wdata,
    output logic [WIDTH-1:0]             ext_a_rdata,

    // Direct access to bank B (if needed)
    input  logic [$clog2(DEPTH)-1:0]     ext_b_addr,
    input  logic                         ext_b_we,
    input  logic [WIDTH-1:0]             ext_b_wdata,
    output logic [WIDTH-1:0]             ext_b_rdata
);

// SRAM bank signals
logic [$clog2(DEPTH)-1:0] a_addr, b_addr;
logic                     a_we, b_we;
logic [WIDTH-1:0]         a_wdata, b_wdata;
logic [WIDTH-1:0]         a_rdata, b_rdata;

sram_bank #(.DEPTH(DEPTH), .WIDTH(WIDTH)) act_a (
    .clk(clk), .addr(a_addr), .we(a_we),
    .wdata(a_wdata), .rdata(a_rdata)
);

sram_bank #(.DEPTH(DEPTH), .WIDTH(WIDTH)) act_b (
    .clk(clk), .addr(b_addr), .we(b_we),
    .wdata(b_wdata), .rdata(b_rdata)
);

// Mux logic based on bank_sel
always_comb begin
    if (!bank_sel) begin
        // bank_sel=0: read from A, write to B (normal layer operation)
        a_addr  = (ext_a_we || ext_a_addr != '0) ? ext_a_addr : rd_addr;
        a_we    = ext_a_we;
        a_wdata = ext_a_wdata;
        b_addr  = (ext_b_we || ext_b_addr != '0) ? ext_b_addr : wr_addr;
        b_we    = ext_b_we || wr_en;
        b_wdata = ext_b_we ? ext_b_wdata : wr_data;
    end else begin
        // bank_sel=1: read from B, write to A
        b_addr  = (ext_b_we || ext_b_addr != '0) ? ext_b_addr : rd_addr;
        b_we    = ext_b_we;
        b_wdata = ext_b_wdata;
        a_addr  = (ext_a_we || ext_a_addr != '0) ? ext_a_addr : wr_addr;
        a_we    = ext_a_we || wr_en;
        a_wdata = ext_a_we ? ext_a_wdata : wr_data;
    end
end

// Read data mux
assign rd_data = bank_sel ? b_rdata : a_rdata;

// Direct access read outputs
assign ext_a_rdata = a_rdata;
assign ext_b_rdata = b_rdata;

endmodule
