`timescale 1ns / 1ps

module tb_ping_pong_buffer;

// Clock and reset
logic clk;
logic rst_n;

localparam CLK_PERIOD = 10;

initial begin
    clk = 1'b0;
    forever #(CLK_PERIOD/2) clk = ~clk;
end

// Reset task
task automatic reset();
    rst_n = 1'b0;
    repeat (5) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);
endtask

// Timeout watchdog
initial begin
    #(CLK_PERIOD * 1_000_000);
    $fatal(1, "TIMEOUT: simulation exceeded 1M cycles");
end

integer errors = 0;
integer checks = 0;


// Check helper
task automatic check_eq(
    input string name,
    input logic signed [31:0] actual,
    input logic signed [31:0] expected
);
    checks++;
    if (actual !== expected) begin
        errors++;
        $display("MISMATCH %s: expected %0d, got %0d (0x%08h vs 0x%08h)",
                 name, expected, actual, expected, actual);
    end
endtask

localparam int DEPTH = 16;
localparam int WIDTH = 64;

logic bank_sel;
logic [$clog2(DEPTH)-1:0] rd_addr, wr_addr;
logic [WIDTH-1:0] rd_data, wr_data;
logic wr_en;
logic [$clog2(DEPTH)-1:0] ext_a_addr, ext_b_addr;
logic ext_a_we, ext_b_we;
logic [WIDTH-1:0] ext_a_wdata, ext_a_rdata, ext_b_wdata, ext_b_rdata;

ping_pong_buffer #(.DEPTH(DEPTH), .WIDTH(WIDTH)) dut (
    .clk(clk), .bank_sel(bank_sel),
    .rd_addr(rd_addr), .rd_data(rd_data),
    .wr_addr(wr_addr), .wr_en(wr_en), .wr_data(wr_data),
    .ext_a_addr(ext_a_addr), .ext_a_we(ext_a_we),
    .ext_a_wdata(ext_a_wdata), .ext_a_rdata(ext_a_rdata),
    .ext_b_addr(ext_b_addr), .ext_b_we(ext_b_we),
    .ext_b_wdata(ext_b_wdata), .ext_b_rdata(ext_b_rdata)
);

initial begin
    $display("=== tb_ping_pong_buffer: starting ===");
    bank_sel = 0; rd_addr = 0; wr_addr = 0; wr_en = 0; wr_data = 0;
    ext_a_addr = 0; ext_a_we = 0; ext_a_wdata = 0;
    ext_b_addr = 0; ext_b_we = 0; ext_b_wdata = 0;

    reset();

    // Test 1: Write to bank B (bank_sel=0), read from bank A
    $display("Test 1: Write B, read A");
    bank_sel = 1'b0;

    // Write to ext_a (bank A) for initial data
    for (int i = 0; i < 4; i++) begin
        ext_a_addr = i[$clog2(DEPTH)-1:0];
        ext_a_we = 1'b1;
        ext_a_wdata = {(WIDTH){1'b0}} | (64'(i + 100));
        @(posedge clk);
    end
    ext_a_we = 1'b0;
    ext_a_addr = '0;

    // Read from A via rd_addr
    for (int i = 0; i < 4; i++) begin
        rd_addr = i[$clog2(DEPTH)-1:0];
        @(posedge clk);
        @(posedge clk);
        checks++;
        if (rd_data !== 64'(i + 100)) begin
            errors++;
            $display("MISMATCH A[%0d]: expected %0d, got %0d", i, i+100, rd_data);
        end
    end

    // Test 2: Swap banks
    $display("Test 2: Bank swap");
    bank_sel = 1'b1;

    // Write to bank A (now write target)
    for (int i = 0; i < 4; i++) begin
        wr_addr = i[$clog2(DEPTH)-1:0];
        wr_en = 1'b1;
        wr_data = 64'(i + 200);
        @(posedge clk);
    end
    wr_en = 1'b0;

    // Swap back and read from A
    bank_sel = 1'b0;
    for (int i = 0; i < 4; i++) begin
        rd_addr = i[$clog2(DEPTH)-1:0];
        @(posedge clk);
        @(posedge clk);
        checks++;
        if (rd_data !== 64'(i + 200)) begin
            errors++;
            $display("MISMATCH swapped A[%0d]: expected %0d, got %0d", i, i+200, rd_data);
        end
    end

    report();
    if (errors > 0) $fatal(1, "FAILED: %0d errors", errors);
    $finish;
end

// Final report
initial begin
    wait(0);  // placeholder — overridden by test
end

task automatic report();
    $display("==============================");
    $display("  tb_ping_pong_buffer");
    $display("  Checks: %0d", checks);
    $display("  Errors: %0d", errors);
    if (errors == 0)
        $display("  PASS");
    else
        $display("  FAIL");
    $display("==============================");
endtask

endmodule
