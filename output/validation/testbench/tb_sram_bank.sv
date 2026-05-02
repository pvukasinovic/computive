`timescale 1ns / 1ps

module tb_sram_bank;

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

logic [$clog2(DEPTH)-1:0] addr;
logic we;
logic [WIDTH-1:0] wdata;
logic [WIDTH-1:0] rdata;

sram_bank #(.DEPTH(DEPTH), .WIDTH(WIDTH), .INIT_FILE("test_sram_init.mem")) dut (
    .clk(clk), .addr(addr), .we(we), .wdata(wdata), .rdata(rdata)
);

initial begin
    $display("=== tb_sram_bank: starting ===");
    addr = 0; we = 0; wdata = '0;

    // Wait for reset
    reset();

    // Test 1: Write-then-read pattern
    $display("Test 1: Write-read pattern");
    for (int i = 0; i < DEPTH; i++) begin
        addr = i[$clog2(DEPTH)-1:0];
        wdata = {(WIDTH/8){i[7:0]}};
        we = 1'b1;
        @(posedge clk);
    end
    we = 1'b0;

    // Read back and verify
    for (int i = 0; i < DEPTH; i++) begin
        addr = i[$clog2(DEPTH)-1:0];
        @(posedge clk);  // 1-cycle read latency
        @(posedge clk);
        checks++;
        if (rdata !== {(WIDTH/8){i[7:0]}}) begin
            errors++;
            $display("MISMATCH at addr %0d: expected 0x%016h, got 0x%016h",
                     i, {(WIDTH/8){i[7:0]}}, rdata);
        end
    end

    // Test 2: Overwrite and re-read
    $display("Test 2: Overwrite");
    addr = 0;
    wdata = 64'hDEAD_BEEF_CAFE_BABE;
    we = 1'b1;
    @(posedge clk);
    we = 1'b0;
    @(posedge clk);
    @(posedge clk);
    checks++;
    if (rdata !== 64'hDEAD_BEEF_CAFE_BABE) begin
        errors++;
        $display("MISMATCH overwrite: got 0x%016h", rdata);
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
    $display("  tb_sram_bank");
    $display("  Checks: %0d", checks);
    $display("  Errors: %0d", errors);
    if (errors == 0)
        $display("  PASS");
    else
        $display("  FAIL");
    $display("==============================");
endtask

endmodule
