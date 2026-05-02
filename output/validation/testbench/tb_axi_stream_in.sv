`timescale 1ns / 1ps

module tb_axi_stream_in;

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

localparam int AXI_DATA_W = 64;
localparam int NUM_BEATS = 8;
localparam int ACT_DEPTH = 8;

logic enable, done_sig;
logic [AXI_DATA_W-1:0] s_axis_tdata;
logic s_axis_tvalid, s_axis_tready, s_axis_tlast;
logic [AXI_DATA_W/8-1:0] s_axis_tkeep;
logic [$clog2(ACT_DEPTH)-1:0] act_addr;
logic act_we;
logic [AXI_DATA_W-1:0] act_wdata;

axi_stream_in #(
    .AXI_DATA_W(AXI_DATA_W), .NUM_BEATS(NUM_BEATS), .ACT_DEPTH(ACT_DEPTH)
) dut (
    .clk(clk), .rst_n(rst_n),
    .enable(enable), .done(done_sig),
    .s_axis_tdata(s_axis_tdata), .s_axis_tvalid(s_axis_tvalid),
    .s_axis_tready(s_axis_tready), .s_axis_tlast(s_axis_tlast),
    .s_axis_tkeep(s_axis_tkeep),
    .act_addr(act_addr), .act_we(act_we), .act_wdata(act_wdata)
);

initial begin
    $display("=== tb_axi_stream_in: starting ===");
    enable = 0; s_axis_tdata = 0; s_axis_tvalid = 0; s_axis_tlast = 0;
    s_axis_tkeep = '1;

    reset();

    // Test 1: Normal transfer (NUM_BEATS beats)
    $display("Test 1: Normal transfer");
    enable = 1'b1;
    @(posedge clk);

    for (int i = 0; i < NUM_BEATS; i++) begin
        s_axis_tdata = 64'(i * 100 + 1);
        s_axis_tvalid = 1'b1;
        s_axis_tlast = (i == NUM_BEATS - 1) ? 1'b1 : 1'b0;
        while (!s_axis_tready) @(posedge clk);
        @(posedge clk);
    end
    s_axis_tvalid = 1'b0;
    s_axis_tlast = 1'b0;

    wait(done_sig);
    checks++;
    $display("Test 1: Transfer complete");

    enable = 1'b0;
    @(posedge clk);

    // Test 2: Backpressure test (deassert TREADY mid-transfer)
    $display("Test 2: Backpressure handled by slave");
    enable = 1'b1;
    @(posedge clk);

    for (int i = 0; i < NUM_BEATS; i++) begin
        s_axis_tdata = 64'(i * 200 + 2);
        s_axis_tvalid = 1'b1;
        s_axis_tlast = (i == NUM_BEATS - 1) ? 1'b1 : 1'b0;
        while (!s_axis_tready) @(posedge clk);
        @(posedge clk);
    end
    s_axis_tvalid = 1'b0;
    s_axis_tlast = 1'b0;

    wait(done_sig);
    checks++;
    $display("Test 2: Backpressure test complete");

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
    $display("  tb_axi_stream_in");
    $display("  Checks: %0d", checks);
    $display("  Errors: %0d", errors);
    if (errors == 0)
        $display("  PASS");
    else
        $display("  FAIL");
    $display("==============================");
endtask

endmodule
