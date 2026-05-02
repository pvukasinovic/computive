`timescale 1ns / 1ps

module tb_requantize;

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

localparam int PARALLELISM = 128;
localparam int ACC_W = 32;
localparam int DATA_W = 8;

logic valid_in;
logic signed [ACC_W-1:0] acc [0:PARALLELISM-1];
logic [ACC_W-1:0] scale_m;
logic [5:0] shift;
logic signed [DATA_W-1:0] zero_point;
logic has_relu;
logic signed [DATA_W-1:0] data_out [0:PARALLELISM-1];
logic valid_out;

requantize #(
    .PARALLELISM(PARALLELISM), .ACC_W(ACC_W), .DATA_W(DATA_W)
) dut (
    .clk(clk), .rst_n(rst_n),
    .valid_in(valid_in), .acc(acc),
    .scale(scale_m), .shift(shift),
    .zero_point(zero_point), .has_relu(has_relu),
    .data_out(data_out), .valid_out(valid_out)
);

initial begin
    $display("=== tb_requantize: starting ===");
    valid_in = 0;
    for (int i = 0; i < PARALLELISM; i++) acc[i] = 0;
    scale_m = 0; shift = 0; zero_point = 0; has_relu = 0;

    reset();

    // Test 1: Identity requant (M=1.0 in 16.16, shift=16, zp=0)
    $display("Test 1: Identity requantization");
    #1;
    scale_m = 32'd65536;  // 1.0 * 2^16
    shift = 6'd16;
    zero_point = 8'sd0;
    has_relu = 1'b0;

    // Set accumulator values
    acc[0] = 32'sd42;
    acc[1] = -32'sd10;
    acc[2] = 32'sd127;
    acc[3] = -32'sd128;
    acc[4] = 32'sd200;   // Should clamp to 127
    acc[5] = -32'sd200;  // Should clamp to -128
    for (int i = 6; i < PARALLELISM; i++) acc[i] = 32'sd0;

    valid_in = 1'b1;
    @(posedge clk); #1;
    valid_in = 1'b0;

    // Wait for valid_out (3-stage pipeline)
    while (!valid_out) @(posedge clk);
    #1;
    check_eq("rq[0]=42", 32'(data_out[0]), 32'sd42);
    check_eq("rq[1]=-10", 32'(data_out[1]), -32'sd10);
    check_eq("rq[2]=127", 32'(data_out[2]), 32'sd127);
    check_eq("rq[3]=-128", 32'(data_out[3]), -32'sd128);
    check_eq("rq[4]=clamp127", 32'(data_out[4]), 32'sd127);
    check_eq("rq[5]=clamp-128", 32'(data_out[5]), -32'sd128);

    // Test 2: ReLU clamping
    $display("Test 2: ReLU clamping");
    #1;
    has_relu = 1'b1;
    acc[0] = 32'sd10;
    acc[1] = -32'sd10;
    for (int i = 2; i < PARALLELISM; i++) acc[i] = 32'sd0;

    valid_in = 1'b1;
    @(posedge clk); #1;
    valid_in = 1'b0;

    while (!valid_out) @(posedge clk);
    #1;
    check_eq("relu_pos", 32'(data_out[0]), 32'sd10);
    check_eq("relu_neg", 32'(data_out[1]), 32'sd0);

    // Test 3: Zero point addition
    $display("Test 3: Zero point offset");
    #1;
    has_relu = 1'b0;
    zero_point = 8'sd5;
    acc[0] = 32'sd10;
    acc[1] = -32'sd10;
    for (int i = 2; i < PARALLELISM; i++) acc[i] = 32'sd0;

    valid_in = 1'b1;
    @(posedge clk); #1;
    valid_in = 1'b0;

    while (!valid_out) @(posedge clk);
    #1;
    check_eq("zp_pos", 32'(data_out[0]), 32'sd15);
    check_eq("zp_neg", 32'(data_out[1]), -32'sd5);

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
    $display("  tb_requantize");
    $display("  Checks: %0d", checks);
    $display("  Errors: %0d", errors);
    if (errors == 0)
        $display("  PASS");
    else
        $display("  FAIL");
    $display("==============================");
endtask

endmodule
