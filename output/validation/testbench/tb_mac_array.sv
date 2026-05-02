`timescale 1ns / 1ps

module tb_mac_array;

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

// DUT parameters
localparam int PARALLELISM = 128;
localparam int DATA_W = 8;
localparam int ACC_W = 32;

// DUT signals
logic start;
logic [15:0] input_dim;
logic has_relu;
logic [PARALLELISM*DATA_W-1:0] weight_data;
logic weight_valid;
logic signed [DATA_W-1:0] input_data;
logic input_valid;
logic signed [ACC_W-1:0] bias_data;
logic bias_valid;
logic [$clog2(PARALLELISM)-1:0] bias_idx;
logic [ACC_W-1:0] scale;
logic [5:0] shift;
logic signed [DATA_W-1:0] zero_point;
logic signed [DATA_W-1:0] output_data [0:PARALLELISM-1];
logic output_valid;
logic done;

mac_array #(
    .PARALLELISM(PARALLELISM),
    .DATA_W(DATA_W),
    .ACC_W(ACC_W)
) dut (
    .clk(clk), .rst_n(rst_n),
    .start(start), .input_dim(input_dim), .has_relu(has_relu),
    .weight_data(weight_data), .weight_valid(weight_valid),
    .input_data(input_data), .input_valid(input_valid),
    .bias_data(bias_data), .bias_valid(bias_valid), .bias_idx(bias_idx),
    .scale(scale), .shift(shift), .zero_point(zero_point),
    .output_data(output_data), .output_valid(output_valid), .done(done)
);

// Test procedure
initial begin
    $display("=== tb_mac_array: starting ===");
    // Initialize
    start = 0; input_dim = 0; has_relu = 0;
    weight_data = '0; weight_valid = 0;
    input_data = 0; input_valid = 0;
    bias_data = 0; bias_valid = 0; bias_idx = 0;
    scale = 0; shift = 0; zero_point = 0;

    reset();

    // Test 1: Zero input → zero output
    $display("Test 1: Zero input");
    input_dim = 16'd4;
    has_relu = 1'b0;
    scale = 32'd65536;  // M=1.0 in 16.16
    shift = 6'd16;
    zero_point = 8'sd0;

    // Set weights to small values
    for (int i = 0; i < PARALLELISM; i++)
        weight_data[i*DATA_W +: DATA_W] = 8'd1;

    #1; start = 1'b1;
    @(posedge clk); #1; start = 1'b0;

    // Wait for BIAS_LOAD state
    @(posedge clk);

    // Load zero biases
    for (int i = 0; i < PARALLELISM; i++) begin
        bias_data = 32'sd0;
        bias_valid = 1'b1;
        bias_idx = i[$clog2(PARALLELISM)-1:0];
        @(posedge clk);
    end
    bias_valid = 1'b0;

    // Feed zero inputs
    for (int i = 0; i < 4; i++) begin
        input_data = 8'sd0;
        input_valid = 1'b1;
        @(posedge clk);
    end
    input_valid = 1'b0;

    // Wait for done
    wait(done);
    @(posedge clk);

    for (int i = 0; i < PARALLELISM; i++) begin
        check_eq($sformatf("out[%0d]", i), 32'(output_data[i]), 32'sd0);
    end
    $display("Test 1: done (%0d errors)", errors);

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
    $display("  tb_mac_array");
    $display("  Checks: %0d", checks);
    $display("  Errors: %0d", errors);
    if (errors == 0)
        $display("  PASS");
    else
        $display("  FAIL");
    $display("==============================");
endtask

endmodule
