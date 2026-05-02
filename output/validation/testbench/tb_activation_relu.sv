`timescale 1ns / 1ps

module tb_activation_relu;

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

localparam int DATA_W = 8;

logic signed [DATA_W-1:0] data_in;
logic enable;
logic signed [DATA_W-1:0] data_out;

activation_relu #(.DATA_W(DATA_W)) dut (
    .data_in(data_in), .enable(enable), .data_out(data_out)
);

initial begin
    $display("=== tb_activation_relu: starting ===");
    enable = 1'b1;

    // Sweep all 256 INT8 values
    for (int i = -128; i <= 127; i++) begin
        data_in = i[DATA_W-1:0];
        #1;
        if (i < 0)
            check_eq($sformatf("relu(%0d)", i), 32'(data_out), 32'sd0);
        else
            check_eq($sformatf("relu(%0d)", i), 32'(data_out), 32'(i[DATA_W-1:0]));
    end

    // Test bypass (enable=0)
    enable = 1'b0;
    data_in = -8'sd50;
    #1;
    check_eq("bypass_neg", 32'(data_out), -32'sd50);

    data_in = 8'sd50;
    #1;
    check_eq("bypass_pos", 32'(data_out), 32'sd50);

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
    $display("  tb_activation_relu");
    $display("  Checks: %0d", checks);
    $display("  Errors: %0d", errors);
    if (errors == 0)
        $display("  PASS");
    else
        $display("  FAIL");
    $display("==============================");
endtask

endmodule
