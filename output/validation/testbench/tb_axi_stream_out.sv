`timescale 1ns / 1ps

module tb_axi_stream_out;

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
logic [AXI_DATA_W-1:0] m_axis_tdata;
logic m_axis_tvalid, m_axis_tready, m_axis_tlast;
logic [AXI_DATA_W/8-1:0] m_axis_tkeep;
logic [$clog2(ACT_DEPTH)-1:0] act_addr;
logic [AXI_DATA_W-1:0] act_rdata;

axi_stream_out #(
    .AXI_DATA_W(AXI_DATA_W), .NUM_BEATS(NUM_BEATS), .ACT_DEPTH(ACT_DEPTH)
) dut (
    .clk(clk), .rst_n(rst_n),
    .enable(enable), .done(done_sig),
    .m_axis_tdata(m_axis_tdata), .m_axis_tvalid(m_axis_tvalid),
    .m_axis_tready(m_axis_tready), .m_axis_tlast(m_axis_tlast),
    .m_axis_tkeep(m_axis_tkeep),
    .act_addr(act_addr), .act_rdata(act_rdata)
);

// Fake SRAM: return address as data
assign act_rdata = {(AXI_DATA_W){1'b0}} | (64'(act_addr) * 64'd100 + 64'd7);

// Capture received data
logic [AXI_DATA_W-1:0] received [0:NUM_BEATS-1];
integer rx_count;
logic saw_tlast;

initial begin
    $display("=== tb_axi_stream_out: starting ===");
    enable = 0; m_axis_tready = 0;
    rx_count = 0; saw_tlast = 0;

    reset();

    // Test 1: Normal readout
    $display("Test 1: Normal readout");
    m_axis_tready = 1'b1;
    enable = 1'b1;

    while (!done_sig) begin
        @(posedge clk);
        if (m_axis_tvalid && m_axis_tready) begin
            received[rx_count] = m_axis_tdata;
            rx_count++;
            if (m_axis_tlast) saw_tlast = 1;
        end
    end

    checks++;
    if (rx_count != NUM_BEATS) begin
        errors++;
        $display("MISMATCH: expected %0d beats, got %0d", NUM_BEATS, rx_count);
    end

    checks++;
    if (!saw_tlast) begin
        errors++;
        $display("MISMATCH: TLAST never asserted");
    end

    enable = 1'b0;
    @(posedge clk);

    // Test 2: Stall mid-transfer
    $display("Test 2: Stall mid-transfer");
    rx_count = 0;
    saw_tlast = 0;
    m_axis_tready = 1'b1;
    enable = 1'b1;

    while (!done_sig) begin
        @(posedge clk);
        if (m_axis_tvalid && m_axis_tready) begin
            received[rx_count] = m_axis_tdata;
            rx_count++;
            // Stall after 3rd beat
            if (rx_count == 3) begin
                m_axis_tready = 1'b0;
                repeat(5) @(posedge clk);
                m_axis_tready = 1'b1;
            end
            if (m_axis_tlast) saw_tlast = 1;
        end
    end

    checks++;
    if (rx_count != NUM_BEATS) begin
        errors++;
        $display("MISMATCH stall: expected %0d beats, got %0d", NUM_BEATS, rx_count);
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
    $display("  tb_axi_stream_out");
    $display("  Checks: %0d", checks);
    $display("  Errors: %0d", errors);
    if (errors == 0)
        $display("  PASS");
    else
        $display("  FAIL");
    $display("==============================");
endtask

endmodule
