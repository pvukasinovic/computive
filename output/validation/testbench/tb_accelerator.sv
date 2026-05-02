`timescale 1ns / 1ps

module tb_accelerator;

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

// ──────────────────────────────────────────────
// Model-specific constants
// ──────────────────────────────────────────────
localparam int INPUT_BYTES = 640;
localparam int OUTPUT_BYTES = 640;
localparam int INPUT_BEATS = 80;
localparam int OUTPUT_BEATS = 80;
localparam int NUM_LAYERS = 4;
localparam int NUM_VECTORS = 100;
localparam int EXPECTED_CYCLES = 2728;

// ──────────────────────────────────────────────
// DUT instantiation (accelerator_top)
// ──────────────────────────────────────────────

// AXI-Lite CSR
logic [7:0] s_axi_awaddr, s_axi_araddr;
logic s_axi_awvalid, s_axi_awready, s_axi_wvalid, s_axi_wready;
logic s_axi_bvalid, s_axi_bready;
logic s_axi_arvalid, s_axi_arready, s_axi_rvalid, s_axi_rready;
logic [31:0] s_axi_wdata, s_axi_rdata;
logic [3:0] s_axi_wstrb;
logic [1:0] s_axi_bresp, s_axi_rresp;

// AXI-Stream input
logic [63:0] s_axis_tdata;
logic s_axis_tvalid, s_axis_tready, s_axis_tlast;
logic [7:0] s_axis_tkeep;

// AXI-Stream output
logic [63:0] m_axis_tdata;
logic m_axis_tvalid, m_axis_tready, m_axis_tlast;
logic [7:0] m_axis_tkeep;

logic irq;

accelerator_top dut (
    .clk(clk), .rst_n(rst_n),
    // AXI-Lite
    .s_axi_awaddr(s_axi_awaddr), .s_axi_awvalid(s_axi_awvalid), .s_axi_awready(s_axi_awready),
    .s_axi_wdata(s_axi_wdata), .s_axi_wstrb(s_axi_wstrb),
    .s_axi_wvalid(s_axi_wvalid), .s_axi_wready(s_axi_wready),
    .s_axi_bresp(s_axi_bresp), .s_axi_bvalid(s_axi_bvalid), .s_axi_bready(s_axi_bready),
    .s_axi_araddr(s_axi_araddr), .s_axi_arvalid(s_axi_arvalid), .s_axi_arready(s_axi_arready),
    .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp),
    .s_axi_rvalid(s_axi_rvalid), .s_axi_rready(s_axi_rready),
    // AXI-Stream input
    .s_axis_tdata(s_axis_tdata), .s_axis_tvalid(s_axis_tvalid),
    .s_axis_tready(s_axis_tready), .s_axis_tlast(s_axis_tlast), .s_axis_tkeep(s_axis_tkeep),
    // AXI-Stream output
    .m_axis_tdata(m_axis_tdata), .m_axis_tvalid(m_axis_tvalid),
    .m_axis_tready(m_axis_tready), .m_axis_tlast(m_axis_tlast), .m_axis_tkeep(m_axis_tkeep),
    // Interrupt
    .irq(irq)
);

// ──────────────────────────────────────────────
// Test data storage
// ──────────────────────────────────────────────
logic [7:0] received_output [0:OUTPUT_BYTES-1];

// ──────────────────────────────────────────────
// AXI-Lite helper tasks
// ──────────────────────────────────────────────
task automatic axi_lite_write(input logic [7:0] addr, input logic [31:0] data);
    s_axi_awaddr = addr;
    s_axi_awvalid = 1'b1;
    s_axi_wdata = data;
    s_axi_wstrb = 4'hF;
    s_axi_wvalid = 1'b1;
    s_axi_bready = 1'b1;
    @(posedge clk);
    while (!s_axi_awready) @(posedge clk);
    s_axi_awvalid = 1'b0;
    while (!s_axi_wready) @(posedge clk);
    s_axi_wvalid = 1'b0;
    while (!s_axi_bvalid) @(posedge clk);
    s_axi_bready = 1'b0;
    @(posedge clk);
endtask

task automatic axi_lite_read(input logic [7:0] addr, output logic [31:0] data);
    s_axi_araddr = addr;
    s_axi_arvalid = 1'b1;
    s_axi_rready = 1'b1;
    @(posedge clk);
    while (!s_axi_arready) @(posedge clk);
    s_axi_arvalid = 1'b0;
    while (!s_axi_rvalid) @(posedge clk);
    data = s_axi_rdata;
    s_axi_rready = 1'b0;
    @(posedge clk);
endtask

// ──────────────────────────────────────────────
// AXI-Stream driver: send random input vector
// ──────────────────────────────────────────────
task automatic send_random_input();
    for (int beat = 0; beat < INPUT_BEATS; beat++) begin
        for (int b = 0; b < 8; b++) begin
            s_axis_tdata[b*8 +: 8] = $urandom_range(0, 255);
        end
        s_axis_tvalid = 1'b1;
        s_axis_tlast = (beat == INPUT_BEATS - 1) ? 1'b1 : 1'b0;
        s_axis_tkeep = 8'hFF;
        while (!s_axis_tready) @(posedge clk);
        @(posedge clk);
    end
    s_axis_tvalid = 1'b0;
    s_axis_tlast = 1'b0;
endtask

// ──────────────────────────────────────────────
// AXI-Stream monitor: capture output vector
// ──────────────────────────────────────────────
task automatic receive_output();
    integer beat_idx = 0;
    m_axis_tready = 1'b1;
    while (beat_idx < OUTPUT_BEATS) begin
        @(posedge clk);
        if (m_axis_tvalid && m_axis_tready) begin
            for (int b = 0; b < 8; b++) begin
                integer byte_idx = beat_idx * 8 + b;
                if (byte_idx < OUTPUT_BYTES)
                    received_output[byte_idx] = m_axis_tdata[b*8 +: 8];
            end
            beat_idx++;
        end
    end
    m_axis_tready = 1'b0;
endtask

// ──────────────────────────────────────────────
// Main test sequence
// ──────────────────────────────────────────────
logic [31:0] read_val;

initial begin
    $display("=== tb_accelerator: starting ===");
    $display("Model: ad_model");
    $display("Input: 640 bytes (80 beats)");
    $display("Output: 640 bytes (80 beats)");
    $display("Layers: 4");

    // Initialize AXI-Lite
    s_axi_awaddr = 0; s_axi_awvalid = 0; s_axi_wdata = 0; s_axi_wstrb = 0;
    s_axi_wvalid = 0; s_axi_bready = 0;
    s_axi_araddr = 0; s_axi_arvalid = 0; s_axi_rready = 0;
    // Initialize AXI-Stream
    s_axis_tdata = 0; s_axis_tvalid = 0; s_axis_tlast = 0; s_axis_tkeep = 0;
    m_axis_tready = 0;

    reset();

    // ── Test 1: VERSION register ──
    $display("Test 1: VERSION register");
    axi_lite_read(8'h18, read_val);
    check_eq("VERSION", read_val, 32'h0001_0000);

    // ── Test 2: SCRATCH register write/read ──
    $display("Test 2: SCRATCH register");
    axi_lite_write(8'h1C, 32'hDEAD_BEEF);
    axi_lite_read(8'h1C, read_val);
    check_eq("SCRATCH", read_val, 32'hDEAD_BEEF);

    // ── Test 3: Single inference (smoke test) ──
    $display("Test 3: Single inference smoke test");
    axi_lite_write(8'h00, 32'h0000_0001);  // CTRL.START
    send_random_input();

    // Wait for completion (poll STATUS.done bit)
    for (int cyc = 0; cyc < 10_000_000; cyc++) begin
        axi_lite_read(8'h04, read_val);
        if (read_val[2]) break;  // done bit
        @(posedge clk);
    end
    checks++;
    if (!read_val[2]) begin
        errors++;
        $display("FAIL: inference did not complete (STATUS=0x%08h)", read_val);
    end else begin
        $display("  Inference completed");
    end

    // Receive output (drains AXI-Stream master)
    receive_output();

    // ── Test 4: IRQ path ──
    $display("Test 4: IRQ path");
    // Enable IRQ_DONE
    axi_lite_write(8'h08, 32'h0000_0001);

    // Start another inference
    axi_lite_write(8'h00, 32'h0000_0001);
    send_random_input();

    // Wait for IRQ
    for (int cyc = 0; cyc < 10_000_000; cyc++) begin
        @(posedge clk);
        if (irq) break;
    end
    checks++;
    if (!irq) begin
        errors++;
        $display("FAIL: IRQ not asserted");
    end

    // Check IRQ_STATUS
    axi_lite_read(8'h0C, read_val);
    check_eq("IRQ_STATUS_set", read_val[0], 1'b1);

    // Clear IRQ (W1C)
    axi_lite_write(8'h0C, 32'h0000_0001);
    axi_lite_read(8'h0C, read_val);
    check_eq("IRQ_STATUS_cleared", read_val[0], 1'b0);

    // Drain output
    receive_output();

    // ── Test 5: Cycle count register ──
    $display("Test 5: Cycle count");
    axi_lite_read(8'h10, read_val);
    checks++;
    if (read_val == 0) begin
        errors++;
        $display("FAIL: CYCLE_COUNT is zero after inference");
    end else begin
        $display("  CYCLE_COUNT = %0d", read_val);
    end
    if (EXPECTED_CYCLES > 0 && read_val > EXPECTED_CYCLES * 110 / 100) begin
        $display("WARNING: cycle count %0d exceeds expected %0d by >10%%",
                 read_val, EXPECTED_CYCLES);
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
    $display("  tb_accelerator");
    $display("  Checks: %0d", checks);
    $display("  Errors: %0d", errors);
    if (errors == 0)
        $display("  PASS");
    else
        $display("  FAIL");
    $display("==============================");
endtask

endmodule
