`timescale 1ns / 1ps

module tb_axi_lite_ctrl;

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

localparam int ADDR_W = 8;
localparam int DATA_W = 32;

// AXI-Lite signals
logic [ADDR_W-1:0] awaddr, araddr;
logic awvalid, awready, wvalid, wready, bvalid, bready;
logic arvalid, arready, rvalid, rready;
logic [DATA_W-1:0] wdata, rdata;
logic [3:0] wstrb;
logic [1:0] bresp, rresp;

// Control/status
logic ctrl_start, ctrl_soft_rst, ctrl_continuous, irq_en_out;
logic status_idle, status_busy, status_done, status_error;
logic [DATA_W-1:0] cycle_count, inf_count, error_code;
logic [1:0] layer_status;
logic irq_done, irq_error, irq;

axi_lite_ctrl #(.ADDR_W(ADDR_W), .DATA_W(DATA_W)) dut (
    .clk(clk), .rst_n(rst_n),
    .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
    .s_axi_wdata(wdata), .s_axi_wstrb(wstrb), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
    .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
    .s_axi_araddr(araddr), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
    .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready),
    .ctrl_start(ctrl_start), .ctrl_soft_rst(ctrl_soft_rst),
    .ctrl_continuous(ctrl_continuous), .irq_en(irq_en_out),
    .status_idle(status_idle), .status_busy(status_busy),
    .status_done(status_done), .status_error(status_error),
    .cycle_count(cycle_count), .inf_count(inf_count),
    .error_code(error_code), .layer_status(layer_status),
    .irq_done(irq_done), .irq_error(irq_error), .irq(irq)
);

// AXI-Lite write task
task automatic axi_write(input logic [ADDR_W-1:0] addr, input logic [DATA_W-1:0] data);
    // Drive valid+data+bready on negedge for clean setup
    #1;
    awaddr = addr;
    awvalid = 1'b1;
    wdata = data;
    wstrb = 4'hF;
    wvalid = 1'b1;
    bready = 1'b1;
    // Posedge: DUT samples awvalid+wvalid (aw_ready and w_ready are high)
    @(posedge clk); #1;
    awvalid = 1'b0;
    wvalid = 1'b0;
    // Wait for bvalid to appear
    while (!bvalid) begin @(posedge clk); #1; end
    // Keep bready high for one more posedge so DUT sees bvalid && bready
    @(posedge clk); #1;
    bready = 1'b0;
    @(posedge clk); #1;
endtask

// AXI-Lite read task
task automatic axi_read(input logic [ADDR_W-1:0] addr, output logic [DATA_W-1:0] data);
    #1;
    araddr = addr;
    arvalid = 1'b1;
    rready = 1'b1;
    // Posedge: DUT samples arvalid (arready is combinational, high initially)
    @(posedge clk); #1;
    arvalid = 1'b0;
    // Wait for rvalid
    while (!rvalid) begin @(posedge clk); #1; end
    data = rdata;
    // Keep rready high for one more posedge so DUT sees rvalid && rready
    @(posedge clk); #1;
    rready = 1'b0;
    @(posedge clk); #1;
endtask

logic [DATA_W-1:0] read_val;

initial begin
    $display("=== tb_axi_lite_ctrl: starting ===");
    awaddr = 0; awvalid = 0; wdata = 0; wstrb = 0; wvalid = 0; bready = 0;
    araddr = 0; arvalid = 0; rready = 0;
    status_idle = 1; status_busy = 0; status_done = 0; status_error = 0;
    cycle_count = 0; inf_count = 0; error_code = 0; layer_status = 0;
    irq_done = 0; irq_error = 0;

    reset();

    // Test 1: Read VERSION register (should be 0x00010000)
    $display("Test 1: Read VERSION");
    axi_read(8'h18, read_val);
    check_eq("VERSION", read_val, 32'h00010000);

    // Test 2: Write/read SCRATCH register
    $display("Test 2: SCRATCH register");
    axi_write(8'h1C, 32'hCAFE_BABE);
    axi_read(8'h1C, read_val);
    check_eq("SCRATCH", read_val, 32'hCAFE_BABE);

    // Test 3: CTRL register self-clearing bits
    $display("Test 3: CTRL self-clear");
    axi_write(8'h00, 32'h0000_0001);  // Set START
    @(posedge clk);  // START auto-clears
    axi_read(8'h00, read_val);
    check_eq("CTRL_after_start", read_val[0], 32'd0);  // Should be cleared

    // Test 4: IRQ_STATUS W1C behavior
    $display("Test 4: IRQ W1C");
    #1; irq_done = 1'b1;
    @(posedge clk); #1; irq_done = 1'b0;
    @(posedge clk); #1;
    axi_read(8'h0C, read_val);
    check_eq("IRQ_STATUS_set", read_val[0], 32'd1);

    // Clear by writing 1
    axi_write(8'h0C, 32'h0000_0001);
    axi_read(8'h0C, read_val);
    check_eq("IRQ_STATUS_cleared", read_val[0], 32'd0);

    // Test 5: STATUS register reflects inputs
    $display("Test 5: STATUS register");
    #1; status_idle = 0; status_busy = 1;
    @(posedge clk); #1;
    axi_read(8'h04, read_val);
    check_eq("STATUS_busy", read_val[1], 32'd1);
    check_eq("STATUS_idle", read_val[0], 32'd0);

    // Test 6: IRQ output
    $display("Test 6: IRQ output");
    axi_write(8'h08, 32'h0000_0001);  // Enable IRQ_DONE
    #1; irq_done = 1'b1;
    @(posedge clk); #1; irq_done = 1'b0;
    repeat(3) @(posedge clk); #1;
    checks++;
    if (!irq) begin
        errors++;
        $display("MISMATCH: IRQ not asserted");
    end

    // Test 7: Read all CSR offsets
    $display("Test 7: Read all CSRs");
    #1; cycle_count = 32'd12345;
    inf_count = 32'd678;
    @(posedge clk); #1;
    axi_read(8'h10, read_val);
    check_eq("CYCLE_COUNT", read_val, 32'd12345);
    axi_read(8'h14, read_val);
    check_eq("INF_COUNT", read_val, 32'd678);

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
    $display("  tb_axi_lite_ctrl");
    $display("  Checks: %0d", checks);
    $display("  Errors: %0d", errors);
    if (errors == 0)
        $display("  PASS");
    else
        $display("  FAIL");
    $display("==============================");
endtask

endmodule
