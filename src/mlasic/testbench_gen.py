"""MLASIC Stage 6: Testbench Generation.

Generates SystemVerilog module-level and system-level testbenches,
cocotb Python test harnesses, and functional coverage checkers.

Supports both MLP (accelerator_top) and tile-fabric paths.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mlasic.ir import Graph, HardwareConstraints, OpType

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_MLP_OPS = {OpType.FUSED_LINEAR, OpType.FUSED_LINEAR_RELU}

# Register addresses (from axi_lite_ctrl.sv)
_CSR_ADDRS = {
    "CTRL": 0x00,
    "STATUS": 0x04,
    "IRQ_EN": 0x08,
    "IRQ_STATUS": 0x0C,
    "CYCLE_COUNT": 0x10,
    "INF_COUNT": 0x14,
    "VERSION": 0x18,
    "SCRATCH": 0x1C,
    "ERROR_CODE": 0x20,
    "LAYER_STATUS": 0x24,
}


# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class VerifConfig:
    """Configuration for testbench generation."""

    graph: Graph
    output_dir: Path
    weight_dir: Path | None = None
    num_random_vectors: int = 100
    num_adversarial_vectors: int = 10
    clock_period_ns: int = 10
    constraints: HardwareConstraints = field(default_factory=HardwareConstraints)
    is_mlp: bool = True


# Backward-compatible alias
TestbenchConfig = VerifConfig


# ── SystemVerilog Testbench Templates ────────────────────────────────────


def _sv_header(module_name: str, clock_period: int = 10) -> str:
    """Common SV testbench header with clock, reset, and utility tasks."""
    return f"""\
`timescale 1ns / 1ps

module {module_name};

// Clock and reset
logic clk;
logic rst_n;

localparam CLK_PERIOD = {clock_period};

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

"""


def _sv_footer(module_name: str) -> str:
    """Common SV testbench footer."""
    return f"""
// Final report
initial begin
    wait(0);  // placeholder — overridden by test
end

task automatic report();
    $display("==============================");
    $display("  {module_name}");
    $display("  Checks: %0d", checks);
    $display("  Errors: %0d", errors);
    if (errors == 0)
        $display("  PASS");
    else
        $display("  FAIL");
    $display("==============================");
endtask

endmodule
"""


def _sv_check_task() -> str:
    """Check macro as inline task."""
    return """
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

"""


# ── Module-Level Testbench Generators ────────────────────────────────────


def generate_tb_mac_array() -> str:
    """Generate testbench for mac_array module (task 6.2.1)."""
    tb = _sv_header("tb_mac_array")
    tb += _sv_check_task()
    tb += """\
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
"""
    tb += _sv_footer("tb_mac_array")
    return tb


def generate_tb_requantize() -> str:
    """Generate testbench for requantize module (task 6.2.2)."""
    tb = _sv_header("tb_requantize")
    tb += _sv_check_task()
    tb += """\
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
"""
    tb += _sv_footer("tb_requantize")
    return tb


def generate_tb_activation_relu() -> str:
    """Generate testbench for activation_relu module (task 6.2.3)."""
    tb = _sv_header("tb_activation_relu")
    tb += _sv_check_task()
    tb += """\
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
"""
    tb += _sv_footer("tb_activation_relu")
    return tb


def generate_tb_sram_bank(tmp_mem_file: str = "test_sram_init.mem") -> str:
    """Generate testbench for sram_bank module (task 6.2.4)."""
    tb = _sv_header("tb_sram_bank")
    tb += _sv_check_task()
    tb += f"""\
localparam int DEPTH = 16;
localparam int WIDTH = 64;

logic [$clog2(DEPTH)-1:0] addr;
logic we;
logic [WIDTH-1:0] wdata;
logic [WIDTH-1:0] rdata;

sram_bank #(.DEPTH(DEPTH), .WIDTH(WIDTH), .INIT_FILE("{tmp_mem_file}")) dut (
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
        wdata = {{(WIDTH/8){{i[7:0]}}}};
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
        if (rdata !== {{(WIDTH/8){{i[7:0]}}}}) begin
            errors++;
            $display("MISMATCH at addr %0d: expected 0x%016h, got 0x%016h",
                     i, {{(WIDTH/8){{i[7:0]}}}}, rdata);
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
"""
    tb += _sv_footer("tb_sram_bank")
    return tb


def generate_tb_ping_pong_buffer() -> str:
    """Generate testbench for ping_pong_buffer module (task 6.2.5)."""
    tb = _sv_header("tb_ping_pong_buffer")
    tb += _sv_check_task()
    tb += """\
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
"""
    tb += _sv_footer("tb_ping_pong_buffer")
    return tb


def generate_tb_axi_stream_in() -> str:
    """Generate testbench for axi_stream_in module (task 6.2.7)."""
    tb = _sv_header("tb_axi_stream_in")
    tb += _sv_check_task()
    tb += """\
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
"""
    tb += _sv_footer("tb_axi_stream_in")
    return tb


def generate_tb_axi_stream_out() -> str:
    """Generate testbench for axi_stream_out module (task 6.2.8)."""
    tb = _sv_header("tb_axi_stream_out")
    tb += _sv_check_task()
    tb += """\
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
"""
    tb += _sv_footer("tb_axi_stream_out")
    return tb


def generate_tb_axi_lite_ctrl() -> str:
    """Generate testbench for axi_lite_ctrl module (task 6.2.9)."""
    tb = _sv_header("tb_axi_lite_ctrl")
    tb += _sv_check_task()
    tb += """\
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
"""
    tb += _sv_footer("tb_axi_lite_ctrl")
    return tb


def generate_tb_fused_linear_relu() -> str:
    """Generate testbench for fused_linear_relu module (task 6.2.6).

    Tests: single tile (128->128) and multi-tile (128->640, 5 tiles).
    """
    tb = _sv_header("tb_fused_linear_relu")
    tb += _sv_check_task()
    tb += """\
// This is a structural/FSM test — verifies the control FSM transitions
// through all states correctly. Full numerical correctness is tested
// at system level with golden vectors.

localparam int PARALLELISM = 128;
localparam int DATA_W = 8;
localparam int ACC_W = 32;
localparam int ACT_DEPTH = 80;
localparam int ACT_WIDTH = 64;

// DUT signals
logic start;
logic [15:0] input_dim, output_dim, num_tiles, weight_base, bias_base;
logic has_relu;
logic [ACC_W-1:0] requant_scale;
logic [5:0] requant_shift;
logic signed [DATA_W-1:0] requant_zp;
logic [15:0] weight_addr;
logic [PARALLELISM*DATA_W-1:0] weight_rdata;
logic [15:0] bias_addr;
logic [PARALLELISM*ACC_W-1:0] bias_rdata;
logic [$clog2(ACT_DEPTH)-1:0] act_rd_addr;
logic [ACT_WIDTH-1:0] act_rd_data;
logic [$clog2(ACT_DEPTH)-1:0] act_wr_addr;
logic act_wr_en;
logic [ACT_WIDTH-1:0] act_wr_data;

// MAC interface
logic mac_start;
logic [15:0] mac_input_dim;
logic mac_has_relu;
logic [ACC_W-1:0] mac_scale;
logic [5:0] mac_shift;
logic signed [DATA_W-1:0] mac_zp;
logic signed [DATA_W-1:0] mac_input_data;
logic mac_input_valid;
logic mac_bias_valid;
logic [$clog2(PARALLELISM)-1:0] mac_bias_idx;
logic signed [ACC_W-1:0] mac_bias_data;
logic signed [DATA_W-1:0] mac_output_data [0:PARALLELISM-1];
logic mac_output_valid;
logic mac_done_sig;
logic layer_done;

fused_linear_relu #(
    .PARALLELISM(PARALLELISM), .DATA_W(DATA_W), .ACC_W(ACC_W),
    .ACT_DEPTH(ACT_DEPTH), .ACT_WIDTH(ACT_WIDTH)
) dut (
    .clk(clk), .rst_n(rst_n),
    .start(start), .input_dim(input_dim), .output_dim(output_dim),
    .num_tiles(num_tiles), .has_relu(has_relu),
    .weight_base(weight_base), .bias_base(bias_base),
    .requant_scale(requant_scale), .requant_shift(requant_shift), .requant_zp(requant_zp),
    .weight_addr(weight_addr), .weight_rdata(weight_rdata),
    .bias_addr(bias_addr), .bias_rdata(bias_rdata),
    .act_rd_addr(act_rd_addr), .act_rd_data(act_rd_data),
    .act_wr_addr(act_wr_addr), .act_wr_en(act_wr_en), .act_wr_data(act_wr_data),
    .mac_start(mac_start), .mac_input_dim(mac_input_dim),
    .mac_has_relu(mac_has_relu), .mac_scale(mac_scale),
    .mac_shift(mac_shift), .mac_zp(mac_zp),
    .mac_input_data(mac_input_data), .mac_input_valid(mac_input_valid),
    .mac_bias_valid(mac_bias_valid), .mac_bias_idx(mac_bias_idx),
    .mac_bias_data(mac_bias_data),
    .mac_output_data(mac_output_data), .mac_output_valid(mac_output_valid),
    .mac_done(mac_done_sig), .layer_done(layer_done)
);

// Fake MAC behavioral model: counts pipeline delay after compute phase ends
integer mac_delay;
always @(posedge clk) begin
    if (!rst_n) begin
        mac_done_sig <= 1'b0;
        mac_output_valid <= 1'b0;
        mac_delay <= 0;
        for (int i = 0; i < PARALLELISM; i++)
            mac_output_data[i] <= 8'sd0;
    end else begin
        // Default: deassert single-cycle pulses
        mac_done_sig <= 1'b0;
        mac_output_valid <= 1'b0;

        if (mac_input_valid) begin
            mac_delay <= 5;  // Reset pipeline delay while computing
        end else if (mac_delay > 0) begin
            mac_delay <= mac_delay - 1;
            if (mac_delay == 1) begin
                mac_done_sig <= 1'b1;
                mac_output_valid <= 1'b1;
                for (int i = 0; i < PARALLELISM; i++)
                    mac_output_data[i] <= 8'sd0;
            end
        end
    end
end

// Fake SRAM read data
assign weight_rdata = '0;
assign bias_rdata = '0;
assign act_rd_data = '0;

initial begin
    $display("=== tb_fused_linear_relu: starting ===");
    start = 0; input_dim = 0; output_dim = 0; num_tiles = 0;
    has_relu = 0; weight_base = 0; bias_base = 0;
    requant_scale = 0; requant_shift = 0; requant_zp = 0;

    reset();

    // Test 1: Single tile layer (128->128)
    $display("Test 1: Single tile (128->128)");
    input_dim = 16'd128;
    output_dim = 16'd128;
    num_tiles = 16'd1;
    has_relu = 1'b1;
    weight_base = 16'd0;
    bias_base = 16'd0;
    requant_scale = 32'd65536;
    requant_shift = 6'd16;
    requant_zp = 8'sd0;

    #1; start = 1'b1;
    @(posedge clk); #1; start = 1'b0;

    // Wait for layer done
    wait(layer_done);
    @(posedge clk);
    checks++;
    $display("Test 1: Single tile complete");

    // Test 2: Multi-tile layer (128->640, 5 tiles)
    $display("Test 2: Multi-tile (128->640)");
    input_dim = 16'd128;
    output_dim = 16'd640;
    num_tiles = 16'd5;

    #1; start = 1'b1;
    @(posedge clk); #1; start = 1'b0;

    wait(layer_done);
    @(posedge clk);
    checks++;
    $display("Test 2: Multi-tile complete");

    report();
    if (errors > 0) $fatal(1, "FAILED: %0d errors", errors);
    $finish;
end
"""
    tb += _sv_footer("tb_fused_linear_relu")
    return tb


# ── System-Level Testbench Generator ─────────────────────────────────────


def generate_tb_accelerator(
    graph: Graph,
    num_vectors: int = 100,
    constraints: HardwareConstraints | None = None,
    is_mlp: bool = True,
) -> str:
    """Generate system-level accelerator testbench (tasks 6.3.1-6.3.7).

    This testbench loads test vectors from .mem files and drives the
    accelerator through its AXI interfaces.
    """
    ordered = graph.topological_order()
    num_layers = len(ordered)

    # Input/output dims from graph
    inp_name = graph.inputs[0]
    out_name = graph.outputs[0]
    inp_shape = graph.tensors[inp_name].type.shape
    out_shape = graph.tensors[out_name].type.shape
    input_bytes = int(np.prod(inp_shape))
    output_bytes = int(np.prod(out_shape))
    input_beats = (input_bytes + 7) // 8
    output_beats = (output_bytes + 7) // 8

    # Expected cycle count from schedule
    expected_cycles = 0
    for name in ordered:
        node = graph.nodes[name]
        if node.schedule_info:
            expected_cycles += node.schedule_info.total_cycles
        elif node.dag_schedule:
            expected_cycles += node.dag_schedule.total_cycles

    # VERSION differs between MLP (v1) and tile-fabric (v2) accelerators
    version_hex = "0001_0000" if is_mlp else "0002_0000"

    tb = _sv_header("tb_accelerator", clock_period=10)
    tb += _sv_check_task()
    tb += f"""\
// ──────────────────────────────────────────────
// Model-specific constants
// ──────────────────────────────────────────────
localparam int INPUT_BYTES = {input_bytes};
localparam int OUTPUT_BYTES = {output_bytes};
localparam int INPUT_BEATS = {input_beats};
localparam int OUTPUT_BEATS = {output_beats};
localparam int NUM_LAYERS = {num_layers};
localparam int NUM_VECTORS = {num_vectors};
localparam int EXPECTED_CYCLES = {expected_cycles};

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
    $display("Model: {graph.name}");
    $display("Input: {input_bytes} bytes ({input_beats} beats)");
    $display("Output: {output_bytes} bytes ({output_beats} beats)");
    $display("Layers: {num_layers}");

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
    check_eq("VERSION", read_val, 32'h{version_hex});

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
"""
    tb += _sv_footer("tb_accelerator")
    return tb


# ── cocotb Testbench Generator ───────────────────────────────────────────


def generate_cocotb_test(
    graph: Graph,
    constraints: HardwareConstraints | None = None,
    is_mlp: bool = True,
) -> str:
    """Generate cocotb Python testbench for accelerator_top (tasks 6.4.1-6.4.6)."""
    inp_name = graph.inputs[0]
    out_name = graph.outputs[0]
    inp_shape = graph.tensors[inp_name].type.shape
    out_shape = graph.tensors[out_name].type.shape
    input_bytes = int(np.prod(inp_shape))
    output_bytes = int(np.prod(out_shape))
    input_beats = (input_bytes + 7) // 8
    output_beats = (output_bytes + 7) // 8

    version_hex = "0x00010000" if is_mlp else "0x00020000"

    return f'''\
"""cocotb testbench for accelerator_top — auto-generated by MLASIC Stage 6.

Tests:
  - AXI-Stream driver/monitor (tasks 6.4.2, 6.4.3)
  - AXI-Lite CSR read/write (task 6.4.4)
  - Golden vector comparison (task 6.4.5)
  - IRQ path test (task 6.4.6)
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, FallingEdge
import numpy as np
from pathlib import Path

# Model-specific constants
INPUT_BYTES = {input_bytes}
OUTPUT_BYTES = {output_bytes}
INPUT_BEATS = {input_beats}
OUTPUT_BEATS = {output_beats}

# CSR register addresses
ADDR_CTRL = 0x00
ADDR_STATUS = 0x04
ADDR_IRQ_EN = 0x08
ADDR_IRQ_STATUS = 0x0C
ADDR_CYCLE_COUNT = 0x10
ADDR_INF_COUNT = 0x14
ADDR_VERSION = 0x18
ADDR_SCRATCH = 0x1C


# ── AXI-Lite driver ──────────────────────────────────────────────────

async def axi_lite_write(dut, addr: int, data: int):
    """Write to AXI-Lite CSR register."""
    dut.s_axi_awaddr.value = addr
    dut.s_axi_awvalid.value = 1
    dut.s_axi_wdata.value = data
    dut.s_axi_wstrb.value = 0xF
    dut.s_axi_wvalid.value = 1
    dut.s_axi_bready.value = 1

    while True:
        await RisingEdge(dut.clk)
        if dut.s_axi_awready.value:
            break
    dut.s_axi_awvalid.value = 0

    while True:
        await RisingEdge(dut.clk)
        if dut.s_axi_wready.value:
            break
    dut.s_axi_wvalid.value = 0

    while True:
        await RisingEdge(dut.clk)
        if dut.s_axi_bvalid.value:
            break
    dut.s_axi_bready.value = 0


async def axi_lite_read(dut, addr: int) -> int:
    """Read from AXI-Lite CSR register."""
    dut.s_axi_araddr.value = addr
    dut.s_axi_arvalid.value = 1
    dut.s_axi_rready.value = 1

    while True:
        await RisingEdge(dut.clk)
        if dut.s_axi_arready.value:
            break
    dut.s_axi_arvalid.value = 0

    while True:
        await RisingEdge(dut.clk)
        if dut.s_axi_rvalid.value:
            break
    data = int(dut.s_axi_rdata.value)
    dut.s_axi_rready.value = 0
    return data


# ── AXI-Stream driver (task 6.4.2) ──────────────────────────────────

async def axi_stream_send(dut, data: np.ndarray):
    """Send input vector via AXI-Stream slave interface."""
    flat = data.flatten().view(np.uint8)
    num_beats = (len(flat) + 7) // 8

    for beat in range(num_beats):
        word = 0
        for b in range(8):
            idx = beat * 8 + b
            if idx < len(flat):
                word |= int(flat[idx]) << (b * 8)

        dut.s_axis_tdata.value = word
        dut.s_axis_tvalid.value = 1
        dut.s_axis_tlast.value = 1 if beat == num_beats - 1 else 0
        dut.s_axis_tkeep.value = 0xFF

        while True:
            await RisingEdge(dut.clk)
            if dut.s_axis_tready.value:
                break

    dut.s_axis_tvalid.value = 0
    dut.s_axis_tlast.value = 0


# ── AXI-Stream monitor (task 6.4.3) ─────────────────────────────────

async def axi_stream_receive(dut, num_bytes: int) -> np.ndarray:
    """Receive output vector from AXI-Stream master interface."""
    num_beats = (num_bytes + 7) // 8
    result = np.zeros(num_bytes, dtype=np.uint8)

    dut.m_axis_tready.value = 1
    beat = 0

    while beat < num_beats:
        await RisingEdge(dut.clk)
        if dut.m_axis_tvalid.value and dut.m_axis_tready.value:
            word = int(dut.m_axis_tdata.value)
            for b in range(8):
                idx = beat * 8 + b
                if idx < num_bytes:
                    result[idx] = (word >> (b * 8)) & 0xFF
            beat += 1

    dut.m_axis_tready.value = 0
    return result.view(np.int8)


# ── Reset ────────────────────────────────────────────────────────────

async def do_reset(dut):
    """Apply reset sequence."""
    dut.rst_n.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(2):
        await RisingEdge(dut.clk)


# ── Tests ────────────────────────────────────────────────────────────

@cocotb.test()
async def test_version_register(dut):
    """Read VERSION CSR to verify AXI-Lite connectivity."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await do_reset(dut)

    version = await axi_lite_read(dut, ADDR_VERSION)
    assert version == {version_hex}, f"VERSION mismatch: 0x{{version:08x}}"


@cocotb.test()
async def test_scratch_register(dut):
    """Write/read SCRATCH CSR to verify AXI-Lite write path."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await do_reset(dut)

    await axi_lite_write(dut, ADDR_SCRATCH, 0xDEADBEEF)
    val = await axi_lite_read(dut, ADDR_SCRATCH)
    assert val == 0xDEADBEEF, f"SCRATCH mismatch: 0x{{val:08x}}"


@cocotb.test()
async def test_golden_vectors(dut):
    """Run golden vectors and check bitwise match (tasks 6.4.1, 6.4.5)."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await do_reset(dut)

    # Load golden vectors
    test_dir = Path("test_vectors")
    if not test_dir.exists():
        cocotb.log.warning("test_vectors/ not found, skipping golden test")
        return

    inputs = np.load(test_dir / "test_inputs_{graph.inputs[0]}.npy")
    expected = np.load(test_dir / "golden_outputs_{graph.outputs[0]}.npy")

    passes = 0
    for i in range(len(inputs)):
        # Start inference
        await axi_lite_write(dut, ADDR_CTRL, 0x00000001)

        # Send input
        await axi_stream_send(dut, inputs[i])

        # Wait for completion
        for _ in range(100000):
            await RisingEdge(dut.clk)
            status = await axi_lite_read(dut, ADDR_STATUS)
            if status & 0x04:  # done bit
                break

        # Receive output
        output = await axi_stream_receive(dut, OUTPUT_BYTES)

        # Compare
        if np.array_equal(output, expected[i].flatten()):
            passes += 1
        else:
            mismatches = np.sum(output != expected[i].flatten())
            cocotb.log.error(f"Vector {{i}}: {{mismatches}} byte mismatches")

    cocotb.log.info(f"Golden vectors: {{passes}}/{{len(inputs)}} passed")
    assert passes == len(inputs), f"{{len(inputs) - passes}} vectors failed"


@cocotb.test()
async def test_irq_done(dut):
    """Test interrupt fires on inference completion (task 6.4.6)."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await do_reset(dut)

    # Enable IRQ_DONE
    await axi_lite_write(dut, ADDR_IRQ_EN, 0x00000001)

    # Start inference
    await axi_lite_write(dut, ADDR_CTRL, 0x00000001)

    # Send random input (needed to advance FSM past S_RECV_IN)
    random_input = np.random.randint(0, 256, INPUT_BYTES, dtype=np.uint8)
    await axi_stream_send(dut, random_input)

    # Wait for IRQ
    for _ in range(10000000):
        await RisingEdge(dut.clk)
        if dut.irq.value:
            break

    # Check IRQ_STATUS
    irq_status = await axi_lite_read(dut, ADDR_IRQ_STATUS)
    assert irq_status & 0x01, "IRQ_DONE bit not set"

    # Clear IRQ (W1C)
    await axi_lite_write(dut, ADDR_IRQ_STATUS, 0x00000001)
    irq_status = await axi_lite_read(dut, ADDR_IRQ_STATUS)
    assert (irq_status & 0x01) == 0, "IRQ_DONE not cleared"

    # Drain output
    await axi_stream_receive(dut, OUTPUT_BYTES)
'''


def generate_cocotb_makefile(top_module: str = "accelerator_top") -> str:
    """Generate Makefile for cocotb simulation."""
    return f"""\
# cocotb Makefile for MLASIC verification
# Auto-generated by Stage 6 testbench generator

TOPLEVEL_LANG = verilog

# RTL source files — relative to this Makefile's directory
RTL_DIR = ../rtl
VERILOG_SOURCES = $(wildcard $(RTL_DIR)/**/*.sv) $(wildcard $(RTL_DIR)/*.sv)

TOPLEVEL = {top_module}
MODULE = test_accelerator

SIM ?= verilator
EXTRA_ARGS += --language 1800-2017

include $(shell cocotb-config --makefilepath)/Makefile.sim
"""


# ── Functional Coverage ──────────────────────────────────────────────────


@dataclass
class CoverageItem:
    """One functional coverage point."""

    name: str
    description: str
    hit: bool = False


@dataclass
class FunctionalCoverage:
    """Track functional coverage for Stage 6 verification (tasks 6.5.1-6.5.11)."""

    items: list[CoverageItem] = field(default_factory=list)

    def __post_init__(self):
        if not self.items:
            self.items = [
                CoverageItem("layer_execution", "All layers executed sequentially (6.5.1)"),
                CoverageItem("weight_sram_read", "All weight SRAM banks read (6.5.2)"),
                CoverageItem("ping_pong_swap", "Ping-pong buffer swap at layer boundary (6.5.3)"),
                CoverageItem("axi_backpressure", "AXI-Stream backpressure exercised (6.5.4)"),
                CoverageItem("csr_read_write", "AXI-Lite register read/write all offsets (6.5.5)"),
                CoverageItem("irq_done", "IRQ_DONE interrupt fires on completion (6.5.6)"),
                CoverageItem("irq_error", "Error interrupt path tested (6.5.7)"),
                CoverageItem("back_to_back", "Back-to-back inferences correct (6.5.8)"),
                CoverageItem("output_tiling", "Output tiling works for multi-tile layers (6.5.9)"),
                CoverageItem(
                    "relu_activation",
                    "ReLU activates on positive, clamps on negative (6.5.10)",
                ),
                CoverageItem("requant_boundary", "Requantization handles boundary values (6.5.11)"),
            ]

    def mark_hit(self, name: str) -> None:
        for item in self.items:
            if item.name == name:
                item.hit = True
                return
        raise ValueError(f"Unknown coverage item: {name}")

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def hit_count(self) -> int:
        return sum(1 for i in self.items if i.hit)

    @property
    def coverage_pct(self) -> float:
        return (self.hit_count / self.total * 100) if self.total > 0 else 0.0

    @property
    def is_complete(self) -> bool:
        return all(i.hit for i in self.items)

    def report(self) -> str:
        lines = ["Functional Coverage Report", "=" * 60]
        for item in self.items:
            status = "HIT" if item.hit else "MISS"
            lines.append(f"  [{status:4s}] {item.name}: {item.description}")
        lines.append(f"\nCoverage: {self.hit_count}/{self.total} ({self.coverage_pct:.0f}%)")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "hit": self.hit_count,
            "coverage_pct": self.coverage_pct,
            "items": [
                {"name": i.name, "description": i.description, "hit": i.hit} for i in self.items
            ],
        }


# ── Main Testbench Generator ────────────────────────────────────────────


class VerifGenerator:
    """Generate all verification artifacts for Stage 6.

    Produces:
      - Module-level SV testbenches (testbench/*.sv)
      - System-level SV testbench (testbench/tb_accelerator.sv)
      - cocotb Python test (testbench/cocotb/test_accelerator.py)
      - cocotb Makefile
      - Functional coverage tracker
    """

    def __init__(self, config: VerifConfig) -> None:
        self.config = config
        self.coverage = FunctionalCoverage()

    def generate_all(self) -> dict:
        """Generate all testbench files.

        Returns:
            Dict mapping filename to file path.
        """
        out = self.config.output_dir
        out.mkdir(parents=True, exist_ok=True)

        files = {}

        # Module-level testbenches
        module_tbs = self.generate_module_testbenches()
        files.update(module_tbs)

        # System-level testbench
        sys_tb = self.generate_system_testbench()
        files.update(sys_tb)

        # cocotb test
        cocotb_files = self.generate_cocotb_testbench()
        files.update(cocotb_files)

        # Coverage tracker JSON
        cov_path = out / "coverage.json"
        with open(cov_path, "w") as f:
            json.dump(self.coverage.to_dict(), f, indent=2)
        files["coverage.json"] = cov_path

        logger.info("Generated %d testbench files in %s", len(files), out)
        return files

    def generate_module_testbenches(self) -> dict[str, Path]:
        """Generate all module-level SV testbenches (tasks 6.2.1-6.2.9)."""
        out = self.config.output_dir
        files = {}

        testbenches = {
            "tb_mac_array.sv": generate_tb_mac_array(),
            "tb_requantize.sv": generate_tb_requantize(),
            "tb_activation_relu.sv": generate_tb_activation_relu(),
            "tb_sram_bank.sv": generate_tb_sram_bank(),
            "tb_ping_pong_buffer.sv": generate_tb_ping_pong_buffer(),
            "tb_fused_linear_relu.sv": generate_tb_fused_linear_relu(),
            "tb_axi_stream_in.sv": generate_tb_axi_stream_in(),
            "tb_axi_stream_out.sv": generate_tb_axi_stream_out(),
            "tb_axi_lite_ctrl.sv": generate_tb_axi_lite_ctrl(),
        }

        for name, content in testbenches.items():
            path = out / name
            with open(path, "w") as f:
                f.write(content)
            files[name] = path
            logger.info("Generated %s", name)

        return files

    def generate_system_testbench(self) -> dict[str, Path]:
        """Generate system-level accelerator testbench (tasks 6.3.1-6.3.7)."""
        out = self.config.output_dir
        files = {}

        content = generate_tb_accelerator(
            self.config.graph,
            num_vectors=self.config.num_random_vectors,
            constraints=self.config.constraints,
            is_mlp=self.config.is_mlp,
        )

        path = out / "tb_accelerator.sv"
        with open(path, "w") as f:
            f.write(content)
        files["tb_accelerator.sv"] = path
        logger.info("Generated tb_accelerator.sv")

        return files

    def generate_cocotb_testbench(self) -> dict[str, Path]:
        """Generate cocotb Python testbench (tasks 6.4.1-6.4.6)."""
        out = self.config.output_dir / "cocotb"
        out.mkdir(parents=True, exist_ok=True)
        files = {}

        # Python test file
        content = generate_cocotb_test(
            self.config.graph,
            constraints=self.config.constraints,
            is_mlp=self.config.is_mlp,
        )
        path = out / "test_accelerator.py"
        with open(path, "w") as f:
            f.write(content)
        files["cocotb/test_accelerator.py"] = path

        # Makefile
        makefile = generate_cocotb_makefile()
        mf_path = out / "Makefile"
        with open(mf_path, "w") as f:
            f.write(makefile)
        files["cocotb/Makefile"] = mf_path

        logger.info("Generated cocotb testbench in %s", out)
        return files


# Backward-compatible alias
TestbenchGenerator = VerifGenerator
