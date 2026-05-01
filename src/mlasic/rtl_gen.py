"""MLASIC Stage 5: RTL Generation.

Generates synthesizable SystemVerilog RTL from a scheduled IR graph.
Produces model-specific parameters.svh and accelerator_top.sv, copies
the module library, and assembles a complete build directory.

Two paths:
  - MLP path: Single shared MAC array, time-multiplexed across layers.
    (original Stage 5, unchanged for backward compatibility)
  - Tile fabric path: Per-tile compute modules with static routing.
    (Stage 5.1, for CNN/Transformer/arbitrary DAG models)

See docs/rtl-interface-spec.md §9.2.
"""

from __future__ import annotations

import logging
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from mlasic.ir import (
    FusedConvAttrs,
    FusedLinearAttrs,
    Graph,
    HardwareConstraints,
    OpType,
    PoolAttrs,
)
from mlasic.tile_mapper import FabricConfig, TileConfig, TileType

logger = logging.getLogger(__name__)

# Root of the RTL module library (relative to this file)
_RTL_LIB_DIR = Path(__file__).resolve().parent.parent.parent / "rtl"

# TileType enum value → numeric TILE_TYPE parameter for tile.sv
_TILE_TYPE_ID: dict[TileType, int] = {
    TileType.MAC: 0,
    TileType.ALU: 1,
    TileType.NORM: 2,
    TileType.SOFTMAX: 3,
    TileType.ACTIVATION: 4,
    TileType.POOL: 5,
    TileType.RESHAPE: 6,
}


@dataclass
class RTLGenerator:
    """Generate SystemVerilog RTL from a scheduled graph.

    Takes a scheduled graph (stage="scheduled" or "scheduled_dag") with
    fused_attrs per node, and generates a complete RTL output directory.

    Two paths selected automatically:
      - MLP path: pure FusedLinear/FusedLinearReLU chains → existing arch
      - Tile fabric path: anything else → tile-based fabric
    """

    graph: Graph
    weight_dir: Path
    output_dir: Path
    constraints: HardwareConstraints = field(default_factory=HardwareConstraints)
    fabric_config: Optional[FabricConfig] = None
    target: str = "fpga"  # "fpga" or "asic"

    def __post_init__(self) -> None:
        self.weight_dir = Path(self.weight_dir)
        self.output_dir = Path(self.output_dir)
        if self.graph.stage not in ("scheduled", "scheduled_dag"):
            raise ValueError(
                f"RTLGenerator requires 'scheduled' or 'scheduled_dag' stage, "
                f"got '{self.graph.stage}'"
            )

    def _is_mlp_linear_chain(self) -> bool:
        """Return True only for pure FusedLinear/FusedLinearReLU chains.

        This selects the original MLP path for backward compatibility.
        Uses tile fabric path when fabric_config is provided or graph
        was DAG-scheduled, even for pure MLP models.
        """
        if self.fabric_config is not None:
            return False
        ordered_names = self.graph.topological_order()
        for name in ordered_names:
            node = self.graph.nodes[name]
            if node.op_type not in (OpType.FUSED_LINEAR, OpType.FUSED_LINEAR_RELU):
                return False
            if not isinstance(node.fused_attrs, FusedLinearAttrs):
                return False
        return True

    def generate_all(self) -> Path:
        """Generate complete RTL output. Returns output directory path."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("RTL Generation: output_dir=%s", self.output_dir)

        if self._is_mlp_linear_chain():
            # Original MLP path — identical output for backward compat
            logger.info("MLP linear chain detected — using MLP path")
            self.copy_module_library()
            self.copy_weight_files()
            self.generate_parameters_svh()
            self.generate_accelerator_top()
            self.validate()
        else:
            # Tile fabric path
            logger.info("Non-MLP model — using tile fabric path")
            self.copy_module_library()
            self.copy_weight_files()
            self.generate_lut_files()
            self.generate_requant_rom_files()
            self.generate_tile_parameters_svh()
            self.generate_tile_fabric_sv()
            self.generate_tile_accelerator_top()
            if self.target == "asic":
                self.generate_asic_rom_modules()
            self.validate_tile_fabric()

        logger.info("RTL Generation complete: %s", self.output_dir)
        return self.output_dir

    # ------------------------------------------------------------------
    # Module library copy
    # ------------------------------------------------------------------

    def copy_module_library(self) -> None:
        """Copy all .sv and .xdc files from the RTL library to output."""
        lib_dir = _RTL_LIB_DIR
        if not lib_dir.is_dir():
            raise FileNotFoundError(f"RTL library not found at {lib_dir}")

        # Copy subdirectories: compute, memory, layer, interface, constraints, tile
        for subdir in ["compute", "memory", "layer", "interface", "constraints", "tile"]:
            src = lib_dir / subdir
            if not src.is_dir():
                continue
            dst = self.output_dir / subdir
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                if f.suffix in (".sv", ".xdc"):
                    shutil.copy2(f, dst / f.name)
                    logger.debug("Copied %s -> %s", f, dst / f.name)

        logger.info("Copied RTL module library (%s)", lib_dir)

    # ------------------------------------------------------------------
    # Weight file copy
    # ------------------------------------------------------------------

    def copy_weight_files(self) -> None:
        """Copy .mem files from Stage 4 weight directory to output."""
        if not self.weight_dir.is_dir():
            raise FileNotFoundError(f"Weight directory not found: {self.weight_dir}")

        for f in self.weight_dir.iterdir():
            if f.suffix == ".mem":
                shutil.copy2(f, self.output_dir / f.name)
                logger.debug("Copied %s -> %s", f, self.output_dir / f.name)

        logger.info("Copied weight files from %s", self.weight_dir)

    # ------------------------------------------------------------------
    # parameters.svh generation
    # ------------------------------------------------------------------

    def generate_parameters_svh(self) -> Path:
        """Generate model-specific parameters.svh from schedule and graph."""
        ordered_names = self.graph.topological_order()
        ordered_nodes = [self.graph.nodes[n] for n in ordered_names]
        num_layers = len(ordered_nodes)

        # Collect per-layer arrays
        in_dims: list[int] = []
        out_dims: list[int] = []
        relus: list[bool] = []
        weight_bases: list[int] = []
        bias_bases: list[int] = []
        requant_ms: list[int] = []
        requant_shifts: list[int] = []
        requant_zps: list[int] = []
        num_tiles_arr: list[int] = []

        max_dim = 0

        for node in ordered_nodes:
            ls = node.schedule_info
            attrs = node.fused_attrs
            assert ls is not None, f"Node {node.name} missing schedule_info"
            assert isinstance(attrs, FusedLinearAttrs), f"Node {node.name} missing FusedLinearAttrs"

            in_dims.append(ls.input_dim)
            out_dims.append(ls.output_dim)
            relus.append(ls.has_relu)
            weight_bases.append(ls.weight_start_row)
            bias_bases.append(ls.bias_start_row)
            num_tiles_arr.append(ls.num_tiles)

            # Requantization params from fused_attrs
            requant_ms.append(attrs.requant_scale_fixed or 0)
            requant_shifts.append(attrs.requant_shift or 0)
            if attrs.output_quant is not None:
                requant_zps.append(attrs.output_quant.zero_point)
            else:
                requant_zps.append(0)

            max_dim = max(max_dim, ls.input_dim, ls.output_dim)

        # Compute SRAM depths from constraints
        weight_depth = self.constraints.weight_bank_depth
        bias_depth = self.constraints.bias_bank_depth
        act_depth = max_dim // 8  # 64-bit words = max_dim / 8 bytes

        # Total weight/bias rows actually used
        total_weight_rows = sum(n.schedule_info.weight_rows for n in ordered_nodes)
        total_bias_rows = sum(n.schedule_info.bias_rows for n in ordered_nodes)

        last_idx = num_layers - 1

        lines = [
            "// parameters.svh — Auto-generated by MLASIC RTL Generator",
            f"// Model: {self.graph.name}",
            f"// Layers: {num_layers}",
            "",
            "/* verilator lint_off UNUSEDPARAM */",
            "",
            f"localparam int NUM_LAYERS   = {num_layers};",
            f"localparam int MAX_DIM      = {max_dim};",
            f"localparam int PARALLELISM  = {self.constraints.max_parallelism};",
            "localparam int AXI_DATA_W   = 64;",
            f"localparam int WEIGHT_DEPTH = {weight_depth};",
            f"localparam int BIAS_DEPTH   = {bias_depth};",
            f"localparam int ACT_DEPTH    = {act_depth};",
            f"localparam int TOTAL_WEIGHT_ROWS = {total_weight_rows};",
            f"localparam int TOTAL_BIAS_ROWS   = {total_bias_rows};",
            "",
            "// Per-layer dimensions",
            f"localparam logic [15:0] LAYER_IN_DIM  [0:{last_idx}] = "
            + "'{"
            + ", ".join(f"16'd{d}" for d in in_dims)
            + "};",
            f"localparam logic [15:0] LAYER_OUT_DIM [0:{last_idx}] = "
            + "'{"
            + ", ".join(f"16'd{d}" for d in out_dims)
            + "};",
            f"localparam logic [15:0] LAYER_NUM_TILES [0:{last_idx}] = "
            + "'{"
            + ", ".join(f"16'd{t}" for t in num_tiles_arr)
            + "};",
            f"localparam logic [0:0]  LAYER_RELU   [0:{last_idx}] = "
            + "'{"
            + ", ".join(f"1'b{int(r)}" for r in relus)
            + "};",
            "",
            "// Weight/bias SRAM base addresses",
            f"localparam logic [15:0] WEIGHT_BASE [0:{last_idx}] = "
            + "'{"
            + ", ".join(f"16'd{b}" for b in weight_bases)
            + "};",
            f"localparam logic [15:0] BIAS_BASE   [0:{last_idx}] = "
            + "'{"
            + ", ".join(f"16'd{b}" for b in bias_bases)
            + "};",
            "",
            "// Requantization parameters",
            f"localparam logic [31:0] REQUANT_M     [0:{last_idx}] = "
            + "'{"
            + ", ".join(f"32'd{m}" for m in requant_ms)
            + "};",
            f"localparam logic [5:0]  REQUANT_SHIFT [0:{last_idx}] = "
            + "'{"
            + ", ".join(f"6'd{s}" for s in requant_shifts)
            + "};",
            f"localparam logic signed [7:0]  REQUANT_ZP    [0:{last_idx}] = "
            + "'{"
            + ", ".join(
                f"-8'sd{abs(z)}" if z < 0 else f"8'sd{z}" for z in requant_zps
            )
            + "};",
        ]

        out_path = self.output_dir / "parameters.svh"
        out_path.write_text("\n".join(lines) + "\n")
        logger.info("Generated %s (%d layers)", out_path, num_layers)
        return out_path

    # ------------------------------------------------------------------
    # accelerator_top.sv generation
    # ------------------------------------------------------------------

    def generate_accelerator_top(self) -> Path:
        """Generate model-specific accelerator_top.sv."""
        ordered_names = self.graph.topological_order()
        num_layers = len(ordered_names)

        # Compute max dims for parameters
        max_dim = 0
        for name in ordered_names:
            ls = self.graph.nodes[name].schedule_info
            max_dim = max(max_dim, ls.input_dim, ls.output_dim)

        parallelism = self.constraints.max_parallelism

        sv = f"""\
// accelerator_top.sv — Auto-generated by MLASIC RTL Generator
// Model: {self.graph.name}
// Layers: {num_layers}

`include "parameters.svh"

module accelerator_top (
    // Clock and reset
    input  logic        clk,
    input  logic        rst_n,

    // AXI-Lite control
    input  logic [7:0]  s_axi_awaddr,
    input  logic        s_axi_awvalid,
    output logic        s_axi_awready,
    input  logic [31:0] s_axi_wdata,
    input  logic [3:0]  s_axi_wstrb,
    input  logic        s_axi_wvalid,
    output logic        s_axi_wready,
    output logic [1:0]  s_axi_bresp,
    output logic        s_axi_bvalid,
    input  logic        s_axi_bready,
    input  logic [7:0]  s_axi_araddr,
    input  logic        s_axi_arvalid,
    output logic        s_axi_arready,
    output logic [31:0] s_axi_rdata,
    output logic [1:0]  s_axi_rresp,
    output logic        s_axi_rvalid,
    input  logic        s_axi_rready,

    // AXI-Stream input
    input  logic [AXI_DATA_W-1:0] s_axis_tdata,
    input  logic        s_axis_tvalid,
    output logic        s_axis_tready,
    input  logic        s_axis_tlast,
    input  logic [7:0]  s_axis_tkeep,

    // AXI-Stream output
    output logic [AXI_DATA_W-1:0] m_axis_tdata,
    output logic        m_axis_tvalid,
    input  logic        m_axis_tready,
    output logic        m_axis_tlast,
    output logic [7:0]  m_axis_tkeep,

    // Interrupt
    output logic        irq
);

// ======================================================================
// Top-level FSM
// ======================================================================

typedef enum logic [2:0] {{
    S_IDLE,
    S_RECV_IN,
    S_RUN_LAYER,
    S_NEXT_LAYER,
    S_SEND_OUT,
    S_DONE
}} accel_state_t;

accel_state_t state, state_next;
logic [15:0] layer_idx;

// ======================================================================
// Internal signals
// ======================================================================

/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off PINMISSING */

// CSR control/status
logic ctrl_start, ctrl_soft_rst, ctrl_continuous, irq_en_out;
logic status_idle, status_busy, status_done_reg, status_error;
logic [31:0] cycle_count_reg, inf_count_reg, error_code_reg;
logic irq_done_pulse, irq_error_pulse;

// Weight SRAM
logic [15:0] weight_addr;
logic [{parallelism}*8-1:0] weight_rdata;

// Bias SRAM
logic [15:0] bias_addr;
logic [{parallelism}*32-1:0] bias_rdata;

// Activation ping-pong
logic input_bank_sel;
logic [$clog2(ACT_DEPTH)-1:0] act_rd_addr, act_wr_addr;
logic [AXI_DATA_W-1:0] act_rd_data, act_wr_data;
logic act_wr_en;

// Direct access for AXI-Stream
logic [$clog2(ACT_DEPTH)-1:0] axis_in_addr, axis_out_addr;
logic axis_in_we;
logic [AXI_DATA_W-1:0] axis_in_wdata;
logic [AXI_DATA_W-1:0] axis_out_rdata;

// AXI-Stream control
logic axis_in_enable, axis_in_done;
logic axis_out_enable, axis_out_done;

// MAC array signals
logic mac_start;
logic [15:0] mac_input_dim;
logic mac_has_relu;
logic signed [7:0] mac_input_data;
logic mac_input_valid;
logic mac_bias_valid;
logic [{int.bit_length(parallelism - 1)}-1:0] mac_bias_idx;
logic signed [31:0] mac_bias_data;
logic [31:0] mac_scale;
logic [5:0] mac_shift;
logic signed [7:0] mac_zp;
logic signed [7:0] mac_output_data [0:{parallelism}-1];
logic mac_output_valid, mac_done_sig;

// Layer controller
logic layer_start, layer_done_sig;

// Cycle counter
logic [31:0] cycle_counter;
logic counting;

// ======================================================================
// SRAM Instances
// ======================================================================

// Unified weight bank
sram_bank #(
    .DEPTH(WEIGHT_DEPTH),
    .WIDTH({parallelism}*8),
    .INIT_FILE("weight_bank.mem")
) weight_bank (
    .clk(clk),
    .addr(weight_addr[$clog2(WEIGHT_DEPTH)-1:0]),
    .we(1'b0),
    .wdata({{{parallelism}*8{{1'b0}}}}),
    .rdata(weight_rdata)
);

// Bias bank
sram_bank #(
    .DEPTH(BIAS_DEPTH),
    .WIDTH({parallelism}*32),
    .INIT_FILE("bias_bank.mem")
) bias_bank (
    .clk(clk),
    .addr(bias_addr[$clog2(BIAS_DEPTH)-1:0]),
    .we(1'b0),
    .wdata({{{parallelism}*32{{1'b0}}}}),
    .rdata(bias_rdata)
);

// Activation ping-pong buffers (A and B)
logic [$clog2(ACT_DEPTH)-1:0] act_a_addr, act_b_addr;
logic act_a_we, act_b_we;
logic [AXI_DATA_W-1:0] act_a_wdata, act_b_wdata;
logic [AXI_DATA_W-1:0] act_a_rdata, act_b_rdata;

sram_bank #(.DEPTH(ACT_DEPTH), .WIDTH(AXI_DATA_W)) act_a (
    .clk(clk), .addr(act_a_addr), .we(act_a_we),
    .wdata(act_a_wdata), .rdata(act_a_rdata)
);

sram_bank #(.DEPTH(ACT_DEPTH), .WIDTH(AXI_DATA_W)) act_b (
    .clk(clk), .addr(act_b_addr), .we(act_b_we),
    .wdata(act_b_wdata), .rdata(act_b_rdata)
);

// Bank select: layer_idx[0] determines read bank
assign input_bank_sel = layer_idx[0];

// Activation bank mux
always_comb begin
    if (state == S_RECV_IN) begin
        // AXI-Stream input writes to bank A
        act_a_addr  = axis_in_addr;
        act_a_we    = axis_in_we;
        act_a_wdata = axis_in_wdata;
        act_b_addr  = '0;
        act_b_we    = 1'b0;
        act_b_wdata = '0;
    end else if (state == S_SEND_OUT) begin
        // AXI-Stream output reads from bank A (last layer output)
        act_a_addr  = axis_out_addr;
        act_a_we    = 1'b0;
        act_a_wdata = '0;
        act_b_addr  = '0;
        act_b_we    = 1'b0;
        act_b_wdata = '0;
    end else if (!input_bank_sel) begin
        // Read from A, write to B
        act_a_addr  = act_rd_addr;
        act_a_we    = 1'b0;
        act_a_wdata = '0;
        act_b_addr  = act_wr_addr;
        act_b_we    = act_wr_en;
        act_b_wdata = act_wr_data;
    end else begin
        // Read from B, write to A
        act_b_addr  = act_rd_addr;
        act_b_we    = 1'b0;
        act_b_wdata = '0;
        act_a_addr  = act_wr_addr;
        act_a_we    = act_wr_en;
        act_a_wdata = act_wr_data;
    end
end

assign act_rd_data = input_bank_sel ? act_b_rdata : act_a_rdata;
assign axis_out_rdata = act_a_rdata;

// ======================================================================
// AXI-Lite Control
// ======================================================================

axi_lite_ctrl #(
    .VERSION(32'h0001_0000)
) u_ctrl (
    .clk(clk), .rst_n(rst_n),
    .s_axi_awaddr(s_axi_awaddr), .s_axi_awvalid(s_axi_awvalid), .s_axi_awready(s_axi_awready),
    .s_axi_wdata(s_axi_wdata), .s_axi_wstrb(s_axi_wstrb),
    .s_axi_wvalid(s_axi_wvalid), .s_axi_wready(s_axi_wready),
    .s_axi_bresp(s_axi_bresp), .s_axi_bvalid(s_axi_bvalid), .s_axi_bready(s_axi_bready),
    .s_axi_araddr(s_axi_araddr), .s_axi_arvalid(s_axi_arvalid),
    .s_axi_arready(s_axi_arready),
    .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp),
    .s_axi_rvalid(s_axi_rvalid), .s_axi_rready(s_axi_rready),
    .ctrl_start(ctrl_start), .ctrl_soft_rst(ctrl_soft_rst),
    .ctrl_continuous(ctrl_continuous), .irq_en(irq_en_out),
    .status_idle(status_idle), .status_busy(status_busy),
    .status_done(status_done_reg), .status_error(status_error),
    .cycle_count(cycle_count_reg), .inf_count(inf_count_reg), .error_code(error_code_reg),
    .layer_status(layer_idx[1:0]),
    .irq_done(irq_done_pulse), .irq_error(irq_error_pulse),
    .irq(irq)
);

// ======================================================================
// AXI-Stream Input
// ======================================================================

axi_stream_in #(
    .AXI_DATA_W(AXI_DATA_W),
    .NUM_BEATS(ACT_DEPTH),
    .ACT_DEPTH(ACT_DEPTH)
) u_axis_in (
    .clk(clk), .rst_n(rst_n),
    .enable(axis_in_enable), .done(axis_in_done),
    .s_axis_tdata(s_axis_tdata), .s_axis_tvalid(s_axis_tvalid),
    .s_axis_tready(s_axis_tready), .s_axis_tlast(s_axis_tlast), .s_axis_tkeep(s_axis_tkeep),
    .act_addr(axis_in_addr), .act_we(axis_in_we), .act_wdata(axis_in_wdata)
);

// ======================================================================
// AXI-Stream Output
// ======================================================================

axi_stream_out #(
    .AXI_DATA_W(AXI_DATA_W),
    .NUM_BEATS(ACT_DEPTH),
    .ACT_DEPTH(ACT_DEPTH)
) u_axis_out (
    .clk(clk), .rst_n(rst_n),
    .enable(axis_out_enable), .done(axis_out_done),
    .m_axis_tdata(m_axis_tdata), .m_axis_tvalid(m_axis_tvalid),
    .m_axis_tready(m_axis_tready), .m_axis_tlast(m_axis_tlast), .m_axis_tkeep(m_axis_tkeep),
    .act_addr(axis_out_addr), .act_rdata(axis_out_rdata)
);

// ======================================================================
// Shared MAC Array
// ======================================================================

mac_array #(
    .PARALLELISM(PARALLELISM),
    .DATA_W(8),
    .ACC_W(32)
) u_mac (
    .clk(clk), .rst_n(rst_n),
    .start(mac_start),
    .input_dim(mac_input_dim),
    .has_relu(mac_has_relu),
    .weight_data(weight_rdata),
    .weight_valid(1'b1),
    .input_data(mac_input_data),
    .input_valid(mac_input_valid),
    .bias_data(mac_bias_data),
    .bias_valid(mac_bias_valid),
    .bias_idx(mac_bias_idx),
    .scale(mac_scale),
    .shift(mac_shift),
    .zero_point(mac_zp),
    .output_data(mac_output_data),
    .output_valid(mac_output_valid),
    .done(mac_done_sig)
);

// ======================================================================
// Layer Controller (fused_linear_relu)
// ======================================================================

fused_linear_relu #(
    .PARALLELISM(PARALLELISM),
    .DATA_W(8),
    .ACC_W(32),
    .ACT_DEPTH(ACT_DEPTH),
    .ACT_WIDTH(AXI_DATA_W)
) u_layer_ctrl (
    .clk(clk), .rst_n(rst_n),
    .start(layer_start),
    .input_dim(LAYER_IN_DIM[layer_idx]),
    .output_dim(LAYER_OUT_DIM[layer_idx]),
    .num_tiles(LAYER_NUM_TILES[layer_idx]),
    .has_relu(LAYER_RELU[layer_idx]),
    .weight_base(WEIGHT_BASE[layer_idx]),
    .bias_base(BIAS_BASE[layer_idx]),
    .requant_scale(REQUANT_M[layer_idx]),
    .requant_shift(REQUANT_SHIFT[layer_idx]),
    .requant_zp(REQUANT_ZP[layer_idx]),
    .weight_addr(weight_addr),
    .weight_rdata(weight_rdata),
    .bias_addr(bias_addr),
    .bias_rdata(bias_rdata),
    .act_rd_addr(act_rd_addr),
    .act_rd_data(act_rd_data),
    .act_wr_addr(act_wr_addr),
    .act_wr_en(act_wr_en),
    .act_wr_data(act_wr_data),
    .mac_start(mac_start),
    .mac_input_dim(mac_input_dim),
    .mac_has_relu(mac_has_relu),
    .mac_scale(mac_scale),
    .mac_shift(mac_shift),
    .mac_zp(mac_zp),
    .mac_input_data(mac_input_data),
    .mac_input_valid(mac_input_valid),
    .mac_bias_valid(mac_bias_valid),
    .mac_bias_idx(mac_bias_idx),
    .mac_bias_data(mac_bias_data),
    .mac_output_data(mac_output_data),
    .mac_output_valid(mac_output_valid),
    .mac_done(mac_done_sig),
    .layer_done(layer_done_sig)
);

// ======================================================================
// Top-level FSM
// ======================================================================

// FSM next-state
always_comb begin
    state_next = state;
    case (state)
        S_IDLE:       if (ctrl_start) state_next = S_RECV_IN;
        S_RECV_IN:    if (axis_in_done) state_next = S_RUN_LAYER;
        S_RUN_LAYER:  if (layer_done_sig) state_next = S_NEXT_LAYER;
        S_NEXT_LAYER: begin
            if (layer_idx == NUM_LAYERS - 1)
                state_next = S_SEND_OUT;
            else
                state_next = S_RUN_LAYER;
        end
        S_SEND_OUT:   if (axis_out_done) state_next = S_DONE;
        S_DONE:       state_next = S_IDLE;
        default:      state_next = S_IDLE;
    endcase
end

// State register
always_ff @(posedge clk) begin
    if (!rst_n || ctrl_soft_rst)
        state <= S_IDLE;
    else
        state <= state_next;
end

// Layer index
always_ff @(posedge clk) begin
    if (!rst_n || state == S_IDLE)
        layer_idx <= '0;
    else if (state == S_NEXT_LAYER)
        layer_idx <= layer_idx + 1;
end

// Control signal generation
assign axis_in_enable  = (state == S_RECV_IN);
assign axis_out_enable = (state == S_SEND_OUT);
assign layer_start     = (state == S_NEXT_LAYER) || (state == S_RECV_IN && axis_in_done);

// Status
assign status_idle = (state == S_IDLE);
assign status_busy = (state != S_IDLE && state != S_DONE);
assign status_error = 1'b0;
assign error_code_reg = 32'b0;

// Done and interrupt
always_ff @(posedge clk) begin
    if (!rst_n || state == S_IDLE) begin
        status_done_reg <= 1'b0;
        irq_done_pulse  <= 1'b0;
    end else if (state == S_DONE) begin
        status_done_reg <= 1'b1;
        irq_done_pulse  <= 1'b1;
    end else begin
        irq_done_pulse <= 1'b0;
    end
end

assign irq_error_pulse = 1'b0;

// Cycle counter
always_ff @(posedge clk) begin
    if (!rst_n || state == S_IDLE) begin
        cycle_counter <= '0;
        counting <= 1'b0;
    end else if (state == S_RECV_IN && !counting) begin
        counting <= 1'b1;
        cycle_counter <= '0;
    end else if (counting && state != S_DONE) begin
        cycle_counter <= cycle_counter + 1;
    end
end

assign cycle_count_reg = cycle_counter;

// Inference counter
always_ff @(posedge clk) begin
    if (!rst_n)
        inf_count_reg <= '0;
    else if (state == S_DONE && state_next == S_IDLE)
        inf_count_reg <= inf_count_reg + 1;
end

endmodule
"""

        out_path = self.output_dir / "accelerator_top.sv"
        out_path.write_text(sv)
        logger.info("Generated %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Post-generation validation: check all expected files exist."""
        expected_sv = [
            "compute/mac_array.sv",
            "compute/requantize.sv",
            "compute/activation_relu.sv",
            "memory/sram_bank.sv",
            "memory/ping_pong_buffer.sv",
            "layer/fused_linear_relu.sv",
            "layer/byte_select.sv",
            "layer/bias_unpack.sv",
            "interface/axi_stream_in.sv",
            "interface/axi_stream_out.sv",
            "interface/axi_lite_ctrl.sv",
            "constraints/constraints.xdc",
        ]

        missing = []
        for f in expected_sv:
            if not (self.output_dir / f).is_file():
                missing.append(f)

        # Check generated files
        for f in ["parameters.svh", "accelerator_top.sv"]:
            if not (self.output_dir / f).is_file():
                missing.append(f)

        # Check weight files
        for f in ["weight_bank.mem", "bias_bank.mem"]:
            if not (self.output_dir / f).is_file():
                missing.append(f)

        if missing:
            raise FileNotFoundError(f"RTL generation incomplete, missing files: {missing}")

        logger.info("Validation passed: all %d expected files present", len(expected_sv) + 4)

    # ==================================================================
    # Tile Fabric Path — Stage 5.1
    # ==================================================================

    def _get_tile_configs(self) -> list[TileConfig]:
        """Get tile configs from fabric_config or build minimal list from graph."""
        if self.fabric_config is not None:
            return self.fabric_config.tiles
        # Fallback: one tile per node, infer type
        from mlasic.tile_mapper import OP_TO_TILE

        tiles = []
        for idx, name in enumerate(self.graph.topological_order()):
            node = self.graph.nodes[name]
            tile_type = OP_TO_TILE.get(node.op_type, TileType.RESHAPE)
            weight_bytes = 0
            bias_bytes = 0
            if node.dag_schedule is not None:
                weight_bytes = node.dag_schedule.weight_bytes
                bias_bytes = node.dag_schedule.bias_bytes
            tiles.append(
                TileConfig(
                    tile_id=idx,
                    tile_type=tile_type,
                    operator_assignment=name,
                    weight_bytes=weight_bytes,
                    bias_bytes=bias_bytes,
                )
            )
        return tiles

    def _get_routes(self) -> list:
        """Get routes from fabric_config or empty list."""
        if self.fabric_config is not None:
            return self.fabric_config.routes
        return []

    # ------------------------------------------------------------------
    # LUT file generation
    # ------------------------------------------------------------------

    def generate_lut_files(self) -> list[Path]:
        """Generate 256-entry .mem LUT files for nonlinear activations.

        Generates: gelu_lut.mem, silu_lut.mem, exp_lut.mem, rsqrt_lut.mem
        All use quantization-aware computation (INT8 input → INT8/INT16 output).
        """
        generated: list[Path] = []

        # GELU LUT: INT8 input [-128..127] → INT8 output
        gelu_table = np.zeros(256, dtype=np.int8)
        for i in range(256):
            x_int = np.int8(i if i < 128 else i - 256)
            x_fp = float(x_int) / 128.0  # rough dequant to [-1, 1]
            gelu_fp = 0.5 * x_fp * (1.0 + math.erf(x_fp / math.sqrt(2.0)))
            gelu_table[i] = np.int8(np.clip(np.floor(gelu_fp * 128.0 + 0.5), -128, 127))
        gelu_path = self.output_dir / "gelu_lut.mem"
        self._write_lut_mem(gelu_path, gelu_table)
        generated.append(gelu_path)

        # SiLU LUT: INT8 input [-128..127] → INT8 output
        silu_table = np.zeros(256, dtype=np.int8)
        for i in range(256):
            x_int = np.int8(i if i < 128 else i - 256)
            x_fp = float(x_int) / 128.0
            silu_fp = x_fp / (1.0 + math.exp(-x_fp))
            silu_table[i] = np.int8(np.clip(np.floor(silu_fp * 128.0 + 0.5), -128, 127))
        silu_path = self.output_dir / "silu_lut.mem"
        self._write_lut_mem(silu_path, silu_table)
        generated.append(silu_path)

        # Exp LUT: INT8 input [0..255] → INT16 output (for softmax)
        # exp(x) where x is (val - max), shifted to [0..255]
        exp_table = np.zeros(256, dtype=np.uint16)
        for i in range(256):
            # i=255 is x-max=0 (max val), i=0 is x-max=-255
            x = i - 255  # range [-255, 0]
            exp_val = math.exp(x / 32.0)  # scaled exponential
            exp_table[i] = min(int(exp_val * 256 + 0.5), 65535)
        exp_path = self.output_dir / "exp_lut.mem"
        self._write_lut_mem(exp_path, exp_table)
        generated.append(exp_path)

        # Reciprocal-sqrt LUT: 8-bit variance index → 16-bit rsqrt
        rsqrt_table = np.zeros(256, dtype=np.uint16)
        for i in range(256):
            if i == 0:
                rsqrt_table[i] = 65535  # max value for zero variance
            else:
                rsqrt_table[i] = min(int(256.0 / math.sqrt(float(i)) + 0.5), 65535)
        rsqrt_path = self.output_dir / "rsqrt_lut.mem"
        self._write_lut_mem(rsqrt_path, rsqrt_table)
        generated.append(rsqrt_path)

        logger.info("Generated %d LUT files", len(generated))
        return generated

    @staticmethod
    def _write_lut_mem(path: Path, data: np.ndarray) -> None:
        """Write a .mem file with one hex value per line."""
        width = data.dtype.itemsize * 2  # hex chars
        lines = []
        for val in data:
            # Convert to unsigned for hex formatting
            if data.dtype == np.int8:
                uval = int(val) & 0xFF
            elif data.dtype == np.uint16:
                uval = int(val) & 0xFFFF
            else:
                uval = int(val)
            lines.append(f"{uval:0{width}x}")
        path.write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # Per-channel requant ROM generation
    # ------------------------------------------------------------------

    def generate_requant_rom_files(self) -> list[Path]:
        """Generate per-channel requantization ROM .mem files for conv tiles.

        For each conv MAC tile, writes tile_{id}_requant.mem with one 32-bit
        M_fixed per output channel in hex. Used by tile.sv conv path to feed
        per-channel scales to the MAC array's requantize module.
        """
        tiles = self._get_tile_configs()
        generated: list[Path] = []

        for tile in tiles:
            if tile.tile_type != TileType.MAC:
                continue

            node = self.graph.nodes.get(tile.operator_assignment)
            if node is None:
                continue

            attrs = node.fused_attrs
            if not isinstance(attrs, FusedConvAttrs):
                continue

            if not attrs.requant_scale_fixed:
                continue

            # Per-channel M_fixed values
            scales = attrs.requant_scale_fixed
            tid = tile.tile_id

            # Write one 32-bit hex value per line (one per output channel)
            lines = []
            for m_fixed in scales:
                lines.append(f"{m_fixed & 0xFFFFFFFF:08x}")

            out_path = self.output_dir / f"tile_{tid}_requant.mem"
            out_path.write_text("\n".join(lines) + "\n")
            generated.append(out_path)
            logger.debug("Generated requant ROM %s (%d channels)", out_path, len(scales))

        logger.info("Generated %d per-channel requant ROM files", len(generated))
        return generated

    # ------------------------------------------------------------------
    # Tile parameters.svh generation
    # ------------------------------------------------------------------

    def generate_tile_parameters_svh(self) -> Path:
        """Generate tile_parameters.svh with per-tile arrays for the fabric."""
        tiles = self._get_tile_configs()
        num_tiles = len(tiles)
        last_idx = max(num_tiles - 1, 0)

        parallelism = self.constraints.max_parallelism

        # Build per-tile arrays
        tile_types: list[int] = []
        weight_depths: list[int] = []
        bias_depths: list[int] = []
        input_dims: list[int] = []
        output_dims: list[int] = []
        has_relu_arr: list[int] = []
        requant_ms: list[int] = []
        requant_shifts: list[int] = []
        requant_zps: list[int] = []
        total_cycles_arr: list[int] = []

        for tile in tiles:
            tile_types.append(_TILE_TYPE_ID.get(tile.tile_type, 6))

            node = self.graph.nodes.get(tile.operator_assignment)
            attrs = node.fused_attrs if node else None
            dag_sched = node.dag_schedule if node else None

            # Weight/bias depth in SRAM rows
            w_rows = max(tile.weight_bytes // (parallelism), 1) if tile.weight_bytes > 0 else 0
            b_rows = max(tile.bias_bytes // 4, 1) if tile.bias_bytes > 0 else 0
            weight_depths.append(w_rows)
            bias_depths.append(b_rows)

            # Dimensions and requant from fused_attrs
            if isinstance(attrs, FusedLinearAttrs):
                input_dims.append(attrs.input_dim)
                output_dims.append(attrs.output_dim)
                has_relu_arr.append(1 if attrs.has_relu else 0)
                requant_ms.append(attrs.requant_scale_fixed or 0)
                requant_shifts.append(attrs.requant_shift or 0)
                requant_zps.append(
                    attrs.output_quant.zero_point if attrs.output_quant else 0
                )
            elif isinstance(attrs, FusedConvAttrs):
                input_dims.append(attrs.in_channels)
                output_dims.append(attrs.out_channels)
                has_relu_arr.append(1 if attrs.has_relu else 0)
                # Per-channel: use first channel's scale for tile-level
                if attrs.requant_scale_fixed:
                    requant_ms.append(attrs.requant_scale_fixed[0])
                else:
                    requant_ms.append(0)
                requant_shifts.append(attrs.requant_shift or 0)
                requant_zps.append(
                    attrs.output_quant.zero_point if attrs.output_quant else 0
                )
            else:
                # Non-weighted tiles
                input_dims.append(0)
                output_dims.append(0)
                has_relu_arr.append(0)
                requant_ms.append(0)
                requant_shifts.append(0)
                requant_zps.append(0)

            # Cycle count
            total_cycles_arr.append(dag_sched.total_cycles if dag_sched else 0)

        # Compute ACT_DEPTH for spatial activations
        max_act = 0
        for name in self.graph.topological_order():
            node = self.graph.nodes[name]
            dag_sched = node.dag_schedule
            if dag_sched:
                max_act = max(max_act, dag_sched.input_bytes, dag_sched.output_bytes)
            for tname in list(node.inputs) + list(node.outputs):
                tensor = self.graph.tensors.get(tname)
                if tensor and tensor.type and tensor.type.shape and not tensor.is_constant:
                    shape = tensor.type.shape
                    total_elems = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
                    max_act = max(max_act, total_elems)
        act_depth = max(math.ceil(max_act / 8), 128)

        lines = [
            "// tile_parameters.svh — Auto-generated by MLASIC RTL Generator (tile fabric)",
            f"// Model: {self.graph.name}",
            f"// Tiles: {num_tiles}",
            "",
            "`ifndef TILE_PARAMETERS_SVH",
            "`define TILE_PARAMETERS_SVH",
            "",
            "/* verilator lint_off UNUSEDPARAM */",
            "",
            f"localparam int NUM_TILES    = {num_tiles};",
            f"localparam int PARALLELISM  = {parallelism};",
            "localparam int AXI_DATA_W   = 64;",
            f"localparam int ACT_DEPTH    = {act_depth};",
            "",
            "// Per-tile type (0=MAC, 1=ALU, 2=NORM, 3=SOFTMAX, 4=ACT, 5=POOL, 6=RESHAPE)",
            f"localparam int TILE_TYPE [0:{last_idx}] = "
            + "'{"
            + ", ".join(str(t) for t in tile_types)
            + "};",
            "",
            "// Per-tile weight/bias ROM depths",
            f"localparam int TILE_WEIGHT_DEPTH [0:{last_idx}] = "
            + "'{"
            + ", ".join(str(d) for d in weight_depths)
            + "};",
            f"localparam int TILE_BIAS_DEPTH [0:{last_idx}] = "
            + "'{"
            + ", ".join(str(d) for d in bias_depths)
            + "};",
            "",
            "// Per-tile compute parameters",
            f"localparam int TILE_INPUT_DIM [0:{last_idx}] = "
            + "'{"
            + ", ".join(str(d) for d in input_dims)
            + "};",
            f"localparam int TILE_OUTPUT_DIM [0:{last_idx}] = "
            + "'{"
            + ", ".join(str(d) for d in output_dims)
            + "};",
            f"localparam int TILE_HAS_RELU [0:{last_idx}] = "
            + "'{"
            + ", ".join(str(r) for r in has_relu_arr)
            + "};",
            "",
            "// Per-tile requantization parameters",
            f"localparam logic [31:0] TILE_REQUANT_M [0:{last_idx}] = "
            + "'{"
            + ", ".join(f"32'd{m}" for m in requant_ms)
            + "};",
            f"localparam logic [5:0] TILE_REQUANT_SHIFT [0:{last_idx}] = "
            + "'{"
            + ", ".join(f"6'd{s}" for s in requant_shifts)
            + "};",
            f"localparam logic signed [7:0] TILE_REQUANT_ZP [0:{last_idx}] = "
            + "'{"
            + ", ".join(
                f"-8'sd{abs(z)}" if z < 0 else f"8'sd{z}" for z in requant_zps
            )
            + "};",
            "",
            "// Per-tile cycle counts",
            f"localparam int TILE_CYCLES [0:{last_idx}] = "
            + "'{"
            + ", ".join(str(c) for c in total_cycles_arr)
            + "};",
            "",
            "/* verilator lint_on UNUSEDPARAM */",
            "",
            "`endif // TILE_PARAMETERS_SVH",
        ]

        out_path = self.output_dir / "tile_parameters.svh"
        out_path.write_text("\n".join(lines) + "\n")
        logger.info("Generated %s (%d tiles)", out_path, num_tiles)
        return out_path

    # ------------------------------------------------------------------
    # Tile fabric SystemVerilog generation
    # ------------------------------------------------------------------

    def generate_tile_fabric_sv(self) -> Path:
        """Generate model-specific tile_fabric.sv with tile instances and routing."""
        tiles = self._get_tile_configs()
        num_tiles = len(tiles)
        parallelism = self.constraints.max_parallelism

        # Build tile instantiation lines
        tile_insts: list[str] = []
        for tile in tiles:
            tid = tile.tile_id
            tt = _TILE_TYPE_ID.get(tile.tile_type, 6)

            node = self.graph.nodes.get(tile.operator_assignment)
            attrs = node.fused_attrs if node else None

            # Compute parameters
            input_dim = 0
            output_dim = 0
            has_relu = 0
            req_m = 0
            req_shift = 0
            req_zp = 0
            w_depth = max(tile.weight_bytes // parallelism, 1) if tile.weight_bytes > 0 else 0
            b_depth = max(tile.bias_bytes // 4, 1) if tile.bias_bytes > 0 else 0

            # Conv-specific params
            is_conv = 0
            in_channels = 1
            out_channels = 1
            in_height = 1
            in_width = 1
            kernel_h = 1
            kernel_w = 1
            stride_h = 1
            stride_w = 1
            pad_h = 0
            pad_w = 0
            group = 1

            # Pool-specific params
            pool_type = 0  # 0=max, 1=avg, 2=global_avg

            # Activation-specific params
            act_type = 0  # 0=relu, 1=gelu, 2=silu, 3=sigmoid
            lut_init_file = ""

            # Requant ROM init file (for conv tiles)
            requant_init_file = ""

            if isinstance(attrs, FusedLinearAttrs):
                input_dim = attrs.input_dim
                output_dim = attrs.output_dim
                has_relu = 1 if attrs.has_relu else 0
                req_m = attrs.requant_scale_fixed or 0
                req_shift = attrs.requant_shift or 0
                req_zp = attrs.output_quant.zero_point if attrs.output_quant else 0
            elif isinstance(attrs, FusedConvAttrs):
                is_conv = 1
                in_channels = attrs.in_channels
                out_channels = attrs.out_channels
                input_dim = in_channels
                output_dim = out_channels
                has_relu = 1 if (attrs.has_relu or attrs.has_relu6) else 0
                group = attrs.group
                kernel_h = attrs.kernel_shape[0] if len(attrs.kernel_shape) >= 1 else 1
                kernel_w = attrs.kernel_shape[1] if len(attrs.kernel_shape) >= 2 else 1
                stride_h = attrs.strides[0] if len(attrs.strides) >= 1 else 1
                stride_w = attrs.strides[1] if len(attrs.strides) >= 2 else 1
                pad_h = attrs.pads[0] if len(attrs.pads) >= 1 else 0
                pad_w = attrs.pads[1] if len(attrs.pads) >= 2 else 0

                # Extract spatial dims from input tensor shape
                if node and len(node.inputs) >= 1:
                    input_tensor = self.graph.tensors.get(node.inputs[0])
                    if input_tensor and input_tensor.type and input_tensor.type.shape:
                        shape = input_tensor.type.shape
                        # NCHW: shape = [N, C, H, W]
                        if len(shape) >= 4:
                            in_height = shape[2]
                            in_width = shape[3]

                # Per-channel requant: use first channel's scale for tile-level param
                if attrs.requant_scale_fixed:
                    req_m = attrs.requant_scale_fixed[0]
                    requant_init_file = f"tile_{tid}_requant.mem"
                req_shift = attrs.requant_shift or 0
                req_zp = attrs.output_quant.zero_point if attrs.output_quant else 0

            # Pool params from PoolAttrs or node op_type
            if node and tile.tile_type == TileType.POOL:
                pool_attrs = getattr(node, "fused_attrs", None)
                if isinstance(pool_attrs, PoolAttrs):
                    kernel_h = pool_attrs.kernel_shape[0] if len(pool_attrs.kernel_shape) >= 1 else 1
                    kernel_w = pool_attrs.kernel_shape[1] if len(pool_attrs.kernel_shape) >= 2 else 1
                    stride_h = pool_attrs.strides[0] if len(pool_attrs.strides) >= 1 else 1
                    stride_w = pool_attrs.strides[1] if len(pool_attrs.strides) >= 2 else 1
                    pad_h = pool_attrs.pads[0] if len(pool_attrs.pads) >= 1 else 0
                    pad_w = pool_attrs.pads[1] if len(pool_attrs.pads) >= 2 else 0

                # Pool type from OpType
                if node.op_type == OpType.MAX_POOL:
                    pool_type = 0
                elif node.op_type == OpType.AVERAGE_POOL:
                    pool_type = 1
                elif node.op_type == OpType.GLOBAL_AVERAGE_POOL:
                    pool_type = 2

                # Spatial dims from input tensor
                if len(node.inputs) >= 1:
                    input_tensor = self.graph.tensors.get(node.inputs[0])
                    if input_tensor and input_tensor.type and input_tensor.type.shape:
                        shape = input_tensor.type.shape
                        if len(shape) >= 4:
                            in_channels = shape[1]
                            in_height = shape[2]
                            in_width = shape[3]
                        elif len(shape) >= 2:
                            in_channels = shape[1]

                # For global avg pool, kernel = full spatial
                if node.op_type == OpType.GLOBAL_AVERAGE_POOL:
                    kernel_h = in_height
                    kernel_w = in_width
                    stride_h = 1
                    stride_w = 1
                    pad_h = 0
                    pad_w = 0

            # Activation params from OpType
            if node and tile.tile_type == TileType.ACTIVATION:
                _act_type_map = {
                    OpType.RELU: 0,
                    OpType.FUSED_GELU: 1,
                    OpType.FUSED_SILU: 2,
                    OpType.SIGMOID: 3,
                    OpType.TANH: 3,
                    OpType.ERF: 3,
                }
                act_type = _act_type_map.get(node.op_type, 0)
                _lut_file_map = {
                    OpType.FUSED_GELU: "gelu_lut.mem",
                    OpType.FUSED_SILU: "silu_lut.mem",
                }
                lut_init_file = _lut_file_map.get(node.op_type, "")

                # Input dim from tensor shape
                if len(node.inputs) >= 1:
                    input_tensor = self.graph.tensors.get(node.inputs[0])
                    if input_tensor and input_tensor.type and input_tensor.type.shape:
                        shape = input_tensor.type.shape
                        input_dim = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]

            # Weight init file path
            w_init = f"tile_{tid}_weights.mem" if tile.weight_bytes > 0 else ""
            b_init = f"tile_{tid}_biases.mem" if tile.bias_bytes > 0 else ""

            num_tiles_param = max(math.ceil(output_dim / parallelism), 1) if output_dim > 0 else 1

            inst = f"""\
    tile #(
        .TILE_ID({tid}),
        .TILE_TYPE({tt}),
        .PARALLELISM({parallelism}),
        .DATA_W(8),
        .ACC_W(32),
        .WEIGHT_DEPTH({w_depth}),
        .BIAS_DEPTH({b_depth}),
        .ACT_DEPTH(ACT_DEPTH),
        .ACT_WIDTH(64),
        .INPUT_DIM({input_dim}),
        .OUTPUT_DIM({output_dim}),
        .NUM_TILES({num_tiles_param}),
        .HAS_RELU({has_relu}),
        .REQUANT_M({req_m}),
        .REQUANT_SHIFT({req_shift}),
        .REQUANT_ZP({req_zp}),
        .WEIGHT_INIT_FILE("{w_init}"),
        .BIAS_INIT_FILE("{b_init}"),
        .LUT_INIT_FILE("{lut_init_file}"),
        .REQUANT_INIT_FILE("{requant_init_file}"),
        .IS_CONV({is_conv}),
        .IN_CHANNELS({in_channels}),
        .OUT_CHANNELS({out_channels}),
        .IN_HEIGHT({in_height}),
        .IN_WIDTH({in_width}),
        .KERNEL_H({kernel_h}),
        .KERNEL_W({kernel_w}),
        .STRIDE_H({stride_h}),
        .STRIDE_W({stride_w}),
        .PAD_H({pad_h}),
        .PAD_W({pad_w}),
        .GROUP({group}),
        .POOL_TYPE({pool_type}),
        .ACT_TYPE({act_type})
    ) u_tile_{tid} (
        .clk(clk),
        .rst_n(rst_n),
        .start(tile_start[{tid}]),
        .done(tile_done[{tid}]),
        .busy(tile_busy[{tid}]),
        .act_in_data(tile_act_in_data_{tid}),
        .act_in_addr(tile_act_in_addr_{tid}),
        .act_out_data(tile_act_out_data_{tid}),
        .act_out_addr(tile_act_out_addr_{tid}),
        .act_out_we(tile_act_out_we_{tid}),
        .act_skip_data(tile_act_skip_data_{tid}),
        .act_skip_addr(tile_act_skip_addr_{tid})
    );"""
            tile_insts.append(inst)

        # Build skip connection map from Routes
        # For ALU tiles (Add), identify the skip source tile.
        # An ALU tile has 2 input_routes: input_routes[0]=main, input_routes[1]=skip
        skip_src: dict[int, int] = {}  # dst_tile_id -> skip_source_tile_id
        for tile in tiles:
            if tile.tile_type == TileType.ALU and len(tile.input_routes) >= 2:
                # The skip source is the input route that isn't the immediately
                # preceding tile in the sequential chain.
                for src_id in tile.input_routes:
                    if src_id != tile.tile_id - 1:
                        skip_src[tile.tile_id] = src_id
                        break
                # If all inputs are from the preceding tile, use second route
                if tile.tile_id not in skip_src:
                    skip_src[tile.tile_id] = tile.input_routes[1]

        # Build static wiring (assign statements)
        wire_lines: list[str] = []
        # First tile reads from fabric input — uses address from tile
        wire_lines.append("    // Static routing — first tile reads from input buffer")
        wire_lines.append("    assign tile_act_in_data_0 = act_in_data;")
        # Each subsequent tile reads from previous tile's output buffer via address
        for i in range(1, num_tiles):
            wire_lines.append(
                f"    assign tile_act_in_data_{i} = act_buf_{i - 1}[tile_act_in_addr_{i}];"
            )
        # Last tile output is fabric output
        if num_tiles > 0:
            wire_lines.append(
                f"    assign act_out_data = tile_act_out_data_{num_tiles - 1};"
            )
        # Skip connection wiring for ALU tiles (read from buffer at skip_addr)
        wire_lines.append("")
        wire_lines.append("    // Skip connection routing (residual add)")
        for tid, src_id in skip_src.items():
            wire_lines.append(
                f"    assign tile_act_skip_data_{tid} = act_buf_{src_id}[tile_act_skip_addr_{tid}];"
            )
        # Default: tie off skip ports for non-ALU tiles
        for tile in tiles:
            if tile.tile_id not in skip_src:
                wire_lines.append(
                    f"    assign tile_act_skip_data_{tile.tile_id} = '0;"
                )

        # Activation buffer declarations
        buf_lines: list[str] = []
        for i in range(num_tiles - 1):
            buf_lines.append(f"    logic [63:0] act_buf_{i} [0:ACT_DEPTH-1];")

        # Buffer write logic — tile output writes to activation buffer
        buf_write_lines: list[str] = []
        for i in range(num_tiles - 1):
            buf_write_lines.append(f"""\
    always_ff @(posedge clk) begin
        if (tile_act_out_we_{i})
            act_buf_{i}[tile_act_out_addr_{i}] <= tile_act_out_data_{i};
    end""")

        # Determine act_depth — max activation buffer size
        # For CNN models with spatial activations (C*H*W), compute from tensor shapes
        max_act = 0
        for name in self.graph.topological_order():
            node = self.graph.nodes[name]
            dag_sched = node.dag_schedule
            if dag_sched:
                max_act = max(max_act, dag_sched.input_bytes, dag_sched.output_bytes)
            # Also check tensor shapes for spatial activations
            for tname in list(node.inputs) + list(node.outputs):
                tensor = self.graph.tensors.get(tname)
                if tensor and tensor.type and tensor.type.shape and not tensor.is_constant:
                    shape = tensor.type.shape
                    # Total elements (exclude batch dim)
                    total_elems = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
                    max_act = max(max_act, total_elems)
        act_depth = max(math.ceil(max_act / 8), 128)  # 64-bit words, min 128

        sv = f"""\
// tile_fabric.sv — Auto-generated by MLASIC RTL Generator (tile fabric)
// Model: {self.graph.name}
// Tiles: {num_tiles}

`include "tile_parameters.svh"

module tile_fabric #(
    parameter int ACT_DEPTH   = {act_depth}
) (
    input  logic        clk,
    input  logic        rst_n,

    // Fabric control
    input  logic        start,
    output logic        done,
    output logic        busy,

    // Activation input (from AXI-Stream)
    input  logic [AXI_DATA_W-1:0] act_in_data,

    // Activation output (to AXI-Stream)
    output logic [AXI_DATA_W-1:0] act_out_data
);

// ======================================================================
// Fabric FSM
// ======================================================================

typedef enum logic [1:0] {{
    S_IDLE,
    S_RUN_TILE,
    S_NEXT_TILE,
    S_DONE
}} fabric_state_t;

fabric_state_t state, state_next;
logic [$clog2(NUM_TILES)-1:0] tile_idx;

// Per-tile control
logic [NUM_TILES-1:0] tile_start;
logic [NUM_TILES-1:0] tile_done;
/* verilator lint_off UNUSEDSIGNAL */
logic [NUM_TILES-1:0] tile_busy;
/* verilator lint_on UNUSEDSIGNAL */

// Per-tile activation wires
"""

        # Per-tile signal declarations
        # Some signals are driven by tiles but only consumed by routing for specific
        # tile pairs — suppress lint warnings for the generated fabric wiring.
        sv += "/* verilator lint_off UNUSEDSIGNAL */\n"
        for i in range(num_tiles):
            sv += f"logic [63:0] tile_act_in_data_{i};\n"
            sv += f"logic [$clog2(ACT_DEPTH)-1:0] tile_act_in_addr_{i};\n"
            sv += f"logic [63:0] tile_act_out_data_{i};\n"
            sv += f"logic [$clog2(ACT_DEPTH)-1:0] tile_act_out_addr_{i};\n"
            sv += f"logic tile_act_out_we_{i};\n"
            sv += f"logic [63:0] tile_act_skip_data_{i};\n"
            sv += f"logic [$clog2(ACT_DEPTH)-1:0] tile_act_skip_addr_{i};\n"
            sv += "\n"
        sv += "/* verilator lint_on UNUSEDSIGNAL */\n\n"

        # Activation buffers
        sv += "// Activation buffers between tiles\n"
        sv += "\n".join(buf_lines) + "\n\n"

        # Buffer write logic
        sv += "// Buffer write logic\n"
        sv += "\n".join(buf_write_lines) + "\n\n"

        # Tile instances
        sv += "// ======================================================================\n"
        sv += "// Tile Instances\n"
        sv += "// ======================================================================\n\n"
        sv += "\n\n".join(tile_insts) + "\n\n"

        # Static routing
        sv += "// ======================================================================\n"
        sv += "// Static Routing\n"
        sv += "// ======================================================================\n\n"
        sv += "\n".join(wire_lines) + "\n\n"

        # Tile start generation (one-hot based on tile_idx)
        sv += "// Tile start — one-hot from tile_idx\n"
        sv += "always_comb begin\n"
        sv += "    tile_start = '0;\n"
        sv += "    if (state == S_RUN_TILE)\n"
        sv += "        tile_start[tile_idx] = 1'b1;\n"
        sv += "end\n\n"

        # FSM
        sv += """\
// ======================================================================
// Fabric FSM
// ======================================================================

always_comb begin
    state_next = state;
    case (state)
        S_IDLE:      if (start) state_next = S_RUN_TILE;
        S_RUN_TILE:  if (tile_done[tile_idx]) state_next = S_NEXT_TILE;
        S_NEXT_TILE: begin
            if (tile_idx == ($clog2(NUM_TILES))'(NUM_TILES - 1))
                state_next = S_DONE;
            else
                state_next = S_RUN_TILE;
        end
        S_DONE:      state_next = S_IDLE;
        default:     state_next = S_IDLE;
    endcase
end

always_ff @(posedge clk) begin
    if (!rst_n)
        state <= S_IDLE;
    else
        state <= state_next;
end

always_ff @(posedge clk) begin
    if (!rst_n || state == S_IDLE)
        tile_idx <= '0;
    else if (state == S_NEXT_TILE)
        tile_idx <= tile_idx + 1;
end

assign done = (state == S_DONE);
assign busy = (state != S_IDLE && state != S_DONE);

endmodule
"""

        out_path = self.output_dir / "tile_fabric.sv"
        out_path.write_text(sv)
        logger.info("Generated %s (%d tiles)", out_path, num_tiles)
        return out_path

    # ------------------------------------------------------------------
    # Tile accelerator_top.sv generation
    # ------------------------------------------------------------------

    def generate_tile_accelerator_top(self) -> Path:
        """Generate tile-based accelerator_top.sv wrapping tile_fabric."""
        tiles = self._get_tile_configs()
        num_tiles = len(tiles)

        # Compute act_depth for AXI-Stream sizing (same logic as tile_fabric)
        max_act = 0
        for name in self.graph.topological_order():
            node = self.graph.nodes[name]
            dag_sched = node.dag_schedule
            if dag_sched:
                max_act = max(max_act, dag_sched.input_bytes, dag_sched.output_bytes)
            for tname in list(node.inputs) + list(node.outputs):
                tensor = self.graph.tensors.get(tname)
                if tensor and tensor.type and tensor.type.shape and not tensor.is_constant:
                    shape = tensor.type.shape
                    total_elems = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
                    max_act = max(max_act, total_elems)
        act_depth = max(math.ceil(max_act / 8), 128)

        sv = f"""\
// accelerator_top.sv — Auto-generated by MLASIC RTL Generator (tile fabric)
// Model: {self.graph.name}
// Tiles: {num_tiles}

/* verilator lint_off VARHIDDEN */
`include "tile_parameters.svh"
/* verilator lint_on VARHIDDEN */

module accelerator_top #(
    parameter int ACT_DEPTH   = {act_depth}
) (
    // Clock and reset
    input  logic        clk,
    input  logic        rst_n,

    // AXI-Lite control
    input  logic [7:0]  s_axi_awaddr,
    input  logic        s_axi_awvalid,
    output logic        s_axi_awready,
    input  logic [31:0] s_axi_wdata,
    input  logic [3:0]  s_axi_wstrb,
    input  logic        s_axi_wvalid,
    output logic        s_axi_wready,
    output logic [1:0]  s_axi_bresp,
    output logic        s_axi_bvalid,
    input  logic        s_axi_bready,
    input  logic [7:0]  s_axi_araddr,
    input  logic        s_axi_arvalid,
    output logic        s_axi_arready,
    output logic [31:0] s_axi_rdata,
    output logic [1:0]  s_axi_rresp,
    output logic        s_axi_rvalid,
    input  logic        s_axi_rready,

    // AXI-Stream input
    input  logic [AXI_DATA_W-1:0] s_axis_tdata,
    input  logic        s_axis_tvalid,
    output logic        s_axis_tready,
    input  logic        s_axis_tlast,
    input  logic [7:0]  s_axis_tkeep,

    // AXI-Stream output
    output logic [AXI_DATA_W-1:0] m_axis_tdata,
    output logic        m_axis_tvalid,
    input  logic        m_axis_tready,
    output logic        m_axis_tlast,
    output logic [7:0]  m_axis_tkeep,

    // Interrupt
    output logic        irq
);

// ======================================================================
// Top-level FSM
// ======================================================================

typedef enum logic [2:0] {{
    S_IDLE,
    S_RECV_IN,
    S_RUN_FABRIC,
    S_SEND_OUT,
    S_DONE
}} accel_state_t;

accel_state_t state, state_next;

// ======================================================================
// Internal signals
// ======================================================================

// CSR control/status
logic ctrl_start, ctrl_soft_rst;
/* verilator lint_off UNUSEDSIGNAL */
logic ctrl_continuous, irq_en_out;  // Reserved for future use
/* verilator lint_on UNUSEDSIGNAL */
logic status_idle, status_busy, status_done_reg, status_error;
logic [31:0] cycle_count_reg, inf_count_reg, error_code_reg;
logic irq_done_pulse, irq_error_pulse;

// Fabric control
logic fabric_start, fabric_done;
/* verilator lint_off UNUSEDSIGNAL */
logic fabric_busy;  // Available for status reporting
/* verilator lint_on UNUSEDSIGNAL */

// Activation I/O for AXI-Stream
/* verilator lint_off UNUSEDSIGNAL */
logic [$clog2(ACT_DEPTH)-1:0] axis_in_addr, axis_out_addr;
logic axis_in_we;
logic [AXI_DATA_W-1:0] axis_in_wdata;
/* verilator lint_on UNUSEDSIGNAL */
logic [AXI_DATA_W-1:0] axis_out_rdata;

// AXI-Stream control
logic axis_in_enable, axis_in_done;
logic axis_out_enable, axis_out_done;

// Fabric input — driven by AXI-Stream input data
logic [AXI_DATA_W-1:0] fabric_act_in;
assign fabric_act_in = s_axis_tdata;

// Cycle counter
logic [31:0] cycle_counter;
logic counting;

// ======================================================================
// AXI-Lite Control
// ======================================================================

axi_lite_ctrl #(
    .VERSION(32'h0002_0000)
) u_ctrl (
    .clk(clk), .rst_n(rst_n),
    .s_axi_awaddr(s_axi_awaddr), .s_axi_awvalid(s_axi_awvalid), .s_axi_awready(s_axi_awready),
    .s_axi_wdata(s_axi_wdata), .s_axi_wstrb(s_axi_wstrb),
    .s_axi_wvalid(s_axi_wvalid), .s_axi_wready(s_axi_wready),
    .s_axi_bresp(s_axi_bresp), .s_axi_bvalid(s_axi_bvalid), .s_axi_bready(s_axi_bready),
    .s_axi_araddr(s_axi_araddr), .s_axi_arvalid(s_axi_arvalid),
    .s_axi_arready(s_axi_arready),
    .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp),
    .s_axi_rvalid(s_axi_rvalid), .s_axi_rready(s_axi_rready),
    .ctrl_start(ctrl_start), .ctrl_soft_rst(ctrl_soft_rst),
    .ctrl_continuous(ctrl_continuous), .irq_en(irq_en_out),
    .status_idle(status_idle), .status_busy(status_busy),
    .status_done(status_done_reg), .status_error(status_error),
    .cycle_count(cycle_count_reg), .inf_count(inf_count_reg), .error_code(error_code_reg),
    .layer_status(2'b0),
    .irq_done(irq_done_pulse), .irq_error(irq_error_pulse),
    .irq(irq)
);

// ======================================================================
// AXI-Stream Input
// ======================================================================

axi_stream_in #(
    .AXI_DATA_W(AXI_DATA_W),
    .NUM_BEATS(ACT_DEPTH),
    .ACT_DEPTH(ACT_DEPTH)
) u_axis_in (
    .clk(clk), .rst_n(rst_n),
    .enable(axis_in_enable), .done(axis_in_done),
    .s_axis_tdata(s_axis_tdata), .s_axis_tvalid(s_axis_tvalid),
    .s_axis_tready(s_axis_tready), .s_axis_tlast(s_axis_tlast), .s_axis_tkeep(s_axis_tkeep),
    .act_addr(axis_in_addr), .act_we(axis_in_we), .act_wdata(axis_in_wdata)
);

// ======================================================================
// AXI-Stream Output
// ======================================================================

axi_stream_out #(
    .AXI_DATA_W(AXI_DATA_W),
    .NUM_BEATS(ACT_DEPTH),
    .ACT_DEPTH(ACT_DEPTH)
) u_axis_out (
    .clk(clk), .rst_n(rst_n),
    .enable(axis_out_enable), .done(axis_out_done),
    .m_axis_tdata(m_axis_tdata), .m_axis_tvalid(m_axis_tvalid),
    .m_axis_tready(m_axis_tready), .m_axis_tlast(m_axis_tlast), .m_axis_tkeep(m_axis_tkeep),
    .act_addr(axis_out_addr), .act_rdata(axis_out_rdata)
);

// ======================================================================
// Tile Fabric
// ======================================================================

tile_fabric #(
    .ACT_DEPTH(ACT_DEPTH)
) u_fabric (
    .clk(clk), .rst_n(rst_n),
    .start(fabric_start),
    .done(fabric_done),
    .busy(fabric_busy),
    .act_in_data(fabric_act_in),
    .act_out_data(axis_out_rdata)
);

// ======================================================================
// Top-level FSM
// ======================================================================

always_comb begin
    state_next = state;
    case (state)
        S_IDLE:       if (ctrl_start) state_next = S_RECV_IN;
        S_RECV_IN:    if (axis_in_done) state_next = S_RUN_FABRIC;
        S_RUN_FABRIC: if (fabric_done) state_next = S_SEND_OUT;
        S_SEND_OUT:   if (axis_out_done) state_next = S_DONE;
        S_DONE:       state_next = S_IDLE;
        default:      state_next = S_IDLE;
    endcase
end

always_ff @(posedge clk) begin
    if (!rst_n || ctrl_soft_rst)
        state <= S_IDLE;
    else
        state <= state_next;
end

// Control signals
assign axis_in_enable  = (state == S_RECV_IN);
assign axis_out_enable = (state == S_SEND_OUT);
assign fabric_start    = (state == S_RECV_IN && axis_in_done);

// Status
assign status_idle  = (state == S_IDLE);
assign status_busy  = (state != S_IDLE && state != S_DONE);
assign status_error = 1'b0;
assign error_code_reg = 32'b0;

// Done and interrupt
always_ff @(posedge clk) begin
    if (!rst_n || state == S_IDLE) begin
        status_done_reg <= 1'b0;
        irq_done_pulse  <= 1'b0;
    end else if (state == S_DONE) begin
        status_done_reg <= 1'b1;
        irq_done_pulse  <= 1'b1;
    end else begin
        irq_done_pulse <= 1'b0;
    end
end

assign irq_error_pulse = 1'b0;

// Cycle counter
always_ff @(posedge clk) begin
    if (!rst_n || state == S_IDLE) begin
        cycle_counter <= '0;
        counting <= 1'b0;
    end else if (state == S_RECV_IN && !counting) begin
        counting <= 1'b1;
        cycle_counter <= '0;
    end else if (counting && state != S_DONE) begin
        cycle_counter <= cycle_counter + 1;
    end
end

assign cycle_count_reg = cycle_counter;

// Inference counter
always_ff @(posedge clk) begin
    if (!rst_n)
        inf_count_reg <= '0;
    else if (state == S_DONE && state_next == S_IDLE)
        inf_count_reg <= inf_count_reg + 1;
end

endmodule
"""

        out_path = self.output_dir / "accelerator_top.sv"
        out_path.write_text(sv)
        logger.info("Generated tile-based %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # ASIC ROM modules
    # ------------------------------------------------------------------

    def generate_asic_rom_modules(self) -> list[Path]:
        """Generate per-tile rom_tile_<id>.sv with hardcoded weight arrays.

        Only used for ASIC target. FPGA path uses sram_bank with $readmemh.
        """
        tiles = self._get_tile_configs()
        parallelism = self.constraints.max_parallelism
        generated: list[Path] = []

        for tile in tiles:
            if tile.weight_bytes == 0:
                continue

            tid = tile.tile_id
            node = self.graph.nodes.get(tile.operator_assignment)
            if node is None:
                continue

            # Extract weight data
            weight_data = None
            if len(node.inputs) >= 2:
                wt = self.graph.tensors.get(node.inputs[1])
                if wt and wt.is_constant and wt.data is not None:
                    weight_data = wt.data

            if weight_data is None:
                continue

            # Apply partitioning
            if tile.partition_info:
                start, end = tile.partition_info["output_range"]
                weight_data = weight_data[start:end]

            # Pack into ROM rows (PARALLELISM * 8 bits per row)
            flat = weight_data.flatten().astype(np.int8)
            row_width = parallelism
            num_rows = max(len(flat) // row_width, 1)

            lines = [
                f"// rom_tile_{tid}.sv — ASIC weight ROM for tile {tid}",
                f"// Node: {tile.operator_assignment}",
                f"// Weight bytes: {tile.weight_bytes}",
                "",
                f"module rom_tile_{tid} #(",
                f"    parameter int DEPTH = {num_rows},",
                f"    parameter int WIDTH = {row_width * 8}",
                ") (",
                "    input  logic clk,",
                "    input  logic [$clog2(DEPTH)-1:0] addr,",
                "    output logic [WIDTH-1:0] rdata",
                ");",
                "",
                f"    logic [WIDTH-1:0] rom [0:{num_rows - 1}];",
                "",
                "    always_ff @(posedge clk) begin",
                "        rdata <= rom[addr];",
                "    end",
                "",
                "    initial begin",
            ]

            # Write hex values row by row
            for row in range(num_rows):
                start_byte = row * row_width
                end_byte = min(start_byte + row_width, len(flat))
                row_bytes = flat[start_byte:end_byte]
                # Pad if needed
                if len(row_bytes) < row_width:
                    row_bytes = np.pad(row_bytes, (0, row_width - len(row_bytes)))
                hex_str = "".join(f"{int(b) & 0xFF:02x}" for b in row_bytes)
                lines.append(f"        rom[{row}] = {row_width * 8}'h{hex_str};")

            lines.append("    end")
            lines.append("")
            lines.append("endmodule")

            out_path = self.output_dir / f"rom_tile_{tid}.sv"
            out_path.write_text("\n".join(lines) + "\n")
            generated.append(out_path)

        logger.info("Generated %d ASIC ROM modules", len(generated))
        return generated

    # ------------------------------------------------------------------
    # Tile fabric validation
    # ------------------------------------------------------------------

    def validate_tile_fabric(self) -> None:
        """Post-generation validation for tile fabric path."""
        expected = [
            "tile_parameters.svh",
            "tile_fabric.sv",
            "accelerator_top.sv",
        ]

        # Module library files (same as MLP path)
        expected_sv = [
            "compute/mac_array.sv",
            "compute/requantize.sv",
            "compute/activation_relu.sv",
            "compute/conv_engine.sv",
            "compute/softmax_unit.sv",
            "compute/layer_norm_unit.sv",
            "compute/activation_unit.sv",
            "compute/pool_unit.sv",
            "memory/sram_bank.sv",
            "memory/ping_pong_buffer.sv",
            "memory/rom_tile.sv",
            "tile/tile.sv",
            "tile/tile_fabric.sv",
            "layer/fused_linear_relu.sv",
            "layer/byte_select.sv",
            "layer/bias_unpack.sv",
            "interface/axi_stream_in.sv",
            "interface/axi_stream_out.sv",
            "interface/axi_lite_ctrl.sv",
            "constraints/constraints.xdc",
        ]

        # LUT files
        expected_lut = [
            "gelu_lut.mem",
            "silu_lut.mem",
            "exp_lut.mem",
            "rsqrt_lut.mem",
        ]

        missing = []
        for f in expected + expected_sv + expected_lut:
            if not (self.output_dir / f).is_file():
                missing.append(f)

        if missing:
            raise FileNotFoundError(
                f"Tile fabric RTL generation incomplete, missing files: {missing}"
            )

        logger.info("Tile fabric validation passed")
