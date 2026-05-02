"""Tile-fabric Graph → RTL IR lowering.

Companion to ``mlasic.rtl_lowering`` for the tile-fabric (CNN/Transformer/
DAG) path. Produces:

  * ``lower_tile_accelerator_top``  → ``RTLModule`` for the AXI/CSR-wrapped
                                       top-level that contains a single
                                       ``tile_fabric`` instance.
  * ``lower_tile_fabric_module``    → ``RTLModule`` for the per-model
                                       ``tile_fabric`` that instantiates each
                                       ``tile`` and wires up the static
                                       activation routing + skip connections.

These two replace ``RTLGenerator.generate_tile_accelerator_top`` and
``RTLGenerator.generate_tile_fabric_sv`` respectively. The per-tile parameter
extraction stays in Python (it's pure data shuffling from the Graph/Fabric
to a parameter dict); the resulting parameter dicts feed straight into
``RTLInstance`` objects so the structure of the fabric is first-class IR.
"""

from __future__ import annotations

import math
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
from mlasic.rtl_ir import RTLDir, RTLModule, RTLType
from mlasic.tile_mapper import FabricConfig, TileConfig, TileType


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


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def compute_act_depth(graph: Graph) -> int:
    """Compute the model-wide activation buffer depth (in 64-bit words)."""
    max_act = 0
    for name in graph.topological_order():
        node = graph.nodes[name]
        dag_sched = node.dag_schedule
        if dag_sched:
            max_act = max(max_act, dag_sched.input_bytes, dag_sched.output_bytes)
        for tname in list(node.inputs) + list(node.outputs):
            tensor = graph.tensors.get(tname)
            if (
                tensor
                and tensor.type
                and tensor.type.shape
                and not tensor.is_constant
            ):
                shape = tensor.type.shape
                total_elems = (
                    int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
                )
                max_act = max(max_act, total_elems)
    return max(math.ceil(max_act / 8), 128)


def render_tile_parameters_svh(
    graph: Graph,
    fabric: FabricConfig,
    constraints: HardwareConstraints,
) -> str:
    """Render the ``tile_parameters.svh`` constants header for the tile fabric.

    Per-tile arrays mirror ``FabricConfig.tiles`` order. ``ACT_DEPTH`` is the
    model-wide activation buffer depth, computed by ``compute_act_depth``.
    """
    tiles = fabric.tiles
    num_tiles = len(tiles)
    last_idx = max(num_tiles - 1, 0)
    parallelism = constraints.max_parallelism

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

        node = graph.nodes.get(tile.operator_assignment)
        attrs = node.fused_attrs if node else None
        dag_sched = node.dag_schedule if node else None

        w_rows = (
            max(tile.weight_bytes // parallelism, 1) if tile.weight_bytes > 0 else 0
        )
        b_rows = max(tile.bias_bytes // 4, 1) if tile.bias_bytes > 0 else 0
        weight_depths.append(w_rows)
        bias_depths.append(b_rows)

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
            if attrs.requant_scale_fixed:
                requant_ms.append(attrs.requant_scale_fixed[0])
            else:
                requant_ms.append(0)
            requant_shifts.append(attrs.requant_shift or 0)
            requant_zps.append(
                attrs.output_quant.zero_point if attrs.output_quant else 0
            )
        else:
            input_dims.append(0)
            output_dims.append(0)
            has_relu_arr.append(0)
            requant_ms.append(0)
            requant_shifts.append(0)
            requant_zps.append(0)

        total_cycles_arr.append(dag_sched.total_cycles if dag_sched else 0)

    act_depth = compute_act_depth(graph)

    def _zp_lit(z: int) -> str:
        return f"-8'sd{abs(z)}" if z < 0 else f"8'sd{z}"

    def _arr(items: list[str]) -> str:
        return "'{" + ", ".join(items) + "}"

    lines = [
        "// tile_parameters.svh — Auto-generated by MLASIC RTL Generator (tile fabric)",
        f"// Model: {graph.name}",
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
        + _arr([str(t) for t in tile_types])
        + ";",
        "",
        "// Per-tile weight/bias ROM depths",
        f"localparam int TILE_WEIGHT_DEPTH [0:{last_idx}] = "
        + _arr([str(d) for d in weight_depths])
        + ";",
        f"localparam int TILE_BIAS_DEPTH [0:{last_idx}] = "
        + _arr([str(d) for d in bias_depths])
        + ";",
        "",
        "// Per-tile compute parameters",
        f"localparam int TILE_INPUT_DIM [0:{last_idx}] = "
        + _arr([str(d) for d in input_dims])
        + ";",
        f"localparam int TILE_OUTPUT_DIM [0:{last_idx}] = "
        + _arr([str(d) for d in output_dims])
        + ";",
        f"localparam int TILE_HAS_RELU [0:{last_idx}] = "
        + _arr([str(r) for r in has_relu_arr])
        + ";",
        "",
        "// Per-tile requantization parameters",
        f"localparam logic [31:0] TILE_REQUANT_M [0:{last_idx}] = "
        + _arr([f"32'd{m}" for m in requant_ms])
        + ";",
        f"localparam logic [5:0] TILE_REQUANT_SHIFT [0:{last_idx}] = "
        + _arr([f"6'd{s}" for s in requant_shifts])
        + ";",
        f"localparam logic signed [7:0] TILE_REQUANT_ZP [0:{last_idx}] = "
        + _arr([_zp_lit(z) for z in requant_zps])
        + ";",
        "",
        "// Per-tile cycle counts",
        f"localparam int TILE_CYCLES [0:{last_idx}] = "
        + _arr([str(c) for c in total_cycles_arr])
        + ";",
        "",
        "/* verilator lint_on UNUSEDPARAM */",
        "",
        "`endif // TILE_PARAMETERS_SVH",
    ]
    return "\n".join(lines) + "\n"


def lower_tile_accelerator_top(
    graph: Graph,
    fabric: FabricConfig,
    constraints: HardwareConstraints,
) -> RTLModule:
    """Lower a scheduled tile-fabric graph to ``accelerator_top`` IR.

    The tile-fabric top is a thin AXI wrapper around a model-specific
    ``tile_fabric`` instance — much smaller than the MLP top because all the
    compute lives inside ``tile_fabric``.
    """
    num_tiles = len(fabric.tiles)
    act_depth = compute_act_depth(graph)

    mod = RTLModule(
        name="accelerator_top",
        header_comment=(
            f"accelerator_top.sv — Auto-generated by MLASIC RTL Generator (tile fabric)\n"
            f"Model: {graph.name}\n"
            f"Tiles: {num_tiles}"
        ),
    )
    mod.add_raw(
        code='/* verilator lint_off VARHIDDEN */\n`include "tile_parameters.svh"\n/* verilator lint_on VARHIDDEN */',
        label="includes",
    )
    mod.add_param("ACT_DEPTH", act_depth)

    _add_tile_top_ports(mod)
    _add_tile_top_signals(mod)
    _add_tile_top_axi_lite(mod)
    _add_tile_top_axi_stream(mod)
    _add_tile_fabric_instance(mod)
    _add_tile_top_fsm(mod)

    mod.verify()
    return mod


def lower_tile_fabric_module(
    graph: Graph,
    fabric: FabricConfig,
    constraints: HardwareConstraints,
) -> RTLModule:
    """Lower a scheduled tile-fabric graph to a model-specific ``tile_fabric``.

    The fabric module owns:
      * One ``tile`` instance per ``TileConfig``, parameterized from the
        graph's fused-attrs.
      * Inter-tile activation buffers (one per producing tile).
      * Static routing assigns (sequential next-tile + skip-connection wires).
      * The fabric FSM that sequences ``tile_idx`` through the chain.
    """
    tiles = fabric.tiles
    num_tiles = len(tiles)
    parallelism = constraints.max_parallelism
    act_depth = compute_act_depth(graph)

    mod = RTLModule(
        name="tile_fabric",
        header_comment=(
            f"tile_fabric.sv — Auto-generated by MLASIC RTL Generator (tile fabric)\n"
            f"Model: {graph.name}\n"
            f"Tiles: {num_tiles}"
        ),
    )
    mod.add_include("tile_parameters.svh")
    mod.add_param("ACT_DEPTH", act_depth)

    L = RTLType.logic
    mod.add_port("clk", RTLDir.INPUT, L())
    mod.add_port("rst_n", RTLDir.INPUT, L())
    mod.add_port("start", RTLDir.INPUT, L())
    mod.add_port("done", RTLDir.OUTPUT, L())
    mod.add_port("busy", RTLDir.OUTPUT, L())
    mod.add_port("act_in_data", RTLDir.INPUT, L(width="AXI_DATA_W"))
    mod.add_port("act_out_data", RTLDir.OUTPUT, L(width="AXI_DATA_W"))

    # FSM state declarations (raw — typedef + reg decl)
    mod.add_raw(
        code=(
            "typedef enum logic [1:0] {\n"
            "    S_IDLE,\n"
            "    S_RUN_TILE,\n"
            "    S_NEXT_TILE,\n"
            "    S_DONE\n"
            "} fabric_state_t;\n"
            "\n"
            "fabric_state_t state, state_next;\n"
            "logic [$clog2(NUM_TILES)-1:0] tile_idx;\n"
            "\n"
            "logic [NUM_TILES-1:0] tile_start;\n"
            "logic [NUM_TILES-1:0] tile_done;\n"
            "/* verilator lint_off UNUSEDSIGNAL */\n"
            "logic [NUM_TILES-1:0] tile_busy;\n"
            "/* verilator lint_on UNUSEDSIGNAL */"
        ),
        defines=("state", "state_next", "tile_idx", "tile_start", "tile_done", "tile_busy"),
        label="Fabric FSM state and per-tile control",
    )

    # Per-tile activation wires (one block; lint suppression around it)
    per_tile_decls: list[str] = ["/* verilator lint_off UNUSEDSIGNAL */"]
    for i in range(num_tiles):
        per_tile_decls.extend(
            [
                f"logic [63:0] tile_act_in_data_{i};",
                f"logic [$clog2(ACT_DEPTH)-1:0] tile_act_in_addr_{i};",
                f"logic [63:0] tile_act_out_data_{i};",
                f"logic [$clog2(ACT_DEPTH)-1:0] tile_act_out_addr_{i};",
                f"logic tile_act_out_we_{i};",
                f"logic [63:0] tile_act_skip_data_{i};",
                f"logic [$clog2(ACT_DEPTH)-1:0] tile_act_skip_addr_{i};",
                "",
            ]
        )
    per_tile_decls.append("/* verilator lint_on UNUSEDSIGNAL */")
    mod.add_raw(
        code="\n".join(per_tile_decls),
        defines=tuple(
            f"tile_act_{kind}_{i}"
            for i in range(num_tiles)
            for kind in (
                "in_data",
                "in_addr",
                "out_data",
                "out_addr",
                "out_we",
                "skip_data",
                "skip_addr",
            )
        ),
        label="Per-tile activation wires",
    )

    # Inter-tile activation buffers (one per producing tile)
    if num_tiles > 1:
        buf_decls = "\n".join(
            f"logic [63:0] act_buf_{i} [0:ACT_DEPTH-1];"
            for i in range(num_tiles - 1)
        )
        mod.add_raw(
            code=buf_decls,
            defines=tuple(f"act_buf_{i}" for i in range(num_tiles - 1)),
            label="Activation buffers between tiles",
        )

        write_blocks = "\n".join(
            (
                f"always_ff @(posedge clk) begin\n"
                f"    if (tile_act_out_we_{i})\n"
                f"        act_buf_{i}[tile_act_out_addr_{i}] <= tile_act_out_data_{i};\n"
                f"end"
            )
            for i in range(num_tiles - 1)
        )
        mod.add_raw(
            code=write_blocks,
            defines=tuple(f"act_buf_{i}" for i in range(num_tiles - 1)),
            uses=tuple(
                f"tile_act_out_{kind}_{i}"
                for i in range(num_tiles - 1)
                for kind in ("we", "addr", "data")
            ),
            label="Buffer write logic",
        )

    # Tile instances — structural IR for each compute tile
    for tile in tiles:
        params = _build_tile_params(tile, graph, parallelism)
        mod.add_instance(
            name=f"u_tile_{tile.tile_id}",
            module_type="tile",
            params={k: str(v) for k, v in params.items()},
            port_map=_build_tile_port_map(tile.tile_id),
            comment=f"Tile {tile.tile_id} ({tile.tile_type.name})",
        )

    # Static routing — sequential next-tile + skip wires
    mod.add_raw(
        code="// Static routing — first tile reads from input buffer\nassign tile_act_in_data_0 = act_in_data;",
        defines=("tile_act_in_data_0",),
        uses=("act_in_data",),
        label="First tile routing",
    )
    seq_assigns = [
        f"assign tile_act_in_data_{i} = act_buf_{i - 1}[tile_act_in_addr_{i}];"
        for i in range(1, num_tiles)
    ]
    if seq_assigns:
        mod.add_raw(
            code="\n".join(seq_assigns),
            defines=tuple(f"tile_act_in_data_{i}" for i in range(1, num_tiles)),
            uses=tuple(f"act_buf_{i - 1}" for i in range(1, num_tiles)),
            label="Sequential next-tile activation routing",
        )
    if num_tiles > 0:
        mod.add_assign(
            "act_out_data",
            f"tile_act_out_data_{num_tiles - 1}",
            comment="Last tile output is fabric output",
        )

    # Skip connection routing (residual add on ALU tiles)
    skip_src = _compute_skip_sources(tiles)
    if skip_src:
        skip_assigns = [
            f"assign tile_act_skip_data_{tid} = act_buf_{src_id}[tile_act_skip_addr_{tid}];"
            for tid, src_id in skip_src.items()
        ]
        mod.add_raw(
            code="// Skip connection routing (residual add)\n" + "\n".join(skip_assigns),
            defines=tuple(f"tile_act_skip_data_{tid}" for tid in skip_src),
            uses=tuple(
                f"act_buf_{src_id}"
                for src_id in skip_src.values()
            ),
            label="Skip-connection routing",
        )
    # Tie off skip ports for tiles that don't consume a skip
    no_skip_assigns = [
        f"assign tile_act_skip_data_{tile.tile_id} = 64'b0;"
        for tile in tiles
        if tile.tile_id not in skip_src
    ]
    if no_skip_assigns:
        mod.add_raw(
            code="\n".join(no_skip_assigns),
            defines=tuple(
                f"tile_act_skip_data_{t.tile_id}"
                for t in tiles
                if t.tile_id not in skip_src
            ),
            label="Skip-port tie-offs (non-ALU tiles)",
        )

    # Tile start (one-hot) + fabric FSM
    mod.add_raw(
        code=(
            "// Tile start — one-hot from tile_idx\n"
            "always_comb begin\n"
            "    tile_start = {NUM_TILES{1'b0}};\n"
            "    if (state == S_RUN_TILE)\n"
            "        tile_start[tile_idx] = 1'b1;\n"
            "end"
        ),
        defines=("tile_start",),
        uses=("state", "tile_idx"),
        label="One-hot tile start",
    )

    mod.add_raw(
        code=(
            "always_comb begin\n"
            "    state_next = state;\n"
            "    case (state)\n"
            "        S_IDLE:      if (start) state_next = S_RUN_TILE;\n"
            "        S_RUN_TILE:  if (tile_done[tile_idx]) state_next = S_NEXT_TILE;\n"
            "        S_NEXT_TILE: begin\n"
            "            if (tile_idx == ($clog2(NUM_TILES))'(NUM_TILES - 1))\n"
            "                state_next = S_DONE;\n"
            "            else\n"
            "                state_next = S_RUN_TILE;\n"
            "        end\n"
            "        S_DONE:      state_next = S_IDLE;\n"
            "        default:     state_next = S_IDLE;\n"
            "    endcase\n"
            "end\n"
            "\n"
            "always_ff @(posedge clk) begin\n"
            "    if (!rst_n)\n"
            "        state <= S_IDLE;\n"
            "    else\n"
            "        state <= state_next;\n"
            "end\n"
            "\n"
            "always_ff @(posedge clk) begin\n"
            "    if (!rst_n || state == S_IDLE)\n"
            "        tile_idx <= {$clog2(NUM_TILES){1'b0}};\n"
            "    else if (state == S_NEXT_TILE)\n"
            "        tile_idx <= tile_idx + 1;\n"
            "end\n"
            "\n"
            "assign done = (state == S_DONE);\n"
            "assign busy = (state != S_IDLE && state != S_DONE);"
        ),
        defines=("state", "state_next", "tile_idx", "done", "busy"),
        uses=("start", "tile_done", "rst_n", "clk"),
        label="Fabric FSM",
    )

    mod.verify()
    return mod


# ---------------------------------------------------------------------------
# Per-tile parameter / port-map extraction
# ---------------------------------------------------------------------------


def _build_tile_params(
    tile: TileConfig, graph: Graph, parallelism: int
) -> dict[str, object]:
    """Compute the tile.sv parameter dict from a TileConfig + Graph."""
    tid = tile.tile_id
    tt = _TILE_TYPE_ID.get(tile.tile_type, 6)

    node = graph.nodes.get(tile.operator_assignment)
    attrs = node.fused_attrs if node else None

    input_dim = 0
    output_dim = 0
    has_relu = 0
    req_m: int = 0
    req_shift: int = 0
    req_zp: int = 0
    w_depth = (
        max(tile.weight_bytes // parallelism, 1) if tile.weight_bytes > 0 else 0
    )
    b_depth = max(tile.bias_bytes // 4, 1) if tile.bias_bytes > 0 else 0

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
    pool_type = 0
    act_type = 0
    lut_init_file = ""
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
        if node and len(node.inputs) >= 1:
            tt_in = graph.tensors.get(node.inputs[0])
            if tt_in and tt_in.type and tt_in.type.shape and len(tt_in.type.shape) >= 4:
                in_height = tt_in.type.shape[2]
                in_width = tt_in.type.shape[3]
        if attrs.requant_scale_fixed:
            req_m = attrs.requant_scale_fixed[0]
            requant_init_file = f"tile_{tid}_requant.mem"
        req_shift = attrs.requant_shift or 0
        req_zp = attrs.output_quant.zero_point if attrs.output_quant else 0

    if node and tile.tile_type == TileType.POOL:
        pool_attrs = getattr(node, "fused_attrs", None)
        if isinstance(pool_attrs, PoolAttrs):
            kernel_h = pool_attrs.kernel_shape[0] if len(pool_attrs.kernel_shape) >= 1 else 1
            kernel_w = pool_attrs.kernel_shape[1] if len(pool_attrs.kernel_shape) >= 2 else 1
            stride_h = pool_attrs.strides[0] if len(pool_attrs.strides) >= 1 else 1
            stride_w = pool_attrs.strides[1] if len(pool_attrs.strides) >= 2 else 1
            pad_h = pool_attrs.pads[0] if len(pool_attrs.pads) >= 1 else 0
            pad_w = pool_attrs.pads[1] if len(pool_attrs.pads) >= 2 else 0
        if node.op_type == OpType.MAX_POOL:
            pool_type = 0
        elif node.op_type == OpType.AVERAGE_POOL:
            pool_type = 1
        elif node.op_type == OpType.GLOBAL_AVERAGE_POOL:
            pool_type = 2
        if len(node.inputs) >= 1:
            tt_in = graph.tensors.get(node.inputs[0])
            if tt_in and tt_in.type and tt_in.type.shape:
                shape = tt_in.type.shape
                if len(shape) >= 4:
                    in_channels = shape[1]
                    in_height = shape[2]
                    in_width = shape[3]
                elif len(shape) >= 2:
                    in_channels = shape[1]
        if node.op_type == OpType.GLOBAL_AVERAGE_POOL:
            kernel_h = in_height
            kernel_w = in_width
            stride_h = 1
            stride_w = 1
            pad_h = 0
            pad_w = 0

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
        if len(node.inputs) >= 1:
            tt_in = graph.tensors.get(node.inputs[0])
            if tt_in and tt_in.type and tt_in.type.shape:
                shape = tt_in.type.shape
                input_dim = (
                    int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
                )

    w_init = f"tile_{tid}_weights.mem" if tile.weight_bytes > 0 else ""
    b_init = f"tile_{tid}_biases.mem" if tile.bias_bytes > 0 else ""
    num_tiles_param = (
        max(math.ceil(output_dim / parallelism), 1) if output_dim > 0 else 1
    )

    return {
        "TILE_ID": tid,
        "TILE_TYPE": tt,
        "PARALLELISM": parallelism,
        "DATA_W": 8,
        "ACC_W": 32,
        "WEIGHT_DEPTH": w_depth,
        "BIAS_DEPTH": b_depth,
        "ACT_DEPTH": "ACT_DEPTH",
        "ACT_WIDTH": 64,
        "INPUT_DIM": input_dim,
        "OUTPUT_DIM": output_dim,
        "NUM_TILES": num_tiles_param,
        "HAS_RELU": has_relu,
        "REQUANT_M": req_m,
        "REQUANT_SHIFT": req_shift,
        "REQUANT_ZP": req_zp,
        "WEIGHT_INIT_FILE": f'"{w_init}"',
        "BIAS_INIT_FILE": f'"{b_init}"',
        "LUT_INIT_FILE": f'"{lut_init_file}"',
        "REQUANT_INIT_FILE": f'"{requant_init_file}"',
        "IS_CONV": is_conv,
        "IN_CHANNELS": in_channels,
        "OUT_CHANNELS": out_channels,
        "IN_HEIGHT": in_height,
        "IN_WIDTH": in_width,
        "KERNEL_H": kernel_h,
        "KERNEL_W": kernel_w,
        "STRIDE_H": stride_h,
        "STRIDE_W": stride_w,
        "PAD_H": pad_h,
        "PAD_W": pad_w,
        "GROUP": group,
        "POOL_TYPE": pool_type,
        "ACT_TYPE": act_type,
    }


def _build_tile_port_map(tid: int) -> dict[str, str]:
    return {
        "clk": "clk",
        "rst_n": "rst_n",
        "start": f"tile_start[{tid}]",
        "done": f"tile_done[{tid}]",
        "busy": f"tile_busy[{tid}]",
        "act_in_data": f"tile_act_in_data_{tid}",
        "act_in_addr": f"tile_act_in_addr_{tid}",
        "act_out_data": f"tile_act_out_data_{tid}",
        "act_out_addr": f"tile_act_out_addr_{tid}",
        "act_out_we": f"tile_act_out_we_{tid}",
        "act_skip_data": f"tile_act_skip_data_{tid}",
        "act_skip_addr": f"tile_act_skip_addr_{tid}",
    }


def _compute_skip_sources(tiles: list[TileConfig]) -> dict[int, int]:
    """ALU tile id → producing tile id of its skip input."""
    skip_src: dict[int, int] = {}
    for tile in tiles:
        if tile.tile_type == TileType.ALU and len(tile.input_routes) >= 2:
            for src_id in tile.input_routes:
                if src_id != tile.tile_id - 1:
                    skip_src[tile.tile_id] = src_id
                    break
            if tile.tile_id not in skip_src:
                skip_src[tile.tile_id] = tile.input_routes[1]
    return skip_src


# ---------------------------------------------------------------------------
# tile-fabric accelerator_top scaffolding
# ---------------------------------------------------------------------------


def _add_tile_top_ports(mod: RTLModule) -> None:
    L = RTLType.logic
    mod.add_port("clk", RTLDir.INPUT, L())
    mod.add_port("rst_n", RTLDir.INPUT, L())

    mod.add_port("s_axi_awaddr", RTLDir.INPUT, L(width=8))
    mod.add_port("s_axi_awvalid", RTLDir.INPUT, L())
    mod.add_port("s_axi_awready", RTLDir.OUTPUT, L())
    mod.add_port("s_axi_wdata", RTLDir.INPUT, L(width=32))
    mod.add_port("s_axi_wstrb", RTLDir.INPUT, L(width=4))
    mod.add_port("s_axi_wvalid", RTLDir.INPUT, L())
    mod.add_port("s_axi_wready", RTLDir.OUTPUT, L())
    mod.add_port("s_axi_bresp", RTLDir.OUTPUT, L(width=2))
    mod.add_port("s_axi_bvalid", RTLDir.OUTPUT, L())
    mod.add_port("s_axi_bready", RTLDir.INPUT, L())
    mod.add_port("s_axi_araddr", RTLDir.INPUT, L(width=8))
    mod.add_port("s_axi_arvalid", RTLDir.INPUT, L())
    mod.add_port("s_axi_arready", RTLDir.OUTPUT, L())
    mod.add_port("s_axi_rdata", RTLDir.OUTPUT, L(width=32))
    mod.add_port("s_axi_rresp", RTLDir.OUTPUT, L(width=2))
    mod.add_port("s_axi_rvalid", RTLDir.OUTPUT, L())
    mod.add_port("s_axi_rready", RTLDir.INPUT, L())

    mod.add_port("s_axis_tdata", RTLDir.INPUT, L(width="AXI_DATA_W"))
    mod.add_port("s_axis_tvalid", RTLDir.INPUT, L())
    mod.add_port("s_axis_tready", RTLDir.OUTPUT, L())
    mod.add_port("s_axis_tlast", RTLDir.INPUT, L())
    mod.add_port("s_axis_tkeep", RTLDir.INPUT, L(width=8))

    mod.add_port("m_axis_tdata", RTLDir.OUTPUT, L(width="AXI_DATA_W"))
    mod.add_port("m_axis_tvalid", RTLDir.OUTPUT, L())
    mod.add_port("m_axis_tready", RTLDir.INPUT, L())
    mod.add_port("m_axis_tlast", RTLDir.OUTPUT, L())
    mod.add_port("m_axis_tkeep", RTLDir.OUTPUT, L(width=8))

    mod.add_port("irq", RTLDir.OUTPUT, L())


def _add_tile_top_signals(mod: RTLModule) -> None:
    mod.add_raw(
        code=(
            "typedef enum logic [2:0] {\n"
            "    S_IDLE,\n"
            "    S_RECV_IN,\n"
            "    S_RUN_FABRIC,\n"
            "    S_SEND_OUT,\n"
            "    S_DONE\n"
            "} accel_state_t;\n"
            "\n"
            "accel_state_t state, state_next;"
        ),
        defines=("state", "state_next"),
        label="Top-level FSM state",
    )

    L = RTLType.logic
    for sig in ("ctrl_start", "ctrl_soft_rst"):
        mod.add_signal(sig, L())
    mod.add_raw(
        code=(
            "/* verilator lint_off UNUSEDSIGNAL */\n"
            "logic ctrl_continuous, irq_en_out;  // Reserved for future use\n"
            "/* verilator lint_on UNUSEDSIGNAL */"
        ),
        defines=("ctrl_continuous", "irq_en_out"),
        label="Reserved CSR fields",
    )
    for sig in (
        "status_idle",
        "status_busy",
        "status_done_reg",
        "status_error",
        "irq_done_pulse",
        "irq_error_pulse",
    ):
        mod.add_signal(sig, L())
    for sig in ("cycle_count_reg", "inf_count_reg", "error_code_reg"):
        mod.add_signal(sig, L(width=32))

    for sig in ("fabric_start", "fabric_done"):
        mod.add_signal(sig, L())
    mod.add_raw(
        code=(
            "/* verilator lint_off UNUSEDSIGNAL */\n"
            "logic fabric_busy;  // Available for status reporting\n"
            "/* verilator lint_on UNUSEDSIGNAL */"
        ),
        defines=("fabric_busy",),
        label="Fabric busy signal (unused at top)",
    )

    mod.add_raw(
        code=(
            "/* verilator lint_off UNUSEDSIGNAL */\n"
            "logic [$clog2(ACT_DEPTH)-1:0] axis_in_addr, axis_out_addr;\n"
            "logic axis_in_we;\n"
            "logic [AXI_DATA_W-1:0] axis_in_wdata;\n"
            "/* verilator lint_on UNUSEDSIGNAL */\n"
            "logic [AXI_DATA_W-1:0] axis_out_rdata;"
        ),
        defines=(
            "axis_in_addr",
            "axis_out_addr",
            "axis_in_we",
            "axis_in_wdata",
            "axis_out_rdata",
        ),
        label="Activation I/O signals",
    )

    for sig in ("axis_in_enable", "axis_in_done", "axis_out_enable", "axis_out_done"):
        mod.add_signal(sig, L())

    mod.add_signal("fabric_act_in", L(width="AXI_DATA_W"))
    mod.add_assign(
        "fabric_act_in", "s_axis_tdata", comment="Fabric input is AXI-Stream data"
    )

    mod.add_signal("cycle_counter", L(width=32))
    mod.add_signal("counting", L())


def _add_tile_top_axi_lite(mod: RTLModule) -> None:
    mod.add_instance(
        name="u_ctrl",
        module_type="axi_lite_ctrl",
        params={"VERSION": "32'h0002_0000"},
        port_map={
            "clk": "clk",
            "rst_n": "rst_n",
            "s_axi_awaddr": "s_axi_awaddr",
            "s_axi_awvalid": "s_axi_awvalid",
            "s_axi_awready": "s_axi_awready",
            "s_axi_wdata": "s_axi_wdata",
            "s_axi_wstrb": "s_axi_wstrb",
            "s_axi_wvalid": "s_axi_wvalid",
            "s_axi_wready": "s_axi_wready",
            "s_axi_bresp": "s_axi_bresp",
            "s_axi_bvalid": "s_axi_bvalid",
            "s_axi_bready": "s_axi_bready",
            "s_axi_araddr": "s_axi_araddr",
            "s_axi_arvalid": "s_axi_arvalid",
            "s_axi_arready": "s_axi_arready",
            "s_axi_rdata": "s_axi_rdata",
            "s_axi_rresp": "s_axi_rresp",
            "s_axi_rvalid": "s_axi_rvalid",
            "s_axi_rready": "s_axi_rready",
            "ctrl_start": "ctrl_start",
            "ctrl_soft_rst": "ctrl_soft_rst",
            "ctrl_continuous": "ctrl_continuous",
            "irq_en": "irq_en_out",
            "status_idle": "status_idle",
            "status_busy": "status_busy",
            "status_done": "status_done_reg",
            "status_error": "status_error",
            "cycle_count": "cycle_count_reg",
            "inf_count": "inf_count_reg",
            "error_code": "error_code_reg",
            "layer_status": "2'b0",
            "irq_done": "irq_done_pulse",
            "irq_error": "irq_error_pulse",
            "irq": "irq",
        },
    )


def _add_tile_top_axi_stream(mod: RTLModule) -> None:
    mod.add_instance(
        name="u_axis_in",
        module_type="axi_stream_in",
        params={
            "AXI_DATA_W": "AXI_DATA_W",
            "NUM_BEATS": "ACT_DEPTH",
            "ACT_DEPTH": "ACT_DEPTH",
        },
        port_map={
            "clk": "clk",
            "rst_n": "rst_n",
            "enable": "axis_in_enable",
            "done": "axis_in_done",
            "s_axis_tdata": "s_axis_tdata",
            "s_axis_tvalid": "s_axis_tvalid",
            "s_axis_tready": "s_axis_tready",
            "s_axis_tlast": "s_axis_tlast",
            "s_axis_tkeep": "s_axis_tkeep",
            "act_addr": "axis_in_addr",
            "act_we": "axis_in_we",
            "act_wdata": "axis_in_wdata",
        },
    )

    mod.add_instance(
        name="u_axis_out",
        module_type="axi_stream_out",
        params={
            "AXI_DATA_W": "AXI_DATA_W",
            "NUM_BEATS": "ACT_DEPTH",
            "ACT_DEPTH": "ACT_DEPTH",
        },
        port_map={
            "clk": "clk",
            "rst_n": "rst_n",
            "enable": "axis_out_enable",
            "done": "axis_out_done",
            "m_axis_tdata": "m_axis_tdata",
            "m_axis_tvalid": "m_axis_tvalid",
            "m_axis_tready": "m_axis_tready",
            "m_axis_tlast": "m_axis_tlast",
            "m_axis_tkeep": "m_axis_tkeep",
            "act_addr": "axis_out_addr",
            "act_rdata": "axis_out_rdata",
        },
    )


def _add_tile_fabric_instance(mod: RTLModule) -> None:
    mod.add_instance(
        name="u_fabric",
        module_type="tile_fabric",
        params={"ACT_DEPTH": "ACT_DEPTH"},
        port_map={
            "clk": "clk",
            "rst_n": "rst_n",
            "start": "fabric_start",
            "done": "fabric_done",
            "busy": "fabric_busy",
            "act_in_data": "fabric_act_in",
            "act_out_data": "axis_out_rdata",
        },
        comment="Model-specific tile fabric (generated by lower_tile_fabric_module)",
    )


def _add_tile_top_fsm(mod: RTLModule) -> None:
    mod.add_raw(
        code=(
            "always_comb begin\n"
            "    state_next = state;\n"
            "    case (state)\n"
            "        S_IDLE:       if (ctrl_start) state_next = S_RECV_IN;\n"
            "        S_RECV_IN:    if (axis_in_done) state_next = S_RUN_FABRIC;\n"
            "        S_RUN_FABRIC: if (fabric_done) state_next = S_SEND_OUT;\n"
            "        S_SEND_OUT:   if (axis_out_done) state_next = S_DONE;\n"
            "        S_DONE:       state_next = S_IDLE;\n"
            "        default:      state_next = S_IDLE;\n"
            "    endcase\n"
            "end\n"
            "\n"
            "always_ff @(posedge clk) begin\n"
            "    if (!rst_n || ctrl_soft_rst)\n"
            "        state <= S_IDLE;\n"
            "    else\n"
            "        state <= state_next;\n"
            "end\n"
            "\n"
            "// Control signals\n"
            "assign axis_in_enable  = (state == S_RECV_IN);\n"
            "assign axis_out_enable = (state == S_SEND_OUT);\n"
            "assign fabric_start    = (state == S_RECV_IN && axis_in_done);\n"
            "\n"
            "// Status\n"
            "assign status_idle  = (state == S_IDLE);\n"
            "assign status_busy  = (state != S_IDLE && state != S_DONE);\n"
            "assign status_error = 1'b0;\n"
            "assign error_code_reg = 32'b0;\n"
            "\n"
            "// Done and interrupt\n"
            "always_ff @(posedge clk) begin\n"
            "    if (!rst_n || state == S_IDLE) begin\n"
            "        status_done_reg <= 1'b0;\n"
            "        irq_done_pulse  <= 1'b0;\n"
            "    end else if (state == S_DONE) begin\n"
            "        status_done_reg <= 1'b1;\n"
            "        irq_done_pulse  <= 1'b1;\n"
            "    end else begin\n"
            "        irq_done_pulse <= 1'b0;\n"
            "    end\n"
            "end\n"
            "\n"
            "assign irq_error_pulse = 1'b0;\n"
            "\n"
            "// Cycle counter\n"
            "always_ff @(posedge clk) begin\n"
            "    if (!rst_n || state == S_IDLE) begin\n"
            "        cycle_counter <= 32'b0;\n"
            "        counting <= 1'b0;\n"
            "    end else if (state == S_RECV_IN && !counting) begin\n"
            "        counting <= 1'b1;\n"
            "        cycle_counter <= 32'b0;\n"
            "    end else if (counting && state != S_DONE) begin\n"
            "        cycle_counter <= cycle_counter + 1;\n"
            "    end\n"
            "end\n"
            "\n"
            "assign cycle_count_reg = cycle_counter;\n"
            "\n"
            "// Inference counter\n"
            "always_ff @(posedge clk) begin\n"
            "    if (!rst_n)\n"
            "        inf_count_reg <= 32'b0;\n"
            "    else if (state == S_DONE && state_next == S_IDLE)\n"
            "        inf_count_reg <= inf_count_reg + 1;\n"
            "end"
        ),
        defines=(
            "state",
            "state_next",
            "axis_in_enable",
            "axis_out_enable",
            "fabric_start",
            "status_idle",
            "status_busy",
            "status_error",
            "error_code_reg",
            "status_done_reg",
            "irq_done_pulse",
            "irq_error_pulse",
            "cycle_counter",
            "counting",
            "cycle_count_reg",
            "inf_count_reg",
        ),
        uses=(
            "ctrl_start",
            "ctrl_soft_rst",
            "axis_in_done",
            "axis_out_done",
            "fabric_done",
            "rst_n",
            "clk",
        ),
        label="Top-level FSM, status, counters",
    )


# ---------------------------------------------------------------------------
# ASIC weight ROM module lowering
# ---------------------------------------------------------------------------


def lower_asic_rom_module(
    tile: TileConfig,
    graph: Graph,
    parallelism: int,
) -> Optional[RTLModule]:
    """Lower a single tile's hardcoded weight ROM to an ``RTLModule``.

    Returns ``None`` for tiles that have no weight data (non-MAC tiles, tiles
    whose source operator has no constant weight tensor). The hardcoded
    ``initial`` block populating the ROM array is emitted as an ``RTLRawBlock``
    — opaque to RTL passes but visible to the emitter.
    """
    if tile.weight_bytes == 0:
        return None
    node = graph.nodes.get(tile.operator_assignment)
    if node is None:
        return None

    weight_data = None
    if len(node.inputs) >= 2:
        wt = graph.tensors.get(node.inputs[1])
        if wt and wt.is_constant and wt.data is not None:
            weight_data = wt.data

    if weight_data is None:
        return None

    if tile.partition_info:
        start, end = tile.partition_info["output_range"]
        weight_data = weight_data[start:end]

    flat = weight_data.flatten().astype(np.int8)
    row_width = parallelism
    num_rows = max(len(flat) // row_width, 1)
    width_bits = row_width * 8

    tid = tile.tile_id
    mod = RTLModule(
        name=f"rom_tile_{tid}",
        header_comment=(
            f"rom_tile_{tid}.sv — ASIC weight ROM for tile {tid}\n"
            f"Node: {tile.operator_assignment}\n"
            f"Weight bytes: {tile.weight_bytes}"
        ),
    )
    mod.add_param("DEPTH", num_rows)
    mod.add_param("WIDTH", width_bits)

    L = RTLType.logic
    mod.add_port("clk", RTLDir.INPUT, L())
    mod.add_port("addr", RTLDir.INPUT, L(width="$clog2(DEPTH)"))
    mod.add_port("rdata", RTLDir.OUTPUT, L(width="WIDTH"))

    mod.add_signal(
        "rom",
        RTLType(width="WIDTH", unpacked_dims=(f"0:{num_rows - 1}",)),
    )

    init_lines: list[str] = []
    for row in range(num_rows):
        start_byte = row * row_width
        end_byte = min(start_byte + row_width, len(flat))
        row_bytes = flat[start_byte:end_byte]
        if len(row_bytes) < row_width:
            row_bytes = np.pad(row_bytes, (0, row_width - len(row_bytes)))
        hex_str = "".join(f"{int(b) & 0xFF:02x}" for b in row_bytes)
        init_lines.append(f"    rom[{row}] = {width_bits}'h{hex_str};")

    mod.add_raw(
        code=(
            "always_ff @(posedge clk) begin\n"
            "    rdata <= rom[addr];\n"
            "end\n"
            "\n"
            "initial begin\n" + "\n".join(init_lines) + "\nend"
        ),
        defines=("rdata", "rom"),
        uses=("addr", "clk"),
    )

    mod.verify()
    return mod
