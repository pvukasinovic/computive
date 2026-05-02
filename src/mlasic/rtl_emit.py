"""SystemVerilog emitter for ``mlasic.rtl_ir.RTLModule``.

A single walker that reads the structural RTL IR and produces synthesizable
SystemVerilog. The emitter is *deliberately dumb*: it does no transforms,
only formatting. All optimizations live in RTL-IR passes (see
``mlasic.rtl_passes``); the emitter must be safe to invoke at any point
after lowering.

Output format conventions (reverse-engineered from the legacy ``rtl_gen``
output to keep diffs minimal during the cutover):
  * Two-space indent inside the module body.
  * Includes emitted before the module declaration.
  * Param block uses ``module foo #( ... ) ( ... );`` form when there are
    parameters; bare ``module foo ( ... );`` otherwise.
  * Port block one-per-line, aligned direction, comma-separated.
  * Body items emitted in declaration order; signal declarations before
    the first body item.
"""

from __future__ import annotations

from typing import Iterable

from mlasic.rtl_ir import (
    RTLContinuousAssign,
    RTLInstance,
    RTLModule,
    RTLRawBlock,
)


def emit_module(mod: RTLModule) -> str:
    """Lower a single ``RTLModule`` to SystemVerilog source text."""
    lines: list[str] = []

    if mod.header_comment:
        for ln in mod.header_comment.splitlines():
            lines.append(f"// {ln}" if ln else "//")
        lines.append("")

    for inc in mod.includes:
        lines.append(f'`include "{inc}"')
    if mod.includes:
        lines.append("")

    lines.append(_render_module_decl(mod))
    lines.append("")

    # Signal declarations
    if mod.signals:
        for sig in mod.signals:
            lines.append("    " + sig.render())
        lines.append("")

    # Body items, separated by a blank line for readability
    for i, item in enumerate(mod.body):
        block = _render_body_item(item)
        if block:
            lines.append(block)
            if i != len(mod.body) - 1:
                lines.append("")

    lines.append("")
    lines.append("endmodule")
    lines.append("")
    return "\n".join(lines)


def emit_modules(mods: Iterable[RTLModule], header: str | None = None) -> str:
    """Concatenate multiple modules into one SV file (used for fabric files)."""
    pieces: list[str] = []
    if header is not None:
        for ln in header.splitlines():
            pieces.append(f"// {ln}" if ln else "//")
        pieces.append("")
    for mod in mods:
        pieces.append(emit_module(mod))
    return "\n".join(pieces)


# ---------------------------------------------------------------------------
# Internal renderers
# ---------------------------------------------------------------------------


def _render_module_decl(mod: RTLModule) -> str:
    """Render ``module foo #(...) (...);`` header with column-aligned ports."""
    lines: list[str] = []
    if mod.params:
        lines.append(f"module {mod.name} #(")
        for i, p in enumerate(mod.params):
            comma = "," if i != len(mod.params) - 1 else ""
            lines.append(f"    {p.render()}{comma}")
        if mod.ports:
            lines.append(") (")
        else:
            lines.append(") ();")
            return "\n".join(lines)
    else:
        if mod.ports:
            lines.append(f"module {mod.name} (")
        else:
            return f"module {mod.name} ();"

    for i, p in enumerate(mod.ports):
        comma = "," if i != len(mod.ports) - 1 else ""
        lines.append(f"    {p.render()}{comma}")
    lines.append(");")
    return "\n".join(lines)


def _render_body_item(item: RTLInstance | RTLContinuousAssign | RTLRawBlock) -> str:
    if isinstance(item, RTLInstance):
        return _render_instance(item)
    if isinstance(item, RTLContinuousAssign):
        return _render_assign(item)
    if isinstance(item, RTLRawBlock):
        return _render_raw_block(item)
    raise TypeError(f"Unknown RTL body item: {type(item).__name__}")


def _render_instance(inst: RTLInstance) -> str:
    """Render a sub-module instance.

        mac_array #(
            .PARALLELISM(PARALLELISM),
            .DATA_W(8)
        ) u_mac (
            .clk(clk),
            .rst_n(rst_n),
            ...
        );
    """
    lines: list[str] = []
    if inst.comment:
        for ln in inst.comment.splitlines():
            lines.append(f"    // {ln}")

    if inst.params:
        lines.append(f"    {inst.module_type} #(")
        param_items = list(inst.params.items())
        for i, (k, v) in enumerate(param_items):
            comma = "," if i != len(param_items) - 1 else ""
            lines.append(f"        .{k}({v}){comma}")
        lines.append(f"    ) {inst.name} (")
    else:
        lines.append(f"    {inst.module_type} {inst.name} (")

    port_items = list(inst.port_map.items())
    if not port_items:
        # Drop the trailing "(" we appended and emit a unit
        lines[-1] = f"    {inst.module_type} {inst.name} ();"
        return "\n".join(lines)

    for i, (k, v) in enumerate(port_items):
        comma = "," if i != len(port_items) - 1 else ""
        lines.append(f"        .{k}({v}){comma}")
    lines.append("    );")
    return "\n".join(lines)


def _render_assign(a: RTLContinuousAssign) -> str:
    suffix = f"  // {a.comment}" if a.comment else ""
    return f"    assign {a.lhs} = {a.rhs};{suffix}"


def _render_raw_block(b: RTLRawBlock) -> str:
    """Re-base the raw block's indent to module-body level (4 spaces).

    Detects the minimum indentation across non-blank lines and rewrites it to
    4 spaces, preserving relative nesting.
    """
    out: list[str] = []
    if b.label:
        out.append(f"    // ── {b.label} ──")
    raw_lines = b.code.splitlines()
    non_blank = [ln for ln in raw_lines if ln.strip()]
    if not non_blank:
        return "\n".join(out)
    min_indent = min(len(ln) - len(ln.lstrip(" ")) for ln in non_blank)
    for ln in raw_lines:
        if ln.strip() == "":
            out.append("")
        else:
            out.append("    " + ln[min_indent:])
    return "\n".join(out)
