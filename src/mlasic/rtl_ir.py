"""MLASIC Hardware IR.

A structural SystemVerilog IR sitting between the scheduled compute graph
(``mlasic.ir.Graph``) and emitted Verilog text. Replaces the f-string
templating in ``rtl_gen.py`` with an explicit object graph that downstream
passes can analyze, transform, and lower.

Design notes
------------
* The IR models SystemVerilog **structurally**: modules, parameter and port
  declarations, signal declarations, sub-module instances with parameter
  and port maps, and continuous assigns. This is the level at which we can
  do structural transforms (dead-instance elimination, parameter coalescing,
  net liveness, structural diff).
* SV constructs that are not worth modeling (FSM ``always_comb`` bodies,
  AXI handshake clauses) are represented by ``RTLRawBlock`` — an opaque
  string that the emitter passes through verbatim. This is the same trade
  LLVM makes with inline-asm: keep the IR small, accept that some leaves
  are opaque.
* Pass-friendly: every container is a plain ``dataclass`` with a stable
  iteration order, supports structural equality, and exposes accessors
  (``ports_by_name``, ``instances_by_type``) so passes don't have to walk
  raw lists. Passes mutate ``RTLModule`` in place and call
  ``module.invalidate_cache()`` — same convention as ``mlasic.ir.Graph``.

Stability invariants enforced by ``RTLModule.verify()``:
    HW-1.1  No two ports share a name.
    HW-1.2  No two instances share a name.
    HW-1.3  No two parameters share a name.
    HW-1.4  Every instance port_map key references a port the sub-module
            declares (validated when sub-module is in a ``ModuleLibrary``).
    HW-1.5  Every signal referenced by an assign or instance port map is
            either a port of this module, a declared signal, a parameter,
            or a literal expression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Direction and primitive declarations
# ---------------------------------------------------------------------------


class RTLDir(Enum):
    """Port direction."""

    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


@dataclass(frozen=True)
class RTLType:
    """Bit-vector type. ``logic`` for unsigned, ``logic signed`` for signed.

    ``width`` is the total bit-width (1 for a scalar). ``packed_dims`` carries
    extra packed dimensions for arrays-of-bytes inside a module port (e.g.
    ``logic [PARALLELISM*8-1:0]``). ``unpacked_dims`` carries unpacked array
    dimensions like ``[0:PARALLELISM-1]``.

    ``width`` may be either an int (resolved) or a string (a Verilog
    expression like ``"PARALLELISM*8"``); strings are passed through as-is.
    """

    width: int | str = 1
    signed: bool = False
    unpacked_dims: tuple[str, ...] = ()

    @staticmethod
    def logic(width: int | str = 1, signed: bool = False) -> "RTLType":
        return RTLType(width=width, signed=signed)

    def render_decl(self, name: str) -> str:
        """Render as ``logic [W-1:0] name [unpacked]``."""
        sign = "logic signed" if self.signed else "logic"
        if isinstance(self.width, int):
            packed = "" if self.width == 1 else f"[{self.width - 1}:0]"
        else:
            packed = f"[{self.width}-1:0]"
        unpacked = "".join(f" [{d}]" for d in self.unpacked_dims)
        return f"{sign} {packed} {name}{unpacked}".replace("  ", " ").strip()


@dataclass
class RTLParamDecl:
    """Module parameter declaration (``parameter int FOO = 5``)."""

    name: str
    default: int | str
    type_str: str = "int"  # "int", "logic [31:0]", etc.

    def render(self) -> str:
        return f"parameter {self.type_str} {self.name} = {self.default}"


@dataclass
class RTLPortDecl:
    """Module port declaration."""

    name: str
    direction: RTLDir
    type: RTLType

    def render(self) -> str:
        # Inputs use the implicit wire type — `input clk`, `input [7:0] d`,
        # `input signed [7:0] d`. Outputs and inouts must declare `logic`
        # because the body assigns them procedurally.
        is_input = self.direction == RTLDir.INPUT
        if is_input:
            type_field = "signed" if self.type.signed else ""
        else:
            type_field = "logic signed" if self.type.signed else "logic"
        if isinstance(self.type.width, int):
            packed = "" if self.type.width == 1 else f"[{self.type.width - 1}:0]"
        else:
            packed = f"[{self.type.width}-1:0]"
        unpacked = "".join(f" [{d}]" for d in self.type.unpacked_dims)
        parts = [self.direction.value]
        if type_field:
            parts.append(type_field)
        if packed:
            parts.append(packed)
        parts.append(f"{self.name}{unpacked}")
        return " ".join(parts)


@dataclass
class RTLSignalDecl:
    """Internal signal declaration (``logic [7:0] foo;``)."""

    name: str
    type: RTLType
    comment: Optional[str] = None

    def render(self) -> str:
        decl = self.type.render_decl(self.name) + ";"
        if self.comment:
            decl += f"  // {self.comment}"
        return decl


# ---------------------------------------------------------------------------
# Body items (instances, assigns, raw blocks)
# ---------------------------------------------------------------------------


@dataclass
class RTLInstance:
    """Sub-module instance.

    ``module_type`` names a module that is either built-in (in the static
    rtl/ library) or another ``RTLModule`` produced by lowering.

    ``params`` is the parameter map (``.DEPTH(WEIGHT_DEPTH)``).
    ``port_map`` maps the sub-module's port name → expression in *this*
    module. Expressions are raw Verilog strings (signal names, literals,
    concats); the emitter does not interpret them, but ``referenced_names()``
    extracts identifier-like substrings for dataflow analysis.
    """

    name: str
    module_type: str
    params: dict[str, str] = field(default_factory=dict)
    port_map: dict[str, str] = field(default_factory=dict)
    comment: Optional[str] = None

    def referenced_names(self) -> set[str]:
        """Names referenced in port_map RHS expressions (best-effort)."""
        return _extract_identifiers(self.port_map.values())


@dataclass
class RTLContinuousAssign:
    """``assign lhs = rhs;``."""

    lhs: str
    rhs: str
    comment: Optional[str] = None

    def referenced_names(self) -> set[str]:
        return _extract_identifiers([self.rhs])


@dataclass
class RTLRawBlock:
    """Opaque SystemVerilog block — passed through verbatim by the emitter.

    Intended for FSMs, AXI handshake bodies, and other constructs that are
    not worth modeling structurally. The block declares the names it
    *defines* (drives) and *uses* so DCE / liveness still work.
    """

    code: str
    defines: tuple[str, ...] = ()
    uses: tuple[str, ...] = ()
    label: Optional[str] = None

    def referenced_names(self) -> set[str]:
        return set(self.uses)


# Body items are ordered so emitted Verilog reads top-to-bottom in a
# predictable way.
RTLBodyItem = RTLInstance | RTLContinuousAssign | RTLRawBlock


# ---------------------------------------------------------------------------
# Module and module library
# ---------------------------------------------------------------------------


@dataclass
class RTLModule:
    """A SystemVerilog module — the central RTL IR object.

    A module is constructed by the lowering pass (``Graph`` → ``RTLModule``),
    optionally rewritten by RTL-IR passes, and finally consumed by
    ``rtl_emit.emit_module`` to produce SystemVerilog text.
    """

    name: str
    params: list[RTLParamDecl] = field(default_factory=list)
    ports: list[RTLPortDecl] = field(default_factory=list)
    signals: list[RTLSignalDecl] = field(default_factory=list)
    body: list[RTLBodyItem] = field(default_factory=list)
    includes: list[str] = field(default_factory=list)
    header_comment: Optional[str] = None

    # Caches — invalidated on mutation
    _ports_by_name: Optional[dict[str, RTLPortDecl]] = field(
        default=None, repr=False, compare=False
    )
    _signals_by_name: Optional[dict[str, RTLSignalDecl]] = field(
        default=None, repr=False, compare=False
    )
    _instances_by_name: Optional[dict[str, RTLInstance]] = field(
        default=None, repr=False, compare=False
    )

    # ------------------------------------------------------------------
    # Builders — mutating helpers used by lowering passes
    # ------------------------------------------------------------------

    def add_param(self, name: str, default: int | str, type_str: str = "int") -> None:
        self.params.append(RTLParamDecl(name=name, default=default, type_str=type_str))
        self.invalidate_cache()

    def add_port(self, name: str, direction: RTLDir, type: RTLType) -> None:
        self.ports.append(RTLPortDecl(name=name, direction=direction, type=type))
        self.invalidate_cache()

    def add_signal(
        self, name: str, type: RTLType, comment: Optional[str] = None
    ) -> None:
        self.signals.append(RTLSignalDecl(name=name, type=type, comment=comment))
        self.invalidate_cache()

    def add_instance(
        self,
        name: str,
        module_type: str,
        params: Optional[dict[str, str]] = None,
        port_map: Optional[dict[str, str]] = None,
        comment: Optional[str] = None,
    ) -> RTLInstance:
        inst = RTLInstance(
            name=name,
            module_type=module_type,
            params=dict(params or {}),
            port_map=dict(port_map or {}),
            comment=comment,
        )
        self.body.append(inst)
        self.invalidate_cache()
        return inst

    def add_assign(
        self, lhs: str, rhs: str, comment: Optional[str] = None
    ) -> RTLContinuousAssign:
        a = RTLContinuousAssign(lhs=lhs, rhs=rhs, comment=comment)
        self.body.append(a)
        return a

    def add_raw(
        self,
        code: str,
        defines: Iterable[str] = (),
        uses: Iterable[str] = (),
        label: Optional[str] = None,
    ) -> RTLRawBlock:
        b = RTLRawBlock(
            code=code,
            defines=tuple(defines),
            uses=tuple(uses),
            label=label,
        )
        self.body.append(b)
        return b

    def add_include(self, path: str) -> None:
        if path not in self.includes:
            self.includes.append(path)

    # ------------------------------------------------------------------
    # Accessors / cached lookups
    # ------------------------------------------------------------------

    def invalidate_cache(self) -> None:
        self._ports_by_name = None
        self._signals_by_name = None
        self._instances_by_name = None

    @property
    def ports_by_name(self) -> dict[str, RTLPortDecl]:
        if self._ports_by_name is None:
            self._ports_by_name = {p.name: p for p in self.ports}
        return self._ports_by_name

    @property
    def signals_by_name(self) -> dict[str, RTLSignalDecl]:
        if self._signals_by_name is None:
            self._signals_by_name = {s.name: s for s in self.signals}
        return self._signals_by_name

    @property
    def instances(self) -> list[RTLInstance]:
        return [b for b in self.body if isinstance(b, RTLInstance)]

    @property
    def assigns(self) -> list[RTLContinuousAssign]:
        return [b for b in self.body if isinstance(b, RTLContinuousAssign)]

    @property
    def raw_blocks(self) -> list[RTLRawBlock]:
        return [b for b in self.body if isinstance(b, RTLRawBlock)]

    @property
    def instances_by_name(self) -> dict[str, RTLInstance]:
        if self._instances_by_name is None:
            self._instances_by_name = {i.name: i for i in self.instances}
        return self._instances_by_name

    def instances_by_type(self, module_type: str) -> list[RTLInstance]:
        return [i for i in self.instances if i.module_type == module_type]

    # ------------------------------------------------------------------
    # Liveness / dataflow helpers
    # ------------------------------------------------------------------

    def all_referenced_names(self) -> set[str]:
        """Union of names referenced by every body item."""
        out: set[str] = set()
        for b in self.body:
            out |= b.referenced_names()
        return out

    def all_driven_names(self) -> set[str]:
        """Names this module drives (assign LHS + raw-block defines).

        Sub-module port_map outputs are *not* included here — that requires
        knowing the sub-module's port directions, i.e. a ``ModuleLibrary``.
        """
        out: set[str] = set()
        for a in self.assigns:
            out.add(_lhs_name(a.lhs))
        for b in self.raw_blocks:
            out.update(b.defines)
        return out

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self, library: Optional["ModuleLibrary"] = None) -> None:
        """Check structural invariants. Raises ``RTLIRValidationError``."""
        seen_ports: set[str] = set()
        for p in self.ports:
            if p.name in seen_ports:
                raise RTLIRValidationError(
                    f"HW-1.1: duplicate port {p.name!r} in module {self.name!r}"
                )
            seen_ports.add(p.name)

        seen_params: set[str] = set()
        for p in self.params:
            if p.name in seen_params:
                raise RTLIRValidationError(
                    f"HW-1.3: duplicate parameter {p.name!r} in module {self.name!r}"
                )
            seen_params.add(p.name)

        seen_inst: set[str] = set()
        for inst in self.instances:
            if inst.name in seen_inst:
                raise RTLIRValidationError(
                    f"HW-1.2: duplicate instance {inst.name!r} in module {self.name!r}"
                )
            seen_inst.add(inst.name)

        if library is not None:
            for inst in self.instances:
                sub = library.get(inst.module_type)
                if sub is None:
                    continue  # External / built-in module — skip check
                sub_ports = {p.name for p in sub.ports}
                for k in inst.port_map:
                    if k not in sub_ports:
                        raise RTLIRValidationError(
                            f"HW-1.4: instance {inst.name!r} of "
                            f"{inst.module_type!r} maps unknown port {k!r}"
                        )


@dataclass
class ModuleLibrary:
    """A collection of ``RTLModule`` definitions for cross-module lookups.

    Holds *signatures* of built-in (rtl/ library) modules and any modules
    produced by the lowering pass. Signatures only need ``name``, ``params``,
    and ``ports`` — body is irrelevant for cross-module checks.
    """

    modules: dict[str, RTLModule] = field(default_factory=dict)

    def add(self, mod: RTLModule) -> None:
        self.modules[mod.name] = mod

    def get(self, name: str) -> Optional[RTLModule]:
        return self.modules.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self.modules


class RTLIRValidationError(Exception):
    """Raised when an ``RTLModule`` violates a structural invariant."""


# ---------------------------------------------------------------------------
# Identifier extraction helpers
# ---------------------------------------------------------------------------


def _extract_identifiers(exprs: Iterable[str]) -> set[str]:
    """Best-effort identifier extraction from Verilog expressions.

    Walks character by character; collects maximal runs of [A-Za-z_][A-Za-z0-9_]*.
    Filters Verilog literals (``8'd5``, ``32'h0001_0000``, ``1'b0``) and the
    handful of keyword-like tokens we use in expressions.
    """
    out: set[str] = set()
    for expr in exprs:
        if not expr:
            continue
        i = 0
        n = len(expr)
        while i < n:
            c = expr[i]
            if c.isalpha() or c == "_":
                j = i + 1
                while j < n and (expr[j].isalnum() or expr[j] == "_"):
                    j += 1
                tok = expr[i:j]
                # Skip Verilog literal suffixes: when the previous non-space
                # char is "'" the token is a literal base ('h'/'d'/'b'/'o').
                k = i - 1
                while k >= 0 and expr[k] == " ":
                    k -= 1
                if k >= 0 and expr[k] == "'":
                    i = j
                    continue
                if tok not in _RESERVED_TOKENS:
                    out.add(tok)
                i = j
            elif c.isdigit():
                # Skip number — possibly with width/base suffix
                j = i + 1
                while j < n and (expr[j].isdigit() or expr[j] == "_"):
                    j += 1
                # Width prefix like "32'h..." — consume base + value
                if j < n and expr[j] == "'":
                    j += 1  # apostrophe
                    if j < n and expr[j] in "hdbosHDBOS":
                        j += 1
                    while j < n and (expr[j].isalnum() or expr[j] == "_"):
                        j += 1
                i = j
            else:
                i += 1
    return out


def _lhs_name(lhs: str) -> str:
    """Extract the bare signal name from an assign LHS like ``foo[7:0]``."""
    s = lhs.strip()
    for delim in "[":
        idx = s.find(delim)
        if idx >= 0:
            s = s[:idx]
            break
    return s.strip()


_RESERVED_TOKENS: frozenset[str] = frozenset(
    {
        # Verilog/SystemVerilog keywords we sometimes embed in expressions
        "begin",
        "end",
        "if",
        "else",
        "logic",
        "signed",
        "unsigned",
        "always_comb",
        "always_ff",
        "posedge",
        "negedge",
        "or",
        "and",
        "not",
        # System functions
        "clog2",
    }
)
