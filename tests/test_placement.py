"""Tests for liveness-driven placement: bank allocator and peak-aware reorder.

These tests verify that placement decisions are computed from the input
model rather than hardcoded:

  - ``BankAllocator`` (linear scheduler): on a strict chain it produces
    A/B alternation by liveness analysis, not by ``idx % 2``. On a DAG
    that would clobber a still-live tensor it raises ``BankConflict``.
  - ``peak_aware_topological_order`` (DAG scheduler): on a branchy
    graph where the canonical topo order produces a high peak, the
    Sethi–Ullman-style reorder reduces it.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlasic.dag_scheduler import (
    estimate_peak_bytes,
    peak_aware_topological_order,
)
from mlasic.ir import (
    FusedLinearAttrs,
    Graph,
    OpNode,
    OpType,
    QuantParams,
    Tensor,
    TensorType,
)
from mlasic.scheduler import BankAllocator, Scheduler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _qp() -> QuantParams:
    return QuantParams(scale=0.1, zero_point=0, calibrated=True)


def _linear_node(
    name: str,
    inp: str,
    out: str,
    in_dim: int,
    out_dim: int,
    has_relu: bool = True,
) -> tuple[OpNode, dict[str, Tensor]]:
    qp = _qp()
    attrs = FusedLinearAttrs(
        input_dim=in_dim,
        output_dim=out_dim,
        has_relu=has_relu,
        weight_quant=qp,
        input_quant=qp,
        output_quant=qp,
        requant_scale_fixed=65536,
        requant_shift=16,
    )
    op = OpType.FUSED_LINEAR_RELU if has_relu else OpType.FUSED_LINEAR
    w, b = f"{name}_w", f"{name}_b"
    node = OpNode(name, op, [inp, w, b], [out])
    node.fused_attrs = attrs
    tensors = {
        inp: Tensor(inp, TensorType((1, in_dim), np.dtype(np.int8))),
        w: Tensor(
            w,
            TensorType((in_dim, out_dim), np.dtype(np.int8)),
            data=np.zeros((in_dim, out_dim), dtype=np.int8),
        ),
        b: Tensor(
            b,
            TensorType((out_dim,), np.dtype(np.int32)),
            data=np.zeros(out_dim, dtype=np.int32),
        ),
        out: Tensor(out, TensorType((1, out_dim), np.dtype(np.int8))),
    }
    return node, tensors


def _linear_chain(dims: list[int]) -> Graph:
    """Build a quantized linear chain ``dims[0] -> dims[1] -> ... -> dims[-1]``."""
    nodes: dict[str, OpNode] = {}
    tensors: dict[str, Tensor] = {}
    prev = "x"
    for i in range(len(dims) - 1):
        out = f"h{i}" if i < len(dims) - 2 else "y"
        node, ts = _linear_node(f"n{i}", prev, out, dims[i], dims[i + 1])
        nodes[node.name] = node
        # Don't overwrite an already-registered tensor (chain reuses prev).
        for tn, tv in ts.items():
            tensors.setdefault(tn, tv)
        prev = out
    return Graph(
        "chain",
        nodes,
        tensors,
        ["x"],
        [prev],
        stage="quantized",
    )


# ---------------------------------------------------------------------------
# BankAllocator
# ---------------------------------------------------------------------------


class TestBankAllocator:
    def test_strict_chain_collapses_to_ping_pong(self):
        """4-layer chain produces A/B/A/B alternation via liveness, not modulo."""
        graph = _linear_chain([640, 128, 128, 128, 640])
        sched = Scheduler().schedule(graph)
        assert sched.layers[0].act_in_bank == "A"
        assert sched.layers[0].act_out_bank == "B"
        assert sched.layers[1].act_in_bank == "B"
        assert sched.layers[1].act_out_bank == "A"
        assert sched.layers[2].act_in_bank == "A"
        assert sched.layers[2].act_out_bank == "B"
        assert sched.layers[3].act_in_bank == "B"
        assert sched.layers[3].act_out_bank == "A"

    def test_input_and_output_banks_always_differ(self):
        """A layer must read from a different bank than it writes to."""
        graph = _linear_chain([128, 64, 32, 16])
        sched = Scheduler().schedule(graph)
        for ls in sched.layers:
            assert ls.act_in_bank != ls.act_out_bank

    def test_handoff_preserved_between_layers(self):
        """Layer i+1's input bank == layer i's output bank."""
        graph = _linear_chain([128, 64, 32, 16, 8])
        sched = Scheduler().schedule(graph)
        for i in range(len(sched.layers) - 1):
            assert sched.layers[i].act_out_bank == sched.layers[i + 1].act_in_bank

    def test_third_live_tensor_raises_bank_conflict(self):
        """When a graph would require a 3rd active bank, fail loud.

        Fan-out from ``x`` to three independent layers. After ``n0``
        fires, ``x`` is still live (two more consumers) so it occupies
        bank A; ``h0`` occupies B. ``n1`` reads ``x`` again — A still
        holds x because it has one consumer left. Both banks are now
        full of live tensors, so n1's output has nowhere to go. The
        allocator must raise ``BankConflict`` rather than silently
        clobbering ``h0`` or ``x``.
        """
        n0, t0 = _linear_node("n0", "x", "h0", 64, 32)
        n1, t1 = _linear_node("n1", "x", "h1", 64, 32)
        n2, t2 = _linear_node("n2", "x", "h2", 64, 32)
        tensors = {**t0, **t1, **t2}
        graph = Graph(
            "fanout3",
            {"n0": n0, "n1": n1, "n2": n2},
            tensors,
            ["x"],
            ["h0", "h1", "h2"],
            stage="quantized",
        )
        alloc = BankAllocator(graph)
        in0, out0 = alloc.assign(n0)
        assert (in0, out0) == ("A", "B")
        with pytest.raises(BankAllocator.BankConflict):
            alloc.assign(n1)


# ---------------------------------------------------------------------------
# peak_aware_topological_order
# ---------------------------------------------------------------------------


class TestPeakAwareReorder:
    def test_linear_chain_unchanged(self):
        """One ready node at every step → reorder == canonical topo."""
        graph = _linear_chain([128, 64, 32, 16, 8])
        canonical = graph.topological_order()
        reordered = peak_aware_topological_order(graph)
        assert canonical == reordered

    def test_branchy_graph_reduces_peak(self):
        """A branchy DAG: deep-narrow branch should run before the wide one.

        Graph:
            x ---n_deep--> d (small, 8 elems)
            x ---n_wide--> w (large, 1024 elems)
            d, w --n_join--> y

        Canonical topo (Kahn's) processes nodes in insertion order, so it
        executes ``n_wide`` first — leaving its 1024-elem tensor alive
        while ``n_deep`` runs. The reorder should pick ``n_deep`` first
        so the small tensor is held alongside whatever ``n_wide``
        produces, lowering peak.
        """
        n_wide, tw = _linear_node("n_wide", "x", "w", 64, 1024, has_relu=False)
        n_deep, td = _linear_node("n_deep", "x", "d", 64, 8, has_relu=False)
        # Join: takes d, w as activation inputs; trivial weights.
        n_join, tj = _linear_node("n_join", "d", "y", 8, 4, has_relu=False)
        n_join.inputs = ["d", "w", n_join.inputs[1], n_join.inputs[2]]
        tensors = {**tw, **td, **tj}
        graph = Graph(
            "branchy",
            # Insertion order biases canonical Kahn's queue toward n_wide.
            {"n_wide": n_wide, "n_deep": n_deep, "n_join": n_join},
            tensors,
            ["x"],
            ["y"],
            stage="quantized",
        )
        canonical = graph.topological_order()
        reordered = peak_aware_topological_order(graph)
        peak_canonical = estimate_peak_bytes(graph, canonical)
        peak_reordered = estimate_peak_bytes(graph, reordered)
        assert peak_reordered <= peak_canonical
        # And the chosen order must place n_deep before n_wide.
        assert reordered.index("n_deep") < reordered.index("n_wide")

    def test_reorder_respects_dependencies(self):
        """Reordered order must still be a valid topological sort."""
        graph = _linear_chain([64, 32, 16])
        order = peak_aware_topological_order(graph)
        seen: set[str] = set()
        for name in order:
            node = graph.nodes[name]
            for inp in node.inputs:
                prod = graph._tensor_to_producer.get(inp)
                if prod and prod in graph.nodes:
                    assert prod in seen, f"{name} reads {inp} before {prod} runs"
            seen.add(name)
