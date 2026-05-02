"""MLASIC Stage 3: Dataflow Scheduling.

Assigns per-layer cycle budgets, SRAM addresses, and activation buffer
assignments to a quantized IR graph. Produces a Schedule object and
optional schedule.json for downstream stages.

RTL spec (corrected §6.4) is authoritative for cycle formulas.

Activation bank assignment is liveness-driven: a `BankAllocator` walks
the graph, tracks which physical bank holds which tensor, and picks an
output bank that is free at the moment the layer fires. For a strict
linear chain this collapses to the historical A/B ping-pong pattern; for
graphs with still-live producers it surfaces a SRAM conflict instead of
silently corrupting a live tensor.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from mlasic.ir import Graph, HardwareConstraints, LayerSchedule, Schedule

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CycleBreakdown — parameterized RTL FSM timing constants
# ---------------------------------------------------------------------------


@dataclass
class CycleBreakdown:
    """RTL FSM timing constants per tile.

    Default values match the corrected RTL spec §6.4:
      BIAS_LOAD(128) + COMPUTE(IN_DIM+1) + REQUANT(3) + WRITE_TILE(16) + NEXT/DONE(1)
      = IN_DIM + 149
    """

    bias_load: int = 128
    pipeline_fill: int = 1
    requant: int = 3
    write_tile: int = 16
    next_or_done: int = 1

    @property
    def overhead(self) -> int:
        """Fixed overhead per tile (everything except input_dim)."""
        return (
            self.bias_load + self.pipeline_fill + self.requant + self.write_tile + self.next_or_done
        )

    def cycles_per_tile(self, input_dim: int) -> int:
        """Total cycles per tile for the given input dimension."""
        return input_dim + self.overhead


# ---------------------------------------------------------------------------
# BankAllocator — liveness-driven activation bank assignment
# ---------------------------------------------------------------------------


@dataclass
class _BankState:
    """Tracks which tensor currently occupies a bank and how many consumers remain."""

    tensor: Optional[str] = None
    remaining_uses: int = 0


class BankAllocator:
    """Liveness-driven allocator for the two-bank activation SRAM.

    The accelerator has exactly two activation banks (A, B). For each
    layer in execution order, this allocator:
      1. Reads the input bank from the bank currently holding the layer's
         input tensor (the first layer's input is loaded into bank ``A``
         by AXI convention).
      2. Decrements the remaining-use counter for that input tensor; if
         it falls to zero, the bank is freed.
      3. Picks an output bank that is currently free (or about to be
         freed by this layer's input consumption). If both banks still
         hold live tensors, raises ``BankConflict``.

    For a strict linear chain (which is what the v0.1 RTL supports) this
    produces the canonical A/B alternation. For any DAG with a
    still-live skip activation we refuse to silently overwrite it.
    """

    INPUT_BANK = "A"
    BANKS = ("A", "B")

    class BankConflict(RuntimeError):
        """Raised when no free bank is available for a layer's output."""

    def __init__(self, graph: Graph):
        self.graph = graph
        graph._build_adjacency()
        self._consumer_count = self._count_consumers(graph)
        self._state: dict[str, _BankState] = {b: _BankState() for b in self.BANKS}
        # External graph input is staged into bank A by the AXI front-end.
        for tname in graph.inputs:
            self._state[self.INPUT_BANK] = _BankState(
                tensor=tname,
                remaining_uses=self._consumer_count.get(tname, 1),
            )
            break

    @staticmethod
    def _count_consumers(graph: Graph) -> dict[str, int]:
        counts: dict[str, int] = {}
        for node in graph.nodes.values():
            for inp in node.inputs:
                counts[inp] = counts.get(inp, 0) + 1
        # A graph output that no node consumes still has at least one
        # virtual reader (the AXI egress path), so reserve one use.
        for out_name in graph.outputs:
            counts.setdefault(out_name, 1)
        return counts

    def assign(self, node) -> tuple[str, str]:
        """Return ``(in_bank, out_bank)`` for ``node`` and update state."""
        in_tensor = self._activation_input_of(node)
        out_tensor = node.outputs[0] if node.outputs else f"{node.name}/out"

        in_bank = self._find_bank(in_tensor)
        if in_bank is None:
            raise self.BankConflict(
                f"Node '{node.name}': input tensor '{in_tensor}' is not resident "
                f"in any activation bank. Bank state: "
                f"A={self._state['A'].tensor!r}, B={self._state['B'].tensor!r}"
            )

        # Consume one use of the input tensor.
        in_state = self._state[in_bank]
        in_state.remaining_uses -= 1
        if in_state.remaining_uses <= 0:
            self._state[in_bank] = _BankState()

        # Pick an output bank that is currently free, preferring the
        # bank that is *not* the input bank so the layer can read and
        # write concurrently from physically separate SRAMs.
        out_bank = self._pick_free_bank(prefer_other_than=in_bank)
        if out_bank is None:
            other = "B" if in_bank == "A" else "A"
            raise self.BankConflict(
                f"Node '{node.name}': cannot place output. "
                f"Other bank ({other}) still holds live tensor "
                f"{self._state[other].tensor!r}. The v0.1 two-bank "
                f"hardware cannot schedule this DAG; promote to the "
                f"tile fabric backend."
            )

        out_uses = self._consumer_count.get(out_tensor, 1)
        self._state[out_bank] = _BankState(tensor=out_tensor, remaining_uses=out_uses)
        return in_bank, out_bank

    def _activation_input_of(self, node) -> str:
        """The first non-constant, in-graph tensor input is the activation."""
        for inp in node.inputs:
            t = self.graph.tensors.get(inp)
            if t is None or t.is_constant:
                continue
            return inp
        # Fall back to the first input — caller will surface a clear error.
        return node.inputs[0] if node.inputs else ""

    def _find_bank(self, tensor: str) -> Optional[str]:
        for bank, state in self._state.items():
            if state.tensor == tensor:
                return bank
        return None

    def _pick_free_bank(self, prefer_other_than: str) -> Optional[str]:
        other = "B" if prefer_other_than == "A" else "A"
        if self._state[other].tensor is None:
            return other
        if self._state[prefer_other_than].tensor is None:
            return prefer_other_than
        return None


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class Scheduler:
    """Main scheduling engine for Stage 3.

    Takes a quantized IR graph with fused operators and produces a Schedule
    with per-layer cycle budgets, SRAM addresses, and activation bank
    assignments.
    """

    def __init__(
        self,
        constraints: Optional[HardwareConstraints] = None,
        cycle_breakdown: Optional[CycleBreakdown] = None,
    ):
        self.constraints = constraints or HardwareConstraints()
        self.cycle_breakdown = cycle_breakdown or CycleBreakdown()

    def schedule(self, graph: Graph) -> Schedule:
        """Schedule the graph. Returns a Schedule and mutates the graph in place.

        Preconditions:
          - graph.stage == "quantized"
          - All fused nodes have fused_attrs with is_quantized == True

        Raises:
          ValueError: If preconditions are not met.
          RuntimeError: If SRAM budget is exceeded.
        """
        self._validate_preconditions(graph)

        ordered_names = graph.topological_order()
        ordered_nodes = [graph.nodes[n] for n in ordered_names]

        bank_allocator = BankAllocator(graph)

        layer_schedules: list[LayerSchedule] = []
        current_cycle = 0
        weight_row_cursor = 0
        bias_row_cursor = 0

        for idx, node in enumerate(ordered_nodes):
            attrs = node.fused_attrs
            input_dim = attrs.input_dim
            output_dim = attrs.output_dim

            parallelism = self._compute_parallelism(output_dim, self.constraints.max_parallelism)
            num_tiles = output_dim // parallelism
            cycles_per_tile = self.cycle_breakdown.cycles_per_tile(input_dim)
            total_cycles = num_tiles * cycles_per_tile

            # Weight SRAM: num_tiles * input_dim rows
            # RTL addressing: weight_addr = BASE + tile_idx * IN_DIM + input_idx
            weight_rows = num_tiles * input_dim
            weight_bytes = weight_rows * self.constraints.weight_row_bytes

            # Bias SRAM: ceil(output_dim / biases_per_row) rows
            bias_rows = math.ceil(output_dim / self.constraints.biases_per_row)
            bias_bytes = bias_rows * self.constraints.biases_per_row * 4  # INT32

            # Activation banks are picked by liveness analysis, not by
            # layer index parity. For a strict linear chain this matches
            # the historical A/B alternation; DAGs that would clobber a
            # still-live tensor raise a clear conflict.
            act_in_bank, act_out_bank = bank_allocator.assign(node)

            ls = LayerSchedule(
                layer_index=idx,
                layer_name=node.name,
                input_dim=input_dim,
                output_dim=output_dim,
                parallelism=parallelism,
                num_tiles=num_tiles,
                has_relu=attrs.has_relu,
                cycles_per_tile=cycles_per_tile,
                total_cycles=total_cycles,
                start_cycle=current_cycle,
                end_cycle=current_cycle + total_cycles,
                weight_start_row=weight_row_cursor,
                weight_rows=weight_rows,
                weight_bytes=weight_bytes,
                bias_start_row=bias_row_cursor,
                bias_rows=bias_rows,
                bias_bytes=bias_bytes,
                act_in_bank=act_in_bank,
                act_out_bank=act_out_bank,
            )

            node.schedule_info = ls
            layer_schedules.append(ls)

            current_cycle += total_cycles
            weight_row_cursor += weight_rows
            bias_row_cursor += bias_rows

        # Check SRAM budget
        total_weight_rows = weight_row_cursor
        total_bias_rows = bias_row_cursor
        if total_weight_rows > self.constraints.weight_bank_depth:
            raise RuntimeError(
                f"Weight SRAM overflow: need {total_weight_rows} rows, "
                f"have {self.constraints.weight_bank_depth}"
            )
        if total_bias_rows > self.constraints.bias_bank_depth:
            raise RuntimeError(
                f"Bias SRAM overflow: need {total_bias_rows} rows, "
                f"have {self.constraints.bias_bank_depth}"
            )

        total_compute_cycles = current_cycle
        total_cycles = total_compute_cycles + self.constraints.axi_overhead_cycles

        total_weight_bytes = sum(ls.weight_bytes for ls in layer_schedules)
        total_bias_bytes = sum(ls.bias_bytes for ls in layer_schedules)
        total_act_bytes = self.constraints.act_buffer_bytes * 2  # ping-pong

        sched = Schedule(
            layers=layer_schedules,
            total_compute_cycles=total_compute_cycles,
            total_cycles=total_cycles,
            total_weight_bytes=total_weight_bytes,
            total_bias_bytes=total_bias_bytes,
            total_act_bytes=total_act_bytes,
            clock_mhz=self.constraints.clock_mhz,
            axi_overhead_cycles=self.constraints.axi_overhead_cycles,
        )

        graph.stage = "scheduled"

        self._log_summary(sched)
        return sched

    @staticmethod
    def _compute_parallelism(output_dim: int, max_par: int) -> int:
        """Compute the largest factor of output_dim <= max_par."""
        par = min(output_dim, max_par)
        while par > 1:
            if output_dim % par == 0:
                return par
            par -= 1
        return 1

    @staticmethod
    def _validate_preconditions(graph: Graph) -> None:
        """Validate that the graph is ready for scheduling."""
        if graph.stage != "quantized":
            raise ValueError(
                f"Graph must be in 'quantized' stage for scheduling, got '{graph.stage}'"
            )

        if len(graph.nodes) == 0:
            raise ValueError("Cannot schedule an empty graph (0 nodes)")

        for node in graph.nodes.values():
            if node.fused_attrs is None:
                raise ValueError(f"Node {node.name} missing fused_attrs")
            if not node.fused_attrs.is_quantized:
                raise ValueError(f"Node {node.name} is not fully quantized")
            if node.fused_attrs.input_dim <= 0:
                raise ValueError(
                    f"Node {node.name} has invalid input_dim={node.fused_attrs.input_dim}"
                )
            if node.fused_attrs.output_dim <= 0:
                raise ValueError(
                    f"Node {node.name} has invalid output_dim={node.fused_attrs.output_dim}"
                )

    @staticmethod
    def _log_summary(sched: Schedule) -> None:
        """Log a human-readable schedule summary."""
        logger.info("=" * 72)
        logger.info("Schedule Summary")
        logger.info("=" * 72)
        logger.info(
            "%-6s %-20s %6s %6s %6s %8s %6s %6s %6s",
            "Layer",
            "Name",
            "InDim",
            "OutDim",
            "Tiles",
            "Cyc/Tile",
            "Total",
            "WtRows",
            "Act",
        )
        logger.info("-" * 72)
        for ls in sched.layers:
            logger.info(
                "%-6d %-20s %6d %6d %6d %8d %6d %6d %s->%s",
                ls.layer_index,
                ls.layer_name,
                ls.input_dim,
                ls.output_dim,
                ls.num_tiles,
                ls.cycles_per_tile,
                ls.total_cycles,
                ls.weight_rows,
                ls.act_in_bank,
                ls.act_out_bank,
            )
        logger.info("-" * 72)
        logger.info("Total compute cycles: %d", sched.total_compute_cycles)
        logger.info("AXI overhead cycles:  %d", sched.axi_overhead_cycles)
        logger.info("Total cycles:         %d", sched.total_cycles)
        logger.info("Latency:              %.2f µs (%.4f ms)", sched.latency_us, sched.latency_ms)
        logger.info(
            "Throughput:           %.0f inferences/sec", sched.throughput_inferences_per_sec
        )
        logger.info("Weight SRAM:          %d bytes", sched.total_weight_bytes)
        logger.info("Bias SRAM:            %d bytes", sched.total_bias_bytes)
        logger.info("Activation buffers:   %d bytes (2x ping-pong)", sched.total_act_bytes)
        logger.info("=" * 72)


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def schedule_to_json(
    schedule: Schedule,
    output_path: Optional[Union[str, Path]] = None,
) -> dict:
    """Serialize a Schedule to a JSON-compatible dict and optionally write to file.

    Args:
        schedule: The Schedule object to serialize.
        output_path: Optional file path to write the JSON to.

    Returns:
        JSON-compatible dict.
    """
    data = schedule.to_json()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Schedule written to %s", path)
    return data
