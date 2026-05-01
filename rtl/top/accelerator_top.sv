// accelerator_top.sv — Template top-level module
// This is a TEMPLATE. The RTLGenerator produces a model-specific version
// in the output directory with parameters.svh included and all values filled in.
//
// Architecture (from rtl-interface-spec.md §9.2):
//   - Single shared MAC array, time-multiplexed across layers
//   - Layer config ROM (localparams from parameters.svh)
//   - Top FSM: IDLE -> RECV_IN -> RUN_LAYER (Nx) -> SEND_OUT -> DONE
//   - Interrupt + cycle/inference counters
//
// See rtl_gen.py:RTLGenerator.generate_accelerator_top() for the actual
// generated code.

// This file exists as documentation; the generated version replaces it.
