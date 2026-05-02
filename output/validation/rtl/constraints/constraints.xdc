# constraints.xdc — Timing and pin constraints for Zynq UltraScale+ (Kria KV260)
# From rtl-interface-spec.md §10

# 100 MHz clock
create_clock -period 10.000 -name clk [get_ports clk]

# Async reset false path (reset synchronizer in PS)
set_false_path -from [get_ports rst_n]
