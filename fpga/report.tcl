# ==============================================================================
# report.tcl -- Post-Implementation Report Generation for MLASIC Accelerator
# ==============================================================================
#
# Purpose:
#   Generates utilization, timing, power, and DRC reports after implementation.
#   Prints a PPA summary to the console and compares against PRD targets.
#
# Usage:
#   Sourced by build.tcl after route_design completes.
#   Can also be run standalone on an open implemented design:
#     vivado -mode batch -source fpga/report.tcl -tclargs <output_dir>
#
# Output files (in output_dir):
#   utilization.rpt  -- Post-implementation resource utilization
#   timing.rpt       -- Timing summary (setup, hold, pulse width)
#   power.rpt        -- Estimated power breakdown
#   drc.rpt          -- Design Rule Check results
#
# Target: Xilinx Kria KV260 (xck26-sfvc784-2LV)
# ==============================================================================

# ------------------------------------------------------------------------------
# Parse output directory
# ------------------------------------------------------------------------------
if {[info exists argv] && [llength $argv] >= 1} {
    set report_dir [lindex $argv 0]
} elseif {![info exists report_dir]} {
    set report_dir "./reports"
}

file mkdir $report_dir

puts "============================================================"
puts "MLASIC Post-Implementation Reports"
puts "  Output directory: $report_dir"
puts "============================================================"

# ------------------------------------------------------------------------------
# PPA Targets from PRD (docs/PRD.md Section 3, Stage 7)
# KV260 resources: 117,120 LUTs, 234,240 FFs, 144 BRAM36K, 1,248 DSP48E2
# ------------------------------------------------------------------------------
set target_lut_pct      15.0
set target_bram_pct     40.0
set target_dsp_pct      50.0
set target_power_mw     50.0
set target_wns_ns       0.0

set kv260_luts          117120
set kv260_ffs           234240
set kv260_bram36        144
set kv260_dsp           1248

# ------------------------------------------------------------------------------
# 1. Utilization Report
# ------------------------------------------------------------------------------
puts "\n--- Generating utilization report ---"
report_utilization -file ${report_dir}/utilization.rpt
report_utilization -hierarchical -file ${report_dir}/utilization_hierarchical.rpt

puts "  Saved: ${report_dir}/utilization.rpt"
puts "  Saved: ${report_dir}/utilization_hierarchical.rpt"

# ------------------------------------------------------------------------------
# 2. Timing Report
# ------------------------------------------------------------------------------
puts "\n--- Generating timing report ---"
report_timing_summary -file ${report_dir}/timing.rpt -max_paths 20
report_timing -sort_by group -max_paths 10 -path_type summary \
    -file ${report_dir}/timing_paths.rpt

puts "  Saved: ${report_dir}/timing.rpt"
puts "  Saved: ${report_dir}/timing_paths.rpt"

# ------------------------------------------------------------------------------
# 3. Power Report
# ------------------------------------------------------------------------------
puts "\n--- Generating power report ---"
# Use switching activity from simulation if available, otherwise Vivado defaults
report_power -file ${report_dir}/power.rpt

puts "  Saved: ${report_dir}/power.rpt"

# ------------------------------------------------------------------------------
# 4. DRC Report
# ------------------------------------------------------------------------------
puts "\n--- Generating DRC report ---"
report_drc -file ${report_dir}/drc.rpt

puts "  Saved: ${report_dir}/drc.rpt"

# ------------------------------------------------------------------------------
# 5. Extract key metrics and print PPA summary
# ------------------------------------------------------------------------------
puts "\n============================================================"
puts "PPA SUMMARY"
puts "============================================================"

# Extract utilization numbers
set util_rpt [report_utilization -return_string]

# Parse LUT usage
set lut_used 0
set lut_avail $kv260_luts
if {[regexp {CLB LUTs\s*\|\s*(\d+)\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)} $util_rpt -> lut_used lut_avail]} {
    # Parsed successfully
} elseif {[regexp {Slice LUTs\s*\|\s*(\d+)\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)} $util_rpt -> lut_used lut_avail]} {
    # Alternative format
}

# Parse FF usage
set ff_used 0
set ff_avail $kv260_ffs
if {[regexp {CLB Registers\s*\|\s*(\d+)\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)} $util_rpt -> ff_used ff_avail]} {
    # Parsed successfully
} elseif {[regexp {Slice Registers\s*\|\s*(\d+)\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)} $util_rpt -> ff_used ff_avail]} {
    # Alternative format
}

# Parse BRAM usage
set bram_used 0
set bram_avail $kv260_bram36
if {[regexp {Block RAM Tile\s*\|\s*([\d.]+)\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)} $util_rpt -> bram_used bram_avail]} {
    # Parsed successfully
}

# Parse DSP usage
set dsp_used 0
set dsp_avail $kv260_dsp
if {[regexp {DSPs\s*\|\s*(\d+)\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)} $util_rpt -> dsp_used dsp_avail]} {
    # Parsed successfully
}

# Calculate percentages
set lut_pct  [expr {$lut_avail > 0  ? double($lut_used)  / $lut_avail  * 100.0 : 0.0}]
set ff_pct   [expr {$ff_avail > 0   ? double($ff_used)   / $ff_avail   * 100.0 : 0.0}]
set bram_pct [expr {$bram_avail > 0 ? double($bram_used) / $bram_avail * 100.0 : 0.0}]
set dsp_pct  [expr {$dsp_avail > 0  ? double($dsp_used)  / $dsp_avail  * 100.0 : 0.0}]

puts ""
puts [format "  Resource       | Used     | Available | Util %%  | Target"]
puts [format "  ---------------+----------+-----------+---------+--------"]
puts [format "  LUTs           | %8s | %9s | %5.1f%%  | <%s%%" $lut_used $lut_avail $lut_pct $target_lut_pct]
puts [format "  Flip-Flops     | %8s | %9s | %5.1f%%  | --" $ff_used $ff_avail $ff_pct]
puts [format "  BRAM (36Kb)    | %8s | %9s | %5.1f%%  | <%s%%" $bram_used $bram_avail $bram_pct $target_bram_pct]
puts [format "  DSP48E2        | %8s | %9s | %5.1f%%  | <%s%%" $dsp_used $dsp_avail $dsp_pct $target_dsp_pct]
puts ""

# Extract timing
set timing_rpt [report_timing_summary -return_string]
set wns "N/A"
set tns "N/A"
if {[regexp {WNS\(ns\)\s+TNS\(ns\).*\n\s*-+\s*\n\s*([-\d.]+)\s+([-\d.]+)} $timing_rpt -> wns tns]} {
    # Parsed successfully
}

puts [format "  Timing         | WNS = %s ns | TNS = %s ns" $wns $tns]
puts ""

# Extract power
set power_rpt [report_power -return_string]
set total_power "N/A"
set dynamic_power "N/A"
set static_power "N/A"
if {[regexp {Total On-Chip Power[^\|]*\|\s*([\d.]+)} $power_rpt -> total_power]} {
    # Total power in W
}
if {[regexp {Dynamic[^\|]*\|\s*([\d.]+)} $power_rpt -> dynamic_power]} {
    # Dynamic power in W
}
if {[regexp {Device Static[^\|]*\|\s*([\d.]+)} $power_rpt -> static_power]} {
    # Static power in W
}

puts [format "  Power          | Total = %s W | Dynamic = %s W | Static = %s W" \
    $total_power $dynamic_power $static_power]
puts ""

# ------------------------------------------------------------------------------
# 6. Pass/Fail assessment against PRD targets
# ------------------------------------------------------------------------------
puts "------------------------------------------------------------"
puts "TARGET COMPARISON"
puts "------------------------------------------------------------"

set all_pass 1

# LUT check
if {[string is double $lut_pct] && $lut_pct <= $target_lut_pct} {
    puts [format "  LUT utilization   : PASS (%.1f%% <= %.1f%%)" $lut_pct $target_lut_pct]
} else {
    puts [format "  LUT utilization   : FAIL (%.1f%% > %.1f%%)" $lut_pct $target_lut_pct]
    set all_pass 0
}

# BRAM check
if {[string is double $bram_pct] && $bram_pct <= $target_bram_pct} {
    puts [format "  BRAM utilization  : PASS (%.1f%% <= %.1f%%)" $bram_pct $target_bram_pct]
} else {
    puts [format "  BRAM utilization  : FAIL (%.1f%% > %.1f%%)" $bram_pct $target_bram_pct]
    set all_pass 0
}

# DSP check
if {[string is double $dsp_pct] && $dsp_pct <= $target_dsp_pct} {
    puts [format "  DSP utilization   : PASS (%.1f%% <= %.1f%%)" $dsp_pct $target_dsp_pct]
} else {
    puts [format "  DSP utilization   : FAIL (%.1f%% > %.1f%%)" $dsp_pct $target_dsp_pct]
    set all_pass 0
}

# Timing check
if {[string is double $wns] && $wns >= $target_wns_ns} {
    puts [format "  Timing (WNS)      : PASS (%s ns >= %.1f ns)" $wns $target_wns_ns]
} else {
    puts [format "  Timing (WNS)      : FAIL (%s ns < %.1f ns)" $wns $target_wns_ns]
    set all_pass 0
}

puts "------------------------------------------------------------"
if {$all_pass} {
    puts "  OVERALL: ALL TARGETS MET"
} else {
    puts "  OVERALL: SOME TARGETS NOT MET -- review reports"
}
puts "============================================================"
puts ""
puts "Reports saved to: $report_dir"
puts "  utilization.rpt, utilization_hierarchical.rpt"
puts "  timing.rpt, timing_paths.rpt"
puts "  power.rpt"
puts "  drc.rpt"
puts "============================================================"
