# ==============================================================================
# build.tcl -- Top-Level FPGA Build Orchestrator for MLASIC Accelerator
# ==============================================================================
#
# Purpose:
#   Runs the full FPGA build flow in Vivado batch mode:
#     1. Package accelerator RTL as IP-XACT (package_ip.tcl)
#     2. Create Zynq UltraScale+ block design (block_design.tcl)
#     3. Run synthesis (synth_design)
#     4. Run implementation (opt_design, place_design, route_design)
#     5. Generate bitstream (write_bitstream)
#     6. Generate reports (report.tcl)
#     7. Export XSA for Vitis/firmware development
#
# Usage:
#   vivado -mode batch -source fpga/build.tcl -tclargs <project_name> <rtl_dir> [part]
#
# Arguments:
#   project_name  -- Name for the Vivado project (e.g., "mlasic_ad")
#   rtl_dir       -- Path to compiler-generated RTL output directory
#   part          -- (Optional) Target FPGA part. Default: xck26-sfvc784-2LV-c
#
# Output:
#   build/<project_name>/
#     *.bit           -- Bitstream for JTAG programming
#     *.xsa           -- Hardware specification archive for Vitis
#     reports/        -- Utilization, timing, power, DRC reports
#
# Target: Xilinx Kria KV260 (xck26-sfvc784-2LV, Zynq UltraScale+)
# ==============================================================================

# ------------------------------------------------------------------------------
# Capture start time
# ------------------------------------------------------------------------------
set build_start_time [clock seconds]

# ------------------------------------------------------------------------------
# Parse command-line arguments
# ------------------------------------------------------------------------------
if {[llength $argv] < 2} {
    puts "============================================================"
    puts "ERROR: Insufficient arguments."
    puts ""
    puts "Usage:"
    puts "  vivado -mode batch -source fpga/build.tcl \\"
    puts "    -tclargs <project_name> <rtl_dir> \[part\]"
    puts ""
    puts "Arguments:"
    puts "  project_name  : Name for Vivado project (e.g., mlasic_ad)"
    puts "  rtl_dir       : Path to RTL sources from compiler"
    puts "  part          : (Optional) FPGA part (default: xck26-sfvc784-2LV-c)"
    puts ""
    puts "Example:"
    puts "  vivado -mode batch -source fpga/build.tcl \\"
    puts "    -tclargs mlasic_ad output/ad_model/rtl"
    puts "============================================================"
    exit 1
}

set project_name [lindex $argv 0]
set rtl_dir      [lindex $argv 1]
set part_name    [expr {[llength $argv] >= 3 ? [lindex $argv 2] : "xck26-sfvc784-2LV-c"}]

# Derived paths
set script_dir   [file dirname [info script]]
set build_dir    "${script_dir}/../build/${project_name}"
set project_dir  "${build_dir}/vivado_project"
set report_dir   "${build_dir}/reports"
set ip_repo_dir  "${script_dir}/ip_repo/accelerator_top_1.0"

# Board part for KV260 SOM
set board_part   "xilinx.com:kv260_som:part0:1.4"

# Clock period (ns) -- 100 MHz
set clk_period   10.000

# Constraints file
set xdc_file     "${script_dir}/constraints.xdc"

puts "============================================================"
puts "MLASIC FPGA Build"
puts "============================================================"
puts "  Project     : $project_name"
puts "  RTL dir     : $rtl_dir"
puts "  Part        : $part_name"
puts "  Board       : $board_part"
puts "  Clock       : [expr {1000.0 / $clk_period}] MHz ($clk_period ns)"
puts "  Build dir   : $build_dir"
puts "  Script dir  : $script_dir"
puts "============================================================"

# Validate RTL directory exists
if {![file isdirectory $rtl_dir]} {
    puts "ERROR: RTL directory does not exist: $rtl_dir"
    exit 1
}

# Create build directories
file mkdir $build_dir
file mkdir $report_dir

# ==============================================================================
# STEP 1: Package Accelerator IP
# ==============================================================================
puts "\n"
puts "============================================================"
puts "STEP 1: Packaging Accelerator IP"
puts "============================================================"

# Source the IP packaging script
# It needs project_dir and rtl_src_dir as argv-like variables
set ::argv [list $project_dir $rtl_dir]
source ${script_dir}/package_ip.tcl

puts "IP packaging complete."

# ==============================================================================
# STEP 2: Create Vivado Project
# ==============================================================================
puts "\n"
puts "============================================================"
puts "STEP 2: Creating Vivado Project"
puts "============================================================"

create_project $project_name $project_dir -part $part_name -force
set_property board_part $board_part [current_project]

# Add constraints
if {[file exists $xdc_file]} {
    add_files -fileset constrs_1 -norecurse $xdc_file
    puts "  Added constraints: $xdc_file"
} else {
    puts "WARNING: Constraints file not found: $xdc_file"
}

# ==============================================================================
# STEP 3: Create Block Design
# ==============================================================================
puts "\n"
puts "============================================================"
puts "STEP 3: Creating Block Design"
puts "============================================================"

# Set variables for block_design.tcl (it reads these directly)
set project_dir $project_dir
set ip_repo_dir $ip_repo_dir

source ${script_dir}/block_design.tcl

puts "Block design complete."

# ==============================================================================
# STEP 4: Synthesis
# ==============================================================================
puts "\n"
puts "============================================================"
puts "STEP 4: Running Synthesis"
puts "============================================================"

set synth_start [clock seconds]

# Configure synthesis settings
set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY rebuilt [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING on [get_runs synth_1]

# Force BRAM inference for memory blocks
set_property STEPS.SYNTH_DESIGN.ARGS.MORE_OPTIONS {-max_bram 200} [get_runs synth_1]

# Launch synthesis
launch_runs synth_1 -jobs [exec nproc]
wait_on_run synth_1

# Check synthesis status
set synth_status [get_property STATUS [get_runs synth_1]]
if {$synth_status ne "synth_design Complete!"} {
    puts "ERROR: Synthesis failed with status: $synth_status"
    exit 1
}

set synth_elapsed [expr {[clock seconds] - $synth_start}]
puts "  Synthesis completed in [format %d:%02d [expr {$synth_elapsed/60}] [expr {$synth_elapsed%60}]]"

# Open synthesized design for post-synth utilization report
open_run synth_1
report_utilization -file ${report_dir}/utilization_post_synth.rpt
puts "  Post-synthesis utilization saved to: ${report_dir}/utilization_post_synth.rpt"
close_design

# ==============================================================================
# STEP 5: Implementation (opt, place, route)
# ==============================================================================
puts "\n"
puts "============================================================"
puts "STEP 5: Running Implementation"
puts "============================================================"

set impl_start [clock seconds]

# Configure implementation settings for timing closure
set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]

# Launch implementation
launch_runs impl_1 -jobs [exec nproc]
wait_on_run impl_1

# Check implementation status
set impl_status [get_property STATUS [get_runs impl_1]]
if {$impl_status ne "route_design Complete!"} {
    puts "ERROR: Implementation failed with status: $impl_status"
    exit 1
}

set impl_elapsed [expr {[clock seconds] - $impl_start}]
puts "  Implementation completed in [format %d:%02d [expr {$impl_elapsed/60}] [expr {$impl_elapsed%60}]]"

# ==============================================================================
# STEP 6: Generate Bitstream
# ==============================================================================
puts "\n"
puts "============================================================"
puts "STEP 6: Generating Bitstream"
puts "============================================================"

set bit_start [clock seconds]

launch_runs impl_1 -to_step write_bitstream -jobs [exec nproc]
wait_on_run impl_1

set bit_elapsed [expr {[clock seconds] - $bit_start}]
puts "  Bitstream generated in [format %d:%02d [expr {$bit_elapsed/60}] [expr {$bit_elapsed%60}]]"

# Copy bitstream to build output directory
set bit_file [glob -nocomplain ${project_dir}/${project_name}.runs/impl_1/*.bit]
if {[llength $bit_file] > 0} {
    file copy -force [lindex $bit_file 0] ${build_dir}/${project_name}.bit
    puts "  Bitstream: ${build_dir}/${project_name}.bit"
} else {
    puts "WARNING: Bitstream file not found in impl_1 directory"
}

# ==============================================================================
# STEP 7: Generate Reports
# ==============================================================================
puts "\n"
puts "============================================================"
puts "STEP 7: Generating Post-Implementation Reports"
puts "============================================================"

# Open implemented design for reporting
open_run impl_1

# Source the report generation script
set report_dir $report_dir
source ${script_dir}/report.tcl

close_design

# ==============================================================================
# STEP 8: Export XSA for Vitis / Firmware Development
# ==============================================================================
puts "\n"
puts "============================================================"
puts "STEP 8: Exporting Hardware Specification Archive (XSA)"
puts "============================================================"

set xsa_file "${build_dir}/${project_name}.xsa"

write_hw_platform -fixed -include_bit -force -file $xsa_file

puts "  XSA exported: $xsa_file"
puts "  Use this XSA in Vitis to create the firmware application."

# ==============================================================================
# Build Summary
# ==============================================================================
set build_elapsed [expr {[clock seconds] - $build_start_time}]

puts ""
puts "============================================================"
puts "BUILD COMPLETE"
puts "============================================================"
puts "  Total build time: [format %d:%02d [expr {$build_elapsed/60}] [expr {$build_elapsed%60}]]"
puts ""
puts "  Outputs:"
puts "    Bitstream : ${build_dir}/${project_name}.bit"
puts "    XSA       : ${build_dir}/${project_name}.xsa"
puts "    Reports   : ${report_dir}/"
puts ""
puts "  Next steps:"
puts "    1. Program the board:    vivado -mode batch -source fpga/program.tcl -tclargs ${build_dir}/${project_name}.bit"
puts "    2. Build firmware:       Use ${xsa_file} in Vitis to create bare-metal app"
puts "    3. Review timing:        Open ${report_dir}/timing.rpt"
puts "    4. Review utilization:   Open ${report_dir}/utilization.rpt"
puts "============================================================"

close_project
exit 0
