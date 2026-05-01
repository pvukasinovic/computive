# ==============================================================================
# program.tcl -- JTAG Bitstream Programming for Kria KV260
# ==============================================================================
#
# Purpose:
#   Programs the KV260 FPGA via JTAG using Vivado Hardware Manager.
#   Supports both direct JTAG programming and XSA export for SD card boot.
#
# Usage:
#   vivado -mode batch -source fpga/program.tcl -tclargs <bitstream_path>
#
# Arguments:
#   bitstream_path  -- Path to the .bit file to program
#
# Prerequisites:
#   - KV260 board connected via USB-JTAG
#   - Vivado hw_server running (started automatically by open_hw_manager)
#
# Target: Xilinx Kria KV260 (xck26-sfvc784-2LV)
# ==============================================================================

# ------------------------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------------------------
if {[llength $argv] < 1} {
    puts "============================================================"
    puts "ERROR: No bitstream file specified."
    puts ""
    puts "Usage:"
    puts "  vivado -mode batch -source fpga/program.tcl -tclargs <bitstream.bit>"
    puts ""
    puts "Example:"
    puts "  vivado -mode batch -source fpga/program.tcl \\"
    puts "    -tclargs build/mlasic_ad/mlasic_ad.bit"
    puts "============================================================"
    exit 1
}

set bit_file [lindex $argv 0]

# Validate bitstream file exists
if {![file exists $bit_file]} {
    puts "ERROR: Bitstream file not found: $bit_file"
    exit 1
}

puts "============================================================"
puts "MLASIC FPGA Programming"
puts "  Bitstream: $bit_file"
puts "============================================================"

# ------------------------------------------------------------------------------
# Open Hardware Manager and connect
# ------------------------------------------------------------------------------
puts "Opening Hardware Manager..."
open_hw_manager

puts "Connecting to hardware server (localhost:3121)..."
connect_hw_server -allow_non_jtag

puts "Opening hardware target..."
open_hw_target

# Auto-detect the Zynq UltraScale+ device
set hw_device [get_hw_devices -filter {PART =~ xck26*}]

if {[llength $hw_device] == 0} {
    # Fallback: try to find any Zynq UltraScale+ device
    set hw_device [get_hw_devices -filter {PART =~ xczu*}]
}

if {[llength $hw_device] == 0} {
    puts "ERROR: No Zynq UltraScale+ device found on JTAG chain."
    puts "  Ensure the KV260 is powered on and connected via USB-JTAG."
    close_hw_target
    disconnect_hw_server
    close_hw_manager
    exit 1
}

set device [lindex $hw_device 0]
puts "  Found device: $device"

current_hw_device $device
refresh_hw_device -update_hw_probes false $device

# ------------------------------------------------------------------------------
# Program the device
# ------------------------------------------------------------------------------
puts "Programming device with: $bit_file"

set_property PROGRAM.FILE $bit_file $device

program_hw_devices $device

puts ""
puts "============================================================"
puts "Programming complete."
puts "  The FPGA is now configured. PL clocks and resets are active."
puts "  Run firmware to interact with the accelerator."
puts "============================================================"

# ------------------------------------------------------------------------------
# Clean up
# ------------------------------------------------------------------------------
close_hw_target
disconnect_hw_server
close_hw_manager

exit 0
