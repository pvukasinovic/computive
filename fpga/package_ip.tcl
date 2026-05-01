# ==============================================================================
# package_ip.tcl -- IP-XACT Packaging Script for MLASIC Accelerator
# ==============================================================================
#
# Purpose:
#   Packages the MLASIC accelerator RTL into a Vivado-compatible IP-XACT
#   component for use in block designs. Defines AXI-Lite slave, AXI-Stream
#   slave/master, clock, reset, and interrupt interfaces.
#
# Usage:
#   vivado -mode batch -source fpga/package_ip.tcl -tclargs <project_dir> <rtl_src_dir>
#
# Arguments:
#   project_dir  -- Path to Vivado project workspace (e.g., ./build)
#   rtl_src_dir  -- Path to directory containing all .sv RTL sources
#
# Output:
#   Packaged IP in fpga/ip_repo/accelerator_top_1.0/
#
# Target: Xilinx Zynq UltraScale+ (Kria KV260, xck26-sfvc784-2LV)
# ==============================================================================

# ------------------------------------------------------------------------------
# Parse command-line arguments
# ------------------------------------------------------------------------------
if {[llength $argv] < 2} {
    puts "ERROR: Usage: vivado -mode batch -source package_ip.tcl \\"
    puts "         -tclargs <project_dir> <rtl_src_dir>"
    puts ""
    puts "  project_dir  : Vivado project workspace directory"
    puts "  rtl_src_dir  : Directory containing .sv RTL source files"
    exit 1
}

set project_dir [lindex $argv 0]
set rtl_src_dir [lindex $argv 1]

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
set ip_vendor     "mlasic"
set ip_library    "user"
set ip_name       "accelerator_top"
set ip_version    "1.0"
set ip_display    "MLASIC Accelerator Top"
set ip_desc       "MLPerf Tiny AD Model Accelerator - INT8 inference engine with AXI-Lite CSR, AXI-Stream data, and interrupt"

set ip_repo_dir   "[file dirname [info script]]/ip_repo/${ip_name}_${ip_version}"
set part_name     "xck26-sfvc784-2LV-c"

# AXI-Lite address range: 10 registers x 4 bytes = 0x28, rounded up to 0x100
set axi_lite_addr_range "0x100"

puts "============================================================"
puts "MLASIC IP Packaging"
puts "============================================================"
puts "  Project dir : $project_dir"
puts "  RTL source  : $rtl_src_dir"
puts "  IP output   : $ip_repo_dir"
puts "  Part        : $part_name"
puts "============================================================"

# ------------------------------------------------------------------------------
# Create a temporary project for IP packaging
# ------------------------------------------------------------------------------
set pkg_project "${project_dir}/ip_pkg_temp"
file mkdir $pkg_project

create_project ip_pkg_proj $pkg_project -part $part_name -force

# ------------------------------------------------------------------------------
# Collect and add RTL source files
# ------------------------------------------------------------------------------
set sv_files [glob -nocomplain ${rtl_src_dir}/**/*.sv ${rtl_src_dir}/*.sv]
set svh_files [glob -nocomplain ${rtl_src_dir}/**/*.svh ${rtl_src_dir}/*.svh]
set mem_files [glob -nocomplain ${rtl_src_dir}/**/*.mem ${rtl_src_dir}/*.mem]

if {[llength $sv_files] == 0} {
    puts "ERROR: No .sv files found in $rtl_src_dir"
    close_project
    exit 1
}

puts "Found [llength $sv_files] SystemVerilog source file(s)"
puts "Found [llength $svh_files] SystemVerilog header file(s)"
puts "Found [llength $mem_files] memory initialization file(s)"

# Add all source files
foreach f $sv_files {
    add_files -norecurse $f
}
foreach f $svh_files {
    add_files -norecurse $f
    set_property IS_GLOBAL_INCLUDE true [get_files [file tail $f]]
}
foreach f $mem_files {
    add_files -norecurse $f
}

# Set the top module
set_property top accelerator_top [current_fileset]
update_compile_order -fileset sources_1

# ------------------------------------------------------------------------------
# Package the project as IP
# ------------------------------------------------------------------------------
ipx::package_project -root_dir $ip_repo_dir -vendor $ip_vendor \
    -library $ip_library -taxonomy /UserIP -import_files -set_current false

ipx::open_ipxact_file ${ip_repo_dir}/component.xml
set ip_core [ipx::current_core]

# ------------------------------------------------------------------------------
# Set IP identification
# ------------------------------------------------------------------------------
set_property vendor          $ip_vendor  $ip_core
set_property library         $ip_library $ip_core
set_property name            $ip_name    $ip_core
set_property version         $ip_version $ip_core
set_property display_name    $ip_display $ip_core
set_property description     $ip_desc    $ip_core
set_property vendor_display_name "MLASIC" $ip_core
set_property company_url     "https://mlasic.dev" $ip_core
set_property supported_families {zynquplus Production} $ip_core

# ------------------------------------------------------------------------------
# Define Clock Interface (clk)
# ------------------------------------------------------------------------------
set clk_intf [ipx::get_bus_interfaces clk -of_objects $ip_core]
if {$clk_intf eq ""} {
    set clk_intf [ipx::add_bus_interface clk $ip_core]
}
set_property abstraction_type_vlnv xilinx.com:signal:clock_rtl:1.0 $clk_intf
set_property bus_type_vlnv xilinx.com:signal:clock:1.0 $clk_intf
set_property interface_mode slave $clk_intf
set_property display_name "Clock" $clk_intf

# Map the physical port
ipx::add_port_map CLK $clk_intf
set_property physical_name clk \
    [ipx::get_port_maps CLK -of_objects $clk_intf]

# Associate clock with all AXI interfaces
ipx::add_bus_parameter ASSOCIATED_BUSIF $clk_intf
set_property value {s_axi:s_axis:m_axis} \
    [ipx::get_bus_parameters ASSOCIATED_BUSIF -of_objects $clk_intf]

# Associate reset with clock
ipx::add_bus_parameter ASSOCIATED_RESET $clk_intf
set_property value {rst_n} \
    [ipx::get_bus_parameters ASSOCIATED_RESET -of_objects $clk_intf]

# ------------------------------------------------------------------------------
# Define Reset Interface (rst_n)
# ------------------------------------------------------------------------------
set rst_intf [ipx::get_bus_interfaces rst_n -of_objects $ip_core]
if {$rst_intf eq ""} {
    set rst_intf [ipx::add_bus_interface rst_n $ip_core]
}
set_property abstraction_type_vlnv xilinx.com:signal:reset_rtl:1.0 $rst_intf
set_property bus_type_vlnv xilinx.com:signal:reset:1.0 $rst_intf
set_property interface_mode slave $rst_intf
set_property display_name "Reset (active-low)" $rst_intf

ipx::add_port_map RST $rst_intf
set_property physical_name rst_n \
    [ipx::get_port_maps RST -of_objects $rst_intf]

# Active-low polarity
ipx::add_bus_parameter POLARITY $rst_intf
set_property value ACTIVE_LOW \
    [ipx::get_bus_parameters POLARITY -of_objects $rst_intf]

# ------------------------------------------------------------------------------
# Define AXI4-Lite Slave Interface (s_axi)
# ------------------------------------------------------------------------------
set axi_lite [ipx::get_bus_interfaces s_axi -of_objects $ip_core]
if {$axi_lite eq ""} {
    set axi_lite [ipx::add_bus_interface s_axi $ip_core]
}
set_property abstraction_type_vlnv xilinx.com:interface:aximm_rtl:1.0 $axi_lite
set_property bus_type_vlnv xilinx.com:interface:aximm:1.0 $axi_lite
set_property interface_mode slave $axi_lite
set_property display_name "AXI-Lite Control/Status" $axi_lite

# Port mappings for AXI-Lite (matching axi_lite_ctrl.sv signals)
set axi_lite_ports {
    AWADDR  s_axi_awaddr
    AWVALID s_axi_awvalid
    AWREADY s_axi_awready
    WDATA   s_axi_wdata
    WSTRB   s_axi_wstrb
    WVALID  s_axi_wvalid
    WREADY  s_axi_wready
    BRESP   s_axi_bresp
    BVALID  s_axi_bvalid
    BREADY  s_axi_bready
    ARADDR  s_axi_araddr
    ARVALID s_axi_arvalid
    ARREADY s_axi_arready
    RDATA   s_axi_rdata
    RRESP   s_axi_rresp
    RVALID  s_axi_rvalid
    RREADY  s_axi_rready
}

foreach {logical physical} $axi_lite_ports {
    ipx::add_port_map $logical $axi_lite
    set_property physical_name $physical \
        [ipx::get_port_maps $logical -of_objects $axi_lite]
}

# Memory map for AXI-Lite: 0x100 address range (10 registers, 4 bytes each)
ipx::add_memory_map s_axi $ip_core
set_property slave_memory_map_ref s_axi $axi_lite

set addr_block [ipx::add_address_block reg0 \
    [ipx::get_memory_maps s_axi -of_objects $ip_core]]
set_property range $axi_lite_addr_range $addr_block
set_property width 32 $addr_block

# ------------------------------------------------------------------------------
# Define AXI4-Stream Slave Interface (s_axis -- input data from DMA)
# ------------------------------------------------------------------------------
set axis_slave [ipx::get_bus_interfaces s_axis -of_objects $ip_core]
if {$axis_slave eq ""} {
    set axis_slave [ipx::add_bus_interface s_axis $ip_core]
}
set_property abstraction_type_vlnv xilinx.com:interface:axis_rtl:1.0 $axis_slave
set_property bus_type_vlnv xilinx.com:interface:axis:1.0 $axis_slave
set_property interface_mode slave $axis_slave
set_property display_name "AXI-Stream Input (Slave)" $axis_slave

set axis_slave_ports {
    TDATA   s_axis_tdata
    TVALID  s_axis_tvalid
    TREADY  s_axis_tready
    TLAST   s_axis_tlast
    TKEEP   s_axis_tkeep
}

foreach {logical physical} $axis_slave_ports {
    ipx::add_port_map $logical $axis_slave
    set_property physical_name $physical \
        [ipx::get_port_maps $logical -of_objects $axis_slave]
}

# ------------------------------------------------------------------------------
# Define AXI4-Stream Master Interface (m_axis -- output data to DMA)
# ------------------------------------------------------------------------------
set axis_master [ipx::get_bus_interfaces m_axis -of_objects $ip_core]
if {$axis_master eq ""} {
    set axis_master [ipx::add_bus_interface m_axis $ip_core]
}
set_property abstraction_type_vlnv xilinx.com:interface:axis_rtl:1.0 $axis_master
set_property bus_type_vlnv xilinx.com:interface:axis:1.0 $axis_master
set_property interface_mode master $axis_master
set_property display_name "AXI-Stream Output (Master)" $axis_master

set axis_master_ports {
    TDATA   m_axis_tdata
    TVALID  m_axis_tvalid
    TREADY  m_axis_tready
    TLAST   m_axis_tlast
    TKEEP   m_axis_tkeep
}

foreach {logical physical} $axis_master_ports {
    ipx::add_port_map $logical $axis_master
    set_property physical_name $physical \
        [ipx::get_port_maps $logical -of_objects $axis_master]
}

# ------------------------------------------------------------------------------
# Define Interrupt Interface (irq)
# ------------------------------------------------------------------------------
set irq_intf [ipx::get_bus_interfaces irq -of_objects $ip_core]
if {$irq_intf eq ""} {
    set irq_intf [ipx::add_bus_interface irq $ip_core]
}
set_property abstraction_type_vlnv xilinx.com:signal:interrupt_rtl:1.0 $irq_intf
set_property bus_type_vlnv xilinx.com:signal:interrupt:1.0 $irq_intf
set_property interface_mode master $irq_intf
set_property display_name "Interrupt" $irq_intf

ipx::add_port_map INTERRUPT $irq_intf
set_property physical_name irq \
    [ipx::get_port_maps INTERRUPT -of_objects $irq_intf]

# Sensitivity: level-high (matches ORed IRQ_STATUS & IRQ_EN in axi_lite_ctrl.sv)
ipx::add_bus_parameter SENSITIVITY $irq_intf
set_property value LEVEL_HIGH \
    [ipx::get_bus_parameters SENSITIVITY -of_objects $irq_intf]

# ------------------------------------------------------------------------------
# Define file groups: synthesis and simulation
# ------------------------------------------------------------------------------
# Synthesis sources
set synth_fg [ipx::get_file_groups xilinx_verilogsynthesis -of_objects $ip_core]
if {$synth_fg eq ""} {
    set synth_fg [ipx::add_file_group xilinx_verilogsynthesis $ip_core]
}
set_property model_name accelerator_top $synth_fg

# Simulation sources
set sim_fg [ipx::get_file_groups xilinx_verilogbehavioralsimulation -of_objects $ip_core]
if {$sim_fg eq ""} {
    set sim_fg [ipx::add_file_group xilinx_verilogbehavioralsimulation $ip_core]
}
set_property model_name accelerator_top $sim_fg

# ------------------------------------------------------------------------------
# Validate and save the packaged IP
# ------------------------------------------------------------------------------
puts "Validating IP core..."
ipx::check_integrity $ip_core

puts "Saving IP-XACT component..."
ipx::save_core $ip_core

# ------------------------------------------------------------------------------
# Clean up temporary project
# ------------------------------------------------------------------------------
close_project

puts "============================================================"
puts "IP packaging complete."
puts "  IP location: $ip_repo_dir"
puts "  VLNV: ${ip_vendor}:${ip_library}:${ip_name}:${ip_version}"
puts ""
puts "To use in a block design, add this IP repository path:"
puts "  set_property ip_repo_paths $ip_repo_dir \[current_project\]"
puts "  update_ip_catalog"
puts "============================================================"
