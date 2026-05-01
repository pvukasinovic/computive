# ==============================================================================
# block_design.tcl -- Zynq UltraScale+ Block Design for MLASIC Accelerator
# ==============================================================================
#
# Purpose:
#   Creates a Vivado block design integrating the MLASIC accelerator IP with
#   the Zynq UltraScale+ Processing System on the Kria KV260 SOM. Configures
#   PS-PL interconnect, AXI DMA, interrupt routing, and address mapping.
#
# Usage:
#   Sourced by build.tcl. Can also be run standalone:
#     vivado -mode batch -source fpga/block_design.tcl -tclargs <project_dir> <ip_repo_dir>
#
# Prerequisites:
#   - Packaged accelerator IP in ip_repo_dir (from package_ip.tcl)
#   - Vivado 2023.2+ with Kria board files installed
#
# Block Design Checklist (from docs/firmware-integration-guide.md Section 12):
#   [x] Accelerator IP added to block design
#   [x] Accelerator AXI-Lite slave -> AXI Interconnect -> PS M_AXI_HPM0_LPD
#   [x] AXI DMA: MM2S + S2MM, 64-bit data, burst 16, SG disabled
#   [x] AXI-Stream: DMA M_AXIS_MM2S -> Accel S_AXIS, Accel M_AXIS -> DMA S_AXIS_S2MM
#   [x] DMA M_AXI_MM2S + M_AXI_S2MM -> AXI SmartConnect -> PS S_AXI_HP0_FPD
#   [x] Interrupt: xlconcat merging accel.irq + DMA interrupts -> pl_ps_irq0
#   [x] proc_sys_reset connected to all reset ports
#   [x] FCLK_CLK0 = 100 MHz, connected to all PL IP clocks
#   [x] Address editor: Accel CSR @ 0xA000_0000, DMA @ 0xA001_0000
#   [x] HP port address range covers DDR buffers
#
# Target: Xilinx Kria KV260 (xck26-sfvc784-2LV)
# ==============================================================================

# ------------------------------------------------------------------------------
# Parse arguments (when run standalone; build.tcl passes these via variables)
# ------------------------------------------------------------------------------
if {[info exists argv] && [llength $argv] >= 2} {
    set project_dir  [lindex $argv 0]
    set ip_repo_dir  [lindex $argv 1]
}

# Validate required variables
if {![info exists project_dir] || ![info exists ip_repo_dir]} {
    puts "ERROR: block_design.tcl requires 'project_dir' and 'ip_repo_dir' variables."
    puts "  Either source from build.tcl or pass as: -tclargs <project_dir> <ip_repo_dir>"
    exit 1
}

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
set bd_name         "mlasic_system"
set board_part      "xilinx.com:kv260_som:part0:1.4"
set pl_clk_freq_hz  100000000
set pl_clk_freq_mhz 100

# Address map (must match firmware-integration-guide.md Section 2)
set accel_base_addr  0xA0000000
set accel_addr_range 4K
set dma_base_addr    0xA0010000
set dma_addr_range   4K

puts "============================================================"
puts "MLASIC Block Design Creation"
puts "============================================================"
puts "  Block design  : $bd_name"
puts "  Board part    : $board_part"
puts "  PL clock      : $pl_clk_freq_mhz MHz"
puts "  Accel base    : [format 0x%08X $accel_base_addr]"
puts "  DMA base      : [format 0x%08X $dma_base_addr]"
puts "============================================================"

# ------------------------------------------------------------------------------
# Add IP repository and update catalog
# ------------------------------------------------------------------------------
set_property ip_repo_paths [list $ip_repo_dir] [current_project]
update_ip_catalog -rebuild

# ------------------------------------------------------------------------------
# Create block design
# ------------------------------------------------------------------------------
if {[llength [get_bd_designs -quiet $bd_name]] > 0} {
    puts "WARNING: Block design '$bd_name' already exists. Removing."
    delete_bd_objs [get_bd_designs $bd_name]
}

create_bd_design $bd_name

# ------------------------------------------------------------------------------
# 1. Add Zynq UltraScale+ PS IP with board preset
# ------------------------------------------------------------------------------
puts "Adding Zynq UltraScale+ Processing System..."

set ps_cell [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_ps]

# Apply board automation to configure PS with KV260 defaults
# This sets DDR, MIO, clocks, and peripheral configurations
apply_board_connection -board_interface "som240_1_connector" \
    -ip_intf "$ps_cell/fixed_io" -diagram $bd_name

# Configure PS clocks and interfaces
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP1          {1}                     \
    CONFIG.PSU__MAXIGP1__DATA_WIDTH     {32}                    \
    CONFIG.PSU__USE__S_AXI_GP2          {1}                     \
    CONFIG.PSU__SAXIGP2__DATA_WIDTH     {128}                   \
    CONFIG.PSU__USE__IRQ0               {1}                     \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $pl_clk_freq_mhz \
    CONFIG.PSU__USE__FABRIC__RST        {1}                     \
    CONFIG.PSU__NUM_FABRIC_RESETS       {1}                     \
] $ps_cell

# Expose clock and reset outputs
# pl_clk0 is the 100 MHz PL fabric clock
# pl_resetn0 is the active-low PL reset

puts "  PS configured: HPM0_LPD (32-bit), HP0_FPD (128-bit), IRQ0, pl_clk0=${pl_clk_freq_mhz}MHz"

# ------------------------------------------------------------------------------
# 2. Add Processor System Reset
# ------------------------------------------------------------------------------
puts "Adding Processor System Reset..."

set ps_reset [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0]

# Connect reset inputs
connect_bd_net [get_bd_pins $ps_cell/pl_clk0]     [get_bd_pins $ps_reset/slowest_sync_clk]
connect_bd_net [get_bd_pins $ps_cell/pl_resetn0]   [get_bd_pins $ps_reset/ext_reset_in]

# The proc_sys_reset produces:
#   peripheral_aresetn  -- active-low reset for AXI peripherals
#   interconnect_aresetn -- active-low reset for AXI interconnects

# ------------------------------------------------------------------------------
# 3. Add MLASIC Accelerator IP
# ------------------------------------------------------------------------------
puts "Adding MLASIC Accelerator IP..."

set accel_cell [create_bd_cell -type ip -vlnv mlasic:user:accelerator_top:1.0 accelerator_0]

# Connect clock and reset to accelerator
connect_bd_net [get_bd_pins $ps_cell/pl_clk0]                     [get_bd_pins $accel_cell/clk]
connect_bd_net [get_bd_pins $ps_reset/peripheral_aresetn]          [get_bd_pins $accel_cell/rst_n]

puts "  Accelerator IP instantiated and clocked at ${pl_clk_freq_mhz} MHz"

# ------------------------------------------------------------------------------
# 4. Add AXI DMA Controller
# ------------------------------------------------------------------------------
puts "Adding AXI DMA Controller..."

set dma_cell [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0]

# Configure DMA:
#   - Scatter-Gather disabled (simple DMA mode for v0.1)
#   - MM2S (Memory-to-Stream) for sending input data to accelerator
#   - S2MM (Stream-to-Memory) for receiving output data from accelerator
#   - 64-bit stream width (8 bytes per beat, matching accelerator AXI-Stream)
#   - 64-bit memory-map width
#   - Burst length 16 (16 x 8 bytes = 128 bytes per burst)
#   - Address width 32 bits (sufficient for Zynq DDR range)
set_property -dict [list \
    CONFIG.c_include_sg              {0}     \
    CONFIG.c_sg_include_stscntrl_strm {0}    \
    CONFIG.c_include_mm2s            {1}     \
    CONFIG.c_include_s2mm            {1}     \
    CONFIG.c_m_axi_mm2s_data_width   {64}   \
    CONFIG.c_m_axis_mm2s_tdata_width {64}    \
    CONFIG.c_m_axi_s2mm_data_width   {64}   \
    CONFIG.c_s_axis_s2mm_tdata_width {64}    \
    CONFIG.c_mm2s_burst_size         {16}    \
    CONFIG.c_s2mm_burst_size         {16}    \
    CONFIG.c_addr_width              {32}    \
    CONFIG.c_include_mm2s_dre        {1}     \
    CONFIG.c_include_s2mm_dre        {1}     \
] $dma_cell

# Connect DMA clock and reset
connect_bd_net [get_bd_pins $ps_cell/pl_clk0]             [get_bd_pins $dma_cell/s_axi_lite_aclk]
connect_bd_net [get_bd_pins $ps_cell/pl_clk0]             [get_bd_pins $dma_cell/m_axi_mm2s_aclk]
connect_bd_net [get_bd_pins $ps_cell/pl_clk0]             [get_bd_pins $dma_cell/m_axi_s2mm_aclk]
connect_bd_net [get_bd_pins $ps_reset/peripheral_aresetn]  [get_bd_pins $dma_cell/axi_resetn]

puts "  DMA configured: SG=off, MM2S+S2MM, 64-bit stream, burst=16"

# ------------------------------------------------------------------------------
# 5. AXI-Stream Connections (DMA <-> Accelerator)
# ------------------------------------------------------------------------------
puts "Connecting AXI-Stream data paths..."

# DMA MM2S output -> Accelerator input (slave)
# DMA produces data from DDR, accelerator consumes it
connect_bd_intf_net [get_bd_intf_pins $dma_cell/M_AXIS_MM2S] \
                    [get_bd_intf_pins $accel_cell/s_axis]

# Accelerator output (master) -> DMA S2MM input
# Accelerator produces inference result, DMA writes to DDR
connect_bd_intf_net [get_bd_intf_pins $accel_cell/m_axis] \
                    [get_bd_intf_pins $dma_cell/S_AXIS_S2MM]

puts "  Stream paths: DMA.MM2S -> Accel.s_axis, Accel.m_axis -> DMA.S2MM"

# ------------------------------------------------------------------------------
# 6. AXI SmartConnect: PS HPM0_LPD -> {Accelerator AXI-Lite, DMA CSR}
# ------------------------------------------------------------------------------
puts "Adding AXI SmartConnect for control path..."

set ctrl_ic [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_ctrl]

# 1 master (PS HPM0_LPD) -> 2 slaves (Accel CSR, DMA CSR)
set_property -dict [list \
    CONFIG.NUM_SI {1}  \
    CONFIG.NUM_MI {2}  \
] $ctrl_ic

# Connect PS HPM0_LPD master to SmartConnect input
connect_bd_intf_net [get_bd_intf_pins $ps_cell/M_AXI_HPM0_LPD] \
                    [get_bd_intf_pins $ctrl_ic/S00_AXI]

# SmartConnect output 0 -> Accelerator AXI-Lite
connect_bd_intf_net [get_bd_intf_pins $ctrl_ic/M00_AXI] \
                    [get_bd_intf_pins $accel_cell/s_axi]

# SmartConnect output 1 -> DMA AXI-Lite CSR
connect_bd_intf_net [get_bd_intf_pins $ctrl_ic/M01_AXI] \
                    [get_bd_intf_pins $dma_cell/S_AXI_LITE]

# Connect SmartConnect clock and reset
connect_bd_net [get_bd_pins $ps_cell/pl_clk0]             [get_bd_pins $ctrl_ic/aclk]
connect_bd_net [get_bd_pins $ps_reset/interconnect_aresetn] [get_bd_pins $ctrl_ic/aresetn]

puts "  Control SmartConnect: PS.HPM0_LPD -> {Accel.s_axi, DMA.CSR}"

# ------------------------------------------------------------------------------
# 7. AXI SmartConnect: DMA Memory-Map -> PS HP0_FPD (DDR access)
# ------------------------------------------------------------------------------
puts "Adding AXI SmartConnect for data path..."

set data_ic [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_data]

# 2 masters (DMA MM2S, DMA S2MM) -> 1 slave (PS HP0_FPD)
set_property -dict [list \
    CONFIG.NUM_SI {2}  \
    CONFIG.NUM_MI {1}  \
] $data_ic

# DMA MM2S memory-map master -> SmartConnect input 0
connect_bd_intf_net [get_bd_intf_pins $dma_cell/M_AXI_MM2S] \
                    [get_bd_intf_pins $data_ic/S00_AXI]

# DMA S2MM memory-map master -> SmartConnect input 1
connect_bd_intf_net [get_bd_intf_pins $dma_cell/M_AXI_S2MM] \
                    [get_bd_intf_pins $data_ic/S01_AXI]

# SmartConnect output -> PS S_AXI_HP0_FPD (high-performance DDR port)
connect_bd_intf_net [get_bd_intf_pins $data_ic/M00_AXI] \
                    [get_bd_intf_pins $ps_cell/S_AXI_HP0_FPD]

# Connect data SmartConnect clock and reset
connect_bd_net [get_bd_pins $ps_cell/pl_clk0]             [get_bd_pins $data_ic/aclk]
connect_bd_net [get_bd_pins $ps_reset/interconnect_aresetn] [get_bd_pins $data_ic/aresetn]

# Connect HP0 clock
connect_bd_net [get_bd_pins $ps_cell/pl_clk0] [get_bd_pins $ps_cell/saxihp0_fpd_aclk]

puts "  Data SmartConnect: {DMA.MM2S, DMA.S2MM} -> PS.HP0_FPD"

# ------------------------------------------------------------------------------
# 8. Interrupt Routing: xlconcat -> PS pl_ps_irq0
# ------------------------------------------------------------------------------
puts "Configuring interrupt routing..."

set irq_concat [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 irq_concat]

# 3 interrupt sources: accelerator done/error, DMA MM2S complete, DMA S2MM complete
set_property -dict [list \
    CONFIG.NUM_PORTS {3} \
] $irq_concat

# In[0] = Accelerator IRQ (inference done or error, from axi_lite_ctrl.sv line 160)
connect_bd_net [get_bd_pins $accel_cell/irq]            [get_bd_pins $irq_concat/In0]

# In[1] = DMA MM2S interrupt (transfer complete)
connect_bd_net [get_bd_pins $dma_cell/mm2s_introut]     [get_bd_pins $irq_concat/In1]

# In[2] = DMA S2MM interrupt (transfer complete)
connect_bd_net [get_bd_pins $dma_cell/s2mm_introut]     [get_bd_pins $irq_concat/In2]

# Concatenated output -> PS PL-to-PS interrupt
connect_bd_net [get_bd_pins $irq_concat/dout]           [get_bd_pins $ps_cell/pl_ps_irq0]

puts "  Interrupts: {accel.irq, dma.mm2s, dma.s2mm} -> xlconcat -> ps.pl_ps_irq0"

# ------------------------------------------------------------------------------
# 9. Address Editor: Assign addresses to all peripherals
# ------------------------------------------------------------------------------
puts "Configuring address map..."

# PS HPM0_LPD address space -> Accelerator CSR
assign_bd_address -target_address_space [get_bd_addr_spaces $ps_cell/Data] \
    [get_bd_addr_segs $accel_cell/s_axi/reg0] \
    -range $accel_addr_range -offset $accel_base_addr
puts "  Accelerator CSR : [format 0x%08X $accel_base_addr] ($accel_addr_range)"

# PS HPM0_LPD address space -> DMA CSR
assign_bd_address -target_address_space [get_bd_addr_spaces $ps_cell/Data] \
    [get_bd_addr_segs $dma_cell/S_AXI_LITE/Reg] \
    -range $dma_addr_range -offset $dma_base_addr
puts "  DMA CSR         : [format 0x%08X $dma_base_addr] ($dma_addr_range)"

# DMA address spaces -> PS HP0_FPD DDR
# The DMA needs access to the full DDR range for input/output buffers.
# Map the entire low DDR range (0x0000_0000 - 0x7FFF_FFFF = 2 GB).
assign_bd_address -target_address_space [get_bd_addr_spaces $dma_cell/Data_MM2S] \
    [get_bd_addr_segs $ps_cell/SAXIGP2/HP0_DDR_LOW] \
    -range 2G -offset 0x00000000
assign_bd_address -target_address_space [get_bd_addr_spaces $dma_cell/Data_S2MM] \
    [get_bd_addr_segs $ps_cell/SAXIGP2/HP0_DDR_LOW] \
    -range 2G -offset 0x00000000

puts "  DMA DDR access  : 0x00000000 - 0x7FFFFFFF (2 GB, HP0_DDR_LOW)"

# ------------------------------------------------------------------------------
# 10. Validate and save block design
# ------------------------------------------------------------------------------
puts "Validating block design..."
validate_bd_design

puts "Regenerating layout..."
regenerate_bd_layout

puts "Saving block design..."
save_bd_design

# ------------------------------------------------------------------------------
# 11. Generate HDL wrapper
# ------------------------------------------------------------------------------
puts "Generating HDL wrapper..."

set bd_file [get_files ${bd_name}.bd]
set wrapper_file [make_wrapper -files $bd_file -top]
add_files -norecurse $wrapper_file
set_property top [file rootname [file tail $wrapper_file]] [current_fileset]
update_compile_order -fileset sources_1

puts "  Top module set to: [file rootname [file tail $wrapper_file]]"

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
puts ""
puts "============================================================"
puts "Block Design Complete: $bd_name"
puts "============================================================"
puts ""
puts "Address Map:"
puts "  Accelerator CSR : [format 0x%08X $accel_base_addr]"
puts "  AXI DMA CSR     : [format 0x%08X $dma_base_addr]"
puts "  DDR (DMA access): 0x00000000 - 0x7FFFFFFF"
puts ""
puts "Interrupt Map:"
puts "  IRQ_F2P[0] = accel.irq"
puts "  IRQ_F2P[1] = dma.mm2s_introut"
puts "  IRQ_F2P[2] = dma.s2mm_introut"
puts ""
puts "Data Path:"
puts "  PS DDR -> DMA MM2S -> Accel s_axis (input)"
puts "  Accel m_axis (output) -> DMA S2MM -> PS DDR"
puts ""
puts "Control Path:"
puts "  PS HPM0_LPD -> SmartConnect -> {Accel s_axi, DMA CSR}"
puts ""
puts "NOTE: The firmware must use these base addresses:"
puts "  #define ACCEL_BASE_ADDR  [format 0x%08X $accel_base_addr]"
puts "  #define DMA_BASE_ADDR    [format 0x%08X $dma_base_addr]"
puts "============================================================"
