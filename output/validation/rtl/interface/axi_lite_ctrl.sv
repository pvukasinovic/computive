// axi_lite_ctrl.sv — AXI-Lite CSR register interface
// From rtl-interface-spec.md §3
// 10 registers: CTRL, STATUS, IRQ_EN, IRQ_STATUS, CYCLE_COUNT,
//               INF_COUNT, VERSION, SCRATCH, ERROR_CODE, LAYER_STATUS
// Self-clearing bits in CTRL, W1C for IRQ_STATUS

module axi_lite_ctrl #(
    parameter int ADDR_W   = 8,
    parameter int DATA_W   = 32,
    parameter int VERSION  = 32'h0001_0000  // v1.0.0
) (
    input clk,
    input rst_n,

    // AXI-Lite slave interface
    input [ADDR_W-1:0] s_axi_awaddr,
    input s_axi_awvalid,
    output logic s_axi_awready,
    input [DATA_W-1:0] s_axi_wdata,
    /* verilator lint_off UNUSEDSIGNAL */
    input [3:0] s_axi_wstrb,  // AXI spec required, not used (full-word writes only)
    /* verilator lint_on UNUSEDSIGNAL */
    input s_axi_wvalid,
    output logic s_axi_wready,
    output logic [1:0] s_axi_bresp,
    output logic s_axi_bvalid,
    input s_axi_bready,
    input [ADDR_W-1:0] s_axi_araddr,
    input s_axi_arvalid,
    output logic s_axi_arready,
    output logic [DATA_W-1:0] s_axi_rdata,
    output logic [1:0] s_axi_rresp,
    output logic s_axi_rvalid,
    input s_axi_rready,

    // Control outputs
    output logic ctrl_start,
    output logic ctrl_soft_rst,
    output logic ctrl_continuous,
    output logic irq_en,

    // Status inputs
    input status_idle,
    input status_busy,
    input status_done,
    input status_error,
    input [DATA_W-1:0] cycle_count,
    input [DATA_W-1:0] inf_count,
    input [DATA_W-1:0] error_code,
    input [1:0] layer_status,

    // Interrupt
    input irq_done,
    input irq_error,
    output logic irq
);

// Register addresses
localparam logic [ADDR_W-1:0] ADDR_CTRL         = 8'h00;
localparam logic [ADDR_W-1:0] ADDR_STATUS       = 8'h04;
localparam logic [ADDR_W-1:0] ADDR_IRQ_EN       = 8'h08;
localparam logic [ADDR_W-1:0] ADDR_IRQ_STATUS   = 8'h0C;
localparam logic [ADDR_W-1:0] ADDR_CYCLE_COUNT  = 8'h10;
localparam logic [ADDR_W-1:0] ADDR_INF_COUNT    = 8'h14;
localparam logic [ADDR_W-1:0] ADDR_VERSION      = 8'h18;
localparam logic [ADDR_W-1:0] ADDR_SCRATCH      = 8'h1C;
localparam logic [ADDR_W-1:0] ADDR_ERROR_CODE   = 8'h20;
localparam logic [ADDR_W-1:0] ADDR_LAYER_STATUS = 8'h24;

// Internal registers
logic [DATA_W-1:0] reg_ctrl;
logic [DATA_W-1:0] reg_irq_en;
logic [DATA_W-1:0] reg_irq_status;
logic [DATA_W-1:0] reg_scratch;

// Write channel handshake
logic aw_ready, w_ready;
logic [ADDR_W-1:0] wr_addr;
logic wr_valid;

assign s_axi_awready = aw_ready;
assign s_axi_wready  = w_ready;
assign s_axi_bresp   = 2'b00;  // OKAY

// Read channel
assign s_axi_arready = !s_axi_rvalid || s_axi_rready;
assign s_axi_rresp   = 2'b00;  // OKAY

// Write channel FSM
always_ff @(posedge clk) begin
    if (!rst_n) begin
        aw_ready       <= 1'b1;
        w_ready        <= 1'b1;
        s_axi_bvalid   <= 1'b0;
        wr_valid       <= 1'b0;
        wr_addr        <= {ADDR_W{1'b0}};
    end else begin
        // Accept write address
        if (s_axi_awvalid && aw_ready) begin
            wr_addr  <= s_axi_awaddr;
            aw_ready <= 1'b0;
        end

        // Accept write data
        if (s_axi_wvalid && w_ready) begin
            w_ready  <= 1'b0;
            wr_valid <= 1'b1;
        end

        // Write response
        if (wr_valid && !aw_ready && !w_ready) begin
            s_axi_bvalid <= 1'b1;
            wr_valid     <= 1'b0;
        end

        if (s_axi_bvalid && s_axi_bready) begin
            s_axi_bvalid <= 1'b0;
            aw_ready     <= 1'b1;
            w_ready      <= 1'b1;
        end
    end
end

// Register write logic
always_ff @(posedge clk) begin
    if (!rst_n) begin
        reg_ctrl       <= {DATA_W{1'b0}};
        reg_irq_en     <= {DATA_W{1'b0}};
        reg_irq_status <= {DATA_W{1'b0}};
        reg_scratch    <= {DATA_W{1'b0}};
    end else begin
        // Self-clearing START and SOFT_RST bits
        reg_ctrl[0] <= 1'b0;  // START auto-clears
        reg_ctrl[1] <= 1'b0;  // SOFT_RST auto-clears

        // Set IRQ status bits on events
        if (irq_done)  reg_irq_status[0] <= 1'b1;
        if (irq_error) reg_irq_status[1] <= 1'b1;

        // Register writes
        if (wr_valid && !aw_ready && !w_ready) begin
            case (wr_addr)
                ADDR_CTRL:       reg_ctrl <= s_axi_wdata;
                ADDR_IRQ_EN:     reg_irq_en <= s_axi_wdata;
                ADDR_IRQ_STATUS: reg_irq_status <= reg_irq_status & ~s_axi_wdata;  // W1C
                ADDR_SCRATCH:    reg_scratch <= s_axi_wdata;
                default: ;  // Read-only registers ignored
            endcase
        end
    end
end

// Control outputs
assign ctrl_start      = reg_ctrl[0];
assign ctrl_soft_rst   = reg_ctrl[1];
assign ctrl_continuous = reg_ctrl[2];
assign irq_en          = reg_irq_en[0];

// Interrupt output
assign irq = |(reg_irq_status & reg_irq_en);

// Read channel
always_ff @(posedge clk) begin
    if (!rst_n) begin
        s_axi_rvalid <= 1'b0;
        s_axi_rdata  <= {DATA_W{1'b0}};
    end else begin
        if (s_axi_arvalid && s_axi_arready) begin
            s_axi_rvalid <= 1'b1;
            case (s_axi_araddr)
                ADDR_CTRL:         s_axi_rdata <= reg_ctrl;
                ADDR_STATUS:       s_axi_rdata <= {28'b0, status_error, status_done, status_busy, status_idle};
                ADDR_IRQ_EN:       s_axi_rdata <= reg_irq_en;
                ADDR_IRQ_STATUS:   s_axi_rdata <= reg_irq_status;
                ADDR_CYCLE_COUNT:  s_axi_rdata <= cycle_count;
                ADDR_INF_COUNT:    s_axi_rdata <= inf_count;
                ADDR_VERSION:      s_axi_rdata <= VERSION;
                ADDR_SCRATCH:      s_axi_rdata <= reg_scratch;
                ADDR_ERROR_CODE:   s_axi_rdata <= error_code;
                ADDR_LAYER_STATUS: s_axi_rdata <= {30'b0, layer_status};
                default:           s_axi_rdata <= {DATA_W{1'b0}};
            endcase
        end else if (s_axi_rvalid && s_axi_rready) begin
            s_axi_rvalid <= 1'b0;
        end
    end
end

endmodule
