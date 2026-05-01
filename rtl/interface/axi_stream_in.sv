// axi_stream_in.sv — AXI-Stream slave for input activation reception
// From rtl-interface-spec.md §4
// Receives 80 beats x 8 bytes = 640 bytes into activation bank A
// TVALID is NOT dependent on TREADY (AXI spec compliant)

/* verilator lint_off VARHIDDEN */
module axi_stream_in #(
    parameter int AXI_DATA_W = 64,
    parameter int NUM_BEATS  = 80,
    parameter int ACT_DEPTH  = 80
) (
    input  logic                             clk,
    input  logic                             rst_n,

    // Control
    input  logic                             enable,
    output logic                             done,

    // AXI-Stream slave interface
    input  logic [AXI_DATA_W-1:0]           s_axis_tdata,
    input  logic                             s_axis_tvalid,
    output logic                             s_axis_tready,
    input  logic                             s_axis_tlast,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [AXI_DATA_W/8-1:0]         s_axis_tkeep,  // AXI spec required, all bytes valid
    /* verilator lint_on UNUSEDSIGNAL */

    // Activation bank write interface
    output logic [$clog2(ACT_DEPTH)-1:0]    act_addr,
    output logic                             act_we,
    output logic [AXI_DATA_W-1:0]           act_wdata
);

// Beat counter
logic [$clog2(NUM_BEATS):0] beat_cnt;
localparam logic [$clog2(NUM_BEATS):0] LAST_BEAT = ($clog2(NUM_BEATS)+1)'(NUM_BEATS - 1);
logic receiving;

// Ready: accept data when enabled and not done
assign s_axis_tready = enable && receiving;

// State
always_ff @(posedge clk) begin
    if (!rst_n) begin
        beat_cnt  <= '0;
        receiving <= 1'b0;
        done      <= 1'b0;
    end else if (enable && !receiving && !done) begin
        receiving <= 1'b1;
        beat_cnt  <= '0;
    end else if (receiving && s_axis_tvalid && s_axis_tready) begin
        beat_cnt <= beat_cnt + 1;
        if (beat_cnt == LAST_BEAT || s_axis_tlast) begin
            receiving <= 1'b0;
            done      <= 1'b1;
        end
    end else if (!enable) begin
        done <= 1'b0;
        receiving <= 1'b0;
        beat_cnt <= '0;
    end
end

// Write to activation bank
assign act_addr  = beat_cnt[$clog2(ACT_DEPTH)-1:0];
assign act_we    = receiving && s_axis_tvalid && s_axis_tready;
assign act_wdata = s_axis_tdata;

endmodule
