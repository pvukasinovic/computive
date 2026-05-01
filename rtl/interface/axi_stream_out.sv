// axi_stream_out.sv — AXI-Stream master for output activation transmission
// From rtl-interface-spec.md §4
// Transmits 80 beats x 8 bytes = 640 bytes from activation bank
// TLAST asserted on beat 80
// Stalls (holds TVALID high, waits for TREADY) if downstream not ready

/* verilator lint_off VARHIDDEN */
module axi_stream_out #(
    parameter int AXI_DATA_W = 64,
    parameter int NUM_BEATS  = 80,
    parameter int ACT_DEPTH  = 80
) (
    input  logic                             clk,
    input  logic                             rst_n,

    // Control
    input  logic                             enable,
    output logic                             done,

    // AXI-Stream master interface
    output logic [AXI_DATA_W-1:0]           m_axis_tdata,
    output logic                             m_axis_tvalid,
    input  logic                             m_axis_tready,
    output logic                             m_axis_tlast,
    output logic [AXI_DATA_W/8-1:0]         m_axis_tkeep,

    // Activation bank read interface
    output logic [$clog2(ACT_DEPTH)-1:0]    act_addr,
    input  logic [AXI_DATA_W-1:0]           act_rdata
);

// Beat counter — sized to hold NUM_BEATS
logic [$clog2(NUM_BEATS):0] beat_cnt;
localparam logic [$clog2(NUM_BEATS):0] LAST_BEAT = ($clog2(NUM_BEATS)+1)'(NUM_BEATS - 1);

// FSM
typedef enum logic [1:0] {
    S_IDLE,
    S_READ,     // Issue SRAM read address (1 cycle latency)
    S_TRANSMIT, // Drive AXI-Stream data
    S_DONE
} out_state_t;

out_state_t state, state_next;

always_comb begin
    state_next = state;
    case (state)
        S_IDLE:     if (enable) state_next = S_READ;
        S_READ:     state_next = S_TRANSMIT;
        S_TRANSMIT: begin
            if (m_axis_tready) begin
                if (beat_cnt == LAST_BEAT)
                    state_next = S_DONE;
                else
                    state_next = S_READ;
            end
        end
        S_DONE: state_next = S_IDLE;
        default: state_next = S_IDLE;
    endcase
end

always_ff @(posedge clk) begin
    if (!rst_n)
        state <= S_IDLE;
    else
        state <= state_next;
end

// Beat counter
always_ff @(posedge clk) begin
    if (!rst_n || state == S_IDLE)
        beat_cnt <= '0;
    else if (state == S_TRANSMIT && m_axis_tready)
        beat_cnt <= beat_cnt + 1;
end

// SRAM read address: prefetch one ahead during transmit
assign act_addr = beat_cnt[$clog2(ACT_DEPTH)-1:0];

// AXI-Stream output
assign m_axis_tdata  = act_rdata;
assign m_axis_tvalid = (state == S_TRANSMIT);
assign m_axis_tlast  = (state == S_TRANSMIT) && (beat_cnt == LAST_BEAT);
assign m_axis_tkeep  = {(AXI_DATA_W/8){1'b1}};  // All bytes valid

// Done
assign done = (state == S_DONE);

endmodule
