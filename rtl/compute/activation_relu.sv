// activation_relu.sv — Signed INT8 ReLU (combinational)
// output = (input[7]) ? 0 : input
// When ENABLE=0, passes input through unchanged

module activation_relu #(
    parameter int DATA_W = 8
) (
    input  logic signed [DATA_W-1:0] data_in,
    input  logic                     enable,
    output logic signed [DATA_W-1:0] data_out
);

always_comb begin
    if (enable && data_in[DATA_W-1])
        data_out = '0;
    else
        data_out = data_in;
end

endmodule
