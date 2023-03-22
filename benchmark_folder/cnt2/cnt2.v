module cnt(input clk, input rst, output reg a, output reg b, output reg c);

always @(posedge clk) begin
  if (rst) begin
    a <= 1;
    b <= 0;
    c <= 0;
  end else begin
    a <= b;
    b <= a;
    c <= a^b;
  end
end

assert property (~b || c);

endmodule
