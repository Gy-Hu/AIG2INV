module cnt(input clk, input rst, output reg a, output reg b, output reg c, output reg d, output reg e, output reg f, output reg g);

always @(posedge clk) begin
  if (rst) begin
    {a,b,c,d,e,f,g} <= 7'b1000000;
  end else begin
    {a,b,c,d,e,f,g} <= {b,c,d,e,f,g,a};
  end
end

assert property (~(a&b));

endmodule