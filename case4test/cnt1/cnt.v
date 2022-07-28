module cnt(input clk, input rst, output reg a, output reg b, output reg c);

always @(posedge clk) begin
  if (rst) begin
    a <= 0;
    b <= 0;
    c <= 0;
  end else begin
    a <= ~b;
    b <= ~c;
    c <= ~a;
  end
end

assert property (~(a&!b&c));

endmodule