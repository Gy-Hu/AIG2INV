#pragma once

#include <z3++.h>
#include <vector>
#include <set>

z3::expr to_var(z3::context& ctx, size_t val);

template <typename T>
z3::expr_vector iterable_to_expr_vec(z3::context& ctx, const T& iterable);
z3::expr_vector vec_to_expr_vec(z3::context& ctx, const std::vector<z3::expr>& vec);