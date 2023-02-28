//
// Created by galls2 on 08/10/19.
//

#include <unordered_set>
#include "z3_utils.h"
#include "version_manager.h"

template <typename T>
z3::expr_vector iterable_to_expr_vec(z3::context& ctx, const T& iterable)
{
    z3::expr_vector expr_vector(ctx);
    for (const auto& it : iterable) expr_vector.push_back(it);
    return expr_vector;
}

std::string expr_vector_to_string(const z3::expr_vector& expr_vec)
{
    std::string res;
    for (unsigned i = 0; i < expr_vec.size(); ++i)
    {
    res += expr_vec[i].to_string() +"  ";
    }

    return res;
}

z3::expr_vector vec_to_expr_vec(z3::context& ctx, const std::vector<z3::expr>& vec)
{
    return iterable_to_expr_vec<std::vector<z3::expr>>(ctx, vec);
}

z3::expr to_var(z3::context& ctx, size_t val)
{
    return ctx.bool_const(VersionManager::new_version(val).data());
}