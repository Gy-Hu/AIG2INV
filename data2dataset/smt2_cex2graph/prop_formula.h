
//
// Created by galls2 on 29/09/19.
//

#pragma once
#include <vector>
#include <z3++.h>
#include <utility>
#include <unordered_map>
#include <map>
#include <cassert>


class PropFormula {
public:
    PropFormula(const z3::expr& formula, const std::map<std::string, z3::expr_vector> &variables); // move?
    const z3::expr& get_raw_formula() const;
    z3::expr get_raw_formula();

    std::string to_string() const;
    const z3::expr_vector& get_vars_by_tag(const std::string& tag) const;
    std::vector<z3::expr> get_all_variables() const;
    bool is_sat() const;
    z3::context& get_ctx() const;
    const std::map<std::string, z3::expr_vector> & get_variables_map() const;

private:
    const z3::expr _formula;
    std::map<std::string, z3::expr_vector> _variables; // TODO remove this, keep this map somewhere and just use the z3::expr

};


