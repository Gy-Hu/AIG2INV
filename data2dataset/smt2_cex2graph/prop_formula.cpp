//
// Created by galls2 on 29/09/19.
//

//#include <configuration/omg_config.h>
#include "prop_formula.h"
//#include "sat_solver.h"

std::vector<z3::expr> PropFormula::get_all_variables() const {
    std::vector<z3::expr> all_vars;
    for (const auto& vars_entry : _variables) {
        const z3::expr_vector &vars = vars_entry.second;
        for (size_t i = 0; i < vars.size(); ++i)
        {
            const z3::expr& var = vars[i];
            all_vars.push_back(var);
        }
    }
    return all_vars;
}

const z3::expr_vector &PropFormula::get_vars_by_tag(const std::string& tag) const {
    return _variables.at(tag);
}

const std::map<std::string, z3::expr_vector> &PropFormula::get_variables_map() const {
    return _variables;
}


std::string PropFormula::to_string() const {
    std::string res = std::string("---------\n")+_formula.to_string() + std::string("\nThe vars are:\n");
    for (const auto& it : _variables) {
        res += it.first + std::string(" : ");
        for (size_t i = 0; i < it.second.size(); ++i)
            res += it.second[i].to_string() + std::string(" ");
        res += '\n';
    }
    res += std::string("---------\n");
    return res;
}

const z3::expr &PropFormula::get_raw_formula() const {
    return _formula;
}

z3::expr PropFormula::get_raw_formula() {
    return _formula;
}


PropFormula::PropFormula(const z3::expr &formula, const std::map<std::string, z3::expr_vector> &variables)  : _formula(formula), _variables(
        variables) {}

z3::context &PropFormula::get_ctx() const {
    return _formula.ctx();
}

