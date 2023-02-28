#include "aig_simple_parser.h"
extern "C" {
#include "aiger.h"
}
#include <iostream>
#include <boost/range/join.hpp>
//#include "aig_to_aag.h"

#include "prop_formula.h"

#include <regex>

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdarg.h>
#include "z3_utils.h"

#include <sstream>
#include <vector>
#include <array>
#include <cassert>
#include <string>

template <size_t MAX_SIZE>
size_t split(const std::string &s, char delim, std::array<std::string, MAX_SIZE> &elems)
{
    static_assert(MAX_SIZE > 0, "Size must be positive");
    
    std::stringstream ss(s);
    std::string item;
    size_t count = 0;

    while(std::getline(ss, item, delim))
    {
#ifdef DEBUG
        assert(count < MAX_SIZE);
#endif
        elems[count++] = item;
    }
    return count;
}

template <size_t N>
std::array<std::string, N> split_to(const std::string& s, char delim)
{
    std::array<std::string, N> arr_to_ret;
    size_t elements_inserted = 0;

    std::stringstream ss(s);
    std::string item;

    while(std::getline(ss, item, delim) && elements_inserted < N) {
        arr_to_ret[elements_inserted++] = item;
    }
    assert(elements_inserted == N);
    return arr_to_ret;
}

typedef struct stream stream;
typedef struct memory memory;

struct stream
{
    double bytes;
    FILE *file;
};

struct memory
{
    double bytes;
    double max;
};

static void *
aigtoaig_malloc (memory * m, size_t bytes)
{
    m->bytes += bytes;
    assert (m->bytes);
    if (m->bytes > m->max)
        m->max = m->bytes;
    return malloc (bytes);
}

static void
aigtoaig_free (memory * m, void *ptr, size_t bytes)
{
    assert (m->bytes >= bytes);
    m->bytes -= bytes;
    free (ptr);
}

struct StringWriter
{
    StringWriter() = default;
    std::string current_line;
    std::vector<std::string> lines;
    void write_char(char ch)
    {
        if (ch == '\n')
        {
            lines.push_back(current_line);
            current_line.clear();
        }
        else
        {
            current_line += ch;
        }
    }
};

int string_writer_put(char to_write, StringWriter* write_to)
{
    write_to->write_char(to_write);
    return (int) to_write;
}

std::vector<std::string> aig_to_aag_lines(const std::string& aig_path) {

    const char *src = aig_path.data(), *error;
    memory memory;
    aiger *aiger;


    memory.max = memory.bytes = 0;
    aiger = aiger_init_mem(&memory,
                           (aiger_malloc) aigtoaig_malloc,
                           (aiger_free) aigtoaig_free);

    error = aiger_open_and_read_from_file(aiger, src);
    if (error) {
        //throw OmgException(std::string("Aig Read error: ").append(error).data());
        std::cout<<"Aig Read error: "<<error<<std::endl;
    }


    StringWriter string_writer;

    if (!aiger_write_generic(aiger, aiger_ascii_mode,
                             &string_writer, (aiger_put) string_writer_put))
        std::cout<<"Aig Write error: "<<aiger_error(aiger)<<std::endl;
        //throw OmgException("Aig Write error!");


    aiger_reset(aiger);


    return string_writer.lines;
}

AigParser::AigParser(const std::string &aig_path)
{
    //std::vector<std::string> file_lines = aig_to_aag_lines(aig_path);
    //aiger * aig = aiger_init();
    
    //freopen(aig_path.c_str(), "r", stdin);
    //const char * msg = aiger_read_from_file(aig, stdin);
    //std::cout<< msg <<std::endl;
    std::vector<std::string> file_lines = aig_to_aag_lines(aig_path);
    extract_metadata(file_lines[0]);
    std::cout<<"Finished extracting metadata"<<std::endl;
    //extract_ap_mapping(file_lines);
    //std::cout<<"Finished extracting ap mapping"<<std::endl;
    extract_literals(file_lines);
    std::cout<<"Finished extracting literals"<<std::endl;
    
    std::unordered_map<size_t, z3::expr> lit_formulas = calc_literal_formulas(file_lines);
    
    std::vector<size_t> lit_ids;
    //print all the formulas in lit_formulas
    for (auto lit_formula : lit_formulas)
    {   
        //if lit_formula.first in _next_state_literals, print it
        for(auto lit_id : _next_state_literals)
        {
            if(lit_id == lit_formula.first)
            {
                std::cout<<lit_formula.first<<" "<<lit_formula.second<<std::endl;
            }
        }
        //std::cout<<lit_formula.first<<" "<<lit_formula.second<<std::endl;
        //std::cout<<lit_formula.first<<std::endl;
        
        //store the formula in the map in the order of the literal id
        //lit_formulas_ordered[lit_formula.first] = lit_formula.second;

        //store the literal id in the vector
        lit_ids.push_back(lit_formula.first);


        /*
        the output like this:

        12 ->34 (and (or |14| |12|) |2|)
        14 ->18 |18|
        16 ->38 (and (or |18| |16|) |4|)
        18 ->22 |22|
        20 ->42 (and (or |22| |20|) |6|)
        22 ->26 |26|
        24 ->46 (and (or |26| |24|) |8|)
        26 ->31 (not |30|)
        28 ->50 (and (not (and |30| (not |28|))) |10|)
        30 ->15 (not |14|)
        */

       /*
       The output tackled by pyPDR

       v12_prime == And(Not(And(Not(v14), Not(v12))), i2), 
       v14_prime == v18,
       v16_prime == And(Not(And(Not(v18), Not(v16))), i4), 
       v18_prime == v22, 
       v20_prime == And(Not(And(Not(v22), Not(v20))), i6), 
       v22_prime == v26, 
       v24_prime == And(Not(And(Not(v26), Not(v24))), i8), 
       v26_prime == Not(v30), v28_prime == And(Not(And(v30, Not(v28))), i10), 
       v30_prime == Not(v14)

       */

    }

    //order the literals in lit_ids according to the literal id
    std::sort(lit_ids.begin(), lit_ids.end());
    for(auto lit_id : lit_ids)
    {
        std::cout<<lit_id<<std::endl;
    }

    //print the size of lit_formulas
    std::cout<<"Finished calculating literal formulas, the size is:"<<lit_formulas.size()<<std::endl;
    std::cout<<"Finished handling all the literal formulas"<<std::endl;

    calculate_tr_formula(lit_formulas);
    std::cout<<"Finished calculating transition relation formula"<<std::endl;
}

AigParser::~AigParser()
{
    std::cout<<"AigParser destructor called"<<std::endl;

}


void AigParser::extract_metadata(const std::string &first_aag_line)
{
    std::array<std::string, 6> components = split_to<6>(first_aag_line, ' ');
    assert(components[0] == std::string("aag"));
    _metadata[M] = std::stoul(components[1]);
    _metadata[I] = std::stoul(components[2]);
    _metadata[L] = std::stoul(components[3]);
    _metadata[O] = std::stoul(components[4]);
    _metadata[A] = std::stoul(components[5]);

    _first_and_literal = (_metadata.at(AigMetadata::I) + _metadata.at(L) + 1) * 2;
}

void AigParser::extract_ap_mapping(const std::vector<std::string> &aag_lines) {

    const std::regex ap_line_regex("^[ilo][0-9].*");
    const size_t start_search_idx = _metadata[A]+_metadata[L]+_metadata[I] + _metadata[O];
    for (size_t i = start_search_idx; i < aag_lines.size(); ++i)
    {
        const std::string &aag_line = aag_lines[i];

        if (std::regex_match(aag_line, ap_line_regex))
        {
            if (_first_ap_index == 0) { _first_ap_index = i; }
            std::array<std::string, 2> parts = split_to<2>(aag_line, ' ');
            _ap_to_symb[parts[1]] = parts[0];
            _symb_to_ap[parts[0]] = parts[1];
        }
    }
    assert(_first_ap_index > 0);
}

void AigParser::extract_literals(const std::vector<std::string> &aag_lines) {
    for (size_t i = 1; i < 1 + _metadata[I]; ++i) {
        _in_literals.push_back(std::stoul(aag_lines[i]));
    }

    for (size_t i = _metadata[I] + 1; i < 1 + _metadata[L] + _metadata[I]; ++i) {
        std::array<std::string, 2> parts = split_to<2>(aag_lines[i], ' ');
        _prev_state_literals.push_back(std::stoul(parts[0]));
        _next_state_literals.push_back(std::stoul(parts[1]));
    }

    for (size_t i = _metadata[I] + _metadata[L] + 1; i < 1 + _metadata[L] + _metadata[O] + _metadata[I]; ++i) {
        _out_literals.push_back(std::stoul(aag_lines[i]));
    }
}

std::unordered_map<size_t, z3::expr> AigParser::calc_literal_formulas(const std::vector<std::string> &aag_lines)
{
    //build this map to store the literal formulas as {literals_id: formula(only contains latches and input)}
    std::unordered_map<size_t, z3::expr> lit_formulas;

    lit_formulas.emplace(0, _ctx.bool_val(false));
    lit_formulas.emplace(1, _ctx.bool_val(true));
    for (auto lit : _in_literals)
        lit_formulas.emplace(lit, _ctx.bool_const(std::to_string(lit).data()));
    for (auto lit : _prev_state_literals)
        lit_formulas.emplace(lit, _ctx.bool_const(std::to_string(lit).data()));

    //size_t first_and_line = _first_ap_index - _metadata[A];
    size_t first_and_line = _metadata[I] + _metadata[L] + _metadata[O] + 1;
    for (auto lit : _next_state_literals) dfs(aag_lines, lit_formulas, first_and_line, lit);
    for (auto lit : _out_literals) dfs(aag_lines, lit_formulas, first_and_line, lit);

    return lit_formulas;
}

const AigParser &
AigParser::dfs(const std::vector<std::string> &lines, std::unordered_map<size_t, z3::expr> &formulas,
               size_t first_line, size_t target_lit) const {
    if (formulas.find(target_lit) == formulas.end()) {
        if (target_lit % 2 == 1) {
            dfs(lines, formulas, first_line, target_lit - 1);
            if (formulas.at(target_lit - 1).is_and()) {
                const size_t and_line_index = first_line + (target_lit - _first_and_literal) / 2;
                const std::string &and_line = lines[and_line_index];
                std::array<std::string, 3> parts = split_to<3>(and_line, ' ');
                size_t left_operand = std::stoul(parts[1]);
                size_t right_operand = std::stoul(parts[2]);
                if (left_operand % 2 == 1 && right_operand % 2 == 1) {
                    formulas.emplace(target_lit, formulas.at(left_operand - 1) || formulas.at(right_operand - 1));
                } else {
                    formulas.emplace(target_lit, !formulas.at(target_lit - 1));
                }
            } else {
                formulas.emplace(target_lit, !formulas.at(target_lit - 1));
            }
        } else {
            const size_t and_line_index = first_line + (target_lit - _first_and_literal) / 2;
            const std::string &and_line = lines[and_line_index];
            std::array<std::string, 3> parts = split_to<3>(and_line, ' ');
            size_t left_operand = std::stoul(parts[1]);
            size_t right_operand = std::stoul(parts[2]);

            dfs(lines, formulas, first_line, left_operand);
            dfs(lines, formulas, first_line, right_operand);
            formulas.emplace(target_lit, formulas.at(left_operand) && formulas.at(right_operand));
        }
    }
    return *this;
}

void AigParser::calculate_tr_formula(const std::unordered_map<size_t, z3::expr> &formulas) {
    size_t new_var_index = (_metadata[AigMetadata::M] + 1) * 2 + 1;

    std::vector<z3::expr> prev_in, prev_latch, prev_out, next_in, next_latch, next_out;
    std::vector<std::reference_wrapper<std::vector<z3::expr>>>
            ins = {{prev_in}, {next_in}},
            latches = {{prev_latch}, {next_latch}},
            outs = {{prev_out}, {next_out}};

    generate_new_names(ins, new_var_index, _metadata[AigMetadata::I]);
    generate_new_names(latches, new_var_index, _metadata[AigMetadata::L]);
    generate_new_names(outs, new_var_index, _metadata[AigMetadata::O]);

    z3::expr_vector orig_in(_ctx), orig_ps(_ctx), orig_ns(_ctx), orig_out(_ctx);
    for (size_t i_lit : _in_literals) orig_in.push_back(_ctx.bool_const(std::to_string(i_lit).data()));
    for (size_t ps_lit : _prev_state_literals) orig_ps.push_back(_ctx.bool_const(std::to_string(ps_lit).data()));
    for (size_t ns_lit : _next_state_literals) orig_ns.push_back(_ctx.bool_const(std::to_string(ns_lit).data()));
    for (size_t o_lit : _out_literals) orig_out.push_back(_ctx.bool_const(std::to_string(o_lit).data()));

    generate_state_formula(formulas, prev_out, orig_in, orig_ps, prev_in, prev_latch);

    z3::expr_vector ltr_parts(_ctx);
    for (size_t i = 0; i < _next_state_literals.size(); ++i) {
        auto &orig = const_cast<z3::expr &>(formulas.at(_next_state_literals[i]));
        z3::expr named_ltr_formula = orig.substitute(orig_in, vec_to_expr_vec(_ctx, prev_in))
                .substitute(orig_ps, vec_to_expr_vec(_ctx, prev_latch));
        z3::expr constraint = next_latch[i] == named_ltr_formula;
        ltr_parts.push_back(constraint);
    }
    z3::expr ltr = z3::mk_and(ltr_parts);

    z3::expr state_next = _state_formula->substitute(vec_to_expr_vec(_ctx, prev_in), vec_to_expr_vec(_ctx, next_in))
            .substitute(vec_to_expr_vec(_ctx, prev_latch), vec_to_expr_vec(_ctx, next_latch))
            .substitute(vec_to_expr_vec(_ctx, prev_out), vec_to_expr_vec(_ctx, next_out));

    z3::expr_vector tr_parts(_ctx);
    tr_parts.push_back(ltr);
    tr_parts.push_back(*_state_formula);
    tr_parts.push_back(state_next);
    z3::expr tr_raw = z3::mk_and(tr_parts);

    z3::expr_vector ps(_ctx), ns(_ctx);
    for (const z3::expr &it : boost::join(prev_latch, prev_out)) ps.push_back(it);
    for (const z3::expr &it : boost::join(next_latch, next_out)) ns.push_back(it);

    std::map<std::string, z3::expr_vector> var_tags =
            {
                    {"in0", vec_to_expr_vec(_ctx, prev_in)},
                    {"in1", vec_to_expr_vec(_ctx, next_in)},
                    {"ps", ps}, {"ns", ns}
            };
    _tr_formula = std::make_unique<PropFormula>(tr_raw, std::move(var_tags));
    // std::cout<<"tr_raw: "<<tr_raw<<std::endl;
    // //print var_tags
    // for(auto it=var_tags.begin();it!=var_tags.end();it++){
    //     std::cout<<it->first<<": ";
    //     for(auto it2=it->second.begin();it2!=it->second.end();it2++){
    //         std::cout<<*it2<<" ";
    //     }
    //     std::cout<<std::endl;
    // }
    //print _tr_formula
    const z3::expr& raw_formula = _tr_formula->get_raw_formula();
    std::cout << "transition relation formula before simplification: \n" << raw_formula << std::endl;
    
    const z3::expr& raw_formula_after_simplification = _tr_formula->get_raw_formula().simplify();
    //const z3::expr& raw_formula = _tr_formula->get_raw_formula();
    std::cout << "transition relation formula after simplication: \n" << raw_formula_after_simplification << std::endl;
    //
}

void AigParser::generate_new_names(std::vector<std::reference_wrapper<std::vector<z3::expr>>> &vec_of_vecs, size_t &start_from,
                                   size_t num_iters) {
    for (size_t i = 0; i < num_iters; ++i)
    {
        for (auto& vec : vec_of_vecs)
            vec.get().push_back(to_var(_ctx, ++start_from));
    }
}

void
AigParser::generate_state_formula(const std::unordered_map<size_t, z3::expr> &formulas, std::vector<z3::expr> &prev_out,
                              const z3::expr_vector &orig_in, const z3::expr_vector &orig_ps,
                               std::vector<z3::expr> &prev_in,
                              std::vector<z3::expr> &prev_latch) {
    z3::expr_vector state_formula_parts(_ctx);
    for (size_t i = 0; i< prev_out.size(); ++i)
    {
        size_t o_lit = _out_literals[i];

        z3::expr out_formula = prev_out[i] == formulas.at(o_lit);
        z3::expr named_out_formula =
                out_formula.substitute(orig_in, vec_to_expr_vec(_ctx, prev_in))
                           .substitute(orig_ps, vec_to_expr_vec(_ctx, prev_latch)); // BUG?
        state_formula_parts.push_back(named_out_formula);
    }
    _state_formula = std::make_unique<z3::expr>(std::move(mk_and(state_formula_parts)));
}