# find the last clauses that block ...
from numpy import var
import z3
import pandas as pd
import os
import ternary_sim
import sys
from tCube import tCube
from natsort import natsorted
#from deps.pydimacs_changed.formula import CNFFormula
#import deps.PyMiniSolvers.minisolvers as minisolvers
import deps.pydimacs_changed.formula
#from pysat.formula import CNF
#from pysat.solvers import Solver
import pysat.formula
import pysat.solvers
import itertools
import sp_converter
import copy


class ExtractCnf(object):
    def __init__(self, aagmodel, clause, name, generalize=False, aig_path='', generate_smt2=False, inv_correctness_check=True, model_checker = 'abc', deep_simplification = True):
        self.aig_path = aig_path
        self.generate_smt2 = generate_smt2 # default: generate smt2
        self.model_checker = model_checker

        # generalize the predescessor?
        self.generalize = generalize
        
        # build clauses
        self.aagmodel = aagmodel
        self.clause_unexpanded = clause.clauses
        self.clauses, self.clauses_var_lst = self._build_clauses(aagmodel.svars, clause.clauses)

        assert len(aagmodel.inputs) == len(aagmodel.primed_inputs)
        assert len(aagmodel.svars) == len(aagmodel.primed_svars)
        # input to input', var to var'
        self.v2prime = dict(list(zip(aagmodel.inputs, aagmodel.primed_inputs )) + list(zip(aagmodel.svars, aagmodel.primed_svars)))
        # latch2next
        self.vprime2nxt = {self.v2prime[v]:expr for v, expr in aagmodel.latch2next.items()}
        # to list
        self.v2prime = list(self.v2prime.items())
        self.vprime2nxt = list(self.vprime2nxt.items())
        
        #XXX: Double Check before running the script
        # Use symbolic simplification to simplify the transition relation 
        self.vprime2nxt_without_simplification = copy.deepcopy(self.vprime2nxt)# backup the original transition relation
        self.vprime2nxt = [(vprime, sp_converter.to_z3(sp_converter.to_sympy(nxt))) for vprime, nxt in self.vprime2nxt]
        #self._check_tr_correctness_after_simplification(self.vprime2nxt_without_simplification, self.vprime2nxt)

        self.init = aagmodel.init
        self.lMap = {str(v):v for v in aagmodel.svars}
        self.Inp_SVars_Map = {str(l): l for l in aagmodel.inputs + aagmodel.svars + aagmodel.primed_inputs + aagmodel.primed_svars}

        self.model_name = name
        self.cex_generalization_ground_truth = []

        # initialize the ternary simulator
        self.ternary_simulator = ternary_sim.AIGBuffer()
        
        # check validation of using the ternary simulator
        self.ternary_simulator_valid = True
        for _, updatefun in self.vprime2nxt: 
            if self.ternary_simulator.check_validation(updatefun) == False:
                self.ternary_simulator_valid = False
                break
        
        if self.ternary_simulator_valid==True and deep_simplification==False:
            for _, updatefun in self.vprime2nxt: self.ternary_simulator.register_expr(updatefun)
        
        #XXX: Double Check before running the script
        # check the duplicated update functions
        # self._check_eq_of_tr()
        
        #XXX: Double Check before running the script
        #if inv_correctness_check: self._check_inv_correctness()
    
    def _check_tr_correctness_after_simplification(self, t1, t2):
        for a,b in zip(t1, t2):
            s = z3.Solver()
            proposition = a[1] == b[1] # assertion is whether b1 and b2 are equal
            s.add(z3.Not(proposition))
            # proposition proved if negation of proposition is unsat
            if s.check() == z3.sat:
                # it should be unsat if the transition relation is simplified correctly -> which is unsat
                print("The transition relation is not simplified correctly!")
        
    def _check_eq_of_tr(self):
        prime_variable = [_[1] for _ in self.vprime2nxt]
        # for each variable, check the equality of them
        for a, b in itertools.combinations(prime_variable, 2):
            s = z3.Solver()
            proposition = a == b # assertion is whether b1 and b2 are equal
            s.add(z3.Not(proposition))
            # proposition proved if negation of proposition is unsat
            if s.check() == z3.unsat: # a == b
                print("Oops!! There are duplicated update functions for the same variable!")
            
    def _parse_dimacs(self,filename):
        # Check this variable is string type or not
        if(isinstance(filename, list)):
            assert len(filename) == 1
            lines = [line for line in filename[0].strip().split("\n")]
            for line in lines:
                if "c" == line.strip().split(" ")[0]:
                    index_c = lines.index(line)
                    break
            header = lines[0].strip().split(" ")
            assert(header[0] == "p")
            n_vars = int(header[2])
            iclauses = [[int(s) for s in line.strip().split(" ")[:-1]] for line in lines[1:index_c]]
            return n_vars, iclauses
        elif(isinstance(filename, str)):
            with open(filename, 'r') as f:
                lines = f.readlines()
            # Find the first index of line that contains string "c"
            for line in lines:
                if "c" == line.strip().split(" ")[0]:
                    index_c = lines.index(line)
                    break
            header = lines[0].strip().split(" ")
            assert(header[0] == "p")
            n_vars = int(header[2])
            iclauses = [[int(s) for s in line.strip().split(" ")[:-1]] for line in lines[1:index_c]]
            return n_vars, iclauses
            
    def _check_satisfiability_by_differentsolver(self, z3_slv, z3_expr=None):
        
        '''
        # use pysat
        '''
        pysat_solver = pysat.solvers.Solver(name='cd')
        f =  pysat.formula.CNF(from_string=z3_slv.dimacs())
        pysat_solver.append_formula(f.clauses, no_return=False)
        res = pysat_solver.solve()
        return res
    
        '''
        # use minisat
        f = CNFFormula.from_z3(z3_expr)
        cnf_string_lst = f.to_dimacs_string()
        n, iclauses = self._parse_dimacs(cnf_string_lst)
        minisolver = minisolvers.MinisatSolver()
        for i in range(n): minisolver.new_var(dvar=True)
        for iclause in iclauses: minisolver.add_clause(iclause)
        is_sat = minisolver.solve()
        
        return is_sat
        '''
    
    def _check_inv_correctness(self):
        '''
        check the correctness of the inductive invariant
        - init -> inv
        - inv -> P
        - inv & T -> inv’
        '''
        
        prop = z3.Not(self.aagmodel.output)
        inv = z3.And(self.clauses) if self.model_checker == 'abc' else z3.And(z3.And(self.clauses),prop) if self.model_checker == 'ic3ref' else None
        assert inv != None, "Not defined model checker!"
        #inv = z3.And(z3.And(self.clauses),prop)
        
        # init -> inv
        slv = z3.Solver()
        slv.add(self.init)
        slv.add(z3.Not(inv))
        #XXX: Double check before running scripts
        #if self._check_satisfiability_by_differentsolver(slv) == True: 
        if slv.check() == z3.sat: 
            self._report2log_inv_check(self.aig_path,\
                "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/log/error_handle/bad_inv.log",\
                "init -> inv is not satisfied")
            print("Invariant is not correct! init -> inv is not satisfied")
            #assert False, "init -> inv is not satisfied"
            sys.exit()
        
        # inv -> P
        slv = z3.Solver()
        slv.add(inv)
        slv.add(z3.Not(prop))
        #if self._check_satisfiability_by_differentsolver(slv) == True:
        if slv.check() == z3.sat:
            self._report2log_inv_check(self.aig_path,\
                "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/log/error_handle/bad_inv.log",\
                "inv -> P is not satisfied")
            print("Invariant is not correct! inv -> P is not satisfied")
            #assert False, "inv -> P is not satisfied"
            sys.exit()
        
        # inv & T -> inv’
        slv = z3.Solver()
        slv.add(inv)
        slv.add(z3.Not(z3.substitute(z3.substitute(inv, self.v2prime), self.vprime2nxt)))
        #XXX: Double check before running scripts
        #if self._check_satisfiability_by_differentsolver(slv) == True: 
        if slv.check() == z3.sat:
            self._report2log_inv_check(self.aig_path,\
                "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/log/error_handle/bad_inv.log", \
                "inv & T -> inv’ is not satisfied")
            print("Invariant is not correct! inv & T -> inv’ is not satisfied")
            #assert False, "inv & T -> inv’ is not satisfied"
            sys.exit()
            
        # with open("/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/log/error_handle/bad_inv.log", "a+") as fout: 
        #     fout.write(f"Finish checking the correctness of the inductive invariant! All good!\n")
        # fout.close()
        print("Finish checking the correctness of the inductive invariant! All good!")
        
        
    def _build_clauses(self, svars, clauses):
        ret_clauses = []
        ret_clauses_var_lst = []
        for cl in clauses:
            cl_z3 = []
            for var,sign in cl:
                lit = svars[var]
                if sign == -1:
                    lit = z3.Not(lit)
                cl_z3.append(lit)
            ret_clauses_var_lst.append(cl_z3)
            ret_clauses.append(z3.Not(z3.And(cl_z3)))
        return ret_clauses, ret_clauses_var_lst

    def _build_clauses(self, svars, clauses):
        ret_clauses = []
        ret_clauses_var_lst = []
        for cl in clauses:
            cl_z3 = []
            for var,sign in cl:
                lit = svars[var]
                if sign == -1:
                    lit = z3.Not(lit)
                cl_z3.append(lit)
            ret_clauses_var_lst.append(cl_z3)
            ret_clauses.append(z3.Not(z3.And(cl_z3)))
        return ret_clauses, ret_clauses_var_lst

    def _find_clause_to_block(self, model_to_block, model_var_lst, generate_smt2=False, cex_double_check_without_generalization=False):
        for idx in range(len(self.clauses)-1, -1, -1): # search backwards, performas like MIC
            cl = self.clauses[idx]
            cl_var_lst = self.clauses_var_lst[idx]
            slv = z3.Solver()
            slv.add(cl)
            slv.add(model_to_block)
            res = slv.check() # some idx will never be visited, it means some clauses are never been used to block the cex
            if res == z3.unsat: # find the subset (which can be generalized)
                #assert idx!=0, "Extreme case: the first clause is always not been used to block the cex"
                if generate_smt2: self._generate_smt2_and_ground_truth(model_to_block, model_var_lst, (cl, self.clause_unexpanded[idx],cl_var_lst))
                return cl, self.clause_unexpanded[idx]
            assert res == z3.sat
        #XXX: Double check before running scripts
        #print (model_to_block)
        
        #XXX: Double check before running scripts
        #for idx in range(len(self.clauses)-1, -1, -1): print(idx,':', self.clauses[idx])
        if cex_double_check_without_generalization:
            # record this case due to mismatched inv.cnf
            print(f"{self.model_name} Mismatched inductive invariant!! Path:{self.aig_path}")
            self._report2log(self.aig_path, "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/log/error_handle/mismatched_inv.log")
            assert False, "BUG: cannot find clause to block bad state"
        else:
            return None, None

    def _report2log(self, aig_path, log_file):
        # append the error message to the log file
        with open(log_file, "a+") as fout:
            fout.write(f"Error: {aig_path} has mismatched inductive invariants. \n")
        fout.close()
        
    def _report2log_inv_check(self, aig_path, log_file, error_msg):
        # append the error message to the log file
        with open(log_file, "a+") as fout:
            fout.write(f"Error: {aig_path} has bad inv. {error_msg}.\n")
        fout.close()
        
    def _cex_deep_simplification(self,expr):
        pass

    def _generate_smt2_and_ground_truth(self, model_to_block, model_var_lst, cl_info):    #smt2_gen_IG is a switch to trun on/off .smt file generation 
        '''
        input:
            cl_info[0]: cl (z3 expression of the clauses, which is the generalization target)
            cl_info[1]: self.clause_unexpanded[idx] (the abstracted clause, only contains index and sign)
            cl_info[2]: cl_var_lst (list of variables in the clause)
        '''
        
        '''
        -------------export the smt2 for graph generation----------------
        '''
        s_smt = z3.Solver()
        cex_prime = (z3.substitute(z3.substitute(model_to_block, self.v2prime), self.vprime2nxt))
        # inductive relative: F[i - 1] & T & Not(badCube) & badCube'
        # graph construction: Not(badCube) & badCube'
        Cube =  z3.And(
                z3.Not(model_to_block), 
                cex_prime
                )
        '''
        Export BoolRef in z3 to the sympy and do the optimization
        - convert z3 expression to AST by using simplify(), then we can leverage __str__ or __repr__ function to get the string
        - directly use str(BoolRef) to get the string (maybe not the best way, the string may be cut off)
        - convert z3 expression to s-expression, like sexpr() -> but how to parse by sympy?
        - solver has __repr__ function, but it is not the best way to get the string(contains '[]')
        - use z3.Tactic, like: simpl = z3.Tactic("simplify"). Also can use z3.TryFor(simpl, 300000) to configure in detail \
        This (simpl) has function as_expr() that can convert the z3 expression to string
        - using the method that mentioned in https://github.com/Z3Prover/z3/issues/4822,\
            https://stackoverflow.com/questions/65022373/z3-boolean-expression-simplification
        - use interp https://www.philipzucker.com/programming-and-interactive-proving-with-z3py/
        - converter for sympy string to z3
        https://stackoverflow.com/questions/75461163/python-how-to-parse-boolean-sympy-tree-expressions-to-boolean-z3py-expressions
        - using prove(f == g) to check if f and g are equivalent:\
            https://stackoverflow.com/questions/60792685/how-to-simplify-ornoty-andy-notx-to-ornoty-notx-with-z3
        '''
        
        '''
        -------------Try to use symbolic simplification before dump smt2----------------
        '''
        #XXX: Double check before running the script
        # maybe make it too simple...
        # sp_converter.to_sympy(Cube)
        
        
        '''
        -----------------------------Dump smt2---------------------------------
        '''
        for literals in model_var_lst: s_smt.add(literals)
        s_smt.add(Cube)
        # new a folder to store the smt2 files
        if not os.path.exists(f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset/bad_cube_cex2graph/expr_to_build_graph/{self.model_name}"): os.makedirs(f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset/bad_cube_cex2graph/expr_to_build_graph/{self.model_name}")
        filename = f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset/bad_cube_cex2graph/expr_to_build_graph/{self.model_name}/{self.model_name}_{len(self.cex_generalization_ground_truth)}.smt2"
        data = {'inductive_check': f"{self.model_name}_{len(self.cex_generalization_ground_truth)}.smt2"} # create a dictionary that store the smt2 file name and its corresponding ground truth
        with open(filename, mode='w') as f: f.write(s_smt.to_smt2())
        f.close() 

        '''
        ---------------------Export the ground truth----------------------
        '''
        mic_like_res = cl_info[2] # Minimum ground truth has been generated
        convert_var_to_str = lambda x: str(x) if x.decl().kind() != z3.Z3_OP_NOT else str(x.children()[0])
        for idx in range(len(self.aagmodel.svars)): # -> ground truth size is q
            var2str = self.aagmodel.svars[idx]
            assert self.aagmodel.svars[idx].decl().kind() != z3.Z3_OP_NOT # assert the variable is not negated
            data[str(var2str)] = 0
        for idx in range(len(mic_like_res)):
            var2str = convert_var_to_str(mic_like_res[idx])
            data[str(var2str)] = 1 # Mark q-like as 1
        #print(list(data.keys()))
        #print(sorted(list(data.keys())))
        
        # assert the sequence of the ground truth has been sorted
        assert list(data.keys()) == natsorted(list(data.keys())), "BUG: the sequence of the ground truth has not been sorted automatically"
        self.cex_generalization_ground_truth.append(data)

    def _check_inductive(self, clause_list):
        slv = z3.Solver()
        prev = z3.And(clause_list)
        not_p_prime = z3.Not(z3.substitute(z3.substitute(prev, self.v2prime), self.vprime2nxt))
        slv.add(prev)
        slv.add(not_p_prime)
        res = slv.check()
        if res == z3.unsat:
            return True
        assert res == z3.sat
        return False

    def _make_cex_prime(self, cex):
        return z3.substitute(z3.substitute(cex, self.v2prime), self.vprime2nxt)

    def _solve_relative(self, prop, cnfs, prop_only, generalize=True): # -> a dict of only the curr_vars
        '''
        prop: the property to be checked
        cnfs: the clauses has been added for blocking
        prop_only: consider the clauses in cnfs ? prev = (cnfs + prop) : prev= prop
        generalize: generalize predecessor or not
        '''
        # initially, we will remove the props
        # then, maybe we can also remove the additional states

        # model of `T /\ P /\ not(P_prime)`
        slv = z3.Solver()
        prev = z3.And(cnfs+[prop]) # all the added claused (for blocking) + the property
        slv.add(prev)

        post = prop if prop_only else prev # check F[i-1](P) /\ T -> P'?
        not_p_prime = z3.Not(z3.substitute(z3.substitute(post, self.v2prime), self.vprime2nxt))
        slv.add(not_p_prime)
        res = slv.check()
        if res == z3.unsat:
            return None, None, None
        assert res == z3.sat
        # res is SAT! counterexample found, now we can generalize it use unsat core and ternary simulation
        model = slv.model()
        
        # make model_after_generalization back to z3 model, this is a naive way to do it
        # model = [literals for literals in model if str(model[literals]) in str(model_after_generalization.cubeLiterals)]
        # assert len(model) != 0 and len(model) <= len(model_after_generalization.cubeLiterals), "BUG: the model is empty after generalization or failed to generalize the model with the naive way"
        
        # make a new z3 ModelRef that copy the model_after_generalization
        if generalize: #if generalize is True, we will use the predecssor generalization method
            model_after_generalization = self._generalize_predecessor(model, z3.Not(post))
            slv = z3.Solver() # initialize a new solver for model conversion
            for literals in model_after_generalization.cubeLiterals: slv.add(literals)
            res = slv.check() ; model = slv.model()
            assert len(model)!=0, "BUG: the model is empty after generalization"
            assert len(model)==len(model_after_generalization.cubeLiterals), "BUG: the model failed to generalize"

        filtered_model, model_list, model_var_list = self._filter_model(model)
        assert len(model_var_list)!=0, "BUG: the model is empty after removing input and prime variable (includes input primes)"
        return filtered_model, model_list, model_var_list # filtered model is a z3 expression, model_list is [(var_index, sign)], var_lst is [vx == T, vx == F...]

    def _extract(self,literaleq):
        # we require the input looks like v==val
        children = literaleq.children()
        assert(len(children) == 2)
        if str(children[0]) in {'True', 'False'}:
            v = children[1]
            val = children[0]
        elif str(children[1]) in {'True', 'False'}:
            v = children[0]
            val = children[1]
        else:
            assert(False)
        return v, val

    def _generalize_predecessor(self, prev_cube, next_cube_expr):    #smt2_gen_GP is a switch to trun on/off .smt file generation
        '''
        :param prev_cube: sat model of CTI (v1 == xx , v2 == xx , v3 == xxx ...)
        :param next_cube_expr: bad state (or CTI), like !P ( ? /\ ? /\ ? /\ ? .....)
        :return:
        '''
        tcube_cp = tCube(0)
        tcube_cp.addModel(self.Inp_SVars_Map, prev_cube, remove_input=False)

        print("original size of CTI (including input variable): ", len(tcube_cp.cubeLiterals))
        print("Begin to generalize predessor")

        #replace the state as the next state (by trans) -> !P (s')
        nextcube = z3.substitute(z3.substitute(next_cube_expr,  self.v2prime),self.vprime2nxt)
        
        # since s -> s', so s & !s' is unsat, we can use unsat core to remove the literals (until fixed point)
        size_before_unsat_core = len(tcube_cp.cubeLiterals)
        shrink_model_times = 0
        while True:
            s = z3.Solver()
            for index, literals in enumerate(tcube_cp.cubeLiterals):
                s.assert_and_track(literals, f'p{str(index)}')
            # prof.Zhang's suggestion
            #s.add(prevF)
            s.add(z3.Not(nextcube))
            assert(s.check() == z3.unsat)
            core = s.unsat_core()
            core = [str(core[i]) for i in range(len(core))]

            for idx in range(len(tcube_cp.cubeLiterals)):
                if f'p{str(idx)}' not in core:
                    tcube_cp.cubeLiterals[idx] = True

            tcube_cp.remove_true()
            size_after_unsat_core = len(tcube_cp.cubeLiterals)
            if size_after_unsat_core == size_before_unsat_core:
                break
            size_before_unsat_core = size_after_unsat_core
            shrink_model_times += 1
        print("size of CTI after removing according to unsat core: ", size_after_unsat_core)

        if self.ternary_simulator_valid: # only if ternary simulator is valid, 
            '''
            if not valid, bug info like:
            
            "ternary_sim.py, line 89, in register_expr\n    op = expr.
            decl().kind()\nAttributeError: \'bool\' object has no attribute \'decl\'\n'"
            
            Thus, we will skip of using ternary simulator
            '''
            # this is the beginning of ternary simulation-based variable reduction
            simulator = self.ternary_simulator.clone() # I just don't want to mess up between two ternary simulations for different outputs
            simulator.register_expr(nextcube)
            simulator.set_initial_var_assignment(dict([self._extract(c) for c in tcube_cp.cubeLiterals]))

            out = simulator.get_val(nextcube)
            if out == ternary_sim._X:  # this is possible because we already remove once according to the unsat core
                return tcube_cp
            assert out == ternary_sim._TRUE
            for i in range(len(tcube_cp.cubeLiterals)):
                v, val = self._extract(tcube_cp.cubeLiterals[i])
                simulator.set_Li(v, ternary_sim._X)
                out = simulator.get_val(nextcube)
                if out == ternary_sim._X:
                    simulator.set_Li(v, ternary_sim.encode(val))  # set to its original value
                    if simulator.get_val(nextcube) != ternary_sim._TRUE:
                        # This is just to help print debug info in case I made mistakes in coding
                        simulator._check_consistency()
                    # after you recover the original input value, the output node should be true again
                    assert simulator.get_val(nextcube) == ternary_sim._TRUE
                else: # the literal is removable
                    # we should never get _FALSE
                    if simulator.get_val(nextcube) != ternary_sim._TRUE:
                        # This is just to help print debug info in case I made mistakes in coding
                        simulator._check_consistency()
                    assert simulator.get_val(nextcube) == ternary_sim._TRUE
                    tcube_cp.cubeLiterals[i] = True
            tcube_cp.remove_true()
            size_after_ternary_sim = len(tcube_cp.cubeLiterals)
            print("size of CTI after removing according to ternary simulation: ", size_after_ternary_sim)
        return tcube_cp


    def _filter_model(self, model): # keep only current state variables
        no_prime_or_input = [l for l in model if str(l)[0] != 'i' and not str(l).endswith('_prime')]
        # assert the length of no_prime_or_input is the same as the number of state variables in aagmodel.svars
        assert len(no_prime_or_input) <= len(self.aagmodel.svars), "Bug occurs because length of model is not the same as the number of latch state variables"
        cex_expr = z3.And([z3.simplify(self.lMap[str(l)] == model[l]) for l in no_prime_or_input])
        cex_list = [ (self.aagmodel.svars.index(self.lMap[str(l)]), 1 if str(model[l]) == 'True' else -1) for l in no_prime_or_input ]
        var_list = [(self.lMap[str(l)] == model[l]) for l in no_prime_or_input]
        return cex_expr, cex_list, var_list

    def _export_ground_truth(self):
        if self.cex_generalization_ground_truth: # When this list is not empty, it return true
            df = pd.DataFrame(self.cex_generalization_ground_truth)
            df = df.fillna(0)
            if not os.path.exists(f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset/bad_cube_cex2graph/ground_truth_table/{self.model_name}"): os.makedirs(f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset/bad_cube_cex2graph/ground_truth_table/{self.model_name}")
            df.to_csv(f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset/bad_cube_cex2graph/ground_truth_table/{self.model_name}/{self.model_name}.csv")

    
    def get_clause_cex_pair(self):
        '''
        Important function that returns a pair of a clause and a counterexample
        Returns:
            cex_clause_pair_list_prop: list of [sat_model, clauses_inv, output?] pairs that block the property
            cex_clause_pair_list_ind: None if cex exists
            is_inductive: check the cex is inductive (fullfil the  `T /\ P /\ not(P_prime)`)
            has_fewer_clauses: the clauses in inv is not used up
        '''
        prop = z3.Not(self.aagmodel.output)
        clause_list = []
        cex_clause_pair_list_prop = []
        cex_clause_pair_list_ind = []

        '''
        Everytime only the property is considered to generate the counterexample
        '''
        # define the number of cex that want to generate
        num_cex = 10
        cex, cex_m, var_lst = self._solve_relative(prop, clause_list, prop_only=True, generalize=self.generalize)
        # Backup the cex without generalization
        cex_without_generalization, cex_m_without_generalization, var_lst_without_generalization = \
            self._solve_relative(prop, clause_list, prop_only=False, generalize=False)
        while cex is not None and num_cex > 0:
            clause, clause_m = self._find_clause_to_block(cex,var_lst,generate_smt2=self.generate_smt2) # find the clause to block the cex
            # if check fail, use the un-generalized version of cex to double check
            if clause is None and clause_m is None: 
                clause, clause_m = self._find_clause_to_block(cex_without_generalization,var_lst_without_generalization,generate_smt2=self.generate_smt2, cex_double_check_without_generalization=True)
            assert clause is not None and clause_m is not None, "Unable to find a clause to block the cex"
            clause_list.append(clause) # add the clause (that has been added to solver for blocking) to the list
            # remove the duplicate clauses in the clause_list
            clause_list = list(set(clause_list))
            cex_prime_expr = self._make_cex_prime(cex) # find the cex expression in z3
            cex_clause_pair_list_prop.append((cex_m, clause_m, cex_prime_expr)) # model generated without using inv.cnf
            cex, cex_m, var_lst = self._solve_relative(prop, clause_list, prop_only=True, generalize=self.generalize)
            num_cex -= 1
        
        
        '''
        Everytime the clauses in clause_list and safety property are considered to generate the counterexample
        which is used to check c -> s (c & !s is unsat) 
        but this is not necessary!
        '''
        # define the number of cex that want to generate
        num_cex = 10
        cex, cex_m, var_lst = self._solve_relative(prop, clause_list, prop_only=False, generalize=self.generalize)
        # Backup the cex without generalization
        cex_without_generalization, cex_m_without_generalization, var_lst_without_generalization = \
            self._solve_relative(prop, clause_list, prop_only=False, generalize=False)
        while cex is not None and num_cex > 0:
            clause, clause_m = self._find_clause_to_block(cex,var_lst,generate_smt2=self.generate_smt2)
            # if check fail, use the un-generalized version of cex to double check
            if clause is None and clause_m is None: 
                clause, clause_m = self._find_clause_to_block(cex_without_generalization,var_lst_without_generalization,generate_smt2=self.generate_smt2, cex_double_check_without_generalization=True)
            assert clause is not None and clause_m is not None, "Unable to find a clause to block the cex"
            clause_list.append(clause)
            cex_prime_expr = self._make_cex_prime(cex)
            cex_clause_pair_list_ind.append((cex_m, clause_m, cex_prime_expr)) # model generated with using inv.cnf
            cex, cex_m, var_lst = self._solve_relative(prop, clause_list, prop_only=False, generalize=self.generalize)
            num_cex -= 1

        is_inductive = self._check_inductive(clause_list)
        has_fewer_clauses = len(clause_list) < len(self.clauses)
        self._export_ground_truth() # export the ground truth to .csv table
        return cex_clause_pair_list_prop, cex_clause_pair_list_ind, is_inductive, has_fewer_clauses

    # now, we can start to build the graphs
    #TODO: Map the CTI list to smt2 (sequence should be defined in CTI.txt)
