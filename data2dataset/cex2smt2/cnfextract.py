# find the last clauses that block ...
from numpy import var
import z3
import pandas as pd
import os
import ternary_sim
from tCube import tCube

class ExtractCnf(object):
    def __init__(self, aagmodel, clause, name):
        
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

        self.init = aagmodel.init
        self.lMap = {str(v):v for v in aagmodel.svars}
        self.Inp_SVars_Map = {str(l): l for l in aagmodel.inputs + aagmodel.svars + aagmodel.primed_inputs + aagmodel.primed_svars}

        self.model_name = name
        self.cex_generalization_ground_truth = []

        # initialize the ternary simulator
        self.ternary_simulator = ternary_sim.AIGBuffer()
        for _, updatefun in self.vprime2nxt:
            self.ternary_simulator.register_expr(updatefun)
        
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

    def _find_clause_to_block(self, model_to_block, model_var_lst, generate_smt2=False):
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
        print (model_to_block)
        for idx in range(len(self.clauses)-1, -1, -1): 
            print(idx,':', self.clauses[idx])
        assert False, "BUG: cannot find clause to block bad state"


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
        # assert the sequence of the ground truth has been sorted
        assert list(data.keys()) == sorted(list(data.keys())), "BUG: the sequence of the ground truth has not been sorted automatically"
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

    def _solve_relative(self, prop, cnfs, prop_only, generalize=False): # -> a dict of only the curr_vars
        '''
        prop: the property to be checked
        cnfs: the clauses has been added for blocking
        prop_only: consider the clauses in cnfs ? prev = (cnfs + prop) : prev= prop
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
        print("size of CTI after removing according to unsat core: ", size_after_unsat_core)

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
        cex, cex_m, var_lst = self._solve_relative(prop, clause_list, prop_only=True, generalize=True)
        while cex is not None:
            clause, clause_m = self._find_clause_to_block(cex,var_lst,generate_smt2=True) # find the clause to block the cex
            clause_list.append(clause) # add the clause (that has been added to solver for blocking) to the list
            # remove the duplicate clauses in the clause_list
            clause_list = list(set(clause_list))
            cex_prime_expr = self._make_cex_prime(cex) # find the cex expression in z3
            cex_clause_pair_list_prop.append((cex_m, clause_m, cex_prime_expr)) # model generated without using inv.cnf
            cex, cex_m, var_lst = self._solve_relative(prop, clause_list, prop_only=True, generalize=True)

        '''
        Everytime the clauses in clause_list and safety property are considered to generate the counterexample
        which is used to check c -> s (c & !s is unsat) 
        but this is not necessary!

        cex, cex_m, var_lst = self._solve_relative(prop, clause_list, prop_only=False)
        while cex is not None:
            clause, clause_m = self._find_clause_to_block(cex,var_lst,generate_smt2=True)
            clause_list.append(clause)
            cex_prime_expr = self._make_cex_prime(cex)
            cex_clause_pair_list_ind.append((cex_m, clause_m, cex_prime_expr)) # model generated with using inv.cnf
            cex, cex_m, var_lst = self._solve_relative(prop, clause_list, prop_only=False)
        '''

        is_inductive = self._check_inductive(clause_list)
        has_fewer_clauses = len(clause_list) < len(self.clauses)
        self._export_ground_truth() # export the ground truth to .csv table
        return cex_clause_pair_list_prop, cex_clause_pair_list_ind, is_inductive, has_fewer_clauses

    # now, we can start to build the graphs
    #TODO: Map the CTI list to smt2 (sequence should be defined in CTI.txt)