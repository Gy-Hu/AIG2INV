# find the last clauses that block ...
from numpy import var
import z3

class ExtractCnf(object):
    def __init__(self, aagmodel, clause, name):
        # build clauses
        self.aagmodel = aagmodel
        self.clause_unexpanded = clause.clauses
        self.clauses = self._build_clauses(aagmodel.svars, clause.clauses)

        assert len(aagmodel.inputs) == len(aagmodel.primed_inputs)
        assert len(aagmodel.svars) == len(aagmodel.primed_svars)
        self.v2prime = dict(list(zip(aagmodel.inputs, aagmodel.primed_inputs )) + list(zip(aagmodel.svars, aagmodel.primed_svars)))
        # latch2next
        self.vprime2nxt = {self.v2prime[v]:expr for v, expr in aagmodel.latch2next.items()}
        
        self.v2prime = list(self.v2prime.items())
        self.vprime2nxt = list(self.vprime2nxt.items())
        
        self.lMap = {str(v):v for v in aagmodel.svars}

        self.model_name = name
        self.cex_generalization_ground_truth = []

    def _build_clauses(self, svars, clauses):
        ret_clauses = []
        for cl in clauses:
            cl_z3 = []
            for var,sign in cl:
                lit = svars[var]
                if sign == -1:
                    lit = z3.Not(lit)
                cl_z3.append(lit)
            ret_clauses.append(z3.Not(z3.And(cl_z3)))
        return ret_clauses

    def _find_clause_to_block(self, model_to_block, model_var_lst, generate_smt2=False):
        for idx in range(len(self.clauses)-1, -1, -1): # search backwards, performas like MIC
            cl = self.clauses[idx]
            slv = z3.Solver()
            slv.add(cl)
            slv.add(model_to_block)
            res = slv.check()
            if res == z3.unsat: # find the subset (which can be generalized)
                if generate_smt2: self._generate_smt2_and_ground_truth(model_to_block, model_var_lst, (cl, self.clause_unexpanded[idx]))
                return cl, self.clause_unexpanded[idx]
            assert res == z3.sat
        print (model_to_block)
        for idx in range(len(self.clauses)-1, -1, -1): 
            print(idx,':', self.clauses[idx])
        assert False, "BUG: cannot find clause to block bad state"

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

    def _generate_smt2_and_ground_truth(self, model_to_block, model_var_lst, cl_info):    #smt2_gen_IG is a switch to trun on/off .smt file generation 
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
        # assert the length of model_var_lst is the same as the length of aagmodel.svars
        for literals in model_var_lst: s_smt.add(literals)
        s_smt.add(Cube)
        filename = '../../dataset/cex2graph/smt2_from_solver/' + (f'{self.model_name}' + f'_{len(self.cex_generalization_ground_truth)}.smt2')
        data = {'inductive_check': filename.split('/')[-1]}
        with open(filename, mode='w') as f: f.write(s_smt.to_smt2())
        f.close() 

        '''
        ---------------------Export the ground truth----------------------
        '''
        q_minimum = cl_info[0] # Minimum ground truth has been generated
        for idx in range(len(q.cubeLiterals)): # -> ground truth size is q
            var, val = self._extract(q.cubeLiterals[idx])
            data[str(var)] = 0
        for idx in range(len(q_minimum.cubeLiterals)):
            var, val = self._extract(q_minimum.cubeLiterals[idx])
            data[str(var)] = 1 # Mark q-like as 1
        self.cex_generalization_ground_truth.append(data)
        return q_minimum

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

    def _solve_relative(self, prop, cnfs, prop_only): # -> a dict of only the curr_vars
        # initially, we will remove the props
        # then, maybe we can also remove the additional states

        # model of `T /\ P /\ not(P_prime)`
        slv = z3.Solver()
        prev = z3.And(cnfs+[prop])
        slv.add(prev)

        post = prop if prop_only else prev
        not_p_prime = z3.Not(z3.substitute(z3.substitute(post, self.v2prime), self.vprime2nxt))
        slv.add(not_p_prime)
        res = slv.check()
        if res == z3.unsat:
            return None, None
        assert res == z3.sat
        model = slv.model()
        filtered_model, model_list, var_list = self._filter_model(model)
        return filtered_model, model_list, var_list # filtered model is a z3 expression, model_list is [(var_index, sign)], var_lst is [vx == T, vx == F...]

    def _filter_model(self, model): # keep only current state variables
        no_prime_or_input = [l for l in model if str(l)[0] != 'i' and not str(l).endswith('_prime')] #
        cex_expr = z3.And([z3.simplify(self.lMap[str(l)] == model[l]) for l in no_prime_or_input])
        cex_list = [ (self.aagmodel.svars.index(self.lMap[str(l)]), 1 if str(model[l]) == 'True' else -1) for l in no_prime_or_input ]
        var_list = [(self.lMap[str(l)] == model[l]) for l in no_prime_or_input]
        return cex_expr, cex_list, var_list

    def get_clause_cex_pair(self):
        '''
        Important function that returns a pair of a clause and a counterexample
        Returns:
            cex_clause_pair_list_prop: list of [sat_model, clauses_inv, output?] pairs that block the property
            cex_clause_pair_list_ind: None if cex exists
            is_inductive: check the cex is inductive (fullfil the  `T /\ P /\ not(P_prime)`)
            has_fewer_clauses: the clauses in inv is not used up
        '''
        prop = z3.Not(self.aagmodel.output)  # not(bad)
        clause_list = []
        cex_clause_pair_list_prop = []
        cex_clause_pair_list_ind = []

        cex, cex_m, var_lst = self._solve_relative(prop, clause_list, prop_only=True)
        while cex is not None:
            clause, clause_m = self._find_clause_to_block(cex,var_lst,generate_smt2=True) # find the clause to block the cex
            clause_list.append(clause) # add the clause (that has been added to solver for blocking) to the list
            cex_prime_expr = self._make_cex_prime(cex) # find the cex expression in z3
            cex_clause_pair_list_prop.append((cex_m, clause_m, cex_prime_expr))
            cex, cex_m, var_lst = self._solve_relative(prop, clause_list, prop_only=True)


        cex, cex_m, var_lst = self._solve_relative(prop, clause_list, prop_only=False)
        while cex is not None:
            clause, clause_m = self._find_clause_to_block(cex,var_lst)
            clause_list.append(clause)
            cex_prime_expr = self._make_cex_prime(cex)
            cex_clause_pair_list_ind.append((cex_m, clause_m, cex_prime_expr))
            cex, cex_m, var_lst = self._solve_relative(prop, clause_list, prop_only=False)

        is_inductive = self._check_inductive(clause_list)
        has_fewer_clauses = len(clause_list) < len(self.clauses)
        return cex_clause_pair_list_prop, cex_clause_pair_list_ind, is_inductive, has_fewer_clauses

    # now, we can start to build the graphs
    
