# find the last clauses that block ...
import z3

class ExtractCnf(object):
    def __init__(self, aagmodel, clause):
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

    def _find_clause_to_block(self, model_to_block):
        for idx in range(len(self.clauses)-1, -1, -1): # search backwards
            cl = self.clauses[idx]
            slv = z3.Solver()
            slv.add(cl)
            slv.add(model_to_block)
            res = slv.check()
            if res == z3.unsat:
                return cl, self.clause_unexpanded[idx]
            assert res == z3.sat
        print (model_to_block)
        for idx in range(len(self.clauses)-1, -1, -1): 
            print(idx,':', self.clauses[idx])
        assert False, "BUG: cannot find clause to block bad state"

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

        if prop_only:
            post = prop
        else:
            post = prev
        
        not_p_prime = z3.Not(z3.substitute(z3.substitute(post, self.v2prime), self.vprime2nxt))
        slv.add(not_p_prime)
        res = slv.check()
        if res == z3.unsat:
            return None, None
        assert res == z3.sat
        model = slv.model()
        filtered_model, model_list = self._filter_model(model)
        return filtered_model, model_list

    def _filter_model(self, model): # keep only current state variables
        no_prime_or_input = [l for l in model if str(l)[0] != 'i' and not str(l).endswith('_prime')] #
        cex_expr = z3.And([z3.simplify(self.lMap[str(l)] == model[l]) for l in no_prime_or_input])
        cex_list = [ (self.aagmodel.svars.index(self.lMap[str(l)]), 1 if str(model[l]) == 'True' else -1) for l in no_prime_or_input ]

        return cex_expr, cex_list

    def get_clause_cex_pair(self):
        prop = z3.Not(self.aagmodel.output)  # not(bad)
        clause_list = []
        cex_clause_pair_list_prop = []
        cex_clause_pair_list_ind = []

        cex, cex_m = self._solve_relative(prop, clause_list, prop_only=True)
        while cex is not None:
            clause, clause_m = self._find_clause_to_block(cex)
            clause_list.append(clause)
            cex_prime_expr = self._make_cex_prime(cex)
            cex_clause_pair_list_prop.append((cex_m, clause_m, cex_prime_expr))
            cex, cex_m = self._solve_relative(prop, clause_list, prop_only=True)


        cex, cex_m = self._solve_relative(prop, clause_list, prop_only=False)
        while cex is not None:
            clause, clause_m = self._find_clause_to_block(cex)
            clause_list.append(clause)
            cex_prime_expr = self._make_cex_prime(cex)
            cex_clause_pair_list_ind.append((cex_m, clause_m, cex_prime_expr))
            cex, cex_m = self._solve_relative(prop, clause_list, prop_only=False)

        is_inductive = self._check_inductive(clause_list)
        has_fewer_clauses = len(clause_list) < len(self.clauses)
        return cex_clause_pair_list_prop, cex_clause_pair_list_ind, is_inductive, has_fewer_clauses

    # now, we can start to build the graphs
    
