from z3 import *
import time
import random
import argparse

class z3_workaround:
    def __init__(self, decls, assertions, seed, old_solver=None):
        self.assertions = assertions
        self.decls = decls
        #self.solver_ctx = Context()
        #self.solver = Solver(ctx=self.solver_ctx)
        self.solver = old_solver
        self.solver.set("random_seed", seed)
        self.solver.set("phase_selection", 5)
        #print("seed is ", seed)

    def check(self):
        decl_vars = {x.name(): Bool(x.name()) for x in self.decls}
        decl_vars_sorted = sorted(decl_vars.keys())

        asts = list(self.assertions)
        asts.sort(key=str)

        for ast in asts:
            '''
            This list is then unpacked using the * operator, 
            which means that the elements of the list will be passed as separate arguments to the substitute_vars() function.
            '''
            ast_substituted = substitute_vars(ast, *[decl_vars[var_name] for var_name in decl_vars_sorted])
            self.solver.add(ast_substituted)

        self.solver.push()
        return self.solver.check()

    def model(self):
        return self.solver.model()
    
def all_smt(s, initial_terms):
    def block_term(s, m, t):
        s.add(t != m.eval(t))
    def fix_term(s, m, t):
        s.add(t == m.eval(t))
    def all_smt_rec(terms):
        if sat == s.check():
           seed = random.randint(0, 2 ** 30)
           _solution = s.model()
           zw = z3_workaround(_solution.decls(), s.assertions(), seed,s)
           zw.check()
           m = zw.model()
           #m = s.model()
           yield m
           for i in range(len(terms)):
               s.push()
               block_term(s, m, terms[i])
               for j in range(i):
                   fix_term(s, m, terms[j])
               yield from all_smt_rec(terms[i:])
               s.pop()   
    yield from all_smt_rec(list(initial_terms))  

def get_variables(expr):
    variables = set()
    stack = [expr]

    while stack:
        current_expr = stack.pop() #if pop(0) then it is BFS (Breadth First Search

        if is_const(current_expr) and current_expr.decl().kind() == Z3_OP_UNINTERPRETED:
            variables.add(current_expr)
        else:
            stack.extend(current_expr.children())

    return variables

if __name__ == "__main__":
    # add parser
    parser = argparse.ArgumentParser(description="Randomly generate uniform distribution of models")
    #parse smt2 file
    parser.add_argument('--smt2-file', type=str, default=None, help='The path to the smt2 file')
    args = parser.parse_args(['--smt2-file', '/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/data2dataset/cex2smt2/test_model_sampler/test.smt2'])
    
    start_time = time.time()
    
    # use z3 to parse smt2 file
    s = Solver()
    # get all terms in the smt2 file and store them in a list
    parsed_exprs = parse_smt2_file(args.smt2_file)
    s.add(parsed_exprs)
    all_variables = []
    for expr in parsed_exprs: all_variables.extend(get_variables(expr))
    #print(all_variables)
    #print(s)

    # Remove duplicates
    all_variables = list(set(all_variables))

    models = list(all_smt(s,all_variables))
    print("All models with random generated seeds:")
    print(models)
    for m in models: print(m)
    print(time.time()-start_time)