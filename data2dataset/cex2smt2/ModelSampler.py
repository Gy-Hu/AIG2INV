from z3 import *
import time
import random

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
        decl_vars = [x.name() for x in self.decls]
        decl_vars.sort()
        for x in decl_vars:
            #print(f'{x} = BitVec(\'{x}\',32)')
            exec(f'{x} = Bool(\'{x}\')')

        asts = [str(x) for x in self.assertions] 
        asts.sort() 
        for ast in asts:
            #print(f"self.solver.add({ast})")
            exec(f"self.solver.add({ast})")

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

if __name__ == "__main__":
    start_time = time.time()
    a, b, c = Bools('a b c')
    s = Solver()
    s.add(Or(a, b, c))
    print("Original constraints:")
    print(s)
    models = list(all_smt(s,[a,b,c]))
    print("All models with random generated seeds:")
    #print(models)
    for m in models:
        print(m)
    print(time.time()-start_time)