'''
Randomize the model from a given z3 context
'''

from z3 import *          
import random
                                                                                                                                                                                     
class z3_workaround:
    def __init__(self, decls, assertions, seed):
        self.assertions = assertions
        self.decls = decls
        self.solver_ctx = Context()
        self.solver = Solver(ctx=self.solver_ctx)
        self.solver.set("random_seed", seed)
        self.solver.set("phase_selection", 5)
        print("seed is ", seed)

    def check(self):
        decl_vars = [x.name() for x in self.decls]
        decl_vars.sort()
        for x in decl_vars:
            #print(f'{x} = BitVec(\'{x}\',32)')
            exec(f'{x} = Bool(\'{x}\', self.solver_ctx)')

        asts = [str(x) for x in self.assertions] 
        asts.sort() 
        for ast in asts:
            #print(f"self.solver.add({ast})")
            exec(f"self.solver.add({ast})")

        self.solver.push()
        return self.solver.check()

    def model(self):
        return self.solver.model()
    
if __name__ == "__main__":
    # generate a random seed
    #seed = random.randint(0, 2 ** 30)
    seed = rand_num = random.SystemRandom().randint(0, 100)
    a, b, c = Bools('a b c')
    prev_solver = Solver()
    #prev_solver.set("model.completion", True)
    prev_solver.add(Or(a, b, c))
    prev_solver.check()
    _solution = prev_solver.model()
    print(_solution.decls())
    zw= z3_workaround(_solution.decls(), prev_solver.assertions(), seed)
    zw.check()
    print(zw.model())