from z3 import *

class BMC:
    def __init__(self, primary_inputs, literals, primes, init, trans, post, pv2next, primes_inp):
        '''
        :param primary_inputs:
        :param literals: Boolean Variables
        :param primes: The Post Condition Variable
        :param init: The initial State
        :param trans: Transition Function
        :param post: The Safety Property
        '''
        self.primary_inputs = primary_inputs
        self.init = init
        self.trans = trans
        #self.literals = literals 
        self.literals = literals + primary_inputs
        self.items = self.primary_inputs + self.literals
        self.inp_prime = primes_inp
        self.lMap = {str(l): l for l in self.items}
        self.post = post
        self.frames = list()
        #self.primes = primes
        self.primes = primes + [Bool(str(pi)+'_prime') for pi in primary_inputs]
        #self.primeMap = [(literals[i], primes[i]) for i in range(len(literals))]
        self.primeMap = [(self.literals[i], self.primes[i]) for i in range(len(self.literals))]
        self.pv2next = pv2next
        self.initprime = substitute(self.init.cube(), self.primeMap)
        self.inp_map = [(primary_inputs[i], primes_inp[i]) for i in range(len(primes_inp))]
        self.vardict = dict()
        #entend literals and prime
        for item in self.inp_map:
            literals.append(item[0])
            primes.append(item[1])

    def vardef(self, n:str):
        if n in self.vardict:
            return self.vardict[n]
        v = Bool(n)
        self.vardict[n] = v
        return v

    def setup(self):
        self.slv = Solver()
        initmap = [(self.literals[i], self.vardef(str(self.literals[i])+"_0")) for i in range(len(self.literals))]
        self.slv.add(substitute(self.init.cube(), initmap))
        self.cnt = 0
    
    def get_map(self, idx):
        curr_map = [(self.literals[i], self.vardef(str(self.literals[i])+"_"+str(idx))) for i in range(len(self.literals))]
        next_map = [(self.primes[i], self.vardef(str(self.literals[i])+"_"+str(idx+1))) for i in range(len(self.literals))]
        return curr_map + next_map
    
    def unroll(self):
        idx = self.cnt
        var_map = self.get_map(idx)
        self.slv.add( substitute(self.trans.cube(), var_map) )
        self.cnt += 1
    
    def add(self, constraint):
        idx = self.cnt
        var_map = self.get_map(idx)
        self.slv.add( substitute(constraint, var_map) )
    
    def check(self):
        return self.slv.check()
        
        
        
        
