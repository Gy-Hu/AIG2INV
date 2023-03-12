# -*- coding: UTF-8 -*- 
from logging import exception
from math import fabs
from z3 import *
from functools import wraps
import collections

def _extract(literaleq):
    # we require the input looks like v==val
    children = literaleq.children()
    assert(len(children) == 2)
    if str(children[0]) in ['True', 'False']:
        v = children[1]
        val = children[0]
    elif str(children[1]) in ['True', 'False']:
        v = children[0]
        val = children[1]
    else:
        assert(False)
    return v, val

class tCube:
    # make a tcube object assosciated with frame t.
    def __init__(self, t=0, cubeLiterals=None):
        self.t = t
        if cubeLiterals is None:
            self.cubeLiterals = list()
        else:
            self.cubeLiterals = cubeLiterals

    def __lt__(self, other):
        return self.t < other.t

    def clone(self):
        ret = tCube(self.t)
        ret.cubeLiterals = self.cubeLiterals.copy()
        return ret
    
    def clone_and_sort(self):
        ret = tCube(self.t)
        ret.cubeLiterals = self.cubeLiterals.copy()
        ret.cubeLiterals.sort(key=lambda x: str(_extract(x)[0]))
        return ret

    def remove_true(self):
        self.cubeLiterals = [c for c in self.cubeLiterals if c is not True]
    
    def __eq__(self, other) : 
        return collections.Counter(self.cubeLiterals) == collections.Counter(other.cubeLiterals)

    def addModel(self, lMap, model, remove_input, aig_path=None): # not remove input' when add model
        no_var_primes = [l for l in model if str(l)[0] == 'i' or not str(l).endswith('_prime')]# no_var_prime -> i2, i4, i6, i8, i2', i4', i6' or v2, v4, v6
        if remove_input:
            no_input = [l for l in no_var_primes if str(l)[0] != 'i'] # no_input -> v2, v4, v6
        else:
            no_input = no_var_primes # no_input -> i2, i4, i6, i8, i2', i4', i6' or v2, v4, v6
        # self.add(simplify(And([lMap[str(l)] == model[l] for l in no_input]))) # HZ:
        for l in no_input:
            # Avoid some cases that has "False==False or True==True" in model
            if (str(l)!= 'True' and str(l) != 'False'): 
                self.add(lMap[str(l)] == model[l]) #TODO: Get model overhead is too high, using C API
            elif aig_path is not None:
                # l -> True or l -> False, and aig_path is not None
                self._report2log_add_model(aig_path,\
                "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/log/error_handle/bad_model.log", \
                f"{aig_path} has bad model {str(l)}")
                print("Strange model, please check the log file.")
                
    def _report2log_add_model(self, aig_path, log_file, error_msg):
        # append the error message to the log file
        with open(log_file, "a+") as fout:
            fout.write(f"Error: {aig_path} has bad model. {error_msg}.\n")
        fout.close()

    def remove_input(self):
        index_to_remove = set()
        for idx, literal in enumerate(self.cubeLiterals):
            children = literal.children()
            assert(len(children) == 2)

            if str(children[0]) in ['True', 'False']:
                v = str(children[1])
            elif str(children[1]) in ['True', 'False']:
                v = str(children[0])
            else:
                assert(False)
            assert (v[0] in ['i', 'v'])
            if v[0] == 'i':
                index_to_remove.add(idx)
        self.cubeLiterals = [self.cubeLiterals[i] for i in range(len(self.cubeLiterals)) if i not in index_to_remove]

    def addAnds(self, ms):
        for i in ms:
            self.add(i)


    def add(self, m):
        self.cubeLiterals.append(m)

    def true_size(self):
        '''
        Remove the 'True' in list (not the BoolRef Variable)
        '''
        return len(self.cubeLiterals) - self.cubeLiterals.count(True) 

    def join(self,  model):
        # first extract var,val from cubeLiteral
        literal_idx_to_remove = set()
        model = {str(var): model[var] for var in model}
        for idx, literal in enumerate(self.cubeLiterals):
            if literal is True:
                continue
            var, val = _extract(literal)
            var = str(var)
            assert(var[0] == 'v')
            if var not in model:
                literal_idx_to_remove.add(idx)
                continue
            val2 = model[var]
            if str(val2) == str(val):
                continue # will not remove
            literal_idx_to_remove.add(idx)
        for idx in literal_idx_to_remove:
            self.cubeLiterals[idx] = True
        return len(literal_idx_to_remove) != 0

    def delete(self, i: int):
        res = tCube(self.t)
        for it, v in enumerate(self.cubeLiterals):
            if i == it:
                res.add(True)
                continue
            res.add(v)
        return res


    def cube(self): #导致速度变慢的罪魁祸首？
        return simplify(And(self.cubeLiterals))

    # Convert the trans into real cube
    def cube_remove_equal(self):
        res = tCube(self.t)
        for literal in self.cubeLiterals:
            children = literal.children()
            assert(len(children) == 2)
            cube_literal = And(Not(And(children[0],Not(children[1]))), Not(And(children[1],Not(children[0]))))
            res.add(cube_literal)
        return res

    def __repr__(self):
        return str(self.t) + ": " + str(sorted(self.cubeLiterals, key=str))