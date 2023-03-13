from ast import Constant
from cmath import exp
from lib2to3.pgen2.pgen import generate_grammar
from unittest import expectedFailure
from xml.dom import registerDOMImplementation
import sympy as sp
from sympy.logic.boolalg import true, false
import z3
from functools import reduce
import pycparser.c_ast as ast
from functools import lru_cache
import multiprocessing as mp
import concurrent.futures
max_workers = mp.cpu_count()
from pysmt.shortcuts import Plus, Pow, Real, Symbol, Times, serialize, And, Or, Not
from pysmt.typing import REAL
from copy import deepcopy

# z3 <-> sympy

'''
Currently the best way is to use parallel z3_to_sympy function, and choose one non-parallel sympy_to_z3 function.
sympy to z3 is really hard to calculate in parallel, due to the context of z3 should not be mismatched.
Thus, the better way to improve performace is using other method than using parallel computing.

--------------------------------------------------sympy to z3--------------------------------------------------
From https://stackoverflow.com/questions/75461163/python-how-to-parse-boolean-sympy-tree-expressions-to-boolean-z3py-expressions
compile_to_z3: Simple map from sympy to z3
compile_to_z3_parallel: Parallel map from sympy to z3 -> contains bug when pickling z3 objects
compile_to_z3_parallel_2: Parallel converter from sympy to z3 -> contains bug due to mismatched context
cvt_parallel: Function that used in compile_to_z3_parallel

From https://github.com/psy054duck/c2c/blob/5135209606bf054d370908a4c8c198991f07d3fb/utils.py
to_z3: Converter from sympy to z3 -> use bfs, which is not efficient
to_z3_parallel: Parallel converter from sympy to z3 -> contains bug when pickling z3 objects

---------------------------------------------------z3 to sympy--------------------------------------------------
to_sympy: Converter from z3 to sympy -> single thread, not efficient
to_sympy_parallel: Parallel converter from z3 to sympy -> Bug fixed. This works! But sometimes may occur bug, due to memory model?

'''


'''
-------------------------------Use to "No Bugs" label function !--------------------------------------
-------------------------------"WIP" label function is still under development.-----------------------
'''


'''
---------
WIP?    |
---------
'''
def z3_deep_simplify(expr):
    # print(expr)
    sim = z3.Tactic('ctx-solver-simplify')
    # sim = z3.Repeat(z3.Then('propagate-ineqs', 'ctx-solver-simplify'))
    simplify_res = sim(expr)
    cond_list = list(simplify_res[0])
    # print(cond_list)
    new_cond_list = []
    for cond in cond_list:
        sp_cond = to_sympy(cond)
        if len(sp_cond.free_symbols) == 1:
            t = to_z3(list(sp_cond.free_symbols)[0])
            s = z3.Solver()
            z3_min, z3_max = z3.Ints('z3_min, z3_max')
            s.push()
            s.add(z3.ForAll(t, z3.Implies(z3.And(t <= z3_max), cond)))
            s.add(z3.ForAll(t, z3.Implies(cond, z3.And(t <= z3_max))))
            if s.check() == z3.sat:
                model = s.model()
                new_cond_list.append(t <= model.eval(z3_max))
                continue
            s.pop()
            s.push()
            s.add(z3.ForAll(t, z3.Implies(z3.And(t >= z3_min), cond)))
            s.add(z3.ForAll(t, z3.Implies(cond, z3.And(t >= z3_min))))
            if s.check() == z3.sat:
                model = s.model()
                new_cond_list.append(t >= model.eval(z3_min))
                continue
            s.pop()
            s.push()
            s.add(z3.ForAll(t, z3.Implies(z3.And(z3_min <= t, t <= z3_max), cond)))
            s.add(z3.ForAll(t, z3.Implies(cond, z3.And(z3_min <= t, t <= z3_max))))
            if s.check() == z3.sat:
                model = s.model()
                new_cond_list.append(z3.And(model.eval(z3_min) <= t, t <= model.eval(z3_max)))
                continue
        new_cond_list.append(to_z3(sp_cond))
    cond_list = new_cond_list

    if len(cond_list) == 0:
        return z3.BoolVal(True)
    elif len(cond_list) == 1:
        return cond_list[0]
    else:
        return z3.And(*[z3_deep_simplify(cond) for cond in cond_list])

'''
---------
No Bugs |
---------
'''
@lru_cache(maxsize=None)
def to_z3(sp_expr):
    #self = sp.simplify(sp_expr)
    self = sp.factor(sp_expr)
    if isinstance(self, sp.Symbol):
        res = z3.Bool(str(self))
    elif self is true:
        res = z3.BoolVal(True)
    elif self is false:
        res = z3.BoolVal(False)
    elif isinstance(self, sp.Not):
        res = z3.Not(to_z3(self.args[0]))
    elif isinstance(self, (sp.And, sp.Or)):
        args = [to_z3(arg) for arg in self.args]
        if isinstance(self, sp.And):
            res = z3.And(*args)
        else:
            res = z3.Or(*args)
    else:
        raise Exception('Conversion for "%s" has not been implemented yet: %s' % (type(self), self))
    #return z3.simplify(res)
    return res

'''
---------
No Bugs |
---------
'''
@lru_cache(maxsize=None)
def to_sympy(expr,t=None):
    # check whether the expression is a sympy expression
    if isinstance(expr, sp.Expr):
        return expr
    
    if z3.is_const(expr):
        if z3.is_bool(expr):
            res = sp.Symbol(str(expr))
        else:
            res = sp.Symbol(str(expr), integer=True)
    elif z3.is_app_of(expr, z3.Z3_OP_NOT):
        res = sp.Not(to_sympy(expr.arg(0)))
    elif z3.is_app_of(expr, z3.Z3_OP_AND) or z3.is_app_of(expr, z3.Z3_OP_OR):
        op = expr.decl().name()
        children = [to_sympy(child) for child in expr.children()]
        if op == 'and':
            res = sp.And(*children)
        elif op == 'or':
            res = sp.Or(*children)
        else:
            raise Exception('Conversion for "%s" has not been implemented yet: %s' % (op, expr))
    else:
        raise Exception('conversion for type "%s" is not implemented: %s' % (type(expr), expr))
    #return sp.simplify(res)
    return res

'''
---------
   WIP  |
---------
'''
@lru_cache(maxsize=None)
def to_z3_parallel(sp_expr):
    #self = sp.factor(sp_expr)
    self = sp.simplify(sp_expr)

    if isinstance(self, sp.Symbol):
        res = z3.Bool(str(self))
    elif self is true:
        res = z3.BoolVal(True)
    elif self is false:
        res = z3.BoolVal(False)
    elif isinstance(self, sp.Not):
        res = z3.Not(to_z3_parallel(self.args[0]))
    elif isinstance(self, (sp.And, sp.Or)):
        args = self.args
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_args = [executor.submit(to_z3_parallel, arg) for arg in args]
            args = [f.result() for f in concurrent.futures.as_completed(future_args)]
        if isinstance(self, sp.And):
            res = z3.And(*args)
        else:
            res = z3.Or(*args)
    else:
        raise Exception('Conversion for "%s" has not been implemented yet: %s' % (type(self), self))
    return res

'''
---------
No Bugs |
---------
'''
#@lru_cache(maxsize=None)
def to_sympy_parallel(expr, t=None):
    # check whether the expression is a sympy expression
    if isinstance(expr, sp.Expr):
        return expr
    
    if z3.is_const(expr):
        if z3.is_bool(expr):
            res = sp.Symbol(str(expr))
        else:
            res = sp.Symbol(str(expr), integer=True)
    elif z3.is_app_of(expr, z3.Z3_OP_NOT):
        res = sp.Not(to_sympy_parallel(expr.arg(0)))
    elif z3.is_app_of(expr, z3.Z3_OP_AND) or z3.is_app_of(expr, z3.Z3_OP_OR):
        op = expr.decl().name()
        children = expr.children()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_args = [executor.submit(to_sympy_parallel, arg) for arg in children]
            children = [f.result() for f in concurrent.futures.as_completed(future_args)]
        if op == 'and':
            res = sp.And(*children)
        elif op == 'or':
            res = sp.Or(*children)
        else:
            raise Exception('Conversion for "%s" has not been implemented yet: %s' % (op, expr))
    else:
        raise Exception('conversion for type "%s" is not implemented: %s' % (type(expr), expr))
    #XXX: Double check before running the script
    #return sp.simplify(res)
    return res

        
# sympy to z3 -> new version from: 
# https://stackoverflow.com/questions/75461163/python-how-to-parse-boolean-sympy-tree-expressions-to-boolean-z3py-expressions
# define cvt outside of compile_to_z3
'''
---------
   WIP  |
---------
'''
def cvt_parallel(expr, pvs, constants, table):
    if expr in pvs:
        return str(pvs[expr])  # convert Z3 object to string

    texpr = type(expr)
    if texpr in constants:
        return str(constants[texpr])  # convert Z3 object to string

    if texpr in table:
        return (table[texpr])(*[cvt_parallel(arg, pvs, constants, table) for arg in expr.args])

    raise NameError("Unimplemented: " + str(expr))

'''
---------
   WIP  |
---------
'''
@lru_cache(maxsize=None)
def compile_to_z3_parallel(exp):
    """Compile sympy expression to z3"""
    
    # Sympy vs Z3. Add more correspondences as necessary!
    table = { sp.logic.boolalg.And    : z3.And
            , sp.logic.boolalg.Or     : z3.Or
            , sp.logic.boolalg.Not    : z3.Not
            , sp.logic.boolalg.Implies: z3.Implies
            }

    # Sympy vs Z3 Constants
    constants = { sp.logic.boolalg.BooleanTrue : z3.BoolVal(True)
                , sp.logic.boolalg.BooleanFalse: z3.BoolVal(False)
                }
    
    # if exp is str, parse it
    if(isinstance(exp, str)):
        pexp = sp.parsing.sympy_parser.parse_expr(exp)
    else:
        pexp = exp
        
    # create Z3 boolean variables using z3.BoolVar
    pvs  = {v: z3.Bool(str(v)) for v in pexp.atoms() if type(v) not in constants}

    # use multiprocessing to run cvt in parallel
    pool = mp.Pool(processes=mp.cpu_count())
    results = [pool.apply_async(cvt_parallel, (expr, pvs, constants, table)) for expr in pexp.args] if len(pexp.args) > 1 else [pool.apply_async(cvt_parallel, (pexp, pvs, constants, table))]
    output = [res.get() for res in results]

    #return cvt(pexp) # non-parallel
    return table[type(pexp)](*output) # parallel

'''
---------
   WIP  |
---------
'''
def compile_to_z3_parallel_2(exp, memo={}):
    # Sympy vs Z3. Add more correspondences as necessary!
    table = { sp.logic.boolalg.And    : z3.And
            , sp.logic.boolalg.Or     : z3.Or
            , sp.logic.boolalg.Not    : z3.Not
            , sp.logic.boolalg.Implies: z3.Implies
            }

    # Sympy vs Z3 Constants
    constants = { sp.logic.boolalg.BooleanTrue : z3.BoolVal(True)
                , sp.logic.boolalg.BooleanFalse: z3.BoolVal(False)
                }
    
    """Compile sympy expression to z3"""
    # if exp is str, parse it
    if(isinstance(exp, str)):
        pexp = sp.parsing.sympy_parser.parse_expr(exp)
    else:
        pexp = exp
        
    # simplify expression by using the simplify function from sympy
    pexp = sp.simplify(pexp)
    
    # use memoization to cache the results of the function
    # Check if expression has already been computed
    if pexp in memo: return memo[pexp]
    
    # Step 1: Compute variables in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        variables = list(pexp.atoms())
        variables_filtered = [v for v in variables if type(v) not in constants]
        # create a new context i_context for the copied expression, i equals to the thread id
        def copy_var(v):
            ctx = z3.Context()
            return (v, z3.Bool(str(v), ctx=ctx), ctx)
        variables_z3 = list(executor.map(copy_var, variables_filtered))
        pvs = dict([(v, z) for v, z, c in variables_z3])
    
    # Step 2: Convert sub-expressions to Z3 expressions in parallel
    def cvt(expr, i_context):
        i_context = z3.Context()
        if expr in pvs:
            return pvs[expr]

        texpr = type(expr)
        if texpr in constants:
            return constants[texpr]

        if texpr in table:
            #return table[texpr](*map(cvt, expr.args)) # if not using memoization
            result = table[texpr](*map(cvt, deepcopy(expr.args), [i_context]*len(expr.args)))
            memo[expr] = result  # Store computed result
            return result

        raise NameError("Unimplemented: " + str(expr))

    result = cvt(pexp, None)
    memo[pexp] = result  # Store computed result
    
    #return cvt(pexp) # if not using memoization
    return result

'''
---------
No Bugs |
---------
'''
def compile_to_z3(exp, memo={}):
    # Sympy vs Z3. Add more correspondences as necessary!
    table = { sp.logic.boolalg.And    : z3.And
            , sp.logic.boolalg.Or     : z3.Or
            , sp.logic.boolalg.Not    : z3.Not
            , sp.logic.boolalg.Implies: z3.Implies
            }

    # Sympy vs Z3 Constants
    constants = { sp.logic.boolalg.BooleanTrue : z3.BoolVal(True)
                , sp.logic.boolalg.BooleanFalse: z3.BoolVal(False)
                }
    
    """Compile sympy expression to z3"""
    # if exp is str, parse it
    if(isinstance(exp, str)):
        pexp = sp.parsing.sympy_parser.parse_expr(exp)
    else:
        pexp = exp
        
    # simplify expression by using the simplify function from sympy
    pexp = sp.simplify(pexp)
    
    # use memoization to cache the results of the function
    # Check if expression has already been computed
    if pexp in memo: return memo[pexp]
    
    pvs  = {v: z3.Bool(str(v)) for v in pexp.atoms() if type(v) not in constants}

    def cvt(expr):
        if expr in pvs:
            return pvs[expr]

        texpr = type(expr)
        if texpr in constants:
            return constants[texpr]

        if texpr in table:
            #return table[texpr](*map(cvt, expr.args)) # if not using memoization
            result = table[texpr](*map(cvt, expr.args))
            memo[expr] = result  # Store computed result
            return result

        raise NameError("Unimplemented: " + str(expr))

    result = cvt(pexp)
    memo[pexp] = result  # Store computed result
    
    #return cvt(pexp) # if not using memoization
    return result

if __name__ == '__main__':
    a = z3.Bool('a')
    b = z3.Bool('b')
    c = z3.Not(z3.And(z3.Not(a), z3.Not(b)))
    #print(z3.simplify(compile_to_z3(to_sympy_parallel(c))))
    #print(sympy2pysmt((to_sympy_parallel(c))))
    print(z3.simplify(compile_to_z3_parallel_2(to_sympy_parallel(c))))