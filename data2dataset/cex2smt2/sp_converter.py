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

# z3 <-> sympy

'''
From https://stackoverflow.com/questions/75461163/python-how-to-parse-boolean-sympy-tree-expressions-to-boolean-z3py-expressions

compile_to_z3: Simple map from sympy to z3
compile_to_z3_parallel: Parallel map from sympy to z3 -> contains bug when pickling z3 objects
cvt_parallel: Function that used in compile_to_z3_parallel

From https://github.com/psy054duck/c2c/blob/5135209606bf054d370908a4c8c198991f07d3fb/utils.py
to_z3: Converter from sympy to z3
to_sympy: Converter from z3 to sympy
to_z3_parallel: Parallel converter from sympy to z3 -> contains bug when pickling z3 objects
to_sympy_parallel: Parallel converter from z3 to sympy -> Bug fixed.

'''


def check_conditions_consistency(conditions):
    s = z3.Solver()
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            if i >= j: continue
            try:
                res = s.check(to_z3(sp.And(cond1, cond2)))
            except:
                res = s.check(z3.And(cond1, cond2))
            print(res)
            if res == z3.sat:
                print(cond1)
                print(cond2)
                print(s.model())
    print('*'*100)

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

@lru_cache(maxsize=None)
def to_sympy(expr):
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


#@lru_cache(maxsize=None)
def to_sympy_parallel(expr):
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
    #return sp.simplify(res)
    return res

def get_app_by_var(var, expr):
    '''for sympy'''
    if expr.func is var:
        return expr
    for arg in expr.args:
        res = get_app_by_var(var, arg)
        if res is not None:
            return res
    return None

def expr2c(expr: sp.Expr):
    if isinstance(expr, sp.Add):
        # assert(len(expr.args) == 2)
        res = ast.BinaryOp('+', expr2c(expr.args[0]), expr2c(expr.args[1]))
        for i in range(2, len(expr.args)):
            res = ast.BinaryOp('+', res, expr2c(expr.args[i]))
    elif isinstance(expr, sp.Mul):
        # print(expr)
        # print(expr.args)
        # assert(len(expr.args) == 2)
        res = ast.BinaryOp('*', expr2c(expr.args[0]), expr2c(expr.args[1]))
        for i in range(2, len(expr.args)):
            res = ast.BinaryOp('*', res, expr2c(expr.args[i]))
    elif isinstance(expr, sp.Integer) or isinstance(expr, int):
        res = ast.Constant('int', str(expr))
    elif expr.is_Function:
        # assert(len(expr.args) == 1)
        # arg = expr.args[0]
        array_ref = ast.ArrayRef(ast.ID(str(expr.func)), expr2c(expr.args[0]))
        for i in range(1, len(expr.args)):
            array_ref = ast.ArrayRef(array_ref, expr2c(expr.args[i]))
        res = array_ref
        # res = ast.ArrayRef(ast.ID(str(expr.func)), *[expr2c(arg) for arg in expr.args])
    elif isinstance(expr, sp.Symbol):
        res = ast.ID(str(expr))
    else:
        raise Exception('conversion for type "%s" is not implemented: %s' % (type(expr), expr))
    return res

def compute_N(cond, closed_form):
    ind_var = closed_form.ind_var
    z3_cond = to_z3(cond.subs(closed_form.to_sympy()))
    ind_var_z3 = to_z3(ind_var)
    N = z3.Int('N')
    e1 = z3.ForAll(ind_var_z3, z3.Implies(z3.And(0 <= ind_var_z3, ind_var_z3 < N), z3_cond))
    e2 = z3.Not(z3.substitute(z3_cond, (ind_var_z3, N)))
    qe = z3.Tactic('qe')
    simplified = qe(z3.And(e1, e2))[0]
    res = solve_k(simplified, N, set())
    return to_sympy(res)

def solve_k(constraints, k, all_ks):
    solver = z3.Solver()
    solver.add(*constraints)
    all_vars = list(reduce(lambda x, y: x.union(y), [z3_all_vars(expr) for expr in constraints]) - set(all_ks))
    all_vars_1 = all_vars + [z3.IntVal(1)]
    # det_vars = [z3.Real('c_%d' % i) for i in range(len(all_vars_1))]
    det_vars = [z3.Int('c_%d' % i) for i in range(len(all_vars_1))]
    template = sum([c*v for c, v in zip(det_vars, all_vars_1)])
    eqs = []
    while solver.check() == z3.sat:
        linear_solver = z3.Solver()
        for _ in range(len(all_vars_1)):
            solver.check()
            m = solver.model()
            eq = (m[k] == m.eval(template))
            # print(m)
            eqs.append(eq)
            for var in (all_vars + list(all_ks)):
                solver.push()
                # var = random.choice(all_vars + list(all_ks))
                solver.add(z3.Not(var == m[var]))
                if solver.check() == z3.unsat:
                    solver.pop()
            # solver.add(*[z3.Not(var == m[var]) for var in all_vars + [k]])
        linear_solver.add(*eqs)
        linear_solver.check()
        m_c = linear_solver.model()
        cur_sol = m_c.eval(template)
        # check whether k is a constant
        solver.push()
        solver.add(z3.Not(k == m.eval(cur_sol)))
        if solver.check() == z3.unsat:
            cur_sol = m.eval(cur_sol)
            break
        solver.pop()
        ###############################
        solver.add(z3.Not(k == cur_sol))
    return cur_sol


def z3_all_vars(expr):
    if z3.is_const(expr):
        if z3.is_int_value(expr):
            return set()
        else:
            return {expr}
    else:
        try:
            return reduce(lambda x, y: x.union(y), [z3_all_vars(ch) for ch in expr.children()])
        except:
            return set()

def my_sp_simplify(expr, assumptions):
    res = expr
    if isinstance(expr, sp.Piecewise):
        s = z3.Solver()
        s.add(to_z3(assumptions))
        remains = []
        for e, cond in expr.args:
            s.push()
            s.add(to_z3(cond))
            z3_res = s.check()
            if z3_res == z3.sat:
                remains.append((e, cond))
            s.pop()
        res = sp.Piecewise(*remains)
    else:
        try:
            res = expr.func(*[my_sp_simplify(arg, assumptions) for arg in expr.args])
        except:
            pass
    return sp.simplify(res)

def collapse_piecewise(expr):
    sim = z3.Tactic('ctx-solver-simplify')
    res = expr
    merged = {}
    if isinstance(expr, sp.Piecewise):
        exprs = [e for e, _ in expr.args]
        conditions = [to_z3(expr.args[0][1])]
        for _, cond in expr.args[1:]:
            conditions.append(z3.And(to_z3(cond), z3.Not(conditions[-1])))
        for i in range(len(conditions)):
            absorbed = reduce(set.union, (merged[idx] for idx in merged), set())
            if i in absorbed: continue
            merged[i] = set()
            for j in range(len(conditions)):
                absorbed = reduce(set.union, (merged[idx] for idx in merged))
                if i == j or j in absorbed: continue
                s = z3.Solver()
                s.add(conditions[j])
                s.push()
                s.add(z3.Not(to_z3(exprs[i]) == to_z3(exprs[j])))
                check_res = s.check()
                print(s.model())
                s.pop()
                if check_res == z3.unsat:
                    merged[i].add(j)
        new_exprs = []
        new_conditions = []
        absorbed = reduce(set.union, (merged[idx] for idx in merged))
        for i in merged:
            cur_condition = to_sympy(z3_deep_simplify(z3.And(z3.BoolVal(True), *sim(z3.Or(conditions[i], *[conditions[j] for j in merged[i]]))[0])))
            # cur_condition = to_sympy(z3_deep_simplify(z3.And(z3.BoolVal(True), *sim(to_z3(sp.Or(conditions[i], *[conditions[j] for j in merged[i]])))[0])))
            if i not in absorbed:
                new_exprs.append(exprs[i])
                new_conditions.append(cur_condition)
            else:
                conditions[i] = cur_condition
        res = sp.Piecewise(*[(e, c) for e, c in zip(new_exprs, new_conditions)])
    return res
        # self.closed_forms = new_closed_forms
        # self.conditions = new_conditions
        
# sympy to z3 -> new version from: 
# https://stackoverflow.com/questions/75461163/python-how-to-parse-boolean-sympy-tree-expressions-to-boolean-z3py-expressions
# define cvt outside of compile_to_z3

def cvt_parallel(expr, pvs, constants, table):
    if expr in pvs:
        return str(pvs[expr])  # convert Z3 object to string

    texpr = type(expr)
    if texpr in constants:
        return str(constants[texpr])  # convert Z3 object to string

    if texpr in table:
        return (table[texpr])(*[cvt(arg, pvs, constants, table) for arg in expr.args])

    raise NameError("Unimplemented: " + str(expr))


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
    results = [pool.apply_async(cvt, (expr, pvs, constants, table)) for expr in pexp.args] if len(pexp.args) > 1 else [pool.apply_async(cvt, (pexp, pvs, constants, table))]
    output = [res.get() for res in results]

    #return cvt(pexp) # non-parallel
    return table[type(pexp)](*output) # parallel


#@lru_cache(maxsize=None)
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
    print(z3.simplify(to_z3_parallel(sp.simplify(to_sympy_parallel(c)))))
    
    # a = sp.Function('a')
    # bb = sp.Function('bb')
    # cc = sp.Function('cc')
    # d = sp.Function('d')
    # _t0 = sp.Symbol('_t0', integer=True)
    # _t1 = sp.Symbol('_t1', integer=True)
    # e = sp.Piecewise((a(0) + bb(0, _t1)*d(0) + cc(0, _t1), sp.Eq(_t0, 0)), (a(_t0) + bb(_t0, _t1)*d(_t0), True))
    # print(collapse_piecewise(e))
    # a = sp.Function('a')
    # i = sp.Symbol('i', integer=True)
    # e = 2*a(i+1) + 1
    # from pycparser import c_generator
    # generator = c_generator.CGenerator()
    # print(generator.visit(expr2c(e)))

    # print(get_app_by_var(a, e))