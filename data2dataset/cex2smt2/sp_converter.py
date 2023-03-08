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

# z3 <-> sympy

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

def to_z3(sp_expr):
    self = sp.factor(sp_expr)
    if isinstance(self, sp.Add):
        res = sum([to_z3(arg) for arg in self.args])
    elif isinstance(self, sp.Mul):
        res = 1
        for arg in reversed(self.args):
            if arg.is_number and not arg.is_Integer:
                res = (res*arg.numerator)/arg.denominator
            else:
                res = res * to_z3(arg)
        return z3.simplify(res)
        # return reduce(lambda x, y: x*y, [to_z3(arg) for arg in reversed(self.args)])
    elif isinstance(self, sp.Piecewise):
        if len(self.args) == 1:
            res = to_z3(self.args[0][0])
        else:
            cond  = to_z3(self.args[0][1])
            res = z3.If(cond, to_z3(self.args[0][0]), to_z3(self.args[1][0]))
    elif isinstance(self, sp.And):
        res = z3.And(*[to_z3(arg) for arg in self.args])
    elif isinstance(self, sp.Or):
        res = z3.Or(*[to_z3(arg) for arg in self.args])
    elif isinstance(self, sp.Not):
        res = z3.Not(*[to_z3(arg) for arg in self.args])
    elif isinstance(self, sp.Gt):
        res = to_z3(self.lhs) > to_z3(self.rhs)
    elif isinstance(self, sp.Ge):
        res = to_z3(self.lhs) >= to_z3(self.rhs)
    elif isinstance(self, sp.Lt):
        res = to_z3(self.lhs) < to_z3(self.rhs)
    elif isinstance(self, sp.Le):
        res = to_z3(self.lhs) <= to_z3(self.rhs)
    elif isinstance(self, sp.Eq):
        res = to_z3(self.lhs) == to_z3(self.rhs)
    elif isinstance(self, sp.Ne):
        res = to_z3(self.lhs) != to_z3(self.rhs)
    elif isinstance(self, sp.Integer) or isinstance(self, int):
        res = z3.IntVal(int(self))
    elif isinstance(self, sp.Symbol):
        #XXX: Double check before running script
        # we assume that all symbols are boolean
        res = z3.Bool(str(self))
        #res = z3.Int(str(self))
    elif isinstance(self, sp.Rational):
        # return z3.RatVal(self.numerator, self.denominator)
        res = z3.IntVal(self.numerator) / z3.IntVal(self.denominator)
    elif isinstance(self, sp.Pow):
        if self.base == 0: res = z3.IntVal(0)
        else: raise Exception('%s' % self)
    elif isinstance(self, sp.Mod):
        res = to_z3(self.args[0]) % to_z3(self.args[1])
    elif isinstance(self, sp.Abs):
        res = z3.Abs(to_z3(self.args[0]))
    elif isinstance(self, sp.Sum):
        s = z3.Function('Sum', z3.IntSort(), z3.IntSort(), z3.IntSort(), z3.IntSort(), z3.IntSort())
        # expr, (idx, start, end) = self.args
        expr, *l = self.args
        res = to_z3(expr)
        for idx, start, end in l:
            res = s(res, to_z3(idx), to_z3(start), to_z3(end))
    elif self is true:
        res = z3.BoolVal(True)
    elif self is false:
        res = z3.BoolVal(False)
    elif self.is_Function:
        func = self.func
        args = self.args
        z3_func = z3.Function(func.name, *([z3.IntSort()]*(len(args) + 1)))
        res = z3_func(*[to_z3(arg) for arg in args])
    else:
        raise Exception('Conversion for "%s" has not been implemented yet: %s' % (type(self), self))
    return z3.simplify(res)

def to_sympy(expr):
    if z3.is_int_value(expr):
        res = expr.as_long()
    elif z3.is_const(expr) and z3.is_bool(expr):
        #XXX: Double check before running script
        #res = sp.S.true if z3.is_true(expr) else sp.S.false
        # initialize the boolean variable
        res = sp.Symbol(str(expr))
    elif z3.is_const(expr):
        res = sp.Symbol(str(expr), integer=True)
    elif z3.is_add(expr):
        res = sum([to_sympy(arg) for arg in expr.children()])
    elif z3.is_sub(expr):
        children = expr.children()
        assert(len(children) == 2)
        res = to_sympy(children[0]) - to_sympy(children[1])
    elif z3.is_mul(expr):
        children = expr.children()
        res = reduce(lambda x, y: x*y, [to_sympy(ch) for ch in children])
    elif z3.is_mod(expr):
        children = expr.children()
        res = to_sympy(children[0]) % to_sympy(children[1])
    elif z3.is_gt(expr):
        children = expr.children()
        res = to_sympy(children[0]) > to_sympy(children[1])
    elif z3.is_lt(expr):
        children = expr.children()
        res = to_sympy(children[0]) < to_sympy(children[1])
    elif z3.is_ge(expr):
        children = expr.children()
        res = to_sympy(children[0]) >= to_sympy(children[1])
    elif z3.is_le(expr):
        children = expr.children()
        res = to_sympy(children[0]) <= to_sympy(children[1])
    elif z3.is_eq(expr):
        children = expr.children()
        res = sp.Eq(to_sympy(children[0]), to_sympy(children[1]))
    elif z3.is_not(expr):
        children = expr.children()
        res = sp.Not(to_sympy(children[0]))
    elif z3.is_and(expr):
        children = expr.children()
        body = [to_sympy(ch) for ch in children]
        res = sp.And(*body)
    elif z3.is_or(expr):
        children = expr.children()
        res = sp.Or(*[to_sympy(ch) for ch in children])
    elif len(expr.children()) == 3 and z3.is_bool(expr.children()[0]):
        children = expr.children()
        cond = to_sympy(children[0])
        res = sp.Piecewise((to_sympy(children[1]), cond), (to_sympy(children[2]), sp.S.true))
    else:
        raise Exception('conversion for type "%s" is not implemented: %s' % (type(expr), expr))
    return sp.simplify(res)
    #return res

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


if __name__ == '__main__':
    a = z3.Bool('a')
    b = z3.Bool('b')
    c = z3.Not(z3.And(z3.Not(a), z3.Not(b)))
    print(to_sympy(c))
    
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