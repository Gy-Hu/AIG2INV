# -*- coding: utf-8 -*-
"""
Use to convert symbolic string to z3 smtlib2
"""

import numpy as np
import collections
import os.path
import platform
import z3
import subprocess
import os
import sympy as sp



class SympyToZ3:
    """SMT-based verification of certificate functions.

    Solvers: dReal, Z3.
    """

    def __init__(self, symbolic_expr=None, options={}):
        self.options = options

        self.sym_dict = collections.OrderedDict({"Add": "+",
                                                 "Mul": "*",
                                                 "Pow": "^",
                                                 "StrictGreaterThan": ">",
                                                 "GreaterThan": ">=",
                                                 "StrictLessThan": "<",
                                                 "LessThan": "<=",
                                                 "Implies": "=>",
                                                 "And": "and",
                                                 "Or": "or",
                                                 "Not": "not",
                                                 "Max": "max",
                                                 "Min": "min",
                                                 "Abs": "abs",
                                                 "Unequality": "!",
                                                 "Equality": "="})

        self.sym_dict_exceptions = collections.OrderedDict(
            {"Max": "max", "Min": "min"})
        self.num_dict = ["Float", "Integer", "Zero", "One",
                         "Symbol", "NegativeOne", "Rational", "Half"]
        #XXX: Double check before running the script -> Use sympy to simplify the expression?
        self.symbolic_expr = sp.simplify(symbolic_expr)
        #self.sympy2smt(symbolic_expr, list(symbolic_expr.atoms()))

    def symbolic_name_to_lisp(self, fun):
        """Translate function between symbolic python and SMT2 function names.

        Part of the symbolic_to_lisp translator.
        """
        expr = fun
        for word in self.sym_dict:
            expr = expr.replace(word, self.sym_dict[word])
        return expr

    def symbolic_to_lisp(self, expr):
        """Translate function from symbolic python to SMT2 expressions."""
        name = expr.func.__name__
        if name in self.num_dict:
            # for rational we must convert to numerical value
            if name == "Rational" or name == "Half":
                expr = float(expr)
            sform = str(expr)
        elif name in self.sym_dict_exceptions:
            sform = "(" + self.symbolic_name_to_lisp(name)
            sform = sform + " " + self.symbolic_to_lisp(expr.args[0])
            for arg in expr.args[1:-1]:
                sform = sform + " (" + self.symbolic_name_to_lisp(name) + \
                    " " + self.symbolic_to_lisp(arg)
            sform = sform + " " + self.symbolic_to_lisp(expr.args[-1])
            for arg in expr.args[1:]:
                sform = sform + ")"
        else:
            sform = "(" + self.symbolic_name_to_lisp(name)
            for arg in expr.args:
                sform = sform + " " + self.symbolic_to_lisp(arg)
            sform = sform + ")"
        return sform

    def make_SMT2_string(self, symbolic_expr, symbol_list, domain=None):
        """Write an SMT file in string format.

        The SMT file is used to check whether inequality symbolic_expr is
        satisfied using dReal.
        """
        # Write settings in a string
        string = ""
        for var in symbol_list:
            string = string + "(declare-fun " + str(var) + " () Bool)\n"
        string = string + "(assert " + self.symbolic_to_lisp(symbolic_expr) + ")\n"
        # check satisfiability
        string = string + "(check-sat)\n"
        string = string + "(exit)"
        return string
    
    def sympy2smt(self):   
        # Write SMT file
        string = self.make_SMT2_string(self.symbolic_expr, list(self.symbolic_expr.atoms()))
        # Translate to Z3 API
        smt_parse = z3.parse_smt2_string(string)
        return smt_parse[0]