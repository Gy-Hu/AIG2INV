from pathlib import Path
import string
from typing import Union, List
import json

import networkx
import z3

PathLike = Union[Path, str]


class MalformedDimacs(Exception):
    """
    Exception raised when the DIMACS
    parsing fail.
    """
    pass


class CNFFormula:
    """
    Class the represent a CNF formula in DIMACS format.
    It can either be populated by a CNF file or by an SMT2
    file. In this later case Z3 is used to parse the file
    and to extract the CNF representation.
    """
    def __init__(self):
        """
        Class construction. A formula is meant to be instanciated
        via the static methods from_z3, from_smt2_file, from_dimacs_string
        and from_dimacs_file.
        """
        self.goal = z3.Goal()
        self.tactics = z3.Then('simplify', 'bit-blast', 'tseitin-cnf')
        self.subgoal = None  # keep it in case of
        # Goal is a collection of constraints we want to find a solution or show to be unsatisfiable (infeasible).
        # Goals are processed using Tactics. A Tactic transforms a goal into a set of subgoals.
        # A goal has a solution if one of its subgoals has a solution.
        # A goal is unsatisfiable if all subgoals are unsatisfiable.

        self.variables_num = None
        self.clauses_num = None
        self.clauses = []

    def _apply_tactics(self) -> None:
        """
        Apply the Z3 tactics to generate the goals.

        :return: None
        """
        self.subgoal = self.tactics(self.goal)
        if len(self.subgoal) > 1:
            print("warning: multiple subgoal for the formula")

    @staticmethod
    def from_z3(smt_formula) -> 'CNFFormula':
        """
        Convert a Z3 expression to a CNF formula.

        :param smt_formula: Z3 expression (all types that can be fed to a Goal instance)
        :return: a CNF Formula instance
        """
        f = CNFFormula()
        f.goal.add(smt_formula)
        f._apply_tactics()
        f._fill_from_dimacs_string(f.subgoal[0].dimacs())
        return f

    @staticmethod
    def from_smt2_file(smt_file: PathLike) -> 'CNFFormula':
        """"
        Load a CNFFormula from an SMT2 file. It uses Z3 to parse
        the SMT2 and to apply the different tactics generating the
        goals in CNF.

        :param smt_file: SMT2 file
        :return: a CNF Formula instance
        """
        p = Path(smt_file)
        if not p.exists():
            raise FileNotFoundError('SMT2 file not found: {}'.format(p))
        x = z3.parse_smt2_file(p.as_posix())
        return CNFFormula.from_z3(x)

    def _fill_from_dimacs_string(self, s: str) -> None:
        """
        Parse the CNF file as a string and fill the
        appropriate attributes of the object.

        :param s: CNF file string
        :return: None
        """
        lines = [x for x in s.split("\n") if x]
        for i in range(len(lines)):
            l = lines[i]
            if l[0] == "c":  # comment
                pass
            elif l[0] == "p":  # problem line
                p, cnf, var_num, clause_num = l.split(" ")
                self.variables_num, self.clauses_num = var_num, clause_num
                if cnf != "cnf":
                    raise MalformedDimacs("Problem line FORMAT should be 'cnf'")
            else:
                # This parsing force each clause to be on a different line
                terms = l.split(" ")
                if terms[-1] != "0":
                    if i != (len(lines)-1):
                        raise MalformedDimacs("DIMACS clause expected to end by '0'")
                    else:
                        terms = [int(x) for x in terms]
                else:
                    terms = [int(x) for x in terms[:-1]]
                self.clauses.append([int(x) for x in terms])

    @staticmethod
    def from_dimacs_string(s: str) -> 'CNFFormula':
        """
        Create a CNFFormula instance from the given
        string which represent a DIMACS file.s

        :param s: DIMACS file as a string
        :return: a CNFFormula instance
        """
        f = CNFFormula()
        f._fill_from_dimacs_string(s)
        return f

    @staticmethod
    def from_dimacs_file(in_file: PathLike) -> 'CNFFormula':
        """
        Create a CNF Formula from a dimacs file.

        :param in_file: input DIMACS file
        :return: a CNFFormula instance
        """
        p = Path(in_file)
        if not p.exists():
            raise FileNotFoundError("DIMACS file {} not found".format(p))
        return CNFFormula.from_dimacs_string(p.read_text())

    @staticmethod
    def __int_clauses_to_z3(clauses: List[List[int]]) -> z3.z3.BoolRef:
        z3_clauses = []
        vars = {}
        for clause in clauses:
            conds = []
            for t in clause:
                a = abs(t)
                if a in vars:
                    b = vars[a]
                else:
                    b = z3.Bool("k!{}".format(a))
                    vars[a] = b
                b = z3.Not(b) if t < 0 else b
                conds.append(b)
            z3_clauses.append(z3.Or(*conds))
        return z3.And(*z3_clauses)

    def to_networkx(self, with_layout=False, dim=2, iter=50) -> networkx.Graph:
        """
        Convert the CNFFormula to a graph representation using networkx.
        Also allows to compute a layout in 2 or 3 dimensions. The default
        algorithm is spring (force directed algorithm).

        :param with_layout: compute or not a layout
        :param dim: dimension of the layout (2 or 3)
        :param iter: number of iteration for the layout computation
        :return: A Graph instance
        """
        g = networkx.Graph()
        for clause in self.clauses:
            for i in range(len(clause)):
                n1 = abs(clause[i])
                g.add_node(n1)
                for j in range(i, len(clause)):
                    n2 = abs(clause[j])
                    g.add_node(n2)
                    g.add_edge(n1, n2)
        if with_layout:
            pos = networkx.spring_layout(g, dim=dim, iterations=iter)
            for i, x in pos.items():
                if dim == 2:
                    g.nodes[i]['x'], g.nodes[i]['y'] = x
                else:
                    g.nodes[i]['x'], g.nodes[i]['y'], g.nodes[i]['z'] = x
        return g

    def to_graph_file(self, output_file: PathLike, with_layout=False, dim=2, iter=50) -> None:
        """
        Convert the formula to a graph representation (using networkx) and
        export it as JSON.

        :param output_file: JSON output file
        :param with_layout: if True compute the layout
        :param dim: dimension of the layout (2 or 3)
        :param iter: number of iteration for the layout computation
        :return: None
        """
        g = self.to_networkx(with_layout, dim, iter)
        json_data = networkx.readwrite.json_graph.node_link_data(g)
        with open(output_file, "w") as f:
            json.dump(json_data, f)

    def to_dimacs_file(self, output_file: PathLike) -> None:
        """
        Write the CNFFormula to a DIMACS file. If it was instanciated
        from z3 just call .dimacs() otherwise dump all the clauses.

        :param output_file: DIMACS output file
        :return: None
        """
        with open(output_file, "w") as f:
            if self.subgoal:
                f.write(self.subgoal[0].dimacs())
            else:
                f.write("p cnf {} {}\n".format(self.variables_num, self.clauses_num))
                for c in self.clauses:
                    f.write("{} 0\n".format(" ".join(str(x) for x in c)))

    def to_dimacs_string(self):
        """
        Write the CNFFormula to a DIMACS file. If it was instanciated
        from z3 just call .dimacs() otherwise dump all the clauses.

        :param output_file: DIMACS output file
        :return: None
        """
        cnf_string_lst = []
        if self.subgoal:
            cnf_string_lst.append(self.subgoal[0].dimacs())
            return cnf_string_lst
        else:
            cnf_string_lst.append("p cnf {} {}\n".format(self.variables_num, self.clauses_num))
            for c in self.clauses: cnf_string_lst.append("{} 0\n".format(" ".join(str(x) for x in c)))
            return cnf_string_lst

            