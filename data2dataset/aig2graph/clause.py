import torch


def check_header(fname: str):
    with open(fname) as fin:
        header = fin.readline()
        header = header.split()
        if len(header) != 2:
            return False
        if header[0] != 'unsat':
            return False
        if header[1] == "0":
            return False
        return True
    return False


class Clauses(object):
    def __init__(self, fname=None, clauses=None, num_input=0, num_sv=0):
        # load from file
        assert fname or clauses
        assert not (fname and clauses)
        if fname:
            self.clauses = []
            with open(fname) as fin:
                header = fin.readline()
                header = header.split()
                assert len(header) == 2
                assert header[0] == 'unsat'
                assert header[1] != "0"
                assert num_input+num_sv > 0
                n_clause = int(header[1])
                for cidx in range(n_clause):
                    l = fin.readline()
                    literals = l.split()
                    lit = [int(l) for l in literals]
                    vlist = []
                    for l in lit:
                        sign = 1 if l % 2 == 0 else -1
                        # NOTE: here we convert to the absolute latch number (start from 0)
                        var = int(l/2)-num_input-1
                        assert var >= 0 and var < num_sv
                        vlist.append((var, sign))
                    self.clauses.append(vlist)
        elif clauses:
            # only if the number of state variables is 0 and fname is not given
            self.clauses = clauses
