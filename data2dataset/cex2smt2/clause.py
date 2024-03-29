import torch
import sys

def check_header(fname:str):
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
    def __init__(self, fname = None, clauses = None, num_input=0, num_sv=0):
        # load from file
        assert fname or clauses
        assert not (fname and clauses)
        if fname:
            self.clauses = []
            with open(fname) as fin:
                header = fin.readline()
                header = header.split()
                if len(header) < 2 or header[1] == "0" : 
                    self.report2log(fname, header, "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/log/error_handle/abnormal_header.log")
                    sys.exit()
                assert len(header) >= 2
                assert header[0] == 'unsat'
                assert header[1] != "0"
                assert num_input+num_sv > 0
                n_clause = int(header[1])
                for _ in range(n_clause):
                    l = fin.readline()
                    literals = l.split()
                    lit = [int(l) for l in literals]
                    vlist = []
                    for l in lit:
                        sign = 1 if l % 2 == 0 else -1
                        var = int(l/2)-num_input-1  # NOTE: here we convert to the absolute latch number (start from 0)
                        assert var >= 0 and var < num_sv
                        vlist.append( (var, sign) )
                    self.clauses.append(vlist)
        elif clauses:
            self.clauses = clauses

    def report2log(self, fname, header, log_file):
        # append the error message to the log file
        with open(log_file, "a+") as fout:
            fout.write(f"Error: {fname} has abormal header: {header} \n")
        fout.close()
        
        

