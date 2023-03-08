import z3


def get_literal(vartable, lid): #BUG: There is a bug here -> when latch has true, the table is incorrect
    expr = vartable[int(lid/2)]
    if lid%2==1:
        expr=z3.Not(expr)
    return expr

class AAGmodel():
    def __init__(self):
        self.inputs = []  # I
        self.svars = []   # L
        self.primed_svars = []
        self.output = None # O
        self.latch2next = dict()

    def from_file(self, fname):
        with open(fname) as fin:
            header=fin.readline()
            header=header.split()
            M,I,L,O,A=int(header[1]),int(header[2]),int(header[3]),int(header[4]),int(header[5])
            latch_update_no = []
            outputidx = None
            var_table=dict()
            var_table[0]=False

            if M == 0 or L == 0 or O != 1:
                return False # parse failed
            for idx in range(I):
                line = fin.readline().split()
                #print (line)
                assert len(line) == 1
                iv = int(line[0])
                assert iv == (idx+1)*2
                vname='i'+str(iv)
                v = z3.Bool(vname)
                self.inputs.append(v)
                var_table[iv/2] = v

            for idx in range(L):
                line = fin.readline().split()
                #print (line)
                assert len(line) == 2, 'cannot have init value for latches'
                latchno = int(line[0])
                assert latchno == I*2+(idx+1)*2
                latch_update_no.append((latchno, int(line[1])))
                #print (latchno, int(line[1]))

                vname='v'+str(latchno)
                v = z3.Bool(vname)
                self.svars.append(v)
                var_table[latchno/2] = v

                vnamep='v'+str(latchno)+"_prime"
                vp = z3.Bool(vnamep)
                self.primed_svars.append(vp)

            for idx in range(O):
                line = fin.readline().split()
                #print (line)
                assert len(line) == 1
                assert outputidx is None
                outputidx=int(line[0])

            for idx in range(A):
                line = fin.readline().split()
                assert len(line) == 3
                aid = int(line[0])
                assert aid == (I+L)*2+(idx+1)*2
                left = int(line[1])
                lexpr = get_literal(var_table, left)
                right = int(line[2])
                rexpr = get_literal(var_table, right)
                var_table[aid/2] = z3.And(lexpr, rexpr)

            # now fill in latch & output
            assert outputidx is not None
            self.output=get_literal(var_table,outputidx)

            for latchno, nxtv in latch_update_no:
                sv = self.svars[int(latchno/2)-1-I]
                self.latch2next[sv]=get_literal(var_table,nxtv)
            return True


if __name__ == '__main__':
    m = AAGmodel()
    m.from_file('../hwmcc07-mod/intel/intel_018.aag')
    print(m.inputs)
    print(m.svars)
    print(m.primed_svars)
    print(m.output)

    for sv, nxt in m.latch2next.items():
        print (sv, ':=', nxt)
