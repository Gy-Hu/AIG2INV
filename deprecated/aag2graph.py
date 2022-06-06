# HZ: This is the function that converts AAG directly to graph (w.o. Z3)

import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from utils import one_hot, add_order_info
import matplotlib.pyplot as plt
from clause import Clauses

AND=0
NOT=1
SV=2
INP=3
FALSE=4
TRUE=5


class AAGmodel():
    def __init__(self):
        self.nid = 0
        self.edges = set()
        self.node_type = {} # map nid -> type (0: AND, 1: NOT, 2: SV-0, 3: INP)
        self.li_node = []
        self.outputnode = None
        self.sv_node = []
        self.num_input = 0
        self.num_sv = 0

    def newnode(self):
        self.nid+=1
        return self.nid # 0 is reserved for false


    def from_file(self, fname):
        with open(fname) as fin:
            header=fin.readline()
            header=header.split()
            M,I,L,O,A=int(header[1]),int(header[2]),int(header[3]),int(header[4]),int(header[5])
            latch_update_no = []
            outputidx = None
            var_table=dict()
            self.num_input = I
            self.num_sv = L

            var_table[0]=0  # from postive literal to nid. 0 is reserved for false
            self.node_type[0]=FALSE

            if M == 0 or L == 0 or O != 1:
                return False # parse failed
            for idx in range(I):
                line = fin.readline().split()
                #print (line)
                assert len(line) == 1
                iv = int(line[0])
                assert iv == (idx+1)*2

                nid = self.newnode()
                var_table[iv]=nid
                self.node_type[nid] = INP

            for idx in range(L):
                line = fin.readline().split()
                #print (line)
                assert len(line) == 2, 'cannot have init value for latches'
                latchno = int(line[0])
                assert latchno == I*2+(idx+1)*2
                latch_update_no.append((latchno, int(line[1])))
                #print (latchno, int(line[1]))
                nid = self.newnode()
                var_table[latchno]=nid
                self.node_type[nid]=SV
                self.sv_node.append(nid)


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
                right = int(line[2])


                if left % 2 == 1:
                    left_not_nid = self.newnode()
                    self.node_type[left_not_nid]=NOT
                if right % 2 == 1:
                    right_not_nid = self.newnode()
                    self.node_type[right_not_nid]=NOT

                nid = self.newnode()
                var_table[aid]=nid
                self.node_type[nid]=AND

                lnode = var_table[int(left/2)*2]
                rnode = var_table[int(right/2)*2]

                if left % 2 == 1:
                    self.edges.add((lnode, left_not_nid))
                    self.edges.add((left_not_nid,nid))
                else:
                    self.edges.add((lnode, nid))

                if right % 2 == 1:
                    self.edges.add((rnode, right_not_nid))
                    self.edges.add((right_not_nid,nid))
                else:
                    self.edges.add((rnode, nid))

            # now fill in latch & output
            assert outputidx is not None
            if outputidx % 2 == 1:
                not_nid = self.newnode()
                self.node_type[not_nid]=NOT
                oid = var_table[outputidx-1]
                self.edges.add((oid, not_nid))
                self.outputnode = not_nid
            else:
                oid = var_table[outputidx]
                self.outputnode = oid

            for latchno, nxtv in latch_update_no:
                if nxtv % 2 == 1:
                    not_nid = self.newnode()
                    self.node_type[not_nid]=NOT
                    li = var_table[nxtv-1]
                    self.edges.add((li, not_nid))
                    self.li_node.append(not_nid)
                else:
                    li = var_table[nxtv]
                    self.li_node.append(li)
            #print (var_table)
            #print (self.node_type)
            return True


    def to_dataframe(self, total_num_node_types, clauses, aag_name='') -> Data:
        nodetype = sorted(self.node_type.items())
        x = []
        sv_node = []
        for idx,tp in nodetype:
            x.append(one_hot(tp, total_num_node_types))

        x = torch.cat(x, dim=0).float()
        edge_index = torch.tensor(list(self.edges)).t().contiguous()  # our graph has more than 1 output !
        g = Data(x=x, edge_index=edge_index)
        add_order_info(g)
        g.__setattr__("clauses", clauses)
        sv_node = torch.tensor(self.sv_node)
        g.__setattr__("sv_node", sv_node)
        outputnode = torch.tensor(self.outputnode)
        g.__setattr__("output_node", outputnode)
        li_node = torch.tensor(self.li_node)
        g.__setattr__("li_node", li_node)
        g.__setattr__("aag_name", aag_name)        
        return g


def test():
    g = AAGmodel()
    g.from_file("testcase1/cnt.aag")
    clause = Clauses(fname='testcase1/inv.cnf', num_input=g.num_input,num_sv=g.num_sv)
    graph = g.to_dataframe(6, clause)
    print (graph.edge_index)
    print (graph.x)
    print (graph.sv_node)
    print (graph.output_node)
    print (graph.li_node)
    print (graph)
    G = to_networkx(graph)
    nx.draw(G)
    plt.savefig("test1.png")

if __name__ == '__main__':
    test()