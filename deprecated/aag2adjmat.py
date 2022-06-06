# HZ: This is the function that converts AAG directly to adjacent matrix

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
LI_UPDATE=5
OUT_UPDATE=6



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

    def newnode(self, tp):
        self.nid+=1
        self.node_type[self.nid] = tp
        # print (self.nid, 'TYPE:',tp)
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

                nid = self.newnode(INP)
                var_table[iv]=nid

            for idx in range(L):
                line = fin.readline().split()
                #print (line)
                assert len(line) == 2, 'cannot have init value for latches'
                latchno = int(line[0])
                assert latchno == I*2+(idx+1)*2
                latch_update_no.append((latchno, int(line[1])))
                #print (latchno, int(line[1]))
                nid = self.newnode(SV)
                var_table[latchno]=nid
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
                    left_not_nid = self.newnode(NOT)
                if right % 2 == 1:
                    right_not_nid = self.newnode(NOT)

                nid = self.newnode(AND)
                var_table[aid]=nid

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
                not_nid = self.newnode(NOT)
                oid = var_table[outputidx-1]
                self.edges.add((oid, not_nid))
                self.outputnode = not_nid
            else:
                oid = var_table[outputidx]
                self.outputnode = oid

            for latchno, nxtv in latch_update_no:
                if nxtv % 2 == 1:
                    not_nid = self.newnode(NOT)
                    li = var_table[nxtv-1]
                    self.edges.add((li, not_nid))
                    self.li_node.append(not_nid)
                else:
                    li = var_table[nxtv]
                    self.li_node.append(li)
            #print (var_table)
            #print (self.node_type)
            self.connect_Li_to_Lo()
            return True

    def count_inoutdegree(self, total_x, edge_index):
        outdeg = torch.bincount(edge_index[0])
        outdeg = (outdeg == 0).long() + outdeg
        # if an element is 0 --> convert to 1 . Because we will need
        # to divide hidden vector by out deg, this is to avoid divide-0

        indeg = torch.bincount(edge_index[1])
        indeg = (indeg == 0).long() + indeg

        return (indeg, outdeg)
        

    def connect_Li_to_Lo(self):
        for idx, svnid in enumerate(self.sv_node):
            li = self.li_node[idx]
            svupdate_node = self.newnode(LI_UPDATE)
            self.edges.add( (li, svupdate_node) )
            self.edges.add( (svupdate_node, svnid) )

        for idx, svnid in enumerate(self.sv_node):
            svupdate_node = self.newnode(OUT_UPDATE)
            self.edges.add( (self.outputnode, svupdate_node) )
            self.edges.add( (svupdate_node, svnid) )
            
        

    def to_dataframe(self, total_num_node_types, clauses, aag_name='') -> Data:
        nodetype = sorted(self.node_type.items())
        x = []
        sv_node = []
        for idx,tp in nodetype:
            x.append(one_hot(tp, total_num_node_types))

        x = torch.cat(x, dim=0).float()
        edge_index = torch.tensor(list(self.edges)).t().contiguous()  # our graph has more than 1 output !
        # x is like        [0,0,0,0,1]         [init_for_type1]
        #                  [0,0,0,0,1]     X   [init_for_type2]
        #                  [0,0,1,0,0]         [init_for_type3]
        # edge_index is like [0,0, ..]
        #                    [1,2, ..]
        #

        ind, outd = self.count_inoutdegree(total_x = x.shape[0], edge_index=edge_index )
        assert ind.shape[0] == outd.shape[0]
        assert ind.shape[0] == x.shape[0]


        g = Data(x=x, edge_index=edge_index)
        g.__setattr__("clauses", clauses)
        sv_node = torch.tensor(self.sv_node)
        g.__setattr__("sv_node", sv_node)
        outputnode = torch.tensor(self.outputnode)
        g.__setattr__("output_node", outputnode)
        li_node = torch.tensor(self.li_node)
        g.__setattr__("li_node", li_node)
        g.__setattr__("ind", ind)
        g.__setattr__("outd", outd)
        g.__setattr__("aag_name", aag_name)        
        return g


# ../hwmcc10-mod/pj2017.aag ../hwmcc10-7200-result/output/pj2017/inv.cnf
def test_speed():
    g = AAGmodel()
    g.from_file("../hwmcc10-mod/pj2017.aag")
    clause = Clauses(fname='../hwmcc10-7200-result/output/pj2017/inv.cnf', num_input=g.num_input,num_sv=g.num_sv)
    graph = g.to_dataframe(7, clause)
    print (graph.edge_index)
    print (graph.x)
    print (graph.sv_node)
    print (graph.output_node)
    print (graph.li_node)
    print (graph.ind)
    print (graph.outd)
    print (graph)
    G = to_networkx(graph)
    #nx.draw(G)
    #plt.savefig("test-large.png")

    # try convert to sparse matrix
    num_nonzero = graph.edge_index.shape[1]
    n_node=graph.x.shape[0]
    AdjMatrix = torch.sparse_coo_tensor(indices=graph.edge_index, values=torch.ones(num_nonzero), size=(n_node, n_node))
    mat2 = torch.randn((n_node, 100))
    print (torch.matmul(AdjMatrix, mat2).shape)

def test():
    g = AAGmodel()
    g.from_file("testcase1/cnt.aag")
    clause = Clauses(fname='testcase1/inv.cnf', num_input=g.num_input,num_sv=g.num_sv)
    graph = g.to_dataframe(7, clause)
    print (graph.edge_index)
    print (graph.x)
    print (graph.sv_node)
    print (graph.output_node)
    print (graph.li_node)
    print (graph.ind)
    print (graph.outd)
    print (graph)
    G = to_networkx(graph)
    nx.draw(G)
    plt.savefig("test2.png")

    # try convert to sparse matrix
    num_nonzero = graph.edge_index.shape[1]
    n_node=graph.x.shape[0]
    AdjMatrix = torch.sparse_coo_tensor(indices=graph.edge_index, values=torch.ones(num_nonzero), size=(n_node, n_node))
    mat2 = torch.randn((n_node, 100))
    print (torch.matmul(AdjMatrix, mat2).shape)
    #print (AdjMatrix)

if __name__ == '__main__':
    test()
