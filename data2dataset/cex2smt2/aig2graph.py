import z3

import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from utils import one_hot, add_order_info
import matplotlib.pyplot as plt
from clause import Clauses


class AigGraph(object):
    # this assumes all zero initial values for 0
    # probably okay for earlier AIGs
    # we expect prop is something already substituted
    def __init__(self, sv, inpv, cex, cnf, prop ):
        self.sv, self.inpv = sv, inpv
        # convert sv, inpv to nodes

        self.nid = 0
        self.node2nid = {}  # map z3expr -> node id
        
        self.node_type = {} # map nid -> type (0: AND, 1: NOT, 2: SV-0, 3: INP, 4: False, 5: True)
        self.edges = set()

        for s in self.sv:
            self.convert_expr(s) # this ensure the svs follow the orders
        for s in self.inpv:
            self.convert_expr(s)

        # extract cex assignment and cnf assignment
        self.cex = cex
        self.cnf = cnf
        self.convert_expr(prop)


    def to_dataframe(self, total_num_node_types, aag_name='') -> Data:
        nodetype = sorted(self.node_type.items())
        x = []
        sv_node = []
        for idx,tp in nodetype:
            x.append(one_hot(tp, total_num_node_types))
            if tp == 2:
                sv_node.append(idx)

        assert sv_node[0] == 0
        assert len(sv_node) == sv_node[-1]+1

        x = torch.cat(x, dim=0).float()
        edge_index = torch.tensor(list(self.edges)).t().contiguous()  # our graph has more than 1 output !
        g = Data(x=x, edge_index=edge_index)  # maybe use a separate input for x?
        add_order_info(g)
        sv_node = torch.tensor(sv_node)
        g.__setattr__("sv_node", sv_node)
        #output_node = self.node2nid[self.output]
        #g.__setattr__("output_node", output_node)
        g.__setattr__("aag_name", aag_name)        
        return g

    
    def convert_expr(self, expr):
        # update self.node_type , self.edges
        bfs_queue = [(expr,None)]

        while len(bfs_queue) != 0:
            n, parent = bfs_queue[0]
            del bfs_queue[0]

            if n in self.node2nid:
                nnid = self.node2nid[n]
                if parent is not None:
                    self.edges.add((nnid, parent))
                continue
            # else:
            nnid = self.nid
            self.node2nid[n] = nnid
            self.nid += 1
            if parent is not None:
                self.edges.add((nnid, parent))

            if n == True or n == False:
                op = z3.Z3_OP_TRUE if n == True else z3.Z3_OP_FALSE
                children = []
            else:
                children = n.children()
                bfs_queue += [(c, nnid) for c in children]
                op = n.decl().kind()

            if op == z3.Z3_OP_AND:
                nodetype = 0
            elif op == z3.Z3_OP_NOT:
                nodetype = 1
            elif op == z3.Z3_OP_FALSE:
                nodetype = 4
            elif op == z3.Z3_OP_TRUE:
                nodetype = 5
            else:
                if n in self.sv:
                    nodetype = 2
                elif n in self.inpv:
                    nodetype = 3
                else:
                    print (n)
                    assert False, 'unrecognized node type'
            self.node_type[nnid] = nodetype
      
