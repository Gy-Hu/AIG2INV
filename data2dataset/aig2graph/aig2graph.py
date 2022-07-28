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
    # probably okay for ealier AIGs
    def __init__(self, sv, inpv, trans, output):
        self.sv, self.inpv = sv, inpv

        self.nid = 0
        self.node2nid = {}  # map z3expr -> node id
        
        self.edges = set() # the id in edges should the same as the sorted nodetypes
        self.node_type = {} # map nid -> type (0: AND, 1: NOT, 2: SV-0, 3: INP)

        self.trans = trans
        self.output = output
        for expr in trans:
            self.convert_expr(expr)
        self.convert_expr(output)


    def to_dataframe(self, total_num_node_types, clauses, aag_name='') -> Data:
        nodetype = sorted(self.node_type.items())
        x = []
        sv_node = []
        for idx,tp in nodetype:
            x.append(one_hot(tp, total_num_node_types)) # one hot embedding according to type
            if tp == 2:
                sv_node.append(idx)

        x = torch.cat(x, dim=0).float()
        edge_index = torch.tensor(list(self.edges)).t().contiguous()  # our graph has more than 1 output !
        g = Data(x=x, edge_index=edge_index)
        add_order_info(g) # add arrows to the graph
        g.__setattr__("clauses", clauses)
        sv_node = torch.tensor(sv_node)
        g.__setattr__("sv_node", sv_node)
        output_node = self.node2nid[self.output]
        g.__setattr__("output_node", output_node)
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
                    assert False, 'unrecognized node type'
            self.node_type[nnid] = nodetype
      

def test():

    x = z3.Bool('x')
    y = z3.Bool('y')
    _1 = z3.And(x,y)
    _2 = z3.And(z3.Not(_1), z3.Not(y))
    _3 = z3.Not(z3.And(_2, z3.Not(_1), y))

    g = AigGraph(sv = [x], inpv = [y], trans = [_3], output=_1)
    for node, nid in g.node2nid.items():
        print (str(node), ' ID:',nid, 'TYPE:', g.node_type[nid] )
    graph = g.to_dataframe(6, None)
    print (graph)
    G = to_networkx(graph)
    nx.draw(G)
    plt.savefig("test1.png")
    
def test2():
    '''
    test2.aag
    2 
    4
    2 5
    4 3
    6
    6 2 5
    '''

    '''
    test3.aag (this is a bug)
    2
    4
    2 3
    4 5
    6
    6 2 5
    '''

    x = z3.Bool('x')
    y = z3.Bool('y')
    output=z3.And(x,z3.Not(y))

    g = AigGraph(sv = [x,y], inpv = [], trans = [z3.Not(y), z3.Not(x)], output=output)
    for node, nid in g.node2nid.items():
        print (str(node), ' ID:',nid, 'TYPE:', g.node_type[nid] )
    clause = Clauses(clauses = [ [(0,1), (1,-1)], [(0,-1), (1,1)]])  # not (x y') and not (x' y)
    graph = g.to_dataframe(total_num_node_types=6, clauses=clause.clauses)
    print (graph)
    G = to_networkx(graph)
    nx.draw(G)
    plt.savefig("test1.png")

if __name__ == '__main__':
    test2()

'''

Not(And(And(Not(And(x, y)), Not(y)), Not(And(x, y)), y))  ID: 0 TYPE: 1
And(And(Not(And(x, y)), Not(y)), Not(And(x, y)), y)  ID: 1 TYPE: 0
And(Not(And(x, y)), Not(y))  ID: 2 TYPE: 0
Not(And(x, y))  ID: 3 TYPE: 1
y  ID: 4 TYPE: 3
Not(y)  ID: 5 TYPE: 1
And(x, y)  ID: 6 TYPE: 0
x  ID: 7 TYPE: 2
tensor([[0., 1., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.],
        [0., 1., 0., 0.],
        [1., 0., 0., 0.],
        [0., 0., 1., 0.]])
tensor([[3, 4, 4, 7, 3, 5, 2, 6, 1, 4],
        [2, 6, 5, 6, 1, 2, 1, 3, 0, 1]])
tensor([[5, 4, 3, 2, 0, 1, 1, 0],
        [0, 1, 2, 3, 4, 5, 6, 7]])

'''
