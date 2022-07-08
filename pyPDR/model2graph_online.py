'''
Generate graph for real-time ic3 
-> only for inductive generalization at present

Still in progress!!!
'''
import z3
import numpy as np
import pandas as pd
import re
import pickle
from natsort import natsorted
import os
import argparse

def mk_adj_matrix(solver, mode=0):
    if mode == 0:
       pass
    elif mode == 1 or 2:
        # Old method to generate graph
        # new_graph = graph(solver, mode=mode)
        # while len(new_graph.bfs_queue) != 0:
        #     new_graph.add()

        #New method to generate graph
        s2 = z3.Solver()
        s2.add(solver.assertions())
        new_graph_2 = graph(s2, mode=mode)
        new_graph_2.add_upgrade_version()

        new_graph = new_graph_2

        new_graph.print()
        new_graph.to_matrix()
        node_ref = {}
        for key,value in new_graph.node2nid.items():
            node_ref[value] = key.sexpr()
        return new_graph, node_ref

def walkFile(dir):
    for root, _, files in os.walk(dir):
        files = natsorted(files)
        files = [os.path.join(root,f) for f in files]
    return files

class graph:
    def __init__(self, solver, mode=0):
        #self.solver = solver
        self.constraints = solver.assertions()
        self.nid = 0
        self.node2nid = {}  # map z3expr -> node id
        self.relations = set()  # the edges
        self.edges = set()
        self.bfs_queue = [self.constraints[-1]]
        self.q_lst = self.constraints[:-1] # Store literals in q to a list
        self.all_node_var = {}
        self.solver = z3.Solver()
        self.solver_node_val = z3.Solver()
        self.mode = mode
        if self.mode == 0: 
            self.eval_node_val()

    def getnid(self, node):
        if node in self.node2nid:
            nnid = self.node2nid[node]
        else:
            nnid = self.nid
            self.node2nid[node] = nnid
            self.nid += 1
        return nnid

    def add(self):
        n = self.bfs_queue[0]
        del self.bfs_queue[0]
        children = n.children()
        self.bfs_queue += list(children)
        remove_duplicated = lambda x: list(dict.fromkeys(x))
        self.bfs_queue = remove_duplicated(self.bfs_queue)

        nnid = self.getnid(n)
        self.calculate_node_value(n, nnid)
        op = n.decl().kind()
        if op == z3.Z3_OP_AND:
            opstr = 'AND'
        elif op == z3.Z3_OP_NOT:
            opstr = 'NOT'
        else:
            opstr = 'OTHER'

        if len(children) != 0:
            rel = f'{nnid} := {opstr} ( {[self.getnid(c) for c in children]} )'
            self.relations.add(rel)

        for c in children:
            cnid = self.getnid(c)
            self.edges.add((nnid, cnid))

    def add_upgrade_version(self):
        remove_duplicated = lambda x: list(dict.fromkeys(x))
        index = 0
        while True:
            #len((self.bfs_queue[index]).children()) != 0
            old_length = len(self.bfs_queue)
            self.bfs_queue += list((self.bfs_queue[index]).children())
            new_length = len(self.bfs_queue)
            if new_length == old_length and index==(len(self.bfs_queue)-1):
                break
            index += 1
        self.bfs_queue = remove_duplicated(self.bfs_queue)

        for n in self.bfs_queue:
            nnid = self.getnid(n)
            self.calculate_node_value(n, nnid)
            op = n.decl().kind()
            if op == z3.Z3_OP_AND:
                opstr = 'AND'
            elif op == z3.Z3_OP_NOT:
                opstr = 'NOT'
            else:
                opstr = 'OTHER'

            if len(n.children())!= 0:
                rel = f'{nnid} := {opstr} ( {[self.getnid(c) for c in n.children()]} )'
                self.relations.add(rel)

            for c in n.children():
                cnid = self.getnid(c)
                self.edges.add((nnid, cnid))

    def print(self):
        print('-------------------')
        print('NODE:')
        for n, nid in self.node2nid.items():
            print(nid, ':', n)

        print('-------------------')
        print('RELATION:')
        relations = sorted(list(self.relations))
        for rel in relations:
            print(rel)

        print('-------------------')
        print('EDGE:')
        print(self.edges)

    def calculate_var(self):
        '''
        :return: a dictionary of variable and its index
        '''
        var = {}
        for item, value in self.node2nid.items():
            op = item.decl().kind()
            if (op!= z3.Z3_OP_AND) and (op != z3.Z3_OP_NOT):
                var[value] = item
        return var

    def to_matrix(self):
        edges = list(self.edges)
        edges = [list(edges[i]) for i in range(0, len(edges), 1)]
        edges = sorted(edges, key=lambda x: (x[0], x[1]), reverse=False)
        node = set()
        var_dict = self.calculate_var()
        for item in edges:
            node.add(item[0])
            node.add(item[1])
            item.append(1)
        n_nodes = len(node)
        n_nodes_var = len(node)
        A = np.zeros((n_nodes_var, n_nodes))
        df_2 = pd.DataFrame(self.all_node_var,index=[0]).T
        for edge in edges:
            i = int(edge[0])
            j = int(edge[1])
            weight = edge[2]
            A[i, j] = weight
        self.adj_matrix = A
        df = pd.DataFrame(self.adj_matrix)

        def map(x):
            ori = x
            for key, value in var_dict.items():
                if key == x:
                    return "n_"+str(value)
            return "m_"+str(ori)

        df.rename(index=map, columns=map, inplace=True)
        df_2.rename(index=map, inplace=True)
        df_2.columns = ['Value']
        df_2 = df_2.reindex(natsorted(df_2.index), axis=0)
        df = df.reindex(natsorted(df.index), axis=0)
        df = df.reindex(natsorted(df.columns), axis=1)
        df = df.reset_index()
        df = df.rename(columns={'index': 'old_index'})
        df = df[~df.old_index.str.contains("n_")] 
        self.adj_matrix = df
        self.all_node_vt = df_2

    def eval_node_val(self):
        '''
        :return: Return a model to evaluate node
        '''
        self.solver_node_val.reset()
        self.solver_node_val.add(self.constraints[:-1])
        self.solver_node_val.check()
        self.model4evl = self.solver_node_val.model()

    def calculate_node_value(self, node, node_id):
        '''
        :return: the node value -> true or false
        '''
        if self.mode == 0:
            pass
        elif self.mode == 1:
            self.solver.reset()
            self.solver.add(self.constraints[:-1])
            self.solver.add(node)
            if self.solver.check() == z3.sat:
                self.all_node_var[node_id] = 1 #-->sat so assign 1 as true
            else:
                self.all_node_var[node_id] = 0 #--> unsat so assign 0 as false


class problem:
    def __init__(self, raw_data, aigname, mode=1,latch_lst=None): 
        self.raw_data = raw_data

        if mode==1:
            self.filename = "../dataset/IG2graph/generalization/" + (aigname.split('/')[-1]).replace('.aag', '.csv')
        elif mode==2:
             self.filename = "../dataset/IG2graph/generalization_no_enumerate/" + (aigname.split('/')[-1]).replace('.aag', '.csv')
        # check self.file is exist or not
        if not os.path.isfile(self.filename):
            self.db_gt = natsorted(latch_lst)
        elif os.path.isfile(self.filename):
            self.db_gt = pd.read_csv(self.filename) #ground truth of the label of literals (database) -> #TODO: refine here, only get one line for one object
            self.db_gt.drop("Unnamed: 0", axis=1, inplace=True)
            self.db_gt = self.db_gt.reindex(natsorted(self.db_gt.columns), axis=1)
        self.unpack_matrix = raw_data[0]
        self.value_table = raw_data[1]
        self.n_vars = self.unpack_matrix.shape[1] - 1 #includes m and variable
        self.n_nodes = self.n_vars - (self.value_table[~self.value_table.index.str.contains('m_')]).shape[0]
        self.adj_matrix = self.unpack_matrix.copy()
        self.adj_matrix = self.adj_matrix.T.reset_index(drop=True).T
        self.adj_matrix.drop(self.adj_matrix.columns[0], axis=1, inplace=True)
        self.edges, self.relations, self.node_ref = raw_data[2][0],raw_data[2][1],raw_data[2][2]
        self.ig_q = raw_data[3]
        self.refined_output = []
    


def run(solver,aigname,mode=0, latch_lst=None):
    if mode == 0: # mode == 'generalized predecessor'
        pass
    elif mode == 1: # mode == 'inductive generalization'
        res, node_ref = mk_adj_matrix(solver,mode)
        adj_matrix_pkl_list = res.adj_matrix
        vt_all_node_pkl_list = res.all_node_vt
        edge_and_relation_pkl_list = [res.edges, res.relations, node_ref]
        q_literal_lst = res.q_lst
        raw_data = [adj_matrix_pkl_list, vt_all_node_pkl_list, edge_and_relation_pkl_list, q_literal_lst]
        prob = problem(raw_data,aigname)
        return prob
    elif mode == 2: # mode == 'inductive generalization upgrade verison, only consider !s & s''
        res, node_ref = mk_adj_matrix(solver,mode=1)
        adj_matrix_pkl_list = res.adj_matrix
        vt_all_node_pkl_list = res.all_node_vt
        edge_and_relation_pkl_list = [res.edges, res.relations, node_ref]
        q_literal_lst = res.q_lst
        raw_data = [adj_matrix_pkl_list, vt_all_node_pkl_list, edge_and_relation_pkl_list, q_literal_lst]
        prob = problem(raw_data,aigname,mode=2, latch_lst=latch_lst)
        return prob
        