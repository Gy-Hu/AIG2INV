import math
import random
import torch
from torch import nn
import torch.nn.init as init
import numpy as np
import copy
from torch_geometric.data import Data



class MLP(nn.Module):
  def __init__(self, in_dim, hidden_dim, out_dim):
    super(MLP, self).__init__()
    self.l1 = nn.Linear(in_dim, hidden_dim)
    self.f1 = nn.ReLU()
    self.l2 = nn.Linear(hidden_dim, hidden_dim)
    self.f2 = nn.ReLU()
    self.l3 = nn.Linear(hidden_dim, out_dim)

  def forward(self, x):
    x = self.l1(x)
    x = self.f1(x)
    x = self.l2(x)
    x = self.f2(x)
    output = self.l3(x)

    return output


class NeuroGraph(nn.Module):
    def __init__(self, nvt=7, vhs=100, chs=30, nrounds=100):
        super(NeuroGraph, self).__init__()
        self.vhs = vhs
        self.nvt = nvt
        self.chs = chs
        self.nrounds = nrounds

        self.device = None

        # AND=0
        # NOT=1
        # SV=2
        # INP=3
        # FALSE=4
        # LI_UPDATE=5
        # OUT_UPDATE=6

        self.init_ts = torch.ones(1)
        self.init_ts.requires_grad = False

        # these are the trainable parameters for different node type
        self.and_init = nn.Linear(1,self.vhs)
        self.not_init = nn.Linear(1, self.vhs)
        self.sv_init = nn.Linear(1,self.vhs)
        self.inp_init = nn.Linear(1, self.vhs)
        self.false_init = nn.Linear(1,self.vhs)
        self.li_init = nn.Linear(1, self.vhs)
        self.out_init = nn.Linear(1, self.vhs)

        self.forward_msg = MLP(self.vhs, self.vhs, self.vhs) #for children to pass message
        self.backward_msg = MLP(self.vhs, self.vhs, self.vhs) #for parents to pass message

        self.forward_update = nn.GRU(self.vhs, self.vhs) #update node (exclude variable)
        self.backward_update = nn.GRU(self.vhs, self.vhs) #udpate variable and node

        # for clause generation
        self.lstm_clause_gen = nn.LSTM(input_size = 2*self.vhs, hidden_size = self.vhs, batch_first = True, bidirectional = True)
        # expect [1 x n_input_node x self.vhs]
        # output [1 x n_input_node x self.vhs] then go through this mapper to -1 0 1

        self.lstm_clause_gen_mapper = nn.Sequential(
                nn.Linear(self.vhs, self.chs),
                nn.ReLU(),
                nn.Linear(self.chs, 1),
                nn.Tanh()
            )
        self.lstm_clause_gen2to1 = nn.Linear(self.vhs*2, self.vhs)

        self.lstm_clause_update = nn.LSTM(input_size = self.vhs, hidden_size = self.vhs, batch_first = True, bidirectional = True)
        self.lstm_clause_update2to1 = nn.Linear(self.vhs*2, self.vhs)

        self.sv_feature_pretrain_layer = nn.Sequential(
            nn.Linear(self.vhs, self.chs),
            nn.ReLU(),
            nn.Linear(self.chs, 1)
            )


    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device

    def forward(self, G, n_clause, transfer_to_device, pretrain_round):
        if transfer_to_device:
            G.x = G.x.to(self.get_device())
            G.sv_node = G.sv_node.to(self.get_device())

            G.edge_index = G.edge_index.to(self.get_device())
            G.output_node = G.output_node.to(self.get_device())
            G.li_node = G.li_node.to(self.get_device())
            G.ind = G.ind.to(self.get_device())
            G.outd = G.outd.to(self.get_device())

        num_nodes_batch = G.x.shape[0]
        num_sv_node = G.sv_node.shape[0]
        assert G.x.shape[1] == self.nvt

        # TODO: build the adjacent matrix
        n_edge = G.edge_index.shape[1]
        edgeones = torch.ones(n_edge, device=self.get_device())
        edge_backward = torch.vstack( (G.edge_index[1],G.edge_index[0]) )
        adj_mat_bwd = torch.sparse_coo_tensor(indices=G.edge_index, values=edgeones, size=(num_nodes_batch,num_nodes_batch))
        adj_mat_fwd = torch.sparse_coo_tensor(indices=edge_backward, values=edgeones, size=(num_nodes_batch,num_nodes_batch))

        init_ts = self.init_ts.to(self.get_device())
        type_vec = torch.vstack(\
          ( self.and_init(init_ts), \
            self.not_init(init_ts), \
            self.sv_init (init_ts), \
            self.inp_init(init_ts), \
            self.false_init(init_ts),\
            self.li_init(init_ts),  \
            self.out_init(init_ts)))

        assert self.nvt == type_vec.shape[0]
        assert type_vec.shape[1] == self.vhs
        var_state = torch.matmul(G.x, type_vec) # this is the initial hidden state vector
        
        ind = G.ind.unsqueeze(0).t()
        outd = G.outd.unsqueeze(0).t()
    
        for _ in range(self.nrounds):
            # forward
            var_pre_msg = self.forward_msg(var_state)
            fwd_msg = torch.matmul(adj_mat_fwd, var_pre_msg)
            fwd_msg = fwd_msg/ind

            # you may want to normalize here ?

            inp = fwd_msg.unsqueeze(0)
            h = var_state.unsqueeze(0) # shape: 1 x n_node x vhs
            var_state, _ = self.forward_update(inp, h)  # basically, the first and second input are the same
            var_state = var_state[0]

            # backward
            var_pre_msg = self.backward_msg(var_state)
            bwd_msg = torch.matmul(adj_mat_bwd, var_pre_msg)
            bwd_msg = bwd_msg/outd
            
            # you may want to normalize here ?

            inp = bwd_msg.unsqueeze(0)
            h = var_state.unsqueeze(0)
            var_state, _ = self.backward_update(inp, h)
            var_state = var_state[0]

        # end of for each round

        sv_node = G.sv_node
        variance = torch.sum(torch.var(var_state[sv_node], dim=0))
        G.variance = variance

        
        shortcut = var_state[sv_node]

        if pretrain_round:
            return (self.sv_feature_pretrain_layer(shortcut).t())[0]

        sv_feature = var_state[sv_node].unsqueeze(0)
        #n_clause = len(G.clauses.clauses)
        result_clauses = []
        for idx in range(n_clause):
            prop_feature, _ = self.lstm_clause_update(sv_feature)
            clause_predict, _ = self.lstm_clause_gen(prop_feature)
            sv_feature = self.lstm_clause_update2to1(prop_feature)

            clause_pred2to1 = self.lstm_clause_gen2to1(clause_predict[0])
            clause_generated = self.lstm_clause_gen_mapper(clause_pred2to1 + shortcut)
            
            prop_feature_has_nan = torch.any(torch.isnan(prop_feature))
            sv_feature_has_nan = torch.any(torch.isnan(sv_feature))
            clause_generated_has_nan = torch.any(torch.isnan(clause_generated))

            if prop_feature_has_nan or sv_feature_has_nan or clause_generated_has_nan:
                print('sv_feature max abs', torch.max(torch.abs(sv_feature)))
                print('clause_predict max abs', torch.max(torch.abs(clause_predict)))
                print('prop_feature max abs', torch.max(torch.abs(prop_feature)))

            result_clauses.append(clause_generated.t())

        # check the result size
        result_clauses = torch.cat(result_clauses, dim=0)
        return result_clauses
