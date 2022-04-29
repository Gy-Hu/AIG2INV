import math
import random
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
import numpy as np
import copy
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing



'''
    Circuit-SAT.
    The model used in this paper named Deep-Gated DAG Recursive Neural Networks (DG-DARGNN).
'''
class DGDAGRNN(nn.Module):
    '''
    The implemnetation of DGDAGRNN with Pytorch Geometric.
    Attributes:
        nvt (integer, default: 3) - # vertex types.
        vhs (integer, default: 100) - the size of hidden state of nodes.
        chs (integer, default: 30) - the size of hidden state of classifier.
        temperature (float, default: 5.0) - the initial value of temperature for soft MIN.
        kstep (float, default: 10.0) - the value of k in soft step function.
        num_rounds (integer, default: 10) - # GRU iterations. 
    '''
    def __init__(self,  nvt=6, vhs=100,  chs=30, nrounds=10):
        super(DGDAGRNN, self).__init__()
        self.nvt = nvt  # number of vertex types
        self.vhs = vhs  # hidden state size of each vertex
        self.chs = chs
        self.nrounds = nrounds
        self.num_layers = 1 # one forward and no backword

        self.device = None

        # 0. GRU-related
        self.init_ts = torch.ones(1)
        self.init_vector_for_nodes = nn.Linear(1, self.vhs, bias=False)

        self.grue_forward_input = nn.GRUCell(self.nvt, self.vhs)  # encode input & message into states
        self.grue_forward_hidden = nn.GRUCell(self.vhs, self.vhs)  # encoder old_state & message into states
        # self.grue_backward = nn.GRUCell(self.vhs, self.vhs)  # backward encoder GRU
        self.gru_latch_li_lo_update = nn.GRUCell(self.vhs, self.vhs)
        self.gru_latch_output_lo_update = nn.GRUCell(self.vhs, self.vhs)


        # 2. gate-related, aggregate
        num_rels = 1    # num_relationship
        self.gate_forward = nn.Sequential(
                nn.Linear(self.vhs, self.vhs), 
                nn.Sigmoid()
                )
        # self.gate_backward = nn.Sequential(
        #         nn.Linear(self.vhs, self.vhs), 
        #         nn.Sigmoid()
        #         )
        self.mapper_forward = nn.Sequential(
                nn.Linear(self.vhs, self.vhs, bias=False),
                )  # disable bias to ensure padded zeros also mapped to zeros
        # self.mapper_backward = nn.Sequential(
        #         nn.Linear(self.vhs, self.vhs, bias=False), 
        #         )

        self.node_aggr_forward = GatedSumConv(self.vhs, num_rels, mapper=self.mapper_forward, gate=self.gate_forward)
        # self.node_aggr_backward = GatedSumConv(self.vhs, num_rels, mapper=self.mapper_backward, gate=self.gate_backward, reverse=True)
        
        # lstm ?
        self.lstm_clause_gen = nn.LSTM(input_size = 2*self.vhs, hidden_size = self.vhs, batch_first = True, bidirectional = True)
        # expect [1 x n_input_node x self.vhs]
        # output [1 x n_input_node x self.vhs] then go through this mapper to -1 0 1

        self.lstm_clause_gen_mapper = nn.Sequential(
                nn.Linear(self.vhs*2, self.chs),
                nn.ReLU(),
                nn.Linear(self.chs, 1),
                nn.Tanh()
            )

        self.lstm_clause_update = nn.LSTM(input_size = self.vhs, hidden_size = self.vhs, batch_first = True, bidirectional = True)
        self.lstm_clause_update2to1 = nn.Linear(self.vhs*2, self.vhs)
        # expect [1 x n_input_node x self.vhs]




    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device
    
    def _get_zeros(self, n, length):
        return torch.zeros(n, length).to(self.get_device()) # get a zero hidden state

    def _get_zero_hidden(self, n=1):
        return self._get_zeros(n, self.hs) # get a zero hidden state

    def _one_hot(self, idx, length):
        if type(idx) in [list, range]:
            if idx == []:
                return None
            idx = torch.LongTensor(idx).unsqueeze(0).t()
            x = torch.zeros((len(idx), length)).scatter_(1, idx, 1).to(self.get_device())
        else:
            idx = torch.LongTensor([idx]).unsqueeze(0)
            x = torch.zeros((1, length)).scatter_(1, idx, 1).to(self.get_device())
        return x

    def _collate_fn(self, G):
        return [copy.deepcopy(g) for g in G]

    def forward(self, G, n_clause, transfer_to_device):
        # GNN computation to get node embeddings
        if transfer_to_device:
            G.x = G.x.to(self.get_device())
            G.sv_node = G.sv_node.to(self.get_device())
            G.forward_layer_index = G.forward_layer_index.to(self.get_device())
            G.backward_layer_index = G.backward_layer_index.to(self.get_device())
            G.edge_index = G.edge_index.to(self.get_device())
            G.output_node = G.output_node.to(self.get_device())
            G.li_node = G.li_node.to(self.get_device())

        num_nodes_batch = G.x.shape[0]
        num_sv_node = G.sv_node.shape[0]
        # print('# nodes for this batch: ', num_nodes_batch)
        num_layers_batch = max(G.forward_layer_index[0]).item() + 1
        # print('# layers for this batch: ', num_layers_batch)

        node_init_val = self.init_vector_for_nodes(self.init_ts.to(self.get_device()))
        G.h = node_init_val.repeat(num_nodes_batch, 1)
        # print('Size of hidden states: ', G.h.size())
        
        
        # forward
        for round_idx in range(self.nrounds):
            # print('######## Round: ', round_idx)
            
            # forwarding
            # print('Forwarding...')
            for l_idx in range(num_layers_batch):
                # print('# layer: ', l_idx)
                layer = G.forward_layer_index[0] == l_idx # pick those which can be handled now
                layer = G.forward_layer_index[1][layer]   # the vertices ID for this batch layer

                if round_idx == 0:
                    inp = G.x[layer]    # input node feature vector
                else:
                    inp = G.h[layer]

                # print("Input feature size: ", inp.size())
                
                if l_idx > 0:   # no predecessors at first layer
                    le_idx = []
                    for n in layer:
                        ne_idx = G.edge_index[1] == n
                        le_idx += [torch.nonzero(ne_idx, as_tuple=False).squeeze(-1)]    # the index of edge edge in edg_index
                    le_idx = torch.cat(le_idx, dim=-1)
                    lp_edge_index = G.edge_index[:, le_idx] # the subset of edge_idx which contains the target vertices ID
                
                    hs1 = G.h
                    ps_h = self.node_aggr_forward(hs1, lp_edge_index, edge_attr=None)[layer]
                    # print('Aggregated hidden size: ', ps_h.size())
                    if round_idx == 0:
                        G.h[layer] = self.grue_forward_input(inp, ps_h)
                    else:
                        G.h[layer] = self.grue_forward_hidden(inp, ps_h)

            # end of forward propagation
            # now update all latch node
            sv_prev = G.h[G.sv_node]
            li = G.h[G.li_node]
            output = G.h[G.output_node]
            output_repeat = output.repeat(num_sv_node, 1)


            after_li_update = self.gru_latch_li_lo_update( li , sv_prev )
            after_out_update = self.gru_latch_output_lo_update( output_repeat , after_li_update )
            G.h[G.sv_node] = after_out_update

            # backwording
            # print('Backwarding')
            # for l_idx in range(num_layers_batch):
            #     # print('# layer: ', l_idx)
            #     layer = G.backward_layer_index[0] == l_idx
            #     layer = G.backward_layer_index[1][layer]   # the vertices ID for this batch layer

            #     inp = G.h[layer]
            #     # print("Input feature size: ", inp.size())

            #     if l_idx > 0:   # no predecessors at first layer
            #         le_idx = []
            #         for n in layer:
            #             ne_idx = G.edge_index[0] == n
            #             le_idx += [torch.nonzero(ne_idx, as_tuple=False).squeeze(-1)]    # the index of edge edge in edg_index
            #         le_idx = torch.cat(le_idx, dim=-1)
            #         lp_edge_index = G.edge_index[:, le_idx] # the subset of edge_idx which contains the target vertices ID
                
            #         # HZ: We don't update the output layer at this time
            #         hs1 = G.h
            #         all_nodes_msg = self.node_aggr_backward(hs1, lp_edge_index, edge_attr=None)
            #         ps_h = all_nodes_msg[layer]
            #         # print('Aggregated hidden size: ', ps_h.size())
            #         G.h[layer] = self.grue_backward(inp, ps_h)



        # 1. find the nodes for state vars
        sv_node = G.sv_node
        sv_feature = G.h[sv_node].unsqueeze(0)
        #n_clause = len(G.clauses.clauses)
        result_clauses = []
        for idx in range(n_clause):
            prop_feature, _ = self.lstm_clause_update(sv_feature)
            clause_predict, _ = self.lstm_clause_gen(prop_feature)
            sv_feature = self.lstm_clause_update2to1(prop_feature)

            clause_generated = self.lstm_clause_gen_mapper(clause_predict[0])
            result_clauses.append(clause_generated.t())

        result_clauses = torch.cat(result_clauses, dim=0)
        return result_clauses


class GatedSumConv(MessagePassing):  # dvae needs outdim parameter
    '''
    Some parameter definitions:
        num_relations (integer): the edge types. If 1, then no information from edge attribute.
                        if not zero, then it reprensent the number of edge types.
        wea (bool): with edge attributes. If num_relations > 1, then the graph is with edge attributes.
        edge_encoder: cast the one-hot edge feature vector into emb_dim size.
    It is not the exactly Deep-Set. Should implement DeepSet later.
    Consider change `aggr` from 'add' to 'mean'. It makes sense when there are AND gates and NOT gates.
    '''
    def __init__(self, emb_dim, num_relations=1, reverse=False, mapper=None, gate=None):
        super(GatedSumConv, self).__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')

        assert emb_dim > 0
        if num_relations > 1:
            self.wea = True
            self.edge_encoder = torch.nn.Linear(num_relations, emb_dim)
        else:
            self.wea = False
        self.mapper = nn.Linear(emb_dim, emb_dim) if mapper is None else mapper
        self.gate = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.Sigmoid()) if gate is None else gate

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        # HACK assume x contains only message sources
        if edge_index is None:
            h = self.gate(x) * self.mapper(x)
            return torch.sum(h, dim=1)

        edge_embedding = self.edge_encoder(edge_attr) if self.wea else None
        return self.propagate(edge_index, x=x, edge_attr=edge_embedding)

    def message(self, x_j, edge_attr):
        h_j = x_j + edge_attr if self.wea else x_j
        return self.gate(h_j) * self.mapper(h_j)

    def update(self, aggr_out):
        return aggr_out
