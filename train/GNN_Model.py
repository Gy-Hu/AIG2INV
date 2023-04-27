import json
import torch
import dgl
import os
import numpy as np
import networkx as nx
import numpy as np
import pandas as pd
from dgl.nn import GraphConv, ChebConv, GATConv, SAGEConv, GINConv
from GlobalVar import *
from torch import nn
from torch.optim import Adam
import dgl.data
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from dgl.dataloading import GraphDataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import math
import dgl
import sympy
import scipy
import numpy as np
from torch import nn
from torch.nn import init
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm, ChebConv, GATConv, HeteroGraphConv

'''
---------------------------------SAGE-BW---------------------------------
'''
def calculate_theta2(d):
    '''
    Adaptive filtering: The model learns the coefficients of the polynomial (theta) using a technique called "theta-calculation",
    which allows it to adaptively capture different types of graph structures. 
    This adaptability can help the model to be more robust when dealing with imbalanced data.
    '''
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        # Beta-Binomial probability distribution
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = [float(coeff[d-i]) for i in range(d+1)]
        thetas.append(inv_coeff)
    return thetas

class SagePolyConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 theta,
                 num_sample_neighbors=5,  # new parameter
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(SagePolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.num_sample_neighbors = num_sample_neighbors  # store the parameter
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin

    def forward(self, graph, feat):

        # Sample neighbors
        sampled_graph = dgl.sampling.sample_neighbors(graph, graph.nodes(), self.num_sample_neighbors)
        
        # Move the graph to CPU before calling to_bidirected(), and then move it back to the device (GPU)
        sampled_graph = sampled_graph.to('cpu')
        sampled_graph = dgl.to_bidirected(sampled_graph)
        sampled_graph = sampled_graph.to(feat.device)


        # Now work with the sampled graph
        with sampled_graph.local_scope():
            D_invsqrt = torch.pow(sampled_graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0] * feat #cheb/GCN theta is constant value
            for k in range(1, self._k):
                sampled_graph.ndata['h'] = feat * D_invsqrt
                sampled_graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                feat = feat - sampled_graph.ndata.pop('h') * D_invsqrt
                h += self._theta[k] * feat

        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
        return h

class SAGE_BW(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, d=2, num_sample_neighbors=5, batch=False):
        super(SAGE_BW, self).__init__()
        self.dropout = nn.Dropout(DROPOUT)
        self.thetas = calculate_theta2(d=d)
        self.conv = nn.ModuleList()  # Use nn.ModuleList to store layers
        for i in range(len(self.thetas)):
            if not batch:
                self.conv.append(SagePolyConv(h_feats, h_feats, self.thetas[i], num_sample_neighbors, lin=False))
            else:
                assert False
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.d = d

    def forward(self, g, in_feat):
        self.g = g
        device = in_feat.device
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0]).to(device)
        for conv in self.conv:
            self.g = self.g.to(device)
            h0 = conv(self.g, h)
            h_final = torch.cat([h_final, h0], -1)
        h = self.linear3(h_final)
        h = self.act(h)
        h = self.linear4(h)
        return h
    
'''
-----------------------------------GCN-----------------------------------
'''

class GCNModel(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCNModel, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_size, num_classes, allow_zero_in_degree=True)
        self.relu = nn.ReLU()

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = self.relu(h)
        h = self.conv2(g, h)
        return h

'''
----------------------------------DualGCN----------------------------------
'''
class DualGCNModel(nn.Module):
    def __init__(self, in_feats_ori, in_feats_struc, hidden_size, num_classes, mlp_hidden_size):
        super(DualGCNModel, self).__init__()
        self.conv1_ori = GraphConv(in_feats_ori, hidden_size, allow_zero_in_degree=True)
        self.conv2_ori = GraphConv(hidden_size, num_classes, allow_zero_in_degree=True)
        
        self.conv1_struc = GraphConv(in_feats_struc, hidden_size, allow_zero_in_degree=True)
        self.conv2_struc = GraphConv(hidden_size, num_classes, allow_zero_in_degree=True)

        self.relu = nn.ReLU()
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_classes, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, num_classes)
        )
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, g, ori_feat, struc_feat):
        h_ori = self.conv1_ori(g, ori_feat)
        h_ori = self.relu(h_ori)
        h_ori = self.conv2_ori(g, h_ori)

        h_struc = self.conv1_struc(g, struc_feat)
        h_struc = self.relu(h_struc)
        h_struc = self.conv2_struc(g, h_struc)

        h_concat = torch.cat((h_ori, h_struc), dim=1)
        h_reduced = self.mlp(h_concat)
        return h_reduced

'''
-----------------------------------GraphSAGE-----------------------------------
'''
class SAGE(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, aggregator_type='gcn', dropout=0):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, hidden_size, aggregator_type))
        self.layers.append(SAGEConv(hidden_size, hidden_size, aggregator_type))
        self.layers.append(SAGEConv(hidden_size, hidden_size, aggregator_type))
        self.layers.append(SAGEConv(hidden_size, hidden_size, aggregator_type))
        self.layers.append(SAGEConv(hidden_size, num_classes, aggregator_type))
        self.dropout = nn.Dropout(DROPOUT)
        self.relu = nn.ReLU()

    def forward(self, graph, feat, eweight=None):
        x = feat # x is the feature of the node
        for l, layers in enumerate(self.layers):
            x = layers(graph, x)
            if l != len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x

'''
-----------------------------------DualGraphSage-------------------------
'''
class DualGraphSAGEModel(nn.Module):
    def __init__(self, in_feats_ori, in_feats_struc, hidden_size, num_classes, mlp_hidden_size):
        super(DualGraphSAGEModel, self).__init__()
        self.sageconv1_ori = SAGEConv(in_feats_ori, hidden_size, 'mean')
        self.sageconv2_ori = SAGEConv(hidden_size, num_classes, 'mean')
        
        self.sageconv1_struc = SAGEConv(in_feats_struc, hidden_size, 'mean')
        self.sageconv2_struc = SAGEConv(hidden_size, num_classes, 'mean')

        self.relu = nn.ReLU()
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_classes, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, num_classes)
        )

    def forward(self, g, ori_feat, struc_feat):
        h_ori = self.sageconv1_ori(g, ori_feat)
        h_ori = self.relu(h_ori)
        h_ori = self.sageconv2_ori(g, h_ori)

        h_struc = self.sageconv1_struc(g, struc_feat)
        h_struc = self.relu(h_struc)
        h_struc = self.sageconv2_struc(g, h_struc)

        h_concat = torch.cat((h_ori, h_struc), dim=1)
        h_reduced = self.mlp(h_concat)
        return h_reduced
    
'''
-----------------------------------GAT-----------------------------------
'''

class GATModel(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, num_heads=1):
        super(GATModel, self).__init__()
        self.gatconv1 = GATConv(in_feats, hidden_size, num_heads=num_heads)
        self.gatconv2 = GATConv(hidden_size * num_heads, num_classes, num_heads=1)
        self.relu = nn.ReLU()

    def forward(self, g, in_feat):
        h = self.gatconv1(g, in_feat)
        h = h.view(h.shape[0], -1)  # Reshape to (N, hidden_size * num_heads)
        h = self.relu(h)
        h = self.gatconv2(g, h)
        h = h.squeeze(1)  # Remove the extra dimension of the output shape (N, 1, num_classes) -> (N, num_classes)
        return h

'''
-----------------------------------GIN-----------------------------------
Graph Isomorphism Networks
'''
class GIN(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, num_layers=3):
        super(GIN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GINConv(in_feats, hidden_size, 'sum'))
        
        for _ in range(num_layers - 1):
            self.layers.append(GINConv(hidden_size, hidden_size, 'sum'))
            
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, g, in_feat):
        h = in_feat
        for layer in self.layers:
            h = torch.nn.functional.relu(layer(g, h))
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.fc(hg)
