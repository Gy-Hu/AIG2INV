import json
import torch
import dgl
import os
import numpy as np
import networkx as nx
import numpy as np
import pandas as pd
from dgl.nn import GraphConv, ChebConv, GATConv, SAGEConv, GINConv
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
---------------------------------BWGNN---------------------------------
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

class PolyConv(nn.Module):
    '''
    Polynomial filtering: The PolyConv layer is designed to learn graph convolutional filters based on polynomial filtering,
    which allows the model to capture different levels of graph information in different scales. 
    This can be helpful in imbalanced data scenarios where certain classes might have different graph structures.
    '''
    def __init__(self,
                 in_feats,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin
        # self.reset_parameters()
        # self.linear2 = nn.Linear(out_feats, out_feats, bias)

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph): #normalized Laplacian transformation
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt
        
        '''
        Set the local scope for the graph: 
        This ensures that any updates made to the graph's node/edge features are discarded once the forward pass is complete. 
        It helps to maintain the original graph structure for future operations.
        '''
        with graph.local_scope():
            #Compute the inverse square root of the node degrees for normalization
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device) 
            h = self._theta[0]*feat #Polynomial expansion
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k]*feat
        '''
        This can introduce more non-linearity and learnable parameters to the model, 
        potentially enhancing its expressiveness and learning capacity.
        Possible to make the training time longer
        '''
        if self.lin: #Apply the optional linear transformation and activation function
            h = self.linear(h)
            h = self.activation(h)
        return h


class PolyConvBatch(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConvBatch, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, block, feat):
        def unnLaplacian(feat, D_invsqrt, block):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            block.srcdata['h'] = feat * D_invsqrt
            block.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - block.srcdata.pop('h') * D_invsqrt

        with block.local_scope():
            D_invsqrt = torch.pow(block.out_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, block)
                h += self._theta[k]*feat
        return h


class BWGNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, d=2, batch=False):
        super(BWGNN, self).__init__()
        '''
        Model architecture: The BWGNN model has multiple linear layers and activation functions, 
        which increases its capacity to learn complex patterns in the data. 
        This increased capacity can help the model perform better on imbalanced datasets.
        '''
        self.dropout = nn.Dropout(0.8)
        self.thetas = calculate_theta2(d=d)
        self.conv = []
        for i in range(len(self.thetas)):
            if not batch:
                '''
                PolyConv implements a polynomial-based graph convolution operation,
                which can adaptively capture different types of graph structures,
                by learning the coefficients of the polynomial basis functions
                '''
                self.conv.append(PolyConv(h_feats, h_feats, self.thetas[i], lin=False))
            else:
                self.conv.append(PolyConvBatch(h_feats, h_feats, self.thetas[i], lin=False))
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats*len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        '''
        ReLU activation: The model uses ReLU activation functions, 
        which are known to have better performance in handling imbalanced datasets compared to other activation functions. 
        ReLU helps in mitigating the vanishing gradient problem, allowing the model to learn better representations for minority classes.
        '''
        self.d = d

    def forward(self, g, in_feat):
        self.g = g
        device = in_feat.device
        h = self.linear(in_feat)
        #h = self.dropout(h)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0]).to(device)
        for conv in self.conv:
            self.g = self.g.to(device)
            h0 = conv(self.g, h)
            h_final = torch.cat([h_final, h0], -1)
            # print(h_final.shape)
            '''
            Concatenation of features: In the BWGNN model, 
            the output features of different PolyConv layers are concatenated together before being fed to the next layer. 
            This allows the model to learn a richer representation of the input features, making it more capable of handling imbalanced data.
            '''
        h = self.linear3(h_final)
        h = self.act(h)
        h = self.linear4(h)
        return h

# heterogeneous graph
class BWGNN_Hetero(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph, d=2):
        '''
        By iterating over the canonical edge types (relations) in the graph, 
        the model is able to learn separate representations for different types of relationships. 
        This is especially useful when the graph data is complex and diverse, 
        as the model can capture different aspects of the graph structure.
        '''
        super(BWGNN_Hetero, self).__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.h_feats = h_feats
        self.conv = [PolyConv(h_feats, h_feats, theta, lin=False) for theta in self.thetas]
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats*len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.LeakyReLU()
        '''
        LeakyReLU activation: The model uses LeakyReLU activation functions instead of ReLU. 
        LeakyReLU is known to improve the performance of models when dealing with imbalanced datasets, 
        as it allows a small gradient for negative values, 
        thus mitigating the vanishing gradient problem and helping the model learn better representations for minority classes.
        '''
        # print(self.thetas)
        for param in self.parameters():
            print(type(param), param.size())

    def forward(self, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_all = []

        for relation in self.g.canonical_etypes:
            # print(relation)
            h_final = torch.zeros([len(in_feat), 0])
            for conv in self.conv:
                h0 = conv(self.g[relation], h)
                h_final = torch.cat([h_final, h0], -1)
                # print(h_final.shape)
            h = self.linear3(h_final)
            h_all.append(h)

        h_all = torch.stack(h_all).sum(0)
        h_all = self.act(h_all)
        h_all = self.linear4(h_all)
        return h_all
    
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
-----------------------------------GraphSAGE-----------------------------------
'''
class SAGEConvModel(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, aggregator_type='mean'):
        super(SAGEConvModel, self).__init__()
        self.sageconv1 = SAGEConv(in_feats, hidden_size, aggregator_type)
        self.sageconv2 = SAGEConv(hidden_size, num_classes, aggregator_type)
        self.relu = nn.ReLU()

    def forward(self, g, in_feat):
        h = self.sageconv1(g, in_feat)
        h = self.relu(h)
        h = self.sageconv2(g, h)
        return h
    
    
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
