#import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

'''
----------------------------Convert to Hypergraph----------------------------

def convert_hypergraph(graph): #dgl graph as input
    indices = torch.stack(graph.edges())
    H = dglsp.spmatrix(indices)
    H = H + dglsp.identity(H.shape)

    X = graph.ndata["feat"]
    Y = graph.ndata["label"]
    train_mask = graph.ndata["train_mask"]
    #val_mask = graph.ndata["val_mask"]
    #test_mask = graph.ndata["test_mask"]
    return H, X, Y, train_mask

# this was given by LLM
def dgl_to_hypergraph(dgl_graph):
    # Extract edge and node information from the DGL graph
    src, dst = dgl_graph.edges()
    num_nodes = dgl_graph.number_of_nodes()
    
    # Initialize a dictionary to store the hyperedges
    hyperedges = {}
    
    # Iterate over the edges and create hyperedges
    for s, d in zip(src, dst):
        edge_id = len(hyperedges)
        hyperedges[edge_id] = [int(s), int(d)]
    
    # Construct a hypergraph adjacency matrix
    adjacency_matrix = np.zeros((num_nodes, len(hyperedges)), dtype=int)
    
    for edge_id, nodes in hyperedges.items():
        for node in nodes:
            adjacency_matrix[node, edge_id] = 1
    
    # The adjacency matrix represents the hypergraph
    return adjacency_matrix, hyperedges
'''

'''
----------------------------Convert to heterograph----------------------------

def convert_heterograph(graph): #dgl graph as input
    pass
'''

def update_adj_cosine_d(graph_list_encoded,graph_list_struc_feat):
    # Iterate through the tuples in both lists
    for (G_ori, _,__,___), (G_struc, struc_feat, ____, _____) in zip(graph_list_encoded, graph_list_struc_feat):
        # Calculate cosine distances between the node embeddings
        dist_matrix = cosine_distances(struc_feat)
        # get the nodes list
        nodes_list = list(G_ori.nodes(data=True))
        # Update the adjacency matrix of the graph in graph_list_ori_feat
        for i in range(dist_matrix.shape[0]):
            for j in range(dist_matrix.shape[1]):
                if i == j: continue
                # You can set a threshold or use a custom function to decide when to update the adjacency matrix
                
                '''
                i, j is the index of the node in the graph (not the node id)
                so we need to get the node id from the nodes_list
                '''
                # if distances are less than 0.5, and (i,j) in graph is state variable, then add edge
                if dist_matrix[i, j] < 0.5 and  G_ori.nodes[nodes_list[i][0]]['type'] == 'variable' and  G_ori.nodes[nodes_list[j][0]]['type'] == 'variable':
                    G_ori.add_edge(i, j)