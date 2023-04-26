import numpy as np
import networkx as nx
import json
import os
import pandas as pd
from math import sin, cos, pi
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor



def count_state_variable_neighbors(G, node_id):
    neighbors = list(G.neighbors(node_id))
    count = 0
    for neighbor_id in neighbors:
        if G.nodes[neighbor_id]["type"] == "variable" and G.nodes[neighbor_id]["application"].startswith("v"):
            count += 1
    return count

# Compute position embedding for a given node
def position_embedding(node_id, max_id):
    pos_emb = np.array([sin(node_id * pi / max_id), cos(node_id * pi / max_id)])
    return pos_emb

# Compute distance to root for each node
def distance_to_root(G, root_id):
    distances = nx.shortest_path_length(G, root_id)
    return distances

# Compute the number of neighboring operators for a given node
def count_neighbor_operators(G, node_id):
    neighbors = list(G.neighbors(node_id))
    count = 0
    for neighbor_id in neighbors:
        if G.nodes[neighbor_id]["type"] == "node":
            count += 1
    return count

def one_hot_encoding(node_data):
    kind_encoding = [0] if node_data['type'] == 'node' else [1]
    value_encoding = None; 

    if node_data['type'] == 'variable':
        if node_data['application'].startswith('v'):
            value_encoding = [1, 0, 0, 0, 0 ,0, 0]
        elif node_data['application'].startswith('i'):
            value_encoding = [0, 1, 0, 0, 0, 0, 0]
        elif node_data['application'].startswith('constant_t'):
            value_encoding = [0, 0, 1, 0, 0, 0, 0]
        elif node_data['application'].startswith('constant_f'):
            value_encoding = [0, 0, 0, 1, 0, 0, 0]

    elif node_data['type'] == 'node':
        if node_data['application'] == 'and':
            value_encoding = [0, 0, 0, 0, 1, 0, 0]
        elif node_data['application'] == 'or':
            value_encoding = [0, 0, 0, 0, 0, 1, 0]
        elif node_data['application'] == 'not':
            value_encoding = [0, 0, 0, 0, 0, 0, 1]
            
    assert (
        len(kind_encoding) == 1 and value_encoding is not None
    ), f"Encoding length is not correct, node data: {node_data}"
    return kind_encoding, value_encoding

def fout_embedding(G, node_id):
    # find all the neighbors of the node that point to it
    # and count the number of each type of node
    neighbors = list(G.predecessors(node_id))
    fout = np.array([0 , 0 , 0]) #represent the number of Not, And, Or
    for _ in neighbors:
        if G.nodes[_]["type"] == "node":
            if G.nodes[_]["application"] == "not":
                fout[0] += 1
            elif G.nodes[_]["application"] == "and":
                fout[1] += 1
            elif G.nodes[_]["application"] == "or":
                fout[2] += 1
    return fout
    
def fin_embedding(G, node_id):
    # find all the neighbors of the node that point to it
    # and count the number of each type of node
    neighbors = list(G.successors(node_id))
    fout = np.array([0 , 0 , 0, 0, 0 ,0, 0]) #represent the number of Not, And, Or
    for _ in neighbors:
        if G.nodes[_]["type"] == "node":
            if G.nodes[_]["application"] == "not":
                fout[0] += 1
            elif G.nodes[_]["application"] == "and":
                fout[1] += 1
            elif G.nodes[_]["application"] == "or":
                fout[2] += 1
                
        elif G.nodes[_]["type"] == "variable":
            if G.nodes[_]["application"].startswith("v"):
                fout[3] += 1
            elif G.nodes[_]["application"].startswith("i"):
                fout[4] += 1
            elif G.nodes[_]["application"].startswith("constant_t"):
                fout[5] += 1
            elif G.nodes[_]["application"].startswith("constant_f"):
                fout[6] += 1
    return fout

# Helper function to generate features for a single node
def generate_single_node_features(G, node_id, node_data , betweenness_centrality, max_node_id):
    kind_enc, value_enc = one_hot_encoding(node_data,G)
    betweenness = betweenness_centrality[node_id]
    degree = G.degree(node_id)
    #root_distance = distance_to_root(G, root_id)[node_id]
    neighbor_ops = count_neighbor_operators(G, node_id)
    pos_emb = position_embedding(node_id, max_node_id)

    features = np.array([betweenness, degree, neighbor_ops])
    features = np.concatenate((kind_enc, value_enc, features, pos_emb))

    return features

def generate_node_features(G):
    # Find the root node id
    # root_id = None
    # for node_id, node_data in G.nodes(data=True):
    #     if node_data["type"] == "node" and node_data["application"] == "and":
    #         if G.in_degree(node_id) == 0:
    #             root_id = node_id
    #             break

    # G = G.to_undirected()
    # Compute betweenness centrality
    root_id = 0
    betweenness_centrality = nx.betweenness_centrality(G)

    # Compute node features
    node_features = []
    max_node_id = len(G.nodes) - 1
    # Initialize the list to store node features
    #node_features = [None] * len(G.nodes)

    for node_id, node_data in G.nodes(data=True):
        kind_enc, value_enc = one_hot_encoding(node_data)
        betweenness = betweenness_centrality[node_id]
        #degree = G.degree(node_id)
        root_distance = distance_to_root(G, root_id)[node_id]
        #neighbor_ops = count_neighbor_operators(G, node_id)
        pos_emb = position_embedding(node_id, max_node_id)
        fout = fout_embedding(G, node_id)
        fin = fin_embedding(G, node_id)

        features = np.array([betweenness, root_distance])
        features = np.concatenate((kind_enc, value_enc, pos_emb, fout, fin, features))
        node_features.append(features)
        ### other features can be added:
        ### - conncetivity between state variables
        ### - k hop neighborhood features (k hop to operator? state variable? input?)
        ### - depth of the node in the graph
        ### - Clustering coefficient: The clustering coefficient of a node measures the degree to which its neighbors are interconnected. This feature can provide insights into the local structure of the graph around the node.
        ### - Fan-in and fan-out of nodes: The fan-in of a node is the number of input edges it has, while the fan-out is the number of output edges. These features can provide information about the connectivity and the role of nodes in the circuit.



    # Convert list of arrays to a 2D NumPy array
    node_features_matrix = np.vstack(node_features)

    # Slice the continuous features (assuming they are the last 5 columns in this case)
    continuous_features = node_features_matrix[:, -8:]

    # Scale the continuous features using standard scaling
    scaler = StandardScaler()
    scaled_continuous_features = scaler.fit_transform(continuous_features)

    # Replace the original continuous features with the scaled ones
    node_features_matrix[:, -8:] = scaled_continuous_features

    # Convert the 2D NumPy array back to the list of arrays
    node_features = list(node_features_matrix)

    return np.array(node_features), G
