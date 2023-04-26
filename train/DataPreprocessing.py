
import json
import torch
import dgl
import os
import numpy as np
import networkx as nx
import numpy as np
import pandas as pd
from dgl.nn import GraphConv
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import dgl.data
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from dgl.dataloading import GraphDataLoader
#from BWGNN import BWGNN_Hetero, BWGNN, PolyConv, PolyConvBatch
import glob 
from GNN_Model import GCNModel, SAGE, GATModel, SAGE_BW, DualGCNModel, DualGraphSAGEModel
from Dataset import CustomGraphDataset
from Loss import FocalLoss
import argparse
from sklearn.metrics import f1_score
from ThersholdFinder import ThresholdFinder
import warnings
from Embedding import deepwalk
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.decomposition import PCA
import pickle
from sklearn.model_selection import train_test_split
import torch.profiler
from NodeFeatureGenerator import generate_node_features
from torch.utils.data import DataLoader
from copy import deepcopy
from GraphConverter import update_adj_cosine_d
from GlobalVar import *

# Preprocess the data before training
def data_preprocessing(args):
    # Get all case folders under the input directory
    case_folders = glob.glob(os.path.join(args.dataset, '*'))
    graph_list = []

    for case_folder in case_folders[:DATASET_SPLIT]:
        JSON_FOLDER = case_folder
        GROUND_TRUTH = os.path.join(JSON_FOLDER.replace('expr_to_build_graph', 'ground_truth_table'), JSON_FOLDER.split('/')[-1]+'.csv')
        if not os.path.exists(GROUND_TRUTH): continue
        json_files = [f for f in os.listdir(JSON_FOLDER) if f.endswith('.json')]
        # For each aiger case, read all the JSON files
        for _ in json_files:
            #print(_) ; graph_list.append(_) ; continue # for bug fix only
            # 1. Parse the JSON graph data and create the graph structure
            with open(os.path.join(JSON_FOLDER, _)) as f:
                graph_data = json.loads(f.read())

            G = nx.DiGraph(name=os.path.splitext(_)[0])

            for node in graph_data:
                G.add_node(node['data']['id'], **node['data'])

            for node in graph_data:
                if 'to' in node['data']:
                    for child_id in node['data']['to']['children_id']:
                        G.add_edge(node['data']['id'], child_id)

            # for node in G.node(data=True): print(node)

            # 2. Assign node features and labels

            # go to `GROUND_TRUTH` to get the labels
            # read the ground truth file
            ground_truth_table = pd.read_csv(GROUND_TRUTH)

            # only keep the particular case
            ground_truth_table = ground_truth_table[ground_truth_table['inductive_check']==os.path.join(JSON_FOLDER, _).split('/')[-1].replace('.json', '.smt2')]

            labels =  ground_truth_table.set_index("inductive_check").iloc[:, 1:].to_dict('records')[0]

            # generate node features and convert the graph to undirected graph
            # node_features, G = generate_node_features(G)

            # Convert the directed graph G to an undirected graph
            # G = G.to_undirected()
            node_features = np.eye(len(G.nodes))
            #node_features = np.eye(G.number_of_nodes())

            node_labels = []

            for node in G.nodes(data=True):
                if node[1]['type'] == 'variable' and node[1]['application'].startswith('v'):
                    node_labels.append(labels[node[1]['application']])
                else:
                    node_labels.append(-1)

            # 3. Split the dataset into training, validation, and test sets
            train_mask = [
                bool(
                    node[1]['type'] == 'variable'
                    and node[1]['application'].startswith('v')
                )
                for node in G.nodes(data=True)
            ]

            graph_list.append((G, node_features, node_labels, train_mask))

    #with open(args.dump_pickle_name, "wb") as f: pickle.dump(graph_list, f) ; exit(0) # for bug fix only
    # if Update_DIM: 
    #     EMBEDDING_DIM = graph_list[0][1].shape[1]
    return graph_list


def employ_graph_embedding(graph_list,args):
    
            
    '''
    ----------------Node2Vec Embedding----------------
    

    # Parallelize the generation of Node2Vec embeddings
    num_threads = 8  # Adjust the number of threads based on your CPU capabilities

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(generate_node2vec_embedding, graph_data) for graph_data in graph_list]

        # Collect the results as they become available
        for idx, future in enumerate(as_completed(futures)):
            graph_list[idx] = future.result()
    '''
    
    '''
    ----------------Deep walk Embedding----------------
    '''
    # Set the DeepWalk parameters
    num_walks = 10
    walk_length = 80
    embedding_dim = EMBEDDING_DIM
    window_size = 10

    # Apply DeepWalk to each graph in the graph_list
    for idx, (G, initial_feat, node_labels, train_mask) in tqdm(enumerate(graph_list), desc="Applying DeepWalk", total=len(graph_list)):
        node_features_dw = deepwalk(G, num_walks, walk_length, embedding_dim, window_size)
        # for every node feature in initial_feat, append the deepwalk embedding
        node_features_final = node_features_dw
        #node_features_final = node_features_dw
        graph_list[idx] = (G, node_features_final, node_labels, train_mask)
    
    
    return graph_list
