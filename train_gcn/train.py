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
import dgl.data
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from dgl.dataloading import GraphDataLoader
#from BWGNN import BWGNN_Hetero, BWGNN, PolyConv, PolyConvBatch
import glob 
from GNN_Model import GCNModel, BWGNN_Hetero, BWGNN
from Dataset import CustomGraphDataset
from Loss import FocalLoss
import argparse
from sklearn.metrics import f1_score
from ThersholdFinder import ThresholdFinder
import warnings
from node2vec import Node2Vec
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.decomposition import PCA

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")



#################################
# WIP:
# 1. generate predicted clauses from json (json -> graph -> model eval() -> predicted clauses)
# 2. use better features (features engineering on the graph: transition relation, initial state, etc.)
# 3. use better model (STGCN, GAT, BWGNN, etc.)
# 4. use more graph (add more graphs to the dataset, hwmcc2007, etc.)
# 5. solve the imbalanced data problem (class weights, moving thereshold, focal loss, etc.)
# 6. weird loss <1
# 7. calculate the perfect accuracy (all the clauses are correct)
# 8. early stop
# 9. use more suggestion from chatgpt
# 10. fix the only one variable bug: /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc2020_all_only_unsat_abc_deep_0/bad_cube_cex2graph/expr_to_build_graph/vcegar_QF_BV_itc99_b13_p10/vcegar_QF_BV_itc99_b13_p10_4.smt2
# 11. train loss nan bug
# 12. some cases may have no data -> check build_data.py
# 13. solve the zero convergence bug in ic3ref
#################################

#JSON_FOLDER = '/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc2007_tip_ic3ref_no_simplification_0-22/bad_cube_cex2graph/expr_to_build_graph/nusmv.syncarb5^2.B/'
#JSON_FOLDER = '/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc2020_all_only_unsat_abc_no_simplification_0/bad_cube_cex2graph/expr_to_build_graph/vcegar_QF_BV_itc99_b13_p10/'
#GROUND_TRUTH = os.path.join(JSON_FOLDER.replace('expr_to_build_graph', 'ground_truth_table'), JSON_FOLDER.split('/')[-2]+'.csv')
HIDDEN_DIM = 128 # 32 default
EMBEDDING_DIM = 128 # 16 default
EPOCH = 500 # 100 default
LR = 0.005 # 0.01 default
BATCH_SIZE = 16 # 2 default

# Use to calculate the weighted in imbalanced data
def calculate_class_weights(train_labels):
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    return total_samples / (len(class_counts) * class_counts)

def generate_node2vec_embedding(graph_data):
    G, _, node_labels, train_mask = graph_data

    # Run Node2Vec on the graph G and obtain the embeddings
    node2vec = Node2Vec(G, dimensions=EMBEDDING_DIM, walk_length=10, num_walks=100, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # Assign Node2Vec embeddings as node features
    node_features = np.zeros((G.number_of_nodes(), EMBEDDING_DIM))
    for i, node_id in enumerate(G.nodes()):
        node_features[i] = model.wv[f"{node_id}"]

    return (G, node_features, node_labels, train_mask)


if __name__ == "__main__":
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='dataset name')
    args = parser.parse_args(['--dataset', 
                              '/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc2007_tip_ic3ref_no_simplification_0-22/bad_cube_cex2graph/expr_to_build_graph/'])
    #args = parser.parse_args()
    
    # Get all case folders under the input directory
    case_folders = glob.glob(os.path.join(args.dataset, '*'))
    for case_folder in case_folders[:]:
        JSON_FOLDER = case_folder
        GROUND_TRUTH = os.path.join(JSON_FOLDER.replace('expr_to_build_graph', 'ground_truth_table'), JSON_FOLDER.split('/')[-1]+'.csv')
        if not os.path.exists(GROUND_TRUTH): continue
        json_files = [f for f in os.listdir(JSON_FOLDER) if f.endswith('.json')]
        graph_list = []
        # For each aiger case, read all the JSON files
        for _ in json_files:
            # 1. Parse the JSON graph data and create the graph structure
            with open(os.path.join(JSON_FOLDER, _)) as f:
                graph_data = json.loads(f.read())

            G = nx.DiGraph()

            for node in graph_data:
                G.add_node(node['data']['id'], **node['data'])

            for node in graph_data:
                if 'to' in node['data']:
                    for child_id in node['data']['to']['children_id']:
                        G.add_edge(node['data']['id'], child_id)

            # Convert the directed graph G to an undirected graph
            G = G.to_undirected()

            # 2. Assign node features and labels

            # go to `GROUND_TRUTH` to get the labels
            # read the ground truth file
            ground_truth_table = pd.read_csv(GROUND_TRUTH)

            # only keep the particular case
            ground_truth_table = ground_truth_table[ground_truth_table['inductive_check']==os.path.join(JSON_FOLDER, _).split('/')[-1].replace('.json', '.smt2')]

            labels =  ground_truth_table.set_index("inductive_check").iloc[:, 1:].to_dict('records')[0]
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
    ----------------Simple Graph Feature Embedding----------------
    '''
    pca = PCA(n_components=EMBEDDING_DIM)
    # Compute the centrality measures for the nodes in the graph
    for idx, (G, _, node_labels, train_mask) in enumerate(graph_list):
        betweenness_centrality = nx.betweenness_centrality(G)
        degree_centrality = nx.degree_centrality(G)

        # Assign the centrality measures as node features
        num_nodes = G.number_of_nodes()
        node_features = np.zeros((num_nodes, 2))
        for i, node_id in enumerate(G.nodes()):
            node_features[i, 0] = betweenness_centrality[node_id]
            node_features[i, 1] = degree_centrality[node_id]

        # Apply PCA to increase the dimensionality of the node features
        node_features_pca = pca.fit_transform(node_features)

        graph_list[idx] = (G, node_features_pca, node_labels, train_mask)
    
    
    '''
    ----------------Random Embedding----------------
    # additional step: get the unique attributes

    # extract unique attributes from all nodes across all graphs:
    unique_attributes = set()
    for G, _, _, _ in graph_list:
        for _, node_data in G.nodes(data=True):
            unique_attributes.add(node_data['type'])
            unique_attributes.add(node_data['application'])
            
    # Create an embedding for each unique attribute and store them in a dictionary
    embedding_dim = EMBEDDING_DIM
    #node2vec = Node2Vec(G, dimensions=EMBEDDING_DIM, walk_length=10, num_walks=100, workers=4)
    #model = node2vec.fit(window=10, min_count=1, batch_words=4)
    #attribute_embedding = {node_id: model.wv[f"{node_id}"] for node_id in G.nodes}
    attribute_embedding = {attr: np.random.rand(embedding_dim) for attr in unique_attributes}

    # Modify the loop that assigns node features to use the attribute embeddings
    for idx, (G, _, node_labels, train_mask) in enumerate(graph_list):
        node_features = np.zeros((G.number_of_nodes(), embedding_dim))
        
        for i, (_, node_data) in enumerate(G.nodes(data=True)):
            node_features[i] = attribute_embedding[node_data['type']] + attribute_embedding[node_data['application']]

        graph_list[idx] = (G, node_features, node_labels, train_mask)
    '''

    # 4. Create a Graph Convolutional Network (GCN) model and custom dataset
        
    # Focal loss


    dataset = CustomGraphDataset(graph_list)
    dataloader = GraphDataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # 5. Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get the input feature dimension
    input_feature_dim = embedding_dim
    #model = GCNModel(input_feature_dim, HIDDEN_DIM, 2).to(device)
    model = BWGNN(input_feature_dim, HIDDEN_DIM, 2).to(device)
    #loss_function = nn.CrossEntropyLoss()
    #loss_function = FocalLoss()
    optimizer = Adam(model.parameters(), lr=LR)


    for epoch in range(EPOCH):
        model.train()
        epoch_loss = 0
        for batched_dgl_G in dataloader:
            batched_dgl_G = batched_dgl_G.to(device)
            logits = model(batched_dgl_G, batched_dgl_G.ndata['feat'])
            # Compute the class weights for the current batch
            train_labels = batched_dgl_G.ndata['label'][batched_dgl_G.ndata['train_mask']].cpu().numpy()
            class_weights = calculate_class_weights(train_labels)
            if len(class_weights)==0: class_weights = [1.0, 1.0] # error handling
            # Update the loss function with the computed class weights
            loss_function = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))

            loss = loss_function(logits[batched_dgl_G.ndata['train_mask']], batched_dgl_G.ndata['label'][batched_dgl_G.ndata['train_mask']])
            #loss = loss_function(logits[batched_dgl_G.ndata['train_mask']], batched_dgl_G.ndata['label'][batched_dgl_G.ndata['train_mask']])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"EPOCH: {epoch}, LOSS: {epoch_loss / len(dataloader)}")


    # Additional step: evaluate the model by using ThersholdFinder
    
    # Instantiate the ThresholdFinder class
    model.eval()
    threshold_finder = ThresholdFinder(dataloader, model, device)

    # Find the best threshold
    best_threshold, best_f1, best_confusion = threshold_finder.find_best_threshold()
    print("Best threshold: ", best_threshold)
    print("Best F1-score: ", best_f1)
    print("Best confusion matrix:\n ", best_confusion)
    
    # 6. Evaluate the model
    
    pred_list = []
    true_labels_list = []
    variable_pred_list = []
    variable_true_labels_list = []
    
    for batched_dgl_G in dataloader:
        batched_dgl_G = batched_dgl_G.to(device)
        logits = model(batched_dgl_G, batched_dgl_G.ndata['feat'])
        pred = logits.argmax(1).cpu().numpy()
        true_labels = batched_dgl_G.ndata['label'].cpu().numpy()

        # Create variable_mask for the batched_dgl_G
        variable_mask = batched_dgl_G.ndata['train_mask'].cpu().numpy()

        pred_list.append(pred)
        true_labels_list.append(true_labels)
        variable_pred_list.append(pred[variable_mask])
        variable_true_labels_list.append(true_labels[variable_mask])

    # Concatenate predictions and true labels
    pred = np.concatenate(pred_list)
    true_labels = np.concatenate(true_labels_list)
    variable_pred = np.concatenate(variable_pred_list)
    variable_true_labels = np.concatenate(variable_true_labels_list)

    # Calculate metrics
    accuracy = accuracy_score(variable_true_labels, variable_pred)
    precision = precision_score(variable_true_labels, variable_pred)
    recall = recall_score(variable_true_labels, variable_pred)
    f1 = f1_score(variable_true_labels, variable_pred)
    confusion = confusion_matrix(variable_true_labels, variable_pred)

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score: ", f1)
    print("Confusion matrix:")
    print(confusion)




