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
from GNN_Model import GCNModel, BWGNN_Hetero, BWGNN, SAGEConvModel, GATModel, BWGNN_Inductive, DualGCNModel, DualGraphSAGEModel
from Dataset import CustomGraphDataset
from Loss import FocalLoss
import argparse
from sklearn.metrics import f1_score
from ThersholdFinder import ThresholdFinder
import warnings
from Embedding import deepwalk, generate_node_features, generate_attribute_node_features
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


#################################
# WIP:
# 1. generate predicted clauses from json (json -> graph -> model eval() -> predicted clauses)
# 2. use better features (features engineering on the graph: transition relation, initial state, etc.)
# 3. use better model (STGCN, GAT, BWGNN, etc.)
# 4. use more graph (add more graphs to the dataset, hwmcc2007, etc.) - DONE
# 5. solve the imbalanced data problem (class weights, moving thereshold, focal loss, etc.)
# 6. weird loss <1
# 7. calculate the perfect accuracy (all the clauses are correct)
# 8. early stop
# 9. use more suggestion from chatgpt
# 10. fix the only one variable bug: dataset_hwmcc2020_all_only_unsat_abc_deep_0/bad_cube_cex2graph/expr_to_build_graph/vcegar_QF_BV_itc99_b13_p10/vcegar_QF_BV_itc99_b13_p10_4.smt2
# 11. train loss nan bug
# 12. some cases may have no data -> check build_data.py
# 13. solve the zero convergence bug in ic3ref
# 14. dump all graph using random walk embedding to a pickle file
# 15. CUDA utilization is low (only 1%, related to conda env?) -> may adjust the loss calculation in advance
# 16. utilize betweenness centrality, degree centrality, etc. as node features as well (additional dimension)
# 17. try heterograph graph training? (Need to calculate the canonical edge types [relations] in the graph in advance)
# 18. the positive weight of cross entropy may be needed to be tuned
# 19. new data construction method: how to avoid generating constant_false graph (only one node?)
# 20. fix loss function weight calculation bug
#################################

#JSON_FOLDER = '/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc2007_tip_ic3ref_no_simplification_0-22/bad_cube_cex2graph/expr_to_build_graph/nusmv.syncarb5^2.B/'
#JSON_FOLDER = '/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc2020_all_only_unsat_abc_no_simplification_0/bad_cube_cex2graph/expr_to_build_graph/vcegar_QF_BV_itc99_b13_p10/'
#GROUND_TRUTH = os.path.join(JSON_FOLDER.replace('expr_to_build_graph', 'ground_truth_table'), JSON_FOLDER.split('/')[-2]+'.csv')
HIDDEN_DIM = 64 # 32 default
EMBEDDING_DIM = 32 # 16 default
EPOCH = 10 # 100 default
LR = 0.001 # =learning rate 0.01 default
BATCH_SIZE = 2 # 2 default
DATASET_SPLIT = 4 # None default, used for testing
WEIGHT_DECAY = 1e-2 # Apply L1 or L2 regularization, [1e-3,1e-2], default 1e-5
DUMP_MODE = False # False default, used for preprocessing graph data

# Best parameters (2023.4.4)
# HIDDEN_DIM = 128, EMBEDDING_DIM = 128, EPOCH = 500, LR = 0.005, BATCH_SIZE = 16

# Testing parameters (2021.4.4) -> not good enough
# HIDDEN_DIM = 128, EMBEDDING_DIM = 128, EPOCH = 300, LR = 0.01, BATCH_SIZE = 64

# Use to calculate the weighted in imbalanced data
def calculate_class_weights(train_labels):
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    # directly calculate the weight of each class
    # return class_counts[0] / class_counts

    # takes into account the class imbalance but also incorporates the total number of samples 
    # and the number of classes in the calculation
    return total_samples / (len(class_counts) * class_counts)

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
            G = G.to_undirected()
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
    ----------------Simple Graph Feature Embedding----------------
    
    generate_node_features(graph_list, EMBEDDING_DIM)
    '''
    
    '''
    ----------------Deep walk Embedding----------------
    '''
    # Set the DeepWalk parameters
    num_walks = 10
    walk_length = 80
    if args.dual: # not change the embedding dim
        embedding_dim = EMBEDDING_DIM
    else: # use the rest of dim to embed the graph
        embedding_dim = EMBEDDING_DIM - graph_list[0][1].shape[1] # one of the inital features
    #embedding_dim = EMBEDDING_DIM
    window_size = 10

    # Apply DeepWalk to each graph in the graph_list
    for idx, (G, initial_feat, node_labels, train_mask) in tqdm(enumerate(graph_list), desc="Applying DeepWalk", total=len(graph_list)):
        node_features_dw = deepwalk(G, num_walks, walk_length, embedding_dim, window_size)
        # for every node feature in initial_feat, append the deepwalk embedding
        if args.dual: # dual graph, no need to concat
            node_features_final = node_features_dw
        else: # concat the initial feature and the deepwalk embedding
            node_features_final = np.concatenate((node_features_dw, initial_feat), axis=1)
        #node_features_final = node_features_dw
        graph_list[idx] = (G, node_features_final, node_labels, train_mask)
    
    # dump all the graph_list to a pickle file 
    # Save the graph_list to a pickle file
    if DUMP_MODE:
        with open(args.dump_pickle_name, "wb") as f:
            pickle.dump(graph_list, f)
        exit(0)
        
    '''
    ----------------Random Embedding----------------
    
    generate_attribute_node_features(graph_list, EMBEDDING_DIM)
    '''

    return graph_list

def model_eval(args, val_dataloader, model, device, save_model=False,thred=None,print_stats=False):
    model.eval()
    
    pred_list = []
    true_labels_list = []
    variable_pred_list = []
    variable_true_labels_list = []

    #for batched_dgl_G in test_dataloader:
    for batched_dgl_G in val_dataloader:
        batched_dgl_G = batched_dgl_G.to(device)
        if args.dual:
            logits = model(batched_dgl_G, batched_dgl_G.ndata['ori_feat'],batched_dgl_G.ndata['struc_feat'])
        else:
            logits = model(batched_dgl_G, batched_dgl_G.ndata['feat'])
        if thred is None:
            pred = logits.argmax(1).cpu().numpy()
        if thred is not None:
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)
            # Compare the probability of class 1 with the threshold value
            pred = (probs[:, 1] > thred).cpu().numpy().astype(int)
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

    if print_stats:
        print('Evaluating the model...')
        #print("Accuracy: ", accuracy, "Precision: ", precision, "Recall: ", recall, "F1-score: ", f1)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1-score: ", f1)
        print("Confusion matrix:")
        print(confusion)
    
    # save the model
    if args.model_name is not None and save_model:
        torch.save(model.state_dict(), f"{args.model_name}.pt")
    
    return f1
    
    

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='dataset name') # no need if load pickle
    parser.add_argument('--load-pickle', type=str, default=None, help='load pickle file name')
    parser.add_argument('--dump-pickle-name', type=str, default=None, help='dump pickle file name') # no need if load pickle
    parser.add_argument('--model-name', type=str, default=None, help='model name to save')
    parser.add_argument('--dual', action='store_true', help='use dual graph')
    #parser.add_argument('--concat', action='store_true', help='concatenate the original feature and the deepwalk embedding')
    # complex
    # args = parser.parse_args(['--dataset', \
    #                           '/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc2020_all_only_unsat_hard_abc_no_simplification_0-9/bad_cube_cex2graph/expr_to_build_graph/',\
    #                           '--dump-pickle-name', 'dataset_hwmcc2020_all_only_unsat_hard_abc_no_simplification_0-9_list_name'
    #                           ])
    # simple
    #args = parser.parse_args(['--load-pickle', '/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/train_gcn/dataset_hwmcc2007_tip_ic3ref_no_simplification_0-22.pickle'])
    #args = parser.parse_args(['--dataset','/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc2020_all_only_unsat_ic3ref_no_simplification_0-38/bad_cube_cex2graph/expr_to_build_graph/','--dual'])
    args = parser.parse_args()
    if args.dump_pickle_name is not None: DUMP_MODE = True

    if args.load_pickle is not None: #  First, load the pickle file if it exists
        graph_list = pickle.load(open(args.load_pickle, "rb"))
    elif args.dataset is not None: # Second, do data preprocessing if the pickle file does not exist
        # apply multi-feature embedding
        graph_list_ori_feat = data_preprocessing(args)
        
        
        updated_graph_list = []
        # pre-defined feature embedding
        for idx, (G, initial_feat, node_labels, train_mask) in tqdm(enumerate(graph_list_ori_feat), desc="Applying inital feature encode", total=len(graph_list_ori_feat)):
            initial_feat, G = generate_node_features(G)
            new_tuple = (G, initial_feat, node_labels, train_mask);updated_graph_list.append(new_tuple)
        
        # Assign the updated list back to the original variable
        graph_list_ori_feat = updated_graph_list
        
        # copy the graph_list
        graph_list_struc_feat = deepcopy(graph_list_ori_feat)
        
        # structure feature embedding
        graph_list_struc_feat = employ_graph_embedding(graph_list_struc_feat,args)
        # apply pre-define feature embedding
        # graph_list = data_preprocessing(args)
        update_adj_cosine_d(graph_list_ori_feat,graph_list_struc_feat,dualGraph=args.dual)
        
        if args.dual:
            graph_list = [(G,
                    np.concatenate((ori_feat, graph_list_struc_feat[i][1]), axis=1),
                    node_labels,
                    train_mask)
                for i, (G, ori_feat, node_labels, train_mask) in enumerate(graph_list_ori_feat)]
        else:
            graph_list = graph_list_struc_feat
        
    else:
        assert False, "Please specify the dataset path to do data preprocessing or load the pickle file."

    # 4. Create a custom dataset
    
    # Split the graph_list into train, val, and test datasets
    # graph_list = graph_list[:]
    #  apply dgl.from_networkx to every graph[0] in graph_list
    assert len(graph_list[0]) == 4, "The graph_list should be a list of tuples (graph, node_features, node_labels, node_train_mask)"
    #graph_list = [(dgl.from_networkx(graph[0]), graph[1],graph[2],graph[3]) for graph in graph_list]

    # train_data = graph_list
    # _, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    
    
    #train_data, test_data = train_test_split(graph_list, test_size=0.05, random_state=42)
    #_, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    
    
    train_data, temp_data = train_test_split(graph_list, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # train_data = graph_list ; val_data = graph_list ; test_data = graph_list
    
    print("Length of val_data: ", len(val_data))
    
    # Create custom datasets for each split
    train_dataset = CustomGraphDataset(train_data, split='train',DIM=EMBEDDING_DIM, dual=args.dual)
    val_dataset = CustomGraphDataset(val_data, split='val',DIM=EMBEDDING_DIM,dual=args.dual)
    #test_dataset = CustomGraphDataset(test_data, split='test')

    # Create dataloaders for each split
    train_dataloader = GraphDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_dataloader = GraphDataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    #test_dataloader = GraphDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)


    # 5. Train the model
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get the input feature dimension
    input_feature_dim = graph_list[0][1].shape[1] # get the feature dimension of the first graph
    ori_feat_input_dim = graph_list_ori_feat[0][1].shape[1] # if using dual graph embedding
    struc_feat_input_dim = graph_list_struc_feat[0][1].shape[1] # if using dual graph embedding
    #model = GCNModel(input_feature_dim, HIDDEN_DIM, 2).to(device)
    #model = BWGNN(input_feature_dim, HIDDEN_DIM, 2).to(device)
    #model = BWGNN_Inductive(input_feature_dim, HIDDEN_DIM, 2).to(device)
    model = DualGraphSAGEModel(ori_feat_input_dim, struc_feat_input_dim, HIDDEN_DIM, 2, 16).to(device)
    #model = SAGEConvModel(input_feature_dim, HIDDEN_DIM, 2).to(device)
    #model = GATModel(input_feature_dim, HIDDEN_DIM, 2).to(device)
    print(model) # for log analysis
    loss_function = nn.CrossEntropyLoss()
    #loss_function = FocalLoss()
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float('inf')
    overall_best_f1 = 0 # overall best f1 score in all epochs
    
    for epoch in range(EPOCH):
        model.train()
        epoch_loss = 0
        # with torch.profiler.profile(
        # schedule=torch.profiler.schedule(wait=0, warmup=1, active=3),
        # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        # record_shapes=True,
        # profile_memory=True,
        # with_stack=True,
        # use_cuda=True,
        # ) as prof:
        for batched_dgl_G in train_dataloader:
            batched_dgl_G = batched_dgl_G.to(device) #this may cost too much time
            if args.dual:
                logits = model(batched_dgl_G, batched_dgl_G.ndata['ori_feat'],batched_dgl_G.ndata['struc_feat'])
            else:
                logits = model(batched_dgl_G, batched_dgl_G.ndata['feat'])
            # Compute the class weights for the current batch
            train_labels = batched_dgl_G.ndata['label'][batched_dgl_G.ndata['train_mask']].cpu().numpy()
            class_weights = calculate_class_weights(train_labels)
            if len(class_weights)==0: class_weights = [1.0, 1.0] # error handling
            # Update the loss function with the computed class weights
            #loss_function = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
            loss_function.weight = torch.FloatTensor(class_weights).to(device)

            loss = loss_function(logits[batched_dgl_G.ndata['train_mask']], batched_dgl_G.ndata['label'][batched_dgl_G.ndata['train_mask']])
            #loss = loss_function(logits[batched_dgl_G.ndata['train_mask']], batched_dgl_G.ndata['label'][batched_dgl_G.ndata['train_mask']])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            #prof.step()
        
        
        # Validation loop
        '''
        model.eval()
        val_loss = 0
        for batched_dgl_G in val_dataloader:
            batched_dgl_G = batched_dgl_G.to(device)
            logits = model(batched_dgl_G, batched_dgl_G.ndata['feat'])
            
            # Compute the class weights for the current batch
            val_labels = batched_dgl_G.ndata['label'][batched_dgl_G.ndata['train_mask']].cpu().numpy()
            class_weights = calculate_class_weights(val_labels)
            if len(class_weights) == 0:
                class_weights = [1.0, 1.0]  # error handling
            
            # Update the loss function with the computed class weights
            loss_function = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
            
            loss = loss_function(logits[batched_dgl_G.ndata['train_mask']], batched_dgl_G.ndata['label'][batched_dgl_G.ndata['train_mask']])
            val_loss += loss.item()
        
        #if val_loss < best_val_loss: best_val_loss = val_loss
        threshold_finder = ThresholdFinder(val_dataloader, model, device)
        _, best_f1, _ = threshold_finder.find_best_threshold() # best f1 score under the best threshold in the current epoch
        if best_f1 > overall_best_f1: overall_best_f1 = best_f1 
        # *10 to make the loss comparable
        print(f"EPOCH: {epoch}, TRAIN LOSS: {(epoch_loss)},VAL LOSS: {(val_loss)}, BEST F1: {overall_best_f1}")
        '''
        f1 = model_eval(args, val_dataloader, model, device, save_model=False, print_stats=False)
        if f1 > overall_best_f1: overall_best_f1 = f1
        print(f"EPOCH: {epoch}, TRAIN LOSS: {(epoch_loss)}, BEST F1: {overall_best_f1}")
        # employ early stop
        if overall_best_f1 == 1: break


    # Additional step: evaluate the model by using ThersholdFinder

    # Instantiate the ThresholdFinder class
    model.eval()
    print("Now evaluating the model on the validation set and find the best thershold...")
    threshold_finder = ThresholdFinder(val_dataloader, model, device,args)
    #threshold_finder = ThresholdFinder(test_dataloader, model, device)

    # Find the best threshold
    best_threshold, best_f1, best_confusion = threshold_finder.find_best_threshold()
    print("Best threshold: ", best_threshold)
    print("Best F1-score: ", best_f1)
    print("Best confusion matrix:\n ", best_confusion)

    # 6. Evaluate the model
    print("Now evaluating the model on the whole training dataset...")
    _ = model_eval(args, train_dataloader, model, device, save_model=True, print_stats=True)
    
    # additional step: evaluate the model on the test set
    print("Now evaluating the model on the test set...")
    test_dataset = CustomGraphDataset(test_data, split='test', DIM=EMBEDDING_DIM, dual=args.dual)
    test_dataloader = GraphDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    _ = model_eval(args, test_dataloader, model, device, save_model=False, thred=best_threshold, print_stats=True)
    
    




