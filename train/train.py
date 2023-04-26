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
from DataPreprocessing import data_preprocessing, employ_graph_embedding
from GlobalVar import *
from dgl.nn import AvgPooling, GNNExplainer

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
    

def model_eval(args, val_dataloader, model, device, save_model=False,thred=None,print_stats=False):
    model.eval()
    
    pred_list = []
    true_labels_list = []
    variable_pred_list = []
    variable_true_labels_list = []

    #for batched_dgl_G in test_dataloader:
    for batched_dgl_G in val_dataloader:
        batched_dgl_G = batched_dgl_G.to(device)
       
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
        variable_mask = batched_dgl_G.ndata['mask'].cpu().numpy()

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
    mf1 = f1_score(variable_true_labels, variable_pred, average='macro',zero_division=1)
    confusion = confusion_matrix(variable_true_labels, variable_pred)

    if print_stats:
        print('Evaluating the model...')
        #print("Accuracy: ", accuracy, "Precision: ", precision, "Recall: ", recall, "F1-score: ", f1)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("MF1-score: ", mf1)
        print("Confusion matrix:")
        print(confusion)
    
    # save the model
    if args.model_name is not None and save_model:
        torch.save(model.state_dict(), f"{args.model_name}.pt")
    
    return mf1
    
    

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='dataset name') # no need if load pickle
    parser.add_argument('--load-pickle', type=str, default=None, help='load pickle file name')
    parser.add_argument('--dump-pickle-name', type=str, default=None, help='dump pickle file name') # no need if load pickle
    parser.add_argument('--model-name', type=str, default=None, help='model name to save')
    parser.add_argument('--update-adjs', action='store_true', help='update adjs by applying knn graph')
    parser.add_argument('--undirected', action='store_true', help='undirected graph')
    
    args = parser.parse_args(['--dataset','/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc2007_tip_ic3ref_no_simplification_0-22/bad_cube_cex2graph/expr_to_build_graph/','--undirected'])
    print(args)
    if args.dump_pickle_name is not None: DUMP_MODE = True

    if args.load_pickle is not None: #  First, load the pickle file if it exists
        graph_list = pickle.load(open(args.load_pickle, "rb"))
    elif args.dataset is not None: # Second, do data preprocessing if the pickle file does not exist
        # initialize the graph list
        graph_list_initialize = data_preprocessing(args)
        
        
        updated_graph_list = []
        # pre-defined feature embedding
        for idx, (G, initial_feat, node_labels, train_mask) in tqdm(enumerate(graph_list_initialize), desc="Applying inital feature encode", total=len(graph_list_initialize)):
            assert G.is_directed(), "G is undirected graph"
            initial_feat, G = generate_node_features(G)
            new_tuple = (G, initial_feat, node_labels, train_mask); updated_graph_list.append(new_tuple)
            assert G.is_directed(), "G is undirected graph"
        
        
        # Assign the updated list back to the original variable
        graph_list_encoded = updated_graph_list
        
        if args.update_adjs:
            # copy the graph_list
            graph_list_knn = deepcopy(graph_list_initialize)
            # structure feature embedding
            graph_list_knn = employ_graph_embedding(graph_list_knn,args)
            # apply pre-define feature embedding
            # graph_list = data_preprocessing(args)
            update_adj_cosine_d(graph_list_encoded,graph_list_knn)
            
        if args.undirected:
            for (G, _, __, ___) in tqdm(graph_list_encoded, desc="Converting to undirected graph", total=len(graph_list_encoded)):
                G.to_undirected()
            
        graph_list = graph_list_encoded
        
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
    train_dataset = CustomGraphDataset(train_data, split='train',DIM=graph_list[0][1].shape[1])
    val_dataset = CustomGraphDataset(val_data, split='val',DIM=graph_list[0][1].shape[1])
    #test_dataset = CustomGraphDataset(test_data, split='test')

    # Create dataloaders for each split
    train_dataloader = GraphDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_dataloader = GraphDataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    #test_dataloader = GraphDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)


    # 5. Train the model
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #model = GCNModel(input_feature_dim, HIDDEN_DIM, 2).to(device)
    #for (G, initial_feat, node_labels, train_mask) in graph_list: G.to_undirected(); model = BWGNN(graph_list[0][1].shape[1], HIDDEN_DIM, 2).to(device)
    model = SAGE_BW(graph_list[0][1].shape[1], HIDDEN_DIM, 2).to(device)
    #model = DualGraphSAGEModel(ori_feat_input_dim, struc_feat_input_dim, HIDDEN_DIM, 2, 16).to(device)
    #model = SAGE(graph_list[0][1].shape[1], HIDDEN_DIM, 2).to(device)
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

            logits = model(batched_dgl_G, batched_dgl_G.ndata['feat'])
            # Compute the class weights for the current batch
            train_labels = batched_dgl_G.ndata['label'][batched_dgl_G.ndata['mask']].cpu().numpy()
            class_weights = calculate_class_weights(train_labels)
            if len(class_weights)==0: class_weights = [1.0, 1.0] # error handling
            # Update the loss function with the computed class weights
            #loss_function = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
            loss_function.weight = torch.FloatTensor(class_weights).to(device)

            loss = loss_function(logits[batched_dgl_G.ndata['mask']], batched_dgl_G.ndata['label'][batched_dgl_G.ndata['mask']])
            #loss = loss_function(logits[batched_dgl_G.ndata['mask']], batched_dgl_G.ndata['label'][batched_dgl_G.ndata['mask']])

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
            val_labels = batched_dgl_G.ndata['label'][batched_dgl_G.ndata['mask']].cpu().numpy()
            class_weights = calculate_class_weights(val_labels)
            if len(class_weights) == 0:
                class_weights = [1.0, 1.0]  # error handling
            
            # Update the loss function with the computed class weights
            loss_function = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
            
            loss = loss_function(logits[batched_dgl_G.ndata['mask']], batched_dgl_G.ndata['label'][batched_dgl_G.ndata['mask']])
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
    test_dataset = CustomGraphDataset(test_data, split='test', DIM=EMBEDDING_DIM)
    test_dataloader = GraphDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    _ = model_eval(args, test_dataloader, model, device, save_model=False, thred=best_threshold, print_stats=True)
    
    # Explain the prediction for graph 0
    # explainer = GNNExplainer(model, num_hops=1)
    # g = test_dataset[0]
    # features = g.ndata['feat']
    # labels = g.ndata['label']
    # train_mask = g.ndata['mask']
    # explainer = GNNExplainer(model, num_hops=1)
    # new_center, sg, feat_mask, edge_mask = explainer.explain_node(10, g , features)
    # print("Feature importance of node 10: ", feat_mask)
    # print("Edge importance of node 10: ", edge_mask)
    
    




