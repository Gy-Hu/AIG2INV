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
from GNN_Model import GCNModel
from Dataset import CustomGraphDataset
from Loss import FocalLoss
import argparse
from sklearn.metrics import f1_score
import torch.nn.functional as F

class ThresholdFinder:
    def __init__(self, dataloader, model, device, args):
        self.dataloader = dataloader
        self.model = model
        self.device = device
        self.args = args

    def make_predictions(self, probs, threshold):
        return (probs[:, 1] > threshold).cpu().numpy().astype(np.int64)

    def find_best_threshold(self):
        if self.dataloader.dataset.split == 'test':
            thresholds = np.linspace(0, 1, 101)
        elif self.dataloader.dataset.split == 'val':
            thresholds = np.linspace(0, 1, 101)
        best_threshold = 0
        best_confusion = None # (tn, fp, fn, tp)
        best_f1 = -1

        for threshold in thresholds:
            variable_pred_list = []
            variable_true_labels_list = []

            for batched_dgl_G in self.dataloader:
                batched_dgl_G = batched_dgl_G.to(self.device)
                
                logits = self.model(batched_dgl_G, batched_dgl_G.ndata['feat'])
                probs = F.softmax(logits, dim=1)
                pred = self.make_predictions(probs, threshold)
                true_labels = batched_dgl_G.ndata['label'].cpu().numpy()
                variable_mask = batched_dgl_G.ndata['mask'].cpu().numpy()
                # if self.dataloader.dataset.split == 'train':
                #     variable_mask = batched_dgl_G.ndata['mask'].cpu().numpy()
                # elif self.dataloader.dataset.split == 'val':
                #     variable_mask = batched_dgl_G.ndata['val_mask'].cpu().numpy()
                # elif self.dataloader.dataset.split == 'test':
                #     variable_mask = batched_dgl_G.ndata['test_mask'].cpu().numpy()
                # else:
                #     assert False, "Invalid dataset, check the dataset constructor."
                variable_pred_list.append(pred[variable_mask])
                variable_true_labels_list.append(true_labels[variable_mask])

            variable_pred = np.concatenate(variable_pred_list)
            variable_true_labels = np.concatenate(variable_true_labels_list)

            mf1 = f1_score(variable_true_labels, variable_pred,average='macro',zero_division=1)

            # Calculate metrics
            confusion = confusion_matrix(variable_true_labels, variable_pred)

            if mf1 > best_f1:
                best_f1 = mf1
                best_threshold = threshold
                best_confusion = confusion

        return best_threshold, best_f1, best_confusion