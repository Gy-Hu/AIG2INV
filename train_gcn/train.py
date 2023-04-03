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
from BWGNN import BWGNN_Hetero, BWGNN, PolyConv, PolyConvBatch

#################################
# WIP:
# 1. generate predicted clauses from json (json -> graph -> model eval() -> predicted clauses)
# 2. use better features (features engineering on the graph: transition relation, initial state, etc.)
# 3. use better model (STGCN, GAT, BWGNN, etc.)
# 4. use more graph (add more graphs to the dataset, hwmcc2007, etc.)
# 5. solve the imbalanced data problem (class weights, moving thereshold, focal loss, etc.)
# 6. weird loss <1
# 7. calculate the perfect accuracy (all the clauses are correct)
#################################

JSON_FOLDER = '/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc2007_tip_ic3ref_no_simplification_0-22/bad_cube_cex2graph/expr_to_build_graph/nusmv.syncarb5^2.B/'
GROUND_TRUTH = os.path.join(JSON_FOLDER.replace('expr_to_build_graph', 'ground_truth_table'), JSON_FOLDER.split('/')[-2]+'.csv')
json_files = [f for f in os.listdir(JSON_FOLDER) if f.endswith('.json')]
graph_list = []

def calculate_class_weights(train_labels):
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    return total_samples / (len(class_counts) * class_counts)

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

# additional step: get the unique attributes

# extract unique attributes from all nodes across all graphs:
unique_attributes = set()
for G, _, _, _ in graph_list:
    for _, node_data in G.nodes(data=True):
        unique_attributes.add(node_data['type'])
        unique_attributes.add(node_data['application'])

# Create an embedding for each unique attribute and store them in a dictionary
embedding_dim = 16
attribute_embedding = {attr: np.random.rand(embedding_dim) for attr in unique_attributes}

# Modify the loop that assigns node features to use the attribute embeddings
for idx, (G, _, node_labels, train_mask) in enumerate(graph_list):
    node_features = np.zeros((G.number_of_nodes(), embedding_dim))
    
    for i, (_, node_data) in enumerate(G.nodes(data=True)):
        node_features[i] = attribute_embedding[node_data['type']] + attribute_embedding[node_data['application']]

    graph_list[idx] = (G, node_features, node_labels, train_mask)


# 4. Create a Graph Convolutional Network (GCN) model and custom dataset
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

class CustomGraphDataset(dgl.data.DGLDataset):
    def __init__(self, graph_list):
        self.graph_list = graph_list
        super(CustomGraphDataset, self).__init__(name='custom')

    def __getitem__(self, idx):
        G, node_features, node_labels, train_mask = self.graph_list[idx]
        dgl_G = dgl.from_networkx(G)
        dgl_G.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32)
        dgl_G.ndata['label'] = torch.tensor(node_labels, dtype=torch.long)
        dgl_G.ndata['train_mask'] = torch.tensor(train_mask)
        return dgl_G

    def __len__(self):
        return len(self.graph_list)
    
# Focal loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

dataset = CustomGraphDataset(graph_list)
dataloader = GraphDataLoader(dataset, batch_size=2, shuffle=True, drop_last=False)

# 5. Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get the input feature dimension
input_feature_dim = embedding_dim
#model = GCNModel(input_feature_dim, 16, 2).to(device)
model = BWGNN(input_feature_dim, 16, 2).to(device)
#loss_function = nn.CrossEntropyLoss()
loss_function = FocalLoss()
optimizer = Adam(model.parameters(), lr=0.01)


for epoch in range(50):
    model.train()
    epoch_loss = 0
    for batched_dgl_G in dataloader:
        batched_dgl_G = batched_dgl_G.to(device)
        logits = model(batched_dgl_G, batched_dgl_G.ndata['feat'])
        # Compute the class weights for the current batch
        train_labels = batched_dgl_G.ndata['label'][batched_dgl_G.ndata['train_mask']].cpu().numpy()
        class_weights = calculate_class_weights(train_labels)

        # Update the loss function with the computed class weights
        loss_function = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))

        loss = loss_function(logits[batched_dgl_G.ndata['train_mask']], batched_dgl_G.ndata['label'][batched_dgl_G.ndata['train_mask']])
        #loss = loss_function(logits[batched_dgl_G.ndata['train_mask']], batched_dgl_G.ndata['label'][batched_dgl_G.ndata['train_mask']])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"EPOCH: {epoch}, LOSS: {epoch_loss / len(dataloader)}")

# 6. Evaluate the model
model.eval()
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




