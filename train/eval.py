
import pickle
import torch
from GNN_Model import GCNModel, BWGNN_Hetero, BWGNN
from Dataset import CustomGraphDataset
from dgl.dataloading import GraphDataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

def update_progress_bar(pbar):
    pbar.update(1)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load pickle file
    with open("/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/train_gcn/dataset_hwmcc2007_tip_ic3ref_no_simplification_0-22.pickle", "rb") as f:
        data = pickle.load(f)
    val_dataset = CustomGraphDataset(data, split='val')
    #val_dataloader = GraphDataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=False)
    # for bug fix only
    # with open("/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/train_gcn/hwmcc2020_all_only_unsat_hard_abc_no_simplification_0-9_list_name", "rb") as f: data_name = pickle.load(f)
    # for idx, ng in enumerate(data): ng[0].name = data_name[idx]
    # with open("/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/train_gcn/hwmcc2020_abc_no_simplification_0-9_only_unsat_hard.pickle", "wb") as f : pickle.dump(data, f) ; exit(0) 
        
        
    # load model 
    model = BWGNN(128, 128, 2).to('cuda:0')
    model.load_state_dict(torch.load("/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/train_gcn/dataset_hwmcc2007_tip_ic3ref_no_simplification_0-22.pt"))
    model.eval()

    # 6. Evaluate the model

    pred_list = []
    true_labels_list = []
    variable_pred_list = []
    variable_true_labels_list = []

    #for batched_dgl_G in test_dataloader:
    with tqdm(total=len(val_dataset)) as pbar:
        for batched_dgl_G in val_dataset:
            batched_dgl_G = batched_dgl_G.to(device)
            logits = model(batched_dgl_G, batched_dgl_G.ndata['feat'])
            pred = logits.argmax(1).cpu().numpy()
            true_labels = batched_dgl_G.ndata['label'].cpu().numpy()

            # Create variable_mask for the batched_dgl_G
            variable_mask = batched_dgl_G.ndata['mask'].cpu().numpy()

            pred_list.append(pred)
            true_labels_list.append(true_labels)
            variable_pred_list.append(pred[variable_mask])
            variable_true_labels_list.append(true_labels[variable_mask])
            update_progress_bar(pbar)

    # Concatenate predictions and true labels
    pred = np.concatenate(pred_list)
    true_labels = np.concatenate(true_labels_list)
    variable_pred = np.concatenate(variable_pred_list)
    variable_true_labels = np.concatenate(variable_true_labels_list)

    # Calculate metrics
    accuracy = accuracy_score(variable_true_labels, variable_pred)
    precision = precision_score(variable_true_labels, variable_pred)
    recall = recall_score(variable_true_labels, variable_pred)
    mf1 = f1_score(variable_true_labels, variable_pred,average='macro',zero_division=1)
    confusion = confusion_matrix(variable_true_labels, variable_pred)

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score: ", mf1)
    print("Confusion matrix:")
    print(confusion)
    




