
import pickle
import torch
from GNN_Model import GCNModel, BWGNN_Hetero, BWGNN
from Dataset import CustomGraphDataset

if __name__ == "__main__":
    # load pickle file
    with open("/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/train_gcn/hwmcc07_tip_ic3ref_no_simplification_0-22.pickle", "rb") as f:
        data = pickle.load(f)
        
    # for bug fix only
    # with open("/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/train_gcn/hwmcc2020_all_only_unsat_hard_abc_no_simplification_0-9_list_name", "rb") as f: data_name = pickle.load(f)
    # for idx, ng in enumerate(data): ng[0].name = data_name[idx]
    # with open("/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/train_gcn/hwmcc2020_abc_no_simplification_0-9_only_unsat_hard.pickle", "wb") as f : pickle.dump(data, f) ; exit(0) 
        
        
    # load model 
    model = BWGNN(128, 128, 2).to('cuda:0')
    model.load_state_dict(torch.load("/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/train_gcn/hwmcc07_ic3ref_tip_no_simplification_0-22.pt"))
    model.eval()
    
    # load data
    test_dataset = CustomGraphDataset(data, split='test')
    batched_dgl_G = test_dataset[67] # 160,163,164,165,166 is small data
    batched_dgl_G = batched_dgl_G.to('cuda:0')
    logits = model(batched_dgl_G, batched_dgl_G.ndata['feat'])
    pred = logits.argmax(1).cpu().numpy()
    true_labels = batched_dgl_G.ndata['label'].cpu().numpy()
    output = [
        (data['application'])
        for idx, (_, data) in enumerate(
            list(data[67][0].nodes(data=True))
        )
        if data['type'] == 'variable'
        and data['application'].startswith('v')
        and batched_dgl_G.ndata['test_mask'].cpu().numpy()[idx]
        and pred[idx]==1
    ]
    print(output)
    #print(pred)
    #print(true_labels)