# load data extract from bad cube and predict it

import os
import sys
import torch
from tqdm import tqdm
sys.path.append('./train_neurograph/')
from train_neurograph.train import GraphDataset
from train_neurograph.neurograph_old import NeuroInductiveGeneralization
# add "../utils" to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.toolbox import walkFile
# add "train_neurograph" to sys.path
import torch.nn as nn


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    sigmoid = nn.Sigmoid()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extracted_bad_cube = 'dataset/bad_cube_cex2graph/json_to_graph_pickle/'
    extracted_bad_cube_after_post_processing = GraphDataset(extracted_bad_cube,mode='predict',case_name='nusmv.syncarb5^2.B',device=device)

    # load pytorch model
    net = NeuroInductiveGeneralization()
    model = torch.load('./neurograph_model/neuropdr_2022-11-24_11:30:11_last.pth.tar',map_location=device)
    net.load_state_dict(model['state_dict'])
    net = net.to(device)
    net.eval()
    # predict, load extracted_bad_cube_after_post_processing one by one
    final_predicted_clauses = []
    for i in tqdm(range(len(extracted_bad_cube_after_post_processing))):
        data = extracted_bad_cube_after_post_processing[i]
        q_index = data[0]['refined_output']
        outputs = net(data)
        torch_select = torch.Tensor(q_index).to(device).int()
        outputs = sigmoid(torch.index_select(outputs, 0, torch_select))
        preds = torch.where(outputs > 0.8, torch.ones(outputs.shape).to(device), torch.zeros(outputs.shape).to(device))
        # choose the state varible based on the preds, and select the 
        # element based on torch_select
        svar_lst = [(data[1][data[0]['n_nodes']:])[i] for i in torch_select.tolist()]
        # convert svar_lst to svar_lst[i]['data']['application'], i based on the preds
        # print svar_lst[i]['data']['application'] in list
        final_predicted_clauses.append([svar_lst[i]['data']['application'] for i in range(len(preds)) if preds[i] == 1])

    # print final_predicted_clauses line by line
    for clause in final_predicted_clauses: print(clause)
    # parse file from case4test/hwmcc_simple
    CTI_file = 'dataset/bad_cube_cex2graph/cti_for_inv_map_checking/nusmv.syncarb5^2.B/nusmv.syncarb5^2.B_inv_CTI.txt'
    Predict_Clauses_File = 'case4test/hwmcc_simple/nusmv.syncarb5^2.B/nusmv.syncarb5^2.B_inv_CTI_predicted.txt'
    with open(CTI_file,'r') as f:
        original_CTI = f.readlines()
    # remove the last '\n'
    original_CTI = [i[:-1] for i in original_CTI]
    # split original_CTI into list with comma
    original_CTI = [clause.split(',') for clause in original_CTI]
    # filter the original_CTI with final_predicted_clauses
    # first, convert final_predicted_clauses to a list that without 'v'
    final_predicted_clauses = [[literal.replace('v','') for literal in clause] for clause in final_predicted_clauses]
    final_generate_res = [] # this will be side loaded to ic3ref
    for i in range(len(original_CTI)):
        # generalize the original_CTI[i] with final_predicted_clauses[i]
        # if the literal in original_CTI[i] is not in final_predicted_clauses[i], then remove it
        cls = [literal for literal in original_CTI[i] if literal in final_predicted_clauses[i] or str(int(literal)-1) in final_predicted_clauses[i]]
        final_generate_res.append(cls)
    # write final_generate_res to Predict_Clauses_File
    with open(Predict_Clauses_File,'w') as f:
        # write the first line with basic info
        f.write(f'unsat {len(final_generate_res)}' + '\n')
        for clause in final_generate_res:
            f.write(' '.join(clause))
            f.write('\n')
            
#TODO: Check the final result, all use them????

    







    
