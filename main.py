# load data extract from bad cube and predict it

import os
import sys
import torch
from tqdm import tqdm
# append './train_neurograph/' to the path
sys.path.append(os.path.join(os.getcwd(), 'train_neurograph'))
# append './data2dataset/cex2smt2/' to the path
sys.path.append(os.path.join(os.getcwd(), 'data2dataset/cex2smt2'))
from train_neurograph.train import GraphDataset
from train_neurograph.neurograph_old import NeuroInductiveGeneralization
from data2dataset.cex2smt2.clause import Clauses
from data2dataset.cex2smt2.aigmodel import AAGmodel
from data2dataset.cex2smt2.aig2graph import AigGraph
from data2dataset.cex2smt2.cnfextract import ExtractCnf
# add "train_neurograph" to sys.path
import torch.nn as nn
from natsort import natsorted
import z3

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''
---------------------------------------------------------
Checker to check if the predicted clauses is satisfiable
---------------------------------------------------------
'''
class CNF_Filter(ExtractCnf):
    def __init__(self, aagmodel, clause, name):
       super(CNF_Filter, self).__init__(aagmodel, clause, name)
       self.init = aagmodel.init
    
    def _solveRelative_upgrade(self, clauses_to_block):
        # sourcery skip: extract-duplicate-method, inline-immediately-returned-variable
        # not(clauses_to_block) is the counterexample (which is also call s)
        # self.aagmodel.output is the bad state
        # prop = safety property which is !bad
        check_init = z3.sat
        slv = z3.Solver()
        slv.add(self.init) # init -> !s ?
        slv.add(z3.Not((clauses_to_block)))
        check_init = slv.check()

        check_relative = z3.sat # init & !s & T -> !s' ?
        cubePrime = z3.substitute(z3.substitute(z3.Not(clauses_to_block), self.v2prime), self.vprime2nxt)
        s = z3.Solver()
        s.add(clauses_to_block)
        s.add(self.init)
        s.add(cubePrime)  # F[i - 1] and T and Not(badCube) and badCube'
        check_relative = s.check()

        if check_init == z3.unsat and check_relative == z3.unsat:
            return 'pass the check'
        else:
            return 'not pass'

    def check_and_reduce(self):
        prop = z3.Not(self.aagmodel.output) # prop - safety property
        pass_clauses = [i for i in range(len(self.clauses)) if self._solveRelative_upgrade(self.clauses[i]) == 'pass the check']
        Predict_Clauses_Before_Filtering = 'case4test/hwmcc_simple/nusmv.syncarb5^2.B/nusmv.syncarb5^2.B_inv_CTI_predicted.txt'
        Predict_Clauses_After_Filtering = 'case4test/hwmcc_simple/nusmv.syncarb5^2.B/nusmv.syncarb5^2.B_predicted_clauses_after_filtering.cnf'
        print(f"Dump the predicted clauses after filtering to {Predict_Clauses_After_Filtering}")
        # copy the line in Predict_Clauses_Before_Filtering to Predict_Clauses_After_Filtering according to pass_clauses
        with open(Predict_Clauses_Before_Filtering, 'r') as f:
            lines = f.readlines()
        with open(Predict_Clauses_After_Filtering, 'w') as f:
            f.write(f'unsat {len(pass_clauses)}' + '\n')
            for i in pass_clauses:
                f.write(lines[i+1])
        
        # prop = z3.Not(self.aagmodel.output) # get the property
        # slv = z3.Solver()
        # prev = z3.And([prop]) # the property, also known as the initial state
        # slv.add(prev)

        # post = prop
        # # get the !s'
        # not_p_prime = z3.Not(z3.substitute(z3.substitute(post, self.v2prime), self.vprime2nxt))
        # slv.add(not_p_prime) # solver: Rk−1 ∧ T ∧ s′ is SAT? 
        # res = slv.check()

'''
-----------------------
Global Used Functions  
-----------------------
'''
def walkFile(self):
    for root, _, files in os.walk(self):
        files = natsorted(files)
        files = [os.path.join(root,f) for f in files]
    return files

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

    # check the final_generate_res with ic3ref -> whether it is fulfill the property
    case = "nusmv.syncarb5^2.B"
    aag_name = f"./case4test/hwmcc_simple/{case}/{case}.aag"
    cnf_name = f"./case4test/hwmcc_simple/{case}/{case}_inv_CTI_predicted.txt"
    model_name = case
    m = AAGmodel()
    m.from_file(aag_name)
    predicted_clauses = Clauses(fname=cnf_name, num_sv = len(m.svars), num_input = len(m.inputs))
    predicted_clauses_filter = CNF_Filter(aagmodel = m, clause = predicted_clauses ,name = model_name)
    predicted_clauses_filter.check_and_reduce()
            
#TODO: Check the final result, all use them????

    







    
