# load data extract from bad cube and predict it

import os
import sys
import torch
from tqdm import tqdm
# append './train_neurograph/' to the path
sys.path.append(os.path.join(os.getcwd(), 'train_neurograph'))
# append './data2dataset/cex2smt2/' to the path
sys.path.append(os.path.join(os.getcwd(), 'data2dataset/cex2smt2'))
from data2dataset.cex2smt2.tCube import tCube
from train_neurograph.train import GraphDataset
# for old(small) cases
#from train_neurograph.neurograph_old import NeuroInductiveGeneralization
# for new(complicated/large) cases
from train_neurograph.neurograph import NeuroInductiveGeneralization

from data2dataset.cex2smt2.clause import Clauses
from data2dataset.cex2smt2.aigmodel import AAGmodel
from data2dataset.cex2smt2.aig2graph import AigGraph
from data2dataset.cex2smt2.cnfextract import ExtractCnf
# add "train_neurograph" to sys.path
import torch.nn as nn
from natsort import natsorted
import z3
import subprocess

# add input arguments
import argparse

'''
---------------------------------------------------------
Checker to check if the predicted clauses is satisfiable
---------------------------------------------------------
'''
class CNF_Filter(ExtractCnf):
    def __init__(self, aagmodel, clause, name, aig_location=None):
       super(CNF_Filter, self).__init__(aagmodel, clause, name)
       # self.init = aagmodel.init
       
       # adjust to perform inductive generalization or not
       self.perform_ig = False
       # record the original aig location
       self.aig_location = aig_location
    
    def _solveRelative_upgrade(self, clauses_to_block):
        # sourcery skip: extract-duplicate-method, inline-immediately-returned-variable
        # not(clauses_to_block) is the counterexample (which is also call s)
        # self.aagmodel.output is the bad state
        # prop = safety property which is !bad
        
        # init /\ s is SAT?
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

        if check_init == z3.unsat and check_relative == z3.unsat and self.perform_ig == False:
            return 'pass the check'
        if check_init == z3.unsat and check_relative == z3.unsat and self.perform_ig == True:
            s_after_ig = self._inductive_generalization(clauses_to_block)
            print('pass the check and generalize the clauses')
            return s_after_ig
        else:
            return 'not pass'

    def _inductive_generalization(self, clauses_to_block):
        # performs unsat core generalization
        # pass

        # perform mic generalization
        tcube2generalize = tCube(0)
        tcube2generalize.cubeLiterals = clauses_to_block.children()[0].children()
        return self._MIC(tcube2generalize)

    def _unsatcore_reduce(self, q, frame):
        # (( not(q) /\ F /\ T ) \/ init' ) /\ q'   is unsat
        slv = z3.Solver()
        slv.set(unsat_core=True)

        l = z3.Or(z3.And(z3.Not(q.cube()), frame), (z3.substitute(z3.substitute(self.init, self.v2prime), self.vprime2nxt)))
        slv.add(l)

        plist = []
        for idx, literal in enumerate(q.cubeLiterals):
            p = 'p'+str(idx)
            slv.assert_and_track((z3.substitute(z3.substitute(literal, self.v2prime), self.vprime2nxt)), p)
            plist.append(p)
        res = slv.check()
        if res == z3.sat:
            model = slv.model()
            print(model.eval(self.initprime))
            assert False,'BUG: !s & F & T & s\' is not inductive or init\' & s\' is not inductive'
        assert (res == z3.unsat)
        core = slv.unsat_core()
        for idx, p in enumerate(plist):
            if z3.Bool(p) not in core:
                q.cubeLiterals[idx] = True
        return q
    
    def _MIC(self, q: tCube):
        sz = q.true_size()
        # perform unsat core reduction first
        self._unsatcore_reduce(q, frame=self.init)
        print('unsatcore', sz, ' --> ', q.true_size())
        q.remove_true()
        if q.true_size() == 1: return q # no need to perform MIC
        return q #FIXME: temporarily disable MIC

        # maybe q can be reduced by MIC further
        for i in range(len(q.cubeLiterals)):
            if q.cubeLiterals[i] is True: #This true does not indicate the literals are true
                continue
            q1 = q.delete(i)
            print(f'MIC try idx:{i}')
            if self._down(q1): 
                q = q1
        q.remove_true()
        print (q)
        return q

    def _down(self, q: tCube):
        while True:
            print(q.true_size(), end=',')
            s = z3.Solver()
            s.push()
            #s.add(And(self.frames[0].cube(), Not(q.cube())))
            s.add(self.frames[0].cube())
            s.add(q.cube())
            #if unsat == s.check():
            if z3.sat == s.check():
                print('F')
                return False
            s.pop()
            s.push()
            s.add(z3.And(self.frames[q.t-1].cube(), z3.Not(q.cube()), self.trans.cube(), #TODO: Check here is t-1 or t
                      z3.substitute(z3.substitute(q.cube(), self.primeMap),self.inp_map)))  # Fi-1 ! and not(q) and T and q'
            if z3.unsat == s.check():
                print('T')
                return True
            
            m = s.model()
            has_removed = q.join(m)
            s.pop()
            assert (has_removed)

    def _sort_passed_clauses(self,lines, passed_clauses):
        '''
        sort the passed clauses according to the number of literals
        lines: the list of clauses
        passed_clauses: the list of passed clauses
        '''
        # get the passed clauses list according to passed_clauses
        passed_clauses = [lines[i+1] for i in passed_clauses]
        # initialize a sorted list to store the clauses
        passed_clauses_sorted = []
        # strip the newline character
        passed_clauses_sorted = [i.strip() for i in passed_clauses]
        # make the clauses to list of string
        passed_clauses_sorted = [i.split(' ') for i in passed_clauses_sorted]
        # for every clause, smaller literals first
        for clause in passed_clauses_sorted: clause.sort()
        # sort the clauses according to first literal
        passed_clauses_sorted.sort(key=lambda x: x[0])
        # sort the clauses according to the length of the clauses and the number of literals
        passed_clauses_sorted.sort(key=lambda x: (len(x), x))
        # change to list of clauses to string, append the newline character
        passed_clauses_sorted = [' '.join(i) + '\n' for i in passed_clauses_sorted]
        return passed_clauses_sorted

    def check_and_reduce(self):
        '''
        Check the predicted clauses, if passed, then dump it without generalization 
        check_and_reduce or check_and_generalize 2 options
        choose one of the two options
        '''
        prop = z3.Not(self.aagmodel.output) # prop - safety property
        passed_clauses = [i for i in range(len(self.clauses)) if self._solveRelative_upgrade(self.clauses[i]) == 'pass the check']
        # process the inductive generalization of the passed clauses -> basic generalization (unsat core) and mic
        # generalized_clauses = [i for i in range(len(passed_clauses)) if self._inductive_generalization(passed_clauses[i]) == 'generalized successfully']
        Predict_Clauses_Before_Filtering = f'{self.aig_location}/{self.model_name}_inv_CTI_predicted.cnf'
        Predict_Clauses_After_Filtering = f'{self.aig_location}/{self.model_name}_predicted_clauses_after_filtering.cnf'
        #print(f"Dump the predicted clauses after filtering to {Predict_Clauses_After_Filtering}")
        print("Finish dumping the predicted clauses after filtering passed clauses (solve relative checking)!!")
        # copy the line in Predict_Clauses_Before_Filtering to Predict_Clauses_After_Filtering according to passed_clauses
        with open(Predict_Clauses_Before_Filtering, 'r') as f:
            lines = f.readlines()
        with open(Predict_Clauses_After_Filtering, 'w') as f:
            f.write(f'unsat {len(passed_clauses)}' + '\n')
            passed_and_sorted_clauses =  self._sort_passed_clauses(lines, passed_clauses) # finish filtering the clauses, and sort the clauses
            for clause in passed_and_sorted_clauses: f.write(clause)
        
    def check_and_generalize(self):
        '''
        check the predicted clause, if passed, then generalize it -> use unsat core and mic
        check_and_reduce or check_and_generalize 2 options
        choose one of the two options
        '''
        prop = z3.Not(self.aagmodel.output)
        pass_and_generalized_clauses = [
            self._solveRelative_upgrade(self.clauses[i]) 
            for i in range(len(self.clauses))
        ]
        # delete the clauses that are not passed, check its type, if string, then it is not passed
        pass_and_generalized_clauses = [
            clause for clause in pass_and_generalized_clauses if type(clause) != str
        ]
        print("finish checking and generalizing the predicted clauses")
        # begin to reduce the duplicated clauses
        pass_and_generalized_clauses = [
            list(t)
            for t in {
                tuple(
                    sorted(
                        cube.cubeLiterals,
                        key=lambda x: int(str((x).children()[0]).replace('v', '')) if (x).children()!=[] else int(str((x)).replace('v', ''))
                    )
                )
                for cube in pass_and_generalized_clauses
            }
        ]
        #pass_and_generalized_clauses = [tCube(original_s_3.t, cube_lt_lst) for cube_lt_lst in pass_and_generalized_clauses]
        pass_and_generalized_clauses_converter = []
        for _ in pass_and_generalized_clauses:
            res = tCube(0)
            res.cubeLiterals = _.copy()
            pass_and_generalized_clauses_converter.append(res)
        pass_and_generalized_clauses = pass_and_generalized_clauses_converter

        Predict_Clauses_After_Filtering_and_Generalization = f'{self.aig_location}/{self.model_name}_predicted_clauses_after_filtering_and_generalization.cnf'
        # write final_generate_res to Predict_Clauses_File
        cubeliteral_to_str = lambda cube_literals: ','.join(map
                                (lambda x: str(x).replace('v','') 
                                # if x is v2, v4, v6 ... rather than Not(v2), Not(v4), Not(v6) ...
                                if x.children() == []
                                else str(int(str(x.children()[0]).replace('v',''))+1),cube_literals))

        with open(Predict_Clauses_After_Filtering_and_Generalization,'w') as f:
            # write the first line with basic info
            f.write(f'unsat {len(pass_and_generalized_clauses)}' + '\n')
            for clause in pass_and_generalized_clauses:
                #FIXME: why every time the clauses are not the same? -> set is disordered
                f.write((cubeliteral_to_str(clause.cubeLiterals)))
                f.write('\n')


'''
-----------------------
Global Used Functions  
-----------------------
'''
def walkFile(self):
    files = []
    for root, _, files in os.walk(self):
        files = natsorted(files)
        files = [os.path.join(root,f) for f in files]
    assert len(files) > 0, f"No files found in {self}"
    return files

def get_dataset(selected_dataset = 'toy'):
    if selected_dataset == 'complete':
        return 'dataset'
    elif selected_dataset == 'toy':
        return 'dataset_20230106_014957_toy'
    elif selected_dataset == 'small':
        return 'dataset_20230106_025223_small'

def compare_ic3ref(aig_original_location, selected_aig_case):
    # compare with ic3ref
    
    '''
    modified ic3ref located in /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/utils/IC3ref/IC3
    '''

    # initialize a shell command
    predicted_clauses_cnf = (f'{aig_original_location}/'+ f'{selected_aig_case}_predicted_clauses_after_filtering.cnf')
    originial_aiger_file = (f'{aig_original_location}/'+ f'{selected_aig_case}.aag')
    cmd = f'/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/utils/IC3ref/IC3 -v -f {predicted_clauses_cnf} < {originial_aiger_file}'
    # run the shell command, and store stream data in terminal output to a variable
    try:
        output = subprocess.check_output(cmd,shell=True,stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e: # normally we will arrive here
        output = f"command '{e.cmd}' return with error (code {e.returncode}): {e.output}"

    # read output, and split it into lines by "\\n"
    output = output.split('\\n')
    # Find the last Level x line, and extract the x
    last_level = ''
    for line in output:
        if 'Level' in line:
            last_level = line
    assert last_level != '', 'No Level x line found'
    last_level = last_level.split(' ')
    last_level = last_level[1]

    '''
    original ic3ref located in /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/utils/IC3ref/IC3ref
    '''
    # initialize a shell command
    cmd = f'/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/utils/IC3ref/IC3 -v < {originial_aiger_file}'
    # run the shell command, and store stream data in terminal output to a variable
    try:
        output = subprocess.check_output(cmd,shell=True,stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e: # normally we will arrive here
        output = f"command '{e.cmd}' return with error (code {e.returncode}): {e.output}"
    
    # read output, and split it into lines by "\\n"
    output = output.split('\\n')
    # Find the last Level x line, and extract the x
    last_level_ic3ref = ''
    for line in output:
        if 'Level' in line:
            last_level_ic3ref = line
    assert last_level_ic3ref != '', 'ic3ref has not found a solution'
    last_level_ic3ref = last_level_ic3ref.split(' ')
    last_level_ic3ref = last_level_ic3ref[1]

    # compare the last level
    if last_level == last_level_ic3ref:
        print('NN-IC3ref has not improved the result')
    else:
        print('NN-IC3ref has improved the result by ',int(last_level_ic3ref) - int(last_level),' frames')

    print('compare with ic3ref done')


if __name__ == "__main__":
    # input arguments to adjust the test case, thershold, and model
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.8, help='threshold for the output of the NN model')
    parser.add_argument('--aig-case-name', type=str, default='nusmv.syncarb5^2.B', help='case name')
    parser.add_argument('--NN-model', type=str, default='neuropdr_2023-01-06_07:56:57_last.pth.tar', help='model name')
    parser.add_argument('--gpu-id', type=str, default='1', help='gpu id')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    sigmoid = nn.Sigmoid()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # choose the dataset that you want to test
    extracted_bad_cube_prefix = get_dataset(selected_dataset='toy')
    # choose the case that you want to test
    selected_aig_case = args.aig_case_name
    extracted_bad_cube_after_post_processing = GraphDataset(f'{extracted_bad_cube_prefix}/bad_cube_cex2graph/json_to_graph_pickle/',mode='predict',case_name=selected_aig_case,device=device)

    # load pytorch model
    net = NeuroInductiveGeneralization()
    # choose the NN model that you want to test
    
    #NN_model_to_load = 'neuropdr_2023-01-05_15:53:59_lowest_training_loss.pth.tar' #TAG: adjust NN model name here
    #NN_model_to_load = 'neuropdr_2022-11-24_11:30:11_last.pth.tar'
    #NN_model_to_load = 'neuropdr_2023-01-06_07:56:57_last.pth.tar'

    NN_model_to_load = args.NN_model
    model = torch.load(f'./neurograph_model/{NN_model_to_load}',map_location=device)
    
    # for small case
    #model = torch.load(f'./neurograph_model/neuropdr_2022-11-24_11:30:11_last.pth.tar',map_location=device)
    # for large case
    #model = torch.load('./neurograph_model/neuropdr_2022-11-28_15:23:41_last.pth.tar',map_location=device)
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
        preds = torch.where(outputs > args.threshold, torch.ones(outputs.shape).to(device), torch.zeros(outputs.shape).to(device))
        # choose the state varible based on the preds, and select the 
        # element based on torch_select
        svar_lst = [(data[1][data[0]['n_nodes']:])[i] for i in torch_select.tolist()]
        # convert svar_lst to svar_lst[i]['data']['application'], i based on the preds
        # print svar_lst[i]['data']['application'] in list
        final_predicted_clauses.append([svar_lst[i]['data']['application'] for i in range(len(preds)) if preds[i] == 1])

    # print final_predicted_clauses line by line
    # for clause in final_predicted_clauses: print(clause) #TAG: uncomment this line to print the predicted clauses
    
    # parse file from aig original location
    aig_original_location = f'case4test/hwmcc_simple/{selected_aig_case}' #TAG: adjust the aig original location
    
    # number_of_subset = 1 #TAG: adjust the number of subset
    # aig_original_location = f'case4test/hwmcc2007/subset{number_of_subset}/{selected_aig_case}'
    
    CTI_file = f'{extracted_bad_cube_prefix}/bad_cube_cex2graph/cti_for_inv_map_checking/{selected_aig_case}/{selected_aig_case}_inv_CTI.txt'
    Predict_Clauses_File = f'{aig_original_location}/{selected_aig_case}_inv_CTI_predicted.cnf'
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

    # remove the duplicate clause in final_generate_res
    # be careful, using set will change the order of the list
    final_generate_res = [
        list(t) for t in {tuple(element) for element in final_generate_res}
    ]

    # write final_generate_res to Predict_Clauses_File
    with open(Predict_Clauses_File,'w') as f:
        # write the first line with basic info
        f.write(f'unsat {len(final_generate_res)}' + '\n')
        for clause in final_generate_res:
            f.write(' '.join(clause))
            f.write('\n')

    # check the final_generate_res with ic3ref -> whether it is fulfill the property
    case = selected_aig_case
    aag_name = f"./{aig_original_location}/{case}.aag" 
    cnf_name = f"./{aig_original_location}/{case}_inv_CTI_predicted.cnf"
    model_name = case
    m = AAGmodel()
    m.from_file(aag_name)
    predicted_clauses = Clauses(fname=cnf_name, num_sv = len(m.svars), num_input = len(m.inputs))
    predicted_clauses_filter = CNF_Filter(aagmodel = m, clause = predicted_clauses ,name = model_name, aig_location=aig_original_location)
    predicted_clauses_filter.check_and_reduce()
    #predicted_clauses_filter.check_and_generalize()#FIXME: Encounter error, the cnf file will become empty
    
    compare_ic3ref(aig_original_location=aig_original_location,selected_aig_case=selected_aig_case)
#TODO: Check the final result, all use them????

    







    