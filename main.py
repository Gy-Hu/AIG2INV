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
import time

# add input arguments
import argparse
import shutil
import re

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

def subset_preproces():
    '''
    This program will copy all the aig files from {all_aig_folder} to {folder_for_prediction_result_store}
    if the aig file has generated graph (which can be used for prediction) in {aig_with_preprocess_data} 
    
    all_aig_folder: the folder that contains all the aig files (will be filtered, only the aiger that has graph will be copied)
    folder_for_prediction_result_store: used for comparision with original model checker
    aig_with_preprocess_data: the folder that contains all the aig files that has been preprocessed (has graph)
    '''
    all_aig_folder=f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/benchmark_folder/{BENCHMARK}"
    folder_for_prediction_result_store=f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/case4comp/{SELECTED_DATASET}_comp"
    aig_with_preprocess_data = f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/{SELECTED_DATASET}/bad_cube_cex2graph/json_to_graph_pickle/"
    if not os.path.exists(folder_for_prediction_result_store):
        os.makedirs(folder_for_prediction_result_store)
    else:
        #assert False, "Delete the folder first! Re-run the code!"
        # break the code
        return
        
    aig_file_list = []
    #get all file names from all the subfolders
    for root, dirs, files in os.walk(all_aig_folder):
        for file in files:
            if file.endswith(".aag"):
                all_aig_folder = os.path.join(root, file)
                aig_file_list.append(all_aig_folder)
    #print(aig_file_list)
    
    # get all folder name in big dataset
    json_path = aig_with_preprocess_data
    # get all files in the json_path
    for root, _, files in os.walk(json_path):
        files = [os.path.join(root, f) for f in files]
    # remove the _{number}.pkl and use set to remove duplicate in files list
    # Create the regular expression 
    regex = re.compile(r'(.*)_[0-9]+\.pkl$')
    # Apply the regular expression to the array elements
    output_array = [regex.match(element)[1] for element in files]
    # Remove duplicate elements from the output array
    output_array = list(set(output_array))
    aig_with_processed_graph = [ _.split('/')[-1] for _ in output_array]
    

    #create a folder for each file
    for aig_path_in_benchmark in aig_file_list :
        if list(filter(lambda x: aig_path_in_benchmark.split('/')[-1].split('.aag')[0] in x, aig_with_processed_graph)) !=[] :
            os.mkdir(f'{folder_for_prediction_result_store}/{aig_path_in_benchmark.split("/")[-1].split(".aag")[0]}')
            # copy the aig file to the folder
            shutil.copy(aig_path_in_benchmark, f'{folder_for_prediction_result_store}/{aig_path_in_benchmark.split("/")[-1].split(".aag")[0]}')
    print('Finish copying all the aig files to the corresponding folders')
    

def get_dataset(selected_dataset):
    assert os.path.exists(f"./{selected_dataset}"), "The dataset path does not exist!"
    return selected_dataset
    
def compare_abc(aig_original_location, selected_aig_case):
    #pass # WIP
    # compare with abc
    '''
    modified abc located in /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/utils/abc/abc
    '''
    # initialize a shell command
    predicted_clauses_cnf = (f'{aig_original_location}/'+ f'{selected_aig_case}_predicted_clauses_after_filtering.cnf')
    # copy the cnf file to current folder and rename to "inv.cnf"
    shutil.copy(predicted_clauses_cnf, f'{aig_original_location}/inv.cnf')
    originial_aiger_file = (f'{aig_original_location}/'+ f'{selected_aig_case}.aag')
    # if .aig not exist, excute /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/utils/aiger_tool_util/aigtoaig {selected_aig_case}.aag {selected_aig_case}.aig
    if not os.path.exists(f'{aig_original_location}/{selected_aig_case}.aig'):
        cmd = f"cd {aig_original_location} && /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/utils/aiger_tool_util/aigtoaig {selected_aig_case}.aag {selected_aig_case}.aig"
        # run the shell command without checking the output
        subprocess.run(cmd,shell=True,stderr=subprocess.STDOUT)
        
    cmd = f"cd /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/{aig_original_location} && \
    /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/utils/abc/abc -c \"&r {selected_aig_case}.aig; &put; fold; pdr\""
    # run the shell command, and store stream data in terminal output to a variable
    start_time = time.monotonic()
    try:
        output = subprocess.check_output(cmd,shell=True,stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e: # normally we will arrive here
        output = f"command '{e.cmd}' return with error (code {e.returncode}): {e.output}"
    end_time = time.monotonic()
    elapsed_time_for_nn_abc = end_time - start_time

    # read output, and split it into lines by "\\n"
    output = str(output).split('\\n')
    # Find the last Level x line, and extract the x
    last_level_nn_abc = ''
    for line in output:
        if 'unsat' in line:
            last_level_nn_abc = line
    assert last_level_nn_abc != '', 'No Level x line found'
    last_level_nn_abc = last_level_nn_abc.split(' ')
    last_level_nn_abc = last_level_nn_abc[2]

    '''
    original abc located in /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/utils/abc/abc
    '''
    # trash the inv.cnf
    os.remove(f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/{aig_original_location}/inv.cnf")
    # initialize a shell command
    cmd = f"cd /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/{aig_original_location} && \
    /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/utils/abc/abc -c \"&r {selected_aig_case}.aig; &put; fold; pdr\""
    start_time = time.monotonic()
    # run the shell command, and store stream data in terminal output to a variable
    try:
        output = subprocess.check_output(cmd,shell=True,stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e: # normally we will arrive here
        output = f"command '{e.cmd}' return with error (code {e.returncode}): {e.output}"
    end_time = time.monotonic()
    elapsed_time_for_abc = end_time - start_time

    # read output, and split it into lines by "\\n"
    output = str(output).split('\\n')
    # Find the last Level x line, and extract the x
    last_level_abc = ''
    for line in output:
        if 'unsat' in line:
            last_level_abc = line
    assert last_level_abc != '', 'abc has not found a solution'
    last_level_abc = last_level_abc.split(' ')
    last_level_abc = last_level_abc[2]

    # compare the last level
    print('NN-abc finished solving ',originial_aiger_file.split('/')[-1])
    if int(last_level_nn_abc) - int(last_level_abc) == 0:
        print('NN-abc has not improved the result')
    elif int(last_level_abc) - int(last_level_nn_abc) > 0 and elapsed_time_for_abc > elapsed_time_for_nn_abc:
        print(
            'NN-abc has been improved with  ',
            int(last_level_abc) - int(last_level_nn_abc),
            ' frames, and has converged ',
            elapsed_time_for_abc - elapsed_time_for_nn_abc,
            ' seconds earlier',
        )
    elif int(last_level_abc) - int(last_level_nn_abc) < 0 and elapsed_time_for_abc > elapsed_time_for_nn_abc:
        print(
            'NN-abc has not reduced frames, but has converged ',
            elapsed_time_for_abc - elapsed_time_for_nn_abc,
            ' seconds earlier',
        )
    elif int(last_level_abc) - int(last_level_nn_abc) > 0 and elapsed_time_for_abc < elapsed_time_for_nn_abc:
        print(
            'NN-abc has been improved with  ',
            int(last_level_abc) - int(elapsed_time_for_nn_abc),
            ' frames',
        )
    else:
        assert int(last_level_abc) - int(last_level_nn_abc) < 0 and elapsed_time_for_abc < elapsed_time_for_nn_abc, 'Something is wrong'
        print(
            'NN-abc is worse than abc. Increased ',
            int(last_level_nn_abc) - int(last_level_abc),
            ' frames',
        )

    # open a file to store the result as table, column contains the following information:
    # aig_case_name, last_level, last_level_ic3ref, elapsed_time_for_nn_ic3, elapsed_time_for_ic3ref
    # if the file does not exist, create it
    if not os.path.exists('log/compare_with_ic3_abc.csv'):
        with open('log/compare_with_ic3_abc.csv', 'w') as f:
            f.write('case name, NN-ABC Frame, ABC Frame, NN-ABC Time, ABC Time\n')
            
    
    with open('log/compare_with_ic3_abc.csv', 'a+') as f:
            
        if (
            last_level_abc.isnumeric()
        ):
            f.write(f'{originial_aiger_file.split("/")[-1].split(".aag")[0]}, {last_level_nn_abc}, {last_level_abc}, {elapsed_time_for_nn_abc}, {elapsed_time_for_abc}\n')

    print('compare with abc done')
    

def compare_ic3ref(aig_original_location, selected_aig_case, ic3ref_basic_generalization="", nnic3_basic_generalization=""):
    # compare with ic3ref

    '''
    modified ic3ref located in /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/utils/IC3ref/IC3
    '''

    # initialize a shell command
    predicted_clauses_cnf = (f'{aig_original_location}/'+ f'{selected_aig_case}_predicted_clauses_after_filtering.cnf')
    originial_aiger_file = (f'{aig_original_location}/'+ f'{selected_aig_case}.aag')
    cmd = f'/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/utils/IC3ref/IC3 -v {nnic3_basic_generalization} -f {predicted_clauses_cnf} < {originial_aiger_file}'
    # run the shell command, and store stream data in terminal output to a variable
    start_time = time.monotonic()
    try:
        output = subprocess.check_output(cmd,shell=True,stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e: # normally we will arrive here
        output = f"command '{e.cmd}' return with error (code {e.returncode}): {e.output}"
    end_time = time.monotonic()
    elapsed_time_for_nn_ic3 = end_time - start_time

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
    cmd = f'/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/utils/IC3ref/IC3 {ic3ref_basic_generalization} -v < {originial_aiger_file}'
    start_time = time.monotonic()
    # run the shell command, and store stream data in terminal output to a variable
    try:
        output = subprocess.check_output(cmd,shell=True,stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e: # normally we will arrive here
        output = f"command '{e.cmd}' return with error (code {e.returncode}): {e.output}"
    end_time = time.monotonic()
    elapsed_time_for_ic3ref = end_time - start_time

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
    print('NN-IC3ref finished solving ',originial_aiger_file.split('/')[-1])
    if last_level == last_level_ic3ref or not(last_level_ic3ref.isnumeric()): #true if last_level_ic3ref is a string
        print('NN-IC3ref has not improved the result')
    elif int(last_level_ic3ref) - int(last_level) > 0 and elapsed_time_for_ic3ref > elapsed_time_for_nn_ic3:
        print(
            'NN-IC3ref has been improved with  ',
            int(last_level_ic3ref) - int(last_level),
            ' frames, and has converged ',
            elapsed_time_for_ic3ref - elapsed_time_for_nn_ic3,
            ' seconds earlier',
        )
    elif int(last_level_ic3ref) - int(last_level) < 0 and elapsed_time_for_ic3ref > elapsed_time_for_nn_ic3:
        print(
            'NN-IC3ref has not reduced frames, but has converged ',
            elapsed_time_for_ic3ref - elapsed_time_for_nn_ic3,
            ' seconds earlier',
        )
    elif int(last_level_ic3ref) - int(last_level) > 0 and elapsed_time_for_ic3ref < elapsed_time_for_nn_ic3:
        print(
            'NN-IC3ref has been improved with  ',
            int(last_level_ic3ref) - int(last_level),
            ' frames',
        )
    else:
        assert int(last_level_ic3ref) - int(last_level) < 0 and elapsed_time_for_ic3ref < elapsed_time_for_nn_ic3, "something wrong"
        print(
            'NN-IC3ref is worse than ic3ref. Increased ',
            int(last_level) - int(last_level_ic3ref),
            ' frames',
        )

    # open a file to store the result as table, column contains the following information:
    # aig_case_name, last_level, last_level_ic3ref, elapsed_time_for_nn_ic3, elapsed_time_for_ic3ref
    # if the file does not exist, create it
    if not os.path.exists('log/compare_with_ic3ref.csv'):
        with open('log/compare_with_ic3ref.csv', 'w') as f:
            f.write('case name, NN-IC3 Frame, IC3ref Frame, NN-IC3 Time, IC3ref Time, NN-IC3-bg, IC3ref-bg\n')
            
    
    with open('log/compare_with_ic3ref.csv', 'a+') as f:
        # convert ic3ref_basic_generalization to 0 or 1
        ic3ref_basic_generalization = 0 if ic3ref_basic_generalization=="" else 1
        nnic3_basic_generalization = 0 if nnic3_basic_generalization=="" else 1
            
        if (
            last_level_ic3ref.isnumeric()
        ):
            f.write(f'{originial_aiger_file.split("/")[-1].split(".aag")[0]}, {last_level}, {last_level_ic3ref}, {elapsed_time_for_nn_ic3}, {elapsed_time_for_ic3ref}, {nnic3_basic_generalization},{ic3ref_basic_generalization} \n')

    print('compare with ic3ref done')
    
def generate_predicted_inv(threshold, aig_case_name, NN_model,aig_original_location_prefix):
    sigmoid = nn.Sigmoid()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # choose the dataset that you want to test
    extracted_bad_cube_prefix = get_dataset(selected_dataset=SELECTED_DATASET)
    # choose the case that you want to test
    selected_aig_case = aig_case_name
    extracted_bad_cube_after_post_processing = GraphDataset(f'{extracted_bad_cube_prefix}/bad_cube_cex2graph/json_to_graph_pickle/',mode='predict',case_name=selected_aig_case,device=device)
    if len(extracted_bad_cube_after_post_processing) == 0: # not a valid case, skip
        # print log 
        with open("log/error_handle/graph_pickle_incomplete.log", "a+") as fout: fout.write(f"Error: {aig_case_name} has no graph generation from json to pickle \n")
        fout.close()
        return False
    # Has error in json_to_graph_pickle
    if len(extracted_bad_cube_after_post_processing)!= len(open(f'{extracted_bad_cube_prefix}/bad_cube_cex2graph/cti_for_inv_map_checking/{selected_aig_case}/{selected_aig_case}_inv_CTI.txt').readlines()):
        # print log 
        with open("log/error_handle/graph_pickle_incomplete.log", "a+") as fout: fout.write(f"Error: {aig_case_name} has incomplete graph generation from json to pickle \n")
        fout.close()
        #XXX: Double check before running the script
        #sys.exit(0)
        return False
 
    # load pytorch model
    net = NeuroInductiveGeneralization()
    # choose the NN model that you want to test

    #NN_model_to_load = 'neuropdr_2023-01-05_15:53:59_lowest_training_loss.pth.tar' #TAG: adjust NN model name here
    #NN_model_to_load = 'neuropdr_2022-11-24_11:30:11_last.pth.tar'
    #NN_model_to_load = 'neuropdr_2023-01-06_07:56:57_last.pth.tar'

    NN_model_to_load = NN_model
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
        with torch.no_grad(): outputs = net(data)
        torch_select = torch.Tensor(q_index).to(device).int()
        outputs = sigmoid(torch.index_select(outputs, 0, torch_select))
        preds = torch.where(outputs > threshold, torch.ones(outputs.shape).to(device), torch.zeros(outputs.shape).to(device))
        # choose the state varible based on the preds, and select the 
        # element based on torch_select
        svar_lst = [(data[1][data[0]['n_nodes']:])[i] for i in torch_select.tolist()]
        # convert svar_lst to svar_lst[i]['data']['application'], i based on the preds
        # print svar_lst[i]['data']['application'] in list
        final_predicted_clauses.append([svar_lst[i]['data']['application'] for i in range(len(preds)) if preds[i] == 1])

    # print final_predicted_clauses line by line
    # for clause in final_predicted_clauses: print(clause) #TAG: uncomment this line to print the predicted clauses

    # parse file from aig original location
    aig_original_location = f'{aig_original_location_prefix}/{selected_aig_case}' #TAG: adjust the aig original location

    # number_of_subset = 1 #TAG: adjust the number of subset
    # aig_original_location = f'benchmark_folder/hwmcc2007/subset{number_of_subset}/{selected_aig_case}'

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
    print(f'{selected_aig_case} is generating predicted clauses...')
    for i in range(len(original_CTI)):
        assert final_predicted_clauses, 'Final predicted clauses is empty!'
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
    return True
    #return aig_original_location, selected_aig_case

    #compare_ic3ref(aig_original_location=aig_original_location,selected_aig_case=selected_aig_case)

def compare_inv_and_draw_table(threshold, NN_model, aig_with_predicted_location_prefix, aig_without_predicted_location_prefix):
    pass

def extract_benchmark(s):
    pattern = r'dataset_((?:(?!_abc|_ic3ref).)+)'
    match = re.search(pattern, s)
    if match:
        return match.group(1)
    else:
        return None

if __name__ == "__main__":
    global SELECTED_DATASET 
    global BENCHMARK
    # input arguments to adjust the test case, thershold, and model
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.8, help='threshold for the output of the NN model')
    #parser.add_argument('--compare_inv', action='store_true', help='compare the inv with ic3ref')
    #parser.add_argument('--aig-case-folder-prefix-for-prediction', type=str, default=None, help='case folder, use for test all cases in the folder, for example: benchmark_folder/hwmcc2007')
    #parser.add_argument('--aig-case-folder-prefix-for-ic3ref', type=str, default=None, help='case folder, contains all ic3ref produced inv.cnf, for example: benchmark_folder/hwmcc2007')
    parser.add_argument('--compare_with_ic3ref_basic_generalization', action='store_true', help='compare with ic3ref basic generalization')
    parser.add_argument('--compare_with_nnic3_basic_generalization', action='store_true', help='compare with nnic3 basic generalization')
    parser.add_argument('--aig-case-name', type=str, default=None, help='case name, use for test single case, for example: cmu.dme1.B')
    #XXX: Double check before running the script
    parser.add_argument('--NN-model', type=str, default='neuropdr_2023-01-06_07:56:57_last.pth.tar', help='model name')
    #parser.add_argument('--benchmark', type=str, default='2007', help='benchmark folder (used to filter the dataset), will convert to hwmcc{benchmark}_all')
    parser.add_argument('--gpu-id', type=str, default='1', help='gpu id')
    parser.add_argument('--compare_with_ic3ref', action='store_true', help='compare with ic3ref')
    parser.add_argument('--compare_with_abc', action='store_true', help='compare with abc')
    parser.add_argument('--selected-built-dataset', type=str, default='big', help='selected dataset to predict the clauses (dataset has been built from build_data.py)')
    args = parser.parse_args()

    

    # for test only
    '''
    
    #XXX: Double check before running the script
    args =  parser.parse_args([
        '--threshold', '0.5',
        #'--aig-case-name', 'eijk.S1423.S', #should has huge improvement
        '--aig-case-name', 'miim',
        #'--aig-case-name', 'nusmv.guidance^6.C',
        #'--aig-case-folder-prefix-for-prediction', 'benchmark_folder/hwmcc2007_big_comp_for_prediction',
        '--NN-model', 'neuropdr_2023-01-06_07:56:51_last.pth.tar',
        '--gpu-id', '1',
        #'--compare_with_ic3ref',
        '--compare_with_abc',
        #'--selected-built-dataset', 'dataset_hwmcc2007_all_no_simplification_23',
        '--selected-built-dataset', 'dataset_hwmcc2020_small_abc_slight_1',
        #'--benchmark', '2007'
        #'--benchmark', '2020'
    ])
    '''
    
    
    args = parser.parse_args([
        '--threshold', '0.5',
        '--selected-built-dataset', 'dataset_hwmcc2020_small_abc_slight_1'])
    
    
    args.compare_with_ic3ref_basic_generalization = "-b" if args.compare_with_ic3ref_basic_generalization else ""
    args.compare_with_nnic3_basic_generalization = "-b" if args.compare_with_nnic3_basic_generalization else ""
    
    
    BENCHMARK = extract_benchmark(args.selected_built_dataset)
    SELECTED_DATASET = args.selected_built_dataset
    aig_case_folder_prefix_for_prediction = f"case4comp/{SELECTED_DATASET}_comp"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    '''
    ----------------- initialize the dataset -----------------
    '''
    # initialize the dataset for validation?
    #subset_preproces()
    subset_preproces()
    # exit(0)

    '''
    ----------------- draw conclusion test -----------------
    '''

    # compare the inv with ic3ref - draw conclusion table
    # if args.compare_inv:
    #     compare_inv_and_draw_table(threshold=args.threshold, NN_model=args.NN_model, aig_with_predicted_location_prefix=aig_case_folder_prefix_for_prediction, aig_without_predicted_location_prefix=args.aig_case_folder_prefix_for_ic3ref)
    #     exit(0)

    '''
    ------------------ test single/all case -----------------
    '''

    # test single case
    if args.aig_case_name is not None:
        generate_predicted_inv_success = (
            True
            if (
                os.path.exists(
                    f'{aig_case_folder_prefix_for_prediction}/{args.aig_case_name}/{args.aig_case_name}_predicted_clauses_after_filtering.cnf'
                )
            )
            else generate_predicted_inv(
                threshold=args.threshold,
                aig_case_name=args.aig_case_name,
                NN_model=args.NN_model,
                aig_original_location_prefix=aig_case_folder_prefix_for_prediction,
            )
        )
        # if the inv is generated, then compare it with ic3ref or abc, if fail, we skip it
        if generate_predicted_inv_success and args.compare_with_abc:
            compare_abc(f'{aig_case_folder_prefix_for_prediction}/{args.aig_case_name}', f'{args.aig_case_name}')
        elif generate_predicted_inv_success and args.compare_with_ic3ref:
            compare_ic3ref(f'{aig_case_folder_prefix_for_prediction}/{args.aig_case_name}', f'{args.aig_case_name}',args.compare_with_ic3ref_basic_generalization,args.compare_with_nnic3_basic_generalization)
    else: # test all cases in specified folder
        # only give aig case folder, not define the aig case name, then test all cases in the folder
        # get all the folder name in the aig_case_folder
        aig_case_list = [ f.path for f in os.scandir(aig_case_folder_prefix_for_prediction) if f.is_dir() ]
        for aig_case in aig_case_list:
            print("Begin to test case: ", aig_case.split('/')[-1], "...")
            #if not(os.path.exists(f'{aig_case_folder_prefix_for_prediction}/{aig_case.split('/')[-1]}/{args.aig_case_name}_predicted_clauses_after_filtering.cnf')):
            generate_predicted_inv_success = (
                True
                if (
                    os.path.exists(
                        f'{aig_case_folder_prefix_for_prediction}/{aig_case.split("/")[-1]}/{aig_case.split("/")[-1]}_predicted_clauses_after_filtering.cnf'
                    )
                )
                else generate_predicted_inv(
                    threshold=args.threshold,
                    aig_case_name=aig_case.split('/')[-1],
                    NN_model=args.NN_model,
                    aig_original_location_prefix=aig_case_folder_prefix_for_prediction,
                )
            )
                
            # begin to compare the inv with ic3ref or abc
            if generate_predicted_inv_success and args.compare_with_abc:
                compare_abc(f"{aig_case_folder_prefix_for_prediction}/{aig_case.split('/')[-1]}", f"{aig_case.split('/')[-1]}",args.compare_with_abc_basic_generalization)
            elif generate_predicted_inv_success and args.compare_with_ic3ref: 
                compare_ic3ref(f"{aig_case_folder_prefix_for_prediction}/{aig_case.split('/')[-1]}", f"{aig_case.split('/')[-1]}",args.compare_with_ic3ref_basic_generalization,args.compare_with_nnic3_basic_generalization)

    '''
    ------------------ check the error log -----------------
    '''
    #print error information in log/error_handle/graph_pickle_incomplete.log if file exist
    if os.path.exists("log/error_handle/graph_pickle_incomplete.log"):
        with open("log/error_handle/graph_pickle_incomplete.log", "r") as f:
            error_info = f.read()
        print(f"Error information: {error_info}")
        # rename the graph_pickle_incomplete.log to graph_pickle_incomplete.log.{time_stamp}
        shutil.move("log/error_handle/graph_pickle_incomplete.log", f"log/error_handle/graph_pickle_incomplete.log.{time.time()}")


    







    
