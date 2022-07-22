'''
check the relation (draw graph) to confirm the amount of latch 
and amount of invariant clauses
'''
from operator import inv
import os

if __name__ == '__main__':
    inv_file_path_prefix = "/data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/output/tip/"
    aiger_file_path_prefix = "/data/guangyuh/coding_env/AIG2INV/pyPDR/hwmcc07_tip_aag"

    # get the list of aiger files
    aiger_file_list = []
    for file in os.listdir(aiger_file_path_prefix):
        if file.endswith(".aag"):
            aiger_file_list.append(file)
    
    # get the list of invariant files
    aig_folder_name = []
    for aig_name in os.listdir(inv_file_path_prefix):
        aig_folder_name.append(aig_name)
    
    inv_file_list = []
    for aig_name in aig_folder_name:
        # get the list of invariant files
        inv_file_list = []
        for file in os.listdir(inv_file_path_prefix+aig_name):
            if file.endswith(".inv"):
                inv_file_list.append(file)
    
    print(aiger_file_list)
    print(inv_file_list)
        