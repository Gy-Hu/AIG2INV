'''
Check the subset and unsat ratio of cex (s) and invariant clause
'''

from numpy import sort
import z3
import random
# import hamming distance from sci-kit learn
#from scipy.spatial.distance import hamming


def levenshtein(cex1: str, cex2: str) -> int:
    len1, len2 = len(cex1), len(cex2)
    # Initialization
    dp = [[0 for _ in range(len2 + 1)] for _ in range(len1 + 1)]
    for i in range(len1+1): 
        dp[i][0] = i
    for j in range(len2+1):
        dp[0][j] = j
        
    # Iteration
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if cex1[i-1] == cex2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    return dp[len1][len2]

# remove duplicate list
def remove_duplicate_list(duplicate_list):
    final_list = []
    for num in duplicate_list:
        if num not in final_list:
            final_list.append(num)
    return final_list

# use edit distance to filter the cex
def filter_cex(cex_list):
    # cast to type to int and sort the list
    cast_and_sort = lambda lst: sorted([int(x) for x in lst])
    
    # check hamming distance for cex and every appended cex in the list
    # check_hamming_distance = lambda cex,filtered_cex: hamming(cex,filtered_cex) > 0.5

    # find max length cex in the list
    max_length = max([len(x) for x in cex_list])
    
    # check Levenshtein distance for cex and every appended cex in the list
    check_levenshtein_distance = lambda cex,filtered_cex: levenshtein(cex,filtered_cex) > max_length/5
    
    cex_list = [cast_and_sort(cex) for cex in cex_list]

    filtered_cex_lst = [cex_list[random.randint(0, len(cex_list)-1)]]
    for cex in cex_list[1:]:
        if cex not in filtered_cex_lst and all(check_levenshtein_distance(cex,filtered_cex) for filtered_cex in filtered_cex_lst):
            filtered_cex_lst.append(cex)
    return filtered_cex_lst


# main function
if __name__ == '__main__':
    file_path_prefix = "/data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/output/tip/"
    file_suffix = "cmu.dme1.B"
    #file_suffix = "eijk.S208o.S"
    #file_suffix = "nusmv.syncarb10^2.B"
    inv_cnf = file_path_prefix + file_suffix + "/inv.cnf"
    with open(inv_cnf, 'r') as f:
        lines = f.readlines()
        f.close()
    inv_lines = [(line.strip()).split() for line in lines]
    print("print the clauses in inv.cnf:")
    print(inv_lines[:3])

    #q_list_cnf = ['110', '141', '192', '206', '211', '231']
    #q_list_cnf =  ['114', '118', '126', '133', '134', '137', '141', '142', '144', '211', '231']

    #cex_file_path = "./cex_before_generalization_without_unsat.txt"
    #cex_file_path = "./nusmv.syncarb10^2.B_complete_CTI.txt"
    #cex_file_path = "./nusmv.syncarb10^2.B_partial_CTI.txt"
    cex_file_path = "./cmu.dme1.B_complete_CTI.txt"
    with open(cex_file_path, 'r') as f:
        lines = f.readlines()
        f.close()
    cex_lines = [(line.strip()).split(',') for line in lines]
    # sort the cex in cex_lines
    print("length of lines:", len(cex_lines),end='---->')
    cex_lines = [sorted(line) for line in cex_lines]
    cex_lines = remove_duplicate_list(cex_lines)
    print(len(cex_lines))
    print("print the clauses in json")
    print(cex_lines[:3])
    
    # decide whether to filter the cex
    # cex_lines = filter_cex(cex_list=cex_lines)

    # Record the sucess rate of finding the inductive clauses in inv.cnf
    subset_fail = 0
    subset_fail_normal = 0
    subset_fail_wrost = 0

    # Record the unsat pass ratio
    unsat_success = 0
    unsat_fail = 0
    unsat_fail_normal = 0
    unsat_fail_wrost = 0

    all_subset_fail = []
    all_subset_success = []
    # scan every clause in inv.cnf
    for clause in inv_lines[1:]:
        this_subset_fail = 0
        this_subset_success = 0
        clause = [int(num) for num in clause]
        # assert all the cex in cex_lines length is the same
        if cex_file_path.split('_')[-2] == 'complete': assert(all(len(l) == len(cex_lines[0]) for l in cex_lines))
        # scan every cex in cex_lines
        for cex in cex_lines:
            cex = [int(num) for num in cex]
            # check the subset
            if set(clause).issubset(set(cex)):
                this_subset_success += 1
            else:
                this_subset_fail += 1
        
        all_subset_fail.append(this_subset_fail)
        all_subset_success.append(this_subset_success)
            
        # Check subset fail
        if this_subset_fail == len(cex_lines):
            subset_fail += 1
    
    print(subset_fail, "inductive invariant has not found mapping model,","fail rate of mapping invariant to model is", (subset_fail/len(inv_lines))* 100, "%")


