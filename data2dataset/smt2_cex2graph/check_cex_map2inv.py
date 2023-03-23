'''
Check the subset and unsat ratio of cex (s) and invariant clause
'''

import z3

# remove duplicate list
def remove_duplicate_list(duplicate_list):
    final_list = []
    for num in duplicate_list:
        if num not in final_list:
            final_list.append(num)
    return final_list

def unsat_test(cex_lines, inv_lines):
    '''
    Check the unsat ratio of cex and invariant clause
    '''
    # Record the unsat pass ratio
    unsat_success = 0
    unsat_fail = 0
    unsat_fail_normal = 0
    unsat_fail_wrost = 0
    for cex_line in cex_lines:
        this_unsat_fail = 0
        for clause in inv_lines[1:]: #scan every clause in inv.cnf
            # Test if the s.cex & clauses from inv is unsat
            s_clause = z3.Solver()
            s_cex = z3.Solver()
            s = z3.Solver()
            for lt in clause:
                if int(lt) % 2 == 1:
                    lt_bool = z3.Bool(lt)
                    s_clause.add(lt_bool==False)
                else:
                    lt_bool = z3.Bool(lt)
                    s_clause.add(lt_bool==True)
            for lt in cex_line:
                if int(lt) % 2 == 1:
                    lt_bool = z3.Bool(lt)
                    s_cex.add(lt_bool==False)
                else:
                    lt_bool = z3.Bool(lt)
                    s_cex.add(lt_bool==True)

            clauses_lst = list(s_clause.assertions())
            cex_lst = list(s_cex.assertions())
            s.add(z3.Not(z3.simplify(z3.And(clauses_lst))))
            s.add(z3.simplify(z3.And(cex_lst)))
            if s.check()==z3.unsat:
                unsat_success += 1
            elif s.check()==z3.sat:
                this_unsat_fail += 1
            else:
                AssertionError

        # Check unsat fail
        if this_unsat_fail == len(inv_lines) - 1:
            unsat_fail += 1
        elif this_unsat_fail == len(inv_lines) - 2:
            unsat_fail_wrost += 1
        #elif this_unsat_fail < len(inv_lines)*0.9:
        elif this_unsat_fail < len(inv_lines)*0.85:
            unsat_fail_normal += 1
            

    print("unsat success ratio:", unsat_success/(unsat_success + unsat_fail) * 100, "%")
    print("wrost unsat success ratio:", unsat_fail_wrost/len(cex_lines) * 100, "%")
    print("normal unsat success ratio:", unsat_fail_normal/len(cex_lines) * 100, "%")
    print("unsat success in all combination:", unsat_success/(len(cex_lines)*(len(inv_cnf)-1)) * 100, "%")

# main function
if __name__ == '__main__':
    file_path_prefix = "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/clause-learning/data-collect/hwmcc07-7200-result/output/tip/"
    #file_suffix = "cmu.dme1.B"
    #file_suffix = "eijk.S208o.S"
    #file_suffix = "nusmv.syncarb10^2.B"
    file_suffix = "nusmv.reactor^4.C"
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
    #cex_file_path = f"./{file_suffix}_complete_CTI.txt"
    #cex_file_path = f"./{file_suffix}_inv_CTI_no_comment.txt"
    cex_file_path = f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_20230106_014957_toy/bad_cube_cex2graph/cti_for_inv_map_checking/{file_suffix}/{file_suffix}_inv_CTI.txt"
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

    # Record the sucess rate of finding the inductive clauses in inv.cnf
    subset_success = 0
    subset_fail = 0
    subset_fail_normal = 0
    subset_fail_wrost = 0

    # Record the unsat pass ratio
    unsat_success = 0
    unsat_fail = 0
    unsat_fail_normal = 0
    unsat_fail_wrost = 0
    for cex_line in cex_lines:
        this_subset_fail = 0
        for clause in inv_lines[1:]: #scan every clause in inv.cnf
            # Test if the s is subset of the invariant clauses 
            if(all(x in cex_line for x in clause)):
                #print("clause: ", clause)
                subset_success += 1
            else:
                this_subset_fail += 1
        
        # Check subset fail
        if this_subset_fail == len(inv_lines) -1 :
            subset_fail += 1
        elif this_subset_fail == len(inv_lines) -2:
            subset_fail_wrost += 1
        #elif this_subset_fail < len(inv_lines)*0.9:
        elif this_subset_fail < len(inv_lines)*0.9:
            subset_fail_normal += 1
            
    print("subset success ratio:", subset_success/(subset_success + subset_fail) * 100, "%")
    print("wrost subset success ratio:", subset_fail_wrost/len(cex_lines) * 100, "%")
    print("normal subset success ratio:", subset_fail_normal/len(cex_lines) * 100, "%")
    print("subset success in all combinations:", subset_success/(len(cex_lines)*(len(inv_cnf)-1)) * 100, "%")

    unsat_test(cex_lines, inv_lines)