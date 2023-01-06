from cnfextract import ExtractCnf
from clause import Clauses
from aigmodel import AAGmodel
from aig2graph import AigGraph
import z3
import os
import argparse


'''
Some assumptions (Maybe useful for debugging):
- the inv.cnf has been sorted (the last one is performs like mic's generated result)
- the latch in .aag has been sorted (influence the order of the svars)
- the model extracted from T /\ P /\ not(P_prime) is complete, number of it equals to the number of latch variables
(actually, it depends on the solver and we how to call it, but I assume it is)
- T /\ P /\ not(P_prime) always returns a counterexample (actually, it depends on the model)
- When the algorithm arrives unsat condition, some clauses are still not used to block the counterexample (in extreme case, all clauses are used)
'''

total_num_node_types = 6

def dump4check_map(cex_clause_pair_list_prop, cex_clause_pair_list_ind, aag_name,m, return_after_finished = False):
    # concat the cex_clause_pair_list_prop and cex_clause_pair_list_ind
    cex_clause_pair_list = cex_clause_pair_list_prop + cex_clause_pair_list_ind
    aag_name = aag_name.split('.aag')[0]
    cubeliteral_to_str = lambda cube_literals: ','.join(map
                                (lambda x: str(m.svars[x[0]]).replace('v','') 
                                if str(x[1])=='1' 
                                else str(int(str(m.svars[x[0]]).replace('v',''))+1),cube_literals))
    # open a file for writing
    if not os.path.exists(f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset/bad_cube_cex2graph/cti_for_inv_map_checking/{aag_name.split('/')[-1]}"): os.makedirs(f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset/bad_cube_cex2graph/cti_for_inv_map_checking/{aag_name.split('/')[-1]}")
    with open(f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset/bad_cube_cex2graph/cti_for_inv_map_checking/{aag_name.split('/')[-1]}/{aag_name.split('/')[-1]}_inv_CTI.txt", "w") as text_file:
        for cti in cex_clause_pair_list:
            text_file.write(cubeliteral_to_str(cti[0]) + "\n")
    
    if return_after_finished:
        print('program finished, only dump cti to file')
        return 'Finish all the work!'

def convert_one_aag(aag_name, cnf_name, model_name):
    m = AAGmodel()
    m.from_file(aag_name)
    inv_cnf = Clauses(fname=cnf_name, num_sv = len(m.svars), num_input = len(m.inputs))
    extractor = ExtractCnf(aagmodel = m, clause = inv_cnf, name = model_name)
    cex_clause_pair_list_prop, cex_clause_pair_list_ind, is_inductive, has_fewer_clauses = extractor.get_clause_cex_pair()
    if dump4check_map(cex_clause_pair_list_prop,cex_clause_pair_list_ind,aag_name,m, return_after_finished = True)!= None: return

def test():
    #case = "nusmv.syncarb5^2.B"
    case = "nusmv.reactor^4.C"
    convert_one_aag(f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/case4test/hwmcc_simple/{case}/{case}.aag", f"/data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/output/tip/{case}/inv.cnf", case) 

if __name__ == "__main__":
    # for testing only
    # test()

    # make a argument parser
    parser = argparse.ArgumentParser(description='Convert aig+inv to graph')
    parser.add_argument('--aag', type=str, help='aag file')
    # parse the arguments to test()
    args = parser.parse_args()
    case = args.aag.split('/')[-1].split('.aag')[0]
    convert_one_aag(args.aag, f"/data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/output/tip/{case}/inv.cnf", case)

