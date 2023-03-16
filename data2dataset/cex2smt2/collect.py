from cnfextract import ExtractCnf
from clause import Clauses
from aigmodel import AAGmodel
from aig2graph import AigGraph
import z3
import os
import argparse
import sys
import copy

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
    if not os.path.exists(f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/{DUMP_FOLDER_PREFIX}/bad_cube_cex2graph/cti_for_inv_map_checking/{aag_name.split('/')[-1]}"): os.makedirs(f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/{DUMP_FOLDER_PREFIX}/bad_cube_cex2graph/cti_for_inv_map_checking/{aag_name.split('/')[-1]}")
    with open(f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/{DUMP_FOLDER_PREFIX}/bad_cube_cex2graph/cti_for_inv_map_checking/{aag_name.split('/')[-1]}/{aag_name.split('/')[-1]}_inv_CTI.txt", "w") as text_file:
        for cti in cex_clause_pair_list:
            text_file.write(cubeliteral_to_str(cti[0]) + "\n")
    
    if return_after_finished:
        print('program finished, only dump cti to file')
        return 'Finish all the work!'

def convert_one_aag(aag_name, cnf_name, model_name, generalize_predecessor, generate_smt2, inv_correctness_check, run_mode, model_checker):
    file_path = aag_name
    m = AAGmodel(SIMPLIFICATION_LEVEL)
    m.from_file(fname=aag_name)
    inv_cnf = Clauses(fname=cnf_name, num_sv = len(m.svars), num_input = len(m.inputs))
    extractor = ExtractCnf(\
        aagmodel = m,\
        clause = inv_cnf,\
        name = model_name,\
        generalize = generalize_predecessor,\
        aig_path=file_path,\
        generate_smt2 = generate_smt2,\
        inv_correctness_check = inv_correctness_check,\
        model_checker = model_checker,\
        simplification_level=SIMPLIFICATION_LEVEL,\
        dump_folder_prefix=DUMP_FOLDER_PREFIX)
    #XXX: Double check before running the script
    if SIMPLIFICATION_LEVEL in ["deep"]: check_extractor_eq(aag_name, cnf_name, model_name, generalize_predecessor, generate_smt2, inv_correctness_check, run_mode, model_checker, copy.deepcopy(extractor))
    if run_mode == 'debug': sys.exit()
    cex_clause_pair_list_prop, cex_clause_pair_list_ind, is_inductive, has_fewer_clauses = extractor.get_clause_cex_pair()
    if dump4check_map(cex_clause_pair_list_prop,cex_clause_pair_list_ind,aag_name,m, return_after_finished = True)!= None: return

def check_extractor_eq(aag_name, cnf_name, model_name, generalize_predecessor, generate_smt2, inv_correctness_check, run_mode, model_checker, extractor):
    file_path = aag_name
    #extractor_old = copy.deepcopy(extractor)
    m_without_simplification = AAGmodel(None)
    m_without_simplification.from_file(fname=aag_name)
    inv_cnf = Clauses(fname=cnf_name, num_sv = len(m_without_simplification.svars), num_input = len(m_without_simplification.inputs))
    extractor_without_simplification = ExtractCnf(\
        aagmodel = m_without_simplification,\
        clause = inv_cnf,\
        name = model_name,\
        generalize = generalize_predecessor,\
        aig_path=file_path,\
        generate_smt2 = generate_smt2,\
        inv_correctness_check = inv_correctness_check,\
        model_checker = model_checker,\
        simplification_level=None,\
        dump_folder_prefix=DUMP_FOLDER_PREFIX)
    
    for a,b in zip(extractor.vprime2nxt, extractor_without_simplification.vprime2nxt):
        # if z3.is_true(a[1]) or z3.is_false(a[1]) or z3.is_true(b[1]) or z3.is_false(b[1]), skip this loop
        if z3.is_true(a[1]) or z3.is_false(a[1]) or z3.is_true(b[1]) or z3.is_false(b[1]):
            assert str(a[1])==str(b[1]), "The transition relation is not simplified correctly!"
            continue
        s = z3.Solver()
        proposition = a[1] == b[1] # assertion is whether b1 and b2 are equal
        s.add(z3.Not(proposition))
        # proposition proved if negation of proposition is unsat
        if s.check() == z3.sat:
            # it should be unsat if the transition relation is simplified correctly -> which is unsat
            print("The transition relation is not simplified correctly!")
            assert False, "The transition relation is not simplified correctly!"
    

def test():
    #case = "nusmv.syncarb5^2.B"
    case = "nusmv.reactor^4.C"
    convert_one_aag(f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/case4test/hwmcc_simple/{case}/{case}.aag", f"/data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/output/tip/{case}/inv.cnf", case) 

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    # for testing only
    # test()

    # make a argument parser
    parser = argparse.ArgumentParser(description='Convert aig+inv to graph')
    parser.add_argument('--aag', type=str, help='aag file')
    parser.add_argument('--generalize', type=str2bool, default=True, help='generalize the predesessor')
    parser.add_argument('--cnf', type=str, default=None, help='cnf file')
    parser.add_argument('--generate_smt2', type=str2bool, default=True, help='generate smt2 file')
    parser.add_argument('--inv-correctness-check', type=str2bool, default=True, help='check the correctness of the invariant')
    parser.add_argument('--run-mode', type=str, default='debug', help='normal or debug. Debug model will exit after inv correctness check')
    parser.add_argument('--model-checker', type=str, default='ic3ref', help='ic3ref or abc')
    parser.add_argument('--thorough-simplification', type=str2bool, default=False, help='use sympy in tr simplification + aig operator simplification during tr construction + z3 simplification + counterexample cube simplification')
    parser.add_argument('--deep-simplification', type=str2bool, default=False, help='use sympy in tr simplification + aig operator simplification during tr construction + z3 simplification')
    parser.add_argument('--moderate-simplification', type=str2bool, default=False, help='aig operator simplification during tr construction + z3 simplification')
    parser.add_argument('--slight-simplification', type=str2bool, default=False, help='z3 simplification + ternary simulation')
    parser.add_argument('--naive-simplification', type=str2bool, default=False, help='only use sympy to simplify the counterexample cube')
    parser.add_argument('--ground-truth-folder-prefix', type=str, default='/data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/output/tip/', help='the prefix of the ground truth folder')
    parser.add_argument('--dump-folder-prefix', type=str, default='/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset', help='the prefix of the dump folder')
    
    # parse the arguments to test()
    args = parser.parse_args()
    
    
    
    '''
    #XXX: Double check before running the script
    for testing only
    
    args = parser.parse_args(['--aag',
        '/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/case4test/hwmcc2020_all/subset_0/simple_alu.aag',
        #'/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/case4test/hwmcc2007/subset_4/nusmv.brp.B.aag',
        #'/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/case4test/hwmcc2007/subset_0/nusmv.syncarb5^2.B.aag',
        #'/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/case4test/hwmcc2007/subset_0/eijk.S208c.S.aag',
        #'/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/case4test/hwmcc2007/subset_2/eijk.S386.S.aag', # encountered a bug in addModel(generalize predecessor)
        #'/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/case4test/hwmcc2007/subset_17/vis.prodcell^03.E.aag',
        #'/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/case4test/hwmcc2007/subset_7/eijk.S953.S.aag',
        #'/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/case4test/hwmcc2007/subset_3/vis.arbiter.E.aag',
        #'/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/case4test/hwmcc2007/subset_20/texas.PI_main^01.E.aag',
        #'/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/case4test/hwmcc2007/subset_21/texas.PI_main^05.E.aag',
        #'/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/case4test/hwmcc2007/subset_15/eijk.S5378.S.aag',
        #'/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/case4test/hwmcc2007/subset_15/eijk.bs3330.S.aag',
        #'/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/case4test/hwmcc2007/subset_15/eijk.bs3271.S.aag', # Solve sat solving when verify the invariants correctness
        #'/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/case4test/hwmcc2007/subset_6/eijk.S838.S.aag', # Solving time of PDR is very long
        #"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/case4test/hwmcc2007/subset_17/vis.prodcell^03.E.aag", # cannot pass inv correctness check by any methods
        #'/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/case4test/hwmcc2007/subset_9/nusmv.reactor^5.C.aag', # z3 convert to dimacs has problem
        '--generalize', 'T',
        #'--cnf',
        #'/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset/re-generate_inv/nusmv.brp.B/inv.cnf',
        #'/data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/output/tip/nusmv.syncarb5^2.B/inv.cnf',
        '--generate_smt2', 
        'T', #XXX: Double check before running scripts
        '--run-mode',
        'normal',
        '--model-checker', 
        'abc', #XXX: Double check before running scripts
        '--deep-simplification', 'T',
        #'--moderate-simplification', 'T',
        '--ground-truth-folder-prefix', '/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/hwmcc20_abc_7200_result'
        #'--deep-simplification',
        #'T' #XXX: Double check before running scripts -> want to use sympy rather than ternary simulation?
        ])
    '''
    
    # set global variables
    global SIMPLIFICATION_LEVEL
    global DUMP_FOLDER_PREFIX
    SIMPLIFICATION_LEVEL = "naive" if args.naive_simplification else "slight" if args.slight_simplification else "moderate" if args.moderate_simplification else "deep" if args.deep_simplification else "thorough" if args.thorough_simplification else "none"
    #assert SIMPLIFICATION_LEVEL != "none", "Please specify the simplification level"
    DUMP_FOLDER_PREFIX = args.dump_folder_prefix
    
    
    case = args.aag.split('/')[-1].split('.aag')[0]
    
    if args.cnf is None: 
        # check which model checker is used, and use the corresponding output folder
        if args.model_checker == 'ic3ref':
            convert_one_aag(args.aag, f"{args.ground_truth_folder_prefix}/{case}/inv.cnf", case, args.generalize, args.generate_smt2, args.inv_correctness_check, args.run_mode, args.model_checker)
            #convert_one_aag(args.aag, f"/data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/output-wrong-result/tip/{case}/inv.cnf", case, args.generalize, args.generate_smt2, args.inv_correctness_check, args.run_mode)
        elif args.model_checker == 'abc':
            convert_one_aag(args.aag, f"{args.ground_truth_folder_prefix}/{case}/inv.cnf", case, args.generalize, args.generate_smt2, args.inv_correctness_check, args.run_mode, args.model_checker)
    else:
        convert_one_aag(args.aag, args.cnf, case, args.generalize, args.generate_smt2, args.inv_correctness_check, args.run_mode)

