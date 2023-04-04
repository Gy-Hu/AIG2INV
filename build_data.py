# this file is for convert aig+inv to graph

'''
----------------- Usage of this tool to generate graph for training and testing -----------------

1. Run in the root project directory
2. The dataset in /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/benchmark_folder/ (e.g. /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/benchmark_folder/hwmcc2007)
3. It should be divided into multiple subsets according to the graph size of counterexample (e.g. .../hwmcc2007/subset_0, .../hwmcc2007/subset_1)


----------------- Generate different graph for training and testing -----------------
1. Small dataset for abc
    1.1. Simplify the transition relation of the model -> use symbolic simplification + z3's simplify function
    1.2. Without deep simplification -> use z3's simplify function only
    1.3  Simplify the last cube + z3 simplify function (Maybe not recommended)
2. Complete dataset for abc 
    2.1. Simplify the transition relation of the model
    2.2. Without deep simplification 
    2.3  Simplify the last cube + z3 simplify function
3. Small dataset for ic3ref -> (skip the graph that could not find the inductive clauses)
    3.1. Simplify the transition relation of the model
    3.2. Without deep simplification
    3.3  Simplify the last cube + z3 simplify function 
4. Complete dataset for ic3ref -> (skip the graph that could not find the inductive clauses)
    4.1. Simplify the transition relation of the model
    4.2  Without deep simplification
    4.3  Simplify the last cube + z3 simplify function 


Note: this is a multi-threading tool, you can change the number of threads in the main function
'''

import os
import argparse
from natsort import natsorted
import subprocess
import multiprocessing
import subprocess
import shlex
from multiprocessing.pool import ThreadPool
from threading import Timer
import shutil
import time
from tqdm import tqdm
import sys
sys.setrecursionlimit(1000000)

def initialization(dir_name, with_re_generate_inv=False):
    
    '''
    with_re-generate_inv: if True, process clean folder function
    
    the folder structure:
    ./dataset
    ├── bad_cube_cex2graph
    │   ├── cti_for_inv_map_checking -> should be empty
    │   ├── expr_to_build_graph -> should be empty
    │   ├── ground_truth_table -> should be empty
    │   └── json_to_graph_pickle -> should be empty
    └── re-generate_inv
        ├── cmu.dme1.B
        ├── cmu.dme2.B
        ├── .......
    '''
    mkdir_cmd = lambda dir_name: os.system(f"mkdir {dir_name}/bad_cube_cex2graph && mkdir {dir_name}/bad_cube_cex2graph/expr_to_build_graph {dir_name}/bad_cube_cex2graph/cti_for_inv_map_checking {dir_name}/bad_cube_cex2graph/ground_truth_table {dir_name}/bad_cube_cex2graph/json_to_graph_pickle")
    if with_re_generate_inv==False:
        # new_dir_name = f"{dir_name}_" + time.strftime("%Y%m%d_%H%M%S", time.localtime())
        # if old directory exists and file size is not zero, then move it to new directory
        if os.path.exists(dir_name):
            assert False, "The folder already exists!"
            '''
            if os.listdir(f"{dir_name}/bad_cube_cex2graph/json_to_graph_pickle")!= [] and os.path.getsize(f"{dir_name}/bad_cube_cex2graph/json_to_graph_pickle") >= 4096: # exist and not empty, change the name
                shutil.move(dir_name, new_dir_name)
            else: # exist but empty, delete it
                # use trash command to delete the empty directory
                os.system(f"trash {dir_name}")
            '''
        # old directory has been handled, then create new directory
        os.mkdir(dir_name)
        # make file folder under dir_name
        mkdir_cmd(dir_name)
        print("Finish initialization!")
    else:
        # clean all the files in the folder bad_cube_cex2graph
        if os.path.exists(f"{dir_name}/bad_cube_cex2graph"): os.system(f"trash {dir_name}/bad_cube_cex2graph")
        # make file folder under dir_name
        mkdir_cmd(dir_name)
        print("Finish cleaning the folder bad_cube_cex2graph!")

def call_proc(cmd):
    """ This runs in a separate thread. Serializes the command and run"""
    #subprocess.call(shlex.split(cmd))  # This will block until cmd finishes
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = p.communicate()
    return (_, err)

def run_cmd(cmd):
    " This directly runs the command "
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = p.communicate()
    return (_, err)

def run_cmd_with_timer(cmd):
    #TODO: This function does not work, need to fix it
    timeout = 7200 #2 hour for model checking
    " This directly runs the command "
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timer = Timer(timeout, lambda x: x.terminate(), [p])
    timer.start()
    _, err = p.communicate()
    code = p.returncode
    if code == -15: #return code of SIGTERM
        print('{cmd} over {timeout}, timeout!'.format(timeout=timeout, cmd=cmd))
    timer.cancel()
    return (_, err)

def walkFile(dir):
    files = []
    for root, _, files in os.walk(dir):
        files = natsorted(files)
        #files = [os.path.join(root,f) for f in files if ".aag" not in f]
    assert len(files) > 0, "No files in the folder!"
    # filtered the files, only keep the .aag file
    files = [file for file in files if file.endswith(".aag")]
    return files

def generate_subset_dir_lst(subset_range, subset_dir):
    if '-' not in subset_range:
        return [subset_dir + subset_range]
    start, end = map(int, subset_range.split('-'))
    return [subset_dir + str(i) for i in range(start, end + 1)]

def find_case_in_selected_dataset_with_inv(model_checker='ic3ref'):
    #generate smt2 file for prediction -> SAT model/conterexample
    print("Start to find the cases with inductive invariants!")
    subset_dir = f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/benchmark_folder/{BENCHMARK}/subset_"
    # if SUBSET_RANGE is a integer in string format, then convert it to integer, otherwise, it is a list
    subset_dir_lst =  generate_subset_dir_lst(SUBSET_RANGE, subset_dir) # 10 is the number for test subset
    
    # get all the generated inductive invariants cases' name
    # store all folder name in '/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/clause-learning/data-collect/hwmcc07-7200-result/output/tip'
    
    #XXX: Double check before running the script
    cases_with_inductive_invariants = os.listdir(f"{GROUND_TRUTH_FOLDER_PREFIX}")
    # check whether it contains inv.cnf in subfolder
    cases_with_inductive_invariants = [
    case
    for case in cases_with_inductive_invariants
    if os.path.exists(f"{GROUND_TRUTH_FOLDER_PREFIX}/{case}/inv.cnf")
    ]

    # initialize the list to store all the abnormal cases
    AigCaseBlackList = [
        #-------- benchmark 2007 stuck in generate smt2-----------
        # 'eijk.bs3271.S',
        # #--------- benchmark 2020 stuck in generate smt2------------
        # 'vcegar_QF_BV_ar',
        # 'rast-p00',
        # 'vis_arrays_bufferAlloc',
        # 'zipversa_composecrc_prf-p10',
        # 'qspiflash_dualflexpress_divthree-p010',
        # 'vcegar_QF_BV_itc99_b13_p10',
        # 'zipcpu-busdelay-p00',
        # 'zipcpu-busdelay-p30',
        # 'qspiflash_qflexpress_divfive-p137',
        # 'qspiflash_dualflexpress_divthree-p046',
        # 'qspiflash_dualflexpress_divfive-p007',
        # 'qspiflash_dualflexpress_divfive-p154',
        # 'qspiflash_dualflexpress_divfive-p009',
        # 'qspiflash_dualflexpress_divfive-p116',
        # 'qspiflash_dualflexpress_divthree-p111',
        # 'elevator.4.prop1-func-interl',
        # 'h_RCU',
        # 'vgasim_imgfifo-p105',
        # 'cal87',
        # 'cal90',
        # 'picorv32-check-p05',
        # 'cal142',
        # 'picorv32-check-p20',
        # 'cal118',
        # 'cal97',
        # 'cal143',
        # 'cal102',
        # 'cal107',
        # 'cal112',
        # 'dspfilters_fastfir_second-p09',
        # 'dspfilters_fastfir_second-p26',
        # 'dspfilters_fastfir_second-p11',
        # 'cal161',
        # 'rushhour.4.prop1-func-interl',
        # 'cal176',
        # #--------- benchmark 2020 stuck in model2graph------------
        # 'dspfilters_fastfir_second-p25',
        # 'dspfilters_fastfir_second-p05',
        # 'dspfilters_fastfir_second-p43',
        # 'dspfilters_fastfir_second-p14',
        # 'dspfilters_fastfir_second-p07',
        # 'dspfilters_fastfir_second-p16',
        # 'dspfilters_fastfir_second-p21',
        # #--------- benchmark 2020 stuck in json2networkx------------
        # 'dspfilters_fastfir_second-p10',
        # 'dspfilters_fastfir_second-p45',
        # 'dspfilters_fastfir_second-p42',
        # 'dspfilters_fastfir_second-p04',
        # 'pgm_protocol.7.prop1-back-serstep',
        # 'dspfilters_fastfir_second-p2'
    ]

    all_cases_name_lst = [] # put into multi-threading pool
    for subset in subset_dir_lst:
        # get file name in each subset
        _ = walkFile(subset)
        # filter the case name list, only keep the case name in cases_with_inductive_invariants
        if _ := [ # if case_name_lst is not empty
            case
            for case in _
            if (case.split('.aag')[0] in cases_with_inductive_invariants and case.split('.aag')[0] not in AigCaseBlackList)
        ]: 
            all_cases_name_lst.extend(f"{subset}/{case}" for case in _)

    return all_cases_name_lst

def update_progress_bar(pbar):
    pbar.update(1)

def generate_smt2(run_mode='normal', model_checker='ic3ref'):
    '''
    If the `run_mode` set to 'debug', collect.py will exit after checking the inv.cnf without smt2 generation.
    Then, this `main.py` will also exit after printing the bad_inv.log
    '''
    pool = ThreadPool(multiprocessing.cpu_count())
    #pool = multiprocessing.Pool(64)
    '''
    First, go to the select dataset, check whether the case has inductive invariants generated in advance,
    if yes, then generate smt2 file for the case, otherwise, skip it.
    '''
    all_cases_name_lst = find_case_in_selected_dataset_with_inv(model_checker)

    results = []
    print(f"Start to generate smt2 for {len(all_cases_name_lst)} aiger in all the subset!")
    with tqdm(total=len(all_cases_name_lst)) as pbar:
        for _, aig_to_generate_smt2 in enumerate(all_cases_name_lst):
            print(f"Start to generate smt2 for No.{_} aiger: {(aig_to_generate_smt2.split('/')[-1]).split('.aag')[0]}")
            results.append(pool.apply_async(
                call_proc,
                (
                    f"python /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/data2dataset/cex2smt2/collect.py --aag {aig_to_generate_smt2} --run-mode {run_mode} --model-checker {model_checker} {SIMPLIFICATION_LABEL} --ground-truth-folder-prefix {GROUND_TRUTH_FOLDER_PREFIX} --dump-folder-prefix {DATASET_FOLDER_PREFIX}",
                ),
                callback=lambda _: update_progress_bar(pbar)
            ))
        pool.close()
        pool.join()
        
    for result in results:
        _, err = result.get()
        # if err is not None, print it 
        if err != b'':
            print(f"err: {err}")
        else:
            print("Congruatulations, no error in subprocess!")
    print("Finish all the subprocess, all the subset has generated smt2.")

    logs = ["abnormal_header.log", "mismatched_inv.log", "bad_inv.log"]
    for log_name in logs:
        log_path = f"log/error_handle/{log_name}"
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                content = f.read()
            print(f"{log_name.capitalize().replace('_', ' ')} information: \n{content}")
            shutil.move(log_path, f"{log_path}.{time.time()}")
    
    if run_mode == 'debug': 
        "Exiting the program after checking the inv.cnf without smt2 generation!"
        exit(0)
    

def generate_smt2_error_handle(log_file=None, only_re_generate_inv=False, ic3ref_basic_generalization=""):
    # parse the log file, find the cases that has mismatched inductive invariants
    # read lines from log file
    with open(log_file, "r") as f:
        lines = f.readlines()
    # get the case name that has mismatched inductive invariants in each line in lines
    cases_with_mismatched_inv = [
        line.split(" ")[1]
        for line in lines
    ]
    
    # make a copy that record this in cases4re_generate_inv
    cases4re_generate_inv = cases_with_mismatched_inv[:]

    print("Begin to re-generate the inv.cnf for the cases that has mismatched inductive invariants!")
    #mkdir for the cases that has mismatched inductive invariants
    
    cases_with_mismatched_inv = [case for case in cases_with_mismatched_inv  \
        if not os.path.exists(f"{DATASET_FOLDER_PREFIX}/re-generate_inv/{case.split('/')[-1].split('.aag')[0]}/inv.cnf")]
    
    if cases_with_mismatched_inv != []:
        # create the folder for the cases that has mismatched inductive invariants (and have not been fixed yet)
        for case in cases_with_mismatched_inv:
            if not os.path.exists(
                f"{DATASET_FOLDER_PREFIX}/re-generate_inv/{case.split('/')[-1].split('.aag')[0]}"
            ):    
                # if the inv.cnf and the folder does not exist, then we create the folder
                os.mkdir(f"{DATASET_FOLDER_PREFIX}/re-generate_inv/{case.split('/')[-1].split('.aag')[0]}")
        # call IC3 to re-generate the inv.cnf for the cases that has mismatched inductive invariants
        pool = ThreadPool(multiprocessing.cpu_count())
        results = []
        results.extend(
            pool.apply_async(
                run_cmd,
                (
                    f"cd {DATASET_FOLDER_PREFIX}/re-generate_inv/{case.split('/')[-1].split('.aag')[0]} && /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/utils/IC3ref/IC3 -d {ic3ref_basic_generalization} < {case}",
                ),
            )
            for case in cases_with_mismatched_inv
        )
        pool.close()
        pool.join()
        for result in results:
            _, err = result.get()
            # if err is not None, print it 
            if err != b'':
                print(f"err: {err}")
            else:
                print("Congruatulations, no error in subprocess!")
        print("Finish all the subprocess, all the abnormal cases have been fixed.")
    
    
    if only_re_generate_inv==False:
        pool = ThreadPool(multiprocessing.cpu_count())
        # begin to generate the smt2 file for the cases that has mismatched inductive invariants
        results = []
        assert cases4re_generate_inv != [], "BUG: cases4re_generate_inv should not be empty! Check the copy operation!"
        print(f"Start to generate smt2 for {len(cases4re_generate_inv)} aiger with mismated inv!")
        for _, aig_to_generate_smt2 in enumerate(cases4re_generate_inv):
            print(f"Start to re-generate smt2 for No.{_} aiger due to mismatched inv.cnf: {(aig_to_generate_smt2.split('/')[-1]).split('.aag')[0]}")
            results.append(pool.apply_async(
                call_proc,
                (
                    f"python /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/data2dataset/cex2smt2/collect.py --aag {aig_to_generate_smt2} --cnf {DATASET_FOLDER_PREFIX}/re-generate_inv/{aig_to_generate_smt2.split('/')[-1].split('.aag')[0]}/inv.cnf",
                ),
            ))
        pool.close()
        pool.join()
        for result in results:
            _, err = result.get()
            # if err is not None, print it 
            if err != b'':
                print(f"err: {err}")
            else:
                print("Congruatulations, no error in subprocess!")
        print("Finish all the subprocess, all the mismatched error cases have re-generated smt2.")
        
    # rename the mismatched_inv.log to mismatched_inv.log.{time_stamp}
    shutil.move("/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/log/error_handle/mismatched_inv.log", f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/log/error_handle/mismatched_inv.log.{time.time()}")
    
    
def generate_pre_graph(simplification):
    # generate pre-graph, constructed as json
    smt2_dir = f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/{DATASET_FOLDER_PREFIX}/bad_cube_cex2graph/expr_to_build_graph/"
    # get all the subfolder name under smt2_dir
    data4json_conversion = os.listdir(smt2_dir)
    simplification = "true" if simplification in ["thorough", "deep", "moderate", "slight", "naive"] else "false"

    pool = ThreadPool(multiprocessing.cpu_count())
    results = []
    with tqdm(total=len(data4json_conversion)) as pbar:
        results.extend(
            pool.apply_async(
                call_proc,
                (
                    # XXX: Double check before running the script -> did you recompile it and update it to the latest version?
                    f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/data2dataset/smt2_cex2graph/model2graph {case_name} {smt2_dir} {simplification}",
                ),
                callback=lambda _: update_progress_bar(pbar),
            )
            for case_name in data4json_conversion
        )
        pool.close()
        pool.join()
    for result in results:
        _, err = result.get()
        # if err is not None, print it 
        if err != b'':
            print(f"err: {err}")
        else:
            print("Congruatulations, no error in subprocess!")
    print("Finish all the subprocess, all the subset has generated json file to present DAG graph.")

def generate_post_graph():
    # generate post-graph, serialization as pickle
    json_dir = f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/{DATASET_FOLDER_PREFIX}/bad_cube_cex2graph/expr_to_build_graph"
    ground_truth_dir = f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/{DATASET_FOLDER_PREFIX}/bad_cube_cex2graph/ground_truth_table"
    pickle_file_name_prefix = f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/{DATASET_FOLDER_PREFIX}/bad_cube_cex2graph/json_to_graph_pickle/"
    # get all the subfolder name under expr_to_build_graph (expression to build graph)
    data4pickle_conversion = os.listdir(json_dir)

    pool = ThreadPool(2)
    results = []
    with tqdm(total=len(data4pickle_conversion)) as pbar:
        results.extend(
            pool.apply_async(
                call_proc,
                (
                    f"python /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/data2dataset/smt2_cex2graph/json2networkx.py \
                    --json_file_path   {json_dir}/{case_name} \
                    --ground_truth_path  {ground_truth_dir}/{case_name} \
                    --pickle_file_name_prefix {pickle_file_name_prefix}",
                ),
                callback=lambda _: update_progress_bar(pbar),
            )
            for case_name in data4pickle_conversion
        )
        pool.close()
        pool.join()

    for result in results:
        _, err = result.get()
        # if err is not None, print it 
        if err != b'':
            print(f"err: {err}")
        else:
            print("Congruatulations, no error in subprocess!")
    print("Finish all the subprocess, all the subset has generated serialization pickle for training.")

    
if __name__ == '__main__':
    
    '''
    ---------------------------------------------------------
    Attention:
    If you are the first time to run this script, do not need to set any option.
    ---------------------------------------------------------
    '''
    
    parser = argparse.ArgumentParser()
    # this option only for that you have .log with mismatched cases list
    parser.add_argument('--only_re_generate_inv', action='store_true', help='only re-generate the inv.cnf for the cases that has mismatched inductive invariants')
    parser.add_argument('--initialization_with_inv_generated', action='store_true', help='initialization with inv generated')
    parser.add_argument('--error_handle_with_ic3ref_basic_generalization', action='store_true', help='error handle with ic3ref basic generalization')
    parser.add_argument('--run-mode', type=str, default="normal", help='run mode, normal or debug. Debug is for testing invariants correctness only')
    parser.add_argument('--model-checker', type=str, default="ic3ref", help='model checker, ic3ref or abc')
    #parser.add_argument('--dataset-folder-prefix', type=str, default="dataset", help='dataset folder prefix, the final aim generated folder, in the root folder')
    parser.add_argument('--simplification-label', type=str, default="no_simplification", help='simplification label')
    parser.add_argument('--benchmark', type=str, default=None, help='benchmark, which is used to generate the training dataset, in benchmark_folder')
    parser.add_argument('--ground_truth_folder_prefix', type=str, default=None, help='ground truth folder prefix, the final aim generated folder, in the root folder')
    parser.add_argument('--subset_range', type=str, default=0, help='subset range, the number of subset')

    
    args = parser.parse_args()
    '''
    # remember to comment out Black List if you do not need it
    args = parser.parse_args(['--model-checker', 'abc', \
        #'--simplification-label', 'naive', \
        '--benchmark', 'hwmcc2007_tip', \
        '--ground_truth_folder_prefix', '/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/clause-learning/data-collect/hwmcc07-7200-abc-result/output/tip/',\
        '--subset_range', '0-22'
        ])
    '''
    
    
    # Global variable assignment
    global DATASET_FOLDER_PREFIX
    global SIMPLIFICATION_LABEL
    global BENCHMARK
    global GROUND_TRUTH_FOLDER_PREFIX
    global SUBSET_RANGE
    DATASET_FOLDER_PREFIX = f"dataset_{args.benchmark}_{args.model_checker}_{args.simplification_label}_{args.subset_range}"
    SIMPLIFICATION_LABEL = "--thorough-simplification T" if args.simplification_label == "thorough" \
        else "--deep-simplification T" if args.simplification_label == "deep" \
        else "--moderate-simplification T" if args.simplification_label == "moderate"\
        else "--slight-simplification T" if args.simplification_label == "slight"\
        else "--naive-simplification T" if args.simplification_label == "naive"\
        else ""
    #assert SIMPLIFICATION_LABEL != "", "simplification label is not correct"
    BENCHMARK = args.benchmark
    GROUND_TRUTH_FOLDER_PREFIX = args.ground_truth_folder_prefix
    SUBSET_RANGE = args.subset_range
    
    # for testing only
    # args = parser.parse_args(['--only_re_generate_inv','--error_handle_with_ic3ref_basic_generalization'])
    '''
    ---------------------------------------------------------
    (step 0: )only re-generate the inv.cnf for the cases that has mismatched inductive invariants?
    
    (only has error log, and user want to generate the inv.cnf only)
    ---------------------------------------------------------
    '''
    # if args.only_re_generate_inv:
    #     args.error_handle_with_ic3ref_basic_generalization = "-b" if args.error_handle_with_ic3ref_basic_generalization else ""
    #     assert not os.path.exists("/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset"), "dataset/re-generate_inv/ folder already exists, please remove it first"
    #     os.mkdir("/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset/"); os.mkdir("/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset/re-generate_inv")
    #     generate_smt2_error_handle("/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/log/error_handle/mismatched_inv.log", only_re_generate_inv=True, ic3ref_basic_generalization=args.error_handle_with_ic3ref_basic_generalization)
    #     exit(0)
    
    '''
    ---------------------------------------------------------
    (step 1: )change the directory name of "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset" with time stamp
    
    Choose mode:
    1. Generate_smt2 for all cases -> default
    2. Generate_smt2 for error cases -> with_re_generate_inv set to True in initialization()
    
    Then, generate smt2 file for prediction (-> inducitve invariant)
    ---------------------------------------------------------
    '''
    dir_name = f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/{DATASET_FOLDER_PREFIX}"
    if args.initialization_with_inv_generated:
        initialization(dir_name, with_re_generate_inv=True)
        generate_smt2_error_handle("/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/log/error_handle/mismatched_inv.log")
    else:
        initialization(dir_name, with_re_generate_inv=False)
        # script folder: /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/data2dataset/cex2smt2/collect.py
        generate_smt2(args.run_mode,args.model_checker) # if mode is debug, the program will exit after inv checking
    
    
    '''
    ---------------------------------------------------------
    (step2: )generate graph file for training
    ---------------------------------------------------------
    '''
    # generate pre-graph (json format)
    # script folder: /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/data2dataset/smt2_cex2graph/model2graph
    generate_pre_graph(args.simplification_label)

    # generate post-graph (pickle format)
    # script folder: /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/data2dataset/smt2_cex2graph/json2networkx.py
    # generate_post_graph()

    print("Finish building data! Ready to train!")


