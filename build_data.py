# this file is for convert aig+inv to graph

'''
----------------- Usage of this tool to generate graph for training and testing -----------------

1. Run in the root project directory
2. The dataset in /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/case4test/ (e.g. /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/case4test/hwmcc2007)
3. It should be divided into multiple subsets according to the graph size of counterexample (e.g. .../hwmcc2007/subset_0, .../hwmcc2007/subset_1)

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

def initialization(old_dir_name):
    import time
    import shutil
    new_dir_name = f"{old_dir_name}_" + time.strftime(
        "%Y%m%d_%H%M%S", time.localtime()
    )
    # if old directory exists and file size is not zero, then move it to new directory
    if os.path.exists(old_dir_name):
        if os.listdir(f"{old_dir_name}/bad_cube_cex2graph/json_to_graph_pickle")!= [] and os.path.getsize(f"{old_dir_name}/bad_cube_cex2graph/json_to_graph_pickle") >= 4096: # exist and not empty, change the name
            shutil.move(old_dir_name, new_dir_name)
        else: # exist but empty, delete it
            # use trash command to delete the empty directory
            os.system(f"trash {old_dir_name}")
    # old directory has been handled, then create new directory
    os.mkdir(old_dir_name)
    # make file folder under old_dir_name
    os.mkdir(f"{old_dir_name}/bad_cube_cex2graph")
    os.mkdir(f"{old_dir_name}/bad_cube_cex2graph/expr_to_build_graph")
    os.mkdir(f"{old_dir_name}/bad_cube_cex2graph/cti_for_inv_map_checking")
    os.mkdir(f"{old_dir_name}/bad_cube_cex2graph/ground_truth_table")
    os.mkdir(f"{old_dir_name}/bad_cube_cex2graph/json_to_graph_pickle")
    print("Finish initialization!")

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

def find_case_in_selected_dataset_with_inv():
    #generate smt2 file for prediction -> SAT model/conterexample
    subset_dir = '/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/case4test/hwmcc2007/subset_'
    subset_dir_lst = [subset_dir+str(i) for i in range(23)] # 10 is the number for test subset
    
    # get all the generated inductive invariants cases' name
    # store all folder name in '/data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/output/tip'
    cases_with_inductive_invariants = os.listdir('/data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/output/tip')
    # check whether it contains inv.cnf in subfolder
    cases_with_inductive_invariants = [
        case
        for case in cases_with_inductive_invariants
        if os.path.exists(f'/data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/output/tip/{case}/inv.cnf')
    ]

    all_cases_name_lst = [] # put into multi-threading pool
    for subset in subset_dir_lst:
        # get file name in each subset
        _ = walkFile(subset)
        # filter the case name list, only keep the case name in cases_with_inductive_invariants
        if _ := [ # if case_name_lst is not empty
            case
            for case in _
            if case.split('.aag')[0] in cases_with_inductive_invariants
        ]: 
            all_cases_name_lst.extend(f'{subset}/{case}' for case in _)

    return all_cases_name_lst
    

def generate_smt2():
    #pool = ThreadPool(multiprocessing.cpu_count())
    pool = multiprocessing.Pool(64)
    '''
    First, go to the select dataset, check whether the case has inductive invariants generated in advance,
    if yes, then generate smt2 file for the case, otherwise, skip it.
    '''
    all_cases_name_lst = find_case_in_selected_dataset_with_inv()

    results = []
    print(f"Start to generate smt2 for {len(all_cases_name_lst)} aiger in all the subset!")
    for _, aig_to_generate_smt2 in enumerate(all_cases_name_lst):
        print(f"Start to generate smt2 for No.{_} aiger: {(aig_to_generate_smt2.split('/')[-1]).split('.aag')[0]}")
        results.append(pool.apply_async(
            call_proc,
            (
                f"python /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/data2dataset/cex2smt2/collect.py --aag {aig_to_generate_smt2}",
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
    print("Finish all the subprocess, all the subset has generated smt2.")

    #print error information in log/error_handle/abnormal_header.log if file exist
    if os.path.exists("/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/log/error_handle/abnormal_header.log"):
        print("However, there are some errors occurred, please check them!")
        with open("/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/log/error_handle/abnormal_header.log", "r") as f:
            error_info = f.read()
        print(f"Error information: {error_info}")

    #print mismatched inv.cnf in log/error_handle/mismatched_inv.log
    if os.path.exists("/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/log/error_handle/mismatched_inv.log"):
        print("There are some mismatched inv.cnf, please check them!")
        with open("/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/log/error_handle/mismatched_inv.log", "r") as f:
            mismatched_inv_info = f.read()
        print(f"Mismatched inv.cnf information: {mismatched_inv_info}") 
        generate_smt2_error_handle(all_cases_name_lst,"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/log/error_handle/mismatched_inv.log")

def generate_smt2_error_handle(log_file=None):
    # parse the log file, find the cases that has mismatched inductive invariants
    # read lines from log file
    with open(log_file, "r") as f:
        lines = f.readlines()
    # get the case name that has mismatched inductive invariants in each line in lines
    cases_with_mismatched_inv = [
        line.split(" ")[1]
        for line in lines
    ]

    print("Begin to re-generate the inv.cnf for the cases that has mismatched inductive invariants!")
    #mkdir for the cases that has mismatched inductive invariants
    
    cases_with_mismatched_inv = [case for case in cases_with_mismatched_inv  \
        if not os.path.exists(f"dataset/re-generate_inv/{case.split('/')[-1].split('.aag')[0]}/inv.cnf")]
    
    if cases_with_mismatched_inv != []:
        # create the folder for the cases that has mismatched inductive invariants (and have not been fixed yet)
        for case in cases_with_mismatched_inv:
            if not os.path.exists(
                f"dataset/re-generate_inv/{case.split('/')[-1].split('.aag')[0]}"
            ):    
                # if the inv.cnf and the folder does not exist, then we create the folder
                os.mkdir(f"dataset/re-generate_inv/{case.split('/')[-1].split('.aag')[0]}")
    # call IC3 to re-generate the inv.cnf for the cases that has mismatched inductive invariants
    pool = ThreadPool(multiprocessing.cpu_count())
    results = []
    results.extend(
        pool.apply_async(
            run_cmd_with_timer,
            (
                f"cd dataset/re-generate_inv/{case.split('/')[-1].split('.aag')[0]} && /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/utils/IC3ref/IC3 -d < {case}",
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
    # remove the log file
    os.remove(log_file)

def generate_pre_graph():
    # generate pre-graph, constructed as json
    smt2_dir = "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset/bad_cube_cex2graph/expr_to_build_graph/"
    # get all the subfolder name under smt2_dir
    data4json_conversion = os.listdir(smt2_dir)

    pool = ThreadPool(multiprocessing.cpu_count())
    results = []
    results.extend(
        pool.apply_async(
            call_proc,
            (
                f"/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/data2dataset/smt2_cex2graph/model2graph {case_name}",
            ),
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
    json_dir = "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset/bad_cube_cex2graph/expr_to_build_graph"
    ground_truth_dir = "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset/bad_cube_cex2graph/ground_truth_table"
    # get all the subfolder name under expr_to_build_graph (expression to build graph)
    data4pickle_conversion = os.listdir(json_dir)

    pool = ThreadPool(multiprocessing.cpu_count())
    results = []
    results.extend(
        pool.apply_async(
            call_proc,
            (
                f"python /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/data2dataset/smt2_cex2graph/json2networkx.py \
                --json_file_path   {json_dir}/{case_name} \
                --ground_truth_path  {ground_truth_dir}/{case_name}",
            ),
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
    generate_smt2_error_handle("/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/log/error_handle/mismatched_inv.log")
    
    # terminate in advance
    exit(0)

    '''
    ---------------------------------------------------------
    step zero, change the directory name of 
    "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset"
    with time stamp
    ---------------------------------------------------------
    '''

    old_dir_name = "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset"
    initialization(old_dir_name)

    '''
    -----------------------------------------------------------------
    first, generate smt2 file for prediction (-> inducitve invariant)
    -----------------------------------------------------------------
    '''
    # script folder: /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/data2dataset/cex2smt2/collect.py
    generate_smt2()

    '''
    -----------------------------------------------------------------
    handle some errors! some aiger's inv.cnf is mismatch
    -----------------------------------------------------------------
    '''
    # re-run the smt2 generation process with the error aiger file - inv.cnf mismatch
    # generate_smt2_error_handle() # I have put this in generate_smt2() function
    
    '''
    ---------------------------------------------------------
    second, generate graph file for training
    ---------------------------------------------------------
    '''
    # generate pre-graph (json format)
    # script folder: /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/data2dataset/smt2_cex2graph/model2graph
    generate_pre_graph()

    # generate post-graph (pickle format)
    # script folder: /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/data2dataset/smt2_cex2graph/json2networkx.py
    generate_post_graph()

    print("Finish building data! Ready to train!")


