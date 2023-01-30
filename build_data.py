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
    """ This runs in a separate thread. """
    #subprocess.call(shlex.split(cmd))  # This will block until cmd finishes
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = p.communicate()
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

def generate_smt2():
    #generate smt2 file for prediction -> SAT model/conterexample
    subset_dir = '/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/case4test/hwmcc2007/subset_'
    subset_dir_lst = [subset_dir+str(i) for i in range(23)] # 10 is the number for test subset
    #pool = ThreadPool(multiprocessing.cpu_count())
    pool = multiprocessing.Pool(32)
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


