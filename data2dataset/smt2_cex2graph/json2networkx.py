import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from natsort import natsorted
import os
# parse input arguments
import argparse

def json2graph_pickle(filename,ground_truth_path, pickle_file_name_prefix):
    with open(filename) as f:
        json_data = json.loads(f.read())

    # since we have already make constant boolean to "constant_true" or "constant_false", we make a assertion here
    assert all(
        elem['data']['application'] not in ['true', 'false']
        for elem in json_data
        if elem['data']['type'] == 'variable'
    ), 'Constant boolean value should be changed to "constant_true" or "constant_false"! Which should be ranked before input variable and latch variable!'

    # if json_data only have one node (it is abnormal graph), we stop the process
    if len(json_data) == 1: 
        '''
        Assert data is like this:
        {
        "data": {
            "application": "constant_false",
            "id": 0,
            "type": "variable"
        }
        }
        '''
        assert all([
            json_data[0]['data']['type'] == 'variable',
            json_data[0]['data']['application'] in ['constant_true', 'constant_false'],
            json_data[0]['data']['id'] == 0
        ]), 'Abnormal graph! Check json file!'
        #record_abnormal_graph(filename)
        return

    # change the node in json_data if it is a constant value like true, false -> can be done as double check
    '''
    for elem in json_data:
        if elem['data']['type'] == 'variable' and elem['data']['application'] in ['true', 'false']:
            # append 'constant_' to the name of the constant value
            elem['data']['application'] = 'constant_' + elem['data']['application']
    '''

    # Count down the amount of variable in json_data, if only one variable, we keep it because it is minimal generalized cex
    count_variable = sum(
        1
        for elem in json_data
        if elem['data']['type'] == 'variable'
        and elem['data']['application'].startswith('v')
    )
    if count_variable == 1: # skip the graph construction
        return
    
    # rank json_data by 'type' and 'id', node - input_var - variable -> sequence like this
    json_data = natsorted(json_data, key=lambda x: (x['data']['type'], x['data']['application']))

    #G = nx.DiGraph()
    G = nx.MultiDiGraph()

    G.add_nodes_from(
        elem['data']['id']
        for elem in json_data
    )

    edge_list = []

    for elem in json_data: 
        if elem['data']['type']=='node':
            edge_list.extend((child_id,elem['data']['id']) for child_id in elem['data']['to']['children_id'])

    G.add_edges_from(
        edge_list
    )

    # Convert graph to numpy matrix
    A=np.array(nx.to_numpy_matrix(G))

    # Get G egdes to dataframe
    edge_df = nx.to_pandas_adjacency(G, dtype=int) # The ordering is produced by G.nodes()

    #Adjust the order of dataframe is possible: read nodes by our preferred order 

    #dump G and json_data to pickle

    #define the pickle file name
    pickle_file_name = pickle_file_name_prefix + filename.split('/')[-1].replace('.json', '.pkl')

    # get ground truth list from '../../dataset/IG2graph/generalization_no_enumerate/' + filename.split('/')[-1].replace('.json', '.csv')
    case_name = ground_truth_path.split('/')[-1]
    ground_truth_table = pd.read_csv(f'{ground_truth_path}/{case_name}.csv')

    # only keep the particular case
    ground_truth_table = ground_truth_table[ground_truth_table['inductive_check']==filename.split('/')[-1].replace('.json', '.smt2')]
    label = ground_truth_table

    with open(pickle_file_name, 'wb') as f:
        pickle.dump([G, json_data, edge_df, label,filename.split('/')[-1].replace('.json', '')], f)


# load the json file in ../../dataset/IG2graph/generalize_IG_no_enumerate
# and convert it to a networkx graph
# only load .json file

# add main
if __name__ == '__main__':
    # parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file_path', type=str, default=None, help='the path of the json file')
    parser.add_argument('--ground_truth_path', type=str, default=None, help='the path of the ground truth table')
    parser.add_argument('--pickle_file_name_prefix', type=str, default=None, help='the prefix of the pickle file name')
    args = parser.parse_args(['--json_file_path', 
            '/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc2020_all_only_unsat_abc_no_simplification_0/bad_cube_cex2graph/expr_to_build_graph/vis_arrays_am2910_p2',
            '--ground_truth_path', 
            '/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc2020_all_only_unsat_abc_no_simplification_0/bad_cube_cex2graph/ground_truth_table/vis_arrays_am2910_p2',
            '--pickle_file_name_prefix',
            '/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc2020_all_only_unsat_abc_no_simplification_0/bad_cube_cex2graph/json_to_graph_pickle/'])
    #args = parser.parse_args()

    # assertion for the input arguments
    assert args.json_file_path is not None, "Please specify the path of the json file"
    assert args.ground_truth_path is not None, "Please specify the path of the ground truth table"

    #json_file_path = "../../dataset/bad_cube_cex2graph/expr_to_build_graph/nusmv.syncarb5^2.B"
    #json_file_path = "../../dataset/bad_cube_cex2graph/expr_to_build_graph/nusmv.reactor^4.C"
    json_file_path = args.json_file_path
    json_file_list = os.listdir(json_file_path)
    json_file_list = [x for x in json_file_list if x.endswith(".json")]

    #ground_truth_file_path = "../../dataset/bad_cube_cex2graph/ground_truth_table/nusmv.syncarb5^2.B"
    #ground_truth_file_path = "../../dataset/bad_cube_cex2graph/ground_truth_table/nusmv.reactor^4.C"
    ground_truth_file_path = args.ground_truth_path
    ground_truth_file_list = os.listdir(ground_truth_file_path)
    ground_truth_file_list = [x for x in ground_truth_file_list if x.endswith(".csv")]

    print("Total number of json files: ", len(json_file_list))

    for preprocess_cases in ground_truth_file_list:
        # filter the json file list by the ground truth file list
        json_file_list = [x for x in json_file_list if x.startswith(preprocess_cases.split('.')[0])]
        for json_file in json_file_list:
            print("Processing file: ", json_file)
            json2graph_pickle(os.path.join(json_file_path, json_file),ground_truth_file_path, pickle_file_name_prefix=args.pickle_file_name_prefix)