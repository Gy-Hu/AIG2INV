import json
import argparse
import os

def average_graph_length(directory):
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    total_length = 0
    count = 0

    for json_file in json_files:
        with open(os.path.join(directory, json_file), 'r') as f:
            graph = json.loads(f.read())
            total_length += len(graph)
            count += 1

    return total_length / count if count > 0 else 0

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Calculate the average length of JSON graphs in multiple directories")
    parser.add_argument('--directories', type=str, nargs='+', help='Input the directories containing JSON graphs')
    args = parser.parse_args([
        '--directories', '/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc2020_all_only_unsat_abc_no_simplification_0/bad_cube_cex2graph/expr_to_build_graph/vis_arrays_am2910_p2/',
        '/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc2020_all_only_unsat_abc_naive_0/bad_cube_cex2graph/expr_to_build_graph/vis_arrays_am2910_p2/',
        '/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc2020_all_only_unsat_abc_slight_0/bad_cube_cex2graph/expr_to_build_graph/vis_arrays_am2910_p2/',
        '/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc2020_all_only_unsat_abc_moderate_0/bad_cube_cex2graph/expr_to_build_graph/vis_arrays_am2910_p2/',
        '/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc2020_all_only_unsat_abc_deep_0/bad_cube_cex2graph/expr_to_build_graph/vis_arrays_am2910_p2/',
        '/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc2020_all_only_unsat_abc_thorough_0/bad_cube_cex2graph/expr_to_build_graph/vis_arrays_am2910_p2/'
    ])

    if args.directories:
        for directory in args.directories:
            avg_length = average_graph_length(directory)
            print(f'The average length of JSON graphs in the directory {directory} is {avg_length}.')
    else:
        print("Please provide directories containing JSON graphs.")
