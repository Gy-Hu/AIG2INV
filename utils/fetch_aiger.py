'''
Fetch aiger from Zhang's csv file, and convert it to aag
'''


# Fetch small aiger from /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/clause-learning/data-collect/stat/

from distutils.dir_util import remove_tree
import pandas as pd
from ast import arg
import subprocess
import sys
from matplotlib.cbook import index_of
from natsort import natsorted
import argparse
from pathlib import Path
import os, os.path, shutil
from itertools import islice

# appen file path to the system path
# sys.path.append(f'{str(Path(__file__).parent.parent)}/code-python_version/')
from generate_aag import delete_redundant_line, chunk, remove_empty_file, remove_trivially_unsat_aiger

import tempfile

import aiger
#from aiger import utils


SIMPLIFY_TEMPLATE = '&r {0}; &put; fold; write_aiger -s {0}'


def simplify(aig_input_path, aag_output_path, verbose=False, abc_cmd='./yosys/yosys-abc', aigtoaig_cmd='./aiger_tool_util/aigtoaig'):
    aig_name = aig_input_path.split('/')[-1]
    # avoids confusion and guarantees deletion on exit
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        # copy the aig to the temporary directory
        shutil.copy(aig_input_path, tmpdir / aig_name)
        aig_path = tmpdir / str(aig_name)     

        command = [
            abc_cmd,
            '-c',
            SIMPLIFY_TEMPLATE.format(aig_path)
        ]

        subprocess.call(command) if verbose else subprocess.call(command, stdout=subprocess.PIPE)
        subprocess.call([aigtoaig_cmd, aig_path, aag_output_path+str(aig_name.replace('.aig','.aag'))])
        # move from tmpdir to output dir without aigtoaig_cmd
        #subprocess.call(['mv', aig_path, aag_output_path+str(aig_name)])

def fetch_aig_from_csv(csv_file):
    # Read this csv file into dataframe
    df = pd.read_csv(csv_file)

    # Then sort the dataframe by 'Res' and 'N_clauses'
    # df = df[df["res"] == "unsat"].sort_values(['res','n_clause'], ascending = True).head(50)
    
    df = df[df["res"] == "unsat"]
    # Export the aag_name column to a list
    aag_list = df["aag_name"].tolist()
    
    hard_aag_list = [
        #--------- benchmark 2020 stuck in generate smt2------------
        'vcegar_QF_BV_ar',
        'rast-p00',
        'vis_arrays_bufferAlloc',
        'zipversa_composecrc_prf-p10',
        'qspiflash_dualflexpress_divthree-p010',
        'vcegar_QF_BV_itc99_b13_p10',
        'zipcpu-busdelay-p00',
        'zipcpu-busdelay-p30',
        'qspiflash_qflexpress_divfive-p137',
        'qspiflash_dualflexpress_divthree-p046',
        'qspiflash_dualflexpress_divfive-p007',
        'qspiflash_dualflexpress_divfive-p154',
        'qspiflash_dualflexpress_divfive-p009',
        'qspiflash_dualflexpress_divfive-p116',
        'qspiflash_dualflexpress_divthree-p111',
        'elevator.4.prop1-func-interl',
        'h_RCU',
        'vgasim_imgfifo-p105',
        'cal87',
        'cal90',
        'picorv32-check-p05',
        'cal142',
        'picorv32-check-p20',
        'cal118',
        'cal97',
        'cal143',
        'cal102',
        'cal107',
        'cal112',
        'dspfilters_fastfir_second-p09',
        'dspfilters_fastfir_second-p26',
        'dspfilters_fastfir_second-p11',
        'cal161',
        'rushhour.4.prop1-func-interl',
        'cal176',
        #--------- benchmark 2020 stuck in model2graph------------
        'dspfilters_fastfir_second-p25',
        'dspfilters_fastfir_second-p05',
        'dspfilters_fastfir_second-p43',
        'dspfilters_fastfir_second-p14',
        'dspfilters_fastfir_second-p07',
        'dspfilters_fastfir_second-p16',
        'dspfilters_fastfir_second-p21',
        #--------- benchmark 2020 stuck in json2networkx------------
        'dspfilters_fastfir_second-p10',
        'dspfilters_fastfir_second-p45',
        'dspfilters_fastfir_second-p42',
        'dspfilters_fastfir_second-p04',
        'pgm_protocol.7.prop1-back-serstep',
        'dspfilters_fastfir_second-p2'
    ]
    
    '''
    filter out the aag_list by hard_aag_list
    '''
    # aag_list = [aag for aag in aag_list if aag.split('/')[-1] in hard_aag_list]
    # filtered_df = df[df['aag_name'].apply(lambda x: any(substring in x for substring in hard_aag_list))]

    '''
    filter out the aag_list by time and n_clause
    '''
    #df_sorted = df.sort_values(by='time', ascending=False)
    #df_filtered = df_sorted[df_sorted['n_clause'] <= 50]
    
    '''
    Adjust options
    '''
    # Set display options
    # pd.set_option("display.max_rows", 100)
    # pd.set_option("display.max_colwidth", None)
    
    # Add file path to the aag_list
    for i in range(len(aag_list)):
        #aag_list[i] = "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/clause-learning/data-collect/hwmcc07/" + aag_list[i] + ".aig"
        aag_list[i] = "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/clause-learning/data-collect/hwmcc20/" + aag_list[i] + ".aig"

    return aag_list

if __name__ == '__main__':
    aag_dir = './pre-dataset/aag4train_hwmcc20_only_unsat_hard/'
    parser = argparse.ArgumentParser(description="Convert aig to aag automatically")
    parser.add_argument('-outdir', type=str, default=aag_dir, help='Export the converted aag to the directory')
    parser.add_argument('-d', type=int, default=1, help='Determin whether to divide files into subset')
    parser.add_argument('-n', type=int, default=10, help='Determin how many files to divide into subset')
    args = parser.parse_args(['-n', '5'])
    '''
    --------------------Get the aig list (and their path)-------------------
    '''
    # make aag_dir if it does not exist
    if not os.path.isdir(aag_dir): 
        os.makedirs(aag_dir)
    csv_file = "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/clause-learning/data-collect/stat/size20.csv"
    aig_list = fetch_aig_from_csv(csv_file)

    for file in aig_list:
        # TODO: Use repr for this, make this command can be ran on Linux and Windows -> avoid Escape Character when use str
        # For instance, str(abc\ndef) will throw exception, using r'xxx' or repr() can avoid this problem
        '''
        --------------------Convert aiger1.9 to aiger1.0 --------------------
        '''
        simplify(file, aag_dir)

        # cmd = [str(Path(__file__).parent.parent/'code/aiger_tools/aigtoaig'), file, '-a']
        # with open(args.outdir + file.split('/')[-1].replace('.aig','.aag'), "w") as outfile:
        #     subprocess.run(cmd, stdout=outfile)

    '''
    -------------------sort the file by size and split into chunks-------------------
    '''
    if args.d != 0:
        sp = subprocess.Popen(f"du -b {aag_dir}/* | sort -n", stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        lst = [line.decode("utf-8").strip('\n').split('\t') for line in sp.stdout.readlines()]
        list_removed_empty = remove_empty_file(lst)
        list_removed_empty = delete_redundant_line(lst)
        list_removed_trivial_unsat = remove_trivially_unsat_aiger(list_removed_empty)
        list_chunks = list(chunk(list_removed_trivial_unsat, args.n))
        for i_tuple in range(len(list_chunks)):
            if not os.path.isdir(f"{aag_dir}/subset_{str(i_tuple)}"): 
                os.makedirs(f"{aag_dir}/subset_{str(i_tuple)}")
            for i_file in range(len(list_chunks[i_tuple])): 
                shutil.copy(list_chunks[i_tuple][i_file][1], f"{aag_dir}/subset_{str(i_tuple)}")

