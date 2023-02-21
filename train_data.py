# this file is for training
from datetime import datetime
# system path append train_neurograph/
from train_neurograph.config import parser
import os
import argparse

# first, choose what model to use

# neurograph: /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/train_neurograph
# neurocircuit: /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/train_circuitsat

def choose_train_parameter(dataset_type=None):
    # choose model
    # choose dataset
    # choose hyperparameter
    
    assert dataset_type is not None, "Please choose a dataset type!"

    #initialize the parameter
    model_name = '' ; dataset = '' ; dimension_of_embedding = 0 ; number_of_rounds = 0 ; 
    number_of_epochs = 0 ; train_file = '' ; val_file = '' ; train_mode = '' ; gpu_id = 0
    task_name = 'neuropdr_'+datetime_str.replace(' ', '_')  ; batch_size = 0 ; possitive_weight_for_loss_fun = 0

    if dataset_type == 'hwmcc07_complete':
        model_name = 'neurograph'
        dataset = 'dataset'
        dimension_of_embedding = 128
        number_of_rounds = 128
        number_of_epochs = 512
        train_file = f'{dataset}/bad_cube_cex2graph/json_to_graph_pickle/'
        val_file = f'{dataset}/bad_cube_cex2graph/json_to_graph_pickle/'
        train_mode = 'train'
        gpu_id = 1
        batch_size = 1
        possitive_weight_for_loss_fun = 4
        learning_rate = 0.00001
    elif dataset_type == 'small':
        model_name = 'neurograph'
        dataset = 'dataset_20230106_025223_small'
        dimension_of_embedding = 128
        number_of_rounds = 512
        number_of_epochs = 400
        train_file = f'{dataset}/bad_cube_cex2graph/json_to_graph_pickle/'
        val_file = f'{dataset}/bad_cube_cex2graph/json_to_graph_pickle/'
        train_mode = 'debug'
        gpu_id = 0
        batch_size = 8
        possitive_weight_for_loss_fun = 2
        learning_rate = 0.0001
    elif dataset_type == 'toy':
        model_name = 'neurograph'
        dataset = 'dataset_20230106_014957_toy'
        dimension_of_embedding = 128
        number_of_rounds = 512
        number_of_epochs = 300
        train_file = f'{dataset}/bad_cube_cex2graph/json_to_graph_pickle/'
        val_file = f'{dataset}/bad_cube_cex2graph/json_to_graph_pickle/'
        train_mode = 'debug'
        gpu_id = 0
        batch_size = 1
        possitive_weight_for_loss_fun = 2
        learning_rate = 0.00001
    
    return (
        f'python train_{model_name}/train.py ' \
        f'--task-name {task_name} --dim {dimension_of_embedding} --n_rounds {number_of_rounds} ' \
        f'--epochs {number_of_epochs} --train-file {train_file} --val-file {val_file} ' \
        f'--mode {train_mode} --gpu-id {gpu_id} ' \
        f'--batch-size {batch_size} ' \
        f'--pos-weight {possitive_weight_for_loss_fun} ' \
        f'--lr {learning_rate} ' \
        f'--dataset-type  {dataset_type}'
    )


if __name__ == "__main__":
    # input argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-type', type=str, default=None)
    args = parser.parse_args(['--dataset-type', 'hwmcc07_complete'])
    dataset_type = args.dataset_type

    datetime_str = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')

    shell_cmd = choose_train_parameter(dataset_type)
    os.system(shell_cmd)
    




    
