'''
Used in training process (parameter setting)
'''
import argparse
from pathlib import Path
prefix_folder = folder = Path(__file__).parent.parent

parser = argparse.ArgumentParser()
parser.add_argument('--task-name', type=str, default='neuropdr', help='task name')
#
parser.add_argument('--local_rank', type=int, default=-1, help='local rank for dpp')
parser.add_argument('--dim', type=int, default=128, help='Dimension of variable and clause embeddings')
parser.add_argument('--n_rounds', type=int, default=26, help='Number of rounds of message passing')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--inf_dev', type=str, default='gpu')
#
# parser.add_argument('--n_pairs', action='store', type=int)
# parser.add_argument('--max_nodes_per_batch', action='store', type=int)
parser.add_argument('--gen_log', type=str, default=str(Path(__file__).parent.parent /'log/data_maker.log'))
# parser.add_argument('--min_n', type=int, default=10, help='min number of variables used for training')
# parser.add_argument('--max_n', type=int, default=40, help='max number of variables used for training')
# parser.add_argument('--p_k_2', action='store', dest='p_k_2', type=float, default=0.3)
# parser.add_argument('--p_geo', action='store', dest='p_geo', type=float, default=0.4)
# parser.add_argument('--py_seed', action='store', dest='py_seed', type=int, default=None)
# parser.add_argument('--np_seed', action='store', dest='np_seed', type=int, default=None)
# parser.add_argument('--one', action='store', dest='one', type=int, default=0)
#
parser.add_argument('--log-dir', type=str, default=str(Path(__file__).parent.parent /'log/'), help='log folder dir')
parser.add_argument('--model-dir', type=str, default=str(Path(__file__).parent.parent /'neurograph_model/'), help='model folder dir')
parser.add_argument('--data-dir', type=str, default=str(Path(__file__).parent.parent /'dataset/'), help='data folder dir')
parser.add_argument('--restore', type=str, default=None, help='continue train from model')
#
parser.add_argument('--train-file', type=str, default=None, help='train file dir')
parser.add_argument('--val-file', type=str, default=None, help='val file dir')
parser.add_argument('--mode', type=str, default=None, help='mode to train or debug')
parser.add_argument('--gpu-id', type=int, default=0, help='gpu id to use')
parser.add_argument('--batch-size', type=int, default=2, help='batch size')
parser.add_argument('--dataset-type', type=str, default=None, help='dataset type, default is None, speical is "toy", will do clipping to train list')
# add arguments to adjust positive weight in BCEWithLogitsLoss
parser.add_argument('--pos-weight', type=float, default=1.0, help='positive weight in BCEWithLogitsLoss')
# add arguments to adjust learning rate
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')