import argparse
import pickle
import os

from zmq import device
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from tqdm import tqdm
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from config import parser
from neurograph import NeuroInductiveGeneralization
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from time import sleep
import z3
# add "../utils" to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.toolbox import walkFile
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from natsort import natsorted

class ReloadedInt(int):
    def __truediv__(self, other):
        if other == 0:
            return 0
        else:
            return super().__truediv__(other)
    def __rtruediv__(self,other):
        if other == 0:
            return 0
        else:
            return super().__truediv__(other)

class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(BCEFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.sum()


class GraphDataset(Dataset):
    def __init__(self,data_root,mode='train'):
        self.data_root = data_root
        self.mode = mode
        self.samples = []
        self.__init_dataset()

    def __len__(self):
        return len(self.samples)

    def __df_to_np(self,df, graph_info=None):
        # copy the df to avoid the warning of SettingWithCopyWarning
        ground_truth_table = df.copy()
        # only keep the columns that are 'variable' that can be found in graph_info based on the column name
        ground_truth_table = ground_truth_table[[graph_info[i]['data']['application'] for i in range(len(graph_info)) if graph_info[i]['data']['type']=='variable' and graph_info[i]['data']['application'].startswith('v')]]
        #ground_truth_table.drop("Unnamed: 0", axis=1, inplace=True)
        ground_truth_table = ground_truth_table.reindex(natsorted(ground_truth_table.columns), axis=1)
        return ((ground_truth_table.values.tolist())[:])[0]

    def __getitem__(self, idx):
        
        #lambda function to sum up the true value in a list
        sum_true = lambda x: sum(i == True for i in x)
        graph_info = self.samples[idx][1]
        # only keep the top n_nodes rows of the adj_matrix
        adj_matrix = (self.samples[idx][2].head(sum_true([graph_info[i]['data']['type']=='node' for i in range(len(graph_info))])))
        ground_truth_label_row = self.samples[idx][3]
        file_name = self.samples[idx][4]
        var_index_in_graph = [i for i in range(len(graph_info)) if graph_info[i]['data']['application'].startswith('v')]
        # achieve the first row first column of the ground_truth_label_row
        assert(file_name == ground_truth_label_row['inductive_check'].iloc[0].replace('.smt2',''))
        prob_main_info = {
            'n_vars' : len(graph_info), #include m and variable (all node)
            # count the number of 'node' in the graph_info (node exclude input, input_prime, variable)
            'n_nodes' : sum_true([graph_info[i]['data']['type']=='node' for i in range(len(graph_info))]),
            'unpack' : (torch.from_numpy(adj_matrix.astype(np.float32).values)).to(device),
            'label' : self.__df_to_np(ground_truth_label_row,graph_info),
            # find the last element in the graph_info that is 'node'
            'refined_output' : list(map(lambda x: x-var_index_in_graph[0],var_index_in_graph)),
            'file_name' : file_name
        }
        assert(len(prob_main_info['label']) == len(prob_main_info['refined_output']))
        # assert sum of prob_main_info['label'] > 1
        assert(sum(prob_main_info['label']) >= 1)
        return prob_main_info, graph_info
    
    def __init_dataset(self):
        if self.mode == 'debug':
            train_lst = walkFile(self.data_root)
            for train_file in train_lst[:100]:
                with open(train_file, 'rb') as f:
                    self.samples.append(pickle.load(f))
        else:
            train_lst = walkFile(self.data_root)
            for train_file in train_lst[:]:
                with open(train_file, 'rb') as f:
                    self.samples.append(pickle.load(f))

def collate_wrapper(batch):
    prob_main_info, dict_vt = zip(*batch)
    return prob_main_info, dict_vt

if __name__ == "__main__":
    #device = 'cuda'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datetime_str = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    

    args = parser.parse_args(['--task-name', 'neuropdr_'+datetime_str.replace(' ', '_'), '--dim', '128', '--n_rounds', '256',
                              '--epochs', '512',
                              #'--log-dir', str(Path(__file__).parent.parent /'log/tmp/'), \
                              '--train-file', '../dataset/bad_cube_cex2graph/json_to_graph_pickle/',  
                              '--val-file', '../dataset/bad_cube_cex2graph/json_to_graph_pickle/',
                              '--mode', 'debug'
                              ])

    if args.mode == 'debug':
        writer = SummaryWriter('../log/tmp/tensorboard'+'-' + datetime_str.replace(' ', '_'))
        args.log_dir = str(Path(__file__).parent.parent /'log/tmp/')
    elif args.mode == 'train':
        writer = SummaryWriter('../log/tensorboard'+'-' + datetime_str.replace(' ', '_'))

    all_train = []
    train = []
    val = []

    all_graph = GraphDataset(args.train_file,args.mode)

    
    if args.mode == 'train' or args.mode == 'debug':
        train_size = int(0.6 * len(all_graph))
        validation_size = int(0.2 * len(all_graph))
        test_size = len(all_graph) - train_size - validation_size

        # Randomly
        train_dataset, test_dataset = torch.utils.data.random_split(all_graph, [train_size + validation_size, test_size])
        _ , validation_dataset = torch.utils.data.random_split(train_dataset, [train_size, validation_size])

        # Sequentially
        #train_dataset = torch.utils.data.Subset(all_graph, range(train_size))
        #validation_dataset = torch.utils.data.Subset(all_graph, range(train_size, train_size + validation_size))
        #test_dataset = torch.utils.data.Subset(all_graph, range(validation_size, len(all_graph)))

        train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_wrapper,
        num_workers=0)

        validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_wrapper,
        num_workers=0)

        test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_wrapper,
        num_workers=0)

    net = NeuroInductiveGeneralization(args)
    net = net.to(device)  # TODO: modify to accept both CPU and GPU version
    # if torch.cuda.device_count() > 1:
    #     net = torch.nn.DataParallel(net)

    log_file = open(os.path.join(args.log_dir, args.task_name + '.log'), 'a+')
    detail_log_file = open(os.path.join(
        args.log_dir, args.task_name + '_detail.log'), 'a+')

    #loss_fn = nn.BCELoss(reduction='sum')
    loss_fn = nn.BCEWithLogitsLoss(reduction='sum',pos_weight=torch.Tensor([4]).to(device))
    #loss_fn = BCEFocalLoss()
    #loss_fn = WeightedBCELosswithLogits()
    optim = optim.Adam(net.parameters(), lr=0.04, weight_decay=1e-10)
    sigmoid = nn.Sigmoid()

    best_acc = 0.0
    best_precision = 0.0
    start_epoch = 0

    if args.restore is not None:
        print('restoring from', args.restore, file=log_file, flush=True)
        model = torch.load(args.restore)
        start_epoch = model['epoch']
        best_acc = model['acc']
        best_precision = model['precision']
        net.load_state_dict(model['state_dict'])

    iteration = 0
    # one batch one iteration at first?
    for epoch in range(start_epoch, args.epochs):
        # Print on terminal
        print('==> %d/%d epoch, previous best precision: %.3f' %
              (epoch+1, args.epochs, best_precision))
        print('==> %d/%d epoch, previous best accuracy: %.3f' %
              (epoch+1, args.epochs, best_acc))
        
        # Write to log file (acc+precision)
        print('==> %d/%d epoch, previous best precision: %.3f' %
              (epoch+1, args.epochs, best_precision), file=log_file, flush=True)
        print('==> %d/%d epoch, previous best accuracy: %.3f' %
              (epoch+1, args.epochs, best_acc), file=log_file, flush=True)

        # Write to detailed log file (acc+precision)
        print('==> %d/%d epoch, previous best precision: %.3f' % (epoch+1,
              args.epochs, best_precision), file=detail_log_file, flush=True)
        print('==> %d/%d epoch, previous best accuracy: %.3f' %
              (epoch+1, args.epochs, best_acc), file=detail_log_file, flush=True)
        '''
        -----------------------train----------------------------------
        '''
        train_bar = tqdm(train_loader)
        TP, TN, FN, FP = torch.zeros(1).long(), torch.zeros(
            1).long(), torch.zeros(1).long(), torch.zeros(1).long()
        net.train()
        all_train_loss = torch.zeros(1).to(device)
        perfection_rate = 0  # Used to record the perfection ratio of the validation set
        all = 0  # Used to record the number of all samples in the validation set
        # FIXME: The train here should be a list contains serverals files (>1)
        
        # For every batch
        for batch_index, (prob,vt_dict) in enumerate(train_bar):
            optim.zero_grad() # set this for every batch
            assert(len(prob) == len(vt_dict))
            loss = torch.zeros(1).to(device)
            iteration += 1
            # For every problem in one batch
            for prob_index in range(len(prob)):
                q_index = prob[prob_index]['refined_output']
                outputs = net((prob[prob_index],vt_dict[prob_index]))
                target = torch.Tensor(prob[prob_index]['label'][:]).to(device).float()

                torch_select = torch.Tensor(q_index).to(device).int()
                outputs = torch.index_select(outputs, 0, torch_select)

                #assert(outputs.size() == target.size())
                this_loss = loss_fn(outputs, target)
                loss = loss+this_loss # loss for every batch

                # Calulate the perfect accuracy
                outputs = sigmoid(outputs)
                preds = torch.where(outputs > 0.6, torch.ones(
                outputs.shape).to(device), torch.zeros(outputs.shape).to(device))
                all = all + 1
                if target.equal(preds): perfection_rate = perfection_rate + 1

                # For every train file
                TP += (preds.eq(1) & target.eq(1)).cpu().sum()
                TN += (preds.eq(0) & target.eq(0)).cpu().sum()
                FN += (preds.eq(0) & target.eq(1)).cpu().sum()
                FP += (preds.eq(1) & target.eq(0)).cpu().sum()

                TOT = TP + TN + FN + FP

            # For every batch (iteration)
            writer.add_scalar('accuracy_per_iteration/train_perfection_ratio',(perfection_rate*1.0/all)*100, iteration)
            writer.add_scalar('confusion_matrix_per_iteration/true_possitive',TP.item()*1.0/TOT.item(), iteration)
            writer.add_scalar('confusion_matrix_per_iteration/false_possitive',FP.item()*1.0/TOT.item(), iteration)
            writer.add_scalar('confusion_matrix_per_iteration/precision', ReloadedInt(TP.item()* 1.0)/(TP.item()*1.0 + FP.item()*1.0), iteration)
            writer.add_scalar('model_evalute_per_iteration/F1-Socre', ReloadedInt(2*TP.item()*1.0) / (2*TP.item()*1.0+FP.item()*1.0+FN.item()*1.0), iteration)
            writer.add_scalar('accuracy_per_iteration/training_accuracy',(TP.item()+TN.item())*1.0/TOT.item(), iteration)
            writer.add_scalar('loss_per_iteration/training_loss', loss.item(), iteration)

            

            # Sum up the loss of every batch -> loss for every epoch
            #all_train_loss+=loss 
            all_train_loss=torch.sum(torch.cat([loss, all_train_loss], 0))
            all_train_loss=all_train_loss.unsqueeze(0) # Add one dimension
            
            # Backward and step
            loss.backward()
            optim.step()

            #for name, parms in net.named_parameters(): print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
            #all_train_loss_cpu_old = all_train_loss_cpu
            #all_train_loss_cpu = all_train_loss.cpu().item()
            #assert(all_train_loss_cpu>all_train_loss_cpu_old) #Assert that loss is increasing

            # Check this will affect training or not?
            #torch.cuda.empty_cache() #Assume that all_train_loss is not equal to zero
        
        # For every epoch
        desc = 'training loss: %.3f, perfection rate: %.3f, acc: %.3f, TP: %.5f, TN: %.5f, FN: %.5f, FP: %.5f' % (
        all_train_loss.item(), # Loss for every epoch
        perfection_rate*1.0/all, (TP.item()+TN.item())*1.0/TOT.item(), TP.item()*1.0/TOT.item(), 
        TN.item()*1.0/TOT.item(), 
        FN.item()*1.0/TOT.item(), 
        FP.item()*1.0/TOT.item())

        print(desc, file=log_file, flush=True)
        writer.add_scalar('accuracy/training_accuracy',(TP.item()+TN.item())*1.0/TOT.item(), epoch)
        writer.add_scalar('loss/training_loss', all_train_loss.item(), epoch)

        '''
        -------------------------validation--------------------------------
        '''

        val_bar = tqdm(validation_loader)
        TP, TN, FN, FP = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long()
        net.eval()
        perfection_rate = 0  # Used to record the perfection ratio of the validation set
        all = 0  # Used to record the number of all samples in the validation set
        loss = torch.zeros(1).to(device)
        #loss_cpu = torch.zeros(1).to('cpu')
        with torch.no_grad():
            for batch_index, (prob,vt_dict) in enumerate(val_bar):
                q_index = prob[0]['refined_output']
                #optim.zero_grad()
                outputs = net((prob[0],vt_dict[0]))
                target = torch.Tensor(prob[0]['label'][:]).to(device).float()
                torch_select = torch.Tensor(q_index).to(device).int()
                outputs = torch.index_select(outputs, 0, torch_select)
                
                this_loss = loss_fn(outputs, target)
                #loss += this_loss
                loss = torch.sum(torch.cat([loss, this_loss.unsqueeze(0)], 0))
                loss = loss.unsqueeze(0)

                outputs = sigmoid(outputs)
                preds = torch.where(outputs > 0.7, torch.ones(
                    outputs.shape).to(device), torch.zeros(outputs.shape).to(device))

                # Calulate the perfect accuracy
                all = all + 1
                if target.equal(preds):
                    perfection_rate = perfection_rate + 1

                TP += (preds.eq(1) & target.eq(1)).cpu().sum()
                TN += (preds.eq(0) & target.eq(0)).cpu().sum()
                FN += (preds.eq(0) & target.eq(1)).cpu().sum()
                FP += (preds.eq(1) & target.eq(0)).cpu().sum()
                #torch.cuda.empty_cache()
        
        # Write log for every epoch
        TOT = TP + TN + FN + FP
        val_precision = ReloadedInt(TP.item()*1.0)/(TP.item()*1.0 + FP.item()*1.0)
        acc =  (TP.item() + TN.item()) * 1.0 / TOT.item()
        desc = 'validation loss: %.3f, perfection rate: %.3f, acc: %.3f, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f' % (
            loss.item(),
            perfection_rate*1.0/all,
            acc, TP.item() *
            1.0 / TOT.item(), TN.item() * 1.0 / TOT.item(),
            FN.item() * 1.0 / TOT.item(), FP.item() * 1.0 / TOT.item())
        print(desc, file=log_file, flush=True)

        # Draw tensorboard for every epoch
        writer.add_scalar('loss/validation_loss', loss.item(), epoch)
        writer.add_scalar('accuracy/validation_predict_perfection_ratio', (perfection_rate*1.0/all)*100, epoch)
        writer.add_scalar('accuracy/validation_accuracy', acc, epoch)
        writer.add_scalar('precision/validation_precision', val_precision, epoch)

        '''
        ------------------------testing-----------------------------------
        '''

        test_bar = tqdm(test_loader)
        TP, TN, FN, FP = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long()
        net.eval()
        perfection_rate = 0  # Used to record the perfection ratio of the validation set
        all = 0  # Used to record the number of all samples in the validation set
        loss = torch.zeros(1).to(device)
        with torch.no_grad():
            for batch_index , (prob,vt_dict) in enumerate(test_bar):
                q_index = prob[0]['refined_output']
                #optim.zero_grad()
                outputs = net((prob[0],vt_dict[0]))
                target = torch.Tensor(prob[0]['label'][:]).to(device).float()
                torch_select = torch.Tensor(q_index).to(device).int()
                outputs = torch.index_select(outputs, 0, torch_select)

                # Calculate loss
                this_loss = loss_fn(outputs, target)
                loss = torch.sum(torch.cat([loss, this_loss.unsqueeze(0)], 0))
                loss = loss.unsqueeze(0)

                outputs = sigmoid(outputs)
                preds = torch.where(outputs > 0.8, torch.ones(
                    outputs.shape).to(device), torch.zeros(outputs.shape).to(device))

                # Calulate the perfect accuracy
                all = all + 1
                if target.equal(preds):
                    perfection_rate = perfection_rate + 1

                # For every batch (for every file if batch size = 1)
                TP += (preds.eq(1) & target.eq(1)).cpu().sum()
                TN += (preds.eq(0) & target.eq(0)).cpu().sum()
                FN += (preds.eq(0) & target.eq(1)).cpu().sum()
                FP += (preds.eq(1) & target.eq(0)).cpu().sum()

        # Write log for every epoch
        TOT = TP + TN + FN + FP
        desc = 'testing loss: %.3f, perfection rate: %.3f, acc: %.3f, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f' % (
            loss.item(),
            perfection_rate*1.0/all,
            (TP.item() + TN.item()) * 1.0 / TOT.item(), TP.item() *
            1.0 / TOT.item(), TN.item() * 1.0 / TOT.item(),
            FN.item() * 1.0 / TOT.item(), FP.item() * 1.0 / TOT.item())
        print(desc, file=log_file, flush=True)

        # Draw tensorboard for every epoch
        acc = (TP.item() + TN.item()) * 1.0 / TOT.item()
        test_precision = ReloadedInt(TP.item()*1.0)/(TP.item()*1.0 + FP.item()*1.0)
        writer.add_scalar('loss/tesing_loss', loss.item(), epoch)
        writer.add_scalar('accuracy/testing_predict_perfection_ratio',(perfection_rate*1.0/all)*100, epoch)
        writer.add_scalar('accuracy/testing_accuracy', acc, epoch)

        
        # Save model for every epoch
        torch.save({'epoch': epoch + 1, 'acc': acc, 'precision': test_precision, 'state_dict': net.state_dict()},
                   os.path.join(args.model_dir, args.task_name + '_last.pth.tar'))
        if test_precision >= best_precision:
            best_precision = test_precision
            torch.save({'epoch': epoch + 1, 'acc': acc, 'precision': best_precision, 'state_dict': net.state_dict()},
                       os.path.join(args.model_dir, args.task_name + '_best_precision.pth.tar'))

        if acc >= best_acc:
            best_acc = acc

    try:
        writer.close()
    except BaseException:
        writer.close()