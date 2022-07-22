#from aig2graph import AigGraph, Clauses
from logging import exception

from zmq import FD
import sys
sys.path.append("../data2dataset/aig2graph/")
from models_1stlayer_bwd import DGDAGRNN
from utils import expand_clause_012, clause_loss, clause_loss_weighted, prediction_has_absone, load_module_state, quantize, measure_012, measure_to_str, set_label_weight, get_label_freq
from tqdm import tqdm
import random
import torch
torch.autograd.set_detect_anomaly(True)
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import config
import pickle
import os
import copy
from loguru import logger
import math
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from sklearn.utils.class_weight import compute_class_weight 


logger.add("train_t1_lastlayer_log.txt")
config.to_str(logger.info)
#logger.info(config)

model = DGDAGRNN(nvt = config.nvt, vhs = config.vhs, nrounds = config.nrounds)
optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
#lossfun = nn.MSELoss()
#lossfun = clause_loss_weighted

#val_lossfun = nn.CrossEntropyLoss(reduction="sum")
# val_lossfun = clause_loss_weighted


model.to(config.device)
logger.info(model)

random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)

def train(epoch, train_data, batch_size, loss_fun):
    model.train()
    train_loss = 0
    TP50, FP50, TN50, FN50, ACC50 = 0,0,0,0,0
    TOT = 0
    
    #random.shuffle(train_data) let's not shuffling for debug purpose
    pbar = tqdm(train_data)
    g_batch = []
    batch_idx = 0
    for i, g in enumerate(pbar):
        # Filtering the graph that too sparse
        # clauses_len_sum = [len(tuple) for tuple in g.clauses]
        # percentage_clauses_node = sum(clauses_len_sum) / g.num_nodes
        # if percentage_clauses_node > 0.1:
        #     g_batch.append(g)
        g_batch.append(g)

        if len(g_batch) == batch_size or (i == len(train_data) - 1 and len(g_batch)!=0):
            batch_idx += 1
            optimizer.zero_grad()
            
            #g_batch = model._collate_fn(g_batch)
            # binary_logit = model(g_batch)
            loss = torch.zeros(1).to(config.device)
            variance = 0
            #loss.requires_grad_(True)
            for data in g_batch:
                # make a copy, so we don't occupy GPU all the time
                data = copy.deepcopy(data)
                n_sv = data.sv_node.shape[0]
                n_clause = len(data.clauses)+1  # the last one is the end (all 00)
                #print (prediction)
                #print (data.clauses[0])
                clauses = expand_clause_012(data.clauses, n_sv = n_sv)
                    
                if clauses is None:
                    print (data.aag_name)
                    exit(1)
                if config.clause_clip != 0:
                    n_clause = min(config.clause_clip, n_clause)
                    clauses = clauses[:n_clause]
                #print (clauses)
                clauses = clauses.to(config.device)

                # use the first clause
                clauses = clauses[-2]
                n_clause = 1

                prediction = model(data, n_clause, True)
                
                variance += data.variance

                if (torch.any(torch.isnan(prediction))):
                    print ('!!! prediction NAN!!!', data.aag_name)
                if (torch.any(torch.isnan(clauses))):
                    print ('!!! target NAN!!!', data.aag_name)
                
                # Use MSE
                #train_lossfun = nn.MSELoss(reduction='sum')
                #this_loss = train_lossfun(prediction, clauses)

                # Use MSE with label weight
                this_loss = loss_fun(prediction, clauses)
                loss += this_loss

                #print (prediction)
                if torch.any(torch.isnan(loss)):
                    print ("!!! loss NAN!!!",  data.aag_name)

                quantized_prediction = torch.max(prediction,dim=1)[1]
                TP, FP, TN, FN, ACC, INC = measure_012(clauses, quantized_prediction)

                assert (ACC + INC == n_clause*n_sv)

                TP50 += TP; FP50 += FP; TN50 += TN; FN50 += FN; ACC50 += ACC
                TOT+=n_clause*n_sv

            # end of for data in batch
            try:
                PRECISION=TP50/(TP50+FP50)
            except:
                PRECISION=0
            msg50=measure_to_str(TP50/TOT, FP50/TOT, TN50/TOT, FN50/TOT, ACC50/TOT, PRECISION)
            loss.backward()

            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            with torch.no_grad():
                maxgrad = torch.zeros(1).to(config.device)
            
                for param in model.parameters():
                    if param.grad is not None:
                        if torch.any(torch.isnan(param.grad)):
                            print ('grad NaN!!!')
                            exit(1)
                        param_grad_max = torch.max(torch.abs(param.grad))
                        if param_grad_max.item() > maxgrad.item():
                            maxgrad = param_grad_max
            
            
            optimizer.step()
            
            for param in model.parameters():
                if torch.any(torch.isnan(param)):
                    print ('param after grad decent NaN!!!')
                    exit(1)

            train_loss += loss.item()
            pbar.set_description('Epoch: %d, loss: %0.4f, %s, var:%0.4f, max grad: %0.5f' % (
                             epoch, loss.item()/len(g_batch), msg50, variance, maxgrad.item()))

            g_batch = []

    train_loss /= len(train_data)
    msg50=measure_to_str(TP50/TOT, FP50/TOT, TN50/TOT, FN50/TOT, ACC50/TOT, PRECISION)


    print('====> Epoch Train: {:d} Average loss: {:.4f}, {}'.format(
          epoch, train_loss, msg50))

    return train_loss



def test(epoch, test_data, batch_size, loss_fun):
    model.eval()
    test_loss = 0
    TP50, FP50, TN50, FN50, ACC50 = 0,0,0,0,0
    TOT = 0

    random.shuffle(test_data)
    pbar = tqdm(test_data)
    g_batch = []
    batch_idx = 0
    for i, g in enumerate(pbar):
        g_batch.append(g)
        if len(g_batch) == batch_size or i == len(test_data) - 1:  # we do not batch during testing
            loss = torch.zeros(1).to(config.device)
            batch_idx += 1
            
            optimizer.zero_grad()
            
            assert (len(g_batch) == 1)
            data = g_batch[0]
            data = copy.deepcopy(data)
            n_sv = data.sv_node.shape[0]
            n_clause = len(data.clauses)+1  # the last one is the end (all 00)
            clauses = expand_clause_012(data.clauses, n_sv = n_sv)
            if config.clause_clip != 0:
                n_clause = min(config.clause_clip, n_clause)
                clauses = clauses[:n_clause]
            clauses = clauses.to(config.device)

            clauses = clauses[-2]
            n_clause = 1
            
            prediction = model(data, n_clause, True)

            this_loss = loss_fun(prediction, clauses)
            quantized_prediction = torch.max(prediction,dim=1)[1]
            TP, FP, TN, FN, ACC, INC = measure_012(clauses, quantized_prediction)



            TP, FP, TN, FN, ACC, INC = measure_012(clauses, quantized_prediction)
            assert (ACC + INC == n_clause*n_sv)

            TP50 += TP; FP50 += FP; TN50 += TN; FN50 += FN; ACC50 += ACC

            #Threshold 80
            # TP, FP, TN, FN, ACC, _ = measure(clauses, quantize_80)
            # TP80 += TP; FP80 += FP; TN80 += TN; FN80 += FN; ACC80 += ACC

            #Threshold 95
            # TP, FP, TN, FN, ACC, _ = measure(clauses, quantize_95)
            # TP95 += TP; FP95 += FP; TN95 += TN; FN95 += FN; ACC95 += ACC

            TOT+=n_clause*n_sv
            try:
                PRECISION=TP50/(TP50+FP50)
            except:
                PRECISION=0
            msg50=measure_to_str(TP50/TOT, FP50/TOT, TN50/TOT, FN50/TOT, ACC50/TOT, PRECISION=PRECISION)
            

            test_loss += this_loss.item()
            pbar.set_description('Epoch: %d, loss: %0.4f, %s ' % (
                             epoch, loss.item()/len(g_batch), msg50))

            g_batch = []

    test_loss /= len(test_data)
    msg50=measure_to_str(TP50/TOT, FP50/TOT, TN50/TOT, FN50/TOT, ACC50/TOT, PRECISION=PRECISION)
    #msg80=measure_to_str(TP80/TOT, FP80/TOT, TN80/TOT, FN80/TOT, ACC80/TOT, 80)
    #msg95=measure_to_str(TP95/TOT, FP95/TOT, TN95/TOT, FN95/TOT, ACC95/TOT, 95)


    print('====> Epoch Test: {:d} Average loss: {:.4f}'.format(
          epoch, test_loss))
    print('====> Epoch Test: {:d} {}'.format(
          epoch, msg50))
    # print('====> Epoch Test: {:d} {}'.format(
    #       epoch, msg80))
    # print('====> Epoch Test: {:d} {}'.format(
    #       epoch, msg95))

    return test_loss, ACC50/TOT, PRECISION

def graph_filter(all_graphs, size):
    retG = []
    for g in all_graphs:
        num_node = g.x.shape[0]
        if num_node > size and size != 0:
            continue
        retG.append(g)
    return retG

def count_zero_one_ratio(all_graphs):
    literal_count = 0
    clause_width = 0
    for g in all_graphs:
        clauses = g.clauses
        n_sv = g.sv_node.shape[0]
        for c in clauses:
            literal_count += len(c)
        clause_width += len(clauses) * n_sv
    return literal_count, clause_width

def save_model(epoch, loss):
    logger.info("Save current model... fname:" + config.modelname)
    ckpt = {'epoch': epoch+1, 'loss': loss, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
    ckpt_name = os.path.join("saved_models", config.modelname + '_model_checkpoint{}.pth'.format(epoch))
    torch.save(ckpt, ckpt_name)

def load_model(fname):
    logger.info('Continue training from {}...'.format(fname))
    ckpt = torch.load(fname)
    start_epoch = ckpt['epoch']
    load_module_state(model, ckpt['state_dict'])
    load_module_state(optimizer, ckpt['optimizer'])
    load_module_state(scheduler, ckpt['scheduler'])

def adjust_learning_rate(learning_rate, learning_rate_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR multiplied by learning_rate_decay(set 0.98, usually) every epoch"""
    learning_rate = learning_rate * (learning_rate_decay ** epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    #return learning_rate

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    datetime_str = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    writer = SummaryWriter('./log/tensorboard_lastlayer_'+'-'+ datetime_str.replace(' ','_'))

    with open(config.dataset,'rb') as fin:
        all_graphs = pickle.load(fin)
    subset_graphs = graph_filter(all_graphs, config.use_size_below_this)
    train_graph, val_graph = train_test_split(subset_graphs, test_size=config.test_size, random_state=config.random_state)

    logger.info('Load %d AIG, will use %d of them.' % (len(all_graphs), len(subset_graphs)))
    is_subset = len(all_graphs) !=  len(subset_graphs)
    literal_count, clause_width = count_zero_one_ratio(subset_graphs)
    logger.info('+/- 1 %d , all %d , ratio:%0.4f' % (literal_count, clause_width, literal_count/clause_width))
    loss_weight = clause_width/literal_count
    loss_fun = nn.CrossEntropyLoss(weight=torch.tensor([loss_weight,1,loss_weight]).to(config.device))
    
    if config.continue_from_model:
        load_model(config.continue_from_model)

    start_epoch = 0
    os.system('date > loss_lastlayer.txt')
    for epoch in range(start_epoch + 1, config.epochs + 1):
        #adjust_learning_rate(config.lr,config.weight_decay,optimizer,epoch)
        train_loss = train(epoch,train_graph,config.batch_size, loss_fun)
        #optimizer.step(train_loss)
        scheduler.step(train_loss)
        with open("loss_lastlayer.txt", 'a') as loss_file:
            loss_file.write("{:.2f} \n".format(
                train_loss
                ))
        writer.add_scalar('train_loss', train_loss, epoch)
        if epoch%10 == 0:
            tloss, acc50, precision = test(epoch,val_graph,1, loss_fun) # use default batch size : 1
            writer.add_scalar('val_accuracy/acc50', acc50, epoch)
            writer.add_scalar('val_accuracy/precision', precision, epoch)


    save_model(config.epochs, train_loss)







