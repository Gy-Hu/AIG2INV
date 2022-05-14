#from aig2graph import AigGraph, Clauses
from models import DGDAGRNN
from utils import expand_clause, clause_loss, clause_loss_weighted, prediction_has_absone, load_module_state, quantize, measure, measure_to_str, set_label_weight
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


logger.add("train_t1_log.txt")
config.to_str(logger.info)
#logger.info(config)

model = DGDAGRNN(nvt = config.nvt, vhs = config.vhs, nrounds = config.nrounds)
optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
#lossfun = nn.MSELoss()
#lossfun = clause_loss_weighted

val_lossfun = nn.CrossEntropyLoss(reduction="sum")

model.to(config.device)
logger.info(model)

random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)

def train(epoch, train_data, batch_size, loss_weight):
    model.train()
    train_loss = 0
    TP50, FP50, TN50, FN50, ACC50 = 0,0,0,0,0
    TP80, FP80, TN80, FN80, ACC80 = 0,0,0,0,0
    TP95, FP95, TN95, FN95, ACC95 = 0,0,0,0,0
    TOT = 0
    
    #random.shuffle(train_data) let's not shuffling for debug purpose
    pbar = tqdm(train_data)
    g_batch = []
    batch_idx = 0
    for i, g in enumerate(pbar):
        g_batch.append(g)
        if len(g_batch) == batch_size or i == len(train_data) - 1:
            batch_idx += 1
            optimizer.zero_grad()
            total_nodes = sum([g.x.shape[0] for g in g_batch])
            model.batch_init(total_nodes)
            #g_batch = model._collate_fn(g_batch)
            # binary_logit = model(g_batch)
            loss = torch.zeros(1).to(config.device)
            for data in g_batch:
                # make a copy, so we don't occupy GPU all the time
                data = copy.deepcopy(data)
                n_sv = data.sv_node.shape[0]
                n_clause = len(data.clauses)+1  # the last one is the end (all 00)
                #print (prediction)
                #print (data.clauses[0])
                clauses = expand_clause(data.clauses, n_sv = n_sv)
                    
                if clauses is None:
                    print (data.aag_name)
                    exit(1)
                if config.clause_clip != 0:
                    n_clause = min(config.clause_clip, n_clause)
                    clauses = clauses[:n_clause]
                #print (clauses)
                clauses = clauses.to(config.device)
                prediction = model(data, n_clause, True)
                
                if (torch.any(torch.isnan(prediction))):
                    print ('!!! prediction NAN!!!', data.aag_name)
                if (torch.any(torch.isnan(clauses))):
                    print ('!!! target NAN!!!', data.aag_name)
                    

                #this_loss = lossfun(clauses, prediction, loss_weight)
                class_weights = (set_label_weight(expand_clauses=clauses,n_sv = n_sv)).to(config.device)
                
                # Use CrossEntropyLoss
                # this_loss = []
                # for idx, preditct_clause in enumerate(prediction):
                #     train_lossfun = nn.CrossEntropyLoss(weight=class_weights[idx],reduction="sum")
                #     this_loss.append(train_lossfun(preditct_clause, clauses[idx]))
                # this_loss = sum(this_loss)
                
                # Use MSE
                train_lossfun = nn.MSELoss(reduction='sum')
                this_loss = train_lossfun(prediction, clauses)

                # this_loss = clause_loss(clauses, prediction)
                #if config.alpha > 0:
                #    this_loss = this_loss + prediction_has_absone(prediction) * config.alpha
                loss = loss + this_loss
                #print (prediction)
                if torch.any(torch.isnan(loss)):
                    print ("!!! loss NAN!!!",  data.aag_name)
                
                quantize_50=quantize(prediction, 0.5)
                #print (quantize_50)
                #quantize_80=quantize(prediction, 0.8)
                #quantize_95=quantize(prediction, 0.95)

                TP, FP, TN, FN, ACC, INC = measure(clauses, quantize_50)
                #print (TP, FP, TN, FN)
                #print (clauses)
                #exit (1)
                assert (ACC + INC == n_clause*n_sv)

                TP50 += TP; FP50 += FP; TN50 += TN; FN50 += FN; ACC50 += ACC
                TOT+=n_clause*n_sv
            # end of for data in batch
            msg50=measure_to_str(TP50/TOT, FP50/TOT, TN50/TOT, FN50/TOT, ACC50/TOT, 50)
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
            pbar.set_description('Epoch: %d, loss: %0.4f, %s, max grad: %0.5f' % (
                             epoch, loss.item()/len(g_batch), msg50, maxgrad.item()))

            g_batch = []

    train_loss /= len(train_data)
    msg50=measure_to_str(TP50/TOT, FP50/TOT, TN50/TOT, FN50/TOT, ACC50/TOT, 50)


    print('====> Epoch Train: {:d} Average loss: {:.4f}, {}'.format(
          epoch, train_loss, msg50))

    return train_loss



def test(epoch, test_data, batch_size, loss_weight):
    model.eval()
    test_loss = 0
    TP50, FP50, TN50, FN50, ACC50 = 0,0,0,0,0
    TP80, FP80, TN80, FN80, ACC80 = 0,0,0,0,0
    TP95, FP95, TN95, FN95, ACC95 = 0,0,0,0,0
    TOT = 0

    random.shuffle(test_data)
    pbar = tqdm(test_data)
    g_batch = []
    batch_idx = 0
    for i, g in enumerate(pbar):
        g_batch.append(g)
        if len(g_batch) == batch_size or i == len(test_data) - 1:
            batch_idx += 1
            total_nodes = sum([g.x.shape[0] for g in g_batch])
            model.batch_init(total_nodes)
            optimizer.zero_grad()
            #g_batch = model._collate_fn(g_batch)
            # binary_logit = model(g_batch)
            assert (len(g_batch) == 1)
            data = g_batch[0]
            data = copy.deepcopy(data)
            n_sv = data.sv_node.shape[0]
            n_clause = len(data.clauses)+1  # the last one is the end (all 00)
            clauses = expand_clause(data.clauses, n_sv = n_sv)
            if config.clause_clip != 0:
                n_clause = min(config.clause_clip, n_clause)
                clauses = clauses[:n_clause]
            clauses = clauses.to(config.device)
            
            prediction = model(data, n_clause, True)
            loss = val_lossfun(clauses, prediction, loss_weight)
            if config.alpha > 0:
                loss = loss + prediction_has_absone(prediction) * config.alpha
            
            quantize_50=quantize(prediction, 0.5)
            quantize_80=quantize(prediction, 0.8)
            quantize_95=quantize(prediction, 0.95)

            TP, FP, TN, FN, ACC, INC = measure(clauses, quantize_50)
            assert (ACC + INC == n_clause*n_sv)

            TP50 += TP; FP50 += FP; TN50 += TN; FN50 += FN; ACC50 += ACC
            TP, FP, TN, FN, ACC, _ = measure(clauses, quantize_80)
            TP80 += TP; FP80 += FP; TN80 += TN; FN80 += FN; ACC80 += ACC
            TP, FP, TN, FN, ACC, _ = measure(clauses, quantize_95)
            TP95 += TP; FP95 += FP; TN95 += TN; FN95 += FN; ACC95 += ACC
            TOT+=n_clause*n_sv

            msg50=measure_to_str(TP50/TOT, FP50/TOT, TN50/TOT, FN50/TOT, ACC50/TOT, 50)
            

            test_loss += loss.item()
            pbar.set_description('Epoch: %d, loss: %0.4f, %s ' % (
                             epoch, loss.item()/len(g_batch), msg50))

            g_batch = []

    test_loss /= len(test_data)
    msg50=measure_to_str(TP50/TOT, FP50/TOT, TN50/TOT, FN50/TOT, ACC50/TOT, 50)
    msg80=measure_to_str(TP80/TOT, FP80/TOT, TN80/TOT, FN80/TOT, ACC80/TOT, 80)
    msg95=measure_to_str(TP95/TOT, FP95/TOT, TN95/TOT, FN95/TOT, ACC95/TOT, 95)


    print('====> Epoch Test: {:d} Average loss: {:.4f}'.format(
          epoch, test_loss))
    print('====> Epoch Test: {:d} {}'.format(
          epoch, msg50))
    print('====> Epoch Test: {:d} {}'.format(
          epoch, msg80))
    print('====> Epoch Test: {:d} {}'.format(
          epoch, msg95))

    return test_loss, ACC50/TOT, ACC80/TOT, ACC95/TOT, TP50/TOT

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
    writer = SummaryWriter('./log/tensorboard'+'-'+ datetime_str.replace(' ','_'))

    with open(config.dataset,'rb') as fin:
        all_graphs = pickle.load(fin)
    subset_graphs = graph_filter(all_graphs, config.use_size_below_this)
    train_graph, val_graph = train_test_split(subset_graphs, test_size=config.test_size, random_state=config.random_state)

    logger.info('Load %d AIG, will use %d of them.' % (len(all_graphs), len(subset_graphs)))
    is_subset = len(all_graphs) !=  len(subset_graphs)
    literal_count, clause_width = count_zero_one_ratio(subset_graphs)
    logger.info('+/- 1 %d , all %d , ratio:%0.4f' % (literal_count, clause_width, literal_count/clause_width))
    loss_weight = math.sqrt(clause_width/literal_count)-1
    assert (loss_weight > 0)

    if config.continue_from_model:
        load_model(config.continue_from_model)

    start_epoch = 0
    os.system('date > loss.txt')
    for epoch in range(start_epoch + 1, config.epochs + 1):
        adjust_learning_rate(config.lr,config.weight_decay,optimizer,epoch)
        train_loss = train(epoch,train_graph,config.batch_size, loss_weight)
        scheduler.step(train_loss)
        with open("loss.txt", 'a') as loss_file:
            loss_file.write("{:.2f} \n".format(
                train_loss
                ))
        writer.add_scalar('train_loss', train_loss, epoch)
        if epoch%10 == 0:
            tloss, acc50, acc80, acc95, tp50 = test(epoch,val_graph,1, loss_weight) # use default batch size : 1
            writer.add_scalar('val_accuracy/acc50', acc50, epoch)
            writer.add_scalar('val_accuracy/acc80', acc80, epoch)
            writer.add_scalar('val_accuracy/acc95', acc95, epoch)
            threshold=None
            if acc50>0.9999 and acc80>0.95 and tp50 > 0.01:
                threshold=0.5
            elif acc80>0.98 and tp50 > 0.01:
                threshold=0.8
            elif acc95>0.95 and tp50 > 0.01:
                threshold=0.95

            if is_subset:
                print ("====> TEST ALL BELOW")
                _, acc50all, acc80all, acc95all = test(epoch, all_graphs, 1)
                print ("====> END OF TEST ALL")

            with open("loss.txt", 'a') as loss_file:
                loss_file.write("Test {:.2f} {:.2f} {:.2f} {:.2f}\n".format(
                    tloss, acc50, acc80, acc95
                    ))
                if is_subset:
                    loss_file.write("Test-ALL {:.2f} {:.2f} {:.2f} {:.2f}\n".format(
                        tloss, acc50, acc80, acc95
                        ))



            if threshold is not None:
                # then we can stop!
                print ('Accuracy is good enough! Training terminates early.')
                break


    save_model(config.epochs, train_loss)







