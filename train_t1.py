from aig2graph import AigGraph, Clauses
from models import DGDAGRNN
from utils import expand_clause, clause_loss, load_module_state, quantize, measure, measure_to_str
from tqdm import tqdm
from random import shuffle
import torch
torch.autograd.set_detect_anomaly(True)
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import z3
from config import config
import pickle
import os
import copy
from loguru import logger

logger.add("train_t1_log.txt")
config.to_str(logger.info)
#logger.info(config)

model = DGDAGRNN(nvt = config.nvt, vhs = config.vhs, nrounds = config.nrounds)
optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

model.to(config.device)
logger.info(model)

torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)

def train(epoch, train_data, batch_size):
    model.train()
    train_loss = 0
    TP50, FP50, TN50, FN50, ACC50 = 0,0,0,0,0
    TP80, FP80, TN80, FN80, ACC80 = 0,0,0,0,0
    TP95, FP95, TN95, FN95, ACC95 = 0,0,0,0,0
    TOT = 0

    shuffle(train_data)
    pbar = tqdm(train_data)
    g_batch = []
    batch_idx = 0
    for i, g in enumerate(pbar):
        g_batch.append(g)
        if len(g_batch) == batch_size or i == len(train_data) - 1:
            batch_idx += 1
            optimizer.zero_grad()
            #g_batch = model._collate_fn(g_batch)
            # binary_logit = model(g_batch)
            loss = torch.zeros(1).to(config.device)
            for data in g_batch:
                # make a copy, so we don't occupy GPU all the time
                data = copy.deepcopy(data)
                n_sv = data.sv_node.shape[0]
                n_clause = len(data.clauses)+1  # the last one is the end (all 00)
                prediction = model(data, n_clause, True)
                #print (prediction)
                #print (data.clauses[0])
                clauses = expand_clause(data.clauses, n_sv = n_sv)
                #print (clauses)
                clauses = clauses.to(config.device)
                loss = loss + clause_loss(clauses, prediction)
                
                quantize_50=quantize(prediction, 0.5)
                quantize_80=quantize(prediction, 0.8)
                quantize_95=quantize(prediction, 0.95)

                TP, FP, TN, FN, ACC, INC = measure(clauses, quantize_50)
                assert (ACC + INC == n_clause*n_sv)

                TP50 += TP; FP50 += FP; TN50 += TN; FN50 += FN; ACC50 += ACC
                TOT+=n_clause*n_sv
            # end of for data in batch
            msg50=measure_to_str(TP50/TOT, FP50/TOT, TN50/TOT, FN50/TOT, ACC50/TOT, 50)
            loss.backward()

            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_description('Epoch: %d, loss: %0.4f, %s' % (
                             epoch, loss.item()/len(g_batch), msg50))

            g_batch = []

    train_loss /= len(train_data)
    msg50=measure_to_str(TP50/TOT, FP50/TOT, TN50/TOT, FN50/TOT, ACC50/TOT, 50)


    print('====> Epoch Train: {:d} Average loss: {:.4f}, {}'.format(
          epoch, train_loss, msg50))

    return train_loss



def test(epoch, test_data, batch_size=1):
    model.eval()
    test_loss = 0
    TP50, FP50, TN50, FN50, ACC50 = 0,0,0,0,0
    TP80, FP80, TN80, FN80, ACC80 = 0,0,0,0,0
    TP95, FP95, TN95, FN95, ACC95 = 0,0,0,0,0
    TOT = 0

    shuffle(test_data)
    pbar = tqdm(test_data)
    g_batch = []
    batch_idx = 0
    for i, g in enumerate(pbar):
        g_batch.append(g)
        if len(g_batch) == batch_size or i == len(test_data) - 1:
            batch_idx += 1
            optimizer.zero_grad()
            #g_batch = model._collate_fn(g_batch)
            # binary_logit = model(g_batch)
            assert (len(g_batch) == 1)
            data = g_batch[0]
            data = copy.deepcopy(data)
            n_sv = data.sv_node.shape[0]
            n_clause = len(data.clauses)+1  # the last one is the end (all 00)
            prediction = model(data, n_clause, True)
            clauses = expand_clause(data.clauses, n_sv = n_sv)
            clauses = clauses.to(config.device)
            loss = clause_loss(clauses, prediction)
            
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

    return test_loss, ACC50/TOT, ACC80/TOT, ACC95/TOT

def graph_filter(all_graphs, size):
    retG = []
    for g in all_graphs:
        num_node = g.x.shape[0]
        if num_node > size and size != 0:
            continue
        retG.append(g)
    return retG

def save_model(epoch, loss):
    logger.info("Save current model...")
    ckpt = {'epoch': epoch+1, 'loss': loss, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
    ckpt_name = os.path.join("saved_models", 'model_checkpoint{}.pth'.format(epoch))
    torch.save(ckpt, ckpt_name)

def load_model(fname):
    logger.info('Continue training from {}...'.format(fname))
    ckpt = torch.load(fname)
    start_epoch = ckpt['epoch']
    load_module_state(model, ckpt['state_dict'])
    load_module_state(optimizer, ckpt['optimizer'])
    load_module_state(scheduler, ckpt['scheduler'])


with open(config.dataset,'rb') as fin:
    all_graphs = pickle.load(fin)
subset_graphs = graph_filter(all_graphs, config.use_size_below_this)
print ('Load %d AIG, will use %d of them.' % (len(all_graphs), len(subset_graphs)))

if config.continue_from_model:
    load_model(config.continue_from_model)

start_epoch = 0
os.system('date > loss.txt')
for epoch in range(start_epoch + 1, config.epochs + 1):
    train_loss = train(epoch,subset_graphs,config.batch_size)
    scheduler.step(train_loss)
    with open("loss.txt", 'a') as loss_file:
        loss_file.write("{:.2f} \n".format(
            train_loss
            ))
    if epoch%10 == 0:
        tloss, acc50, acc80, acc95 = test(epoch,subset_graphs,1) # use default batch size : 1
        threshold=None
        if acc50>0.99:
            threshold=0.5
        elif acc80>0.95:
            threshold=0.8
        elif acc95>0.95:
            threshold=0.95
        with open("loss.txt", 'a') as loss_file:
            loss_file.write("Test {:.2f} {:.2f} {:.2f} {:.2f}\n".format(
                tloss, acc50, acc80, acc95
                ))


        if threshold is not None:
            # then we can stop!
            print ('Accuracy is good enough! Training terminates early.')
            break


save_model(config.epochs, train_loss)







