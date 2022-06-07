

from cProfile import label
import numpy
import torch
import sys

def one_hot(idx, length):
    idx = torch.LongTensor([idx]).unsqueeze(0)
    x = torch.zeros((1, length)).scatter_(1, idx, 1)
    return x


# see https://github.com/unbounce/pytorch-tree-lstm/blob/66f29a44e98c7332661b57d22501107bcb193f90/treelstm/util.py#L8
# assume nodes consecutively named starting at 0
#
def top_sort(edge_index, graph_size):

    node_ids = numpy.arange(graph_size, dtype=int)

    node_order = numpy.zeros(graph_size, dtype=int)
    unevaluated_nodes = numpy.ones(graph_size, dtype=bool)

    parent_nodes = edge_index[0]
    child_nodes = edge_index[1]

    n = 0
    while unevaluated_nodes.any():
        # Find which parent nodes have not been evaluated
        unevaluated_mask = unevaluated_nodes[parent_nodes]

        # Find the child nodes of unevaluated parents
        unready_children = child_nodes[unevaluated_mask]

        # Mark nodes that have not yet been evaluated
        # and which are not in the list of children with unevaluated parent nodes
        nodes_to_evaluate = unevaluated_nodes & ~numpy.isin(node_ids, unready_children)

        node_order[nodes_to_evaluate] = n
        unevaluated_nodes[nodes_to_evaluate] = False

        n += 1

    return torch.from_numpy(node_order).long()


def add_order_info(graph):
    ns = torch.LongTensor([i for i in range(graph.num_nodes)])
    layers = torch.stack([top_sort(graph.edge_index, graph.num_nodes), ns], dim=0)
    ei2 = torch.LongTensor([list(graph.edge_index[1]), list(graph.edge_index[0])])
    layers2 = torch.stack([top_sort(ei2, graph.num_nodes), ns], dim=0)
    
    graph.__setattr__("forward_layer_index", layers)
    graph.__setattr__("backward_layer_index", layers2)
    

def expand_clause(clauses, n_sv):
    all_clauses = []
    for cidx, c in enumerate(clauses):
        x = torch.zeros((1, n_sv))
        for idx, v in c:
            if idx >= n_sv:
                # this is a bug
                print (cidx, ":", c)
                print ('n_sv :', n_sv)
                return None
            x[0][idx] = v
        all_clauses.append(x)
    all_clauses.append(torch.zeros((1, n_sv))) # this is the terminal symbol
    # this indicates no more clauses
    return torch.cat(all_clauses, dim=0).float()

    
def expand_clause_012(clauses, n_sv):
    all_clauses = []
    for cidx, c in enumerate(clauses):
        x = torch.ones((1, n_sv),dtype=torch.long) #x = torch.zeros((1, n_sv))
        for idx, v in c:
            if idx >= n_sv:
                # this is a bug
                print (cidx, ":", c)
                print ('n_sv :', n_sv)
                return None
            x[0][idx] = int(1+v)  # v
        all_clauses.append(x)
    all_clauses.append(torch.ones((1, n_sv),dtype=torch.long)) # all_clauses.append(torch.zeros((1, n_sv))) # this is the terminal symbol
    # this indicates no more clauses
    return torch.cat(all_clauses, dim=0).long()
    
def sum_clause(clauses):
    return torch.sum(clauses**2, dim=0)

def clause_loss(groundtruth, prediction):
    return torch.sum( (groundtruth-prediction)**2 )

def clause_loss_weighted(groundtruth, prediction, weight): # mse loss with weight
    return torch.sum( (groundtruth-prediction)**2 * (torch.abs(groundtruth)+weight)**2 )

def set_label_weight(expand_clauses, n_sv):
    label_weight = []
    for clauses_label in expand_clauses:
        x = torch.zeros((1, n_sv))
        for idx, v in enumerate(clauses_label.tolist()):
            if v in [-1, 1]:
                x[0][idx] = 5
            elif v == 0:
                x[0][idx] = 0.1
    
        label_weight.append(x)
    return torch.cat(label_weight, dim=0).float()

def get_label_freq(label):
        class_count = numpy.zeros((3)) 
        for label in label.tolist():
            if label == -1.0:       
                class_count[0] += 1
            elif label == 0.0:
                class_count[1] += 1
            elif label == 1.0:
                class_count[2] += 1
        class_count = numpy.where(class_count==0,sys.maxsize,class_count)
        return class_count

def prediction_has_absone(prediction):
    v, _ = torch.max(prediction[:-1]**2, dim=1)
    return torch.sum(1-v)


def load_module_state(model, state_name):
    model_dict = model.state_dict()
    
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    return
    

def quantize(logits, threshold):
    return ((logits>threshold).long()-(logits<-threshold).long())

def quantize_max(logits):
    assert False

def measure(expected, predicted):
    abs_expected = torch.abs(expected)
    abs_predict  = torch.abs(predicted)
    TP = torch.sum(torch.logical_and(abs_expected==1, abs_predict==1).long()).cpu().item()
    FP = torch.sum(torch.logical_and(abs_expected==0, abs_predict==1).long()).cpu().item()
    TN = torch.sum(torch.logical_and(abs_expected==0, abs_predict==0).long()).cpu().item()
    FN = torch.sum(torch.logical_and(abs_expected==1, abs_predict==0).long()).cpu().item()
    ACC = torch.sum((expected==predicted).long()).cpu().item()
    INC = torch.sum((expected!=predicted).long()).cpu().item()
    return TP, FP, TN, FN, ACC, INC
    
def measure_012(expected, predicted):
    abs_expected = torch.abs(expected-1)
    abs_predict  = torch.abs(predicted-1)
    TP = torch.sum(torch.logical_and(abs_expected==1, abs_predict==1).long()).cpu().item()
    FP = torch.sum(torch.logical_and(abs_expected==0, abs_predict==1).long()).cpu().item()
    TN = torch.sum(torch.logical_and(abs_expected==0, abs_predict==0).long()).cpu().item()
    FN = torch.sum(torch.logical_and(abs_expected==1, abs_predict==0).long()).cpu().item()
    ACC = torch.sum((expected==predicted).long()).cpu().item()
    INC = torch.sum((expected!=predicted).long()).cpu().item()
    return TP, FP, TN, FN, ACC, INC

def measure_to_str(TP, FP, TN, FN, ACC,PRECISION):
    return 'TP: %0.3f, FP: %0.3f, TN: %0.3f,FN: %0.3f,ACC: %0.3f, PRECISION: %0.3f' % (TP,FP,TN,FN,ACC,PRECISION)


