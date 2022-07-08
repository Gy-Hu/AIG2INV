import torch
import torch.nn as nn
import z3
import numpy as np
import pandas as pd

#from code.data_gen import problem


'''
-----------------Non-linear MLP--------------------
'''
class MLP(nn.Module):
  def __init__(self, in_dim, hidden_dim, out_dim):
    super(MLP, self).__init__()
    self.l1 = nn.Linear(in_dim, hidden_dim)
    self.l1_dropout = nn.Dropout(0.5)
    self.f1 = nn.ReLU()
    self.l2 = nn.Linear(hidden_dim, hidden_dim)
    self.l2_dropout = nn.Dropout(0.5)
    self.f2 = nn.ReLU()
    self.l3 = nn.Linear(hidden_dim, out_dim)

  def forward(self, x):
    x = self.l1(x)
    x = self.l1_dropout(x)
    x = self.f1(x)
    x = self.l2(x)
    x = self.l2_dropout(x)
    x = self.f2(x)
    output = self.l3(x)
    return output



class NeuroPredessor(nn.Module):
    def __init__(self,args = None):
        super(NeuroPredessor, self).__init__()
        self.args = args
        if args!=None:
            self.n_rounds = args.n_rounds
            if args.inf_dev == 'gpu':
                self.inf_device = 'cuda'
            elif args.inf_dev == 'cpu':
                self.inf_device = 'cpu'
            self.dim = args.dim
        else:
            self.n_rounds = 120
            self.inf_device = 'cuda'
            self.dim = 128

        self.init_ts = torch.ones(1)
        self.init_ts.requires_grad = False

        #TODO: Using 2 separated nn.Linear to do literals embedding
        self.true_init = nn.Linear(1,self.dim) #for node return true
        self.false_init = nn.Linear(1, self.dim) #for node return false

        self.children_msg = MLP(self.dim, self.dim, self.dim) #for children to pass message
        self.parent_msg = MLP(self.dim, self.dim, self.dim) #for parents to pass message

        self.node_update = nn.LSTM(self.dim, self.dim) #update node (exclude variable)
        self.var_update = nn.LSTM(self.dim, self.dim) #udpate variable and node

        #FIXME: fix here (how to defind new vote?)
        self.var_vote = MLP(self.dim, self.dim, 1) #vote for variable and node
        self.denom = torch.sqrt(torch.Tensor([self.dim]))


    def forward(self, wrapper):
        problem = wrapper[0]
        dict_vt = wrapper[1]
        init_ts = self.init_ts.to(self.inf_device)

        # TODO: change the init part to true/false init
        # dict_vt = dict(zip((problem.value_table).index, (problem.value_table).Value))

        #true_tensor = torch.tensor([]).to('cuda')
        #false_tensor = torch.tensor([]).to('cuda')
        all_init = torch.tensor([]).to(self.inf_device)
        for key, value in dict_vt.items():
            if value == 1:
                tmp_tensor = self.true_init(init_ts).view(1, 1, -1) #<-assign true init tensor
                #true_tensor = torch.cat((true_tensor,tmp_true_tensor),dim=1)
            else:
                tmp_tensor = self.false_init(init_ts).view(1, 1, -1) #<-assign false init tensor
                #false_tensor = torch.cat((false_tensor, tmp_false_tensor), dim=1)
            all_init = torch.cat((all_init,tmp_tensor),dim=1)

        #all_init = torch.cat((true_tensor, false_tensor),dim=1)

        # var_init = self.var_init(init_ts).view(1, 1, -1) # encode true or false here
        # node_init = self.node_init(init_ts).view(1, 1, -1) # re-construct the dimension, size = [1, 1, 128]
        # var_init = var_init.repeat(1, n_var, 1)
        # node_init = node_init.repeat(1, n_node, 1)

        var_state = (all_init[:], torch.zeros(1, problem['n_vars'], self.dim).to(self.inf_device)) # resize for LSTM, (ht, ct)
        '''
        var_state[:] -> all node includes input, input_prime, variable
        var_state[:?] -> node exclude input, input_prime, variable
        var_state[?:] -> only input, input_prime, variable (without m node)
        '''

        # adj_martix initialize here

        # message passing procedure
        #TODO: refine the n_rounds
        for _ in range(self.n_rounds): #TODO: Using LSTM to eliminate the error brought by symmetry

            var_pre_msg = self.children_msg(var_state[:][0].squeeze(0))
            child_to_par_msg = torch.matmul(problem['unpack'], var_pre_msg) #TODO: ask question "two embedding of m here"
            #FIXME: Expected hidden[0] size (1, 204, 128), got (1, 231, 128), fix the size in adj_matrix (where's prime needed?) , and re-run this
            #var_state_slice = ((var_state[0])[:,:problem.n_nodes,:], (var_state[1])[:,:problem.n_nodes,:])
            var_node_state = ((var_state[0])[:, :problem['n_nodes'], :], (var_state[1])[:, :problem['n_nodes'], :])
            var_rest_state = ((var_state[0])[:, problem['n_nodes']: , :], (var_state[1])[:, problem['n_nodes']: , :])
            _, var_node_state = self.var_update(child_to_par_msg.unsqueeze(0), var_node_state)
            #_, ((var_state[0])[:, :problem.n_nodes, :], (var_state[1])[:, :problem.n_nodes, :]) = self.var_update(child_to_par_msg.unsqueeze(0), ((var_state[0])[:, :problem.n_nodes, :], (var_state[1])[:, :problem.n_nodes, :]))
            #_, var_state = self.var_update(torch.cat(child_to_par_msg.unsqueeze(0), ((var_state[0])[:,:problem.n_nodes,:], (var_state[1])[:,:problem.n_nodes,:])))  #TODO: replace node_state with the partial var_state
            var_state = (torch.cat((var_node_state[0],var_rest_state[0]),dim=1),torch.cat((var_node_state[1],var_rest_state[1]),dim=1))

            node_pre_msg = self.parent_msg(((var_state[0])[:,:problem['n_nodes'],:]).squeeze(0))
            par_to_child_msg = torch.matmul(problem['unpack'].t(), node_pre_msg)
            _, var_state = self.node_update(par_to_child_msg.unsqueeze(0), var_state)

        logits = var_state[0].squeeze(0)
        #TODO: update here with the correct number
        vote = self.var_vote(logits[problem['n_nodes']:,:]) # (a+b) * dim -> a * dim
        #return the index of larger value
        #vote = torch.argmax(vote, dim=1)
        #vote_mean = torch.mean(vote, dim=1)
        return vote.squeeze()



