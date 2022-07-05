# -*- coding: UTF-8 -*- 
from logging import exception
from math import fabs
import profile
import string
from z3 import *
import time
import sys
import argparse
import csv
import numpy as np
import copy
from queue import PriorityQueue
#from config import parser
#from line_profiler import LineProfiler
from functools import wraps
import pandas as pd
from bmc import BMC
#from pyPDR.check_inv import remove_duplicate_list
import ternary_sim
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from scipy.special import comb
#import build_graph_online
#from deps.pydimacs_changed.formula import CNFFormula
import torch
import torch.nn as nn
#import neuro_ig_no_enumerate
from datetime import datetime
from operator import itemgetter
#import deps.PyMiniSolvers.minisolvers as minisolvers
import random
import collections
from scipy.spatial.distance import hamming
# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats,output_unit=1e-03)

# from line_profiler import LineProfiler
# profile = LineProfiler()
# profile = line_profiler.LineProfiler()
#from train import extract_q_like, refine_cube


# 查询接口中每行代码执行的时间
# def func_line_time(f):
#     @wraps(f)
#     def decorator(*args, **kwargs):
#         func_return = f(*args, **kwargs)
#         lp = LineProfiler()
#         lp_wrap = lp(f)
#         lp_wrap(*args, **kwargs)
#         lp.print_stats()
#         return func_return
#     return decorator

#TODO: Using Z3 to check the 3 properties of init, trans, safe, inductive invariant

# conjunction of literals.

class Frame:
    def __init__(self, lemmas):
        self.Lemma = lemmas
        self.pushed = [False] * len(lemmas)

    def cube(self):
        return And(self.Lemma)


    def add(self, clause, pushed=False):
        self.Lemma.append(clause)
        self.pushed.append(pushed)

    def __repr__(self):
        return str(sorted(self.Lemma, key=str))


class tCube:
    # make a tcube object assosciated with frame t.
    def __init__(self, t=0, cubeLiterals=None):
        self.t = t
        if cubeLiterals is None:
            self.cubeLiterals = list()
        else:
            self.cubeLiterals = cubeLiterals

    def __lt__(self, other):
        return self.t < other.t

    def clone(self):
        ret = tCube(self.t)
        ret.cubeLiterals = self.cubeLiterals.copy()
        return ret
    
    def clone_and_sort(self):
        ret = tCube(self.t)
        ret.cubeLiterals = self.cubeLiterals.copy()
        ret.cubeLiterals.sort(key=lambda x: str(_extract(x)[0]))
        return ret

    def remove_true(self):
        self.cubeLiterals = [c for c in self.cubeLiterals if c is not True]
    
    def __eq__(self, other) : 
        return collections.Counter(self.cubeLiterals) == collections.Counter(other.cubeLiterals)
        #return self.cubeLiterals == other.cubeLiterals
        # self.t == other.t

#TODO: Using multiple timer to caculate which part of the code has the most time consumption
    # 解析 sat 求解出的 model, 并将其加入到当前 tCube 中 #TODO: lMap should incudes the v prime and i prime
    def addModel(self, lMap, model, remove_input): # not remove input' when add model
        no_var_primes = [l for l in model if str(l)[0] == 'i' or not str(l).endswith('_prime')]# no_var_prime -> i2, i4, i6, i8, i2', i4', i6' or v2, v4, v6
        if remove_input:
            no_input = [l for l in no_var_primes if str(l)[0] != 'i'] # no_input -> v2, v4, v6
        else:
            no_input = no_var_primes # no_input -> i2, i4, i6, i8, i2', i4', i6' or v2, v4, v6
        # self.add(simplify(And([lMap[str(l)] == model[l] for l in no_input]))) # HZ:
        for l in no_input:
            self.add(lMap[str(l)] == model[l]) #TODO: Get model overhead is too high, using C API

    def remove_input(self):
        index_to_remove = set()
        for idx, literal in enumerate(self.cubeLiterals):
            children = literal.children()
            assert(len(children) == 2)

            if str(children[0]) in ['True', 'False']:
                v = str(children[1])
            elif str(children[1]) in ['True', 'False']:
                v = str(children[0])
            else:
                assert(False)
            assert (v[0] in ['i', 'v'])
            if v[0] == 'i':
                index_to_remove.add(idx)
        self.cubeLiterals = [self.cubeLiterals[i] for i in range(len(self.cubeLiterals)) if i not in index_to_remove]


    # 扩增 CNF 式
    def addAnds(self, ms):
        for i in ms:
            self.add(i)

    # 增加一个公式到当前 tCube() 中
    def add(self, m):
        self.cubeLiterals.append(m) # HZ: does not convert to cnf for the time being
        # g = Goal()
        # g.add(m) #TODO: Check 这边CNF会不会出现问题（试试arb-start那个case）
        # t = Tactic('tseitin-cnf')  # 转化得到该公式的 CNF 范式 #TODO:弄清楚这边转CNF如何转，能不能丢入Parafrost加速
        # for c in t(g)[0]:
        #     self.cubeLiterals.append(c)
        # if len(t(g)[0]) == 0:
        #     self.cubeLiterals.append(True)

    def true_size(self):
        '''
        Remove the 'True' in list (not the BoolRef Variable)
        '''
        return len(self.cubeLiterals) - self.cubeLiterals.count(True) 

    def join(self,  model):
        # first extract var,val from cubeLiteral
        literal_idx_to_remove = set()
        model = {str(var): model[var] for var in model}
        for idx, literal in enumerate(self.cubeLiterals):
            if literal is True:
                continue
            var, val = _extract(literal)
            var = str(var)
            assert(var[0] == 'v')
            if var not in model:
                literal_idx_to_remove.add(idx)
                continue
            val2 = model[var]
            if str(val2) == str(val):
                continue # will not remove
            literal_idx_to_remove.add(idx)
        for idx in literal_idx_to_remove:
            self.cubeLiterals[idx] = True
        return len(literal_idx_to_remove) != 0
        # for each variable in cubeLiteral, check if it has negative literal in model
        # if so, remove this literal
        # return False if there is no removal (which should not happen)


    # 删除第 i 个元素，并返回新的tCube
    def delete(self, i: int):
        res = tCube(self.t)
        for it, v in enumerate(self.cubeLiterals):
            if i == it:
                res.add(True)
                continue
            res.add(v)
        return res

    #TODO: 验证这个cube()是否导致了求解速度变慢

    def cube(self): #导致速度变慢的罪魁祸首？
        return simplify(And(self.cubeLiterals))

    # Convert the trans into real cube
    def cube_remove_equal(self):
        res = tCube(self.t)
        for literal in self.cubeLiterals:
            children = literal.children()
            assert(len(children) == 2)
            cube_literal = And(Not(And(children[0],Not(children[1]))), Not(And(children[1],Not(children[0]))))
            res.add(cube_literal)
        return res


    # def ternary_sim(self, index_of_x):
    #     # first extract var,val from cubeLiteral
    #     s = Solver()
    #     for idx, literal in enumerate(self.cubeLiterals):
    #         if idx !=index_of_x:
    #             s
    #             var = str(var)
    #
    # def cube(self):
    #     return And(*self.cubeLiterals)

    def __repr__(self):
        return str(self.t) + ": " + str(sorted(self.cubeLiterals, key=str))

def _extract(literaleq):
    # we require the input looks like v==val
    children = literaleq.children()
    assert(len(children) == 2)
    if str(children[0]) in ['True', 'False']:
        v = children[1]
        val = children[0]
    elif str(children[1]) in ['True', 'False']:
        v = children[0]
        val = children[1]
    else:
        assert(False)
    return v, val

class PDR:
    def __init__(self, primary_inputs, literals, primes, init, trans, post, pv2next, primes_inp, filename):
        '''
        :param primary_inputs:
        :param literals: Boolean Variables
        :param primes: The Post Condition Variable
        :param init: The initial State
        :param trans: Transition Function
        :param post: The Safety Property
        '''
        self.primary_inputs = primary_inputs
        self.init = init
        self.trans = trans
        self.literals = literals
        self.items = self.primary_inputs + self.literals + primes_inp + primes
        self.lMap = {str(l): l for l in self.items}
        self.post = post 
        self.frames = list()
       # self.primaMap_new = [(literals[i], primes[i]) for i in range(len(literals))] #TODO: Map the input to input' (input prime)
        self.primeMap = [(literals[i], primes[i]) for i in range(len(literals))]
        self.inp_map = [(primary_inputs[i], primes_inp[i]) for i in range(len(primes_inp))]
        #self.inp_prime = primes_inp
        self.pv2next = pv2next
        self.initprime = substitute(self.init.cube(), self.primeMap)
        # for debugging purpose
        self.bmc = BMC(primary_inputs=primary_inputs, literals=literals, primes=primes,
                       init=init, trans=trans, post=post, pv2next=pv2next, primes_inp = primes_inp)
        self.generaliztion_data_GP = []# Store the ground truth data of generalized predecessor
        self.generaliztion_data_IG = []# Store the ground truth data of inductive generalization 
        #TODO: Use self.generaliztion_data_IG to store the ground truth data of inductive generalization 
        self.filename = filename
        # create a ternary simulator and buffer the update functions in advance
        self.ternary_simulator = ternary_sim.AIGBuffer()
        for _, updatefun in self.pv2next.items():
            self.ternary_simulator.register_expr(updatefun)
        '''
        --------------The following variables are used to calculate the reducing rate--------
        '''
        self.sum_MIC = 0 # Sum of the literals produced by MIC
        self.sum_IG_GT = 0 # Sum of the literals produced by combinations
        self.sum_GP = 0 # Sum of the literals of predecessor (unsat core or other methods)
        self.sum_GP_GT = 0 #Sum of the minimum literals of predecessor (MUST, ternary simulation etc.)
        '''
        --------------Switch to open/close the ground truth data generation------------------
        '''
        self.smt2_gen_IG = 0
        self.smt2_gen_GP = 0
        '''
        --------------Switch to open/close the NN-guided inductive generalization------------------
        '''
        self.test_IG_NN = 0
        self.test_GP_NN = 0
        '''
        ---------------Count down the success/fail of NN-guided inductive generalization------------------
        '''
        self.NN_guide_ig_success = 0
        self.NN_guide_ig_fail = 0
        self.NN_guide_ig_iteration = 0
        self.NN_guide_ig_passed_ratio = []
        '''
        ---------------Time consuming of NN-guided/MIC inductive generalization------------------
        '''
        self.NN_guide_ig_time_sum = 0
        self.MIC_time_sum = 0
        self.pushLemma_time_sum = 0
        '''
        ---------------Determine whether append NN-guided ig append to MIC------------------
        '''
        self.NN_guide_ig_append = 0
        '''
        ---------------Collect the inductive invariant
        '''
        self.collect_inductive_invariant = 0
        '''
        ---------------Collect the result
        
        '''
        self.record_result = 0
        self.record_result_dict = {}
        '''
        --------------Set the prediction thershold------------------
        '''
        self.prediction_threshold = 0.5
        '''
        --------------Set the model name of NN to predict-----------
        '''
        self.model_name = None
        '''
        --------------Test mic?------------------------------------
        '''
        self.test_mic = 0
        '''
        -------------- Time consuming of random MIC-------------------
        '''
        self.test_random_mic_time_sum = 0
        '''
        ----------------Store the history append cex ------------
        '''
        self.history_append_cex = []
        '''
        ----------------Store the NN attempt ------------
        '''
        self.NN_attempt_fail = 0
        '''
        ------------------Store the folder name of test case ----------
        '''
        self.folder_name = None
        '''
        -----------------Device to inf----------------------
        '''
        self.inf_device = 'gpu'
        '''
        ------------------Check solve relative before export the CTI---------
        '''
        self.check_CTI_before_export = 0
        
    def check_init(self):
        s = Solver()
        s.add(self.init.cube())
        s.add(Not(self.post.cube()))
        res1 = s.check()
        if res1 == sat:
            return False
        s = Solver()
        s.add(self.init.cube())
        s.add(self.trans.cube())
        s.add(substitute(substitute(Not(self.post.cube()), self.primeMap),self.inp_map))
        res2 = s.check()
        if res2 == sat:
            return False
        return True

    #@profile
    def run(self, agent=None):

        if not self.check_init():
            print("Found trace ending in bad state")
            return False

        self.agent = agent
        self.frames = list() # list for Frame
        self.frames.append(Frame(lemmas=[self.init.cube()]))
        self.frames.append(Frame(lemmas=[self.post.cube()]))

        while True:
            c, all_model_lst_complete, all_model_lst_partial = self.getBadCube() # conduct generalize predecessor here
            if c is not None:
                # print("get bad cube!")
                trace = self.recBlockCube((c,all_model_lst_complete, all_model_lst_partial)) # conduct generalize predecessor here (in solve relative process)
                #TODO: 找出spec3-and-env这个case为什么没有recBlock
                if trace is not None:
                    # Generate ground truth of generalized predecessor
                    if self.generaliztion_data_GP: # When this list is not empty, it return true
                        df = pd.DataFrame(self.generaliztion_data_GP)
                        df = df.fillna(1)
                        df.to_csv("../dataset/GP2graph/generalization/" + (self.filename.split('/')[-1]).replace('.aag', '.csv'))
                    
                    # Generate ground truth of inductive generalization
                    if self.generaliztion_data_IG: # When this list is not empty, it return true
                        df = pd.DataFrame(self.generaliztion_data_IG)
                        df = df.fillna(0)
                        # Enumerate the inductive clauses
                        # df.to_csv("../dataset/IG2graph/generalization/" + (self.filename.split('/')[-1]).replace('.aag', '.csv'))
                        
                        # Data from the result of inductive invariant
                        df.to_csv("../dataset/IG2graph/generalization_no_enumerate/" + (self.filename.split('/')[-1]).replace('.aag', '.csv'))
                    
                    # Print out the improvement space of inductive generalization
                    if self.sum_IG_GT != 0:
                        print("Sum of the literals produced by MIC(): ",self.sum_MIC)
                        print("Sum of the literals produced by enumeration in indutive generalization: ",self.sum_IG_GT)
                        print("Reducing ",((self.sum_MIC-self.sum_IG_GT)/self.sum_MIC)*100,"% ")
                    
                    print("Found trace ending in bad state:")
                    
                    
                    self._debug_trace(trace)
                    # If want to print the trace, remember to comment the _debug_trace() function
                    while not trace.empty():
                        idx, cube = trace.get()
                        print(cube)
                    return False
                print("recBlockCube Ok! F:")

            else:
                inv = self.checkForInduction()
                if inv != None:
                    print("Found inductive invariant")
                    # Print out the improvement space of inductive generalization
                    if self.sum_IG_GT != 0:
                        print("Sum of the literals produced by MIC(): ",self.sum_MIC)
                        print("Sum of the literals produced by enumeration in indutive generalization: ",self.sum_IG_GT)
                        print("Reducing ",((self.sum_MIC-self.sum_IG_GT)/self.sum_MIC)*100,"% ")
                    
                    
                    # Generate ground truth of generalized predecessor
                    if self.generaliztion_data_GP: # When this list is not empty, it return true
                        df = pd.DataFrame(self.generaliztion_data_GP)
                        df = df.fillna(1)
                        #df.iloc[:,2:] = df.iloc[:,2:].apply(pd.to_numeric)
                        df.to_csv("../dataset/GP2graph/generalization/" + (self.filename.split('/')[-1]).replace('.aag', '.csv'))
                    
                    # Generate ground truth of inductive generalization
                    if self.generaliztion_data_IG: # When this list is not empty, it return true
                        df = pd.DataFrame(self.generaliztion_data_IG)
                        df = df.fillna(0)

                        # Enumerate the inductive clauses
                        # df.to_csv("../dataset/IG2graph/generalization/" + (self.filename.split('/')[-1]).replace('.aag', '.csv'))
                        
                        # Data from the result of inductive invariant
                        df.to_csv("../dataset/IG2graph/generalization_no_enumerate/" + (self.filename.split('/')[-1]).replace('.aag', '.csv'))


                    print ('Total F', len(self.frames), ' F[-1]:', len(self.frames[-1].Lemma))
                    self._debug_print_frame(len(self.frames)-1)

                    if self.record_result == 1 and self.folder_name is not None:
                        if self.test_IG_NN == 0 and self.NN_guide_ig_append == 0:
                            # Add info to the result recorder
                            self.record_result_dict['filename'] = self.filename
                            self.record_result_dict['Total Frame'] = len(self.frames)
                            self.record_result_dict['Number of clauses'] = len(self.frames[-1].Lemma)
                            self.record_result_dict["Time Consuming"] = time.time() - self.start_time
                            print("Export the result to csv file")
                            root = '/data/guangyuh/coding_env/ML4PDR/log/'
                            name = 'small_subset_without_NN' + "_" + self.folder_name
                            file_exists = os.path.isfile(root+name+".csv")
                            with open(os.path.join(root, name+".csv"), 'a+', newline='') as csvfile:
                                fieldnames = ['filename', 'Total Frame', 'Number of clauses', 'Time Consuming']
                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                if not file_exists: writer.writeheader() 
                                writer.writerow(self.record_result_dict)
                                csvfile.close()
                        elif self.test_IG_NN == 1 and self.NN_guide_ig_append == 1:
                            self.record_result_dict['filename'] = self.filename
                            self.record_result_dict['Total Frame'] = len(self.frames)
                            self.record_result_dict['Number of clauses'] = len(self.frames[-1].Lemma)
                            self.record_result_dict["Time Consuming"] = time.time() - self.start_time
                            self.record_result_dict["Time reduce INF time"] = self.record_result_dict["Time Consuming"] - self.NN_guide_ig_time_sum
                            self.record_result_dict["Prediction Thershold"] = self.prediction_threshold
                            try:
                                self.record_result_dict["Passing Ratio"] = str((self.NN_guide_ig_success/(self.NN_guide_ig_success + self.NN_guide_ig_fail))*100)+"%"
                            except: # Check this
                                self.record_result_dict["Passing Ratio"] = "nan"
                            print("Export the result to csv file")
                            root = '/data/guangyuh/coding_env/ML4PDR/log/'
                            name = 'small_subset_experiment_with_NN' + "_" + self.folder_name
                            file_exists = os.path.isfile(root+name+".csv")
                            with open(os.path.join(root, name+".csv"), 'a+', newline='') as csvfile:
                                fieldnames = ['filename', 'Total Frame', 'Number of clauses', 'Time Consuming',"Time reduce INF time","Prediction Thershold","Passing Ratio"]
                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                if not file_exists: writer.writeheader()  # file doesn't exist yet, write a header
                                writer.writerow(self.record_result_dict)
                                csvfile.close()
                
                    

                    return True
                print("Did not find invariant, adding frame " + str(len(self.frames)) + "...")


                print("Adding frame " + str(len(self.frames)) + "...")
                self.frames.append(Frame(lemmas=[])) # property can be directly pushed here

                # TODO: Append P, and get bad cube change to F[-1] /\ T /\ !P' (also can do generalization), check it is sat or not
                # [init, P]
                # init /\ bad   ?sat
                # init /\T /\ bad'  ?sat

                #TODO: Try new way to pushing lemma (like picking >=2 clause at once to add in new frame)
                for idx in range(1,len(self.frames)-1):
                    pushLemma_start_time = time.time()
                    self.pushLemma(idx)
                    pushLemma_consuming_t = time.time() - pushLemma_start_time
                    self.pushLemma_time_sum += pushLemma_consuming_t

                #--------------remove this due to the overhead-------
                #self._sanity_check_frame()
                print("Now print out the size of frames")
                for index in range(len(self.frames)):
                    push_cnt = self.frames[index].pushed.count(True)
                    print("F", index, 'size:', len(self.frames[index].Lemma), 'pushed: ', push_cnt)
                    assert (len(self.frames[index].Lemma) == len(self.frames[index].pushed))
                for index in range(1, len(self.frames)):
                    print (f'--------F {index}---------')
                    #-----------remove this due to the overhead------
                    #self._debug_print_frame(index, skip_pushed=True)
                #input() # pause





    def checkForInduction(self):
        #print("check for Induction now...")
        # check Fi+1 => Fi ?
        Fi2 = self.frames[-2].cube()
        Fi = self.frames[-1].cube()
        s = Solver()
        s.add(Fi)
        s.add(Not(Fi2))
        if s.check() == unsat:
            return Fi
        return None


    def pushLemma(self, Fidx:int):
        fi: Frame = self.frames[Fidx]

        for lidx, c in enumerate(fi.Lemma):
            if fi.pushed[lidx]:
                continue
            s = Solver()
            s.add(fi.cube())
            s.add(self.trans.cube())
            s.add(substitute(Not(substitute(c, self.primeMap)),self.inp_map))

            # f = CNFFormula.from_z3(s.assertions())
            # cnf_string_lst = f.to_dimacs_string()
            # n, iclauses = self.parse_dimacs(cnf_string_lst)
            # minisolver = minisolvers.MinisatSolver()
            # for i in range(n): minisolver.new_var(dvar=True)
            # for iclause in iclauses: minisolver.add_clause(iclause)
            # is_sat = minisolver.solve()
            # assert((is_sat==False and s.check()==unsat) or (is_sat==True and s.check()==sat))

            #if is_sat==False:
            if s.check()==unsat:
                fi.pushed[lidx] = True
                self.frames[Fidx + 1].add(c)
    
    def frame_trivially_block(self, st: tCube):
        Fidx = st.t
        slv = Solver()
        slv.add(self.frames[Fidx].cube())
        slv.add(st.cube())
        if slv.check() == unsat:
            return True
        return False

    #TODO: 解决这边特殊case遇到safe判断成unsafe的问题
    
    def export_CTI_lst(self, cti_lst_complete: list, cti_lst_partial:list):
        if len(cti_lst_complete)==0 or len(cti_lst_partial) == 0:
            return
        
        if self.check_CTI_before_export == 1:
            #check_init = lambda s,cti : s.add(self.init.cube(),cti.cube())
            check_solve_relative = lambda x: self._solveRelative_upgrade(x) == 'pass the check'
            #check_solve_relative = lambda x: self._solveRelative(x) == unsat
            assert(all(check_solve_relative(x) for x in cti_lst_complete))
            assert(all(check_solve_relative(x) for x in cti_lst_partial))
        
        # change the list to string, use comma to split
        cubeliteral_to_str = lambda cube_literals: ','.join(map
                                (lambda x: str(_extract(x)[0]).replace('v','') 
                                if str(_extract(x)[1])=='True' 
                                else str(int(str(_extract(x)[0]).replace('v',''))+1),cube_literals))
        # open a file for writing
        with open("./" + self.filename.split('/')[-1].replace('.aag', '') + "_complete_CTI.txt", "w") as text_file:
            for cti in cti_lst_complete:
                text_file.write(cubeliteral_to_str(cti.cubeLiterals) + "\n")
        
        with open("./" + self.filename.split('/')[-1].replace('.aag', '') + "_partial_CTI.txt", "w") as text_file:
            for cti in cti_lst_partial:
                text_file.write(cubeliteral_to_str(cti.cubeLiterals) + "\n")

    #@profile
    def recBlockCube(self, wrapper):
        '''
        :param s0: CTI (counterexample to induction, represented as cube)
        :return: Trace (cex, indicates that the system is unsafe) or None (successfully blocked)
        '''
        s0 = wrapper[0]
        model_lst_complete = wrapper[1]
        model_lst_partial = wrapper[2]
        self.export_CTI_lst(cti_lst_complete=model_lst_complete, cti_lst_partial=model_lst_partial)
        Q = PriorityQueue()
        print("recBlockCube now...")
        Q.put((s0.t, s0))
        prevFidx = None
        while not Q.empty():
            print (Q.qsize())
            s:tCube = Q.get()[1]
            if s.t == 0:
                return Q

            assert(prevFidx != 0)
            if prevFidx is not None and prevFidx == s.t-1:
                # local lemma push
                pushLemma_start_time = time.time()
                self.pushLemma(prevFidx)
                pushLemma_consuming_t = time.time() - pushLemma_start_time
                self.pushLemma_time_sum += pushLemma_consuming_t
            prevFidx = s.t
            # check Frame trivially block
            if self.frame_trivially_block(s):
                #Fmin = s.t+1
                #Fmax = len(self.frames)
                #if Fmin < Fmax:
                #    s_copy = s.clone()
                #    s_copy.t = Fmin
                #    Q.put((Fmin, s_copy)) #TODO: Open this will cause the problem in bmc check
                continue

            z = self.solveRelative(s)
            if z is None:
                sz = s.true_size()
                original_s_1 = s.clone() # For generating ground truth
                original_s_2 = s.clone() # For testing the NN-guided inductive generalization
                original_s_3 = s.clone() # For random ordering MIC
                original_s_4 = s.clone() # for calculate the hamming distance
                s_enumerate = self.generate_GT_no_enumerate(original_s_1) #Generate ground truth here
                #s_enumerate = self.generate_GT(original_s_1)
                # if self.test_IG_NN and self.NN_guide_ig_iteration > 5 and self.NN_guide_ig_success / (self.NN_guide_ig_success + self.NN_guide_ig_fail) < 0.5:
                #     self.test_IG_NN = 0
                NN_guide_start_time = time.time()
                s_NN = self.NN_guided_inductive_generalization(original_s_2,no_enumerate=True)
                if self.test_IG_NN != 0:
                    self.NN_guide_ig_passed_ratio.append(((self.NN_guide_ig_success / (self.NN_guide_ig_success + self.NN_guide_ig_fail)) * 100, self.NN_guide_ig_iteration))

                NN_guide_consuming_t = time.time() - NN_guide_start_time
                self.NN_guide_ig_time_sum += NN_guide_consuming_t

                # -------------------Random MIC Procedure--------------
                Random_MIC_start_time = time.time()
                s_random = self.Random_MIC(original_s_3)
                # Remove duplicated tcubes according to the list of tcubes.cubeLiterals
                if s_random is not None:
                    s_random = [list(t) for t in set(tuple(sorted(cube.cubeLiterals,key=lambda x: int(str(_extract(x)[0]).replace('v','')))) for cube in s_random)]
                    #s_random = [tCube(original_s_3.t, cube_lt_lst) for cube_lt_lst in s_random]
                    s_random_converter = []
                    for _ in s_random:
                        res = tCube(original_s_3.t)
                        res.cubeLiterals = _.copy()
                        s_random_converter.append(res)
                    s_random = s_random_converter
                Random_MIC_consuming_t = time.time() - Random_MIC_start_time
                self.test_random_mic_time_sum += Random_MIC_consuming_t
                # -------------------MIC Procedure---------------------
                MIC_start_time = time.time()
                s = self.MIC(s)
                self._check_MIC(s)
                print ('MIC ', sz, ' --> ', s.true_size(),  'F', s.t)
                MIC_consuming_t = time.time() - MIC_start_time
                self.MIC_time_sum += MIC_consuming_t
                self.sum_MIC = self.sum_MIC + s.true_size()
                # Append MIC to history appended clauses
                s_sorted = s.clone_and_sort()
                self.history_append_cex.append(s_sorted)
                self.frames[s.t].add(Not(s.cube()), pushed=False)
                for i in range(1, s.t):
                    self.frames[i].add(Not(s.cube()), pushed=True) #TODO: Try RL here


                if s_NN is not None:
                    self.NN_attempt_fail = 0
                    print ("NN-guide method find minimum", sz,' --> ', s_NN.true_size(),  'F', s_NN.t)
                    #if not((len(s_NN.cubeLiterals) == len(s.cubeLiterals)) and (len(s_NN.cubeLiterals) == sum([1 for i, j in zip(s_NN.cubeLiterals, s.cubeLiterals) if i == j]))):
                    if not (collections.Counter(s_NN.cubeLiterals) == collections.Counter(s.cubeLiterals)):
                        #assert(s_random[min_dis_cost_h_index] not in self.history_append_cex)
                        if s_NN not in self.history_append_cex:
                            self.frames[s_NN.t].add(Not(s_NN.cube()), pushed=False)
                            for i in range(1, s_NN.t):
                                self.frames[i].add(Not(s_NN.cube()), pushed=True)
                            s_NN_sorted = s_NN.clone_and_sort()
                            self.history_append_cex.append(s_NN_sorted)
                elif s_NN is None:
                    self.NN_attempt_fail += 1
                    print ("NN-guide method failed")
                    if self.NN_attempt_fail == 3 and round(self.prediction_threshold,1) > 0.5:
                        self.prediction_threshold -= 0.1
                        self.NN_attempt_fail -= 1

                if s_random is not None:
                    # Fetch the qnew at first (the standard answer of this generalization)
                    file_path_prefix = "/data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/output/tip/"
                    file_suffix = self.filename.split('/')[-1].replace('.aag', '')
                    inv_cnf = file_path_prefix + file_suffix + "/inv.cnf"
                    with open(inv_cnf, 'r') as f:
                        lines = f.readlines()
                        f.close()
                    lines = [(line.strip()).split() for line in lines]
                    q_list_cnf = [str(_extract(literal)[0]).replace('v','') if _extract(literal)[1] == True else str(int(str(_extract(literal)[0]).replace('v',''))+1) for literal in original_s_4.cubeLiterals]
                    
                    found_subset = 0
                    for clause in lines[1:]: #scan every clause
                        if(all(x in q_list_cnf for x in clause)): #the clause is subset of q
                            found_subset = 1
                            qnew = tCube(original_s_4.t)
                            ref_lst = [x in clause for x in q_list_cnf]
                            qnew.cubeLiterals = [original_s_4.cubeLiterals[i] for i in range(len(original_s_4.cubeLiterals)) if ref_lst[i] == True]
                    assert(found_subset == 1)
                    
                    # Find the min/max cost of hamming distance
                    min_dis_cost_h = hamming([1 if i in qnew.cubeLiterals else 0 for i in original_s_4.cubeLiterals], [1 if i in s_random[0].cubeLiterals else 0 for i in original_s_4.cubeLiterals])
                    min_dis_cost_h_index = 0
                    max_dis_cost_g = hamming([1 if i in s.cubeLiterals else 0 for i in original_s_4.cubeLiterals], [1 if i in s_random[0].cubeLiterals else 0 for i in original_s_4.cubeLiterals])
                    max_dis_cost_g_index = 0
                    for idx, s_random_tcube in enumerate(s_random):
                        if not (collections.Counter(s_random_tcube.cubeLiterals) == collections.Counter(s.cubeLiterals)):
                            print("Found different generalization by random ordering MIC")
                            s_one_hot_endcode = [1 if i in s.cubeLiterals else 0 for i in original_s_4.cubeLiterals]
                            s_random_one_hot_endcode = [1 if i in s_random_tcube.cubeLiterals else 0 for i in original_s_4.cubeLiterals]
                            s_inv_one_hot_encode = [1 if i in qnew.cubeLiterals else 0 for i in original_s_4.cubeLiterals]
                            dis_cost_g = hamming(s_one_hot_endcode, s_random_one_hot_endcode)
                            dis_cost_h = hamming(s_inv_one_hot_encode, s_random_one_hot_endcode)
                            #min_fn = dis_cost_g + dis_cost_h
                            if dis_cost_h <= min_dis_cost_h: 
                                min_dis_cost_h = dis_cost_h
                                min_dis_cost_h_index = idx
                            if dis_cost_g >= max_dis_cost_g:
                                max_dis_cost_g = dis_cost_h
                                max_dis_cost_g_index = idx
                    # print the min/max distance clauses
                    print("Min cost h:", min_dis_cost_h, "at index:", min_dis_cost_h_index)
                    print("Max cost g:", max_dis_cost_g, "at index:", max_dis_cost_g_index)
                    #push this to the frame
                    if(0<min_dis_cost_h<=0.1):
                        #assert(s_random[min_dis_cost_h_index] not in self.history_append_cex)
                        if s_random[min_dis_cost_h_index] not in self.history_append_cex:
                            self.frames[s_random[min_dis_cost_h_index].t].add(Not(s_random[min_dis_cost_h_index].cube()), pushed=False)
                            for i in range(1, s_random[min_dis_cost_h_index].t):
                                self.frames[i].add(Not(s_random[min_dis_cost_h_index].cube()), pushed=True)
                            s_random_min_sorted = s_random[min_dis_cost_h_index].clone_and_sort()
                            c.append(s_random_min_sorted)
                    if(max_dis_cost_g>=0.5):
                        #assert(s_random[max_dis_cost_g_index] not in self.history_append_cex)
                        if s_random[max_dis_cost_g_index] not in self.history_append_cex:
                            self.frames[s_random[max_dis_cost_g_index].t].add(Not(s_random[max_dis_cost_g_index].cube()), pushed=False)
                            for i in range(1, s_random[max_dis_cost_g_index].t):
                                self.frames[i].add(Not(s_random[max_dis_cost_g_index].cube()), pushed=True)
                            s_random_max_sorted = s_random[max_dis_cost_g_index].clone_and_sort()
                            self.history_append_cex.append(s_random_max_sorted)

                #-------------Append the NN-generated cube and MIC to frames------------- 
                # else: 
                #     print("NN-guided inductive generalization failed, begin MIC process")
                #     MIC_start_time = time.time()
                #     s = self.MIC(s)
                #     self._check_MIC(s)
                #     print ('MIC ', sz, ' --> ', s.true_size(),  'F', s.t)
                #     MIC_consuming_t = time.time() - MIC_start_time
                #     self.MIC_time_sum += MIC_consuming_t
                #     self.sum_MIC = self.sum_MIC + s.true_size()
                #     self.frames[s.t].add(Not(s.cube()), pushed=False)
                #     for i in range(1, s.t):
                #         self.frames[i].add(Not(s.cube()), pushed=True) #TODO: Try RL here

                if s_enumerate is not None: 
                    print ("Enueration find minimum", sz,' --> ', s_enumerate.true_size(),  'F', s_enumerate.t)
                    self.sum_IG_GT = self.sum_IG_GT + s_enumerate.true_size()
                else:
                    print ("Minimum not found by enueration")
                    self.sum_IG_GT = self.sum_IG_GT + s.true_size()


                '''
                Determine whether use MIC
                
                self._check_MIC(s)
                self.frames[s.t].add(Not(s.cube()), pushed=False)
                for i in range(1, s.t):
                    self.frames[i].add(Not(s.cube()), pushed=True) #TODO: Try RL here
                '''

                '''
                Only Use NN-guided IG
                
                self._check_MIC(s_NN)
                self.frames[s_NN.t].add(Not(s_NN.cube()), pushed=False)
                for i in range(1, s_NN.t):
                    self.frames[i].add(Not(s_NN.cube()), pushed=True)
                '''


                '''
                Add NN-guided inductive generalization generated answer
                
                #Use unsat core reduce
                if (s_NN is not None) and (self.NN_guide_ig_append!=0): 
                    original_s_3 = s_NN.clone()
                    original_s_4 = s.clone()
                    #self.unsatcore_reduce(original_s_3, trans=self.trans.cube(), frame=self.frames[original_s_3.t-1].cube())
                    #original_s_3.remove_true()
                    #if not((len(s_NN.cubeLiterals)== len(s.cubeLiterals)) and (len(s_NN.cubeLiterals) == sum([1 for i, j in zip(s_NN.cubeLiterals, s.cubeLiterals) if i == j]))):   
                    original_s_3.cubeLiterals.sort(key=lambda x: str(_extract(x)[0]))
                    original_s_4.cubeLiterals.sort(key=lambda x: str(_extract(x)[0]))
                    if not(original_s_3.cubeLiterals == original_s_4.cubeLiterals): 
                        self.frames[s_NN.t].add(Not(s_NN.cube()), pushed=False)
                        for i in range(1, s.t):
                            self.frames[i].add(Not(s_NN.cube()), pushed=True)
                '''


                        #Not use unsat core reduce
                        # if (s_NN is not None) and (self.NN_guide_ig_append!=0): 
                        #     self.frames[s.t].add(Not(s_NN.cube()), pushed=False)
                        #     for i in range(1, s.t):
                        #         self.frames[i].add(Not(s_NN.cube()), pushed=True)
                        

                        # reQueue : see IC3 PDR Friends
                        #Fmin = original_s.t+1
                        #Fmax = len(self.frames)
                        #if Fmin < Fmax:
                        #    s_copy = original_s #s.clone()
                        #    s_copy.t = Fmin
                        #    Q.put((Fmin, s_copy))

            else: #SAT condition
                assert(z.t == s.t-1)
                Q.put((s.t, s))
                Q.put((s.t-1, z))
        return None


    # def recBlockCube_RL(self, s0: tCube):
    #     print("recBlockCube now...")
    #     Q = [s0]
    #     while len(Q) > 0:
    #         s = Q[-1]
    #         if s.t == 0:
    #             return Q

    #         # solve if cube s was blocked by the image of the frame before it
    #         z, u = self.solveRelative_RL(s)

    #         if (z == None):
    #             # Cube 's' was blocked by image of predecessor:
    #             # block cube in all previous frames
    #             Q.pop()  # remove cube s from Q
    #             for i in range(1, s.t + 1):
    #                 # if not self.isBlocked(s, i):
    #                 self.R[i] = And(self.R[i], Not(u))
    #         else:
    #             # Cube 's' was not blocked by image of predecessor
    #             # it will stay on the stack, and z (the model which allowed transition to s) will we added on top
    #             Q.append(z)
    #     return None

    def _solveRelative(self, tcube) -> tCube:
        '''
        #FIXME: The inductive relative checking should subtitue input -> input'
        '''
        #cubePrime = substitute(tcube.cube(), self.primeMap)
        cubePrime = substitute(substitute(tcube.cube(), self.primeMap),self.inp_map)
        s = Solver()
        s.add(Not(tcube.cube()))
        s.add(self.frames[tcube.t - 1].cube())
        s.add(self.trans.cube())
        s.add(cubePrime)  # F[i - 1] and T and Not(badCube) and badCube'
        return s.check()

    def _solveRelative_upgrade(self, tcube) -> tCube:
        check_init = sat
        slv = Solver()
        slv.add(self.init.cube())
        slv.add(tcube.cube())
        check_init = slv.check()

        check_relative = sat
        cubePrime = substitute(substitute(tcube.cube(), self.primeMap),self.inp_map)
        s = Solver()
        s.add(Not(tcube.cube()))
        s.add(self.frames[tcube.t - 1].cube())
        s.add(self.trans.cube())
        s.add(cubePrime)  # F[i - 1] and T and Not(badCube) and badCube'
        check_relative = s.check()

        if check_init == unsat and check_relative == unsat:
            return 'pass the check'
        else:
            return 'not pass'

    def _test_MIC1(self, q: tCube):
        passed_single_q = []
        for i in range(len(q.cubeLiterals)):
            qnew = tCube(q.t)
            var, val = _extract(q.cubeLiterals[i])
            if str(val) != "True": # will intersect with init: THIS IS DIRTY check
                continue
            qnew.cubeLiterals = [q.cubeLiterals[i]]
            if self._solveRelative(qnew) == unsat:
                passed_single_q.append(qnew)
        return passed_single_q

    def _test_MIC2(self, q: tCube):
        def check_init(c: tCube):
            slv = Solver()
            slv.add(self.init.cube())
            slv.add(c.cube())
            return slv.check()

        passed_single_q = []
        for i in range(len(q.cubeLiterals)):
            for j in range(i+1, len(q.cubeLiterals)):
                qnew = tCube(q.t)
                qnew.cubeLiterals = [q.cubeLiterals[i], q.cubeLiterals[j]]
                if check_init(qnew) == sat:
                    continue
                if self._solveRelative(qnew) == unsat:
                    passed_single_q.append(qnew)
        return passed_single_q

    def Random_MIC(self, q: tCube):
        if self.test_mic == 0:
            return None
        elif self.test_mic == 1:
            q_lst = [q.clone(), q.clone(),q.clone(), q.clone(), q.clone(), q.clone(), q.clone(), q.clone(), q.clone(), q.clone()]
            for q in q_lst:
                sz = q.true_size()
                self.unsatcore_reduce(q, trans=self.trans.cube(), frame=self.frames[q.t-1].cube())
                print('unsatcore', sz, ' --> ', q.true_size())
                q.remove_true()
                random_cubeliterals = [x for x in q.cubeLiterals]
                random.shuffle(random_cubeliterals)
                q.cubeLiterals = random_cubeliterals
                for i in range(len(q.cubeLiterals)):
                    q1 = q.delete(i)
                    print(f'MIC try idx:{i}')
                    if self.down(q1): 
                        q = q1
                q.remove_true()
                print (q)
            return q_lst
        

    def MIC(self, q: tCube): #TODO: Check the algorithm is correct or not
        #passed_single_q_sz1 = self._test_MIC1(q)
        #passed_single_q_sz2 = []
        #if len(passed_single_q_sz1) == 0:
        #    passed_single_q_sz2 = self._test_MIC2(q)

        sz = q.true_size()
        self.unsatcore_reduce(q, trans=self.trans.cube(), frame=self.frames[q.t-1].cube())
        print('unsatcore', sz, ' --> ', q.true_size())
        q.remove_true()

        for i in range(len(q.cubeLiterals)):
            if q.cubeLiterals[i] is True: #This true does not indicate the literals are true
                continue
            q1 = q.delete(i)
            print(f'MIC try idx:{i}')
            if self.down(q1): 
                q = q1
        q.remove_true()
        print (q)
        # FIXME: below shows the choice of var is rather important
        # I think you may want to first run some experience to confirm
        # that if can achieve minimum, it will be rather useful
        # if q.true_size() > 1 and len(passed_single_q_sz1) != 0:
        #     q = passed_single_q_sz1[0] # should be changed!
        #     print ('Not optimal!!!')
        # if q.true_size() > 2 and len(passed_single_q_sz2) != 0:
        #     for newq in passed_single_q_sz2:
        #         if 'False' in str(newq): #Ask this, why is 'False' in str(newq)?
        #             q = newq
        #     # should be changed!
        #     print ('Not optimal!!!')
        return q
        # i = 0
        # while True:
        #     print(i)
        #     if i < len(q.cubeLiterals) - 1:
        #         i += 1
        #     else:
        #         break
        #     q1 = q.delete(i)
        #     if self.down(q1):
        #         q = q1
        # return q
    
    #TODO: Add assertion on this to check inductive relative
    #TODO: Add assertion to check there is no 'True' and 'False' in the cubeLiterals list
    def generate_GT(self,q: tCube): #smt2_gen_IG is a switch to trun on/off .smt file generation
        
        if self.smt2_gen_IG == 0:
            return None
        elif self.smt2_gen_IG == 1:
            assert(q.cubeLiterals.count(True)==0)
            assert(q.cubeLiterals.count(False)==0)
            '''
            ---------------------Generate .smt2 file (for building graph)--------------
            
            
            #FIXME: This .smt generation still exists problem, remember to fix this
            s_smt = Solver()  #use to generate SMT-lib2 file

            #This "Cube" is a speical circuit of combining two conditions of solve relative (determine inductive generalization)

            # s_smt.add(Not(q.cube()))
            # s_smt.add(self.frames[q.t - 1].cube())
            # s_smt.add(self.trans.cube())
            # s_smt.add(substitute(substitute(q.cube(), self.primeMap),self.inp_map))
            
            Cube = Not(
                And(
                    Not(
                      And(self.frames[q.t-1].cube(), Not(q.cube()), self.trans.cube_remove_equal().cube(),
                      substitute(substitute(substitute(q.cube(), self.primeMap),self.inp_map),list(self.pv2next.items())))
                      #substitute(q.cube(), self.primeMap))
                      )  # Fi-1 ! and not(q) and T and q'
                ,
                    Not(And(self.frames[0].cube(),q.cube()))
                    )
            )

            #Cube = substitute(Cube,list(self.pv2next.items()))

            for index, literals in enumerate(q.cubeLiterals): 
                s_smt.add(literals) 
                # s_smt.assert_and_track(literals,'p'+str(index))
            
            s_smt.add(Cube)  # F[i - 1] and T and Not(badCube) and badCube'

            assert (s_smt.check() == unsat)

            filename = '../dataset/IG2graph/generalize_IG/' + (self.filename.split('/')[-1]).replace('.aag', '_'+ str(len(self.generaliztion_data_IG)) +'.smt2')
            with open(filename, mode='w') as f:
                f.write(s_smt.to_smt2())
            f.close()
            '''
            

            '''
            -------------------Generate ground truth--------------
            '''

            def check_init(c: tCube):
                slv = Solver()
                slv.add(self.init.cube())
                slv.add(c.cube())
                return slv.check()

            # sz = q.true_size()
            # self.unsatcore_reduce(q, trans=self.trans.cube(), frame=self.frames[q.t-1].cube())
            # print('unsatcore', sz, ' --> ', q.true_size())
            # q.remove_true()

            end_lst = []
            passed_minimum_q = []
            is_looping = True
            for i in range(1,len(q.cubeLiterals)+1): #When i==len(q.cubeLiterals), this means it met wrost case
                for c in combinations(q.cubeLiterals, i):
                    if len(end_lst) > 3000:
                        is_looping = False
                        break
                    end_lst.append(c)
                if is_looping==False:
                    break

            #FIXME: This may cause memory exploration of list (length 2^n, n is the length of original q)

            '''
            1 -> 0
            2 -> Cn1
            3 -> Cn1+Cn2
            4 -> Cn1+Cn2+Cn3
            5 -> Cn1+Cn2+Cn3+Cn4
            ...
            n -> Cn1+Cn2+Cn3+Cn4+...+Cnn -> 2^n - 1 
            '''
            
            #TODO: Using multi-thread to handle inductive relative checking
            # dict_n = {}
            # dict_n[1] = 0
            # dict_n[2] = int(comb(len(end_lst),1))
            # dict_n[3] = int(comb(len(end_lst),1) + comb(len(end_lst),2))
            # dict_n[4] = int(comb(len(end_lst),1) + comb(len(end_lst),2) \
            #     + comb(len(end_lst),2)+comb(len(end_lst),3))
            
            data = {} # Store ground truth, and output to .csv
            for tuble in end_lst:
                if len(passed_minimum_q) > 0:
                    break
                elif len(passed_minimum_q) == 0:
                    qnew = tCube(q.t)
                    qnew.cubeLiterals = [tcube for tcube in tuble]
                    if check_init(qnew) == sat:
                        continue
                    # if self._solveRelative(qnew) == sat:
                    #     print("Did not pass inductive relative check")
                    #     continue
                    if self._solveRelative(qnew) == unsat:
                        passed_minimum_q.append(qnew)
                else:
                    raise AssertionError
                #ADD: When len(passed_single_q) != 0, break the for loop
            if len(passed_minimum_q)!= 0:
                '''
                ---------------------Generate .smt2 file (for building graph)--------------

                Not generate the .smt2 file when enumerate combinations of literals could not find ground truth
                '''
                
                s_smt = Solver()
                Cube = Not(
                    And(
                        Not(
                        And(self.frames[q.t-1].cube(), 
                        Not(q.cube()), 
                        substitute(substitute(substitute(q.cube(), self.primeMap),self.inp_map),
                        list(self.pv2next.items()))
                        )),
                        Not(And(self.frames[0].cube(),q.cube()))
                        ))
                for index, literals in enumerate(q.cubeLiterals): s_smt.add(literals)
                s_smt.add(Cube)
                assert (s_smt.check() == unsat)
                filename = '../dataset/IG2graph/generalize_IG/' + (self.filename.split('/')[-1]).replace('.aag', '_'+ str(len(self.generaliztion_data_IG)) +'.smt2')
                data['inductive_check'] = filename.split('/')[-1] #Store the name of .smt file
                with open(filename, mode='w') as f: f.write(s_smt.to_smt2())
                f.close() 
                

                '''
                ---------------------Export the ground truth----------------------
                '''
                q_minimum = passed_minimum_q[0] # Minimum ground truth has been generated
                for idx in range(len(q.cubeLiterals)): # -> ground truth size is q
                    var, val = _extract(q.cubeLiterals[idx])
                    data[str(var)] = 0
                # for idx in range(len(Cube.cubeLiterals)): # -> ground truth size is Cube (combine of two check)
                #     var, val = _extract(Cube.cubeLiterals[idx])
                #     data[str(var)] = 0
                for idx in range(len(q_minimum.cubeLiterals)):
                    var, val = _extract(q_minimum.cubeLiterals[idx])
                    data[str(var)] = 1 # Mark q-like as 1
                self.generaliztion_data_IG.append(data)
                return q_minimum
            else:
                print("The ground truth has not been found")
                return None

    def generate_GT_no_enumerate(self,q: tCube): #smt2_gen_IG is a switch to trun on/off .smt file generation 
        if self.smt2_gen_IG == 0:
            return None
        elif self.smt2_gen_IG == 1:
            assert(q.cubeLiterals.count(True)==0)
            assert(q.cubeLiterals.count(False)==0)

            def check_init(c: tCube):
                slv = Solver()
                slv.add(self.init.cube())
                slv.add(c.cube())
                return slv.check()

            file_path_prefix = "/data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/output/tip/"
            #file_path_prefix = "/data/guangyuh/coding_env/IC3ref_zhang/"
            file_suffix = self.filename.split('/')[-1].replace('.aag', '')
            inv_cnf = file_path_prefix + file_suffix + "/inv.cnf"
            with open(inv_cnf, 'r') as f:
                lines = f.readlines()
                f.close()
            lines = [(line.strip()).split() for line in lines]
            q_list_cnf = [str(_extract(literal)[0]).replace('v','') if _extract(literal)[1] == True else str(int(str(_extract(literal)[0]).replace('v',''))+1) for literal in q.cubeLiterals]
            
            data = {} # Store ground truth, and output to .csv
            passed_minimum_q = []
            found_subset_flag = 0
            for clause in lines[1:]: #scan every clause
                if(all(x in q_list_cnf for x in clause)): #the clause is subset of q
                    found_subset_flag = 1
                    qnew = tCube(q.t)
                    ref_lst = [x in clause for x in q_list_cnf]
                    qnew.cubeLiterals = [q.cubeLiterals[i] for i in range(len(q.cubeLiterals)) if ref_lst[i] == True]
                    if check_init(qnew) == unsat and self._solveRelative(qnew) == unsat:
                        passed_minimum_q.append(qnew)
            
            assert(found_subset_flag == 1)
                    
            #q_list_cnf = [_extract(literals)[1]==True?_extract(literals)[0]:Not(_extract(literals)[0]) for literals in q.cubeLiterals]


            if len(passed_minimum_q)!= 0:
                '''
                ---------------------Generate .smt2 file (for building graph)--------------

                Not generate the .smt2 file when enumerate combinations of literals could not find ground truth
                '''
                
                s_smt = Solver()
                Cube = Not(
                    And(
                        Not(q.cube()), 
                        substitute(substitute(substitute(q.cube(), self.primeMap),self.inp_map),
                        list(self.pv2next.items()))
                        ))
                for index, literals in enumerate(q.cubeLiterals): s_smt.add(literals)
                s_smt.add(Cube)
                #assert (s_smt.check() == unsat)
                filename = '../dataset/IG2graph/generalize_IG_no_enumerate/' + (self.filename.split('/')[-1]).replace('.aag', '_'+ str(len(self.generaliztion_data_IG)) +'.smt2')
                data['inductive_check'] = filename.split('/')[-1] #Store the name of .smt file
                with open(filename, mode='w') as f: f.write(s_smt.to_smt2())
                f.close() 
                

                '''
                ---------------------Export the ground truth----------------------
                '''
                q_minimum = passed_minimum_q[0] # Minimum ground truth has been generated
                for idx in range(len(q.cubeLiterals)): # -> ground truth size is q
                    var, val = _extract(q.cubeLiterals[idx])
                    data[str(var)] = 0
                for idx in range(len(q_minimum.cubeLiterals)):
                    var, val = _extract(q_minimum.cubeLiterals[idx])
                    data[str(var)] = 1 # Mark q-like as 1
                self.generaliztion_data_IG.append(data)
                return q_minimum
            else:
                print("The ground truth has not been found")
                return None

    def parse_prob(self,prob,device):
        prob_main_info = {
            'n_vars' : prob.n_vars,
            'n_nodes' : prob.n_nodes,
            'unpack' : (torch.from_numpy(prob.adj_matrix.astype(np.float32).values)).to(device),
            'refined_output' : prob.refined_output
        }
        dict_vt = dict(zip((prob.value_table).index, (prob.value_table).Value))
        return prob_main_info, dict_vt
    
    def NN_guided_inductive_generalization(self, q: tCube, no_enumerate=False):
        '''
        Test the NN-version inductive generalization
        '''
        self.NN_guide_ig_iteration += 1

        # Generate a backup of the original frames -> using unsat core reduce method to reduce
        q4unsatcore = q.clone()
        self.unsatcore_reduce(q4unsatcore, trans=self.trans.cube(), frame=self.frames[q4unsatcore.t-1].cube())
        q4unsatcore.remove_true()

        if self.test_IG_NN == 0:
            return None
            #pass
        elif self.test_IG_NN == 1 and self.NN_attempt_fail <=3:
            # s_smt = Solver()
            # Cube = Not(
            #     And(
            #         Not(
            #         And(self.frames[q.t-1].cube(), 
            #         Not(q.cube()), 
            #         substitute(substitute(substitute(q.cube(), self.primeMap),self.inp_map),
            #         list(self.pv2next.items()))
            #         )),
            #         Not(And(self.frames[0].cube(),q.cube()))
            #         ))
            # for index, literals in enumerate(q.cubeLiterals): s_smt.add(literals)
            # s_smt.add(Cube)
            # assert (s_smt.check() == unsat)
            # res = build_graph_online.run(s_smt,self.filename,self.test_IG_NN+1) #-> this is a list to guide which literals should be kept/throwed
            # # Conductive two relative check of the return q-like
            # print('restoring from: ', "../dataset/model/neuropdr_2022-06-07_06:31:22_last_copy.pth.tar")
            # # Load model to predict
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # net = neuro_predessor.NeuroPredessor()
            # model = torch.load("../model/neuropdr_2022-06-07_06:31:22_last_copy.pth.tar")
            # net.load_state_dict(model['state_dict'])
            # net = net.to(device)
            # sigmoid  = nn.Sigmoid()
            # torch.no_grad()

            if no_enumerate == False:
                s_smt = Solver()
                Cube = Not(
                    And(
                        Not(
                        And(self.frames[q.t-1].cube(), 
                        Not(q.cube()), 
                        substitute(substitute(substitute(q.cube(), self.primeMap),self.inp_map),
                        list(self.pv2next.items()))
                        )),
                        Not(And(self.frames[0].cube(),q.cube()))
                        ))
                for index, literals in enumerate(q.cubeLiterals): s_smt.add(literals)
                s_smt.add(Cube)
                assert (s_smt.check() == unsat)
                res = build_graph_online.run(s_smt,self.filename,self.test_IG_NN+1) #-> this is a list to guide which literals should be kept/throwed
                # Conductive two relative check of the return q-like
                print('restoring from: ', "../dataset/model/"+self.model_name+".pth.tar")
                # Load model to predict
                #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if self.inf_device == 'gpu':
                    device = torch.device("cuda")
                elif self.inf_device == 'cpu':
                    device = torch.device("cpu")
                net = neuro_ig_no_enumerate.NeuroPredessor()
                model = torch.load("../model/"+self.model_name+".pth.tar",map_location=device)
                net.load_state_dict(model['state_dict'])
                net = net.to(device)
                if torch.cuda.device_count >1:
                    net = nn.DataParallel(net,device_ids=[0,1])
                sigmoid  = nn.Sigmoid()
                torch.no_grad()
                #q_index = extract_q_like(res)
                q_index = []
                tmp_lst_all_node = res.value_table.index.to_list()[res.n_nodes:]
                ig_q = res.ig_q # original q in inductive generalization
                for q_literal in ig_q: # literals in q (in inductive generalization process)
                    q_index.append(tmp_lst_all_node.index('n_'+str(q_literal.children()[0])))

                q_index.sort() # Fixed the bug of indexing the correct literals
                res = self.parse_prob(res)
                outputs = sigmoid(net(res))
                torch_select = torch.Tensor(q_index).to(device).int() 
                outputs = torch.index_select(outputs, 0, torch_select)
                top_k_outputs = list(sorted(enumerate(outputs.tolist()), key = itemgetter(1)))[-2:]
                preds = torch.where(outputs>0.997, torch.ones(outputs.shape).to(device), torch.zeros(outputs.shape).to(device))

                '''
                Generate the new q (which is also q-like) under the NN-given answer
                '''
                q.cubeLiterals.sort(key=lambda x: str(_extract(x)[0]))
                q_like = tCube(q.t)
                for idx, preds_ans in enumerate(preds.tolist()):
                    if preds_ans == 1:
                        for top_outputs in top_k_outputs:
                            if top_outputs[0] == idx: q_like.cubeLiterals.append(q.cubeLiterals[idx])
            elif no_enumerate == True:
                s_smt = Solver()
                Cube = Not(
                    And(
                        Not(q.cube()), 
                        substitute(substitute(substitute(q.cube(), self.primeMap),self.inp_map),
                        list(self.pv2next.items()))
                        ))
                for index, literals in enumerate(q.cubeLiterals): s_smt.add(literals)
                s_smt.add(Cube)
                # store the key in self.pv2next to list
                latch_lst = [str(key).replace('_prime','') for key in self.pv2next.keys()]
                res = build_graph_online.run(s_smt,self.filename,mode=2,latch_lst=latch_lst) #-> this is a list to guide which literals should be kept/throwed
                # Conductive two relative check of the return q-like
                print('restoring from: ', "../dataset/model/"+self.model_name+".pth.tar")
                # Load model to predict
                # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if self.inf_device == 'gpu':
                    device = torch.device("cuda")
                elif self.inf_device == 'cpu':
                    device = torch.device("cpu")
                args = parser.parse_args(['--dim', '128', '--n_rounds', '512','--inf_dev',self.inf_device])
                net = neuro_ig_no_enumerate.NeuroPredessor(args)
                model = torch.load("../model/"+self.model_name+".pth.tar",map_location=device)
                net.load_state_dict(model['state_dict'])
                net = net.to(device)
                sigmoid  = nn.Sigmoid()
                torch.no_grad()
                tmp_lst_all_node = res.value_table.index.to_list()[res.n_nodes:]
                ig_q = res.ig_q # original q in inductive generalization

                single_node_index = []  # store the index
                if isinstance(res.db_gt, list):
                    var_list = res.db_gt
                else:
                    var_list = list(res.db_gt)
                    var_list.pop(0)  # remove "filename_nextcube"
                
                tmp = res.value_table[~res.value_table.index.str.contains('m_')]
                tmp.index = tmp.index.str.replace("n_", "")

                for i, element in enumerate(var_list):
                    if element not in tmp.index.tolist():
                        single_node_index.append(i)

                '''
                now try to refine the output 
                '''
                var_index = [] # Store the index that is in the graph and in the ground truth table
                q_index = []
                #tmp_lst_var = list(res.db_gt)[1:]

                # The groud truth we need to focus on
                #focus_gt = [e[1] for e in enumerate(tmp_lst_var) if e[0] not in single_node_index]
                # The q that we need to focus on 
                focus_q = [_extract(e[1])[0] for e in enumerate(ig_q)]
                # Try to fetch the index of the variable in the value table (variable in db_gt)
                tmp_lst_all_node = res.value_table.index.to_list()[res.n_nodes:]
                for element in focus_q:
                    #var_index.append(tmp_lst_all_node.index('n_'+str(element)))
                    q_index.append(tmp_lst_all_node.index('n_'+str(element)))
                #res.refined_output = var_index
                #res.refined_output = q_index
                
                #q_index = var_index
                q_index.sort() # Fixed the bug of indexing the correct literals
                res,vt_dict = self.parse_prob(res,device)
                outputs = sigmoid(net((res,vt_dict)))
                torch_select = torch.Tensor(q_index).to(device).int() 
                outputs = torch.index_select(outputs, 0, torch_select)
                top_k_outputs = list(sorted(enumerate(outputs.tolist()), key = itemgetter(1)))[:]
                preds = torch.where(outputs>self.prediction_threshold, torch.ones(outputs.shape).to(device), torch.zeros(outputs.shape).to(device))
                '''
                Generate the new q (which is also q-like) under the NN-given answer
                '''
                q.cubeLiterals.sort(key=lambda x: str(_extract(x)[0]))
                q_like = tCube(q.t)
                for idx, preds_ans in enumerate(preds.tolist()):
                    if preds_ans == 1:
                        q_like.cubeLiterals.append(q.cubeLiterals[idx])
            
            '''
            Check whether this answer pass the inductive relative check
            ''' 
            def check_init(c: tCube):
                    slv = Solver()
                    slv.add(self.init.cube())
                    slv.add(c.cube())
                    return slv.check()

            if len(q_like.cubeLiterals)!= 0:
                if check_init(q_like) == unsat:
                    s = Solver()
                    s.add(And(self.frames[q_like.t-1].cube(), Not(q_like.cube()), self.trans.cube(), 
                                substitute(substitute(q_like.cube(), self.primeMap),self.inp_map)))  
                    if s.check() == unsat:
                        # Pass both check
                        print("Congratulation, the NN-guide inductive generalization is correct")
                        self.NN_guide_ig_success += 1
                        return q_like
                        # if len(q_like.cubeLiterals) > len(q4unsatcore.cubeLiterals) + 1:
                        #     return q4unsatcore
                        # else:
                        #     return q_like
                    else:
                        # Not pass the second check
                        self.NN_guide_ig_fail += 1
                        return None
                else:
                    # Not pass the first check
                    self.NN_guide_ig_fail += 1
                    return None
            else:
                self.NN_guide_ig_fail += 1
                return None

        

            
    def unsatcore_reduce(self, q:  tCube, trans, frame):
        # (( not(q) /\ F /\ T ) \/ init' ) /\ q'   is unsat
        slv = Solver()
        slv.set(unsat_core=True)

        l = Or( And(Not(q.cube()), trans, frame), self.initprime)
        slv.add(l)

        plist = []
        for idx, literal in enumerate(q.cubeLiterals):
            p = 'p'+str(idx)
            slv.assert_and_track(substitute(substitute(literal, self.primeMap),self.inp_map), p)
            plist.append(p)
        res = slv.check()
        if res == sat:
            model = slv.model()
            print(model.eval(self.initprime))
            assert False
        assert (res == unsat)
        core = slv.unsat_core()
        for idx, p in enumerate(plist):
            if Bool(p) not in core:
                q.cubeLiterals[idx] = True
        return q


    def down(self, q: tCube):
        while True:
            print(q.true_size(), end=',')
            s = Solver()
            s.push()
            #s.add(And(self.frames[0].cube(), Not(q.cube())))
            s.add(self.frames[0].cube())
            s.add(q.cube())
            #if unsat == s.check():
            if sat == s.check():
                print('F')
                return False
            s.pop()
            s.push()
            s.add(And(self.frames[q.t-1].cube(), Not(q.cube()), self.trans.cube(), #TODO: Check here is t-1 or t
                      substitute(substitute(q.cube(), self.primeMap),self.inp_map)))  # Fi-1 ! and not(q) and T and q'
            if unsat == s.check():
                print('T')
                return True
            # TODO: this is not the down process !!!
            m = s.model()
            has_removed = q.join(m)
            s.pop()
            assert (has_removed)
            #return False

    # def tcgMIC(self, q: tCube, d: int):
    #     for i in range(len(q.cubeLiterals)):
    #         q1 = q.delete(i)
    #         if self.ctgDown(q1, d):
    #             q = q1
    #     return q
    #
    # def ctgDown(self, q: tCube, d: int):
    #     ctgs = 0
    #     while True:
    #         s = Solver()
    #         s.push()
    #         s.add(And(self.R[0].cube(), Not(q.cube())))
    #         if unsat == s.check():
    #             return False
    #         s.pop()
    #         s.push()
    #         s.add(And(self.R[q.t].cube(), Not(q.cube()), self.trans.cube(),
    #                   substitute(q.cube(), self.primeMap)))  # Fi and not(q) and T and q'
    #         if unsat == s.check():
    #             return True
    #         m = s.model()

    def _debug_print_frame(self, fidx, skip_pushed=False):
        for idx, c in enumerate(self.frames[fidx].Lemma):
            if skip_pushed and self.frames[fidx].pushed[idx]:
                continue
            if 'i' in str(c):
                print('C', idx, ':', 'property')
            else:
                print('C', idx, ':', str(c))


    def _debug_c_is_predecessor(self, c, t, f, not_cp):
        s = Solver()
        s.add(c)
        s.add(t)
        if f is not True:
            s.add(f)
        s.add(not_cp)
        assert (s.check() == unsat)

    # tcube is bad state

    def _check_MIC(self, st:tCube):
        cubePrime = substitute(substitute(st.cube(), self.primeMap),self.inp_map)
        s = Solver()
        s.add(Not(st.cube()))
        s.add(self.frames[st.t - 1].cube())
        s.add(self.trans.cube())
        s.add(cubePrime)
        assert (s.check() == unsat)

    def parse_dimacs(self,filename):
        # Check this variable is string type or not
        if(isinstance(filename, list)):
            assert len(filename) == 1
            lines = [line for line in filename[0].strip().split("\n")]
            for line in lines:
                if "c" == line.strip().split(" ")[0]:
                    index_c = lines.index(line)
                    break
            header = lines[0].strip().split(" ")
            assert(header[0] == "p")
            n_vars = int(header[2])
            iclauses = [[int(s) for s in line.strip().split(" ")[:-1]] for line in lines[1:index_c]]
            return n_vars, iclauses
        elif(isinstance(filename, str)):
            with open(filename, 'r') as f:
                lines = f.readlines()
            # Find the first index of line that contains string "c"
            for line in lines:
                if "c" == line.strip().split(" ")[0]:
                    index_c = lines.index(line)
                    break
            header = lines[0].strip().split(" ")
            assert(header[0] == "p")
            n_vars = int(header[2])
            iclauses = [[int(s) for s in line.strip().split(" ")[:-1]] for line in lines[1:index_c]]
            return n_vars, iclauses

    # for tcube, check if cube is blocked by R[t-1] AND trans (check F[i−1]/\!s/\T/\s′ is sat or not)
    def solveRelative(self, tcube) -> tCube:
        '''
        :param tcube: CTI (counterexample to induction, represented as cube)
        :return: None (relative solved! Begin to block bad state) or
        predecessor to block (Begin to enter recblock() again)
        '''
        cubePrime = substitute(substitute(tcube.cube(), self.primeMap),self.inp_map)
        s = Solver()
        s.add(Not(tcube.cube()))
        s.add(self.frames[tcube.t - 1].cube())
        s.add(self.trans.cube())
        s.add(cubePrime)  # F[i - 1] and T and Not(badCube) and badCube'
        
        # Use minisat
        # f = CNFFormula.from_z3(s.assertions())
        # cnf_string_lst = f.to_dimacs_string()
        # n, iclauses = self.parse_dimacs(cnf_string_lst)
        # minisolver = minisolvers.MinisatSolver()
        # for _ in range(n): minisolver.new_var(dvar=True)
        # for iclause in iclauses: minisolver.add_clause(iclause)
        # is_sat = minisolver.solve()
        # assert((is_sat==False and s.check()==unsat) or (is_sat==True and s.check()==sat))
        
        if s.check()==sat: # F[i-1] & !s & T & s' is sat!!
            model = s.model()
            c = tCube(tcube.t - 1)
            c.addModel(self.lMap, model, remove_input=False)  # c = sat_model, get the partial model of c
            #return c
            prev_c_length = len(c.cubeLiterals)
            # print("cube size: ", len(c.cubeLiterals), end='--->')
            # FIXME: check1 : c /\ T /\ F /\ Not(cubePrime) : unsat

            #Comment out this line to check if the cube is blocked by R[t-1]
            #self._debug_c_is_predecessor(c.cube(), self.trans.cube(), self.frames[tcube.t-1].cube(), Not(cubePrime))
            
            generalized_p = self.generalize_predecessor(c, next_cube_expr = tcube.cube(), prevF=self.frames[tcube.t-1].cube())  # c = get_predecessor(i-1, s')
            #print(len(generalized_p.cubeLiterals))
            after_c_length = len(generalized_p.cubeLiterals)
            print("cube size after generalized predecessor: ", prev_c_length, '--->', after_c_length)
            #
            # FIXME: sanity check: gp /\ T /\ F /\ Not(cubePrime)  unsat
            
            #Comment out this line to check if the cube is blocked by R[t-1]
            #self._debug_c_is_predecessor(generalized_p.cube(), self.trans.cube(), self.frames[tcube.t-1].cube(), Not(cubePrime))
            
            generalized_p.remove_input()
            return generalized_p #TODO: Using z3 eval() to conduct tenary simulation
        else:
            # Get the unsat core from s
            # s_unsat = Solver()
            # for index, literals in enumerate(tcube.cubeLiterals):
            #     s_unsat.add(literals) 
            #     s_unsat.assert_and_track(literals,'p'+str(index)) # -> ['p1','p2','p3']
            return None

    #(X ∧ 0 = 0), (X ∧ 1 = X), (X ∧ X = X), (¬X = X).
    # def ternary_operation(self, ternary_candidate):
    #     for
    #     False = And(x,True)
    #     x = And(x,True)
    #     x = And(x,x)
    #     x = Not(x)

#TODO: Get bad cude should generalize as well!
    def generalize_predecessor(self, prev_cube:tCube, next_cube_expr, prevF, smt2_gen_GP=0): #smt2_gen_GP is a switch to trun on/off .smt file generation
        '''
        :param prev_cube: sat model of CTI (v1 == xx , v2 == xx , v3 == xxx ...)
        :param next_cube_expr: bad state (or CTI), like !P ( ? /\ ? /\ ? /\ ? .....)
        :return:
        '''
        data = {}
        #data['previous cube'] = str(prev_cube)
        #data['previous cube'] = (prev_cube.cube()).sexpr()


        #check = tcube.cube()

        tcube_cp = prev_cube.clone() #TODO: Solve the z3 exception warning
        ground_true = prev_cube.clone()
        print("original size of !P (or CTI): ", len(tcube_cp.cubeLiterals))
        #print("Begin to generalize predessor")

        #replace the state as the next state (by trans) -> !P (s')
        nextcube = substitute(substitute(substitute(next_cube_expr, self.primeMap),self.inp_map), list(self.pv2next.items())) # s -> s'
        #data['nextcube'] = str(nextcube)

        # try:
        #     nextcube = substitute(substitute(next_cube_expr, self.primeMap), list(self.pv2next.items()))
        # except Exception:
        #     pass
        index_to_remove = []

        #sanity check
        #s = Solver()
        #s.add(prev_cube.cube())
        #s.check()
        #assert(str(s.model().eval(nextcube)) == 'True')
        
        if self.smt2_gen_GP==1:
            s = Solver()
            s_smt = Solver()  #use to generate SMT-lib2 file
            for index, literals in enumerate(tcube_cp.cubeLiterals):
                s_smt.add(literals) 
                s.assert_and_track(literals,'p'+str(index)) # -> ['p1','p2','p3']
            s.add(prevF)
            s.add(Not(nextcube))
            s_smt.add(prevF)  # TODO: we need to think about it, do we really need to add it? --Hongce
            s_smt.add(Not(nextcube)) 
            assert(s.check() == unsat and s_smt.check() == unsat)
            core = s.unsat_core()
            core = [str(core[i]) for i in range(0, len(core), 1)] # -> ['p1','p3'], core -> nextcube

            filename = '../dataset/GP2graph/generalize_pre/' + (self.filename.split('/')[-1]).replace('.aag', '_'+ str(len(self.generaliztion_data_GP)) +'.smt2')
            data['nextcube'] = filename.split('/')[-1]
            with open(filename, mode='w') as f:
                f.write(s_smt.to_smt2())
            f.close()

        elif self.smt2_gen_GP==0:
            s = Solver()
            for index, literals in enumerate(tcube_cp.cubeLiterals):
                s.assert_and_track(literals,'p'+str(index)) # -> ['p1','p2','p3']
            # prof.Zhang's suggestion
            #s.add(prevF)
            s.add(Not(nextcube))
            assert(s.check() == unsat)
            core = s.unsat_core()
            core = [str(core[i]) for i in range(0, len(core), 1)] # -> ['p1','p3'], core -> nextcube
        
        # cube_list = []
        # for index, literals in enumerate(tcube_cp.cubeLiterals):
        #     if index in core_list:
        for idx in range(len(tcube_cp.cubeLiterals)):
            var, val = _extract(prev_cube.cubeLiterals[idx])
            data[str(var)] = 0
            if 'p'+str(idx) not in core:
                tcube_cp.cubeLiterals[idx] = True
                data[str(var)] = 1
        # for the time being, completely rely on unsat core reduce



        # tcube_cp.cubeLiterals = cube_list
        #For loop in all previous cube

        # for i in range(len(ground_true.cubeLiterals)):
        #     var, val = _extract(prev_cube.cubeLiterals[i])
        #     data[str(var)] = 0 #TODO: Solve the issue that it contains float or int
        #     assert (type(data[str(var)]) is int)
        #     ground_true.cubeLiterals[i] = Not(ground_true.cubeLiterals[i]) # Flip the variable in f(v1,v2,v3,v4...)
        #     s = Solver()
        #     s.add(ground_true.cube()) #check if the miniterm (state) is sat or not
        #     res = s.check() #check f(v1,v2,v3,v4...) is
        #     #print("The checking result after fliping literal: ",res)
        #     assert (res == sat)
        #     # check the new sat model can transit to the CTI (true means it still can reach CTI)
        #     if str(s.model().eval(nextcube)) == 'True': #TODO: use tenary simulation -> solve the memeory exploration issue
        #         index_to_remove.append(i)
        #         # children = literal.children()
        #         # assert (len(children) == 2)
        #         #
        #         # if str(children[0]) in ['True', 'False']:
        #         #     v = str(children[1])
        #         # elif str(children[1]) in ['True', 'False']:
        #         #     v = str(children[0])
        #         # else:
        #         #     assert (False)
        #         # assert (v[0] in ['i', 'v'])
        #         # if v[0] == 'i':
        #         #     index_to_remove.add(idx)
        #         data[str(var)] = 1
        #         assert (type(data[str(var)]) is int)
        #         # substitute its negative value into nextcube
        #         v, val = _extract(prev_cube.cubeLiterals[i]) #TODO: using unsat core to reduce the literals (as preprocess process), then use ternary simulation
        #         #nextcube = simplify(And(substitute(nextcube, [(v, Not(val))])))
        #         nextcube = simplify(And(substitute(nextcube, [(v, Not(val))]), substitute(nextcube, [(v, val)])))
        #
        #     ground_true.cubeLiterals[i] = prev_cube.cubeLiterals[i]
        #TODO: Compare the ground true and unsat core

        # prev_cube.cubeLiterals = [prev_cube.cubeLiterals[i] for i in range(0, len(prev_cube.cubeLiterals), 1) if i not in index_to_remove]
        # tcube_cp.cubeLiterals = [tcube_cp.cubeLiterals[i] for i in range(0, len(tcube_cp.cubeLiterals), 1) if i not in index_to_remove]
        # return tcube_cp

        if self.smt2_gen_GP==1: self.generaliztion_data_GP.append(data)

        tcube_cp.remove_true()
        size_after_unsat_core = len(tcube_cp.cubeLiterals)
        #print("After generalization by using unsat core : ",len(tcube_cp.cubeLiterals))
        #print("After generalization by dropping literal one by one : ", len(index_to_remove))
        
        # Hongce: this is the beginning of ternary simulation-based variable reduction
        simulator = self.ternary_simulator.clone() # I just don't want to mess up between two ternary simulations for different outputs
        simulator.register_expr(nextcube)
        simulator.set_initial_var_assignment(dict([_extract(c) for c in tcube_cp.cubeLiterals]))

        out = simulator.get_val(nextcube)
        if out == ternary_sim._X:  # this is possible because we already remove once according to the unsat core
            return tcube_cp
        assert out == ternary_sim._TRUE
        for i in range(len(tcube_cp.cubeLiterals)):
            v, val = _extract(tcube_cp.cubeLiterals[i])
            simulator.set_Li(v, ternary_sim._X)
            out = simulator.get_val(nextcube)
            if out == ternary_sim._X:
                simulator.set_Li(v, ternary_sim.encode(val))  # set to its original value
                if simulator.get_val(nextcube) != ternary_sim._TRUE:
                    # This is just to help print debug info in case I made mistakes in coding
                    simulator._check_consistency()
                # after you recover the original input value, the output node should be true again
                assert simulator.get_val(nextcube) == ternary_sim._TRUE
            else: # the literal is removable
                # we should never get _FALSE
                if simulator.get_val(nextcube) != ternary_sim._TRUE:
                    # This is just to help print debug info in case I made mistakes in coding
                    simulator._check_consistency()
                assert simulator.get_val(nextcube) == ternary_sim._TRUE
                tcube_cp.cubeLiterals[i] = True
        tcube_cp.remove_true()
        size_after_ternary_sim = len(tcube_cp.cubeLiterals)
        return tcube_cp

    # def solveRelative_RL(self, tcube):
    #         cubePrime = substitute(substitute(tcube.cube(), self.primeMap),self.inp_map)
    #         s = Solver()
    #         s.add(self.frames[tcube.t - 1].cube())
    #         s.add(self.trans.cube())
    #         s.add(Not(tcube.cube()))
    #         s.add(cubePrime)  # F[i - 1] and T and Not(badCube) and badCube'
    #         if (s.check() != unsat):  # cube was not blocked, return new tcube containing the model
    #             model = s.model()
    #             # c = tCube(tcube.t - 1) #original verison
    #             # c.addModel(self.lMap, model)  # c = sat_model, original verison
    #             # return c #original verison
    #             return tCube(model, self.lMap, tcube.t - 1), None
    #         else:
    #             res,h= self.RL(tcube)
    #             return None, res

    # use edit distance as filter to promise the diveristy of the cex
    def levenshtein_distance(self, cex1: str, cex2: str) -> int:
        len1, len2 = len(cex1), len(cex2)
        # Initialization
        dp = [[0 for _ in range(len2 + 1)] for _ in range(len1 + 1)]
        for i in range(len1+1): 
            dp[i][0] = i
        for j in range(len2+1):
            dp[0][j] = j
            
        # Iteration
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if cex1[i-1] == cex2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        return dp[len1][len2]

    def get_all_model_partial(self, s_original, t):
        model_lst = []
        tCube_lst = []
        s = Solver()
        # copy s_original to s
        for c in s_original.assertions():
            s.add(c)
        res = s.check()
        assert(s_original.check() == res)

        #remove_duplicate_tcube = lambda lst : [list(t) for t in set(tuple(t) for t in lst)]

        # partial model
        
        while (res == sat):
            m = s.model()
            #print(m)
            #model_lst.append(m)
            #assert(len(model_lst) == 1)
            block = []
            for var in m: block.append(var() != m[var])
            s.add(Or(block))
            res = s.check()
            res2tcube = tCube(t)
            res2tcube.addModel(self.lMap, m, remove_input=True)
            if res2tcube not in tCube_lst:
                tCube_lst.append(res2tcube)
                if len(tCube_lst) >= 45:
                    break

        # older version -> has bug to determine when to stop (or how many models to generate in advance)
        # while (res == sat and len(model_lst) < 45):
        #     m = s.model()
        #     #print(m)
        #     model_lst.append(m)
        #     #assert(len(model_lst) == 1)
        #     block = []
        #     for var in m:
        #         block.append(var() != m[var])
        #     s.add(Or(block))
        #     res = s.check()
        #     #model_lst = remove_duplicate_tcube(model_lst)


        # for m in model_lst:
        #     res = tCube(t)
        #     res.addModel(self.lMap, m, remove_input=True)
        #     if res not in tCube_lst:
        #         tCube_lst.append(res)
        #         if len(tCube_lst) >= 45:
        #             break

        return tCube_lst


    def get_all_model_complete(self, s_original,t):
        model_lst = []
        tCube_lst = []
        s = Solver()
        # copy s_original to s
        for c in s_original.assertions():
            s.add(c)
        res = s.check()
        assert(s_original.check() == res)

        # partial model
        # while (res == sat and len(model_lst) < 45):
        #     m = s.model()
        #     #print(m)
        #     model_lst.append(m)
        #     #assert(len(model_lst) == 1)
        #     block = []
        #     for var in m:
        #         block.append(var() != m[var])
        #     s.add(Or(block))
        #     res = s.check()

        # for m in model_lst:
        #     res = tCube(t)
        #     res.addModel(self.lMap, m, remove_input=True)
        #     tCube_lst.append(res)
        
        # complete model
        latch_lst = [Bool(str(key).replace('_prime','')) for key in self.pv2next.keys()]
        while (res == sat and len(model_lst) < 45):
            m = s.model()
            block = []
            this_solution = Solver()
            # extract all variable in z3 solver

            for var in latch_lst:
                v = m.eval(var, model_completion=True)
                block.append(var != v)
                this_solution.add((var == True) if is_true(v) else (var == False))

            s.add(Or(block))
            res = s.check()
            model_lst.append(this_solution.assertions())

        for m in model_lst:
            res = tCube(t,cubeLiterals=m)
            tCube_lst.append(res)

        return tCube_lst


    def getBadCube(self):
        print("seek for bad cube...")

        s = Solver() #TODO: the input should also map to input'(prime)
        s.add(substitute(substitute(Not(self.post.cube()), self.primeMap),self.inp_map)) #TODO: Check the correctness here
        s.add(self.frames[-1].cube())
        s.add(self.trans.cube())

        if s.check() == sat: #F[-1] /\ T /\ !P(s') is sat! CTI (cex to induction) found!
            
            res = tCube(len(self.frames) - 1)
            res.addModel(self.lMap, s.model(), remove_input=False)  # res = sat_model
            print("get bad cube size:", len(res.cubeLiterals), end=' --> ') # Print the result
            # sanity check - why?
            self._debug_c_is_predecessor(res.cube(), self.trans.cube(), self.frames[-1].cube(), substitute(substitute(self.post.cube(), self.primeMap),self.inp_map)) #TODO: Here has bug
            new_model = self.generalize_predecessor(res, Not(self.post.cube()), self.frames[-1].cube()) #new_model: predecessor of !P extracted from SAT witness
            print(len(new_model.cubeLiterals)) # Print the result
            self._debug_c_is_predecessor(new_model.cube(), self.trans.cube(), self.frames[-1].cube(), substitute(substitute(self.post.cube(), self.primeMap),self.inp_map))
            new_model.remove_input()

            all_tcube_lst_complete = self.get_all_model_complete(s, new_model.t)
            all_tcube_lst_partial = self.get_all_model_partial(s, new_model.t)
            return new_model, all_tcube_lst_complete, all_tcube_lst_partial
        else:
            return None

    # def RandL(self, tcube):
    #     STEPS = self.agent.action_size
    #     done = False
    #     M = tcube.M
    #     orig = np.array([i for i in tcube.M if '\'' not in str(i)])
    #     cp = np.copy(orig)
    #     for ti in range(STEPS):
    #         action = np.random.randint(STEPS) % len(cp)
    #         cp = np.delete(cp, action);
    #         cubeprime = substitute(substitute(And(*[self.lMap[str(l)] == M.get_interp(l) for l in cp]), self.primeMap),self.inp_map)
    #         s = Solver()
    #         s.add(Not(And(*[self.lMap[str(l)] == M.get_interp(l) for l in cp])))
    #         s.add(self.R[tcube.t - 1])
    #         s.add(self.trans.cube())
    #         s.add(cubeprime)
    #         start = time.time()
    #         SAT = s.check();
    #         interv = time.time() - start
    #         if SAT != unsat:
    #             break
    #         else:
    #             if (self.isInitial(And(*[self.lMap[str(l)] == M.get_interp(l) for l in cp]), self.init)):
    #                 break
    #             else:
    #                 orig = np.copy(cp)
    #     return And(*[self.lMap[str(l)] == M.get_interp(l) for l in orig]), None

    # def RL(self, tcube):
    #     '''
    #     :param tcube:
    #     :return: res -> generalized q (q-like) , h -> None
    #     '''
    #     STEPS = self.agent.action_size
    #     # agent.load("./save/cartpole-ddqn.h5")
    #     done = False
    #     batch_size = 10
    #     history_QL = [0]
    #     state = [-1] * 10
    #     state = np.reshape(state, [1, self.agent.state_size])

    #     M = tcube.M
    #     orig = np.array([i for i in tcube.M if '\'' not in str(i)])
    #     cp = np.copy(orig)
    #     for ti in range(STEPS):
    #         # env.render()
    #         action = self.agent.act(state) % len(cp) # MLP return back the index of throwing literal
    #         cp = np.delete(cp, action);
    #         cubeprime = substitute(substitute(And(*[self.lMap[str(l)] == M.get_interp(l) for l in cp]), self.primeMap),self.inp_map)
    #         s = Solver()
    #         s.add(Not(And(*[self.lMap[str(l)] == M.get_interp(l) for l in cp])))
    #         s.add(self.R[tcube.t - 1])
    #         s.add(self.trans.cube())
    #         s.add(cubeprime)
    #         start = time.time()
    #         SAT = s.check();
    #         interv = time.time() - start
    #         if SAT != unsat:
    #             reward = -1
    #             done = True
    #         else:
    #             if (self.isInitial(And(*[self.lMap[str(l)] == M.get_interp(l) for l in cp]), self.init)):
    #                 reward = 0
    #                 done = True
    #             else:
    #                 reward = max(10 / interv, 1)
    #                 orig = np.copy(cp)

    #         next_state = [b for (a, b) in s.statistics()][:-4]
    #         if (len(next_state) > 10):
    #             next_state = next_state[0:10]
    #         else:
    #             i = len(next_state)
    #             while (i < 10):
    #                 next_state = np.append(next_state, -1)
    #                 i += 1
    #         # print(next_state)
    #         history_QL[-1] += reward
    #         next_state = np.reshape(next_state, [1, self.agent.state_size])
    #         self.agent.remember(state, action, reward, next_state, done)
    #         state = next_state
    #         if done:
    #             history_QL.append(0)
    #             self.agent.update_target_model()
    #             break
    #         if len(self.agent.memory) > batch_size:
    #             self.agent.replay(batch_size)
    #     # And(*[self.lMap[str(l)] == M.get_interp(l) for l in orig])-> generlization (when unsat core not exists)
    #     tmp_cube = And(*[self.lMap[str(l)] == M.get_interp(l) for l in orig])
    #     return And(*[self.lMap[str(l)] == M.get_interp(l) for l in orig]), history_QL

    def _debug_trace(self, trace: PriorityQueue):
        prev_fidx = 0
        self.bmc.setup()
        while not trace.empty():
            idx, cube = trace.get()
            assert (idx == prev_fidx+1)
            self.bmc.unroll()
            self.bmc.add(cube.cube())
            reachable = self.bmc.check()
            if reachable:
                print (f'F {prev_fidx} ---> {idx}')
            else:
                print(f'F {prev_fidx} -/-> {idx}')
                assert(False)
            prev_fidx += 1
        self.bmc.unroll()
        self.bmc.add(Not(self.post.cube()))
        assert(self.bmc.check() == sat)


    def _sanity_check_inv(self, inv):
        pass

    def _sanity_check_frame(self):
        for idx in range(0,len(self.frames)-1):
            # check Fi => Fi+1
            # Fi/\T => Fi+1
            Fi = self.frames[idx].cube()
            Fiadd1 = self.frames[idx+1].cube()
            s1 = Solver()
            s1.add(Fi)
            s1.add(Not(Fiadd1))
            assert( s1.check() == unsat)
            s2 = Solver()
            s2.add(Fi)
            s2.add(self.trans.cube())
            s2.add(substitute(substitute(Not(Fiadd1), self.primeMap),self.inp_map))
            assert( s2.check() == unsat)




if __name__ == '__main__':
    pass
