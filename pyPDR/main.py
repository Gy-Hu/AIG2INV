# -*- coding: UTF-8 -*- 
'''
Main function to run PDR (extract the graph as well)
'''

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from datetime import datetime
from datetime import timedelta
from multiprocessing import Process
from threading import Thread

from time import sleep
import sys
sys.path.append("..")
import model
import pdr
import time
import matplotlib.pyplot as plt
import csv
# from env import QL

# When you need to run all folder, setup this
test_file_path = "../dataset/"
test_file_folder_path = "../dataset/aag4train/" #open this if run through a folder

if __name__ == '__main__':
    #sys.stdout = open('file', 'w') #open this when we need the log
    help_info = "Usage: python main.py <file-name>.aag"
    parser = argparse.ArgumentParser(description="Run tests examples on the PDR algorithm")
    parser.add_argument('fileName', type=str, help='The name of the test to run', default=None, nargs='?')
    parser.add_argument('--mode',type=int,help='choose the mode to run the program, 0 means only run one file, 1 means run through the files in folder',default=0)
    parser.add_argument('-t', type=int, help='the time limitation of one test to run', default=900)
    parser.add_argument('-p', type=str, help='file path of mode 1', default=None)
    parser.add_argument('-c', help='switch to counting time', action='store_true')
    parser.add_argument('-d', type=str, help='switch to do data generation in generalized predecessor or inductive generalization', default='off')
    parser.add_argument('-n', type=str, help='switch to use neural network in inductive generalization or generalized predecessor', default='off')
    parser.add_argument('-a', type=str, help='Use NN-guided IG and append to MIC', default='off')
    parser.add_argument('-s', type=str, help='Save the inductive invariant', default='off')
    parser.add_argument('-r', type=str, help='Record the result', default='off')
    parser.add_argument('-th', type=float, help='threshold for the inductive invariant', default=0.5)
    parser.add_argument('-mn', type=str, help='model name of NN', default=None)
    parser.add_argument('-tm', type=str, help='test mic', default='off')
    parser.add_argument('-inf_dev', type=str, help='device do inference', default='gpu')
    parser.add_argument('-check_cti', type=str, help='check CTI before exporting', default='on')

    #args = parser.parse_args(["./nusmv.reactor^4.C.aag"])
    args = parser.parse_args(['./nusmv.syncarb10^2.B.aag','-d','ig'])
    if (args.fileName is not None) and (args.mode==0):
        file = args.fileName
        m = model.Model()

        # state_size = 10  # set up RL
        # action_size = 8  # set up RL
        # agent = None #QL(state_size, action_size)  # set up RL

        print("============= Running test ===========")

        # Not using RL
        solver = pdr.PDR(*m.parse(file))

        # Switch to turn on/off using neural network to guide generalization (predecessor/inductive generalization)
        if args.n=='off':
            solver.test_IG_NN = 0
            solver.test_GP_NN = 0
        elif args.n=='on':
            solver.test_IG_NN = 1
            solver.test_GP_NN = 1
        elif args.n=='ig':
            solver.test_IG_NN = 1
            solver.test_GP_NN = 0
        elif args.n=='gp':
            solver.test_IG_NN = 0
            solver.test_GP_NN = 1

        # Switch to turn on/off the data generation of generalized predecessor or inductive generalization
        if args.d=='off':
            solver.smt2_gen_GP = 0
            solver.smt2_gen_IG = 0
        elif args.d=='on':
            solver.smt2_gen_GP = 1
            solver.smt2_gen_IG = 1
        elif args.d=='ig':
            solver.smt2_gen_IG = 1
            solver.smt2_gen_GP = 0
        elif args.d=='gp':
            solver.smt2_gen_IG = 0
            solver.smt2_gen_GP = 1

        # On/off the NN-guided ig append to MIC
        if args.a=='off':
            solver.NN_guide_ig_append = 0
        elif args.a=='on':
            solver.NN_guide_ig_append = 1

        # On/off the collection of inductive invariant
        if args.s=='off':
            solver.collect_inductive_invariant = 0
        elif args.s=='on':
            solver.collect_inductive_invariant = 1

        # On/off the recording of result
        if args.r=='off':
            solver.record_result = 0
        elif args.r=='on':
            solver.record_result = 1

        # Set the thershold of prediction
        solver.prediction_threshold = args.th

        # Set the model name to predict
        solver.model_name = args.mn

        # Set the switch to test mic
        if args.tm=='off':
            solver.test_mic = 0
        elif args.tm=='on':
            solver.test_mic = 1

        # switch device to do inference
        if args.inf_dev=='cpu':
            solver.inf_device = 'cpu'
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        elif args.inf_dev=='gpu':
            solver.inf_device = 'gpu'

        # switch to check cti before exporting
        if args.check_cti=='off':
            solver.check_CTI_before_export = 0
        elif args.check_cti=='on':
            solver.check_CTI_before_export = 1

        startTime = time.time()
        # Record start time
        solver.start_time = time.time()
        solver.run()
        endTime = time.time()
        print("Finish runing aiger file:"+args.fileName)
        if args.c:
            if solver.NN_guide_ig_time_sum != 0:
                print("TIME CONSUMING IN TOTAL: ",(endTime - startTime) ,"seconds")
                print("TIME CONSUMING WITH NN, WITHOUT INF TIME: " ,(endTime - startTime - solver.NN_guide_ig_time_sum) , "seconds")  
                print("TIME CONSUMING IN PUSH LEMMA", solver.pushLemma_time_sum)
                print("TIME CONSUMING WITHOUT RANDOM MIC TIME: ",(endTime - startTime - solver.test_random_mic_time_sum) , "seconds")
                if solver.test_IG_NN : 
                    print("NN-guided inductive generalization success rate: ",(solver.NN_guide_ig_success/(solver.NN_guide_ig_success + solver.NN_guide_ig_fail))*100,"%")             
                    y_nn_ig_pass_ratio, x_nn_ig_pass_ratio = zip(*solver.NN_guide_ig_passed_ratio)
                    plt.plot(x_nn_ig_pass_ratio,y_nn_ig_pass_ratio)
                    plt.savefig("../log/NN_guided_IG_pass_ratio.jpg") 
            else:
                print("TIME CONSUMING: " ,(endTime - startTime) , "seconds")
    elif args.mode==1: # 1 means runs through all the folder
        print("================ Test the ./aag directory ========")
        agent = None
        for root, dirs, files in os.walk(args.p): #TODO: 把aig原本的二进制文件也搬进来，这边代码改成仅把.aag加入处理的文件队列里面
            for name in files:
                if name.endswith('.aag'):
                    print("============ Testing " + str(name) + " ==========")
                    m = model.Model()
                    solver = pdr.PDR(*m.parse(os.path.join(root, name)))

                    # Switch to turn on/off using neural network to guide generalization (predecessor/inductive generalization)
                    if args.n=='off':
                        solver.test_IG_NN = 0
                        solver.test_GP_NN = 0
                    elif args.n=='on':
                        solver.test_IG_NN = 1
                        solver.test_GP_NN = 1
                    elif args.n=='ig':
                        solver.test_IG_NN = 1
                        solver.test_GP_NN = 0
                    elif args.n=='gp':
                        solver.test_IG_NN = 0
                        solver.test_GP_NN = 1

                    # Switch to turn on/off the data generation of generalized predecessor or inductive generalization
                    if args.d=='off':
                        solver.smt2_gen_GP = 0
                        solver.smt2_gen_IG = 0
                    elif args.d=='on':
                        solver.smt2_gen_GP = 1
                        solver.smt2_gen_IG = 1
                    elif args.d=='ig':
                        solver.smt2_gen_IG = 1
                        solver.smt2_gen_GP = 0
                    elif args.d=='gp':
                        solver.smt2_gen_IG = 0
                        solver.smt2_gen_GP = 1

                    # On/off the NN-guided ig append to MIC
                    if args.a=='off':
                        solver.NN_guide_ig_append = 0
                    elif args.a=='on':
                        solver.NN_guide_ig_append = 1

                    # On/off the collection of inductive invariant
                    if args.s=='off':
                        solver.collect_inductive_invariant = 0
                    elif args.s=='on':
                        solver.collect_inductive_invariant = 1

                    # On/off the recording of result
                    if args.r=='off':
                        solver.record_result = 0
                    elif args.r=='on':
                        solver.record_result = 1
                    
                    # Set the thershold of prediction
                    solver.prediction_threshold = args.th

                    # Set the model name to predict
                    solver.model_name = args.mn

                    # Set the switch to test mic
                    if args.tm=='off':
                        solver.test_mic = 0
                    elif args.tm=='on':
                        solver.test_mic = 1


                    # switch device to do inference
                    if args.inf_dev=='cpu':
                        solver.inf_device = 'cpu'
                        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                    elif args.inf_dev=='gpu':
                        solver.inf_device = 'gpu'
                        if args.gpu_id != -1:
                            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

                    # switch to check cti before exporting
                    if args.check_cti=='off':
                        solver.check_CTI_before_export = 0
                    elif args.check_cti=='on':
                        solver.check_CTI_before_export = 1

                    solver.folder_name =  args.p.split('/')[-1]

                    startTime = time.time()
                    timeout = False

                    # Record start time
                    solver.start_time = time.time()

                    # t = Thread(target=solver.run)
                    # t.daemon = True
                    # t.start() # start the thread
                    # t.join(timeout=args.t) #FIXME: If timeout, the thread will throw exception and program core dump
                    
                    p = Process(target=solver.run, name="PDR")
                    p.start()
                    # Wait a maximum of 10 seconds for foo
                    # Usage: join([timeout in seconds])
                    p.join(timeout=int(args.t))
                    endTime = time.time()
                    # If thread is active
                    if p.is_alive():
                        timeout = True
                        print("Finish runing aiger file:"+str(name))
                        print("PDR run out of the time... let's kill it...")
                        # Terminate foo
                        p.terminate()
                        p.join()

                    # if timeout == True:
                    #     sleep(20)
                    # elif timeout != True:
                    if timeout != True:
                        solve_time = (endTime - startTime)
                        print("Finish runing aiger file:"+str(name))
                        print("Done in time")
                        #sleep(20)
                        if args.c:
                            print("TIME CONSUMING: ", solve_time, "seconds")
                        
    else:
        print("Wrong input, please give a vaild input or check the document")

#TODO: Run all the dataset (includes tip..)