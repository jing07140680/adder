#!/usr/bin/env python3
import argparse
import gym
import drx_env_plus
import random
from lib import model
import math
import numpy as np
import torch
import multiprocessing
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import ptan
import logging
import sys
from genusg import genTraffic, genrow, genrecord
import scipy.stats as stats
import pandas as pd
import os
import csv
from scipy.stats import percentileofscore
import pickle

ENV_ID = "Drx-v1"


def test_edrx(record, dsmap, L, E, lock,gold):
    beliefstate = dsmap
    gold = np.percentile(beliefstate, 30)    
    action=[-1,1,1,-1]
    cur_time = 0
    done = 0
    res = 0
    while True:
        belief = max(beliefstate[math.floor(cur_time/60000) : math.floor(cur_time/60000)+60]) 
        obs = [gold*100, (belief)*100]
        state = genTraffic(record,cur_time)
        env.fillin(obs,state)             
        obs_, reward, done, info = env.step(action)
        cur_time = info[0]
        with lock:
            L.append(info[1]) #latency
            E.append(info[2])
        if done:
            obs = env.reset()
            break
    
def test(record,dsmap,act_net,L,E,lock,gold):
    beliefstate = dsmap
    gold = np.percentile(beliefstate, 30)
    cur_time = 0
    done = 0
    res = 0  
    while True:
        belief = max(beliefstate[math.floor(cur_time/60000) : math.floor(cur_time/60000)+60])
        obs = [gold*100, (belief)*100]
        state = genTraffic(record,cur_time)
        obs_v = torch.FloatTensor([obs])
        mu_v = act_net(obs_v)
        formatted_mu_v = [float("{:.2f}".format(x)) for x in mu_v.tolist()[0]]
        print(obs_v, formatted_mu_v)
        action = mu_v.squeeze(dim=0).data.numpy()
        action = np.clip(action, -1, 1)
        action[0] = -1
        env.fillin(obs,state)
        obs_, reward, done, info = env.step(action)
        cur_time = info[0]

        with lock:
            L.append(info[1]) #latency
            E.append(info[2])
        if done:
            obs = env.reset()
            break


def analyze_validation(L,B,S,E):
    data = np.array(L)/1000
    mean = np.mean(data)
    median = np.median(data)
    mode = stats.mode(data)
    minimum = np.min(data)
    maximum = np.max(data)
    range_ = np.ptp(data)
    std_dev = np.std(data)
    variance = np.var(data)
    q1 = np.percentile(data, 25)  # 25th percentile (Q1)
    q3 = np.percentile(data, 75)  # 75th percentile (Q3)
    iqr = q3 - q1  # Interquartile range
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    summary = pd.DataFrame({
        "Measure": ["Mean", "Median", "Mode", "Min", "Max", "Range", "Standard Deviation", "Variance", "Skewness", "Kurtosis", "Q1","Q3","IQR"],
        "Value": [mean, median, mode[0][0], minimum, maximum, range_, std_dev, variance, skewness, kurtosis, q1, q3, iqr]
    })  
    print(summary) 
    logging.info(summary)
    if sum(E)<sum(S):
        logging.info("energy saved: %.3f", 1-sum(E)/sum(S))
    else:
        logging.info("energy waste: %.3f", 1-sum(S)/sum(E))
    return

'''
def modelbufsave(L_buffer,B_buffer,S_buffer,E_buffer,T_buffer):
    #B_buffer_ = [percentileofscore(B_buffer, x) for x in B_buffer]
    with open('L.pkl', 'wb') as f:
        pickle.dump(L_buffer, f)
    with open('B.pkl', 'wb') as f:
        pickle.dump(B_buffer, f)
    with open('S.pkl', 'wb') as f:
        pickle.dump(S_buffer, f)
    with open('E.pkl', 'wb') as f:
        pickle.dump(E_buffer, f)
    with open('T.pkl', 'wb') as f:
        pickle.dump(T_buffer, f)
'''
def datasave(L1,L2,L3,E1,E2,E3):
    with open('L_act.pkl', 'wb') as f:
        pickle.dump(L1, f)
    with open('L_pred.pkl', 'wb') as f:
        pickle.dump(L2, f)
    with open('L_edrx.pkl', 'wb') as f:
        pickle.dump(L3, f)
    with open('E_act.pkl', 'wb') as f:
        pickle.dump(E1, f)
    with open('E_pred.pkl', 'wb') as f:
        pickle.dump(E2, f)
    with open('E_edrx.pkl', 'wb') as f:
        pickle.dump(E3, f)


    #zipped_data = list(zip(L_buffer,B_buffer,S_buffer,E_buffer,T_buffer))
    #with open('buffers/R_test.csv', 'w', newline='') as file:
    #    writer = csv.writer(file)
    #    writer.writerow(['Latency','Belief','Standard','Energy','Time Arrive'])
    #    writer.writerows(zipped_data)
          
if __name__ == "__main__":
    logging.basicConfig(filename='ddpg.log', level=logging.INFO)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    parser = argparse.ArgumentParser()
    parser.add_argument("-am", "--amodel", required=False, help="actor Model file to load")
    parser.add_argument("-cm", "--cmodel", required=False, help="critic Model file to load")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment name to use, default=" + ENV_ID)
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    args = parser.parse_args()
    env = gym.make(args.env)
    device = torch.device("cpu")
    if args.record:
        env = gym.wrappers.Monitor(env, args.record) 
    pact_net = model.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device) 
    pcrt_net = model.DDPGCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    pact_net.load_state_dict(torch.load(args.amodel))
    pcrt_net.load_state_dict(torch.load(args.cmodel))
    act_net = pact_net
    crt_net = pcrt_net

 
    # load actual statistic
    dsmap = []
    dsf = open("../scooterdata/actual.txt")
    Lines = dsf.readlines()
    dsf.close()
    for line in Lines:
        dsmap.append([float(x) for x in line.strip().split(" ")])
  
    # load prediction
    dspmap = []
    dspf = open("/adder/scooterdata/predict.txt")
    Lines = dspf.readlines()
    dspf.close()
    for line in Lines:
        dspmap.append([float(x) for x in line.strip().split(" ")])
         
    output_numbers = []
    with open('../scooterdata/actual.txt', 'r') as f:
        for line in f:
            numbers = line.strip().split()  # Assuming numbers are space-separated
            # Extract every 15th number starting from the first
            unique_numbers = numbers[::60]
            output_numbers.append([float(x) for x in unique_numbers])
    f.close()   
   
    L_act,E_act,L_pred,E_pred,L_edrx,E_edrx = [],[],[],[],[],[]
    for R_ in range(5):
        records, gold = genrecord(R_,output_numbers)
        dsmap_ = dsmap[R_]
        dspmap_ = dspmap[R_]
        record = [genrow(records[x]) for x in range(100)]
 
        with multiprocessing.Manager() as manager:
            lock = multiprocessing.Lock()
            L = manager.list()
            E = manager.list()
            for loop in range(4):
                processes = []
                for trail in range(25):
                    trail = loop*25+trail
                    process = multiprocessing.Process(target=test, args=(record[trail], dsmap_, act_net, L, E, lock, gold))
                    processes.append(process)
                    process.start()
                    
                # Waiting for all processes to finish
                for process in processes:
                    process.join()                    
            L_act.extend(list(L))
            E_act.extend(list(E))
            
        with multiprocessing.Manager() as manager:
            lock = multiprocessing.Lock()
            L = manager.list()
            E = manager.list()
            for loop in range(4):
                processes = []
                for trail in range(25):
                    trail = loop*25+trail
                    process = multiprocessing.Process(target=test, args=(record[trail], dspmap_, act_net, L, E, lock, gold))
                    processes.append(process)
                    process.start()
                            
                # Waiting for all processes to finish
                for process in processes:
                    process.join() 
                            
            L_pred.extend(list(L))
            E_pred.extend(list(E))

        with multiprocessing.Manager() as manager:
            lock = multiprocessing.Lock()
            L = manager.list()
            E = manager.list()
            for loop in range(4):
                processes = []
                for trail in range(25):
                    trail = loop*25+trail
                    process = multiprocessing.Process(target=test_edrx, args=(record[trail], dspmap_, L, E, lock, gold))
                    processes.append(process)
                    process.start()
                            
                # Waiting for all processes to finish
                for process in processes:
                    process.join() 
                            
            L_edrx.extend(list(L))
            E_edrx.extend(list(E))

            #analyze_validation(list(L),list(B),list(S),list(E), list(T))
    #modelbufsave(L_buffer,B_buffer,S_buffer,E_buffer,T_buffer)
    datasave(L_act,L_pred,L_edrx,E_act,E_pred,E_edrx)
