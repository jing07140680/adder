#!/usr/bin/env python3
import argparse
import gym
import drx_env_plus
import drx_env_AC
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

ENV1_ID = "Drx-v1"
ENV2_ID = "Drx-v2"
 
max_subframes = 120*60*1000
def test_edrx(beliefs, gold, states, env, L, E):
    L_, E_ = [], []
    action=[-1,1,1,-1]
    for i in range(20):
        for j in range(5):
            obs = [gold, beliefs[i]]
            state = states[i][j]
            env.fillin(obs,state)           
            obs_, reward, done, info = env.step(action)
            L_.append(info[1]) #latency
            E_.append(info[2])
    obs = env.reset()
    L.append(L_)
    E.append(E_)

def test_AC(beliefs, gold, states, env, act_net, L, E):
    L_,E_ = [],[]
    obs = [5]*5
    for i in range(20):
        belief = beliefs[i]
        state = states[i]
        obs_v = torch.FloatTensor([obs])
        mu_v = act_net(obs_v)
        formatted_mu_v = [float("{:.2f}".format(x)) for x in mu_v.tolist()[0]]
        action = mu_v.squeeze(dim=0).data.numpy()
        action = np.clip(action, -1, 1)
        action[0] = -1
        env.fillin(belief,gold,obs,state)
        obs, reward, done, info = env.step(action)
        if state !=max_subframes:
            L_.append(info[1]) #latency
        E_.append(info[2])
        obs = env.reset()
    L.append(L_)
    E.append(E_)

    
def test_wyj(beliefs, gold, states, env, act_net, L, E):
    res = 0
    L_,E_ = [],[]
    for i in range(20):
        for j in range(5):
            obs = [gold, beliefs[i]]
            state = states[i][j]
            obs_v = torch.FloatTensor([obs])
            mu_v = act_net(obs_v)
            formatted_mu_v = [float("{:.2f}".format(x)) for x in mu_v.tolist()[0]]
            print(obs_v, formatted_mu_v)
            action = mu_v.squeeze(dim=0).data.numpy()
            action = np.clip(action, -1, 1)
            action[0] = -1
            env.fillin(obs,state)
            obs_, reward, done, info = env.step(action)
            if state != max_subframes:
                L_.append(info[1]) #latency
            E_.append(info[2])
    obs = env.reset()
    L.append(L_) #latency
    E.append(E_)
 
 
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
 
def modelbufsave(L1,L2,L3,L4,E1,E2,E3,E4):
    #B_buffer_ = [percentileofscore(B_buffer, x) for x in B_buffer]
    with open('L_10.pkl', 'wb') as f:
        pickle.dump(L1, f)
    with open('L_AC.pkl', 'wb') as f:
        pickle.dump(L2, f)
    with open('L_edrx.pkl', 'wb') as f:
        pickle.dump(L3, f)
    with open('L_40.pkl', 'wb') as f:
        pickle.dump(L4, f)
    with open('E_10.pkl', 'wb') as f:
        pickle.dump(E1, f)
    with open('E_AC.pkl', 'wb') as f:
        pickle.dump(E2, f)
    with open('E_edrx.pkl', 'wb') as f:
        pickle.dump(E3, f)
    with open('E_40.pkl', 'wb') as f:
        pickle.dump(E4, f)
        
    #zipped_data = list(zip(L_buffer,B_buffer,S_buffer,E_buffer,T_buffer))
    #with open('buffers/R_test.csv', 'w', newline='') as file:
    #    writer = csv.writer(file)
    #    writer.writerow(['Latency','Belief','Standard','Energy','Time Arrive'])
    #    writer.writerows(zipped_data)
          
if __name__ == "__main__":
    logging.basicConfig(filename='AC.log', level=logging.INFO)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    parser = argparse.ArgumentParser()
    parser.add_argument("-am1", "--amodel1", required=False, help="actor Model file to load 1")
    parser.add_argument("-cm1", "--cmodel1", required=False, help="critic Model file to load 1")
    parser.add_argument("-am2", "--amodel2", required=False, help="actor Model file to load 2")
    parser.add_argument("-cm2", "--cmodel2", required=False, help="critic Model file to load 2")
    parser.add_argument("-e1", "--env1", default=ENV1_ID, help="Environment name to use, default=" + ENV1_ID)
    parser.add_argument("-e2", "--env2", default=ENV2_ID, help="Environment name to use, default=" + ENV2_ID)
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    args = parser.parse_args()
    env1 = gym.make(args.env1) 
    env2 = gym.make(args.env2)
    device = torch.device("cpu")
    pact_net1 = model.DDPGActor(env1.observation_space.shape[0], env1.action_space.shape[0]).to(device) 
    pcrt_net1 = model.DDPGCritic(env1.observation_space.shape[0], env1.action_space.shape[0]).to(device)
    pact_net2 = model.DDPGActor(5, env2.action_space.shape[0]).to(device)
    pcrt_net2 = model.DDPGCritic(5, env2.action_space.shape[0]).to(device)
    pact_net1.load_state_dict(torch.load(args.amodel1))
    pcrt_net1.load_state_dict(torch.load(args.cmodel1))
    pact_net2.load_state_dict(torch.load(args.amodel2))
    pcrt_net2.load_state_dict(torch.load(args.cmodel2))
    act_net1 = pact_net1
    crt_net1 = pcrt_net1
    act_net2 = pact_net2
    crt_net2 = pcrt_net2

    #numbers = [0,10,30,40,70,80]
    manager = multiprocessing.Manager()
    L1,E1 = manager.list(), manager.list()
    L2,E2 = manager.list(), manager.list()
    L3,E3 = manager.list(), manager.list()
    L4,E4 = manager.list(), manager.list()
    for R_ in range(100):
        #beliefs = [random.choice(numbers) for _ in range(20)]
        beliefs = [random.uniform(0, 100) for _ in range(20)]
        print(beliefs)
        states = []
        for x in range(len(beliefs)):
            non_zero_elements_count = int(5*beliefs[x]/100)
            non_zero_elements = np.random.randint(1, max_subframes, size=non_zero_elements_count)
            elements = [max_subframes]*5
            elements[:non_zero_elements_count] = non_zero_elements
            np.random.shuffle(elements)
            states.append(elements)
            
        print(states)
        
        processes = []
        
        process = multiprocessing.Process(target=test_wyj, args=(beliefs,10,states,env1,act_net1,L1,E1))
        processes.append(process)
        process.start()

        process = multiprocessing.Process(target=test_AC, args=(beliefs,10,states,env2,act_net2,L2,E2))
        processes.append(process)
        process.start()
 
        process = multiprocessing.Process(target=test_edrx, args=(beliefs,10,states,env1,L3,E3))
        processes.append(process)
        process.start()
        
        process = multiprocessing.Process(target=test_wyj, args=(beliefs,40,states,env1,act_net1,L4,E4))
        processes.append(process)
        process.start()
        
        #test_wyj(beliefs,10,states,env1,act_net1,L1,E1)
        #test_AC(beliefs,10,states,env2,act_net2,L2,E2)
        #test_edrx(beliefs,10,states,env1,L3,E3)
        #test_wyj(beliefs,40,states,env1,act_net1,L4,E4)
        
        for process in processes:
            process.join()

         
        ###################################################
        # Process and analyze the results                            
        ###################################################
        logging.info("###############Validate %d ###########",R_)

    #logging.info("wyj latency:%s",L1)
    #logging.info("AC latency:%s",[item for sublist in L2 for item in sublist])
    #logging.info("standard:%s",S1)
    #logging.info("wyj energy:%s",E1)
    #logging.info("AC energy:%s",[item for sublist in E2 for item in sublist])
    modelbufsave(list(L1),list(L2),list(L3),list(L4),list(E1),list(E2),list(E3),list(E4))
