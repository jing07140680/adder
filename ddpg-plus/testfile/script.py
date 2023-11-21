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

GAMMA = 0.9
LEARNING_RATE_ = 0.00001 #actor's learning_rate
LEARNING_RATE = 0.05 #critic's learning_rate
ENV_ID = "Drx-v1"
gold = 0
  
# off-line training
def train(records,trail,dsmap,act_net,gold):
    record = genrow(records[trail])
    beliefstate = dsmap
    cur_time = 0
    done = 0
    res = 0 
    states = []
    actions = []
    rewards = []
    while True:
        belief = max(beliefstate[math.floor(cur_time/60000) : math.floor(cur_time/60000)+15])
        obs = [gold*100, (belief)*100]
        states.append(obs)  
        state = genTraffic(record,cur_time)
        obs_v = torch.FloatTensor([obs])
        mu_v = act_net(obs_v) 
        formatted_mu_v = [float("{:.2f}".format(x)) for x in mu_v.tolist()[0]]
        print(obs_v, formatted_mu_v)
        action = mu_v.squeeze(dim=0).data.numpy()

        dice = np.random.uniform(0,1)
        if dice < 0.2:
            noise = np.random.uniform(-0.1, 0.1,size=action.shape)
        else:
            action = np.random.uniform(-1, 1,size=action.shape).astype(np.float32)
            noise = np.array([0,0,0,0])
            
        action += noise
        action = np.clip(action, -1, 1)
        action[0] = -1
        actions.append(action) 
        env.fillin(obs,state)
        obs_, reward, done, info = env.step(action)
        cur_time = info[0]
 
        latency = info[1]
        energy = info[2] 
        standard = info[3]
   
        act0 = int(((action[0] - (-1)) / (1 - (-1))) * 29999 + 1)
        act1 = int(((action[1] - (-1)) / (1 - (-1))) * (900000-10)+10)
        act2 = int(((action[2] - (-1)) / (1 - (-1))) * (900000/10-1)+1)
        act3 = int(((action[3] - (-1)) / (1 - (-1))) * (900000))
        #print(action) 
        #print(act0,act1,act2*10,act3)
        res_time=900000
        res_time -= act0
        act1 = min(res_time, act1)
        res_time -= act1
        act3=min(res_time, act3)
        act2=min(act1//10, act2)
        act_ = min(act1,act2*10)
        #print(act0,act1,act2*10,act3,act_,max(act_,act3))
        #print("action max_delay:", max(act_,act3))
 
        ###################################################################################
        # Reward Function:
        # Rule: limit on latency when belief is high
        # the closer to 20000 the better if belief is high, to save more energy
        if belief > gold:
            reward = (belief/gold)*(20000-max(act_,act3))/9000  + 10*(standard/energy) -10*(latency/90000)
        else:
            reward = (belief/gold)*(20000-max(act_,act3))/9000  + 10*(standard/energy) -10*(latency/90000) + 110
        
        # max(act_,act3) = 900000:           satisfy energy                        satisfy latency
        # belief/gold > 1:  -100*(belief/gold) + 10*(standard/energy)     1/9000 + 10*(standard/energy)
        # belief/gold < 1:  -100*(belief/gold <=1) + 10*(standard/energy)  <1/9000 + 10*(standard/energy)

        # -100*a < 1/9000    -100 + b >  1/9000  
        rewards.append(reward)  
        formatted_action = [float("{:.2f}".format(x)) for x in action]
        print(formatted_action,latency,"{:.0f}".format(energy), standard,"{:.4f}".format(reward))
        #print(formatted_action,reward)
        print('------------')
        ######################################################################################

        if done:  
            obs = env.reset()
            break

    return torch.tensor(states),torch.tensor(actions),torch.tensor(rewards)



def test(records,trail,dsmap,act_net,L,B,S,E,lock,gold):
    record = genrow(records[trail])
    beliefstate = dsmap
    cur_time = 0
    done = 0
    res = 0  
    while True:
        belief = max(beliefstate[math.floor(cur_time/60000) : math.floor(cur_time/60000)+15])
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

        ####################################################################################
        # Record Latency buffer, Energe buffer, Standard buffer, belief buffer
        ###################################################################################
        if state != 900000:
            with lock:
                L.append(info[1])
                E.append(info[2])
                S.append(info[3])
                B.append(obs[1])

        if done:
            obs = env.reset()
            break
    return


def analyze_validation(L,B,S,E,best):
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
 
def modelbufsave(R,L_buffer,B_buffer,S_buffer,E_buffer,best):

    filenames = ["buffers/L"+str(R)+".csv","buffers/B"+str(R)+".csv","buffers/S"+str(R)+".csv","buffers/E"+str(R)+".csv" ]
    buffers = [L_buffer,B_buffer,S_buffer,E_buffer]
    for i in range(len(buffers)):
        with open(filenames[i], 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(buffers[i])
    inner_averages = []
    # Convert the list of lists to a NumPy array
    for i in range(len(L_buffer)):
        inner_averages.append(np.mean(L_buffer[i])/1000)
    # Calculate the average of averages
    overall_average = np.mean(inner_averages)
    print("Averages latency validations:", inner_averages)
    logging.info("Averages latency validations: %s", inner_averages)
    print("Average of average L:", overall_average) 
    logging.info("Average of average L: %.3f", overall_average)

    tmp = []
    for i in range(len(E_buffer)):
        e = sum(E_buffer[i])
        s = sum(S_buffer[i])
        tmp.append(1-e/s if e<s else -(1-s/e))
    print("Saved energy:",tmp)
    logging.info("Saved energy: %s",tmp)
    res = np.mean(tmp) 
    print("Average saved energy: %.3f",res)
    logging.info("Average saved energy: %.3f",res)
   
    if overall_average < 20 and res > best:
        print("Best energy updated  %.3f -> %.3f" % (best, res))
        logging.info("Best energy updated  %.3f -> %.3f", best, res) 
        save_path = os.path.join("saves", "models")
        actname = "best_%+.3f_%d_act.dat" % (res*100, R)
        crtname = "best_%+.3f_%d_crt.dat" % (res*100, R)
        actfname = os.path.join(save_path, actname)
        crtfname = os.path.join(save_path, crtname)
        torch.save(act_net.state_dict(), actfname)
        torch.save(crt_net.state_dict(), crtfname)
        best = res
    return 
         
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


    writer = SummaryWriter(comment="-ddpg_" + "a")
    agent = model.AgentDDPG(act_net, device=device)
    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE_)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)
    frame_idx = 0
    best_reward = None


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

    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=1) as tb_tracker:
            ################################################################
            # The first 60% records for trainning data
            ###############################################################
            '''
            best = -1
            buffer_dir = os.path.join("buffers")
            os.makedirs(buffer_dir, exist_ok=True)
            logging.info('Started')
            
            for R in range(20): 
                records, gold = genrecord(R,output_numbers)        
                dsmap_ = dsmap[R] 
                for trail in range(100):
                    print("train trail:",trail)
                    logging.info("train trail: %d",trail)
                    states_v, actions_v, rewards_v = train(records,trail,dsmap_,act_net,gold)
                    crt_opt.zero_grad()
                    q_v = crt_net(states_v, actions_v)
                    q_ref_v = rewards_v.unsqueeze(dim=-1)# + q_last_v * GAMMA
                    critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                     
                    critic_loss_v.backward() 
                    crt_opt.step()
                    tb_tracker.track("loss_critic", critic_loss_v, frame_idx)
                    tb_tracker.track("critic_ref", q_ref_v.mean(), frame_idx)

                    # train actor
                    act_opt.zero_grad()
                    cur_actions_v = act_net(states_v)
                    actor_loss_v = -crt_net(states_v, cur_actions_v)  
                    #actor_loss_v = torch.mul(actor_loss_v, states_v)
                    actor_loss_v = actor_loss_v.mean()
                    #actor_loss_v =  actor_loss_v.sum()
                    print(actor_loss_v)
                    actor_loss_v.backward()
                    act_opt.step() 
                    tb_tracker.track("loss_actor", actor_loss_v, frame_idx)
                    frame_idx += 1
            '''
            ################################################################
            # The first 20% records for validating
            ###############################################################
            best = -1
            L_buffer, B_buffer, S_buffer, E_buffer = [], [], [], []
            for R_ in range(20,25):
                records, gold = genrecord(R_,output_numbers)
                dsmap_ = dsmap[R_]
                with multiprocessing.Manager() as manager:
                    lock = multiprocessing.Lock()
                    L = manager.list()
                    B = manager.list()
                    E = manager.list()
                    S = manager.list()
                    for loop in range(5):
                        processes = []
                        for trail in range(10):
                            trail = loop*10+trail
                            print("test trail:",trail)
                            #logging.info("test trail: %d",trail)
                            process = multiprocessing.Process(target=test, args=(records,trail, dsmap_, act_net, L, B, S, E, lock, gold))
                            processes.append(process)
                            process.start()
                            
                        # Waiting for all processes to finish
                        for process in processes:
                            process.join() 
                            
                    ###################################################
                    # Process and analyze the results                            
                    ###################################################
                    logging.info("###############Validate %d ###########",R_)
                    if not L or not S or not E:
                        logging.info("Got L NULL !!!!")
                    else:  
                        L_buffer.append(list(L))
                        B_buffer.append(list(B))
                        S_buffer.append(list(S))
                        E_buffer.append(list(E))
                        analyze_validation(list(L),list(B),list(S),list(E),best)
                #modelbufsave(R,L_buffer,B_buffer,S_buffer,E_buffer,best)
 
