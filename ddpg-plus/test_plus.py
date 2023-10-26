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
import torch.nn.functional as F

ENV_ID = "Drx-v1"
# energy consumption here use average current
# Energy usage, the unit is uA(current)(to fix!!!)
CE = 35000 # energy consumed when fully connected
IE = 2620  # energy consumed when connected in edrx idle
BE = 5  # Basic energy consumed when released in edrx idle
SE = 2.7  # Sleep energy
gold = 0
gold2 = 0

from torch.utils.data import Dataset

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(3, 8)
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, state): 
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_mean = self.sigmoid(self.fc3(x))
        action_stddev = torch.ones_like(action_mean) * 0.1
        return action_mean, action_stddev
    def sample_action(self, state):
        action_mean, action_stddev = self(state)
        #print("sample_action",state,action_mean)
        normal_dist = torch.distributions.Normal(action_mean, action_stddev)
        action = normal_dist.sample()
        action =  torch.clamp(action, 0, 1)
        return action,normal_dist.log_prob(action), action_mean
                            
                                                                 
 
def genTraffic(record,cur_time):
    #print(record,cur_time)
    if not record:
        state = 900000
    else:
        if cur_time < record[0] and cur_time+900000 >= record[0]:
            state = record[0]-cur_time
            record.pop(0)
        else:
            state = 900000
    return state
  

def genrecord(R,output_numbers):
    probabilities = output_numbers[R]
    #gold = np.percentile(probabilities, 80)
    #gold2 = np.percentile(probabilities, 30)
    gold = np.percentile(probabilities, 80)
    gold2 = np.percentile(probabilities, 40)
    rows = 100
    columns = 24  
    matrix = np.zeros((rows, columns))
 
    for col, prob in enumerate(probabilities):
        matrix[:, col] = np.random.choice([0, 1], size=(rows,), p=[1-prob, prob])
    # Replace 1s with random value from 0 to 60
    matrix[matrix == 1] = [random.randint(0, 60) for _ in range(int(np.sum(matrix)))]
    return matrix, gold, gold2

def genrow(record): 
    res = []
    cnt = 0
    for i in range(1,len(record)):
        if record[i] != 0:
            res.append((cnt+record[i])*60000)
            cnt += 60
        else:
            cnt += 60
    #print(res)
    return res
 

def compute_loss(log_probs, returns_tensor):
    loss = []
    for log_prob, R in zip(log_probs,returns_tensor):
        loss.append(-log_prob * R)
    return torch.cat(loss).sum()
 
def compute_returns(rewards, discount_factor=0):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + discount_factor * R
        returns.insert(0, R)
    return returns
                         
def adaptor(records,trail,dsmap,act_net,policy,optimizer,gold, gold2):
    record = genrow(records[trail])
    beliefstate = dsmap[0]
    cur_time = 0
    done = 0
    res = 0
    rewards = []
    states = []
    mean_beliefs = []
    sample_beliefs = []
    log_probs = [] 
    cnt = 0
    while True:
        if cnt == 20:
            done = 1
        else:
            done = 0
        belief = max(beliefstate[math.floor(cur_time/60000) : math.floor(cur_time/60000)+15])
        states.append(torch.tensor([belief,gold,gold2], requires_grad=True))
        sample_belief,_,_= policy.sample_action(torch.tensor([belief,gold,gold2], dtype=torch.float32))
        sample_beliefs.append(sample_belief)
        obs = [20, (sample_belief)*100]
        state = genTraffic(record,cur_time)
        obs_v = torch.FloatTensor([obs])
        mu_v = act_net(obs_v)
        action = mu_v.squeeze(dim=0).data.numpy()
        #if (state != 900000):
        #    print(obs[1], action)    
        env.fillin(obs,state)
        obs_, _, done, info = env.step(action)
        cur_time = info[0]

        reward = 0 
        if belief > gold:
            if 0.7 < sample_belief :
                reward = 2
            else: 
                reward =-2
        else:
            if sample_belief < 0.3:
                reward = 1
            else:
                reward = -1
        print("get reward",belief,sample_belief,reward)
        rewards.append(reward)
        if done:
            obs = env.reset()
            break
        cnt += 1
    states = torch.stack(states)
    #sample_beliefs = torch.stack(sample_beliefs)
    rewards = torch.tensor(rewards)
    #returns = compute_returns(rewards)
    #returns_tensor = torch.tensor(returns)
    #print("states",states)
    #print("sample",sample_beliefs)
    #print('R',rewards)

    R = 0
    policy_loss = []
    reversed_states = torch.flip(states, [0])
    reversed_rewards = torch.flip(rewards, [0])
    gamma = 0.5
    printed_s = {}
    for s, r in zip(reversed_states,reversed_rewards):
        R = r + gamma * R  
        action_prob,log_prob,action_mean = policy.sample_action(s)
        if s[0].item() not in printed_s:
            print(s,action_mean,action_prob,R)
            printed_s[s[0].item()] = 1
        policy_loss.append(-log_prob * R)  
       
    #params_before = [p.clone() for p in policy.parameters()]
    #print("before:",params_before)    
    optimizer.zero_grad()
    #loss = compute_loss(log_probs,returns_tensor)
    loss = torch.cat(policy_loss).sum()    
    print(loss)
    loss.backward()
    optimizer.step()
    #params_after = [p.clone() for p in policy.parameters()]
    #print("after:",params_after)          
                                        
            
def test(records,trail,dsmap,act_net,L,uL,S,E,lock,policy):
    record = genrow(records[trail])
    beliefstate = dsmap[0]
    cur_time = 0
    done = 0 
    res = 0
    while True:
        belief = max(beliefstate[math.floor(cur_time/60000) : math.floor(cur_time/60000)+15])
        belief = policy(torch.FloatTensor([belief]))[0].item()
        obs = [20, (belief)*100]
        state = genTraffic(record,cur_time)
        obs_v = torch.FloatTensor([obs])
        mu_v = act_net(obs_v)
        action = mu_v.squeeze(dim=0).data.numpy()
        if (state != 900000):
            print(obs[1], action)
        env.fillin(obs,state) 
        obs_, reward, done, info = env.step(action)
        cur_time = info[0]
        #print(info[4],info[1],obs[1])
        with lock:
            if state != 900000 and obs[1] != 30:
                L.append(info[1])
            elif state != 900000 and obs[1] == 0:
                print(state,obs[1])
                uL.append(info[1])
            E.append(info[2])
            S.append(info[3])

        if done:
            obs = env.reset()
            break
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-am", "--amodel", required=False, help="actor Model file to load")
    #parser.add_argument("-cm", "--cmodel", required=False, help="critic Model file to load")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment name to use, default=" + ENV_ID)
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    args = parser.parse_args()
    env = gym.make(args.env)
    device = torch.device("cpu")
    if args.record:
        env = gym.wrappers.Monitor(env, args.record) 
    act_net = model.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    #crt_net = model.DDPGCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    act_net.load_state_dict(torch.load(args.amodel))
    #crt_net.load_state_dict(torch.load(args.cmodel))
    policy = PolicyNetwork()
    q_net = QNetwork()
    optimizer = optim.Adam(policy.parameters(), lr=0.05)
    qoptimizer = optim.Adam(q_net.parameters(), lr=0.05)
    
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
 
    # train
    for R in range(1): 
        records, gold, gold2 = genrecord(R,output_numbers)        
        for trail in range(100):
            print(trail)
            adaptor(records,trail,dsmap,act_net,policy,optimizer,gold, gold2)
            #params_after = [p.clone() for p in policy.parameters()]
            #print("trail",params_after[0][0])
    '''
    with multiprocessing.Manager() as manager:
        lock = multiprocessing.Lock()
        L = manager.list()
        uL = manager.list()
        E = manager.list()
        S = manager.list()
        processes = []
        for R in range(1,2):
            records, gold, gold2 = genrecord(R,output_numbers)
            for trail in range(10):
                process = multiprocessing.Process(target=test, args=(records,trail,dsmap,act_net,L,uL,S,E,lock,policy))
                processes.append(process)
                process.start()

            for process in processes:
                process.join()

        
            if not L or not S or not E:
                print("L NULL")
            else:
                print("latency:",L)
                print(len(L), len(uL))
                print("average latency:", np.mean(list(L)),"max latency:",max(list(L)))
                print("energy saved:", 1-sum(E)/sum(S)) if sum(E)<sum(S) else  print("energy waste:", 1-sum(S)/sum(E))
    '''
