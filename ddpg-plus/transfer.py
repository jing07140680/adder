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
GAMMA = 0.9
LEARNING_RATE_ = 0.0001 #act
LEARNING_RATE = 0.05 
ENV_ID = "Drx-v1"
# energy consumption here use average current
# Energy usage, the unit is uA(current)(to fix!!!)
gold = 0
gold2 = 0 
 
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
    gold = np.percentile(probabilities, 80)
    gold2 = np.percentile(probabilities, 40)
    rows = 1000
    columns = 24  
    matrix = np.zeros((rows, columns))
 
    for col, prob in enumerate(probabilities):
        matrix[:, col] = np.random.choice([0, 1], size=(rows,), p=[1-prob, prob])
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
 


class NewAct(nn.Module):
    def __init__(self, pretrained_model):
        super(NewAct, self).__init__()
        
        # Add a layer to handle 3-dimensional input and produce 2-dimensional output
        self.new_layer = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 2),
            nn.Sigmoid(),
            nn.Linear(2, 2, bias=False),
        )
        nn.init.constant_(self.new_layer[-1].weight, 100)
        # Use the remaining parts of the pretrained model
        self.pretrained = pretrained_model
  
    def forward(self, x): 
        x = self.new_layer(x)  # Convert 3D input to 2D
        x[:, 0] = 20
        return self.pretrained(x)  # Process with the pretrained model
 
class NewCrt(nn.Module):
    def __init__(self, pretrained_critic):        
        super(NewCrt, self).__init__()
        # Preprocessing layer
        self.preprocess = nn.Sequential(
            nn.Linear(2, 32),  # 2 is the original input size of DDPGCritic
            nn.LeakyReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid(),
            nn.Linear(2, 2, bias=False),
        )
        nn.init.constant_(self.preprocess[-1].weight, 100)                
        # Pre-trained critic
        self.pretrained_critic = pretrained_critic

    def forward(self, x, a):
        x_transformed = self.preprocess(x)
        x_transformed[:, 0] = 20
        return self.pretrained_critic(x_transformed, a)
                    

def transfer(records,trail,dsmap,act_net,gold):
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
        #position = np.random.choice(len(action))
        if np.random.uniform(0,1) < 0.7:
            noise = np.random.uniform(-2, 2,size=action.shape)
            action += noise
        #position = np.random.choice(len(action))
        #noise = np.random.uniform(-2, 2)
        #action[position] += noise
        action = np.clip(action, -1, 1)
        action[0] = -1
        actions.append(action) 
        #if (state != 900000):
        #    print(obs[1], action)    
        env.fillin(obs,state)
        obs_, reward, done, info = env.step(action)
        #cur_time += 900000
        cur_time = info[0]
 
        latency = info[1]
        energy = info[2] 
        standard = info[3]
        #reward shapping
   
        reward = 0
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
 
        if belief > gold:
            if max(act_,act3) > 20000:
                #standard < energy 
                reward = (20000-max(act_,act3))/900000 
            else:
                #standard > energy
                reward = (1-(20000-max(act_,act3))/900000) + standard/energy
                # t1 up energy up reward down                
        else: 
            reward = (standard/energy) 
      
        '''
        if max(act_,act3) > 20000:
            reward = (standard/energy) + (belief>=gold)*(20000-max(act_,act3))/900000
        else:
            reward = (standard/energy) + (belief<gold)*(20000-max(act_,act3))/900000
        '''  
        rewards.append(reward)
        formatted_action = [float("{:.2f}".format(x)) for x in action]
        print(formatted_action,latency,"{:.0f}".format(energy), standard,"{:.4f}".format(reward))
        #print(formatted_action,reward)
        print('------------')

        if done:  
            obs = env.reset()
            break
    return torch.tensor(states),torch.tensor(actions),torch.tensor(rewards)

        
def test(records,trail,dsmap,act_net,L,uL,S,E,gold):
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
        #if (state != 900000):
        #    print(obs[1], action)
        env.fillin(obs,state)
        obs_, reward, done, info = env.step(action)
        cur_time = info[0]
        if state != 900000 and obs[1] > gold*100:
            L.append(info[1])
        elif state != 900000 and obs[1] < gold*100:
            #print(state,obs[1])
            uL.append(info[1])
            E.append(info[2])
            S.append(info[3])
        if done:
            obs = env.reset()
            break
    return
 
  
         
if __name__ == "__main__":
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
    #Lock the parameters of the pre-trained model
    #for param in pact_net.parameters():
    #        param.requires_grad = False
    #act_net = NewAct(pact_net)    
    act_net = pact_net
    #for param in pcrt_net.parameters():
    #    param.requires_grad = False
    #crt_net = NewCrt(pcrt_net)
    crt_net = pcrt_net


    writer = SummaryWriter(comment="-ddpg_" + "transfer_fine")
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

    #print(output_numbers[0],output_numbers[1])
    #print(dsmap[0],dsmap[1])
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=1) as tb_tracker:
            # transfer trainning
            for R in range(10): 
                records, gold, gold2 = genrecord(R,output_numbers)        
                dsmap_ = dsmap[R]
                for trail in range(100):
                    print("trail:",trail)
                    states_v, actions_v, rewards_v = transfer(records,trail,dsmap_,act_net,gold2)
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
                
                with multiprocessing.Manager() as manager:
                    lock = multiprocessing.Lock()
                    L = manager.list()
                    uL = manager.list()
                    E = manager.list()
                    S = manager.list()
                    processes = []
                    for trail in range(10):
                        print("test trail:",trail)
                        process = multiprocessing.Process(target=test, args=(records,trail, dsmap_, act_net, L,uL,S,E,gold2))
                        processes.append(process)
                        process.start()


                    # Waiting for all processes to finish
                    for process in processes:
                        process.join() 
                                                          
                    if not L or not S or not E:
                        print("L NULL")
                    else:
                        print("latency:",L)
                        print(len(L), len(uL))
                        print("average latency:", np.mean(list(L)),"max latency:",max(list(L)))
                        print("energy saved:", 1-sum(E)/sum(S)) if sum(E)<sum(S) else  print("energy waste:", 1-sum(S)/sum(E))

