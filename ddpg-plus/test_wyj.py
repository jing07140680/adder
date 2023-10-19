#!/usr/bin/env python3
import argparse
import gym
import drx_env_plus
import random
from lib import model
import math
import numpy as np
import torch
 
ENV_ID = "Drx-v1"
# energy consumption here use average current
# Energy usage, the unit is uA(current)(to fix!!!)
CE = 35000 # energy consumed when fully connected
IE = 2620  # energy consumed when connected in edrx idle
BE = 5  # Basic energy consumed when released in edrx idle
SE = 2.7  # Sleep energy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=False, help="Model file to load")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment name to use, default=" + ENV_ID)
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    args = parser.parse_args()
    env = gym.make(args.env)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)
  
    net = model.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0])
    net.load_state_dict(torch.load(args.model))
 

    #dsmap
    dsmap = []
    dsf = open("/adder/ddpg-plus/dsmap.txt")
    Lines = dsf.readlines()
    for line in Lines:
        dsmap.append([float(x) for x in line.strip().split(" ")])
        dsf.close()
  
    #dspmap
    dspmap = []
    dspf = open("/adder/ddpg-plus/dspmap.txt")
    Lines = dspf.readlines()
    for line in Lines:
        dspmap.append([float(x) for x in line.strip().split(" ")])
        dspf.close()
    for trail in range(len(dsmap)):
        rf = 0 
        obs = [20,3*dspmap[trail][0]*100]#,(rf%1024)/10]
        done = 0
        P = []
        L = []
        E = []
        S = []
        res = 0
        total_reward = 0
        total_steps = 0
        while True:
            print("obs:",obs)
            obs_v = torch.FloatTensor([obs])
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.numpy()
            env.fillin(obs)
            print(action)
            obs, reward, done, info = env.step(action)
            # print(info)
            # print(math.floor(info[0]/1000/60))
            obs = [20,3*dspmap[0][math.floor(info[0]/1000/60)]*100]#, obs[2]]
            if info[4] != 900000:
                L.append(info[1])
            P.append(3*dspmap[0][math.floor(info[0]/1000/60)]*100)
            E.append(info[2])
            S.append(info[3])
            
            total_reward += reward
            total_steps += 1
            if done:
                obs = env.reset()
                break

        if not L:
            continue
        #print("In %d steps we got %.3f reward" % (total_steps, total_reward))
        print(L)
        print(E)
        print(S)
        print(P)
        print(len(E))
        #print("average latency:", np.mean(L),"max latency:",max(L))
        '''
        aL = []
        aP = []
        for r in range(len(L)):
            if L[r] < 20000:
                aL.append(L[r])
            else:
                aP.append(P[r]) 
        print("average latency during peak hours:",np.mean(aL))
        print("could miss:", aP)
        '''
        print("energy saved:", 1-sum(E)/sum(S)) if sum(E)<sum(S) else  print("energy waste:", 1-sum(S)/sum(E))
        '''
        for r in range(len(E)):
            standard = S[r]*20000/np.mean(L)
            #print(E[r],20000/np.mean(L),S[r],standard)
            if E[r] < standard:
                res += 1-E[r]/standard
            else:
                res -= 1-standard/E[r]
        
        print("energy saved:",res/len(E))
        '''
