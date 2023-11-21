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

ENV_ID = "Drx-v1"
# energy consumption here use average current
# Energy usage, the unit is uA(current)(to fix!!!)
CE = 35000 # energy consumed when fully connected
IE = 2620  # energy consumed when connected in edrx idle
BE = 5  # Basic energy consumed when released in edrx idle
SE = 2.7  # Sleep energy
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
        
def test(records,trail,dsmap,net,L,uL,S,E,lock, gold, gold2):
    record = genrow(records[trail])
    beliefstate = dsmap[0]
    cur_time = 0
    done = 0
    res = 0
    while True: 
        belief = max(beliefstate[math.floor(cur_time/60000) : math.floor(cur_time/60000)+15])
        #print(belief)
        if belief > gold:
            #belief = 0.73
            belief = 0.73
        elif gold2 < belief <= gold:
            #belief = 0.71
            belief = 0.7 
        else: 
            #belief = 0.3
            belief = 0
        obs = [20, (belief)*100]
        state = genTraffic(record,cur_time)
        obs_v = torch.FloatTensor([obs])
        mu_v = net(obs_v)
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
    parser.add_argument("-m", "--model", required=False, help="Model file to load")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment name to use, default=" + ENV_ID)
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    args = parser.parse_args()
    env = gym.make(args.env)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record) 
    net = model.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0])
    net.load_state_dict(torch.load(args.model))
  
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

    '''
    # load prediction
    records = []
    rcdf = open("/adder/scooterdata/records.txt")
    Lines = rcdf.readlines()
    rcdf.close()
    for k, line in enumerate(Lines):
        if k%2 == 0:
            continue
        else:
            tmprecords = [float(x) for x in line.strip().split(",")]
            tmprecords.insert(0,0)
            records.append([60000*(tmprecords[i]+tmprecords[i-1]) for i in range(1,len(tmprecords))])
    '''

    with multiprocessing.Manager() as manager:
        lock = multiprocessing.Lock()
        L = manager.list()
        uL = manager.list()
        E = manager.list()
        S = manager.list()
        processes = []

        for R in range(2): 
            records, gold, gold2 = genrecord(R,output_numbers)        
            # start motivation
            for loop in range(1):
                for trail in range(20):#len(records)):
                    #print(trail)
                    process = multiprocessing.Process(target=test, args=(records,loop*20+trail,dsmap,net,L,uL,S,E,lock,gold, gold2))
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

        
 
