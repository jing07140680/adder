#!/usr/bin/env python3
import gym
import drx_env_plus
#import drx_env_AC
import numpy as np
import random
import argparse
import time

ENV_ID = "Drx-v1"
t1 = -1
t2 = -1
t3 = -1
t4 = -1
test = 0

###################### SIMUTLATOR TEST ########################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t1","--t1", default=0.5, help='rrc release')
    parser.add_argument("-t2","--t2", default=0.2, help='T3324')
    parser.add_argument("-t3","--t3", default=-0.2,  help='edrx_cycle')
    parser.add_argument("-t4","--t4", default=0.1, help='PSM')
    parser.add_argument("-T","--test", default=1, type = int, help='Enable test mode')
    parser.add_argument("-B","--belief",default=0, help='test belief')
    args = parser.parse_args()
    t1 = float(args.t1)
    t2 = float(args.t2)
    t3 = float(args.t3)
    t4 = float(args.t4)
    test = args.test
    belief = float(args.belief)
    print(t1,t2,t3,t4,test,belief)
    env = gym.make(ENV_ID, debug=True, timelineplot=False, test=test, belief = belief)
    env.reset()
    ob,reward,terminated,_ = env.step([t1,t2,t3,t4])
    print(ob,reward)

 

################### RUNTIME TEST #####################
"""
np.random.seed(0)

if __name__ == "__main__":
    start_time = time.time()  # record the start time
    env = gym.make(ENV_ID, debug=False, timelineplot=False)
    cnt = 0
    while True:
        random_numbers = np.random.uniform(-1, 1, 4)
        ob,reward,terminated,_ = env.step(random_numbers)
        cnt += 1
        if terminated == 1:
            break
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Time elapsed:', elapsed_time, "cnt:",cnt)
    
"""
