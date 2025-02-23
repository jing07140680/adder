#!/usr/bin/env python3
import gym
import drx_env_plus
import numpy as np
import random
import argparse
import time

ENV_ID = "Drx-v1"
t1 = 0.5
t2 = -0.8
t3 = -0.9
t4 = -0.9

###################### SIMUTLATOR TEST ########################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    env = gym.make(ENV_ID, debug=True, timelineplot=True)
    parser.add_argument("-t1","--t1", default=0.5, help='rrc release')
    parser.add_argument("-t2","--t2", default=0.2, help='T3324')
    parser.add_argument("-t3","--t3", default=-0.2,  help='edrx_cycle')
    parser.add_argument("-t4","--t4", default=0.1, help='PSM')
    args = parser.parse_args()
    #env.reset()
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
