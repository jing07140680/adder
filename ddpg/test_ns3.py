#!/usr/bin/env python3
from lib import model, common
import os
import ptan
import time

import torch
import torch.optim as optim
import torch.nn.functional as F

import argparse
from ns3gym import ns3env
import collections
from itertools import product
import gym
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Concatenate
from keras.optimizers import Adam
import keras.backend as K
from tensorboardX import SummaryWriter
import tensorflow as tf
import math
import random
from collections import deque
import pdb
import time

if __name__ == "__main__":
    L = []
    E = 0
    S = 0
    port = 5555
    stepTime = 0.5
    startSim = 0
    debug = 0
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    args = parser.parse_args()
    env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, debug=debug)
    net = model.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0])
    net.load_state_dict(torch.load(args.model))

    obs = env.reset()
    total_reward = 0.0
    total_steps = 0


    while True:
        obs_v = torch.FloatTensor([obs])
        mu_v = net(obs_v)
        action = mu_v.squeeze(dim=0).data.numpy()
        action = np.clip(action, -1, 1)

        obs, reward, done, info = env.step(action)
        print(info)
        info = info.split('|')
        print(info)
        L.append(float(info[1]))
        E += float(info[2])
        S += float(info[3])
        total_reward += reward
        total_steps += 1
        if info[0]=='done':
            break
    print("In %d steps we got %.3f reward" % (total_steps, total_reward))
    print("average latency:", np.mean(L),"max latency:",max(L))
    print("energy saved:", 1-E/S)
