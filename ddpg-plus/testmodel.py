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
LEARNING_RATE_ = 0.0005
LEARNING_RATE = 0.0005
ENV_ID = "Drx-v1"

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
    #pcrt_net = model.DDPGCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    #pact_net = model.DDPGActor(5, env.action_space.shape[0]).to(device)
    pact_net.load_state_dict(torch.load(args.amodel))
    #pcrt_net.load_state_dict(torch.load(args.cmodel))

    for i in range(10):
        print("act:", pact_net(torch.FloatTensor([40,i*10])))

   #print("act:", pact_net(torch.FloatTensor([3,3,3,3,3])))
