#!/usr/bin/env python3
# -*- coding: utf-8 -*-import argparse
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

GAMMA = 0.9
BATCH_SIZE = 64
#LEARNING_RATE = 0.00005
#LEARNING_RATE_ = 0.000005
REPLAY_SIZE = 10000
REPLAY_INITIAL = 1000

def timer(func, *args, **kwargs):
      start = time.time()
      func(*args, **kwargs)
      end = time.time()
      print("time:",end-start)

def train(arg):
      frame_idx = 0 
      while frame_idx < 10:
            frame_idx += 1
            arg.populate(1)
            rewards_steps = exp_source.pop_rewards_steps()
             
GAMMA = 0.9
BATCH_SIZE = 64
LEARNING_RATE = 0.00005
LEARNING_RATE_ = 0.000005
REPLAY_SIZE = 10000
REPLAY_INITIAL = 1000
TEST_ITERS = 1000

parser = argparse.ArgumentParser(description='Start simulation script on/off')
parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
parser.add_argument('--start',
                    type=int,
                    default=0,
                    help='Start ns-3 simulation script 0/1, Default: 1')
parser.add_argument('--iterations',
                    type=int,
                    default=1,
                    help='Number of iterations, Default: 1')
parser.add_argument('--port',
                    type=int,
                    default=5555,
                    help='Number of iterations, Default: 5555')

parser.add_argument('--filename',
                    type=str,
                    default="/opt/NB-IoT/test.txt",
                    help='Number of iterations, Default: test.txt')

args = parser.parse_args()
device = torch.device("cuda" if args.cuda else "cpu")
    
startSim = bool(args.start)
iterationNum = int(args.iterations)
port = int(args.port)
filename = str(args.filename)

simTime = 200  # seconds
stepTime = 0.5  # seconds 
seed = 0
simArgs = {"--simTime": simTime,
           "--testArg": 123}
debug = True
values = collections.defaultdict(float)
#trial_len = []

'''
# open traffic pattern test.txt
f=open(filename)
Lines = f.readlines() 
print("filename: ",filename)

# num of trial in total
num_trial = int(len(Lines)/2)
print("total trial: ",num_trial)
'''

env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim,
                    simSeed=seed, simArgs=simArgs, debug=debug)
print("observation_space",env.observation_space.shape)
print("action_space",env.action_space.shape)

act_net = model.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
crt_net = model.DDPGCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)

tgt_act_net = ptan.agent.TargetNet(act_net)
tgt_crt_net = ptan.agent.TargetNet(crt_net)

writer = SummaryWriter(comment="ddpg-opengym-2")
agent = model.AgentDDPG(act_net, device=device)
exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)
buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE_)
crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)

frame_idx = 0
best_reward = None 



with ptan.common.utils.RewardTracker(writer) as tracker:
      with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            while True:
                  print(frame_idx)
                  frame_idx += 1
                   
                  buffer.populate(1) # call act_net agentDDPG

                  rewards_steps = exp_source.pop_rewards_steps()
                  if rewards_steps:
                        rewards, steps = zip(*rewards_steps)
                        tb_tracker.track("episode_steps", steps[0], frame_idx)
                        tracker.reward(rewards[0], frame_idx)

                  if len(buffer) < REPLAY_INITIAL:
                        continue

                  #reach the batch size, traning begin
                  batch = buffer.sample(BATCH_SIZE)
                  states_v, actions_v, rewards_v, dones_mask, last_states_v = common.unpack_batch_ddqn(batch, device)
                  #print("dones_mask",dones_mask)
                  # train critic
                  crt_opt.zero_grad()
                  q_v = crt_net(states_v, actions_v)
                  q_ref_v = rewards_v.unsqueeze(dim=-1)
                  critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                  critic_loss_v.backward()
                  crt_opt.step()
                  tb_tracker.track("loss_critic", critic_loss_v, frame_idx)
                  tb_tracker.track("critic_ref", q_ref_v.mean(), frame_idx)
                  # train actor
                  act_opt.zero_grad()
                  cur_actions_v = act_net(states_v)
                  actor_loss_v = -crt_net(states_v, cur_actions_v) 
 
                  actor_loss_v = torch.mul(actor_loss_v, states_v)
                  actor_loss_v = actor_loss_v.mean()
                  actor_loss_v.backward()
                  act_opt.step()
                  tb_tracker.track("loss_actor", actor_loss_v, frame_idx)
                  
                  tgt_act_net.alpha_sync(alpha=1 - 0.001)
                  tgt_crt_net.alpha_sync(alpha=1 - 0.001)

                  



'''
total_episode = 100
while total_episode:
      print(total_episode)
      timer(train,arg=buffer)
      total_episode -= 1
#n_s, r, done, _ = env.step([0,0,1,1])
#print(n_s,r,done,_)
'''
