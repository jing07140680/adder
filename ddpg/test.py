#!/usr/bin/env python3
import argparse
import gym
import drx_env
from lib import model

import numpy as np
import torch

ENV_ID = "Drx-v0"


if __name__ == "__main__":
    L = []
    E = 0
    S = 0
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment name to use, default=" + ENV_ID)
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    args = parser.parse_args()

    spec = gym.envs.registry.spec(args.env)
    #spec._kwargs['render'] = False
    env = gym.make(args.env)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)
 
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
        #print(info)
        L.append(info[0])
        E += info[1]
        S += info[2]
        total_reward += reward
        total_steps += 1
        if done:
            break
    print("In %d steps we got %.3f reward" % (total_steps, total_reward))
    print("average latency:", np.mean(L),"max latency:",max(L))
    print("energy saved:", 1-E/S)
