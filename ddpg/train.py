#!/usr/bin/env python3
import os
import ptan
import time
import gym
import pybullet_envs
import argparse
from tensorboardX import SummaryWriter
import numpy as np
import drx_env
import random

from lib import model, common

import torch
import torch.optim as optim
import torch.nn.functional as F
 

ENV_ID = "Drx-v0"
GAMMA = 0.9
BATCH_SIZE = 64
#LEARNING_RATE = 0.00005
#LEARNING_RATE_ = 0.000005
REPLAY_SIZE = 10000
REPLAY_INITIAL = 1000

TEST_ITERS = 1000
new_min = 1
new_max = 14

def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)   
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            #print("test:",obs_v, action)
            #action = np.clip(action, 1, 15)
            #action[0] = int(((action[0] - (-1)) / (1 - (-1))) * (new_max - 1) + 1)
            #action[1] = 15-action[0]
            #print('state:',obs_v,"act:",action)
            obs, reward, done, _ = env.step(action)
            act = [0,0]
            act[0] = int(((action[0] - (-1)) / (1 - (-1))) * (new_max - 1) + 1)
            act[1] = int(((action[1] - (-1)) / (1 - (-1))) * (new_max - new_min) + new_min)

            print('state:',obs_v,"act:",act,"reward:",reward)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count

  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-a", "--LEARNING_RATE", required=True,type=int, help="actor learning rate")
    parser.add_argument("-c", "--LEARNING_RATE_", required=True,type=int, help="critic learning rate")
     
    args = parser.parse_args()
    LEARNING_RATE = args.LEARNING_RATE/1000000
    LEARNING_RATE_ = args.LEARNING_RATE_/100000000
    device = torch.device("cuda" if args.cuda else "cpu")
 
    save_path = os.path.join("saves", "ddpg-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = gym.make(ENV_ID)
    test_env = gym.make(ENV_ID)

    #print(env.observation_space.shape[0])

    
    act_net = model.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    crt_net = model.DDPGCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    
    tgt_act_net = ptan.agent.TargetNet(act_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)

    writer = SummaryWriter(comment="-ddpg_" + args.name)
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
 
                #vectors from batch
                #print("state:",states_v,"action:", actions_v,"reward:", rewards_v)
                # train critic
                crt_opt.zero_grad()
                q_v = crt_net(states_v, actions_v)
                #last_act_v = tgt_act_net.target_model(last_states_v)
                #q_last_v = tgt_crt_net.target_model(last_states_v, last_act_v)
                #q_last_v[dones_mask] = 0.0

                q_ref_v = rewards_v.unsqueeze(dim=-1)# + q_last_v * GAMMA
                critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                critic_loss_v.backward()
                crt_opt.step()
                tb_tracker.track("loss_critic", critic_loss_v, frame_idx)
                tb_tracker.track("critic_ref", q_ref_v.mean(), frame_idx)
  
                # train actor 
                act_opt.zero_grad()
                cur_actions_v = act_net(states_v)
                actor_loss_v = -crt_net(states_v, cur_actions_v) #we need to update the actor's weights in a directionthat will decrease the critic's output.

                actor_loss_v = torch.mul(actor_loss_v, states_v)
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()
                act_opt.step()
                tb_tracker.track("loss_actor", actor_loss_v, frame_idx)
 
                tgt_act_net.alpha_sync(alpha=1 - 0.001)
                tgt_crt_net.alpha_sync(alpha=1 - 0.001)
  
                #print("frame_idx:",frame_idx)
                if frame_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = test_net(act_net, test_env, device=device) 
                    print("Test done in %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, frame_idx)
                    writer.add_scalar("test_steps", steps, frame_idx)
                    if best_reward is None or best_reward < rewards:  
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, frame_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(act_net.state_dict(), fname)
                        best_reward = rewards

    pass
 
 
