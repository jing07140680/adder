#!/usr/bin/env python3 
import os
import ptan
import time
import gym 
import pybullet_envs
import argparse
from tensorboardX import SummaryWriter
import numpy as np
import drx_env_plus
import random
from lib import model, common
import torch
import torch.optim as optim
import torch.nn.functional as F
import sys
from ptan import experience
import multiprocessing
import time
import os


 
ENV_ID = "Drx-v1"
GAMMA = 0.9
BATCH_SIZE = 256 
#LEARNING_RATE = 0.000005
#LEARNING_RATE_ = 0.0000005
LEARNING_RATE = 0.0005
LEARNING_RATE_ = 0.00005
REPLAY_SIZE = 1000  
REPLAY_INITIAL = 100  
TEST_ITERS = 10  
max_minutes = 15 
max_subframes = max_minutes*60*1000
debug_lr = 0 
  
 
#class CustomBufferManager(BaseManager):
#    pass

def worker(local_buffer,exp_source, idx, shared_buffer):
    for i in range(5):
        local_buffer.populate(1,idx)
    for sample in local_buffer: 
        shared_buffer.share_add(sample,idx)
        
def test_net(net, env, count=1, device="cpu"):
    rewards = 0.0 
    steps = 0
    for _ in range(count):
        obs = env.reset()
        for step in range(20):
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            obs, reward, done, info = env.step(action)
            act = [0]*4
            act[0] = int(((action[0] - (-1)) / (1 - (-1))) * 29999 + 1) 
            act[1] = int(((action[1] - (-1)) / (1 - (-1))) * (max_subframes-10)+10)
            act[2] = int(((action[2] - (-1)) / (1 - (-1))) * (max_subframes/10-1)+1)
            act[3] = int(((action[3] - (-1)) / (1 - (-1))) * (max_subframes))
            #print('cur state:', obs_v, 'next state:',obs, 'cur act:',act,'cur reward:',reward, 'cur T_d:',info)
            print("obs:",obs_v[0][1],"T_d:",info[-1],"action:",act, "reward:",reward)
            rewards += reward
            steps += 1 
            if done: 
                break 
    return rewards / count, steps / count
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA') 
    parser.add_argument("-n", "--name", required=True, help="Name of the run") 
    parser.add_argument("-m", "--mode", default=0, type=int,  help="mode: 0: random 1: periodic, 2: periodic w/ congestion")
    #parser.add_argument("-a", "--LEARNING_RATE_", required=True,type=int, help="actor learning rate")
    #parser.add_argument("-c", "--LEARNING_RATE", required=True,type=int, help="critic learning rate")
 
    
    args = parser.parse_args()
    #LEARNING_RATE = args.LEARNING_RATE/1000000
    #LEARNING_RATE_ = args.LEARNING_RATE_/1000000
    device = torch.device("cuda" if args.cuda else "cpu")
    save_path = os.path.join("saves", "ddpg-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    print(args.mode)
    if args.mode == 0:
        train_ = 0
        test_ = 1
    elif args.mode == 1:
        train_ = 2
        test_ = 2
    else:
        train_ = 0
        test_ = 1
        
    print("train:", train_, "test:", test_)
    env = gym.make(ENV_ID, test=train_)
    test_env = gym.make(ENV_ID, test=test_) 

    #print(env.observation_space.shape[0])
    #print(len(env.action_space.spaces))
    act_net = model.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    crt_net = model.DDPGCritic(env.observation_space.shape[0],env.action_space.shape[0]).to(device)
    #act_net = model.DDPGActor(env.observation_space.shape[0], len(env.action_space.spaces)).to(device)
    #crt_net = model.DDPGCritic(env.observation_space.shape[0],len(env.action_space.spaces)).to(device)
     
    tgt_act_net = ptan.agent.TargetNet(act_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)
    
    writer = SummaryWriter(comment="-ddpg_" + args.name)
    agent = model.AgentDDPG(act_net, device=device)
   
    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE_)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1) 
    shared_buffer = experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    
    frame_idx = 0
    best_reward = None
 
 
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=1) as tb_tracker:
            while True:
                print("frame_idx:",frame_idx)
                #buffer one step 
                #shared_buffer.populate(1) # call act_net agentDDPG
                processes = []
                buffers = []
                for i in range(20):
                    env_ = gym.make(ENV_ID, test=train_)
                    exp_source = ptan.experience.ExperienceSourceFirstLast(env_, agent, gamma=GAMMA, steps_count=1)
                    buffer =experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
                    buffers.append(buffer)
                    p = multiprocessing.Process(target=worker, args=(buffer, exp_source, i, shared_buffer)) 
                    processes.append(p)
                    p.start()
                      
                #start_time = time.time()
                while shared_buffer.size() < REPLAY_INITIAL:
                    pass

                #end_time = time.time()
                #print("size:",shared_buffer.size())
                #print("time:", end_time-start_time)
                for p in processes:
                    p.join()

                for buffer in buffers:
                    buffer.close()
                #print("size:",shared_buffer.size())
                #end_time = time.time()
                #print("time:", end_time-start_time)

                '''
                if shared_buffer.size() < REPLAY_INITIAL:
                    continue
                '''

                #reach the batch size, traning begin
                batch = shared_buffer.share_sample(BATCH_SIZE) 
                frame_idx += 1
                states_v, actions_v, rewards_v, dones_mask, last_states_v = common.unpack_batch_ddqn(batch, device)
                #for o in range(10):
                #    print("state:",states_v[o][1],"act:",actions_v[o],"R:",rewards_v[o])
                 
                #vectors from batch
                #print("state:",states_v,"action:", actions_v,"reward:", rewards_v)
                # train critic
                crt_opt.zero_grad()
                q_v = crt_net(states_v, actions_v)
                #last_act_v = tgt_act_net.target_model(last_states_v)
                #q_last_v = tgt_crt_net.target_model(last_states_v, last_act_v)
                #q_last_v[dones_mask] = 0.0
                #print(rewards_v)
                q_ref_v = rewards_v.unsqueeze(dim=-1)# + q_last_v * GAMMA
                critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                #print("q_ref_v.detach:", q_ref_v.detach())
                #print("critic_loss_v:",critic_loss_v)
                critic_loss_v.backward()
                crt_opt.step()
                tb_tracker.track("loss_critic", critic_loss_v, frame_idx)
                tb_tracker.track("critic_ref", q_ref_v.mean(), frame_idx)
    
                # train actor 
                act_opt.zero_grad()
                #print(states_v)
                cur_actions_v = act_net(states_v)
                #print(cur_actions_v) 
                actor_loss_v = -crt_net(states_v, cur_actions_v) #we need to update the actor's weights in a directionthat will decrease the critic's output.
                #print('actor_loss_v',actor_loss_v)
                #print(states_v)
                actor_loss_v = torch.mul(actor_loss_v, states_v)
                #print('actor_loss_v',actor_loss_v)
                actor_loss_v = actor_loss_v.mean() 
                #print('actor_loss_v',actor_loss_v) 
                actor_loss_v.backward()

                # debug the parameters change in each layer and gradients in each layer
                if debug_lr:
                    #for name, param in act_net.named_parameters():
                    for name, param in act_net.named_parameters():
                        if name == "net.2.bias":
                            print(param.grad)
                            print(param.data)
                act_opt.step()
                # debug the parameters change in each layer and gradients in each layer
                if debug_lr:
                    # Print gradients
                    for name, param in act_net.named_parameters(): 
                        if name == "net.2.bias":
                            print("after gradients")
                            print(param.data)
                    
                tb_tracker.track("loss_actor", actor_loss_v, frame_idx)
                tgt_act_net.alpha_sync(alpha=1 - 0.001)
                tgt_crt_net.alpha_sync(alpha=1 - 0.001)

                
                # Go To Test
                if frame_idx % TEST_ITERS == 0:
                    print("Test iters:")
                    ts = time.time() 
                    rewards, steps = test_net(act_net, test_env, device=device) 
                    print("Test done in %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, frame_idx)
                    writer.add_scalar("test_steps", steps, frame_idx)
                    if best_reward is None or best_reward < rewards:  
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            actname = "best_%+.3f_%d_act.dat" % (rewards, frame_idx)
                            crtname = "best_%+.3f_%d_crt.dat" % (rewards, frame_idx)
                            actfname = os.path.join(save_path, actname)
                            crtfname = os.path.join(save_path, crtname)
                            torch.save(act_net.state_dict(), actfname)
                            torch.save(crt_net.state_dict(), crtfname)
                        best_reward = rewards
                    if frame_idx % 100 == 0: 
                        print("Newest reward updated")
                        actname = "newest_act.dat"
                        crtname = "newest_crt.dat"
                        actfname = os.path.join(save_path, actname)
                        crtfname = os.path.join(save_path, crtname)
                        torch.save(act_net.state_dict(), actfname)
                        torch.save(crt_net.state_dict(), crtfname)
