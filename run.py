import gym
import drx_env
import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument('a0', type=str, help='action0')
#parser.add_argument('a1', type=str, help='action1')

#args = parser.parse_args()
#a0 = args.a0
#a1 = args.a1



env = gym.make('Drx-v0')

for episode in range(10):
    env.reset()
    print("episode:", episode)
    for i in range(100):
        ob,reward,terminated,_ = env.step([-0.9, 0.9, -1])
        print(ob,reward)
        if terminated == 1:
            break
