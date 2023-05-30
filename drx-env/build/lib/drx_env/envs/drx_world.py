import gym
from gym import spaces
import numpy as np
import random
new_min = 1
new_max = 14
np.random.seed(34)
SE = 1
IE = 10
def trafficpattern():
    # Set mean and standard deviation
    mu, sigma = 0, 1
    
    # Generate random data from Gaussian distribution
    data = np.random.normal(mu, sigma, 1000)
    
    # Compute histogram bins and frequencies
    n_bins = 96
    freq, bins = np.histogram(data, bins=n_bins)
    # Compute bin width and normalize frequency values
    bin_width = bins[1] - bins[0]
    prob_density = freq / (np.sum(freq) * bin_width)
    prob_density = list(np.array(prob_density)*100)
    return prob_density
      

def binomial_probability(p):
    result = random.random() <= p
    return result
 
def gentraffic(belief,uplinktime):
    has_traffic = binomial_probability(belief)
    #has_traffic = belief
    downlinktime = float(np.random.uniform(low=1, high=15, size=(1,))[0]) if has_traffic else uplinktime
    return downlinktime
     
 
class DRXEnv(gym.Env):
    def __init__(self, render_mode=None):

        self.observation_space = spaces.Box(0, 100, shape=(2,1), dtype=float)
        #self.observation_space = spaces.Box(0, 100, shape=(2,1), dtype=int)
        #self.observation_space = spaces.Dict({"time": spaces.Box(0, 100, shape=(1,), dtype=int), "downlink": spaces.Discrete(2)})
          
        self.time = 0
        self.uplinktime = 15

        #for testing
        self.pattern = trafficpattern()
        self.belief = self.pattern[int(self.time//15)]

        #for training
        #self.belief = int(np.random.uniform(low=0, high=100, size=(1,))[0])

        self.state = gentraffic(self.belief, self.uplinktime)
          
        # We have 2 actions
        self.action_space = spaces.Box(1, 30, shape=(3,1), dtype=int)
        #self.action_space = spaces.Tuple((spaces.Box(1, 15, shape=(2,1), dtype=int), spaces.Box(1, 30, shape=(1,),dtype=int)))
    def _get_obs(self):
        return [self.belief, 0.2]# SE, IE] #self.time] 
 
    def reset(self, seed=None, options=None): 
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.time = 0

        #for testing
        self.belief = self.pattern[int(self.time//15)]

        #for training
        #self.belief = int(np.random.uniform(low=0, high=100, size=(1,))[0])

        self.state = gentraffic(self.belief,self.uplinktime)
        observation = self._get_obs()
        return observation
 
    def delay(self, interval, action):
        recv_time = 0
        sleeptime = 0
        idletime = 0
        latency = 0
        if interval == 0:
            return 0, 0, 0
        
        if __debug__:
            print("inter:",interval,"act:",action)

        while interval> 0:
            if interval < action[0]:
                latency = action[0]-interval
                sleeptime += action[0]
                recv_time += action[0]
                energy = sleeptime*SE+idletime*IE
                if __debug__:
                    print('sleep_t:',sleeptime)
                    print('idle_t:',idletime)
                    print('recv_t:',recv_time)
                    print('latency:',latency)
                    print("energy:",energy)
                return latency, energy, recv_time
            
            sleeptime += action[0]
            interval = interval - action[0]
            recv_time += action[0]

            # start here
            if interval < action[1]:
                while interval > 0:
                    if interval <2*action[2]/60:
                        if interval < action[2]/60:
                            latency = 0
                            recv_time += interval
                            idletime += 2*interval
                            energy = sleeptime*SE+idletime*IE
                            if __debug__:
                                print('sleep_t:',sleeptime)
                                print('idle_t:',idletime)
                                print('recv_t:',recv_time)
                                print('latency:',latency)
                                print("energy:",energy)
                            return latency, energy, recv_time
                        else:
                            latency = (2*action[2]/60 - interval)
                            idletime += 2*action[2]/60
                            recv_time += 2*action[2]/60
                            energy = sleeptime*SE+idletime*IE
                            if __debug__:
                                print('sleep_t:',sleeptime)
                                print('idle_t:',idletime)
                                print('recv_t:',recv_time)
                                print('latency:',latency)
                                print("energy:",energy)
                            return latency, energy, recv_time
                        
                    idletime += 2*action[2]/60
                    interval -= 2*action[2]/60
                    recv_time += 2*action[2]/60

            else:
                n = (action[1]*60)//action[2]
                if n % 2 == 0:
                    idletime += (n*action[2]+(action[1]*60)%action[2])/60
                else:
                    idletime += (n+1)*action[2]/60
                recv_time += action[1]
                interval -= action[1]
                
            '''
            if interval < action[1]:
                latency = 0 if interval%2 == 0 else interval%2
                #print(latency)
                idletime += interval + latency
                recv_time += interval + latency
                break
            idletime += action[1]
            recv_time += action[1]
            interval = interval - action[1]
            '''

        energy = sleeptime*SE+idletime*IE
        if __debug__:
            print('sleep_t:',sleeptime)
            print('idle_t:',idletime)
            print('recv_t:',recv_time)
            print('latency:',latency)
            print('energy:',energy)
        return latency, energy, recv_time
    
    def step(self, action, dt=0):
        if __debug__:
            print('belief:',self.belief,'downlink:',self.state ,'action:',action)
        reward = 0
        latency = 0
        energy = 0

        act = [0,0,0]
        act[0] = int(((action[0] - (-1)) / (1 - (-1))) * (new_max - new_min) + new_min)
        act[1] = int(((action[1] - (-1)) / (1 - (-1))) * (new_max - new_min) + new_min)
        act[2] = int(((action[2] - (-1)) / (1 - (-1))) * (30 - new_min) + new_min)
        
        #print(action,act)
        uplinktime = self.uplinktime
        downlinktime = self.state
        belief = self.belief

        # next_state
        #for testing
        self.belief = self.pattern[int(self.time//15)]

        #for training
        self.belief = int(np.random.uniform(low=0, high=100, size=(1,))[0])

        observation = self._get_obs()
        self.state = gentraffic(self.belief,uplinktime)
 

         # clip drx parameters
        if act[0] >= uplinktime:
            act[0] = uplinktime
            act[1] = 0
        else:
            act[1] = min(uplinktime-act[0],act[1])
 
        
        if downlinktime <= uplinktime:
            #do not have to consider uplink condition
            latency, energy, recv_time = self.delay(downlinktime,act)

        # In the 15 mins, no downlink or downlink at the 15th min
        else:
            #latency,recv_time = self.delay(downlinktime%uplinktime,action)
            #recv_time += uplinktime*(downlinktime//uplinktime)
            recv_time = uplinktime
            
        self.time = self.time + recv_time
        
        terminated = 1 if self.time >= 60*24 else 0  
 
        if latency > 0.2:
            reward = -belief*(latency/recv_time)*50
        else:
            reward = -(100-belief)*(energy/(recv_time*max(SE,IE)))
        if __debug__:
            print("reward:",energy, recv_time*max(SE,IE))
            
        '''
        if belief > 30:
            reward = -latency/recv_time
        else:
            reward = -energy/(recv_time*10)
        '''
        '''
        if latency > 2:
            #print(belief)
            reward = belief * (1-latency/15) # unstable
        else: 
            #reward += energy/15*15 #unstable
            reward = (100-belief) * (1-energy/(recv_time*10))
        '''    
        return observation, reward, terminated, [latency,energy,recv_time*max(SE,IE)]
                                                 
'''
env = DRXEnv()
env.reset()
for i in range(10):
    print(env.step([5,3]))
''' 
    
 
 
