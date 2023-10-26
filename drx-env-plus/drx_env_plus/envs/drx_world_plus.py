import gym
from gym import spaces
import numpy as np
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import copy


#np.random.seed(34)
sf = 0  # subframe
rf = 0  # radio frame
max_minutes = 15
max_subframes = max_minutes*60*1000
#action_to_po = {0: 4, 1: 9}
max_delay = 20000
 
 
# energy consumption here use average current
# Energy usage, the unit is uA(current)(to fix!!!)
CE = 35000 # energy consumed when fully connected
IE = 2620  # energy consumed when connected in edrx idle
BE = 5  # Basic energy consumed when released in edrx idle
SE = 2.7  # Sleep energy

 
def trafficpattern( pattern = 1, steps = 30):
    ''' 
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
    #print("trafficpattern:",prob_density)
    return prob_density
    '''
    np.random.seed(0)
    traffic = [random.randint(0, 100) for _ in range(steps)]
    return traffic


    
  
def binomial_probability(p):
    result = random.random() <= p
    return result
 
    
# Use belief to generate uplink traffic, if there is 0 uplink traffic,
# then uplink traffic will happen for GPS location update set to 15 min.
def gentraffic(belief, T_u):
    has_traffic = binomial_probability(belief/100)
    T_d = np.random.randint(1, max_subframes, 1)[0] if has_traffic else T_u
    #if belief < 80:
    #    T_d = T_u
    #else:
    #    T_d = np.random.randint(1, max_subframes, 1)[0]
    return T_d    
'''
def genfixed():
    num_columns = 20
    num_rows = 20
    matrix = np.zeros((num_rows, num_columns))
    for i in range(num_columns):
        matrix[:, i] = (np.random.rand(num_rows) < self.pattern[i]/100).astype(int)
    matrix[matrix == 0] = 300000
    matrix[matrix == 1] = 900000
    return matrix
'''
# periodic traffic
def genperiodic(): 
    return 300000
 
class DRXEnv(gym.Env):

    fig = make_subplots(rows=1, cols=1)
    x = []
    y = []

    def _set_obs(self):
        # Train mode: random
        if self.test == 0:
            #print("TEST mode")
            #self.pattern = trafficpattern()
            #self.belief = self.pattern[int(self.time//max_subframes)]
            self.belief = np.random.uniform(low=0, high=100, size=(1,))[0]
            self.state = gentraffic(self.belief, self.uplinktime)
             
        # test mode: fixed random 
        elif self.test == 1:
            self.belief = self.tmppattern.pop(0)
            #self.state = self.tmpavl.pop(0)
            self.state = gentraffic(self.belief, self.uplinktime)

        # mode: periodic
        elif self.test == 2:
            self.belief = 100
            self.state = genperiodic()
 
            
    def _get_obs(self):
        global rf
        return [20, self.belief]#, (rf%1024)/10]
                           
    def __init__(self, render_mode=None, debug=False, timelineplot=False, test=0, belief = 0):
        self.debug = debug 
        self.timelineplot = timelineplot
        self.test = test
          
        ########################## Observation ###############################
        # observation: 1. belif value 2. expected maximum latency
        self.observation_space = spaces.Box(0, 100, shape=(2, 1), dtype=float)
        # initial time
        self.time = 0
        # there is at least an uplink traffic each max_subframes
        self.uplinktime = max_subframes
 
        if self.test == 1:
            self.pattern = trafficpattern(pattern = 1, steps = 30)
            self.tmppattern = copy.deepcopy(self.pattern)
            print(self.pattern)
 
               
        self._set_obs()
        #print("init:", self.belief, self.state)        
        ########################## Action  #################################
        self.action_space = spaces.Box(-1, 1, shape=(4,1), dtype=float)
     
   
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)
        self.time = 0
        global sf
        global rf
        sf = 0
        rf = 0
        if self.test == 1:
            self.tmppattern = copy.deepcopy(self.pattern)
        self._set_obs() 
        observation = self._get_obs()
        return observation
 
    # return: latency, energy, recv_time
    # note: recv_time != T_d
    # PSM = T3412-T3324

    def simulator(self, T_d, act):
        rrc_release, T3324, edrx_cycle, PSM = act
        PO = 4
        T3412 = T3324+PSM
        Cycle = T3412+edrx_cycle
        global sf
        global rf
        nofcycle = T_d // (rrc_release+T3412)
        offset  = T_d % (rrc_release+T3412)
        tmp_nofcycle = nofcycle
        fig = self.fig
        cnt = 0

        
        def tplot(powLevel):
            if self.timelineplot:
                self.x.append(cur_time+cnt*(rrc_release+T3412))
                self.y.append(powLevel)
 
        energy = 0
        
        while tmp_nofcycle:
            
            ############ rrc_connected ##############
            cur_time = 0
            tplot(CE)
            energy += rrc_release*CE
            rf = (rf + rrc_release // 10 + (sf+ rrc_release % 10) // 10) % 1024
            sf = (sf + rrc_release) % 10
            cur_time = rrc_release
            tplot(CE)
            #print("rrc_connected:",energy)
            ############ rrc_idle #############
            flag = 0
            PF = 0
            while cur_time < rrc_release + T3324:
                if rf == PF and sf == PO:
                    flag = 1
                    PF = (PF + edrx_cycle)
                    energy += IE
                    tplot(IE)
                else:
                    energy += BE
                    tplot(BE)

                # update time
                cur_time += 1
                sf = (sf+1) % 10
                if sf == 0:
                    if flag:
                        rf += 1
                    else:
                        rf = (rf+1) % 1024
            #print("rrc_idle:",energy)
            tplot(SE)
            energy += PSM*SE
            cur_time += PSM
            rf = rf % 1024
            rf = (rf + PSM // 10 + (sf+ PSM % 10) // 10) % 1024
            sf = (sf + PSM) % 10
            tplot(SE)
            tmp_nofcycle -= 1
            cnt += 1
            #print("cycle:",energy)

        #print("end of cycle:",energy)
        cur_time = 0
        recv_time = 0
        triggered = 0
        connected = 0
        flag = 0
        PF = 0
        while not recv_time:
            if cur_time == offset:
                triggered = 1

            tmp_time = cur_time % (T3412+rrc_release)
            if tmp_time < rrc_release:
                connected = 1
                energy += CE
                tplot(CE)

            if rrc_release <= tmp_time < rrc_release + T3324: 
                if rf == PF and sf == PO:
                    flag = 1
                    PF = (PF + edrx_cycle) 
                    connected = 1 
                    energy += IE
                    tplot(IE)
                else:
                    connected = 0
                    energy += BE
                    tplot(BE)
  
            if rrc_release + T3324 <= tmp_time < rrc_release + T3324 + PSM:
                rf = rf % 1024
                connected = 0
                energy += SE
                tplot(SE)


            # print(T_d, cur_time, tmp_time, connected, triggered)
            # record the time when UE received downlink traffic 
            if triggered and connected:
                recv_time = cur_time + (rrc_release+T3412)*nofcycle
                #print("!!!", cur_time, (rrc_release+T3412)*nofcycle, nofcycle)
                break

            if triggered and cur_time + (rrc_release+T3412)*nofcycle ==900000:
                recv_time = 900000
                break
 
            # update time
            cur_time += 1 
            sf = (sf+1) % 10
            if sf == 0:
                if flag == 0:
                    rf = (rf+1) % 1024
                else:
                    rf += 1
                    
        latency = recv_time - T_d
 
        if self.debug: 
            print('T_d:', T_d)
            print('recv_t:', recv_time)
            print('latency:', latency)
            print('energy:', energy)
 
        if self.timelineplot:
            self.fig.update_xaxes(range=[0, 900000])
            self.fig.update_layout(
                title='LTE-M DRX Energy and Traffic monitor per 15 mintues',  # Set the main plot title
                title_x=0.5,  # Set the x-position of the main title (0.5 centers the title)
            )
            # Set axis titles
            self.fig.update_xaxes(title_text='Time (subframes)')  # Set the title of the x-axis
            self.fig.update_yaxes(title_text='Energy')  # Set the title of the y-axis
            
            self.fig.add_scatter(x=self.x, y=self.y, )
            #self.fig.add_scatter(x=[T_d, T_d], y=[0,100], line=dict(color="crimson",width=1))
            # Add arrow annotation
            self.fig.add_annotation(
                    x=T_d,  # x-coordinate of the arrow's head
                    y=0,  # y-coordinate of the arrow's head
                    ax=0,  # x-coordinate of the arrow's tail
                    ay=-200,  # y-coordinate of the arrow's tail
                    xref='x',  # sets the x-coordinate system for x and ax (can be 'x', 'x domain', etc.)
                    yref='y',  # sets the y-coordinate system for y and ay (can be 'y', 'y domain', etc.)
                    arrowhead=2,  # size of the arrow head
                    arrowsize=1,  # scale factor for the size of the arrow
                    arrowwidth=2,  # width of the arrow line in pixels
                    arrowcolor='red',  # color of the arrow
                    text="DL traffic arrives at eNB"
                )

            self.fig.add_annotation(
                x=recv_time,  # x-coordinate of the arrow's head
                y=0,  # y-coordinate of the arrow's head
                ax=0,  # x-coordinate of the arrow's tail
                ay=-500,  # y-coordinate of the arrow's tail
                xref='x',  # sets the x-coordinate system for x and ax (can be 'x', 'x domain', etc.)
                yref='y',  # sets the y-coordinate system for y and ay (can be 'y', 'y domain', etc.)
                arrowhead=2,  # size of the arrow head
                arrowsize=1,  # scale factor for the size of the arrow
                arrowwidth=2,  # width of the arrow line in pixels
                arrowcolor='green',  # color of the arrow
                text='DL traffic received by UE'
            )
            
            self.fig.show()
            print("timelineplot")

        return latency, energy, recv_time

    
    def fillin(self, obs, state):
        self.belief = obs[1]
        #self.state = gentraffic(self.belief,self.uplinktime)
        self.state = state
        #print(self.belief,self.state)
        
         
    def step(self, action):
        global sf
        global rf
        reward = 0
        latency = 0
        energy = 0 
         
        ############################################ Prepare observation ##################################################

        # current state and
        T_u = self.uplinktime
        T_d = self.state
        belief = self.belief
        #print("T_d:",T_d)
        ############### Apply Mask and Constraint Enforcement to the action space ###################### 
        #print(action)
        act=[0]*4
        act[0]=int(((action[0] - (-1)) / (1 - (-1))) * 29999 + 1)  #action['rrc_release']
        act[1]=int(((action[1] - (-1)) / (1 - (-1))) * (max_subframes-10)+10)  #action['T3324']
        act[2]=int(((action[2] - (-1)) / (1 - (-1))) * (max_subframes/10-1)+1) #action['edrx_cycle'] 
        #act[3]=action['PO']
        act[3]=int(((action[3] - (-1)) / (1 - (-1))) * (max_subframes))  #action['PSM'] 
        
        #print(act)
        # set constraint for each timer:
        res_time=T_u
        res_time=T_u-act[0]
        act[1]=min(res_time, act[1])
        res_time -= act[1]
        act[3]=min(res_time, act[3])
        act[2]=min(act[1]//10, act[2])
        #act[3]=action_to_po[act[3]]
        #print("coverted action:",act)

  
        ########################################### Reward Function ###################################################
        # simulate the episode 
         
        latency, energy, recv_time = self.simulator(T_d, act) 
        self.time=self.time + recv_time
        #print("time:", self.time)
        terminated=1 if self.time >= 85500000 else 0 
        #terminated=1 if self.time >= max_subframes*20 else 0
        nofawake = int(recv_time/max_delay)
        standard = nofawake*IE+(recv_time-nofawake)*BE+CE
        #nofawake = int(T_d/max_delay)
        #standard = nofawake*IE+(T_d-nofawake)*BE+CE
        
        if latency > max_delay:
            #reward=-belief*(latency/recv_time)*50  
            reward= 0.5*(-(1-max_delay/latency)-1)    
        else:         
            #reward=-(100-belief)*(energy/((recv_time+1)*CE))
            if energy > standard: 
                #reward = -(100-belief)*(1-standard/energy)/100
                reward = -0.5*(1-standard/energy)  
            else:  
                #reward = (100-belief)*(1-(energy/standard))/100 
                reward = 1-energy/standard
            #reward = pow(reward,19)              
        if self.debug: 
            print("reward: ", reward) 
            print("used energy:", energy, "standard",standard) 
        self._set_obs()
        observation = self._get_obs()
        #print(observation)
        return observation, reward, terminated, [self.time,latency,energy,standard,T_d] 
  
 
 


 
 
