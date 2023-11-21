import gym
from gym import spaces
import numpy as np
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import copy

sf = 0  # subframe
rf = 0  # radio frame
max_minutes = 120
max_subframes = max_minutes*60*1000
#action_to_po = {0: 4, 1: 9}
#max_delay = 20000
max_delay = 10000
 
# energy consumption here use average current
# Energy usage, the unit is uA(current)(to fix!!!)
#CE = 35000 # energy consumed when fully connected
#IE = 2620  # energy consumed when connected in edrx idle
#BE = 5  # Basic energy consumed when released in edrx idle
#SE = 2.7  # Sleep energy
CE = 15345
IE = 5913
BE = 23.82
SE = 2.7
 
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
    if pattern == 1:
        np.random.seed(0)
        traffic = [random.randint(0, 100) for _ in range(steps)]
    if pattern == 2:
        numbers = [0, 30, 80]
        traffic = [random.choice(numbers) for _ in range(steps)]
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
 
    def _set_obs(self, DW=0):
        # Train mode: random
        if self.test == 0:
            #print("TEST mode")
            #self.pattern = trafficpattern() 
            #self.belief = self.pattern[int(self.time//max_subframes)]
            if DW == 0:
                self.belief = np.random.uniform(low=0, high=100, size=(1,))[0]
                '''
                #dice               
                dice = np.random.uniform()
                if dice < 0.33:
                    self.belief = 0
                elif 0.33 <= dice < 0.66:
                    self.belief = 30
                else:
                    self.belief = 80
                '''
            else:
                self.belief = self.belief
            self.gold = np.random.uniform(low=0, high=100, size=(1,))[0]
            self.state = gentraffic(self.belief, self.uplinktime)
            self.S = [5]*5
        # test mode: fixed random 
        elif self.test == 1:
            self.belief = self.tmppattern.pop(0)
            #self.state = self.tmpavl.pop(0)
            self.gold = 40
            self.state = gentraffic(self.belief, self.uplinktime)
            self.S  = [5]*5
        # mode: periodic
        elif self.test == 2:
            self.belief = 100
            self.gold = np.random.uniform(low=0, high=100, size=(1,))[0]
            self.state = genperiodic()
            self.S = [5]*5
            
    def _get_obs(self):
        global rf
        return self.S
                           
    def __init__(self, render_mode=None, debug=False, timelineplot=False, test=0, belief = 0):
        self.debug = debug 
        self.timelineplot = timelineplot
        self.test = test
          
        ########################## Observation ###############################
        # observation: 1. belif value 2. expected maximum latency
        # self.observation_space = spaces.Box(0, 100, shape=(2, 1), dtype=float)
        self.observation_space = spaces.MultiDiscrete([6]*5)
        # initial time
        self.time = 0
        # there is at least an uplink traffic each max_subframes
        self.uplinktime = max_subframes
 
        if self.test == 1:
            self.pattern = trafficpattern(pattern = 1, steps = 200)
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
        S = 0
        #print("simulator:",T_d, act)
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
        connected_ = 0
        flag_connected_ = 0
        flag = 0
        PF = 0
        
        while not recv_time:
                if cur_time == offset:
                    triggered = 1

                tmp_time = cur_time % (T3412+rrc_release)
                if tmp_time == 0:
                    if triggered and flag_connected_ == 0:
                        connected_ = 5
                        flag_connected_ = 1
                if 0 < tmp_time < rrc_release:
                    if triggered and flag_connected_ == 0:
                        connected_ = 1
                        flag_connected_ = 1
                    if tmp_time >= 2000:
                        connected = 1  
                    energy += CE
                    tplot(CE)  

                if rrc_release <= tmp_time < rrc_release + T3324: 
                    if rf == PF and sf == PO:
                        flag = 1
                        PF = (PF + edrx_cycle) 
                        if triggered and flag_connected_ == 0:
                            connected_ = 2
                            flag_connected_ = 1
                        connected = 1
                        energy += IE
                        tplot(IE)
                    else:
                        if triggered and flag_connected_ == 0:
                            connected_=3
                            flag_connected_ = 1
                        connected = 0 
                        energy += BE
                        tplot(BE)
   
                if rrc_release + T3324 <= tmp_time < rrc_release + T3324 + PSM:
                    rf = rf % 1024
                    if triggered and flag_connected_ == 0: 
                        connected_ = 4
                        flag_connected_ = 1
                    connected = 0
                    energy += SE 
                    tplot(SE)

                # print(T_d, cur_time, tmp_time, connected, triggered)
                # record the time when UE received downlink traffic 
                # the downlink handled in the end
                if triggered and connected:
                    recv_time = cur_time + (rrc_release+T3412)*nofcycle
                    S = connected_
                    break

                # the downlink didn't handeled in the end and handled by TAU
                if triggered and cur_time + (rrc_release+T3412)*nofcycle ==max_subframes:
                    recv_time = max_subframes
                    # no traffic in the step
                    if T_d == max_subframes:
                        S = 5
                    #traffic not handled
                    else:
                        S = connected_
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
        #print(recv_time, T_d)
        if self.debug: 
            print('T_d:', T_d)
            print('recv_t:', recv_time)
            print('latency:', latency)
            print('energy:', energy)
  
        if self.timelineplot:
            self.fig.update_xaxes(range=[0, max_subframes])
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
        return latency, energy, recv_time, S, nofcycle    


    def fillin(self, belief, gold, obs, state):
        self.belief = belief
        self.gold = gold
        self.S = obs
        self.State = state
        self.state = state[0]
        #print(self.belief,self.state)

    def step(self, action):
        global sf
        global rf
        reward = 0
        latency = 0 
        energy = 0

        ######### Apply Mask and Constraint Enforcement to the action space ############
        T_u = self.uplinktime
        #print(action)
        act=[0]*4
        act[0]=int(((action[0] - (-1)) / (1 - (-1))) * 29999 + 2001)  #action['rrc_release']
        act[1]=int(((action[1] - (-1)) / (1 - (-1))) * (max_subframes-10)+10)  #action['T3324']
        act[2]=int(((action[2] - (-1)) / (1 - (-1))) * (max_delay/10-1)+1) #action['edrx_cycle']
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
        act_ = min(act[1],act[2]*10)
        #act[3]=action_to_po[act[3]]
        #print("coverted action:",act)
        
        S_ = []
        K_ = []
        M_ = []
        L_ = []
        B_ = []
        LT_ = []
        ET_ = []
        T_off = act[2]*10 - 1
        for I in range(5):
            ######################## Prepare observation ######################
            # current state and
            #self.state = self.State[I]
            T_d = self.state
            belief = self.belief
            gold = self.gold
            #print("T_d, belief:",T_d,belief )
            ######################### Reward Function ####################
            # simulate the episode 

            latency, energy, recv_time, S, nofcycle = self.simulator(T_d, act)  
            self.time=self.time + recv_time
            terminated = 0 
             
            S_.append(S)
            if S == 2 or S == 3:
                K = nofcycle+1
            else:
                K = 0
            if S == 4:
                M = nofcycle+1+ act[3]/T_off
            else:
                M = 0
            K_.append(K) 
            M_.append(M)

            if S == 3 or S == 4 or S == 5:
                L = latency/T_off
            else:
                L = 0
            L_.append(L)
            LT_.append(latency)
            ET_.append(energy)
            B_.append(belief)
            #self.state = gentraffic(self.belief, self.uplinktime)
            self._set_obs(1)
            #print("AC energy:",energy)
        if np.mean(L_)*T_off <= max_delay:
            reward = T_off/self.time*(sum(K_)+sum(M_)) 
 
        else:
            reward = -100000
        
        self.S = S_
        print("belief:",belief,"next obs:",self.S,"action:",act,"reward:",reward)
        self._set_obs(0)
        return S_, reward, terminated, [self.time, LT_, ET_, B_, T_d]  
   
  
 
  

 
 
