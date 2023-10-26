import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self.buffer[self.position] = (state, action, reward, next_state)
            self.position = (self.position + 1) % self.capacity
            
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = map(np.stack, zip(*batch))
        return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)


# Define Actor and Critic Networks
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, state):
        state = state.unsqueeze(1)
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
         
    def forward(self, state, action):
        state = state.unsqueeze(1)
        action = action.unsqueeze(1)
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
# Hyperparameters
GAMMA = 0.99 
TAU = 0.005
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
BATCH_SIZE = 64

actor = Actor()
critic = Critic()
 
optimizer_actor = optim.Adam(actor.parameters(), lr=LR_ACTOR)
optimizer_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC)
criterion = nn.MSELoss()

# Assuming a replay buffer exists (not shown here for brevity), let's write the update function
def update(replay_buffer):
    states, actions, rewards, next_state = replay_buffer.sample(BATCH_SIZE)
      
    # Convert to tensors
    states = torch.FloatTensor(states)
    actions = torch.FloatTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_state = torch.FloatTensor(next_state)
    
    # Update critic
    q_current = critic(states, actions)
    loss_critic = criterion(q_current, rewards.unsqueeze(1))
    optimizer_critic.zero_grad()
    loss_critic.backward()
    optimizer_critic.step()
    
    # Update actor
    actions_pred = actor(states).squeeze(1)
    loss_actor = -critic(states, actions_pred).mean()
    optimizer_actor.zero_grad()
    loss_actor.backward()
    optimizer_actor.step()
    

# Define reward function
def reward_function(input_,output):
    reward = 0
    # Assuming some function f. Modify as needed.
    if input_ > 0.05:
        if output > 0.5:
            reward = 1
        else:
            reward = -1
    else:
        if output < 0.5:
            reward = 2
        else:
            reward = -2
    return reward

# Training loop
NUM_EPISODES = 10
buffer = ReplayBuffer(capacity=10000)
for episode in range(NUM_EPISODES):
    total_reward = 0
    for t in range(100):
        state = np.random.rand() * 0.13
        state_tensor = torch.FloatTensor([state])
        action = actor(state_tensor).item()
        reward = reward_function(state,action)
        next_state = np.random.rand() * 0.13  # You might want a more meaningful next state transition
        # Assuming a replay buffer named 'buffer'
        print(state, action, reward,next_state)
        buffer.push(state, action, reward,next_state)
        total_reward += reward
        state = next_state
        
        if len(buffer) > BATCH_SIZE:
            update(buffer)
    

