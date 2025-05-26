# agents/full_text_agent.py

from agents.search_agent import DQN  # Reuse architecture
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class FullTextAgent:
    def __init__(self, input_dim=768, action_dim=2, lr=1e-3, gamma=0.99):
        self.policy_net = DQN(input_dim, action_dim)
        self.target_net = DQN(input_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        states = torch.FloatTensor(state_batch)
        actions = torch.LongTensor(action_batch).unsqueeze(1)
        rewards = torch.FloatTensor(reward_batch)
        next_states = torch.FloatTensor(next_state_batch)
        dones = torch.FloatTensor(done_batch)

        q_values = self.policy_net(states).gather(1, actions).squeeze()
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
