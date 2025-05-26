# agents/title_abstract_filter.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPOPolicy(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PPOPolicy, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value
