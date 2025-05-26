# agents/search_agent.py

import os
from pathlib import Path
from agents.shared_enhanced_dqn import EnhancedDQNAgent

class SearchAgent:
    def __init__(self, state_dim=384, action_dim=5, model_dir="models"):
        self.agent = EnhancedDQNAgent(state_dim, action_dim)
        self.model_path = Path(model_dir) / "search_agent.pth"
        self.load_model()

    def act(self, state, training=True):
        return self.agent.act(state, training)

    def remember(self, state, action, reward, next_state, done):
        self.agent.remember(state, action, reward, next_state, done)

    def train(self):
        self.agent.replay()

    def save_model(self):
        self.agent.save_model(str(self.model_path))

    def load_model(self):
        if self.model_path.exists():
            self.agent.load_model(str(self.model_path))
