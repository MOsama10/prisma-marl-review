# agents/prisma_checker.py

from rewards.enhanced_reward_system import EnhancedRewardSystem

class PRISMAChecker:
    def __init__(self):
        self.reward_system = EnhancedRewardSystem()

    def compute_global_reward(self, agent_rewards: dict) -> float:
        """
        Compute the average of all agent rewards as a global cooperative reward.
        """
        if not agent_rewards:
            return 0.0
        return sum(agent_rewards.values()) / len(agent_rewards)

    def compute_prisma_reward(self, review_data: dict) -> float:
        """
        Compute compliance score based on PRISMA checklist.
        """
        return self.reward_system.compute_prisma_reward(review_data)
