# agents/prisma_checker.py

class PRISMAChecker:
    def compute_global_reward(self, rewards):
        """
        Accepts a dict of agent rewards and returns a global reward score.
        Example: {'search': 0.8, 'title_abstract': 0.9, 'full_text': 0.7}
        """
        return sum(rewards.values()) / len(rewards) if rewards else 0.0
