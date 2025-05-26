# agents/prisma_checker.py

from rewards.enhanced_reward_system import EnhancedRewardSystem

class PRISMAChecker:
    def __init__(self):
        self.reward_system = EnhancedRewardSystem()

    def evaluate_search_reward(self, papers, query_embedding, human_feedback=None):
        """
        Evaluate search quality based on relevance, diversity, and feedback.
        """
        return self.reward_system.compute_search_reward(papers, query_embedding, human_feedback)

    def evaluate_abstract_reward(self, paper_summary, decision, ground_truth=None):
        """
        Evaluate title/abstract decision quality.
        """
        return self.reward_system.compute_filter_reward(
            {"abstract": paper_summary, "citation_count": 0},
            decision,
            ground_truth
        )

    def evaluate_fulltext_reward(self, paper_summary, decision, ground_truth=None):
        """
        Evaluate full-text decision quality.
        """
        return self.reward_system.compute_filter_reward(
            {"abstract": paper_summary, "citation_count": 0},
            decision,
            ground_truth
        )

    def evaluate_prisma_score(self, checklist_dict):
        """
        Return a PRISMA compliance score from 0.0 to 1.0.
        """
        return self.reward_system.compute_prisma_reward(checklist_dict)
