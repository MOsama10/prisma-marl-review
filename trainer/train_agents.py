import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import logging
import arxiv

from agents.search_agent import SearchAgent
from agents.title_abstract_filter import TitleAbstractFilterAgent
from agents.full_text_agent import FullTextAgent
from agents.prisma_checker import PRISMAChecker
from rewards.enhanced_reward_system import EnhancedRewardSystem

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PRISMAAgentTrainer:
    def __init__(self):
        self.search_agent = SearchAgent()
        self.abstract_agent = TitleAbstractFilterAgent()
        self.fulltext_agent = FullTextAgent()
        self.prisma = PRISMAChecker()
        self.reward_system = EnhancedRewardSystem()

    def train(self, training_data, epochs=10):
        for epoch in range(epochs):
            rewards = {'search': [], 'abstract': [], 'fulltext': []}

            for sample in training_data:
                query = sample['query']
                papers = sample['papers']
                query_embed = self.reward_system.embed_text(query)

                # Search Agent Step
                search_action = sample.get('search_action', 0)
                search_reward = self.prisma.evaluate_search_reward(
                    papers, query_embed, sample.get('human_feedback')
                )
                next_state = np.concatenate([query_embed, [len(papers), search_reward]])
                self.search_agent.remember(query_embed, search_action, search_reward, next_state, True)
                rewards['search'].append(search_reward)

                # Abstract Agent Step
                filter_decisions = sample.get('filter_decisions', [])
                for i, paper in enumerate(papers[:5]):
                    paper_embed = self.reward_system.embed_text(paper.summary)
                    decision = filter_decisions[i] if i < len(filter_decisions) else 1
                    ground_truth = sample.get('ground_truth_labels', {}).get(i)
                    reward = self.prisma.evaluate_abstract_reward(paper.summary, decision, ground_truth)
                    self.abstract_agent.remember(paper_embed, decision, reward, paper_embed, True)
                    rewards['abstract'].append(reward)

                # FullText Agent Step
                for paper in papers[:5]:
                    embed = self.reward_system.embed_text(paper.summary)
                    decision = np.random.choice([0, 1])  # Simulated
                    reward = self.prisma.evaluate_fulltext_reward(paper.summary, decision)
                    self.fulltext_agent.remember(embed, decision, reward, embed, True)
                    rewards['fulltext'].append(reward)

            # Train all agents
            self.search_agent.train()
            self.abstract_agent.train()
            self.fulltext_agent.train()

            # Log average rewards
            logger.info(
                f"[Epoch {epoch}] "
                f"Search: {np.mean(rewards['search']):.3f}, "
                f"Abstract: {np.mean(rewards['abstract']):.3f}, "
                f"FullText: {np.mean(rewards['fulltext']):.3f}"
            )

        self.search_agent.save_model()
        self.abstract_agent.save_model()
        self.fulltext_agent.save_model()
        logger.info("✅ Models saved after training.")

# 🔧 Entry point with real arXiv data
if __name__ == "__main__":
    # Pull real papers from arXiv
    query = "deep reinforcement learning"
    search = arxiv.Search(query=query, max_results=10, sort_by=arxiv.SortCriterion.Relevance)
    papers = list(search.results())

    # Training data sample
    dummy_training_data = [
        {
            'query': query,
            'papers': papers,
            'search_action': 1,
            'filter_decisions': [1, 2, 0, 1, 2],
            'ground_truth_labels': {0: 1, 1: 2, 2: 0, 3: 1, 4: 2},
            'human_feedback': {'relevance': 0.8, 'quality': 0.7}
        }
    ]

    trainer = PRISMAAgentTrainer()
    trainer.train(dummy_training_data, epochs=10)
