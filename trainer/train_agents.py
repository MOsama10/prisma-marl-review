# trainer/train_agents.py

import numpy as np
import logging
from pathlib import Path
from agents.search_agent import SearchAgent
from agents.title_abstract_filter import TitleAbstractFilterAgent
from agents.full_text_agent import FullTextAgent
from rewards.enhanced_reward_system import EnhancedRewardSystem

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PRISMAAgentTrainer:
    def __init__(self):
        self.search_agent = SearchAgent()
        self.abstract_agent = TitleAbstractFilterAgent()
        self.fulltext_agent = FullTextAgent()
        self.reward_system = EnhancedRewardSystem()

    def train(self, training_data, epochs=50):
        for epoch in range(epochs):
            search_rewards, abstract_rewards, fulltext_rewards = [], [], []

            for sample in training_data:
                query = sample['query']
                papers = sample.get('papers', [])
                query_embed = self.reward_system.embed_text(query)

                # ----- Search Agent -----
                search_action = sample.get('search_action', 0)
                search_reward = self.reward_system.compute_search_reward(
                    papers, query_embed, sample.get('human_feedback')
                )
                next_state = np.concatenate([query_embed, [len(papers), search_reward]])
                self.search_agent.remember(query_embed, search_action, search_reward, next_state, True)
                search_rewards.append(search_reward)

                # ----- Abstract Filter Agent -----
                filter_decisions = sample.get('filter_decisions', [])
                for i, paper in enumerate(papers[:5]):
                    paper_embed = self.reward_system.embed_text(paper.summary)
                    decision = filter_decisions[i] if i < len(filter_decisions) else 1
                    ground_truth = sample.get('ground_truth_labels', {}).get(i)
                    reward = self.reward_system.compute_filter_reward(
                        {'abstract': paper.summary, 'citation_count': 0},
                        decision,
                        ground_truth
                    )
                    self.abstract_agent.remember(paper_embed, decision, reward, paper_embed, True)
                    abstract_rewards.append(reward)

                # ----- Full-Text Agent -----
                for paper in papers[:5]:
                    embed = self.reward_system.embed_text(paper.summary)
                    decision = np.random.choice([0, 1])  # Placeholder
                    reward = self.reward_system.compute_filter_reward(
                        {'abstract': paper.summary, 'citation_count': 0},
                        decision
                    )
                    self.fulltext_agent.remember(embed, decision, reward, embed, True)
                    fulltext_rewards.append(reward)

            # Train agents
            self.search_agent.train()
            self.abstract_agent.train()
            self.fulltext_agent.train()

            # Log
            logger.info(f"[Epoch {epoch}] Search R: {np.mean(search_rewards):.3f}, "
                        f"Abstract R: {np.mean(abstract_rewards):.3f}, "
                        f"FullText R: {np.mean(fulltext_rewards):.3f}")

        # Save all models
        self.search_agent.save_model()
        self.abstract_agent.save_model()
        self.fulltext_agent.save_model()
        logger.info("✅ All models saved.")


# 🔧 Example usage
if __name__ == "__main__":
    # Simulated training data (replace with real queries and papers)
    dummy_training_data = [
        {
            'query': 'deep learning in healthcare',
            'papers': [],
            'search_action': 1,
            'filter_decisions': [1, 2, 0],
            'ground_truth_labels': {0: 1, 1: 2, 2: 0},
            'human_feedback': {'relevance': 0.7, 'quality': 0.6}
        }
    ]

    trainer = PRISMAAgentTrainer()
    trainer.train(dummy_training_data, epochs=10)
