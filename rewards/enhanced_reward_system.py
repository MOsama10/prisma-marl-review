# rewards/enhanced_reward_system.py

import numpy as np
from sentence_transformers import SentenceTransformer
from collections import deque
from typing import List, Dict, Optional

class EnhancedRewardSystem:
    """
    Advanced reward system with human feedback integration and adaptive learning
    """

    def __init__(self):
        self.relevance_threshold = 0.7
        self.diversity_bonus = 0.1
        self.human_feedback_weight = 0.3
        self.feedback_history = deque(maxlen=1000)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def compute_search_reward(self, papers: List, query_embedding: np.ndarray, 
                              human_feedback: Optional[Dict] = None) -> float:
        if not papers:
            return -1.0

        relevance_scores = []
        for paper in papers:
            paper_embedding = self.embed_text(paper.summary)
            similarity = np.dot(query_embedding, paper_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(paper_embedding)
            )
            relevance_scores.append(similarity)

        avg_relevance = np.mean(relevance_scores)
        diversity_score = self.calculate_diversity(papers)

        feedback_score = 0.0
        if human_feedback:
            feedback_score = self.integrate_human_feedback(human_feedback)

        reward = (avg_relevance +
                  self.diversity_bonus * diversity_score +
                  self.human_feedback_weight * feedback_score)

        return np.clip(reward, -1.0, 1.0)

    def compute_filter_reward(self, paper_data: Dict, decision: int, 
                              ground_truth: Optional[int] = None) -> float:
        base_reward = 0.0

        has_methodology = any(word in paper_data.get('abstract', '').lower()
                              for word in ['method', 'approach', 'algorithm', 'framework'])
        has_results = any(word in paper_data.get('abstract', '').lower()
                          for word in ['result', 'performance', 'evaluation', 'experiment'])
        citation_count = paper_data.get('citation_count', 0)

        if decision == 1:
            base_reward = 0.5
            if has_methodology and has_results:
                base_reward += 0.3
            if citation_count > 10:
                base_reward += 0.2
        elif decision == 0:
            base_reward = 0.1
            if not has_methodology or not has_results:
                base_reward += 0.2

        if ground_truth is not None:
            if decision == ground_truth:
                base_reward += 0.5
            else:
                base_reward -= 0.3

        return np.clip(base_reward, -1.0, 1.0)

    def compute_prisma_reward(self, review_data: Dict) -> float:
        checklist_items = [
            'search_strategy_documented',
            'inclusion_criteria_clear',
            'exclusion_criteria_clear',
            'study_selection_process',
            'data_extraction_systematic',
            'quality_assessment_performed',
            'results_synthesized',
            'limitations_discussed'
        ]
        compliance_score = sum(review_data.get(item, 0) for item in checklist_items) / len(checklist_items)

        if compliance_score > 0.8:
            compliance_score += 0.2

        return np.clip(compliance_score, 0.0, 1.0)

    def calculate_diversity(self, papers: List) -> float:
        if len(papers) < 2:
            return 0.0

        embeddings = [self.embed_text(p.summary) for p in papers]
        similarities = []

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)

        avg_similarity = np.mean(similarities)
        diversity = 1.0 - avg_similarity
        return max(0.0, diversity)

    def integrate_human_feedback(self, feedback: Dict) -> float:
        self.feedback_history.append(feedback)
        recent_feedback = list(self.feedback_history)[-10:]

        if not recent_feedback:
            return 0.0

        relevance_scores = [f.get('relevance', 0.5) for f in recent_feedback]
        quality_scores = [f.get('quality', 0.5) for f in recent_feedback]

        weighted_score = 0.6 * np.mean(relevance_scores) + 0.4 * np.mean(quality_scores)
        return (weighted_score - 0.5) * 2  # Normalize to [-1, 1]

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)
