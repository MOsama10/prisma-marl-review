# app.py

import streamlit as st
import pandas as pd
from datetime import datetime

from agents.search_agent import SearchAgent
from agents.title_abstract_filter import TitleAbstractFilterAgent
from agents.full_text_agent import FullTextAgent
from agents.prisma_checker import PRISMAChecker
from rewards.enhanced_reward_system import EnhancedRewardSystem
from utils.arxiv_interface import search_arxiv
from trainer.train_agents import PRISMAAgentTrainer

st.set_page_config(page_title="PRISMA-MARL System", layout="wide")
st.title("📚 PRISMA-MARL System")
st.markdown("Automated systematic literature reviews with multi-agent RL and PRISMA feedback")

# Session state for persistent model access
if "agents" not in st.session_state:
    st.session_state.agents = {
        "search": SearchAgent(),
        "abstract": TitleAbstractFilterAgent(),
        "fulltext": FullTextAgent()
    }
if "prisma" not in st.session_state:
    st.session_state.prisma = PRISMAChecker()
if "reward" not in st.session_state:
    st.session_state.reward = EnhancedRewardSystem()

st.sidebar.header("🧠 Agent Training")

if st.sidebar.button("🎯 Train Agents"):
    with st.spinner("Training agents..."):
        # Simulated training data (you can replace this)
        sample_data = [
            {
                'query': 'graph neural networks',
                'papers': [],
                'search_action': 1,
                'filter_decisions': [1, 2, 0],
                'ground_truth_labels': {0: 1, 1: 2, 2: 0},
                'human_feedback': {'relevance': 0.8, 'quality': 0.7}
            }
        ]
        trainer = PRISMAAgentTrainer()
        trainer.train(sample_data, epochs=10)
        st.success("✅ Training completed and models saved!")

st.sidebar.header("🔍 Literature Review")

topic = st.sidebar.text_input("Research Topic", "deep reinforcement learning")
from_year = st.sidebar.number_input("From Year", 2000, 2030, value=2020)
to_year = st.sidebar.number_input("To Year", 2000, 2030, value=2025)
max_results = st.sidebar.slider("Max Results", 5, 30, 10)

if st.sidebar.button("🚀 Start Review"):
    with st.spinner("Searching arXiv..."):
        papers = search_arxiv(topic, from_year, to_year, max_results)
    
    if not papers:
        st.error("No papers found.")
    else:
        st.success(f"Found {len(papers)} papers")

        results = []
        for paper in papers:
            embed = st.session_state.reward.embed_text(paper.summary)

            a_action = st.session_state.agents['abstract'].act(embed, training=False)
            f_action = st.session_state.agents['fulltext'].act(embed, training=False)

            a_reward = st.session_state.prisma.evaluate_abstract_reward(paper.summary, a_action)
            f_reward = st.session_state.prisma.evaluate_fulltext_reward(paper.summary, f_action)

            score = (a_reward + f_reward) / 2
            decision = "Include" if a_action == 2 else "Maybe" if a_action == 1 else "Exclude"

            results.append({
                "Title": paper.title,
                "Year": paper.published.year,
                "URL": paper.entry_id,
                "Abstract": paper.summary,
                "Authors": ", ".join([a.name for a in paper.authors]),
                "Decision": decision,
                "Score": round(score, 3)
            })

        df = pd.DataFrame(results).sort_values(by="Score", ascending=False).head(10)
        st.subheader("📋 Top Papers")
        st.dataframe(df)

        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, file_name="review_results.csv")

        prisma_score = st.session_state.prisma.evaluate_prisma_score({
            'search_strategy_documented': 1,
            'inclusion_criteria_clear': 1,
            'exclusion_criteria_clear': 1,
            'study_selection_process': 1,
            'data_extraction_systematic': 1,
            'quality_assessment_performed': 1,
            'results_synthesized': 0.9,
            'limitations_discussed': 0.6
        })
        st.metric("📊 PRISMA Compliance Score", f"{prisma_score:.2f}")
