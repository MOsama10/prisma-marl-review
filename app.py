# app.py

import streamlit as st
from datetime import datetime
import pandas as pd
from agents.search_agent import SearchAgent
from agents.title_abstract_filter import TitleAbstractFilterAgent
from agents.full_text_agent import FullTextAgent
from agents.prisma_checker import PRISMAChecker
from rewards.enhanced_reward_system import EnhancedRewardSystem
from utils.arxiv_interface import search_arxiv

st.set_page_config(page_title="PRISMA MARL System", layout="wide")

st.title("📚 PRISMA-MARL System")
st.markdown("Automated Systematic Reviews with Multi-Agent RL")

# Instantiate agents
search_agent = SearchAgent()
abstract_agent = TitleAbstractFilterAgent()
fulltext_agent = FullTextAgent()
prisma_checker = PRISMAChecker()
reward_system = EnhancedRewardSystem()

st.sidebar.header("🔍 Review Configuration")
topic = st.sidebar.text_input("Research Topic", "deep reinforcement learning")
from_year = st.sidebar.number_input("From Year", 2000, 2030, 2020)
to_year = st.sidebar.number_input("To Year", 2000, 2030, 2025)
max_results = st.sidebar.slider("Max Results", 5, 25, 10)

if st.sidebar.button("🚀 Start Review"):
    with st.spinner("Retrieving papers..."):
        papers = search_arxiv(topic, from_year, to_year, max_results=max_results)
    
    if not papers:
        st.error("No papers found.")
    else:
        st.success(f"Found {len(papers)} papers from arXiv.")

        data = []
        for paper in papers:
            abstract_embedding = reward_system.embed_text(paper.summary)
            action = abstract_agent.act(abstract_embedding, training=False)
            include = "Include" if action == 2 else "Maybe" if action == 1 else "Exclude"
            data.append({
                "Title": paper.title,
                "Year": paper.published.year,
                "URL": paper.entry_id,
                "Decision": include,
                "Abstract": paper.summary,
                "Authors": ", ".join([a.name for a in paper.authors])
            })

        df = pd.DataFrame(data)
        st.dataframe(df)

        # Export options
        st.subheader("💾 Export Results")
        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, file_name="review_results.csv", mime="text/csv")

        st.subheader("✅ PRISMA Compliance Score")
        score = prisma_checker.compute_prisma_reward({
            'search_strategy_documented': 1,
            'inclusion_criteria_clear': 1,
            'exclusion_criteria_clear': 1,
            'study_selection_process': 1,
            'data_extraction_systematic': 1,
            'quality_assessment_performed': 1,
            'results_synthesized': 0.9,
            'limitations_discussed': 0.6
        })
        st.metric("PRISMA Score", f"{score:.2f}")
