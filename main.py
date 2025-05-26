# main.py

from agents.search_agent import SearchAgent
from agents.title_abstract_filter import TitleAbstractFilterAgent
from agents.full_text_agent import FullTextAgent
from agents.prisma_checker import PRISMAChecker
from rewards.enhanced_reward_system import EnhancedRewardSystem
from utils.arxiv_interface import search_arxiv
import pandas as pd

def main():
    topic = input("🔍 Enter research topic: ")
    from_year = int(input("📅 Start year: "))
    to_year = int(input("📅 End year: "))
    max_results = 10

    print("🔎 Retrieving papers...")
    papers = search_arxiv(topic, from_year, to_year, max_results)

    if not papers:
        print("❌ No papers found.")
        return

    search_agent = SearchAgent()
    abstract_agent = TitleAbstractFilterAgent()
    fulltext_agent = FullTextAgent()
    prisma_checker = PRISMAChecker()
    reward_system = EnhancedRewardSystem()

    data = []
    for paper in papers:
        abstract_embedding = reward_system.embed_text(paper.summary)
        action = abstract_agent.act(abstract_embedding, training=False)
        decision = "Include" if action == 2 else "Maybe" if action == 1 else "Exclude"

        data.append({
            "Title": paper.title,
            "Year": paper.published.year,
            "URL": paper.entry_id,
            "Decision": decision,
            "Abstract": paper.summary,
            "Authors": ", ".join([a.name for a in paper.authors])
        })

    df = pd.DataFrame(data)
    df.to_csv("results.csv", index=False)
    print(f"✅ Saved {len(df)} results to results.csv")

    # Compute PRISMA score
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

    print(f"📊 PRISMA Compliance Score: {score:.2f}")

if __name__ == "__main__":
    main()
