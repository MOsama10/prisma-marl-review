# main.py

from env.prisma_env import PRISMAEnv
from agents.search_agent import SearchAgent
from agents.full_text_agent import FullTextAgent
from agents.prisma_checker import PRISMAChecker
from utils.tokenizer import embed_text
from utils.logger import get_logger
from utils.arxiv_interface import search_arxiv

import numpy as np
import csv

logger = get_logger()

# Save the papers to CSV
def save_papers_to_csv(papers, filename="results.csv"):
    with open(filename, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Title", "Year", "URL", "Abstract"])  # Header
        for paper in papers:
            writer.writerow([
                paper.title,
                paper.published.year,
                paper.entry_id,
                paper.summary.replace("\n", " ")
            ])

def main():
    # Get topic and date range from user
    topic = input("🔍 Enter a research topic: ")
    from_year = int(input("📅 Start year (e.g., 2020): "))
    to_year = int(input("📅 End year (e.g., 2025): "))

    # Query arXiv
    papers = search_arxiv(topic, from_year, to_year, max_results=10)
    if not papers:
        print("❌ No papers found for that topic and date range.")
        return

    print(f"✅ Found {len(papers)} papers from {from_year} to {to_year}.\n")
    save_papers_to_csv(papers)
    print("📁 Results saved to results.csv\n")

    # Init environment and agents
    env = PRISMAEnv()
    env.reset()

    agents = {
        "search": SearchAgent(),
        "full_text": FullTextAgent(),
        "prisma_checker": PRISMAChecker()
    }

    rewards = {}

    # Run one round of MARL loop
    while env.agents:
        agent_name = env.agent_selection
        obs = env.observe(agent_name)

        if agent_name == "search":
            print(f"🔎 Search Agent retrieved papers for topic: '{topic}'")
            state = embed_text(topic)
            action = agents["search"].act(state)
            env.step(action)
            rewards[agent_name] = env.rewards[agent_name]

        elif agent_name == "title_abstract":
            print(f"🧠 Title & Abstract Agent filtering paper: {papers[0].title}")
            env.step(np.random.choice(3))  # placeholder classification
            rewards[agent_name] = env.rewards[agent_name]

        elif agent_name == "full_text":
            print(f"📄 Full Text Agent reviewing: {papers[0].title}")
            state = embed_text(papers[0].summary)
            action = agents["full_text"].act(state)
            env.step(action)
            rewards[agent_name] = env.rewards[agent_name]

        elif agent_name == "prisma_checker":
            global_reward = agents["prisma_checker"].compute_global_reward(rewards)
            logger.info(f"Global PRISMA reward: {global_reward:.4f}")
            env.step(0)

if __name__ == "__main__":
    main()
