
```markdown
# 📚 Automated Systematic Literature Review using PRISMA & Multi-Agent Reinforcement Learning

This project is a **proof-of-concept (PoC)** system for automating systematic literature reviews in compliance with the **PRISMA guidelines**, using **Multi-Agent Reinforcement Learning (MARL)** with **Centralized Training and Decentralized Execution (CTDE)**.

It simulates expert behavior in:
- Paper Search (via arXiv)
- Title/Abstract Filtering
- Full-Text Review
- PRISMA Rule Compliance

All papers are retrieved from [arXiv.org](https://arxiv.org), filtered and scored by reinforcement learning agents trained with human feedback and reward shaping.

---
````

## 🚀 Features

- 🔎 arXiv search for academic papers (2020–2025)
- 🤖 Modular agents (search, abstract, full text)
- 🧠 DQN agents with replay buffer, epsilon decay, and target networks
- 📊 PRISMA compliance scoring
- 💡 Streamlit UI for interactive usage
- 💾 Export results as CSV/JSON

---

## 🧠 Architecture Overview

User Input
│
├──▶ Search Agent ──▶ Title/Abstract Agent ──▶ Full-Text Agent
│                                            │
└───────────────────────────────────────────▶ PRISMA Checker

````

## 📁 Project Structure

```bash
prisma\_marl\_project/
├── agents/                   # Modular DQN-based agents
│   ├── search\_agent.py
│   ├── full\_text\_agent.py
│   ├── title\_abstract\_filter.py
│   ├── prisma\_checker.py
│   └── shared\_enhanced\_dqn.py
│
├── rewards/                 # Advanced reward system with feedback
│   └── enhanced\_reward\_system.py
│
├── trainer/                 # Agent training loop
│   └── train\_agents.py
│
├── utils/                   # Helper functions
│   ├── arxiv\_interface.py
│   ├── tokenizer.py
│   └── logger.py
│
├── models/                  # Saved DQN model checkpoints
│
├── app.py                   # Streamlit UI
├── main.py                  # CLI runner
├── requirements.txt         # Python dependencies
├── results.csv              # Saved output from main/app

````

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/MOsama10/prisma-marl-review.git
cd prisma-marl-review
````

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🧪 Usage

### 🔹 CLI Mode

```bash
python main.py
```

You'll be asked to enter:

* Research topic
* Start year (e.g. 2020)
* End year (e.g. 2025)

The output will be saved to `results.csv`.

### 🔹 Streamlit Web UI

```bash
streamlit run app.py
```

Explore and export your results visually.

---

## 🎯 Training Agents

You can train agents using the `trainer` module:

```bash
python trainer/train_agents.py
```

This trains the agents with simulated feedback and saves models to `models/`.

---

## 📦 Output

* **results.csv** – Top papers (title, year, abstract, link, decision)
* **.pth files** – Saved weights for each agent
* **PRISMA score** – Compliance metric between 0.0 and 1.0

---

## 📝 PRISMA Compliance Explained

The PRISMA score is computed using an 8-point checklist that covers:

* Search documentation
* Inclusion/exclusion clarity
* Study selection, extraction, synthesis
* Quality assessment

A score of 1.0 indicates **perfect compliance**.

---

## 📌 Future Enhancements

* RL fine-tuning with human-in-the-loop feedback (RLHF)
* Integration with PubMed, Semantic Scholar APIs
* Auto-summarization of included papers
* Visual PRISMA flow diagrams
* Hugging Face or Streamlit Cloud deployment

---

## 👤 Author

**Mohamed Osama**
[GitHub @MOsama10](https://github.com/MOsama10)

---

## 📄 License

This project is licensed under the **MIT License**.

---

