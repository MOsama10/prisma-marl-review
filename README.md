

# 📚 PRISMA-MARL: Automated Systematic Literature Review with Multi-Agent Reinforcement Learning

This project is a **proof-of-concept (PoC)** for automating systematic literature reviews based on **PRISMA guidelines**, using a **Multi-Agent Reinforcement Learning (MARL)** system. Each agent operates independently and is trained using **agent-specific reward loops**, evaluated via a dynamic **PRISMA Checker**. All documents are pulled from [arXiv.org](https://arxiv.org).

---

## ✨ Features

- 🔍 Automated arXiv search based on topic and date range
- 🧠 Multi-agent system: search, abstract filter, full-text decision
- 🧪 PRISMA-based reward functions for each agent
- 🔁 Centralized Training with Decentralized Execution (CTDE)
- 📊 Streamlit web interface + CSV/JSON export
- 💾 Model saving/loading with live evaluation

---

## ⚙️ Architecture (Reward Loop)

```markdown


                  +--------------------------+
                  |        User Input        |
                  +--------------------------+
                           |
           +---------------+---------------+---------------+
           |               |               |               |
    +------+-----+  +------+-----+  +------+-----+  +------+-----+
    | Search Agent|  | Abstract Agent|  | FullText Agent|  |   ...      |
    +------+-----+  +------+-----+  +------+-----+  +------+-----+
           |               |               |
    +------+-----+  +------+-----+  +------+-----+
    | PRISMA Eval |  | PRISMA Eval |  | PRISMA Eval |
    +------+-----+  +------+-----+  +------+-----+
           |               |               |
     Reward + Replay  Reward + Replay  Reward + Replay
           ↑               ↑               ↑
           +---------------+---------------+
                   (Repeat per epoch)
```



---

## 📁 Project Structure
```markdown

prisma\_marl\_project/
├── agents/                   # Modular DQN agents
│   ├── search\_agent.py
│   ├── title\_abstract\_filter.py
│   ├── full\_text\_agent.py
│   ├── prisma\_checker.py
│   └── shared\_enhanced\_dqn.py
│
├── rewards/                 # Custom PRISMA reward logic
│   └── enhanced\_reward\_system.py
│
├── trainer/                 # Training pipeline
│   └── train\_agents.py
│
├── utils/                   # ArXiv interface, tokenizer, etc.
│   ├── arxiv\_interface.py
│   └── logger.py
│
├── models/                  # Saved PyTorch model weights
│
├── app.py                   # Streamlit UI
├── main.py                  # CLI review pipeline
├── requirements.txt         # Dependencies
├── README.md

````

---

## 🚀 Getting Started

### ✅ 1. Clone the Repo

```bash
git clone https://github.com/MOsama10/prisma-marl-review.git
cd prisma-marl-review
````

### ✅ 2. Create & Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# or
source venv/bin/activate   # macOS/Linux
```

### ✅ 3. Install Requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔁 Training Agents (Reward Loop)

Train the MARL agents with real arXiv data and PRISMA feedback:

```bash
python trainer/train_agents.py
```

📦 Trained models are saved to `models/`.

---

## 🧪 CLI Inference Mode

Run the review pipeline interactively from terminal:

```bash
python main.py
```

✔ Enter a topic and year range
📄 Outputs top 10 papers to `results.csv`
📊 Shows PRISMA compliance score

---

## 🌐 Streamlit Web Interface

```bash
streamlit run app.py
```

Features:

* Topic input + date range
* Run reviews interactively
* Download CSV/JSON
* Train agents from sidebar
* Visual PRISMA score and relevance ranking

---

## 📊 PRISMA Compliance Scoring

Each agent receives structured feedback on:

* Search coverage and strategy
* Abstract clarity and inclusion accuracy
* Methodology and results from full-text
* Synthesis and limitation discussion

The global score is calculated from 8 standard PRISMA checkpoints.

---

## 👤 Author

**Mohamed Osama**
GitHub: [@MOsama10](https://github.com/MOsama10)





