## Complexity-Aware Reinforcement Learning for Automated Model Selection

This project explores the use of Reinforcement Learning (RL) to automate model selection for supervised learning tasks. 
Instead of relying on static rules or trial-and-error pipelines, the system learns how dataset complexity influences model performance and adapts its choices accordingly.

### Key Idea
Model selection is framed as a **Markov Decision Process (MDP)**:
- **State:** Data complexity and meta-features
- **Action:** Choice among 12 base and ensemble models (XGBoost, LightGBM, Random Forest, Logistic Regression, and their combinations)
- **Reward:** Validation accuracy, with penalties for redundant or failed models

### Core Components
- **LinUCB Agent:** A contextual bandit that balances exploration and exploitation to learn optimal model-selection policies.
- **ModelSelectionEnv:** Custom RL environment encoding dataset states and actions.
- **Complexity Analyzer:** Computes classical dataset complexity measures (Ho & Basu indices) and meta-features.
- **Reward Engine:** Evaluates model performance, detects deterministic targets, and applies penalties or bonuses.
- **Dataset Suite:** Tested on 5 datasets — Iris, Digits, RCV1, 20 Newsgroups, and C3F (IoT).

### Results
- Outperforms or matches **FLAML** on multiple datasets.
- Learns cross-dataset adaptability and skips redundant model training.
- Proves that RL can be a viable AutoML alternative when dataset memory matters.

### Next Steps
- Expand to a larger meta-dataset for broader generalization.
- Integrate hyperparameter optimization within the RL loop.
- Explore richer RL algorithms beyond LinUCB (e.g., PPO, DQN for discrete action spaces).

### Dependencies
- Python 3.10+
- scikit-learn, xgboost, lightgbm
- numpy, pandas, scipy
- matplotlib, plotly (optional for plotting)
- flaml (for baseline comparison)
