# 🍔 Prizo – Dynamic Pricing using Reinforcement Learning

**Multi-Armed Bandits for Profit-Aware Discounts in Food Delivery**

🔗 **Live Demo:**  
https://prizo-dynamic-prizing-x598itxyalyvmjx78pmwzr.streamlit.app/

---

## 📌 Overview
Prizo is a reinforcement learning–based dynamic pricing system for food delivery platforms.  
Instead of using fixed discounts, it learns which discount level maximizes **profit** using:

- **Epsilon-Greedy**
- **UCB1**
- **Thompson Sampling** (best performer)

The system models discount selection as a **Multi-Armed Bandit (MAB)** problem with a **profit-aware reward function** and supports **non-stationary user behavior**.

A Streamlit app is included for real-time simulation and visualization.

---

## 🚀 Features
- Profit-aware rewards  
- Multi-armed bandit algorithms  
- Non-stationary drift simulation  
- Interactive Streamlit dashboard  
- Offline experiment script  
- Deployment-ready project  

---

## 📂 Project Structure
📁 prizo-dynamic-pricing
│
├── streamlit_app.py        # Streamlit UI (live app)
├── main.py                 # Offline simulation experiments
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

---

## 🧠 Algorithms Used
Epsilon-Greedy, UCB1, Thompson Sampling

## 💸 Reward
Profit-aware reward = revenue after discount – cost
## ▶️ Run the Project Locally

▶️ Run the Project Locally

Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run streamlit_app.py

👤 Author

Madhupa Vinod
MSc Data Science – Christ University


