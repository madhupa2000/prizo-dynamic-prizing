# streamlit_app.py
# Interactive Streamlit dashboard for Dynamic Pricing using RL & Bandits

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.markdown("""
<div style="padding:15px; border-radius:10px; 
            background: linear-gradient(90deg, #4b79a1, #283e51); 
            text-align:center;">
    <h1 style="color:white; margin:0;">🍔 Dynamic Pricing using Reinforcement Learning</h1>
    <p style="color:#eee; font-size:18px; margin-top:5px;">
        Bandits • Profit-Aware Learning • Non-Stationary Environment • Streamlit App
    </p>
</div>
""", unsafe_allow_html=True)


# Use non-interactive backend
import matplotlib
matplotlib.use("Agg")

# ================================
# 1. RL ENVIRONMENT (Profit-aware)
# ================================
class DiscountBanditEnv:
    def __init__(self, discounts, conv_probs, profit_mode=True,
                 order_value_mean=400, order_value_sd=40, cost=0):

        self.discounts = np.array(discounts)
        self.conv_probs = np.array(conv_probs)
        self.n_arms = len(discounts)

        self.profit_mode = profit_mode
        self.order_value_mean = order_value_mean
        self.order_value_sd = order_value_sd
        self.cost = cost

    def step(self, arm):
        p = self.conv_probs[arm]
        order = np.random.rand() < p

        if order:
            value = max(0, np.random.normal(self.order_value_mean, self.order_value_sd))
            profit = value * (1 - self.discounts[arm]) - self.cost
            return profit, 1

        return 0, 0

    def set_conv_probs(self, new_probs):
        self.conv_probs = np.array(new_probs)


# ================================
# 2. Bandit Agents
# ================================
class EpsilonGreedy:
    def __init__(self, n_arms, eps=0.1):
        self.eps = eps
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select(self):
        if np.random.rand() < self.eps:
            return np.random.randint(len(self.values))
        return np.argmax(self.values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


class UCB1:
    def __init__(self, n_arms):
        self.t = 0
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select(self):
        self.t += 1
        for i in range(len(self.counts)):
            if self.counts[i] == 0:
                return i
        ucb = self.values + np.sqrt(2 * np.log(self.t) / self.counts)
        return np.argmax(ucb)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


class ThompsonSampling:
    def __init__(self, n_arms):
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select(self):
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, arm, ordered):
        self.alpha[arm] += ordered
        self.beta[arm] += 1 - ordered


# ================================
# 3. Simulation
# ================================
def run_bandit(env, agent, horizon, drift=False, drift_std=0.005):
    rewards = np.zeros(horizon)

    for t in range(horizon):
        arm = agent.select()
        reward, ordered = env.step(arm)

        # Thompson needs binary feedback
        if isinstance(agent, ThompsonSampling):
            agent.update(arm, ordered)
        else:
            agent.update(arm, reward)

        rewards[t] = reward

        # non-stationary drift
        if drift and t % 300 == 0 and t > 0:
            env.set_conv_probs(env.conv_probs + np.random.normal(0, drift_std, len(env.conv_probs)))

    return np.cumsum(rewards)



# ================================
# STREAMLIT UI
# ================================


st.write("This dashboard compares different bandit strategies for selecting optimal discounts in a food delivery platform.")

st.sidebar.header("Configuration")

discounts = st.sidebar.multiselect(
    "Select Discount Percentages",
    [0, 5, 10, 20, 30],
    default=[0, 5, 10, 20, 30]
)
discounts = [d/100 for d in sorted(discounts)]

conv_probs = st.sidebar.multiselect(
    "Base Conversion Probabilities per Discount",
    [0.04, 0.06, 0.095, 0.12, 0.11],
    default=[0.04, 0.06, 0.095, 0.12, 0.11]
)

if len(conv_probs) != len(discounts):
    st.error("⚠ Conversion probabilities must match number of discounts!")
    st.stop()

horizon = st.sidebar.slider("Time Steps", 500, 10000, 3000)
runs = st.sidebar.slider("Repeated Runs", 1, 50, 10)
drift = st.sidebar.checkbox("Enable Non-Stationary Drift", value=True)

algo = st.selectbox("Choose Bandit Algorithm", ["Epsilon-Greedy", "UCB1", "Thompson Sampling"])

if st.button("Run Simulation"):
    st.write("### Results")

    all_runs = []

    for r in range(runs):
        env = DiscountBanditEnv(discounts, conv_probs, profit_mode=True)
        if algo == "Epsilon-Greedy":
            agent = EpsilonGreedy(len(discounts), eps=0.1)
        elif algo == "UCB1":
            agent = UCB1(len(discounts))
        else:
            agent = ThompsonSampling(len(discounts))

        rewards = run_bandit(env, agent, horizon, drift=drift)
        all_runs.append(rewards)

    all_runs = np.array(all_runs)
    mean_curve = all_runs.mean(axis=0)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(mean_curve, label=f"{algo} Mean Profit")
    ax.set_title("Cumulative Profit Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative Profit")
    ax.legend()

    st.pyplot(fig)

    st.write("### Summary")
    st.write(f"**Final Mean Profit:** {mean_curve[-1]:.2f}")

    st.write("### Raw Data")
    st.dataframe(pd.DataFrame(all_runs).T)


st.write("---")
st.write("Developed with ❤️ using Streamlit + RL")
