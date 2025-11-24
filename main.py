import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ============================================================
# 1. ENVIRONMENT (Profit-aware + Non-Stationary support)
# ============================================================
class DiscountBanditEnv:
    def __init__(self, discounts, conv_probs, profit_mode=False,
                 order_value_mean=400, order_value_sd=40, cost=0):

        self.discounts = np.array(discounts)
        self.conv_probs = np.array(conv_probs)
        self.n_arms = len(discounts)

        self.profit_mode = profit_mode
        self.order_value_mean = order_value_mean
        self.order_value_sd = order_value_sd
        self.cost = cost

    def step(self, arm):
        """Simulate choosing a discount arm."""
        p = self.conv_probs[arm]
        order = np.random.rand() < p   # Bernoulli conversion

        if order:
            value = max(0, np.random.normal(self.order_value_mean, self.order_value_sd))
            profit = value * (1 - self.discounts[arm]) - self.cost
            return (profit if self.profit_mode else 1), 1

        return (0, 0)

    def set_conv_probs(self, new_probs):
        """Update probabilities (for non-stationary drift)."""
        self.conv_probs = np.array(new_probs)


# ============================================================
# 2. MAB AGENTS
# ============================================================
class EpsilonGreedy:
    def __init__(self, n_arms, eps=0.1):
        self.eps = eps
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select(self):
        if np.random.rand() < self.eps:
            return np.random.randint(self.n_arms)
        return np.argmax(self.values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


class UCB1:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.t = 0

    def select(self):
        self.t += 1

        for a in range(self.n_arms):
            if self.counts[a] == 0:
                return a

        ucb = self.values + np.sqrt(2 * np.log(self.t) / self.counts)
        return np.argmax(ucb)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


class ThompsonSamplingBernoulli:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select(self):
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, arm, ordered):
        self.alpha[arm] += ordered
        self.beta[arm] += 1 - ordered


# ============================================================
# 3. RUN SINGLE EXPERIMENT
# ============================================================
def run_one(env, agent, horizon=5000):
    rewards = np.zeros(horizon)

    for t in range(horizon):
        arm = agent.select()
        reward, ordered = env.step(arm)

        # Thompson uses only binary feedback
        if isinstance(agent, ThompsonSamplingBernoulli):
            agent.update(arm, ordered)
        else:
            agent.update(arm, reward)

        rewards[t] = reward

    return np.cumsum(rewards)


# ============================================================
# 4. BASE SETTINGS
# ============================================================
discounts = [0.0, 0.05, 0.10, 0.20, 0.30]
true_probs = [0.04, 0.06, 0.095, 0.12, 0.11]

HORIZON = 5000
RUNS = 20


def env_factory(seed, profit_mode):
    np.random.seed(seed)
    return DiscountBanditEnv(
        discounts, true_probs,
        profit_mode=profit_mode,
        order_value_mean=400, order_value_sd=40
    )


# ============================================================
# 5. PROFIT-AWARE EXPERIMENTS
# ============================================================
def run_profit_experiments():
    agents = {
        "Epsilon-0.1": lambda: EpsilonGreedy(len(discounts), eps=0.1),
        "UCB1": lambda: UCB1(len(discounts)),
        "Thompson": lambda: ThompsonSamplingBernoulli(len(discounts))
    }

    profit_results = {}

    for name, ctor in agents.items():
        print(f"\nRunning profit-aware model: {name}")
        all_runs = np.zeros((RUNS, HORIZON))

        for r in range(RUNS):
            env = env_factory(r, profit_mode=True)
            agent = ctor()
            all_runs[r] = run_one(env, agent, HORIZON)

        profit_results[name] = all_runs

    return profit_results


# ============================================================
# 6. NON-STATIONARY EXPERIMENTS
# ============================================================
def run_nonstationary(agent_ctor, horizon=5000, runs=15):
    all_runs = np.zeros((runs, horizon))

    for r in range(runs):
        np.random.seed(r)
        probs = np.array(true_probs)
        env = DiscountBanditEnv(discounts, probs.copy(), profit_mode=False)
        agent = agent_ctor()

        rewards = np.zeros(horizon)

        for t in range(horizon):
            # Drift every 300 steps
            if t % 300 == 0 and t > 0:
                probs += np.random.normal(0, 0.004, size=len(probs))
                probs = np.clip(probs, 0.01, 0.5)
                env.set_conv_probs(probs)

            arm = agent.select()
            reward, ordered = env.step(arm)

            if isinstance(agent, ThompsonSamplingBernoulli):
                agent.update(arm, ordered)
            else:
                agent.update(arm, reward)

            rewards[t] = reward

        all_runs[r] = np.cumsum(rewards)

    return all_runs


# ============================================================
# 7. MAIN EXECUTION
# ============================================================
if __name__ == "__main__":

    # -----------------------------------------
    # PROFIT-AWARE RESULTS
    # -----------------------------------------
    profit_results = run_profit_experiments()

    plt.figure()
    for name, data in profit_results.items():
        plt.plot(data.mean(axis=0), label=name)

    plt.title("Cumulative Profit Over Time")
    plt.xlabel("Time step")
    plt.ylabel("Cumulative Profit")
    plt.legend()
    plt.show()

    print("\n=== PROFIT SUMMARY ===")
    for name, data in profit_results.items():
        finals = data[:, -1]
        print(f"{name}: Mean={finals.mean():.2f}, 5%={np.percentile(finals,5):.2f}, 95%={np.percentile(finals,95):.2f}")

    # -----------------------------------------
    # NON-STATIONARY RESULTS
    # -----------------------------------------
    ns_agents = {
        "Epsilon-0.1": lambda: EpsilonGreedy(len(discounts)),
        "UCB1": lambda: UCB1(len(discounts)),
        "Thompson": lambda: ThompsonSamplingBernoulli(len(discounts))
    }

    plt.figure()
    for name, ctor in ns_agents.items():
        print(f"\nRunning Non-Stationary: {name}")
        res = run_nonstationary(ctor)
        plt.plot(res.mean(axis=0), label=name)

    plt.title("Non-Stationary: Cumulative Reward")
    plt.xlabel("Time step")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()
