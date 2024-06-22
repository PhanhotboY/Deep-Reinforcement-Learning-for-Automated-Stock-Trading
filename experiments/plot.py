import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import pandas as pd
import numpy as np

result = pd.read_csv("trade_result.csv")
result.set_index("date", inplace=True)
columns = ["A2C", "DDPG", "PPO", "TD3", "SAC", "Ensemble", "VNINDEX"]

# cummulative return
plt.rcParams["figure.figsize"] = (15, 5)
plt.figure()
result.plot(title="Cumulative Return").get_figure().savefig(
    "experiments/results/cumulative.png"
)

# anualized return
df = result.copy()
annualized_return = pd.DataFrame(index=[0])
for col in columns:
    cumulative = (df.tail(1)[col].values[0] - 10_000_000_000) / 10_000_000_000
    annualized_return[col] = [(1 + cumulative) ** (1 / 5) - 1]  # 5 years

fig, ax = plt.subplots(figsize=(12, 8))
annualized_return.T.plot(kind="bar", legend=False, ax=ax, width=0.8)
ax.set_title("Annualized Return")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
ax.set_xlabel("Model")
ax.set_ylabel("Return (%)")
ax.grid(axis="y", linestyle="--", alpha=0.7)

for idx, row in annualized_return.iterrows():
    ax.bar_label(
        ax.containers[idx],
        label_type="center",
        fmt="%.2f%%",
        labels=[f"{v:.2%}" for v in row],
    )

fig.savefig("experiments/results/annualized_return.png")

# annualized vol
df = result.copy()
df = df.pct_change(1).dropna()
annualized_vol = pd.DataFrame(index=[0])
for col in columns:
    annualized_vol[col] = df[col].std() * np.sqrt(252)  # daily to annualized vol
print("annualized_vol: ", annualized_vol)

fig, ax = plt.subplots(figsize=(12, 8))
annualized_vol.T.plot(kind="bar", legend=False, ax=ax, width=0.8)
ax.set_title("Annualized Volatility")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
ax.set_xlabel("Model")
ax.set_ylabel("Volatility (%)")
ax.grid(axis="y", linestyle="--", alpha=0.7)

for idx, row in annualized_vol.iterrows():
    ax.bar_label(
        ax.containers[idx],
        label_type="center",
        fmt="%.2f%%",
        labels=[f"{v:.2%}" for v in row],
    )

fig.savefig("experiments/results/annualized_vol.png")


# sharpe ratio
risk_free_rate = 0.02788
sharpe = pd.DataFrame(index=[0])
for col in columns:
    sharpe[col] = (annualized_return[col] - risk_free_rate) / annualized_vol[col]
    print(f"Sharpe Ratio {col}: ", sharpe[col].values[0])

print("Sharpe Ratio: ", sharpe)

fig, ax = plt.subplots(figsize=(12, 8))
sharpe.T.plot(kind="bar", legend=False, ax=ax, width=0.8)
ax.set_title("Sharpe Ratio")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
ax.set_xlabel("Model")
ax.set_ylabel("Sharpe ratio (%)")
ax.grid(axis="y", linestyle="--", alpha=0.7)

for idx, row in sharpe.iterrows():
    ax.bar_label(
        ax.containers[idx],
        label_type="center",
        fmt="%.2f%%",
        labels=[f"{v:.2%}" for v in row],
    )

fig.savefig("experiments/results/sharpe.png")

# max drawdown
df = result.copy()
max_dd = pd.DataFrame(index=[0])
for col in columns:
    cumulative_returns = df[col].values

    peak = cumulative_returns[0]
    trough = cumulative_returns[0]
    max_drawdown = 0.0

    for i in range(1, len(cumulative_returns)):
        if cumulative_returns[i] > peak:
            peak = cumulative_returns[i]
            trough = cumulative_returns[i]
        elif cumulative_returns[i] < trough:
            trough = cumulative_returns[i]

        drawdown = (peak - trough) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    max_dd[col] = [max_drawdown]
print("Max Drawdown: ", max_dd)

fig, ax = plt.subplots(figsize=(12, 8))
max_dd.T.plot(kind="bar", legend=False, ax=ax, width=0.8)
ax.set_title("Max Drawdown")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
ax.set_xlabel("Model")
ax.set_ylabel("Max Drawdown (%)")
ax.grid(axis="y", linestyle="--", alpha=0.7)

for idx, row in max_dd.iterrows():
    ax.bar_label(
        ax.containers[idx],
        label_type="center",
        fmt="%.2f%%",
        labels=[f"{v:.2%}" for v in row],
    )

fig.savefig("experiments/results/max_drawdown.png")
