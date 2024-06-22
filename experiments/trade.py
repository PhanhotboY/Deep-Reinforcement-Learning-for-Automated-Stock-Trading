import sys
import pandas as pd
import numpy as np

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from finrl.config import INDICATORS

TRAINED_MODEL_DIR = "src/models/trained_models"

train = pd.read_csv("data/train_data.csv")
trade = pd.read_csv("data/trade_data.csv")
vni = pd.read_csv("data/df_vni.csv")

# If you are not using the data generated from part 1 of this tutorial, make sure
# it has the columns and index in the form that could be make into the environment.
# Then you can comment and skip the following lines.
train = train.set_index(train.columns[0])
train.index.names = [""]
trade = trade.set_index(trade.columns[0])
trade.index.names = [""]

trained_a2c = A2C.load(TRAINED_MODEL_DIR + "/agent_a2c")
trained_ddpg = DDPG.load(TRAINED_MODEL_DIR + "/agent_ddpg")
trained_ppo = PPO.load(TRAINED_MODEL_DIR + "/agent_ppo")
trained_td3 = TD3.load(TRAINED_MODEL_DIR + "/agent_td3")
trained_sac = SAC.load(TRAINED_MODEL_DIR + "/agent_sac")

stock_dimension = len(trade.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.0015] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 10_000_000_000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
}

e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)
env_trade, obs_trade = e_trade_gym.get_sb_env()

df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
    model=trained_a2c, environment=e_trade_gym
)

df_account_value_ddpg, df_actions_ddpg = DRLAgent.DRL_prediction(
    model=trained_ddpg, environment=e_trade_gym
)

df_account_value_ppo, df_actions_ppo = DRLAgent.DRL_prediction(
    model=trained_ppo, environment=e_trade_gym
)

df_account_value_td3, df_actions_td3 = DRLAgent.DRL_prediction(
    model=trained_td3, environment=e_trade_gym
)

df_account_value_sac, df_actions_sac = DRLAgent.DRL_prediction(
    model=trained_sac, environment=e_trade_gym
)

# set precision for printing results
np.set_printoptions(precision=3, suppress=True)

unique_trade_date = trade.date.unique()

df_trade_date = pd.DataFrame({"datadate": unique_trade_date})

rebalance_window = 63
validation_window = 63

print(rebalance_window + validation_window, len(unique_trade_date) + 1)
df_account_value = pd.DataFrame()
for i in range(
    rebalance_window + validation_window, len(unique_trade_date) + 1, rebalance_window
):
    try:
        print(i)
        temp = pd.read_csv(
            "results/account_value_trade_{}_{}.csv".format("ensemble", i)
        )
        df_account_value = pd.concat([df_account_value, temp], ignore_index=True)
    except Exception as e:
        print(e)
        pass
sharpe = (
    (252**0.5)
    * df_account_value.account_value.pct_change(1).mean()
    / df_account_value.account_value.pct_change(1).std()
)
print("Sharpe Ratio: ", sharpe)
df_account_value = df_account_value.join(
    df_trade_date[validation_window:].reset_index(drop=True)
)

df_result_ensemble = pd.DataFrame(
    {"date": df_account_value["date"], "ensemble": df_account_value["account_value"]}
)
df_result_ensemble = df_result_ensemble.set_index("date")

print("df_result_ensemble.columns: ", df_result_ensemble.columns)
print("df_trade_date: ", df_trade_date)

df_result_ensemble.to_csv("df_result_ensemble.csv")
print("df_result_ensemble: ", df_result_ensemble)

df_vni = pd.DataFrame()
df_vni["date"] = vni["date"]
first_day = vni["close"][0]
df_vni["vnindex"] = vni["close"] / first_day * 10_000_000_000
df_vni.set_index("date", inplace=True)

df_result_a2c = df_account_value_a2c.set_index(df_account_value_a2c.columns[0])
df_result_ddpg = df_account_value_ddpg.set_index(df_account_value_ddpg.columns[0])
df_result_ppo = df_account_value_ppo.set_index(df_account_value_ppo.columns[0])
df_result_td3 = df_account_value_td3.set_index(df_account_value_td3.columns[0])
df_result_sac = df_account_value_sac.set_index(df_account_value_sac.columns[0])

result = pd.DataFrame()

result = pd.merge(result, df_result_a2c, how="outer", left_index=True, right_index=True)

result = pd.merge(
    result,
    df_result_ddpg,
    how="outer",
    left_index=True,
    right_index=True,
    suffixes=("", "_drop"),
)
result = pd.merge(
    result,
    df_result_ppo,
    how="outer",
    left_index=True,
    right_index=True,
    suffixes=("", "_drop"),
)
result = pd.merge(
    result,
    df_result_td3,
    how="outer",
    left_index=True,
    right_index=True,
    suffixes=("", "_drop"),
)
result = pd.merge(
    result,
    df_result_sac,
    how="outer",
    left_index=True,
    right_index=True,
    suffixes=("", "_drop"),
)
result = pd.merge(
    result, df_result_ensemble, how="outer", left_index=True, right_index=True
).fillna(method="bfill")

result = pd.merge(
    result, df_vni, how="outer", left_index=True, right_index=True
).fillna(method="bfill")

result = result.dropna()

result.columns = ["A2C", "DDPG", "PPO", "TD3", "SAC", "Ensemble", "VNINDEX"]

result.index = pd.to_datetime(result.index)
result = result.loc["2019-01-01":"2024-01-01"]
result.to_csv("trade_result.csv")
