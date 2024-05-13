import os
import pandas as pd

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent, DRLEnsembleAgent
from stable_baselines3.common.logger import configure
from finrl import config_tickers
from finrl.main import check_and_make_directories
from finrl.config import INDICATORS, RESULTS_DIR

TRAINED_MODEL_DIR = "src/models/trained_models"

check_and_make_directories([TRAINED_MODEL_DIR])

train = pd.read_csv("data/train_data.csv")
trade = pd.read_csv("data/trade_data.csv")

# If you are not using the data generated from part 1 of this tutorial, make sure
# it has the columns and index in the form that could be make into the environment.
# Then you can comment and skip the following two lines.
train = train.set_index(train.columns[0])
train.index.names = [""]

trade = trade.set_index(trade.columns[0])
trade.index.names = [""]

df_full = pd.concat([train, trade], ignore_index=True)
# df_full.index = df_full["date"].factorize()[0]
df_full.index.names = [""]


stock_dimension = len(train.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.0015] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "buy_cost_pct": 0.0015,
    "sell_cost_pct": 0.0015,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
    "print_verbosity": 5,
}

env_kwargs_gym = {
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


e_train_gym = StockTradingEnv(df=train, **env_kwargs_gym)

env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))

TRAIN_START_DATE = "2012-04-01"
TRAIN_END_DATE = "2023-03-31"
TEST_START_DATE = "2023-04-01"
TEST_END_DATE = "2024-04-01"

rebalance_window = 63  # rebalance_window is the number of days to retrain the model
validation_window = 63  # validation_window is the number of days to do validation and trading (e.g. if validation_window=63, then both validation and trading period will be 63 days)

print(df_full.tail())
ensemble_agent = DRLEnsembleAgent(
    df=df_full,
    train_period=(TRAIN_START_DATE, TRAIN_END_DATE),
    val_test_period=(TEST_START_DATE, TEST_END_DATE),
    rebalance_window=rebalance_window,
    validation_window=validation_window,
    **env_kwargs,
)

A2C_model_kwargs = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0007}

PPO_model_kwargs = {
    "ent_coef": 0.01,
    "n_steps": 2048,
    "learning_rate": 0.00025,
    "batch_size": 128,
}

DDPG_model_kwargs = {
    # "action_noise":"ornstein_uhlenbeck",
    "buffer_size": 10_000,
    "learning_rate": 0.0005,
    "batch_size": 64,
}

TD3_model_kwargs = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}

SAC_model_kwargs = {
    "batch_size": 128,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}

timesteps_dict = {
    "a2c": 1_000,
    "ppo": 1_000,
    "ddpg": 1_000,
    "td3": 1_000,
    "sac": 1_000,
}

df_summary = ensemble_agent.run_ensemble_strategy(
    A2C_model_kwargs=A2C_model_kwargs,
    PPO_model_kwargs=PPO_model_kwargs,
    DDPG_model_kwargs=DDPG_model_kwargs,
    TD3_model_kwargs=TD3_model_kwargs,
    SAC_model_kwargs=SAC_model_kwargs,
    timesteps_dict=timesteps_dict,
)

df_summary.to_csv(RESULTS_DIR + "/ensemble_summary.csv")

"""A2C Model"""
agent = DRLAgent(env=env_train)
model_ddpg = agent.get_model("ddpg", model_kwargs=DDPG_model_kwargs)

# set up logger
tmp_path = RESULTS_DIR + "/ddpg"
new_logger_ddpg = configure(tmp_path, ["stdout", "csv", "tensorboard"])
# Set new logger
model_ddpg.set_logger(new_logger_ddpg)

trained_ddpg = agent.train_model(
    model=model_ddpg, tb_log_name="ddpg", total_timesteps=timesteps_dict["ddpg"]
)
trained_ddpg.save(TRAINED_MODEL_DIR + "/agent_ddpg")

"""PPO Model"""
agent = DRLAgent(env=env_train)

model_ppo = agent.get_model("ppo", model_kwargs=PPO_model_kwargs)

# set up logger
tmp_path = RESULTS_DIR + "/ppo"
new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
# Set new logger
model_ppo.set_logger(new_logger_ppo)

trained_ppo = agent.train_model(
    model=model_ppo, tb_log_name="ppo", total_timesteps=timesteps_dict["ppo"]
)
trained_ppo.save(TRAINED_MODEL_DIR + "/agent_ppo")

"""TD3 Model"""
agent = DRLAgent(env=env_train)

model_td3 = agent.get_model("td3", model_kwargs=TD3_model_kwargs)

# set up logger
tmp_path = RESULTS_DIR + "/td3"
new_logger_td3 = configure(tmp_path, ["stdout", "csv", "tensorboard"])
# Set new logger
model_td3.set_logger(new_logger_td3)

trained_td3 = agent.train_model(
    model=model_td3, tb_log_name="td3", total_timesteps=timesteps_dict["td3"]
)
trained_td3.save(TRAINED_MODEL_DIR + "/agent_td3")

"""SAC Model"""
agent = DRLAgent(env=env_train)

model_sac = agent.get_model("sac", model_kwargs=SAC_model_kwargs)

# set up logger
tmp_path = RESULTS_DIR + "/sac"
new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
# Set new logger
model_sac.set_logger(new_logger_sac)

trained_sac = agent.train_model(
    model=model_sac, tb_log_name="sac", total_timesteps=timesteps_dict["sac"]
)
trained_sac.save(TRAINED_MODEL_DIR + "/agent_sac")
