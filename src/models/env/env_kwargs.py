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


A2C_model_kwargs = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0007}

PPO_model_kwargs = {
    "ent_coef": 0.01,
    "n_steps": 2048,
    "learning_rate": 0.00025,
    "batch_size": 128,
}

DDPG_model_kwargs = {
    # "action_noise":"ornstein_uhlenbeck",
    "buffer_size": 50_000,
    "learning_rate": 0.0005,
    "batch_size": 128,
}

TD3_model_kwargs = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}

SAC_model_kwargs = {
    "batch_size": 128,
    "buffer_size": 50_000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}

timesteps_dict = {
    "a2c": 50_000,
    "ppo": 50_000,
    "ddpg": 50_000,
    "td3": 50_000,
    "sac": 50_000,
}
