import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from env.env_kwargs import *

MODELS = {"a2c": [], "ddpg": [], "td3": [], "sac": [], "ppo": []}

val_test_period = (TEST_START_DATE, TEST_END_DATE)
unique_trade_date = df_full[
    (df_full.date > val_test_period[0]) & (df_full.date <= val_test_period[1])
].date.unique()

rebalance_window = 63  # rebalance_window is the number of days to retrain the model
validation_window = 63  # validation_window is the number of days to do validation and trading (e.g. if validation_window=63, then both validation and trading period will be 63 days)


def get_validation_sharpe(iteration, model_name):
    """Calculate Sharpe ratio based on validation results"""
    df_total_value = pd.read_csv(
        f"results/account_value_validation_{model_name}_{iteration}.csv"
    )
    # If the agent did not make any transaction
    if df_total_value["daily_return"].var() == 0:
        if df_total_value["daily_return"].mean() > 0:
            return np.inf
        else:
            return 0.0
    else:
        return (
            (4**0.5)
            * df_total_value["daily_return"].mean()
            / df_total_value["daily_return"].std()
        )


model_use = []
validation_start_date_list = []
validation_end_date_list = []
iteration_list = []
sharpe_list = []
model_dct = {k: {"sharpe_list": [], "sharpe": -1} for k in MODELS.keys()}

for i in range(
    rebalance_window + validation_window,
    len(unique_trade_date),
    rebalance_window,
):
    validation_start_date = unique_trade_date[i - rebalance_window - validation_window]
    validation_end_date = unique_trade_date[i - rebalance_window]

    validation_start_date_list.append(validation_start_date)
    validation_end_date_list.append(validation_end_date)
    iteration_list.append(i)

    for model_name in MODELS.keys():
        sharpe = get_validation_sharpe(i, model_name=model_name)
        print(f"{model_name} Sharpe Ratio: ", sharpe)
        model_dct[model_name]["sharpe_list"].append(sharpe)


df_summary = pd.DataFrame(
    [
        iteration_list,
        validation_start_date_list,
        validation_end_date_list,
        model_use,
        model_dct["a2c"]["sharpe_list"],
        model_dct["ppo"]["sharpe_list"],
        model_dct["ddpg"]["sharpe_list"],
        model_dct["sac"]["sharpe_list"],
        model_dct["td3"]["sharpe_list"],
    ]
).T
df_summary.columns = [
    "Iter",
    "Val Start",
    "Val End",
    "Model Used",
    "A2C Sharpe",
    "PPO Sharpe",
    "DDPG Sharpe",
    "SAC Sharpe",
    "TD3 Sharpe",
]

print(df_summary.head())
df_summary.to_csv(RESULTS_DIR + "/ensemble/ensemble_summary.csv")
