import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from env.env_kwargs import *

ensemble_agent = DRLEnsembleAgent(
    df=df_full,
    train_period=(TRAIN_START_DATE, TRAIN_END_DATE),
    val_test_period=(TRADE_START_DATE, TRADE_END_DATE),
    rebalance_window=rebalance_window,
    validation_window=validation_window,
    **env_kwargs,
)

df_summary = ensemble_agent.run_ensemble_strategy(
    A2C_model_kwargs=A2C_model_kwargs,
    PPO_model_kwargs=PPO_model_kwargs,
    DDPG_model_kwargs=DDPG_model_kwargs,
    TD3_model_kwargs=TD3_model_kwargs,
    SAC_model_kwargs=SAC_model_kwargs,
    timesteps_dict=timesteps_dict,
)
print(df_summary.head())
df_summary.to_csv(RESULTS_DIR + "/ensemble/ensemble_summary.csv")
