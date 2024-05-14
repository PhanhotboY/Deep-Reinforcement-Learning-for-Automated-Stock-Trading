from env.env_kwargs import *

ensemble_agent = DRLEnsembleAgent(
    df=df_full,
    train_period=(TRAIN_START_DATE, TRAIN_END_DATE),
    val_test_period=(TEST_START_DATE, TEST_END_DATE),
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

df_summary.to_csv(RESULTS_DIR + "/ensemble_summary.csv")

"""DDPG Model"""
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
