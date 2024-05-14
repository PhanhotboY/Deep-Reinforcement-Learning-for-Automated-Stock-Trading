import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from env.env_kwargs import *

agent = DRLAgent(env=env_train)
model = agent.get_model("ppo")

# set up logger
tmp_path = RESULTS_DIR + "/ppo"
new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
# Set new logger
model.set_logger(new_logger_ppo)

print("Training model ppo...")
trained_ppo = agent.train_model(
    model=model, tb_log_name="ppo", total_timesteps=timesteps_dict["ppo"]
)

trained_ppo.save(TRAINED_MODEL_DIR + "/agent_ppo")
