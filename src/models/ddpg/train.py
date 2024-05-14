import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from env.env_kwargs import *

agent = DRLAgent(env=env_train)
model = agent.get_model("ddpg")

# set up logger
tmp_path = RESULTS_DIR + "/ddpg"
new_logger_ddpg = configure(tmp_path, ["stdout", "csv", "tensorboard"])
# Set new logger
model.set_logger(new_logger_ddpg)

print("Training model ddpg...")
trained_ddpg = agent.train_model(
    model=model, tb_log_name="ddpg", total_timesteps=timesteps_dict["ddpg"]
)

trained_ddpg.save(TRAINED_MODEL_DIR + "/agent_ddpg")
