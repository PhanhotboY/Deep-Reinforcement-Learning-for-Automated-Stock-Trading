import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from env.env_kwargs import *

agent = DRLAgent(env=env_train)
model = agent.get_model("sac")

# set up logger
tmp_path = RESULTS_DIR + "/sac"
new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
# Set new logger
model.set_logger(new_logger_sac)

print("Training model sac...")
trained_sac = agent.train_model(
    model=model, tb_log_name="sac", total_timesteps=timesteps_dict["sac"]
)

trained_sac.save(TRAINED_MODEL_DIR + "/agent_sac")
