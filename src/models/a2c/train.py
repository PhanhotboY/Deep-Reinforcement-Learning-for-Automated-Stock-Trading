import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from env.env_kwargs import *

agent = DRLAgent(env=env_train)
model = agent.get_model("a2c")

# set up logger
tmp_path = RESULTS_DIR + "/a2c"
new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
# Set new logger
model.set_logger(new_logger_a2c)

trained_a2c = agent.train_model(
    model=model, tb_log_name="a2c", total_timesteps=timesteps_dict["a2c"]
)

trained_a2c.save(TRAINED_MODEL_DIR + "/agent_a2c")
