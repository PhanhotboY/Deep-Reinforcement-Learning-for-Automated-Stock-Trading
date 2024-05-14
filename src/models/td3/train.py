import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from env.env_kwargs import *

agent = DRLAgent(env=env_train)
model = agent.get_model("td3")

# set up logger
tmp_path = RESULTS_DIR + "/td3"
new_logger_td3 = configure(tmp_path, ["stdout", "csv", "tensorboard"])
# Set new logger
model.set_logger(new_logger_td3)

print("Training model td3...")
trained_td3 = agent.train_model(
    model=model, tb_log_name="td3", total_timesteps=timesteps_dict["td3"]
)

trained_td3.save(TRAINED_MODEL_DIR + "/agent_td3")
