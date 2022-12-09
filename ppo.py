import logging
import pprint
import random
from typing import Tuple

import numpy as np
import tensorflow as tf

# This is a hacky fix for tensorflow imports to work with intellisense
from rllib.BaseAgent import BaseAgent
from rllib.PPOAgent import PPOAgent
from rllib.PPOSimpleAgent import PPOSimpleAgent
from rllib.Environments import LunarLander
from rllib.Experiment import PPOExperiment
from rllib.Network import Network, SimpleNetwork
from rllib.PlotHelper import PlotHelper
from rllib.utils import logging_setup

# Logging to stdout and file with logging class

log = logging_setup(file=__file__, name=__name__, level=logging.INFO)
# os.environ['WANDB_SILENT'] = "true"

# Set seed for experiment reproducibility
seed = 42

tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

def main():
    # env = PongEnvironment(draw=True, draw_speed=None, state_scaler_enable=True)
    env = LunarLander(draw=True, draw_speed=None, state_scaler_enable=True)

    # nn = SimpleNetwork("simplenet",
    #              env.num_states, env.num_actions,
    #              max_timesteps=1, learning_rate=3e-4)
    # agent = PPOSimpleAgent(nn, epsilon=0.2,
    #                        steps_per_epoch=300,
    #                        train_iterations=10,
    #                        clip_ratio=.2)
    nn = Network("transformer",
                 env.num_states, env.num_actions,
                 max_timesteps=5, learning_rate=0.0001)
    agent = PPOAgent(nn, epsilon=0.2,
                           steps_per_epoch=500,
                           train_iterations=4,
                           clip_ratio=.2)

    exp = PPOExperiment(env, agent, use_wandb=False)

    exp.run_experiment()


if __name__ == "__main__":
    main()
