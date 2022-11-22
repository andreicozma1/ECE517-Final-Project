import logging
import os
import random

import numpy as np
import tensorflow as tf

from Environment import Environment
from Experiment import Experiment
from NeuralNet import NeuralNet
from Agent import Agent
# This is a hacky fix for tensorflow imports to work with intellisense
from rllib.utils import logging_setup

# Logging to stdout and file with logging class

log = logging_setup(file=__file__, name=__name__, level=logging.INFO)
os.environ['WANDB_SILENT'] = "true"

# Set seed for experiment reproducibility
seed = 42

tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)


def main():
    env = Environment(draw=True)
    nn = NeuralNet("dense", env.num_states, env.num_actions)
    agent = Agent(nn)

    exp = Experiment(env, agent)
    exp.run_experiment()


if __name__ == "__main__":
    main()
