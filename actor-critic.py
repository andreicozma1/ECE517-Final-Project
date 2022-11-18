import collections
import logging
import os
import pprint
import random

import numpy as np
import tensorflow as tf
import tqdm

import wandb
from Environment import Environment
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


class Experiment:
    def __init__(self, env: Environment, agent: Agent):
        super().__init__()
        self.env: Environment = env
        self.agent: Agent = agent
        logging.info(f"Agent Args: {pprint.pformat(self.__dict__)}")

    def run_experiment(self,
                       min_episodes_criterion=500,
                       max_episodes=50000,
                       max_steps_per_episode=2500,
                       use_wandb: bool = True):
        # self.init_experiment(training, use_wandb=use_wandb)

        stats = {}
        episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)
        tq = tqdm.trange(max_episodes)
        for ep in tq:
            episode_stats = self.agent.train_step(self.env, max_steps_per_episode)

            episodes_reward.append(episode_stats["total_reward"])
            stats |= {
                    "episode"       : ep,
                    "running_reward": np.mean(episodes_reward),
                    **episode_stats
            }
            # wandb.log(stats)

            tq.set_postfix(episode_reward=stats['total_reward'],
                           running_reward=stats['running_reward'],
                           steps=stats['steps'])


def main():
    env = Environment(draw=True)
    nn = NeuralNet("dense", env.num_states, env.num_actions)
    agent = Agent(nn)

    exp = Experiment(env, agent)
    exp.run_experiment()


if __name__ == "__main__":
    main()
