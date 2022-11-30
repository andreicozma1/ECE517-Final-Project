import collections
import logging
import pprint

import numpy as np
import tqdm

from BaseAgent import BaseAgent
from Environment import Environment


class Experiment:
    def __init__(self, env: Environment, agent: BaseAgent):
        super().__init__()
        self.env: Environment = env
        self.agent: BaseAgent = agent
        logging.info(f"Agent Args: {pprint.pformat(self.__dict__)}")

    def run_experiment(self,
                       min_episodes_criterion=50,
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
            stats.update({
                    "episode"       : ep,
                    "running_reward": np.mean(episodes_reward),
                    **episode_stats
            })
            # wandb.log(stats)

            tq.set_postfix(episode_reward=stats['total_reward'],
                           running_reward=stats['running_reward'],
                           steps=stats['steps'])
