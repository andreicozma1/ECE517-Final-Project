import collections
import logging
import pprint
from math import inf

import numpy as np
import tqdm

from BaseAgent import BaseAgent
from PongEnvironment import PongEnvironment


class Experiment:
    def __init__(self, env: PongEnvironment, agent: BaseAgent):
        super().__init__()
        self.env: PongEnvironment = env
        self.agent: BaseAgent = agent
        logging.info(f"Agent Args: {pprint.pformat(self.__dict__)}")

    def run_experiment(self,
                       min_episodes_criterion=25,
                       max_episodes=50000,
                       max_steps_per_episode=2500,
                       use_wandb: bool = True):
        # self.init_experiment(training, use_wandb=use_wandb)
        stats = {}
        episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

        tq = tqdm.trange(max_episodes, desc="Experiment", leave=True)
        for _ in tq:
            episode_stats = self.agent.train_step(self.env, max_steps_per_episode)

            episodes_reward.append(episode_stats["total_reward"])
            stats |= {
                    "running_reward": np.mean(episodes_reward),
                    "max_steps"     : max(stats.get("max_steps", 0), episode_stats["steps"]),
                    "max_reward"    : max(stats.get("max_reward", -inf), episode_stats["total_reward"]),
                    "min_reward"    : min(stats.get("min_reward", inf), episode_stats["total_reward"]),
                    **episode_stats
            }

            # if use_wandb:
            # wandb.log(stats)

            tq.set_postfix(stats)
