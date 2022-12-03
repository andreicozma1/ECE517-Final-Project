import collections
import logging
import pprint
from math import inf

import numpy as np
import tqdm
import tensorflow as tf

from BaseAgent import BaseAgent
from PongEnvironment import BaseEnvironment, PongEnvironment


class Experiment:
    def __init__(self, env: BaseEnvironment, agent: BaseAgent):
        super().__init__()
        self.env: BaseEnvironment = env
        self.agent: BaseAgent = agent
        logging.info(f"Args:\n{pprint.pformat(self.__dict__, width=30)}")

    def run_experiment(self,
                       max_episodes=1000,
                       max_steps_per_episode=2500,
                       running_rew_len=50,
                       use_wandb: bool = True):
        stats = {}
        episodes_reward: collections.deque = collections.deque(maxlen=running_rew_len)
        episodes_loss: collections.deque = collections.deque(maxlen=running_rew_len)

        tq = tqdm.trange(max_episodes, desc="Train", leave=True)
        for ep in tq:
            steps, total_reward, loss = self.agent.train_step(self.env, max_steps_per_episode)
            steps, total_reward, loss = steps.numpy(), total_reward.numpy(), loss.numpy()
            episodes_reward.append(total_reward)
            episodes_loss.append(loss)
            stats |= {
                    "steps"         : steps,
                    "total_reward"  : total_reward,
                    "running_reward": np.mean(episodes_reward),
                    "loss"          : loss,
                    "running_loss"  : np.mean(episodes_loss),
                    "max_steps"     : max(stats.get("max_steps", 0), steps),
                    "min_reward"    : min(stats.get("min_reward", inf), total_reward),
                    "max_reward"    : max(stats.get("max_reward", -inf), total_reward),
            }
            tq.set_postfix(stats)

            # if use_wandb:
            # wandb.log(stats)

        return stats

    def run_test(self, max_episodes=500,
                 max_steps_per_episode=2500,
                 running_rew_len=50):
        tq = tqdm.trange(max_episodes, desc="Test", leave=True)
        stats = {}
        episodes_reward: collections.deque = collections.deque(maxlen=running_rew_len)

        for _ in tq:
            rewards, action_probs_hist, critic_returns_hist, steps, total_reward = self.agent.run_episode(self.env,
                                                                                                          max_steps_per_episode,
                                                                                                          deterministic=True)
            steps, total_reward = steps.numpy(), total_reward.numpy()
            episodes_reward.append(np.sum(rewards))
            stats |= {
                    "running_reward": np.mean(episodes_reward),
                    "max_steps"     : max(stats.get("max_steps", 0), steps),
                    "max_reward"    : max(stats.get("max_reward", -inf), total_reward),
                    "min_reward"    : min(stats.get("min_reward", inf), total_reward),
                    "steps"         : steps,
                    "total_reward"  : total_reward,
            }
            tq.set_postfix(stats)

            # if use_wandb:
            # wandb.log(stats)

        return stats
