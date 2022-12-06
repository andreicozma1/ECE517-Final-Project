import collections
import hashlib
import json
import logging
import os
import pprint
from math import inf

import numpy as np
import tqdm
import tensorflow as tf
import wandb

from BaseAgent import BaseAgent
from Environment import PongEnvironment
from BaseEnvironment import BaseEnvironment


class Experiment:
    def __init__(self, env: BaseEnvironment, agent: BaseAgent):
        super().__init__()
        self.env: BaseEnvironment = env
        self.agent: BaseAgent = agent
        logging.info(f"Args:\n{pprint.pformat(self.__dict__, width=30)}")

    @property
    def config(self):
        return {
                "environment": self.env.config,
                "agent"      : self.agent.config,
        }

    def run_experiment(self,
                       max_episodes=1000,
                       max_steps_per_episode=2500,
                       running_rew_len=50,
                       training=True,
                       use_wandb: bool = True):

        hash_all = hashlib.md5(json.dumps(self.config).encode('utf-8')).hexdigest()
        model_hash = self.config["agent"]["nn"]["config_hash"]
        wandb.init(project="ECE517",
                   entity="utkteam",
                   mode="online" if use_wandb else "disabled",
                   name=hash_all,
                   group=self.config["agent"]["nn"]["name"],
                   job_type="train" if training else "test",
                   tags=[model_hash],
                   config=self.config)
        model_img_filename = f"{model_hash}.png"
        tf.keras.utils.plot_model(self.agent.nn.model,
                                  to_file=model_img_filename,
                                  show_layer_names=True,
                                  show_shapes=True,
                                  show_dtype=True,
                                  expand_nested=True,
                                  show_layer_activations=True,
                                  dpi=120)
        wandb.log({
                "model": wandb.Image(model_img_filename)
        })
        os.remove(model_img_filename)

        stats = {}
        episodes_reward: collections.deque = collections.deque(maxlen=running_rew_len)
        episodes_loss: collections.deque = collections.deque(maxlen=running_rew_len)

        tq = tqdm.trange(max_episodes, desc="Train", leave=True)
        for ep in tq:
            steps, total_reward, loss = self.agent.train_step(self.env, max_steps_per_episode)
            steps, total_reward, loss = steps.numpy(), total_reward.numpy(), loss.numpy()
            episodes_reward.append(total_reward)
            loss = np.sum(loss)
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
            wandb.log(stats)

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
