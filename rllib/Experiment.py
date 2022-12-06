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

from rllib.BaseAgent import BaseAgent
from rllib.BaseEnvironment import BaseEnvironment


class Experiment:
    def __init__(self, env: BaseEnvironment, agent: BaseAgent, use_wandb: bool = True):
        super().__init__()
        self.env: BaseEnvironment = env
        self.agent: BaseAgent = agent
        self.use_wandb = use_wandb
        logging.info(f"Args:\n{pprint.pformat(self.__dict__, width=30)}")

    @property
    def config(self):
        return {
                "env"      : self.env.config,
                "agent"    : self.agent.config,
                "use_wandb": self.use_wandb,
        }

    def run_experiment(self,
                       max_episodes=1000,
                       max_steps_per_episode=2500,
                       running_rew_len=50,
                       training=True):
        self._init_experiment(training)
        self.plot_model()

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

            wandb.log(stats)

        return stats

    def _init_experiment(self, training):
        hash_config = hashlib.md5(json.dumps(self.config).encode('utf-8')).hexdigest()
        hash_model = self.config["agent"]["nn"]["config_hash"]
        wandb.init(project="ECE517",
                   entity="utkteam",
                   mode="online" if self.use_wandb else "disabled",
                   name=hash_config,
                   group=self.config["agent"]["nn"]["name"],
                   job_type="train" if training else "test",
                   tags=[hash_model],
                   config=self.config)
        return hash_config, hash_model

    def plot_model(self, filename="model", temporary=False):
        # if doesnt have extension
        if not os.path.splitext(filename)[1]:
            filename += ".png"

        filename = os.path.join(self.save_path, filename)
        tf.keras.utils.plot_model(self.agent.nn.model,
                                  to_file=filename,
                                  show_layer_names=True,
                                  show_shapes=True,
                                  show_dtype=True,
                                  expand_nested=True,
                                  show_layer_activations=True,
                                  dpi=120)
        if self.use_wandb:
            wandb.log({
                    "model": wandb.Image(filename)
            })
        if temporary:
            os.remove(filename)
