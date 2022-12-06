import hashlib
import json
import os
import pprint

import tensorflow as tf

import wandb


class RLExperimentFramework:

    def __init__(self):
        self._agent = None

        self._config = {}
        self.update_config()

    def init_model(self, model_func_name: str, **kwargs):
        pass
        # models = Models(self.e_num_states, self.e_num_actions, learning_rate=self.a_learning_rate)
        # init_model = getattr(models, model_func_name)
        # self._a_model, self._a_optimizer, self._a_loss = init_model(**kwargs)

    def init_experiment(self, training, use_wandb=True):
        self.update_config()
        log.info(f"Config:\n{pprint.pformat(self._config)}")

    def update_config(self):
        self._config.update({k: v for k, v in self.__dict__.items() if not k.startswith("_")})
