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
        hash_all = hashlib.md5(json.dumps(self._config).encode('utf-8')).hexdigest()
        model_hash = self._config["model"]["config_hash"]
        wandb.init(project="ECE517",
                   entity="utkteam",
                   mode="online" if use_wandb else "disabled",
                   name=hash_all,
                   group=self._config["model"]["func_name"],
                   job_type="train" if training else "test",
                   tags=[f"Opt:{self._a_optimizer.__class__.__name__}",
                         f"Loss:{self._a_loss.__class__.__name__}",
                         model_hash],
                   config=self._config)
        model_img_filename = f"{model_hash}.png"
        tf.keras.utils.plot_model(self._a_model,
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

    def update_config(self):
        self._config.update({k: v for k, v in self.__dict__.items() if not k.startswith("_")})
