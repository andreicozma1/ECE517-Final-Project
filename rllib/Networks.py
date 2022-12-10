import abc
import hashlib
import json
import logging
import os
import pprint
from typing import Tuple

import numpy as np
import tensorflow as tf

from rllib.nn.Common import CommonTransformer
from rllib.nn.Heads import ActorCriticHead

keras = tf.keras


class BaseNetwork:

    def __init__(self, name: str, num_states: int, num_actions: int,
                 max_timesteps: int = 1,
                 learning_rate: float = 0.0001,
                 **kwargs):
        self.name: str = name
        self.num_features: int = num_states
        self.num_actions: int = num_actions
        self.learning_rate: float = learning_rate
        self.max_timesteps: int = max_timesteps
        self.kwargs = kwargs

        self.inp_p_shape = (None, self.max_timesteps)
        self.inp_s_shape = (None, self.max_timesteps, self.num_features)
        self.inp_a_shape = (None, self.max_timesteps, self.num_actions)

        self.model, self.optimizer = self.build_model()
        self.model.summary()

        self.model_config = self.model.to_json()
        self.model_config_hash = hashlib.md5(json.dumps(json.loads(self.model_config)).encode('utf-8')).hexdigest()
        self.path_network = os.path.join("saves", "networks", self.model_config_hash)
        os.makedirs(self.path_network, exist_ok=True)
        self.plot_model()
        logging.info(f"Args:\n{pprint.pformat(self.__dict__, width=30)}")

    def plot_model(self, filename="model"):
        # if doesnt have extension
        if not os.path.splitext(filename)[1]:
            filename += ".png"

        filename = os.path.join(self.path_network, filename)
        tf.keras.utils.plot_model(self.model,
                                  to_file=filename,
                                  show_layer_names=True,
                                  show_shapes=True,
                                  show_dtype=True,
                                  expand_nested=True,
                                  show_layer_activations=True,
                                  dpi=120)

    @abc.abstractmethod
    def build_model(self) -> Tuple[keras.Model, keras.optimizers.Optimizer]:
        pass

    @property
    def config(self):
        return {
                "name"       : self.name,
                "kwargs"     : self.kwargs,
                "num_layers" : len(self.model.layers),
                "num_params" : self.model.count_params(),
                "config"     : self.model_config,
                "config_hash": self.model_config_hash,
        }


class TransformerNetwork(BaseNetwork):
    def __init__(self, num_states: int, num_actions: int,
                 max_timesteps: int = 1,
                 learning_rate: float = 0.0001,
                 **kwargs):
        super().__init__("transformer_net", num_states, num_actions, max_timesteps, learning_rate, **kwargs)

    def build_model(self) -> Tuple[keras.Model, keras.optimizers.Optimizer]:
        inp_p_layer = keras.layers.Input(batch_input_shape=self.inp_p_shape, name="inp_pos_layer")
        inp_s_layer = keras.layers.Input(batch_input_shape=self.inp_s_shape, name="inp_states_layer")
        inp_a_layer = keras.layers.Input(batch_input_shape=self.inp_a_shape, name="inp_actions_layer")
        inputs = [inp_p_layer, inp_s_layer, inp_a_layer]

        common = CommonTransformer(seq_len=self.max_timesteps)
        common_out = common(inputs)

        head = ActorCriticHead(num_actions=self.num_actions)
        head_out = head(common_out)

        model = keras.Model(inputs=inputs, outputs=head_out, name=self.name)
        optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        return model, optimizer


class SimpleNetwork(BaseNetwork):
    def __init__(self, num_states: int, num_actions: int,
                 max_timesteps: int = 1,
                 learning_rate: float = 0.0001,
                 **kwargs):
        super().__init__("simple_net", num_states, num_actions, max_timesteps, learning_rate, **kwargs)

    def predict(self, state):
        actor_probs, critic_out = self.model(state)
        actor_probs = tf.nn.log_softmax(actor_probs)
        return actor_probs, critic_out

    def get_log_probs(self, state):
        actor_probs, _ = self.model(state)
        return tf.nn.log_softmax(actor_probs)

    def get_action(self, state):
        actor_probs, _ = self.model(state)
        return tf.squeeze(tf.random.categorical(actor_probs, 1), axis=-1)

    def build_model(self) -> Tuple[keras.Model, keras.optimizers.Optimizer]:
        inp_s_shape = (None, self.max_timesteps, self.num_features)

        inp_s_layer = keras.layers.Input(batch_input_shape=inp_s_shape, name="inp_states_layer")
        # common state encoder
        body = keras.layers.Dense(64, activation="relu")(inp_s_layer)
        body = keras.layers.Dense(64, activation="relu")(body)
        body = keras.layers.Dense(64, activation="relu")(body)
        body = keras.layers.Dense(64, activation="relu")(body)

        # actor and critic head
        head = ActorCriticHead(num_actions=self.num_actions)
        head_out = head(body)

        model = keras.Model(inputs=[inp_s_layer], outputs=head_out, name=self.name)
        optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)

        return model, optimizer
