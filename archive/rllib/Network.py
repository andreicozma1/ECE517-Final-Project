import hashlib
import json
import logging
import os
import pprint

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from rllib.CustomLayers import ActorCriticLayer, ActorLoss, StateAndPositionEmbedding, \
    TransformerActorCritic, \
    TransformerEncoders

keras = tf.keras


class Network:
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

        self.optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)

        self.inp_p_shape = (1, self.max_timesteps)
        self.inp_s_shape = (1, self.max_timesteps, self.num_features)
        self.inp_a_shape = (1, self.max_timesteps, self.num_actions)

        self.inp_p_layer = keras.layers.Input(batch_input_shape=self.inp_p_shape, name="inp_pos_layer")
        self.inp_s_layer = keras.layers.Input(batch_input_shape=self.inp_s_shape, name="inp_states_layer")
        self.inp_a_layer = keras.layers.Input(batch_input_shape=self.inp_a_shape, name="inp_actions_layer")
        inputs = [self.inp_p_layer, self.inp_s_layer, self.inp_a_layer]

        common = TransformerActorCritic(seq_len=self.max_timesteps)
        common_out = common(inputs)

        head = ActorCriticLayer(num_actions=self.num_actions)
        head_out = head(common_out)

        self.model = keras.Model(inputs=inputs, outputs=head_out, name=self.name)
        self.model.summary()

        self.model_config = self.model.to_json()
        self.model_config_hash = hashlib.md5(json.dumps(json.loads(self.model_config)).encode('utf-8')).hexdigest()
        self.path_network = os.path.join("saves", "networks", self.model_config_hash)
        os.makedirs(self.path_network, exist_ok=True)
        logging.info(f"Args:\n{pprint.pformat(self.__dict__, width=30)}")

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


class SimpleNetwork:
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
        self.action_space = np.arange(self.num_actions)
        self.optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)

        self.inp_s_shape = (None, self.max_timesteps, self.num_features)
        self.inp_s_layer = keras.layers.Input(batch_input_shape=self.inp_s_shape, name="inp_states_layer")

        # common state encoder
        body = keras.layers.Dense(64, activation="relu")(self.inp_s_layer)
        body = keras.layers.Dense(64, activation="relu")(body)
        body = keras.layers.Dense(64, activation="relu")(body)
        body = keras.layers.Dense(64, activation="relu")(body)

        # actor and critic head
        head = ActorCriticLayer(num_actions=self.num_actions)
        head_out = head(body)

        self.model = keras.Model(inputs=[self.inp_s_layer], outputs=head_out, name=self.name)
        self.model.summary()

        self.model_config = self.model.to_json()
        self.model_config_hash = hashlib.md5(json.dumps(json.loads(self.model_config)).encode('utf-8')).hexdigest()
        self.path_network = os.path.join("saves", "networks", self.model_config_hash)
        os.makedirs(self.path_network, exist_ok=True)
        logging.info(f"Args:\n{pprint.pformat(self.__dict__, width=30)}")

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



