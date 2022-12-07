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

        ###################################################################
        # OPTIMIZER
        # optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        # optimizer = keras.optimizers.Nadam(learning_rate=self.learning_rate)
        # optimizer = tfa.optimizers.AdamW(
        #         learning_rate=self.learning_rate, weight_decay=0.000005, amsgrad=True
        # )
        self.optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)

        ###################################################################
        # LOSS
        # critic_loss = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.critic_loss = keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
        self.actor_loss = ActorLoss(reduction=tf.keras.losses.Reduction.NONE)

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
                "critic_loss": self.critic_loss.__class__.__name__,
                "actor_loss" : self.actor_loss.__class__.__name__,
        }


if __name__ == "__main__":
    main()
