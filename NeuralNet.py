import hashlib
import json
import logging
import pprint

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from CustomLayers import A2C, ActorLoss, StateAndPositionEmbedding, TransformerEncoder, \
    TransformerEncoderBlock

keras = tf.keras


class NeuralNet:
    def __init__(self, name: str, num_states: int, num_actions: int,
                 max_timesteps: int = 1,
                 learning_rate: float = 0.0001,
                 **kwargs):
        self.name: str = name
        self.num_features: int = num_states
        self.num_actions: int = num_actions
        self.learning_rate: float = learning_rate
        self.max_timesteps: int = max_timesteps
        self.input_shape = (1, self.max_timesteps, self.num_features)
        self.kwargs = kwargs
        self.model, self.optimizer, self.critic_loss, self.actor_loss = self.create_model(name, **kwargs)
        logging.info(f"Args:\n{pprint.pformat(self.__dict__, width=30)}")

    @property
    def config(self):
        config = self.model.to_json()
        config_str = json.dumps(json.loads(config))
        config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()
        return {
                "name"       : self.name,
                "kwargs"     : self.kwargs,
                "num_layers" : len(self.model.layers),
                "num_params" : self.model.count_params(),
                "config"     : config,
                "config_hash": config_hash,
                "critic_loss": self.critic_loss.__class__.__name__,
                "actor_loss" : self.actor_loss.__class__.__name__,
        }

    def create_model(self, model_name: str, **kwargs):
        logging.info(f"Model: {model_name}")
        logging.info(f"kwargs: {pprint.pformat(kwargs)}")
        c_final_act = self.get_activation("c_final_act", "linear", kwargs)

        inputs = keras.layers.Input(batch_input_shape=self.input_shape, name="input")

        actor_inputs, critic_inputs = getattr(self, f"inner_{model_name}")(inputs, **kwargs)

        ac_layer = A2C(self.num_actions, critic_activation=c_final_act)
        ac_outputs = ac_layer(actor_inputs=actor_inputs, critic_inputs=critic_inputs)

        model = keras.Model(inputs=inputs, outputs=ac_outputs, name=model_name)
        model.summary()

        ###################################################################
        # OPTIMIZER
        # optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        # optimizer = keras.optimizers.Nadam(learning_rate=self.learning_rate)
        # optimizer = tfa.optimizers.AdamW(
        #         learning_rate=self.learning_rate, weight_decay=0.000005, amsgrad=True
        # )
        optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)

        ###################################################################
        # LOSS
        # critic_loss = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        critic_loss = keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
        actor_loss = ActorLoss(reduction=tf.keras.losses.Reduction.NONE)

        return model, optimizer, critic_loss, actor_loss

    def inner_transformer(self, inputs, **kwargs):
        comm_act, a_act, c_act = self.get_a2c_activations(kwargs)

        emb_dim = 64
        t_layers = 1
        t_num_heads = 10
        t_dropout = 0.1
        t_ff_dim = [256, 256, 256, 256, 256]

        emb = StateAndPositionEmbedding(input_dim=self.input_shape,
                                        embed_dim=emb_dim)
        emb_out, padding_mask = emb(inputs)

        transformer = TransformerEncoder(num_layers=t_layers, embed_dim=emb_dim,
                                         num_heads=t_num_heads,
                                         ff_dim=t_ff_dim, dropout=t_dropout)
        transformer_out = transformer(emb_out, training=True, mask=padding_mask)

        # flatten
        common = keras.layers.Flatten()(transformer_out)
        # common = keras.layers.GlobalAveragePooling1D()(transformer_out)
        common = keras.layers.Dense(256, activation=comm_act)(common)
        common = keras.layers.Dense(256, activation=comm_act)(common)
        common = keras.layers.Dense(256, activation=comm_act)(common)
        common = keras.layers.Dense(256, activation=comm_act)(common)
        common = keras.layers.Dense(256, activation=comm_act)(common)
        common = keras.layers.Dense(256, activation=comm_act)(common)

        actor_inp = keras.layers.Dense(64, activation=a_act)(common)
        actor_inp = keras.layers.Dense(64, activation=a_act)(actor_inp)

        critic_inp = keras.layers.Dense(64, activation=c_act)(common)
        critic_inp = keras.layers.Dense(64, activation=c_act)(critic_inp)

        return actor_inp, critic_inp

    def get_a2c_activations(self, kwargs):
        common_activation = self.get_activation("common_activation", "elu", kwargs)
        actor_activation = self.get_activation("actor_activation", "elu", kwargs)
        critic_activation = self.get_activation("critic_activation", "linear", kwargs)
        return common_activation, actor_activation, critic_activation

    def get_activation(self, key, default_val, kwargs):
        val = default_val
        if key in kwargs:
            val = kwargs[key]
        if val == "leaky_relu":
            val = keras.layers.LeakyReLU()
        return val


def main():
    # state shape (1, 10, 4)
    # set the first 7 timesteps to 0
    # set the last 3 timesteps to random values

    num_timesteps = 10
    num_features = 4
    state = np.zeros((1, num_timesteps, num_features))
    state[:, 7:, :] = np.random.rand(1, 3, num_features)

    print(state)

    emb_dim = 16

    input_embedding, padding_mask = StateAndPositionEmbedding(input_dim=(1, num_timesteps, num_features),
                                                              embed_dim=emb_dim)(state)

    print(input_embedding)
    print(padding_mask)

    # position_embedding = keras.layers.Embedding(input_dim=num_timesteps, output_dim=emb_dim)
    # state_embedding = keras.layers.Dense(emb_dim, activation="relu")
    #
    # embedded = position_embedding(np.arange(num_timesteps))
    # # embedded = tf.expand_dims(embedded, axis=0)
    # print(embedded.shape)
    # print(embedded)
    #
    # # embed the state
    # embedded_state = state_embedding(state)
    # print(embedded_state.shape)
    # print(embedded_state)
    # #
    # # # create a mask
    # masking = keras.layers.Masking(mask_value=0.0)
    # mask = masking(embedded_state)
    # print(mask.shape)
    # print(mask)
    #
    # mask_bool = tf.cast(mask, tf.bool)
    # print(mask_bool)


if __name__ == "__main__":
    main()
