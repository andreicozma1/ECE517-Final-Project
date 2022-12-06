import logging
import pprint
from typing import Tuple

import tensorflow as tf

keras = tf.keras


class A2C(keras.layers.Layer):
    def __init__(self, num_actions, name="A2C", critic_activation="linear", **kwargs):
        super(A2C, self).__init__(name=name, **kwargs)
        self.num_actions = num_actions
        self.critic_activation = critic_activation
        self.actor = keras.layers.Dense(num_actions, name="actor")
        self.critic = keras.layers.Dense(1, activation=critic_activation, name="critic")

    def get_config(self):
        config = super().get_config()
        config.update({
                "num_actions"      : self.num_actions,
                "critic_activation": self.critic_activation,
                "actor"            : self.actor.get_config(),
                "critic"           : self.critic.get_config(),
        })
        return config

    def call(self, actor_inputs: tf.Tensor, critic_inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.actor(actor_inputs), self.critic(critic_inputs)


class ActorLoss(tf.keras.losses.Loss):
    def __init__(self, name='actor_loss', reduction=tf.keras.losses.Reduction.NONE, **kwargs):
        super().__init__(name=name, reduction=reduction, **kwargs)

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, action_probs: tf.Tensor, advantage: tf.Tensor):
        advantage_probs = tf.math.multiply(advantage, tf.math.log(action_probs))
        advantage_probs = -tf.math.reduce_sum(advantage_probs, axis=-1)
        return advantage_probs


class TransformerEncoderBlock(keras.layers.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 ff_dim,
                 ff_activation="leaky_relu",
                 dropout=0.1,
                 name="TransformerEncoderBlock",
                 **kwargs):
        super(TransformerEncoderBlock, self).__init__(name=name, **kwargs)
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.ff_dim, self.ff_activation = ff_dim, ff_activation
        self.dropout_rate = dropout

        self.att = tf.keras.layers.MultiHeadAttention(key_dim=embed_dim, num_heads=num_heads)

        if isinstance(ff_dim, int):
            ff_dim = [ff_dim]

        ffn_layers = []
        for dim in ff_dim:
            ff_activation = keras.layers.LeakyReLU() if ff_activation == "leaky_relu" else keras.layers.Activation(
                    ff_activation)
            ffn_layers.extend((keras.layers.Dense(dim, activation=ff_activation), keras.layers.Dropout(dropout)))

        ffn_layers.append(keras.layers.Dense(embed_dim))

        self.ffn = keras.Sequential(ffn_layers)

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(dropout)
        self.dropout2 = keras.layers.Dropout(dropout)

    def get_config(self):
        config = super().get_config()
        config.update({
                "embed_dim"    : self.embed_dim,
                "num_heads"    : self.num_heads,
                "ff_dim"       : self.ff_dim,
                "ff_activation": self.ff_activation,
                "dropout_rate" : self.dropout_rate,
                "att"          : self.att.get_config(),
                "ffn"          : self.ffn.get_config(),
                "layernorm1"   : self.layernorm1.get_config(),
                "layernorm2"   : self.layernorm2.get_config(),
                "dropout1"     : self.dropout1.get_config(),
                "dropout2"     : self.dropout2.get_config(),
        })
        return config

    def call(self, inputs, training, mask=None):
        # Multi-Head Attemtion
        attn_output = self.att(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        # Add & Norm
        out1 = self.layernorm1(inputs + attn_output)
        # Feed Forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # Add & Norm
        return self.layernorm2(out1 + ffn_output)


class TransformerEncoder(keras.layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, activation="leaky_relu", dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.enc_layers = [TransformerEncoderBlock(embed_dim, num_heads, ff_dim, activation, dropout) for _ in
                           range(num_layers)]

    def get_config(self):
        config = super().get_config()
        config.update({
                "num_layers": self.num_layers,
                "enc_layers": [layer.get_config() for layer in self.enc_layers],
        })
        return config

    def call(self, inputs, training, mask=None):
        x = inputs
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x


class StateAndPositionEmbedding(keras.layers.Layer):

    def __init__(self, input_dim, embed_dim):
        super(StateAndPositionEmbedding, self).__init__()
        """
        Returns positional encoding for a given sequence length and embedding dimension,
        as well as the padding mask for 0 values.
        """

        # inputs to the model are of shape (1, num_timesteps, num_features)
        self.num_timesteps = input_dim[1]
        self.num_features = input_dim[2]
        self.embed_dim = embed_dim

        self.position_embedding = keras.layers.Embedding(input_dim=self.num_timesteps,
                                                         output_dim=self.embed_dim)
        self.state_embedding = keras.layers.Dense(self.embed_dim, activation="linear")

    def get_config(self):
        config = super().get_config()
        config.update({
                "num_timesteps"     : self.num_timesteps,
                "num_features"      : self.num_features,
                "embed_dim"         : self.embed_dim,
                "position_embedding": self.position_embedding.get_config(),
                "state_embedding"   : self.state_embedding.get_config(),
        })
        return config

    def call(self, positions, inputs):
        # embed each timestep
        # pos_encoding = self.position_embedding(tf.range(start=0, limit=self.num_timesteps, delta=1))
        pos_encoding = self.position_embedding(positions)
        state_embedding = self.state_embedding(inputs)
        return state_embedding + pos_encoding
