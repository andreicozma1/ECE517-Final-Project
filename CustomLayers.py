from typing import Tuple

import tensorflow as tf
import tensorflow_addons as tfa

keras = tf.keras


class A2C(keras.layers.Layer):
    def __init__(self, num_actions):
        super(A2C, self).__init__()
        self.num_actions = num_actions
        self.actor = keras.layers.Dense(num_actions, name="actor")
        self.critic = keras.layers.Dense(1, name="critic")

    def get_config(self):
        config = super().get_config()
        config.update({
                "num_actions": self.num_actions,
                "actor"      : self.actor.get_config(),
                "critic"     : self.critic.get_config(),
        })
        return config

    def call(self, actor_inputs: tf.Tensor, critic_inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.actor(actor_inputs), self.critic(critic_inputs)


class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim, self.num_heads, self.ff_dim, self.dropout_rate = embed_dim, num_heads, ff_dim, dropout_rate
        self.att = tfa.layers.MultiHeadAttention(head_size=embed_dim, num_heads=num_heads)
        self.ffn = keras.Sequential([*[keras.layers.Dense(ffd, activation="relu") for ffd in ff_dim],
                                     keras.layers.Dense(embed_dim), ])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)

    def get_config(self):
        config = super().get_config()
        config.update({
                "embed_dim"   : self.embed_dim,
                "num_heads"   : self.num_heads,
                "ff_dim"      : self.ff_dim,
                "dropout_rate": self.dropout_rate,
                "att"         : self.att.get_config(),
                "ffn"         : self.ffn.get_config(),
                "layernorm1"  : self.layernorm1.get_config(),
                "layernorm2"  : self.layernorm2.get_config(),
                "dropout1"    : self.dropout1.get_config(),
                "dropout2"    : self.dropout2.get_config(),
        })
        return config

    def call(self, inputs, training, mask=None):
        attn_output = self.att(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
