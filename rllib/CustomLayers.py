import logging
import pprint
from typing import Tuple, Union

import keras_nlp
import tensorflow as tf

keras = tf.keras


class ActorCriticLayer(keras.layers.Layer):
    def __init__(self, num_actions, name="A2C", critic_activation="linear", **kwargs):
        super(ActorCriticLayer, self).__init__(name=name, **kwargs)
        self.num_actions = num_actions
        self.critic_activation = critic_activation
        self.actor_dense = [keras.layers.Dense(64, activation="elu") for _ in range(4)]
        self.crictic_dense = [keras.layers.Dense(64, activation="elu") for _ in range(4)]
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

    def call(self, common):
        actor_inputs = common
        for layer in self.actor_dense:
            actor_inputs = layer(actor_inputs)
        actor = self.actor(actor_inputs)

        critic_inputs = common
        for layer in self.crictic_dense:
            critic_inputs = layer(critic_inputs)
        critic = self.critic(critic_inputs)
        return actor, critic


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


class TransformerEncoders(keras.layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, activation="leaky_relu", dropout=0.1):
        super(TransformerEncoders, self).__init__()
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


class TransformerDecoderBlock(keras.layers.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 ff_dim,
                 ff_activation="leaky_relu",
                 dropout=0.1,
                 name="TransformerDecoderBlock",
                 **kwargs):
        super(TransformerDecoderBlock, self).__init__(name=name, **kwargs)
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.ff_dim, self.ff_activation = ff_dim, ff_activation
        self.dropout_rate = dropout

        self.att1 = tf.keras.layers.MultiHeadAttention(key_dim=embed_dim, num_heads=num_heads)
        self.att2 = tf.keras.layers.MultiHeadAttention(key_dim=embed_dim, num_heads=num_heads)

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
        self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(dropout)
        self.dropout2 = keras.layers.Dropout(dropout)
        self.dropout3 = keras.layers.Dropout(dropout)

    def get_config(self):
        config = super().get_config()
        config.update({
                "embed_dim"    : self.embed_dim,
                "num_heads"    : self.num_heads,
                "ff_dim"       : self.ff_dim,
                "ff_activation": self.ff_activation,
                "dropout_rate" : self.dropout_rate,
                "att1"         : self.att1.get_config(),
                "att2"         : self.att2.get_config(),
                "ffn"          : self.ffn.get_config(),
                "layernorm1"   : self.layernorm1.get_config(),
                "layernorm2"   : self.layernorm2.get_config(),
                "layernorm3"   : self.layernorm3.get_config(),
                "dropout1"     : self.dropout1.get_config(),
                "dropout2"     : self.dropout2.get_config(),
                "dropout3"     : self.dropout3.get_config(),
        })
        return config

    def call(self, inputs, encoder_outputs, training, look_ahead_mask=None, padding_mask=None):
        # Multi-Head Attention (self-attention)
        attn1 = self.att1(inputs, inputs, attention_mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(inputs + attn1)
        # Multi-Head Attention (encoder-decoder attention)
        attn2 = self.att2(out1, encoder_outputs, attention_mask=padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        # Feed Forward
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        # Add & Norm
        return self.layernorm3(out2 + ffn_output)


class TransformerDecoders(keras.layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, activation="leaky_relu", dropout=0.1):
        super(TransformerDecoders, self).__init__()
        self.num_layers = num_layers
        self.dec_layers = [TransformerDecoderBlock(embed_dim, num_heads, ff_dim, activation, dropout) for _ in
                           range(num_layers)]

    def get_config(self):
        config = super().get_config()
        config.update({
                "num_layers": self.num_layers,
                "dec_layers": [layer.get_config() for layer in self.dec_layers],
        })
        return config

    def call(self, inputs, encoder_outputs, training, look_ahead_mask=None, padding_mask=None):
        x = inputs
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, encoder_outputs, training, look_ahead_mask, padding_mask)
        return x


class StateAndPositionEmbedding(keras.layers.Layer):

    def __init__(self, num_timesteps: int, embed_dim: Union[int, list], **kwargs):
        super(StateAndPositionEmbedding, self).__init__(name="StateAndPositionEmbedding", **kwargs)
        """
        Returns positional encoding for a given sequence length and embedding dimension,
        as well as the padding mask for 0 values.
        """
        self.num_timesteps = num_timesteps
        self.embed_dim = [embed_dim] if isinstance(embed_dim, int) else embed_dim

        self.emb_pos = [keras_nlp.layers.PositionEmbedding(sequence_length=self.num_timesteps) for edim in
                        self.embed_dim]

        self.emb_inputs = [keras.layers.Dense(edim, activation="linear") for edim in self.embed_dim]

    def get_config(self):
        config = super().get_config()
        config.update({
                "num_timesteps": self.num_timesteps,
                "embed_dim"    : self.embed_dim,
                "emb_pos"      : self.emb_pos.get_config(),
                "emb_inputs"   : self.emb_inputs
        })
        return config

    def call(self, pos_arr, inputs):
        # embed each timestep
        # print("pos_encoding", pos_encoding)
        # embed each input

        encoding = []
        for i in range(len(self.embed_dim)):
            emb_i = self.emb_inputs[i](inputs[i])
            pos_i = self.emb_pos[i](emb_i)
            added = keras.layers.Add()([emb_i, pos_i])
            encoding.append(added)
        # print("encoding", encoding)
        # add the two embeddings
        return encoding


# class CustomInputs(keras.layers.Input):
#     def __init__(self, seq_len, num_features, num_actions):
#         super(CustomInputs, self).__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.num_actions = num_actions
#         self.inp_pos_shape = (1, self.seq_len)
#         self.inp_state_shape = (1, self.seq_len, self.num_features)
#         self.inp_actions_shape = (1, self.seq_len, self.num_actions)
#         # self.inp_pos_layer = keras.layers.Input(batch_input_shape=self.inp_pos_shape, name="inp_pos_layer")
#         # self.inp_states_layer = keras.layers.Input(batch_input_shape=self.inp_state_shape, name="inp_states_layer")
#         # self.inp_actions_layer = keras.layers.Input(batch_input_shape=self.inp_actions_shape, name="inp_actions_layer")
#
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#                 "seq_len"          : self.seq_len,
#                 "num_features"     : self.num_features,
#                 "num_actions"      : self.num_actions,
#                 "inp_pos_shape"    : self.inp_pos_shape,
#                 "inp_state_shape"  : self.inp_state_shape,
#                 "inp_actions_shape": self.inp_actions_shape,
#                 "inp_pos_layer"    : self.inp_pos_layer.get_config(),
#                 "inp_states_layer" : self.inp_states_layer.get_config(),
#                 "inp_actions_layer": self.inp_actions_layer.get_config()
#         })
#         return config


class TransformerActorCritic(keras.layers.Layer):
    def __init__(self, seq_len, name="transformer_actor_critic", **kwargs):
        super().__init__(name=name, **kwargs)
        self.seq_len = seq_len

        emb_dim = [32, 32]
        t_layers = 1
        t_num_heads = 5
        t_dropout = 0.1
        t_ff_dim = [256, 512, 256]

        self.embedding_layer = StateAndPositionEmbedding(self.seq_len, emb_dim)
        self.masking_layer = keras.layers.Masking(mask_value=0.0)
        self.norm_layer = keras.layers.LayerNormalization(epsilon=1e-6)

        self.transformer_encoder = TransformerEncoders(num_layers=t_layers,
                                                       embed_dim=sum(emb_dim),
                                                       num_heads=t_num_heads,
                                                       ff_dim=t_ff_dim,
                                                       dropout=t_dropout)

        self.pooling_layer = keras.layers.GlobalAveragePooling1D()
        self.common_layers = [keras.layers.Dense(64, activation="elu") for _ in range(2)]

    def get_config(self):
        config = super().get_config()
        config.update({
                "seq_len": self.seq_len
        })
        return config

    def call(self, inputs):
        inp_p, inp_s, inp_a = inputs[0], inputs[1], inputs[2]
        # print("inp_pos_layer", inp_pos_layer)
        # print("inp_states_layer", inp_states_layer)
        # print("inp_actions_layer", inp_actions_layer)
        mask_att = self.masking_layer.compute_mask(inp_s)
        mask_att = tf.cast(mask_att, tf.bool)

        # print("mask_att", mask_att)
        # print("attention_mask", attention_mask)
        emb_state, emb_actions = self.embedding_layer(inp_p, [inp_s, inp_a])
        # print("emb_state", emb_state)
        # print("emb_actions", emb_actions)

        x = tf.concat([emb_state, emb_actions], axis=-1)
        layer_norm_out = self.norm_layer(x)
        transformer_out = self.transformer_encoder(layer_norm_out, training=True, mask=mask_att)

        output = self.pooling_layer(transformer_out)
        for layer in self.common_layers:
            output = layer(output)
        return output
