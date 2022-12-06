import logging
import pprint
from typing import Tuple

import tensorflow as tf

keras = tf.keras


class ActorCriticLayer(keras.layers.Layer):
    def __init__(self, num_actions, name="A2C", critic_activation="linear", **kwargs):
        super(ActorCriticLayer, self).__init__(name=name, **kwargs)
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

    def __init__(self, num_inputs, num_timesteps, embed_dim):
        super(StateAndPositionEmbedding, self).__init__()
        """
        Returns positional encoding for a given sequence length and embedding dimension,
        as well as the padding mask for 0 values.
        """
        self.num_inputs = num_inputs
        self.num_timesteps = num_timesteps
        self.embed_dim = embed_dim
        self.position_embedding = keras.layers.Embedding(input_dim=self.num_timesteps,
                                                         output_dim=self.embed_dim)
        # self.state_embedding = keras.layers.Dense(self.embed_dim, activation="linear")
        # Create dense layers for each input
        self.state_embedding = [keras.layers.Dense(self.embed_dim, activation="linear") for _ in range(self.num_inputs)]

    def get_config(self):
        config = super().get_config()
        config.update({
                "num_timesteps"     : self.num_timesteps,
                "embed_dim"         : self.embed_dim,
                "position_embedding": self.position_embedding.get_config(),
                "state_embedding"   : self.state_embedding
        })
        return config

    def call(self, pos_arr, inputs):
        # embed each timestep
        pos_encoding = self.position_embedding(pos_arr)
        # embed each input
        input_encoding = [self.state_embedding[i](inputs[i]) + pos_encoding for i in range(self.num_inputs)]
        # sum the positional and input encoding
        return tf.math.add_n(input_encoding)


class TransformerActorCritic(keras.Model):
    def __init__(self, seq_len, num_features, num_actions,
                 critic_final_activation="linear",
                 name="transformer_actor_critic", **kwargs):
        super().__init__(name=name, **kwargs)
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_actions = num_actions
        self.critic_final_activation = critic_final_activation

        self.inp_pos_shape = (1, self.seq_len)
        self.inp_state_shape = (1, self.seq_len, self.num_features)
        self.inp_actions_shape = (1, self.seq_len, self.num_actions)

        self.inp_pos_layer = keras.layers.Input(batch_input_shape=self.inp_pos_shape, name="input_t")
        self.inp_states_layer = keras.layers.Input(batch_input_shape=self.inp_state_shape, name="input_states")
        self.inp_actions_layer = keras.layers.Input(batch_input_shape=self.inp_actions_shape, name="input_actions")

        self.actor_critic_layer = ActorCriticLayer(self.num_actions, critic_activation=self.critic_final_activation)

        self.state_embedding_layer = StateAndPositionEmbedding(self.num_features, self.seq_len, 64)
        self.transformer_encoder = TransformerEncoders(num_layers=2, embed_dim=64, num_heads=4, ff_dim=128)
        self.transformer_decoder = TransformerDecoders(num_layers=2, embed_dim=64, num_heads=4, ff_dim=128)
        self.dense = keras.layers.Dense(self.num_actions)

    def get_config(self):
        config = super().get_config()
        config.update({
                "seq_len"                : self.seq_len,
                "num_features"           : self.num_features,
                "num_actions"            : self.num_actions,
                "critic_final_activation": self.critic_final_activation,
                "input_positions_shape"  : self.inp_pos_shape,
                "input_state_shape"      : self.inp_state_shape,
                "input_actions_shape"    : self.inp_actions_shape,
                "ac_layer"               : self.actor_critic_layer.get_config(),
                "state_embedding"        : self.state_embedding_layer.get_config(),
                "transformer_encoder"    : self.transformer_encoder.get_config(),
                "transformer_decoder"    : self.transformer_decoder.get_config(),
                "dense"                  : self.dense.get_config(),
        })
        return config

    def action_value(self, states):
        # states is a tensor of shape [batch_size, sequence_length, num_features]
        # returns a tensor of shape [batch_size, sequence_length, num_actions]
        sequence_length = tf.shape(states)[1]
        positions = tf.range(start=0, limit=sequence_length, delta=1)
        positions = tf.expand_dims(positions, axis=0)
        actions = tf.zeros(shape=(1, sequence_length, self.num_actions))
        return self((positions, states, actions))

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y, sample_weight = data

        with tf.GradientTape() as tape:
            logits, value = self(x, training=True)
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, logits, sample_weight, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, logits, sample_weight)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        x, y, sample_weight = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=False):
        # inputs is a tuple of (positions, states, actions)
        positions, states, actions = inputs
        # embed the states and positions
        x = self.state_embedding_layer(positions, states)
        # encode the states
        x = self.transformer_encoder(x)
        # decode the actions
        x = self.transformer_decoder(x, actions)
        # predict the logits and value
        return self.actor_critic_layer(x)
