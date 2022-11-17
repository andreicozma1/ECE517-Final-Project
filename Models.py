import logging
import pprint
import tensorflow as tf

import tensorflow_addons as tfa

keras = tf.keras


class Models:
    def __init__(self, num_states, num_actions, learning_rate):
        self.e_num_states = num_states
        self.e_num_actions = num_actions
        self.learning_rate = learning_rate
        logging.info(f"Models Args: {pprint.pformat(self.__dict__)}")

    def m_dense(self, **kwargs):
        logging.info("Model: m_dense")
        logging.info(f"kwargs: {pprint.pformat(kwargs)}")
        inputs = keras.layers.Input(shape=(1, self.e_num_states,), name="input")

        # Dense
        common = keras.layers.Dense(4096, name="l1")(inputs)
        common = keras.layers.LeakyReLU(alpha=0.05)(common)
        common = keras.layers.Dropout(0.1)(common)
        common = keras.layers.Dense(2048, name="l2")(common)
        common = keras.layers.LeakyReLU(alpha=0.05)(common)
        common = keras.layers.Dropout(0.1)(common)
        common = keras.layers.Dense(1024, name="l3")(common)
        common = keras.layers.LeakyReLU(alpha=0.05)(common)
        common = keras.layers.Dropout(0.1)(common)

        # common = keras.layers.LeakyReLU(alpha=0.1)(common)
        # actor = keras.layers.Dense(512)(common)
        # critic = keras.layers.Dense(512)(common)

        ac_layer = A2C(self.e_num_actions)(actor_inputs=common, critic_inputs=common)

        model = keras.Model(inputs=inputs, outputs=ac_layer, name="m_dense")
        # RMSProp
        optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        # optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        # optimizer = keras.optimizers.Nadam(learning_rate=0.001)
        # optimizer = tfa.optimizers.AdamW(
        #     learning_rate=0.01, weight_decay=0.3, amsgrad=True
        # )
        # loss = keras.losses.Huber()
        loss = keras.losses.MeanSquaredError()

        return model, optimizer, loss

    def m_rnn(self, **kwargs):
        logging.info("Model: m_rnn")
        logging.info(f"kwargs: {pprint.pformat(kwargs)}")
        inputs = keras.layers.Input(batch_input_shape=(1, None, self.e_num_states), name="input")

        rnn = keras.layers.SimpleRNN(512, return_sequences=True, stateful=True)(inputs)
        rnn = keras.layers.SimpleRNN(256, return_sequences=False, stateful=True)(rnn)

        dense = keras.layers.Dense(512, activation="relu")(rnn)
        dense = keras.layers.Dense(1024, activation="relu")(dense)
        dense = keras.layers.Dense(2048, activation="relu")(dense)
        dense = keras.layers.Dense(512, activation="relu")(dense)

        critic = keras.layers.Dense(512, activation="linear")(dense)
        critic = keras.layers.Dense(128, activation="linear")(critic)

        actor = keras.layers.Dense(512, activation="relu", name="abcv")(dense)
        # actor = keras.layers.Dropout(0.2)(actor)
        actor = keras.layers.Dense(128, activation="relu", name="dev")(actor)
        # actor = keras.layers.Dropout(0.2)(actor)

        # critic = keras.layers.LeakyReLU(alpha=0.1)(critic)

        ac_layer = A2C(self.e_num_actions)(actor_inputs=actor, critic_inputs=critic)

        model = keras.Model(inputs=inputs, outputs=ac_layer, name="m_rnn")
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        # optimizer = keras.optimizers.Nadam(learning_rate=0.001)
        # optimizer = tfa.optimizers.AdamW(
        #     learning_rate=0.01, weight_decay=0.3, amsgrad=True
        # )
        loss = keras.losses.Huber()
        # loss = keras.losses.MeanSquaredError()

        return model, optimizer, loss

    def m_lstm(self, **kwargs):
        logging.info("Model: m_lstm")
        logging.info(f"kwargs: {pprint.pformat(kwargs)}")
        inputs = keras.layers.Input(shape=(1, self.e_num_states,), name="input")

        # LSTM
        common = keras.layers.LSTM(256, return_sequences=True)(inputs)
        common = keras.layers.Dropout(0.2)(common)
        common = keras.layers.LSTM(256, return_sequences=False)(common)
        common = keras.layers.Dropout(0.2)(common)

        ac_layer = A2C(self.e_num_actions)(actor_inputs=common, critic_inputs=common)

        model = keras.Model(inputs=inputs, outputs=ac_layer, name="m_lstm")
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        # optimizer = keras.optimizers.Nadam(learning_rate=0.001)
        # optimizer = tfa.optimizers.AdamW(
        #     learning_rate=0.01, weight_decay=0.3, amsgrad=True
        # )
        huber_loss = keras.losses.Huber()
        loss = keras.losses.MeanSquaredError()

        return model, optimizer, huber_loss

    def m_attention(self, **kwargs):
        logging.info("Model: m_attention")
        logging.info(f"kwargs: {pprint.pformat(kwargs)}")
        # The current state of the environment
        inputs = keras.layers.Input(  # shape=(
                #     None,
                #     self.n_inputs,
                # ),
                batch_input_shape=(1, 30, self.e_num_states), )

        common = keras.layers.LSTM(128, return_sequences=True, stateful=False)(inputs)
        common = keras.layers.Dropout(0.2)(common)

        # LSTM Actor
        lstm_actor = keras.layers.LSTM(128, return_sequences=False)(common)
        lstm_actor = keras.layers.Dropout(0.5)(lstm_actor)
        # LSTM Critic
        lstm_critic = keras.layers.LSTM(128, return_sequences=False)(common)
        lstm_critic = keras.layers.Dropout(0.5)(lstm_critic)

        # # Dense Actor
        dense_actor = keras.layers.Dense(128, activation="relu")(inputs)
        dense_actor = keras.layers.Dropout(0.2)(dense_actor)

        # Dense Critic
        dense_critic = keras.layers.Dense(128, activation="relu")(inputs)
        dense_critic = keras.layers.Dropout(0.2)(dense_critic)

        # Merge
        att_a = keras.layers.MultiHeadAttention(num_heads=6, key_dim=64, dropout=0.2)(dense_critic, dense_actor)
        att_a = keras.layers.Dropout(0.2)(common)

        att_c = keras.layers.MultiHeadAttention(num_heads=6, key_dim=64, dropout=0.2)(dense_critic, dense_critic)
        att_c = keras.layers.Dropout(0.2)(common)

        # Multiply the actor by the attention of the critic
        att_a = keras.layers.Multiply()([dense_actor, att_a])
        att_a = keras.layers.Dropout(0.2)(att_a)

        # Multiply the critic by the attention of the actor
        att_c = keras.layers.Multiply()([dense_critic, att_c])
        att_c = keras.layers.Dropout(0.2)(att_c)

        # Flatten
        actor = keras.layers.Flatten()(att_a)
        critic = keras.layers.Flatten()(att_c)

        # Output
        ac_layer = A2C(self.e_num_actions)(actor_inputs=actor, critic_inputs=critic)

        model = keras.Model(inputs=inputs, outputs=ac_layer)
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        # optimizer = keras.optimizers.Nadam(learning_rate=0.001)
        # optimizer = tfa.optimizers.AdamW(
        #     learning_rate=0.01, weight_decay=0.3, amsgrad=True
        # )
        loss = keras.losses.Huber()

        return model, optimizer, loss


class A2C(keras.layers.Layer):
    def __init__(self, num_actions):
        super(A2C, self).__init__()
        self.num_actions = num_actions
        self.actor = keras.layers.Dense(num_actions, activation="softmax", name="actor")
        self.critic = keras.layers.Dense(1, activation="linear", name="critic")

    def get_config(self):
        config = super().get_config()
        config.update({
                "num_actions": self.num_actions,
                "actor"      : self.actor.get_config(),
                "critic"     : self.critic.get_config(),
        })
        return config

    def call(self, actor_inputs, critic_inputs):
        actor = self.actor(actor_inputs)
        critic = self.critic(critic_inputs)
        return [actor, critic]


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
