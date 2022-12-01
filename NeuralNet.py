import logging
import pprint

import tensorflow as tf
import tensorflow_addons as tfa

from CustomLayers import A2C, ActorLoss, TransformerBlock

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
        self.model, self.optimizer, self.critic_loss, self.actor_loss = self.create_model(name, **kwargs)
        logging.info(f"Args:\n{pprint.pformat(self.__dict__, width=30)}")

    def create_model(self, model_name: str, **kwargs):
        logging.info(f"Model: {model_name}")
        logging.info(f"kwargs: {pprint.pformat(kwargs)}")
        inputs = keras.layers.Input(batch_input_shape=self.input_shape, name="input")

        actor_inputs, critic_inputs = getattr(self, f"inner_{model_name}")(inputs, **kwargs)

        ac_layer = A2C(self.num_actions)(actor_inputs=actor_inputs, critic_inputs=critic_inputs)

        model = keras.Model(inputs=inputs, outputs=ac_layer, name=model_name)
        model.summary()

        # config = model.get_config()
        # config_hash = hashlib.md5(json.dumps(config).encode('utf-8')).hexdigest()
        # self._config.update({
        #         "model": {
        #                 "name"       : model_name,
        #                 "kwargs"     : kwargs,
        #                 "num_layers" : len(model.layers),
        #                 "num_params" : model.count_params(),
        #                 "config"     : config,
        #                 "config_hash": config_hash
        #         }
        # })

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
        critic_loss = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        # critic_loss = keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE, delta=0.5)
        actor_loss = ActorLoss()

        return model, optimizer, critic_loss, actor_loss

    def inner_lstm(self, inputs, **kwargs):
        common = keras.layers.Dense(256, activation="elu", kernel_initializer="he_uniform")(inputs)
        # common = keras.layers.Dense(128, activation="elu", kernel_initializer="he_uniform")(common)
        # common = keras.layers.Dense(128, activation="elu", kernel_initializer="he_uniform")(common)

        sequences, state_h, state_c = keras.layers.LSTM(128, return_sequences=True, return_state=True)(common)
        # sequences, state_h, state_c = keras.layers.LSTM(128, return_sequences=True, return_state=True)(sequences)
        # sequences, state_h, state_c = keras.layers.LSTM(128, return_sequences=True, return_state=True)(sequences)

        lstm = keras.layers.LSTM(128, return_sequences=False)(sequences)

        common = keras.layers.Dropout(0.5)(lstm)

        actor = keras.layers.Dense(128, activation="elu", kernel_initializer="he_uniform")(common)
        actor = keras.layers.Dense(128, activation="elu", kernel_initializer="he_uniform")(actor)
        actor = keras.layers.Dense(128, activation="elu", kernel_initializer="he_uniform")(actor)

        critic = keras.layers.Dense(128, activation="elu", kernel_initializer="he_uniform")(common)
        critic = keras.layers.Dense(128, activation="elu", kernel_initializer="he_uniform")(critic)
        critic = keras.layers.Dense(128, activation="elu", kernel_initializer="he_uniform")(critic)

        return actor, critic

    def inner_dense(self, inputs, **kwargs):
        common = keras.layers.Flatten()(inputs)

        common = keras.layers.Dense(64)(common)
        common = keras.layers.LeakyReLU(alpha=0.05)(common)
        common = keras.layers.Dense(128)(common)
        common = keras.layers.LeakyReLU(alpha=0.05)(common)
        common = keras.layers.Dense(256)(common)
        common = keras.layers.LeakyReLU(alpha=0.05)(common)
        common = keras.layers.Dense(512)(common)
        common = keras.layers.LeakyReLU(alpha=0.05)(common)
        common = keras.layers.Dense(256)(common)
        common = keras.layers.LeakyReLU(alpha=0.05)(common)
        common = keras.layers.Dense(128)(common)
        common = keras.layers.LeakyReLU(alpha=0.05)(common)
        common = keras.layers.Dense(64)(common)
        common = keras.layers.LeakyReLU(alpha=0.05)(common)
        # common = keras.layers.Dropout(0.1)(common)

        # common = keras.layers.LeakyReLU(alpha=0.1)(common)
        # actor = keras.layers.Dense(512)(common)
        # critic = keras.layers.Dense(512)(common)

        return common, common

    def inner_attention(self, inputs, **kwargs):
        common = keras.layers.LSTM(128, return_sequences=True, stateful=False)(inputs)
        common = keras.layers.Dropout(0.2)(common)

        # LSTM Actor
        lstm_actor = keras.layers.LSTM(128, return_sequences=False)(common)
        lstm_actor = keras.layers.Dropout(0.5)(lstm_actor)
        # LSTM Critic
        lstm_critic = keras.layers.LSTM(128, return_sequences=False)(common)
        lstm_critic = keras.layers.Dropout(0.5)(lstm_critic)

        # # Dense Actor
        dense_actor = keras.layers.Dense(128, activation="relu")(lstm_actor)
        dense_actor = keras.layers.Dropout(0.2)(dense_actor)

        # Dense Critic
        dense_critic = keras.layers.Dense(128, activation="relu")(lstm_critic)
        dense_critic = keras.layers.Dropout(0.2)(dense_critic)

        # Merge
        att_a = keras.layers.MultiHeadAttention(num_heads=6, key_dim=64, dropout=0.2)(dense_critic, dense_actor)
        att_a = keras.layers.Dropout(0.2)(att_a)

        att_c = keras.layers.MultiHeadAttention(num_heads=6, key_dim=64, dropout=0.2)(dense_critic, dense_critic)
        att_c = keras.layers.Dropout(0.2)(att_c)

        # Multiply the actor by the attention of the critic
        att_a = keras.layers.Multiply()([dense_actor, att_a])
        att_a = keras.layers.Dropout(0.2)(att_a)

        # Multiply the critic by the attention of the actor
        att_c = keras.layers.Multiply()([dense_critic, att_c])
        att_c = keras.layers.Dropout(0.2)(att_c)

        # Flatten
        actor = keras.layers.Flatten()(att_a)
        critic = keras.layers.Flatten()(att_c)

        return actor, critic
