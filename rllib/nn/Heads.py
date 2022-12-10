import tensorflow as tf

keras = tf.keras


class ActorCriticHead(keras.layers.Layer):
    def __init__(self, num_actions, name="A2C", critic_activation="linear", **kwargs):
        super(ActorCriticHead, self).__init__(name=name, **kwargs)
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
