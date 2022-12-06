    def build_actor_critic_network(self):
        input = keras.layers.Input(
            batch_input_shape=(1, None, self.input_dims),
            name="input",
        )
        advantage = keras.layers.Input(
            batch_input_shape=(1, None, 1),
            name="advantage",
        )
        layers = [512, 256]
        x = keras.layers.Dense(layers[0], activation="relu", name="dense_1")(input)
        for i in range(1, len(layers)):
            x = keras.layers.Dense(layers[i], activation="relu", name=f"dense_{i+1}")(x)

        probs = keras.layers.Dense(self.n_actions, activation="softmax", name="probs")(
            x
        )
        values = keras.layers.Dense(1, activation="linear", name="values")(x)

        def custom_loss(y_true, y_pred):
            out = keras.backend.clip(y_pred, 1e-8, 1 - 1e-8)
            log_lik = y_true * keras.backend.log(out)

            return keras.backend.sum(-log_lik * advantage)

        actor = keras.models.Network(
            inputs=[input, advantage], outputs=[probs], name="actor"
        )
        actor_opt = keras.optimizers.Adam(lr=self.lr)
        actor.compile(optimizer=actor_opt, loss=custom_loss)

        critic = keras.models.Network(inputs=[input], outputs=[values], name="critic")
        critic_opt = keras.optimizers.Adam(lr=self.entropy_loss_multiplier)
        critic_loss = keras.losses.MeanSquaredError()
        critic.compile(optimizer=critic_opt, loss=critic_loss)

        policy = keras.models.Network(inputs=[input], outputs=[probs], name="policy")

        return actor, critic, policy