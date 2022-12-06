    def model_attention(self):
        inputs = keras.layers.Input(shape=(self.n_inputs,))

        # Dense
        common = keras.layers.Dense(1024, activation="relu")(inputs)
        common = keras.layers.Dropout(0.2)(common)

        # MultiHeadAttention
        common = keras.layers.Reshape((1, 1024))(common)
        att = keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)(
            common, common, return_attention_scores=True
        )
        common = keras.layers.Dropout(0.2)(att[0])
        common = keras.layers.Flatten()(common)

        # Use attention scores as input to Dense layer
        att_scores = keras.layers.Flatten()(att[1])
        common = keras.layers.Concatenate()([common, att_scores])

        # Dense
        common = keras.layers.Dense(1024, activation="relu")(common)
        common = keras.layers.Dropout(0.2)(common)

        action = keras.layers.Dense(self.n_actions, activation="softmax")(common)
        critic = keras.layers.Dense(1)(common)

        model = keras.Network(inputs=inputs, outputs=[action, critic])
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        # optimizer = keras.optimizers.Nadam(learning_rate=0.001)
        # optimizer = tfa.optimizers.AdamW(
        #     learning_rate=0.01, weight_decay=0.3, amsgrad=True
        # )
        huber_loss = keras.losses.Huber()

        return model, optimizer, huber_loss

    def model_transformer(self):
        inputs = keras.layers.Input(shape=(self.n_inputs,))

        # Transformer
        t_embed_dim = 64  # Embedding size for each token
        t_num_heads = 6  # Number of attention heads
        t_ff_dim = [
            32,
            32,
        ]  # Hidden layer size in feed forward network inside transformer
        # Embedding for transformer block
        common = keras.layers.Reshape((1, self.n_inputs))(inputs)
        common = keras.layers.Dense(t_embed_dim, activation="relu")(common)
        common = keras.layers.Dropout(0.5)(common)
        # Now transformer block
        common = TransformerBlock(t_embed_dim, t_num_heads, t_ff_dim)(common)
        common = keras.layers.Dropout(0.5)(common)
        # Now dense layers
        common = keras.layers.Dense(16, activation="relu")(common)
        common = keras.layers.Dropout(0.5)(common)

        common = keras.layers.Flatten()(common)

        action = keras.layers.Dense(self.n_actions, activation="softmax")(common)
        critic = keras.layers.Dense(1)(common)

        model = keras.Network(inputs=inputs, outputs=[action, critic])
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        # optimizer = keras.optimizers.Nadam(learning_rate=0.001)
        # optimizer = tfa.optimizers.AdamW(
        #     learning_rate=0.01, weight_decay=0.3, amsgrad=True
        # )
        huber_loss = keras.losses.Huber()

        return model, optimizer, huber_loss