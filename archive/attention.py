# Add attention layer to the deep learning network
class attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape)
        self.W = self.add_weight(
            name="kernel",
            shape=(input_shape[-1], 1),
            initializer="uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="kernel",
            shape=(input_shape[1],),
            initializer="uniform",
            trainable=True,
        )
        super(attention, self).build(input_shape)

    def call(self, x):
        # Alignment scores. Pass them through tanh function
        e = keras.backend.tanh(keras.backend.dot(x, self.W) + self.b)
        # Remove dimension of size 1
        e = keras.backend.squeeze(e, axis=-1)
        # Compute the weights
        alpha = keras.backend.softmax(e)
        # Reshape to tensorFlow format
        alpha = keras.backend.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = keras.backend.sum(context, axis=1)
        return context
