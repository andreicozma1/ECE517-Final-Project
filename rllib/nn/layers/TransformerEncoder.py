import tensorflow as tf

keras = tf.keras


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
