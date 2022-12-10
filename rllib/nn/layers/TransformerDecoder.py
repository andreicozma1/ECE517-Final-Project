import tensorflow as tf

keras = tf.keras


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
