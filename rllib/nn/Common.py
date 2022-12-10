import tensorflow as tf

from rllib.nn.layers.Embeddings import StateAndPositionEmbedding
from rllib.nn.layers.TransformerEncoder import TransformerEncoders

keras = tf.keras


class CommonTransformer(keras.layers.Layer):
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

        output = tf.concat([emb_state, emb_actions], axis=-1)
        output = self.norm_layer(output)
        output = self.transformer_encoder(output, training=True, mask=mask_att)
        output = self.pooling_layer(output)
        for layer in self.common_layers:
            output = layer(output)
        return output
