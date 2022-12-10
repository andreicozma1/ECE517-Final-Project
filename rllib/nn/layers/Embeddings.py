from typing import Union
import tensorflow as tf
import keras_nlp

keras = tf.keras


class StateAndPositionEmbedding(keras.layers.Layer):

    def __init__(self, num_timesteps: int, embed_dim: Union[int, list], **kwargs):
        super(StateAndPositionEmbedding, self).__init__(name="StateAndPositionEmbedding", **kwargs)
        """
        Returns positional encoding for a given sequence length and embedding dimension,
        as well as the padding mask for 0 values.
        """
        self.num_timesteps = num_timesteps
        self.embed_dim = [embed_dim] if isinstance(embed_dim, int) else embed_dim
        self.emb_pos = [keras_nlp.layers.PositionEmbedding(sequence_length=self.num_timesteps) for _ in self.embed_dim]
        self.emb_inputs = [keras.layers.Dense(edim, activation="linear") for edim in self.embed_dim]

    def get_config(self):
        config = super().get_config()
        config.update({
                "num_timesteps": self.num_timesteps,
                "embed_dim"    : self.embed_dim,
                "emb_inputs"   : self.emb_inputs
        })
        return config

    def call(self, pos_arr, inputs):
        # embed each timestep
        # print("pos_encoding", pos_encoding)
        # embed each input

        encoding = []
        for i in range(len(self.embed_dim)):
            emb_i = self.emb_inputs[i](inputs[i])
            pos_i = self.emb_pos[i](emb_i)
            added = keras.layers.Add()([emb_i, pos_i])
            encoding.append(added)
        # print("encoding", encoding)
        # add the two embeddings
        return encoding
