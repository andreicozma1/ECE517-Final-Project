from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import transformers
from transformers import TFGPT2Model, GPT2Config
from tf_agents.environments import random_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.networks import encoding_network
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.specs import array_spec
from tf_agents.utils import common as common_utils
from tf_agents.utils import nest_utils
import numpy as np
import tensorflow as tf

# from trajectory_gpt2 import GPT2Model

"""
https://arxiv.org/pdf/2106.01345.pdf
https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py 
"""
class DecisionTransformer(tf.keras.layers.Layer):

    def __init__(self,
                 state_dim,
                 act_dim,
                 hidden_size,
                 max_length=None,
                 max_ep_len=4096,
                 action_tanh=True,
                 **kwargs):
        super(DecisionTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

        config = GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )
        self.transformer = TFGPT2Model(config)

        self.embed_timestep = tf.keras.layers.Embedding(max_ep_len, hidden_size)
        self.embed_return = tf.keras.layers.Dense(self.hidden_size, input_shape=(None, 1))
        self.embed_state = tf.keras.layers.Dense(self.hidden_size, input_shape=(None, self.state_dim))
        self.embed_action = tf.keras.layers.Dense(self.hidden_size, input_shape=(None, self.act_dim))

        # check params
        self.embed_ln = tf.keras.layers.LayerNormalization(axis=-1, scale=True, center=True)

        # note: we don't predict states or returns for the paper
        self.predict_state = tf.keras.layers.Dense(self.hidden_size, input_shape=(None, self.state_dim))
        self.predict_action = tf.keras.Sequential(
            *([tf.keras.layers.Dense(self.act_dim)] + ([tf.keras.layers.Activation('tanh')] if action_tanh else []))
        )
        self.predict_return = tf.keras.layers.Dense(1, input_shape=(None, self.hidden_size))

    # def forward(self, states, actions, rewards, masks=None, attention_mask=None):
    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = tf.ones((batch_size, seq_length), dtype=tf.int64)

        # embed each modaliry with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        return_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        return_embeddings = return_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = tf.stack([return_embeddings, state_embeddings, action_embeddings], axis=1)
        stacked_inputs = tf.reshape(stacked_inputs, (batch_size, seq_length * 3, self.hidden_size))
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = tf.stack([attention_mask, attention_mask, attention_mask], axis=1)
        stacked_attention_mask = tf.reshape(stacked_attention_mask, (batch_size, seq_length * 3))

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            input_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape((batch_size, seq_length, 3, self.hidden_size))

        # get predictions
        return_preds = self.predict_return(x[:, 2])     # predict next return given state and action
        state_preds = self.predict_state(x[:, 2])       # predict next state given state and action
        action_preds = self.predict_action(x[:, 1])     # predict next action given state

        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):

        states = states.reshape((1, -1, self.state_dim))
        actions = actions.reshape((1, -1, self.act_dim))
        returns_to_go = returns_to_go.reshape((1, -1, 1))
        timesteps = timesteps.reshape((1, -1))

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            # pad all tokens to sequence length
            attention_mask = tf.concat([
                tf.zeros(self.max_length - states.shape[1], dtype=tf.int64),
                tf.ones(states.shape[1], dtype=tf.int64)], axis=0)
            attention_mask = attention_mask.reshape((1, -1))

            states = tf.concat([
                tf.zeros((states.shape[0], self.max_length - states.shape[1], self.state_dim), dtype=tf.float32),
                states], axis=1)
            actions = tf.concat([
                tf.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim), dtype=tf.float32),
                actions], axis=1)
            returns_to_go = tf.concat([
                tf.zeros((returns_to_go.shape[0], self.max_length - returns_to_go.shape[1], 1), dtype=tf.float32),
                returns_to_go], axis=1)
            timesteps = tf.concat([
                tf.zeros((timesteps.shape[0], self.max_length - timesteps.shape[1]), dtype=tf.int64),
                timesteps], axis=1)

            attention_mask = tf.cast(attention_mask, dtype=tf.int64)
            states = tf.cast(states, dtype=tf.float32)
            actions = tf.cast(actions, dtype=tf.float32)
            returns_to_go = tf.cast(returns_to_go, dtype=tf.float32)
            timesteps = tf.cast(timesteps, dtype=tf.int64)

        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]
