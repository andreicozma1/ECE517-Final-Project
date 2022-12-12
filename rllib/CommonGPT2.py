from typing import Tuple

import torch
from torch import nn
from torch import Tensor
import transformers
from rllib.GPT2 import GPT2Model

class CommonGPT2(nn.Module):
    """Simple MLP network."""

    def __init__(self, n_states: int, n_actions: int,
                 out_features: int,
                 max_episode_len: int,
                 batch_size: int,
                 seq_len: int,
                 hidden_size: int = 64,
                 **kwargs):
        """
        Args:
            n_states: observation shape of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        print(kwargs)
        self.n_states = n_states
        self.n_actions = n_actions
        self.out_features = out_features
        self.batch_size = batch_size
        self.max_length = seq_len
        self.hidden_size = hidden_size

        self.n_layer = kwargs.get("n_layer", 2)
        self.n_head = kwargs.get("n_head", 8)
        self.activation_function = kwargs.get("activation_function", "gelu")
        self.dropout = kwargs.get("dropout", 0.5)
        self.n_positions = kwargs.get("n_positions", 1024)
        # Using extra padding token (-1) to avoid confusion with 0 (which is a valid timestep)
        self.emb_p = nn.Embedding(max_episode_len, hidden_size)
        self.emb_s = nn.Linear(n_states, hidden_size)
        self.emb_a = nn.Linear(n_actions, hidden_size)
        self.emb_r = nn.Linear(1, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_inner=4*hidden_size,
            activation_function=self.activation_function,
            n_positions=self.n_positions,
            resid_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            # **kwargs
        )
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)


    def setup_shapes(self, positions, states, actions, batched=False, attention_mask=None):
        positions = positions.int()
        if not batched:
            positions = positions.reshape(1, -1)
            states = states.reshape(1, -1, self.n_states)
            actions = actions.reshape(1, -1, self.n_actions)
        pad_mask = (positions == -1).bool()

        batch_size, seq_length = states.shape[0], states.shape[1]
        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            positions = positions[:, -self.max_length:]

            # pad all tokens to sequence length
            if attention_mask is None:
                attention_mask = torch.ones((1,self.max_length)).to(dtype=torch.long, device=states.device)
                attention_mask[pad_mask] = 0

            states[pad_mask] = 0
            actions[pad_mask] = 0
            positions[pad_mask] = 0

            seq_length = self.max_length
        else:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        states.to(dtype=torch.float32)
        actions.to(dtype=torch.float32)
        positions.to(dtype=torch.long)
        return batch_size, seq_length, positions, states, actions, attention_mask

    def forward(self, input_x, batched=False, attention_mask=None):
        positions, states, actions = input_x

        (batch_size,
         seq_length,
         positions,
         states,
         actions,
         attention_mask) = self.setup_shapes(positions, states, actions, batched=batched, attention_mask=attention_mask)

        # embed each modality with a different head
        state_embeddings = self.emb_s(states)
        action_embeddings = self.emb_a(actions)
        time_embeddings = self.emb_p(positions)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings

        stacked_inputs = torch.stack(
                (state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2 * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)


        stacked_attention_mask = torch.stack(
                (attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 2 * seq_length).bool()

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # states (0), or actions (1); i.e. x[:,0,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)

        # state x[:,0] is the output of the transformer
        # state and action x[:,1] is the output of the transformer
        state_pred = x[:,0]  # only state
        state_action_pred = x[:,1]  # state and action


        # slice last prediction
        state_pred = state_pred[:, -1]
        state_action_pred = state_action_pred[:, -1]
        return state_pred, state_action_pred, attention_mask
