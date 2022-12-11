from typing import Tuple

import torch
from pl_bolts.models import VAE
from torch import nn
from torch import Tensor


class CommonTransformer(nn.Module):
    """Simple MLP network."""

    def __init__(self, n_states: int, n_actions: int,
                 max_episode_len: int, out_features: int,
                 ctx_len: int, hidden_size: int = 128):
        """
        Args:
            n_states: observation shape of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        self.seq_len = ctx_len
        self.pos_emb = nn.Embedding(max_episode_len + 1, hidden_size)
        self.state_emb = nn.Linear(n_states, hidden_size)
        self.action_emb = nn.Linear(n_actions, hidden_size)

        # self.pos_emb = nn.Embedding(self.seq_len, 128)
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder, num_layers=2)
        self.pool = nn.AvgPool1d(self.seq_len)  # [5,128] -> [5,1]
        self.linear = nn.Linear(hidden_size, out_features)
        self.relu = nn.ReLU()
        # state and position embedding
        # the state embedding is float, the position embedding is int

    def forward(self, input_x):
        positions, states, actions = input_x
        positions = torch.squeeze(positions, dim=-1)
        # print("=" * 80)

        # padding_mask_pos = torch.where(positions == -1, torch.tensor(1), torch.tensor(0))
        # repeat the position padding mask to match the state shape
        # padding_mask_state = padding_mask_pos.unsqueeze(-1).repeat(1, 1, states.shape[-1])
        # print("padding_mask_pos", padding_mask_pos)
        # print("padding_mask_pos", padding_mask_pos.shape)
        # print("padding_mask_state", padding_mask_state)
        # print(positions)
        # print(positions.shape)
        # print(states.shape)
        state_emb = self.state_emb(states)
        action_emb = self.action_emb(actions)

        # TODO: add padding_idx to pos_emb call

        positions_no_neg = torch.where(positions == -1, torch.zeros_like(positions), positions)
        position_embedding = self.pos_emb(positions_no_neg)
        # print("-" * 80)

        # print(position_embedding.shape)
        # # print(state_emb.shape)
        batch_size = 1 if states.dim() == 2 else states.shape[0]
        state_emb = state_emb + position_embedding
        action_emb = action_emb + position_embedding
        stacked_inputs = torch.stack(
                (state_emb, action_emb), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2 * self.seq_len, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # print(padding_mask_pos.shape)  # [5]
        # print(padding_mask_pos.squeeze().shape)
        x = self.transformer_encoder(stacked_inputs)
        # print(x.shape)
        if x.dim() == 2:
            x = self.pool(x.permute(1, 0)).permute(1, 0)
        elif x.dim() == 3:
            x = self.pool(x.permute(0, 2, 1)).permute(0, 2, 1)
        # print(x.shape)

        x = self.linear(x)
        x = self.relu(x)
        # print(x.shape)
        x = x.squeeze()
        # print(x.shape)
        # [5, 1, 64]
        # [1, 64]
        # [64]
        return x

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
