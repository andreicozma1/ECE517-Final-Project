from typing import Tuple

import torch
from pl_bolts.models import VAE
from torch import nn
from torch import Tensor


class CommonTransformer(nn.Module):
    """Simple MLP network."""

    def __init__(self, input_shape: Tuple[int], out_features, ctx_len):
        """
        Args:
            input_shape: observation shape of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        self.seq_len = ctx_len
        self.state_emb = nn.Linear(input_shape[0], 128)
        self.pos_emb = nn.Embedding(self.seq_len, 128)
        max_timesteps
        # self.pos_emb = nn.Embedding(, 128)
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder, num_layers=2)
        self.pool = nn.AvgPool1d(self.seq_len)  # [5,128] -> [5,1]
        self.linear = nn.Linear(128, out_features)
        self.relu = nn.ReLU()
        # state and position embedding
        # the state embedding is float, the position embedding is int

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(self, input_x):
        positions, states, actions = input_x
        # positions = torch.squeeze(positions, dim=-1)
        # print("=" * 80)

        # print(positions.shape)
        # print(states.shape)
        state_emb = self.state_emb(states)
        # TODO: add padding_idx to pos_emb call
        # position_embedding = self.pos_emb(positions)
        # print("-" * 80)

        # print(position_embedding.shape)
        # print(state_emb.shape)
        # x = state_emb + position_embedding
        x = state_emb
        # print("-" * 80)

        # print(x.shape)
        x = self.transformer_encoder(x)
        # print(x.shape)
        if x.dim() == 2:
            x = self.pool(x.permute(1, 0)).permute(1, 0)
        elif x.dim() == 3:
            x = self.pool(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = self.linear(x)
        x = self.relu(x)
        # print(x.shape)
        return x
