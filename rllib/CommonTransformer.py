from typing import Tuple

import torch
from pl_bolts.models import VAE
from torch import nn
from torch import Tensor


class CommonTransformer(nn.Module):
    """Simple MLP network."""

    def __init__(self, n_states: int, n_actions: int,
                 max_episode_len: int, out_features: int,
                 seq_len: int, hidden_size: int = 128):
        """
        Args:
            n_states: observation shape of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.out_features = out_features
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        self.emb_pos = nn.Embedding(max_episode_len + 1, hidden_size)
        self.emb_states = nn.Linear(n_states, hidden_size)
        self.emb_action = nn.Linear(n_actions, hidden_size)

        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder, num_layers=2)
        self.pool = nn.AvgPool1d(self.seq_len)  # [5,128] -> [5,1]
        # self.pool = nn.AvgPool1d(2 * self.seq_len)  # [5,128] -> [5,1]

        self.out = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, out_features),
        )

    def forward(self, input_x):
        positions, states, actions = input_x
        # print("positions:", positions)
        # print("states:", states)
        # print("actions:", actions)
        positions = torch.squeeze(positions, dim=-1)

        # print("=" * 80)
        # print(positions)
        # print(positions.shape)
        # print(states)
        # print(states.shape)
        # print(actions)
        # print(actions.shape)

        padding_mask_pos = torch.where(positions == -1, 1, 0).bool()  # shape: [25]
        positions = torch.where(positions == -1, torch.zeros_like(positions), positions)
        position_embedding = self.emb_pos(positions)

        state_emb = self.emb_states(states)
        action_emb = self.emb_action(actions)

        # TODO: add padding_idx to pos_emb call

        # print("-" * 80)
        # print(position_embedding)
        # print(position_embedding.shape)
        # print(state_emb)
        # print(state_emb.shape)
        # print(action_emb)
        # print(action_emb.shape)

        # print(position_embedding.shape)
        # # print(state_emb.shape)
        batch_size = 1 if states.dim() == 2 else states.shape[0]
        state_emb = state_emb + position_embedding
        action_emb = action_emb + position_embedding
        if state_emb.dim() == 2:
            state_emb = state_emb.unsqueeze(0)
            action_emb = action_emb.unsqueeze(0)
        # print(state_emb.shape)
        # print(action_emb.shape)

        # torch.Size([1, 2, 25, 128])
        # torch.Size([1, 25, 2, 128])
        # print(torch.stack(
        #         (state_emb, action_emb), dim=1
        # ).shape)
        # print(torch.stack(
        #         (state_emb, action_emb), dim=1
        # ).permute(0, 2, 1, 3).shape)

        # stacked_inputs = torch.stack(
        #         (state_emb, action_emb), dim=1
        # ).permute(0, 2, 1, 3).reshape(batch_size, 2 * self.seq_len, self.hidden_size)

        # print(stacked_inputs.shape)
        # print('-------------------')
        # print(padding_mask_pos)
        # print(padding_mask_pos.shape)
        # print(padding_mask_pos.dim())
        if padding_mask_pos.dim() == 1:
            padding_mask_pos = padding_mask_pos.unsqueeze(0)
        # print(padding_mask_pos.shape)

        # stacked_attention_mask = torch.stack(
        #         (padding_mask_pos, padding_mask_pos), dim=1
        # ).permute(0, 2, 1).reshape(batch_size, 2 * self.seq_len).bool()

        # print(padding_mask_pos.shape)  # [5]
        # print(padding_mask_pos.squeeze().shape)

        # def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
        #     to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
        #     Binary and byte masks are supported.
        #     For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
        #     the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
        # need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
        #     Default: ``True``.
        # attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
        #     :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
        #     :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
        #     broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
        #     Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
        #     corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
        #     corresponding position is not allowed to attend. For a float mask, the mask values will be added to
        #     the attention weight.

        trans_imp = state_emb

        trans_out = self.transformer_encoder(trans_imp)
        # x = self.transformer_encoder(stacked_inputs, src_key_padding_mask=stacked_attention_mask)

        # print('-------------------')
        # print(x)
        # print(x.shape)
        # x = x[1]
        # print(x.shape)
        if trans_out.dim() == 2:
            trans_out = self.pool(trans_out.permute(1, 0)).permute(1, 0)
        elif trans_out.dim() == 3:
            trans_out = self.pool(trans_out.permute(0, 2, 1)).permute(0, 2, 1)

        ###############################################################################
        # PPO1:
        # out: torch.Size([64]) <--- correct
        ###############################################################################
        out_imp = trans_out.squeeze()
        out = self.out(out_imp)
        return out

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
