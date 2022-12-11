from typing import Tuple

import torch
from pl_bolts.models import VAE
from torch import nn
from torch import Tensor


class CommonTransformer(nn.Module):
    """Simple MLP network."""

    def __init__(self, n_states: int, n_actions: int,
                 max_episode_len: int,
                 out_features: int,
                 batch_size: int,
                 seq_len: int,
                 hidden_size: int = 128):
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
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        self.emb_pos = nn.Embedding(max_episode_len + 1, hidden_size)
        self.emb_states = nn.Linear(n_states, hidden_size)
        self.emb_action = nn.Linear(n_actions, hidden_size)

        # self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_size,
        #                                                       nhead=hidden_size // 8,
        #                                                       dim_feedforward=256,
        #                                                       dropout=0.1,
        #                                                       activation="gelu",
        #                                                       batch_first=True)
        # self.transformer = nn.TransformerEncoder(self.transformer_encoder, num_layers=1)
        self.transformer = nn.Transformer(d_model=hidden_size,
                                          nhead=hidden_size // 8,
                                          num_encoder_layers=2,
                                          num_decoder_layers=2,
                                          dropout=0.1,
                                          activation="gelu",
                                          batch_first=True,
                                          norm_first=False
                                          )
        self.pool = nn.AvgPool1d(self.seq_len)
        self.conv = nn.Conv1d(self.seq_len, 1, 1)
        # self.pool = nn.AvgPool1d(2 * self.seq_len)  # [5,128] -> [5,1]

        self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, out_features),
        )

    def forward(self, input_x, batched=False):
        positions, states, actions = input_x
        # print("=" * 80)

        # positions = torch.squeeze(positions, dim=-1)
        padding_mask_pos = torch.where(positions == -1, 1, 0).bool()
        # print("-" * 80)
        # print(padding_mask_pos)
        # print(padding_mask_pos.shape)
        # print("-" * 80)
        # print(positions)
        # print(positions.shape)
        positions = torch.where(positions == -1, torch.zeros_like(positions), positions)
        # print(positions)
        # print(positions.shape)
        # print("-" * 80)
        # print(states)
        # print(states.shape)
        # print("-" * 80)
        # print(actions)
        # print(actions.shape)

        positions_emb = self.emb_pos(positions)
        states_emb = self.emb_states(states)
        actions_emb = self.emb_action(actions)

        # batch_size = 1 if states.dim() == 2 else states.shape[0]
        states_emb = states_emb + positions_emb
        actions_emb = actions_emb + positions_emb

        if not batched:
            states_emb = states_emb.unsqueeze(0)
            actions_emb = actions_emb.unsqueeze(0)

        # torch.Size([1, 2, 25, 128])
        # torch.Size([1, 25, 2, 128])

        # stacked_inputs = torch.stack(
        #         (state_emb, action_emb), dim=1
        # ).permute(0, 2, 1, 3).reshape(batch_size, 2 * self.seq_len, self.hidden_size)

        # if padding_mask_pos.dim() == 1:
        #     padding_mask_pos = padding_mask_pos.unsqueeze(0)

        # stacked_attention_mask = torch.stack(
        #         (padding_mask_pos, padding_mask_pos), dim=1
        # ).permute(0, 2, 1).reshape(batch_size, 2 * self.seq_len).bool()

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

        trans_src = states_emb
        # the source should be one step before the target
        trans_out = self.transformer(trans_src, trans_src)
        # print("trans_out.shape", trans_out.shape)

        # trans_out shape is [batch_size, seq_len, hidden_size]
        # for example, it could be [1, 25, 128], [5, 25, 128], etc.
        # use the average pooling to get the final output
        # print(trans_out.shape)

        conv_inp = trans_out
        # print("conv_inp.shape", conv_inp.shape)
        conv_out = self.conv(conv_inp)
        # print("conv_out.shape", conv_out.shape)

        # if trans_out.dim() == 2:
        #     trans_out = self.pool(trans_out.permute(1, 0)).permute(1, 0)
        # elif trans_out.dim() == 3:
        #     trans_out = self.pool(trans_out.permute(0, 2, 1)).permute(0, 2, 1)

        ###############################################################################
        # PPO1:
        # out: torch.Size([64]) <--- correct
        ###############################################################################
        fc_imp = conv_out.squeeze()
        fc_out = self.fc(fc_imp)

        if batched:
            out_shape_expected = torch.Size([self.batch_size, self.out_features])
        else:
            out_shape_expected = torch.Size([self.out_features])
        assert fc_out.shape == out_shape_expected, f"fc_out.shape: {fc_out.shape} != {out_shape_expected}"
        return fc_out

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
