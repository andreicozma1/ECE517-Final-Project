from typing import Tuple

import torch
from pl_bolts.models import VAE
from torch import nn
from torch import Tensor
from torch.nn import Transformer


class CommonTransformer(nn.Module):
    """Simple MLP network."""

    def __init__(self, n_states: int, n_actions: int,
                 out_features: int,
                 max_episode_len: int,
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

        # Using extra padding token (-1) to avoid confusion with 0 (which is a valid timestep)
        self.emb_p = nn.Embedding(max_episode_len + 1, hidden_size)
        self.emb_s = nn.Linear(n_states, hidden_size)
        self.emb_a = nn.Linear(n_actions, hidden_size)
        self.emb_r = nn.Linear(n_actions, hidden_size)

        # self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_size,
        #                                                       nhead=hidden_size // 8,
        #                                                       dim_feedforward=2048,
        #                                                       dropout=0.1,
        #                                                       activation="gelu",
        #                                                       batch_first=True,
        #                                                       norm_first=True)
        # self.transformer = nn.TransformerEncoder(self.transformer_encoder, num_layers=2)
        self.transformer: Transformer = nn.Transformer(d_model=hidden_size,
                                                       nhead=hidden_size // 8,
                                                       num_encoder_layers=4,
                                                       num_decoder_layers=4,
                                                       dim_feedforward=128,
                                                       dropout=0.1,
                                                       activation="gelu",
                                                       batch_first=True,
                                                       norm_first=True
                                                       )
        self.pool = nn.AvgPool1d(self.seq_len)
        self.conv = nn.Conv1d(self.seq_len * 2 - 1, 1, 1)
        # self.pool = nn.AvgPool1d(2 * self.seq_len)  # [5,128] -> [5,1]

        self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, out_features),
        )

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)

    def forward(self, input_x, training: bool):
        pos, states, actions = input_x
        pos = pos.int()
        # print("=" * 80)
        # print("-" * 80)
        # print("pos", pos)
        # print("pos", pos.shape)
        # print("-" * 80)
        # print("states", states)
        # print("states", states.shape)
        # print("-" * 80)
        # print("actions", actions)
        # print("actions", actions.shape)
        # pos_s = pos + 1
        # pos_a = torch.where(pos == -1, torch.zeros_like(pos), pos)
        # print("-" * 80)
        # print("pos_s", pos_s)
        # print("pos_s", pos_s.shape)
        # print("-" * 80)
        # print("pos_a", pos_a)
        # print("pos_a", pos_a.shape)

        pad_s = self.create_pad_mask(pos, pad_token=-1)
        # pad_a = self.create_pad_mask(pos_a, pad_token=0)
        # print("-" * 80)
        # print("pad_s", pad_s)
        # print("pad_s", pad_s.shape)
        # print("-" * 80)
        # print("pad_a", pad_a)
        # print("pad_a", pad_a.shape)

        t_emb_s = self.emb_p(pos_s)
        # t_emb_a = self.emb_p(pos_a)

        s_emb = self.emb_s(states)
        # a_emb = self.emb_a(actions)

        s_emb = s_emb + t_emb_s
        # a_emb = a_emb + t_emb_a

        s_emb = s_emb.reshape(-1, self.seq_len, self.hidden_size)
        # a_emb = a_emb.reshape(-1, self.seq_len, self.hidden_size)

        # print("s_emb", s_emb)
        # print("s_emb", s_emb.shape)
        # print("a_emb", a_emb)
        # print("a_emb", a_emb.shape)
        # trans_inp = torch.stack([a_emb, s_emb], dim=-1).permute(0, 1, 3, 2)
        trans_inp = s_emb
        # print("trans_inp", trans_inp.squeeze())
        # print("trans_inp", trans_inp.shape)
        # trans_inp = trans_inp.reshape(-1, 2 * self.seq_len, self.hidden_size)
        # print("trans_inp", trans_inp.squeeze())
        # print("trans_inp", trans_inp.shape)

        pad_mask = pad_s.reshape(-1, self.seq_len)
        # pad_mask = torch.stack([pad_a, pad_s], dim=-1)
        # pad_mask = pad_mask.reshape(-1, 2 * self.seq_len)
        # print("pad_mask", pad_mask)
        # print("pad_mask", pad_mask.shape)
        attn_mask = self.transformer.generate_square_subsequent_mask(self.seq_len - 1, device="cuda")
        # print("attn_mask", attn_mask)
        # print("attn_mask", attn_mask.shape)

        # create the transformer source and target sequences
        # the current sequences in trans_imp are (a0, s0, a1, s1, a2, s2, a3, s3, a4, s4, a5, s5)
        # we want to generate the target sequences (s0, a1, s1, a2, s2, a3, s3, a4, s4, a5, a6, s6)
        # so we need to shift the sequences by one timestep but keep the same shape
        # we do this by padding the sequences with -1 and then shifting them by one timestep
        trans_src = trans_inp[:, :-1, :].squeeze()
        trans_tgt = trans_inp[:, 1:, :].squeeze()
        trans_src_pad = pad_mask[:, :-1].squeeze()
        trans_tgt_pad = pad_mask[:, 1:].squeeze()
        # print("trans_src", trans_src)
        # print("trans_src", trans_src.shape)
        # print("trans_src_pad", trans_src_pad)
        # print("trans_src_pad", trans_src_pad.shape)
        # print("trans_tgt", trans_tgt)
        # print("trans_tgt", trans_tgt.shape)
        # print("trans_tgt_pad", trans_tgt_pad)
        # print("trans_tgt_pad", trans_tgt_pad.shape)
        trans_out = self.transformer(src=trans_src, tgt=trans_tgt, tgt_mask=attn_mask,
                                     src_key_padding_mask=trans_src_pad)

        # print("trans_out.shape", trans_out.shape)

        # trans_out shape is [batch_size, seq_len, hidden_size]
        # for example, it could be [1, 25, 128], [5, 25, 128], etc.
        # use the average pooling to get the final output
        # print(trans_out.shape)
        # print("trans_out", trans_out)

        # conv_inp = trans_out
        # # print("conv_inp.shape", conv_inp.shape)
        # conv_out = self.conv(conv_inp)
        # # print("conv_out.shape", conv_out.shape)

        # if trans_out.dim() == 2:
        #     trans_out = self.pool(trans_out.permute(1, 0)).permute(1, 0)
        # elif trans_out.dim() == 3:
        #     trans_out = self.pool(trans_out.permute(0, 2, 1)).permute(0, 2, 1)

        ###############################################################################
        # PPO1:
        # out: torch.Size([64]) <--- correct
        ###############################################################################
        # fc_imp = conv_out.squeeze()
        fc_imp = trans_out
        # print("fc_imp", fc_imp)

        fc_out = self.fc(fc_imp)

        # if training:
        #     out_shape_expected = torch.Size([self.batch_size, self.out_features])
        # else:
        #     out_shape_expected = torch.Size([self.out_features])
        # assert fc_out.shape == out_shape_expected, f"fc_out.shape: {fc_out.shape} != {out_shape_expected}"
        # print("fc_out.shape", fc_out.shape)
        # print("fc_out", fc_out)
        return fc_out

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
