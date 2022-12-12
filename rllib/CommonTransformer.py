import torch
from torch import nn


class CommonTransformer(nn.Module):

    def __init__(self, n_states: int, n_actions: int,
                 out_features: int,
                 max_episode_len: int,
                 batch_size: int,
                 seq_len: int,
                 hidden_size: int = 64):
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

        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_size,
                                                              nhead=hidden_size // 8,
                                                              dim_feedforward=2048,
                                                              dropout=0.1,
                                                              activation="gelu",
                                                              batch_first=True,
                                                              norm_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_encoder, num_layers=2)

        self.conv = nn.Conv1d(self.seq_len * 2, 1, 1)
        # self.pool = nn.AvgPool1d(self.seq_len)

        self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, out_features),
        )

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)

    def forward(self, input_x, batched=False):
        positions, states, actions = input_x
        positions = positions + 1

        pad_mask = self.create_pad_mask(positions, pad_token=0)

        t_emb = self.emb_p(positions)
        s_emb = self.emb_s(states)
        a_emb = self.emb_a(actions)

        s_emb = s_emb + t_emb
        a_emb = a_emb + t_emb

        pad_mask = pad_mask.reshape(-1, self.seq_len, 1)
        s_emb = s_emb.reshape(-1, self.seq_len, self.hidden_size)
        a_emb = a_emb.reshape(-1, self.seq_len, self.hidden_size)

        trans_src = torch.stack([s_emb, a_emb], dim=-1).permute(0, 1, 3, 2)
        trans_src = trans_src.reshape(-1, 2 * self.seq_len, self.hidden_size).squeeze()
        pad_mask = torch.stack([pad_mask, pad_mask], dim=-1).permute(0, 1, 3, 2)
        pad_mask = pad_mask.reshape(-1, 2 * self.seq_len, 1).squeeze()

        trans_out = self.transformer(trans_src, src_key_padding_mask=pad_mask)

        conv_inp = trans_out
        conv_out = self.conv(conv_inp)

        fc_imp = conv_out.squeeze()
        fc_out = self.fc(fc_imp)

        if batched:
            out_shape_expected = torch.Size([self.batch_size, self.out_features])
        else:
            out_shape_expected = torch.Size([self.out_features])
        assert fc_out.shape == out_shape_expected, f"fc_out.shape: {fc_out.shape} != {out_shape_expected}"
        return fc_out
