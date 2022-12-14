import torch
from torch import nn


class CommonTransformer(nn.Module):
    """Simple MLP network."""

    def __init__(self, n_states: int, n_actions: int,
                 out_features: int,
                 max_episode_len: int,
                 batch_size: int,
                 seq_len: int,
                 hidden_size: int = 64):
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

        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_size,
                                                              nhead=hidden_size // 8,
                                                              dim_feedforward=2048,
                                                              dropout=0.1,
                                                              activation="gelu",
                                                              batch_first=True,
                                                              norm_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_encoder, num_layers=2)
        # self.transformer: Transformer = nn.Transformer(d_model=hidden_size,
        #                                                nhead=hidden_size // 8,
        #                                                num_encoder_layers=4,
        #                                                num_decoder_layers=4,
        #                                                dim_feedforward=128,
        #                                                dropout=0.1,
        #                                                activation="gelu",
        #                                                batch_first=True,
        #                                                norm_first=True
        #                                                )

        # TODO: I got Comv1d to sorta work but not sure if it's proper way to do it.
        self.conv = nn.Conv1d(self.seq_len * 2, 1, 1)
        self.pool = nn.AvgPool1d(self.seq_len)
        # self.pool = nn.AvgPool1d(2 * self.seq_len)  # [5,128] -> [5,1]

        self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, out_features),
        )

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)

    def forward(self, input_x, batched=False):
        positions, states, actions = input_x
        positions = positions + 1

        pad_mask = self.create_pad_mask(positions, pad_token=0)
        # print("pad_mask", pad_mask.shape)
        # print("-" * 80)
        # print(padding_mask_pos)
        # print(padding_mask_pos.shape)
        # print("-" * 80)
        # print(positions)
        # print(positions.shape)
        # positions = torch.where(positions == -1, torch.zeros_like(positions), positions)
        # print(positions)
        # print(positions.shape)
        # print("-" * 80)
        # print(states)
        # print(states.shape)
        # print("-" * 80)
        # print(actions)
        # print(actions.shape)

        t_emb = self.emb_p(positions)
        s_emb = self.emb_s(states)
        a_emb = self.emb_a(actions)

        # print("t_emb", t_emb)
        # print("t_emb", t_emb.shape)
        # print("s_emb", s_emb)
        # print("s_emb", s_emb.shape)

        # batch_size = 1 if states.dim() == 2 else states.shape[0]
        s_emb = s_emb + t_emb
        a_emb = a_emb + t_emb

        pad_mask = pad_mask.reshape(-1, self.seq_len, 1)
        s_emb = s_emb.reshape(-1, self.seq_len, self.hidden_size)
        a_emb = a_emb.reshape(-1, self.seq_len, self.hidden_size)
        # if not batched:
        #     pad_mask = pad_mask.unsqueeze(0)
        #     s_emb = s_emb.unsqueeze(0)
        #     a_emb = a_emb.unsqueeze(0)

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

        # print("s_emb", s_emb)
        # print("s_emb", s_emb.shape)
        # print("a_emb", a_emb)
        # print("a_emb", a_emb.shape)
        trans_src = torch.stack([s_emb, a_emb], dim=-1).permute(0, 1, 3, 2)
        trans_src = trans_src.reshape(-1, 2 * self.seq_len, self.hidden_size).squeeze()
        pad_mask = torch.stack([pad_mask, pad_mask], dim=-1).permute(0, 1, 3, 2)
        pad_mask = pad_mask.reshape(-1, 2 * self.seq_len, 1).squeeze()
        # print("trans_src", trans_src)
        # print("trans_src", trans_src.shape)
        # print("pad_mask", pad_mask)
        # print("pad_mask", pad_mask.shape)

        ###############################################################################
        # attn_mask = self.transformer.generate_square_subsequent_mask(self.seq_len - 1, device="cuda")
        # print("attn_mask", attn_mask)
        # print("attn_mask", attn_mask.shape)
        ###############################################################################

        ###############################################################################
        # This would create sequences as if we're to predict the next state.
        # However, I didn't finish trying this yet.
        # It would use the whole transformer encoder-decoder architecture.
        # And no pooling/conv at the end.
        # trans_src = trans_inp[:, :-1, :].squeeze()
        # trans_tgt = trans_inp[:, 1:, :].squeeze()
        # trans_src_pad = pad_mask[:, :-1].squeeze()
        # trans_tgt_pad = pad_mask[:, 1:].squeeze()
        ######################################
        # print("trans_src", trans_src)
        # print("trans_src", trans_src.shape)
        # print("trans_src_pad", trans_src_pad)
        # print("trans_src_pad", trans_src_pad.shape)
        # print("trans_tgt", trans_tgt)
        # print("trans_tgt", trans_tgt.shape)
        # print("trans_tgt_pad", trans_tgt_pad)
        # print("trans_tgt_pad", trans_tgt_pad.shape)
        ###############################################################################

        # TODO: Idea -> use expected returns to attend to the state-action pairs?
        ###############################################################################
        # This is for whole transformer
        # trans_out = self.transformer(src=trans_src, tgt=trans_tgt, tgt_mask=attn_mask,
        #                              src_key_padding_mask=trans_src_pad)
        ######################################
        # This is for transformer encoder only
        trans_out = self.transformer(trans_src, src_key_padding_mask=pad_mask)
        # print("trans_out.shape", trans_out.shape)
        ###############################################################################

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
