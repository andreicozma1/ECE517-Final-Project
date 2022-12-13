from typing import Tuple

import torch
from torch import nn


class CommonAC(nn.Module):
    """Simple MLP network."""

    def __init__(self, actual_state_features: int, pred_state_features: int, out_features: int):
        """
        Args:
            in_features: observation shape of the environment
            n_actions: number of discrete actions available in the environment
            out_features: size of hidden layers
        """
        super().__init__()
        emb_dim = 64
        comb_hidden = 64
        comb_out = 32
        act_hid = 32
        crit_hid = 32
        self.actual_state = nn.Linear(actual_state_features, emb_dim)
        self.pred_state = nn.Linear(pred_state_features, emb_dim)

        self.combined = nn.Sequential(
                nn.Linear(emb_dim, comb_hidden),
                nn.ReLU(),
                nn.Linear(comb_hidden, comb_hidden),
                nn.ReLU(),
                nn.Linear(comb_hidden, comb_hidden),
                nn.ReLU(),
                nn.Linear(comb_hidden, comb_out),
        )

        self.actor = nn.Sequential(
                nn.Linear(comb_out, act_hid),
                nn.ReLU(),
                nn.Linear(act_hid, act_hid),
                nn.ReLU(),
                nn.Linear(act_hid, act_hid),
                nn.ReLU(),
                nn.Linear(act_hid, out_features),
        )

        self.critic = nn.Sequential(
                nn.Linear(comb_out, crit_hid),
                nn.ReLU(),
                nn.Linear(crit_hid, crit_hid),
                nn.ReLU(),
                nn.Linear(crit_hid, crit_hid),
                nn.ReLU(),
                nn.Linear(crit_hid, out_features),
        )

    def forward(self, last_state, pred_state, pred_state_action):
        state_actual = self.actual_state(last_state)
        state_pred = self.pred_state(pred_state)
        print(state_actual.shape, state_pred.shape)
        # average the two states over the last dimension
        combined = (state_actual + state_pred) / 2
        combined = self.combined(combined)
        actor = self.actor(state_actual)
        critic = self.critic(combined)
        return actor, critic
