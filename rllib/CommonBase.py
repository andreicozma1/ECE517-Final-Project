from typing import Tuple

from torch import nn


class CommonBase(nn.Module):
    """Simple MLP network."""

    def __init__(self, in_features: int, out_features: int):
        """
        Args:
            in_features: observation shape of the environment
            n_actions: number of discrete actions available in the environment
            out_features: size of hidden layers
        """
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.ReLU(),
                nn.Linear(out_features, out_features),
        )

    def forward(self, input_x):
        """Forward pass through network.

        Args:
            x: input to network

        Returns:
            output of network
        """
        inp = input_x.float()
        out = self.net(inp)
        return out
