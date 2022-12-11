from typing import Tuple

from torch import nn


class CommonBase(nn.Module):
    """Simple MLP network."""

    def __init__(self, input_shape: Tuple[int], hidden_size: int = 128):
        """
        Args:
            input_shape: observation shape of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(input_shape[0], hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, input_x):
        """Forward pass through network.

        Args:
            x: input to network

        Returns:
            output of network
        """
        return self.net(input_x.float())
