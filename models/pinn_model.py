"""PINN model definitions."""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional


class PINNModel(nn.Module):
    """Physics-Informed Neural Network model for option pricing."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: int = 6,
        hidden_units: int = 128,
        output_dim: int = 1,
        activation: Optional[nn.Module] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialize the PINN model.

        Args:
            input_dim: Number of input features (asset prices + time).
            hidden_layers: Number of hidden layers.
            hidden_units: Number of units per hidden layer.
            output_dim: Number of output dimensions.
            activation: Activation function to use between layers.
            device: Device for model tensors.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.activation = activation if activation is not None else nn.Tanh()
        self.device = device

        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(self.activation)

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(self.activation)

        layers.append(nn.Linear(hidden_units, output_dim))
        self.network = nn.Sequential(*layers)

        self.to(self.device)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass through the PINN network.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        if x.device != self.device:
            x = x.to(self.device)
        return self.network(x)