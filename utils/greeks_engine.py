"""Greeks extraction utilities for PINN option pricing models."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import grad
from typing import Optional


class GreeksEngine:
    """Compute Greeks from a trained PINN option pricing model."""

    def __init__(self, model: nn.Module, device: torch.device = torch.device("cpu")) -> None:
        """Initialize the Greeks engine.

        Args:
            model: Trained PINN model.
            device: Device for evaluation.
        """
        self.model = model.to(device)
        self.device = device

    @staticmethod
    def _check_tensor(x: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(x).all():
            raise ValueError("Input tensor contains non-finite values.")
        return x

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device).float()
        x = self._check_tensor(x)
        if x.ndim != 2:
            raise ValueError("Input tensor must have shape (batch_size, input_dim).")
        x.requires_grad_(True)
        return x

    def _split_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < 2:
            raise ValueError("Input tensor must contain asset prices and time.")
        return x[:, :-1]

    def compute_delta(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Delta for each asset using exact automatic differentiation.

        Args:
            x: Input tensor of shape (batch_size, asset_dim + 1).

        Returns:
            Tensor of shape (batch_size, asset_dim) containing asset deltas.
        """
        x = self._prepare_input(x)
        prices = self._split_input(x)

        output = self.model(x)
        if output.ndim == 1:
            output = output.unsqueeze(-1)

        delta = grad(
            outputs=output,
            inputs=prices,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return delta

    def compute_gamma(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the Gamma matrix for a multi-asset system.

        Args:
            x: Input tensor of shape (batch_size, asset_dim + 1).

        Returns:
            Tensor of shape (batch_size, asset_dim, asset_dim) containing second derivatives.
        """
        x = self._prepare_input(x)
        prices = self._split_input(x)
        delta = self.compute_delta(x)

        gamma_rows = []
        for asset_index in range(delta.shape[1]):
            second_grad = grad(
                outputs=delta[:, asset_index],
                inputs=prices,
                grad_outputs=torch.ones_like(delta[:, asset_index]),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            gamma_rows.append(second_grad)

        gamma = torch.stack(gamma_rows, dim=1)
        return gamma

    def plot_delta(self, delta: torch.Tensor, asset_names: Optional[list[str]] = None) -> None:
        """Visualize delta values for each asset."""
        delta = delta.detach().cpu().numpy()
        asset_dim = delta.shape[1]

        plt.figure(figsize=(10, 6))
        for idx in range(asset_dim):
            label = asset_names[idx] if asset_names is not None and idx < len(asset_names) else f"Asset {idx + 1}"
            plt.plot(delta[:, idx], label=label)

        plt.title("Delta Across Batch")
        plt.xlabel("Sample Index")
        plt.ylabel("Delta")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_gamma(self, gamma: torch.Tensor, asset_names: Optional[list[str]] = None) -> None:
        """Visualize diagonal Gamma values for each asset."""
        gamma = gamma.detach().cpu().numpy()
        diag_gamma = gamma[:, range(gamma.shape[1]), range(gamma.shape[1])]
        asset_dim = diag_gamma.shape[1]

        plt.figure(figsize=(10, 6))
        for idx in range(asset_dim):
            label = asset_names[idx] if asset_names is not None and idx < len(asset_names) else f"Asset {idx + 1}"
            plt.plot(diag_gamma[:, idx], label=label)

        plt.title("Diagonal Gamma Across Batch")
        plt.xlabel("Sample Index")
        plt.ylabel("Gamma")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
