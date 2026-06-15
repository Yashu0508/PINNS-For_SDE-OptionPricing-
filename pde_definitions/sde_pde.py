"""Black-Scholes PDE definitions for PINNs."""

from __future__ import annotations

import torch
from torch import nn
from typing import Callable, Optional


class BlackScholesPDE:
    """Black-Scholes PDE residuals for correlated multi-asset systems."""

    def __init__(
        self,
        asset_dim: int,
        risk_free_rate: float = 0.01,
        sigma: Optional[list[float]] = None,
        corr_matrix: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialize Black-Scholes PDE coefficients.

        Args:
            asset_dim: Number of underlying assets.
            risk_free_rate: Risk-free interest rate.
            sigma: Volatility vector for each asset.
            corr_matrix: Correlation matrix for asset returns.
            device: Torch device for computation.
        """
        self.asset_dim = asset_dim
        self.risk_free_rate = risk_free_rate
        self.device = device

        if sigma is None:
            sigma = [0.2] * asset_dim
        if len(sigma) != asset_dim:
            raise ValueError("Length of sigma must equal asset_dim.")
        self.sigma = torch.tensor(sigma, dtype=torch.float32, device=self.device)

        if corr_matrix is None:
            corr_matrix = torch.eye(asset_dim, dtype=torch.float32, device=self.device)
        else:
            corr_matrix = corr_matrix.to(self.device).float()
            if corr_matrix.shape != (asset_dim, asset_dim):
                raise ValueError("corr_matrix must have shape (asset_dim, asset_dim).")
        self.corr_matrix = corr_matrix

        self.cov_matrix = torch.diag(self.sigma) @ self.corr_matrix @ torch.diag(self.sigma)

    def _split_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split input tensor into asset prices and time."""
        if x.ndim != 2 or x.shape[1] != self.asset_dim + 1:
            raise ValueError(f"Input tensor must have shape (batch_size, {self.asset_dim + 1}).")
        prices = x[:, : self.asset_dim]
        time = x[:, -1:].clone()
        return prices, time

    def _gradients(self, outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Compute first-order gradients of outputs with respect to inputs."""
        grad = torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return grad

    def _second_derivatives(self, gradients: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Compute Hessian for the asset price gradient.

        Args:
            gradients: First-order gradients with respect to asset prices.
            inputs: Input tensor including prices and time.

        Returns:
            Hessian tensor of shape (batch_size, asset_dim, asset_dim).
        """
        hessian = []
        for dim in range(self.asset_dim):
            grad_row = torch.autograd.grad(
                gradients[:, dim],
                inputs,
                grad_outputs=torch.ones_like(gradients[:, dim]),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            hessian.append(grad_row[:, : self.asset_dim])
        return torch.stack(hessian, dim=1)

    def compute_pde_residual(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Compute interior PDE residuals using automatic differentiation.

        Args:
            model: PINN model that maps (prices, time) to option price.
            x: Input tensor of shape (batch_size, asset_dim + 1).

        Returns:
            Residual tensor of shape (batch_size, 1).
        """
        x = x.to(self.device).float().requires_grad_(True)
        prices, _ = self._split_input(x)
        u = model(x)

        if u.ndim == 1:
            u = u.unsqueeze(-1)

        grads = self._gradients(u, x)
        u_s = grads[:, : self.asset_dim]
        u_t = grads[:, -1:]

        hessian = self._second_derivatives(u_s, x)

        price_outer = prices.unsqueeze(2) * prices.unsqueeze(1)
        cov = self.cov_matrix.unsqueeze(0)
        diffusion = 0.5 * (cov * price_outer * hessian).sum(dim=(1, 2))

        drift = self.risk_free_rate * (prices * u_s).sum(dim=1)
        discount = self.risk_free_rate * u.squeeze(-1)

        residual = u_t.squeeze(-1) + diffusion + drift - discount
        return residual.unsqueeze(-1)

    def compute_boundary_loss(
        self,
        model: nn.Module,
        x_boundary: torch.Tensor,
        boundary_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        boundary_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute boundary loss against prescribed boundary conditions.

        Args:
            model: PINN model.
            x_boundary: Boundary input tensor.
            boundary_fn: Callable returning boundary values for x_boundary.
            boundary_values: Precomputed boundary values.

        Returns:
            Mean squared boundary loss.
        """
        if boundary_values is None and boundary_fn is None:
            raise ValueError("Either boundary_fn or boundary_values must be provided.")

        x_boundary = x_boundary.to(self.device).float()
        predictions = model(x_boundary)

        if boundary_values is None:
            boundary_values = boundary_fn(x_boundary.to(self.device)).to(self.device).float()

        return torch.mean((predictions - boundary_values) ** 2)

    def compute_terminal_loss(
        self,
        model: nn.Module,
        x_terminal: torch.Tensor,
        payoff_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Compute terminal payoff loss at maturity.

        Args:
            model: PINN model.
            x_terminal: Terminal-time input tensor.
            payoff_fn: Callable that returns payoff values for input state.

        Returns:
            Mean squared terminal loss.
        """
        x_terminal = x_terminal.to(self.device).float()
        predictions = model(x_terminal)
        payoffs = payoff_fn(x_terminal).to(self.device).float()
        return torch.mean((predictions - payoffs) ** 2)