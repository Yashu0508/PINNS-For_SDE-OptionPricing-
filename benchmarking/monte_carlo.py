"""Monte Carlo benchmark engine for multi-asset basket options."""

from __future__ import annotations

import time

import numpy as np
import torch
from typing import Callable, Dict, Optional, Sequence, Union


TensorLike = Union[torch.Tensor, np.ndarray, float, int]


class MonteCarloPricer:
    """Monte Carlo pricer for correlated multi-asset basket options."""

    def __init__(
        self,
        asset_dim: int,
        risk_free_rate: float = 0.01,
        mu: Optional[Sequence[float]] = None,
        sigma: Optional[Sequence[float]] = None,
        corr_matrix: Optional[torch.Tensor] = None,
        seed: int = 42,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the Monte Carlo pricer.

        Args:
            asset_dim: Number of underlying assets.
            risk_free_rate: Risk-free interest rate.
            mu: Expected drift for each asset.
            sigma: Volatility vector for each asset.
            corr_matrix: Correlation matrix for asset returns.
            seed: Random seed for reproducibility.
            device: Torch device for simulation.
        """
        self.asset_dim = asset_dim
        self.risk_free_rate = risk_free_rate
        self.seed = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

        self.mu = torch.tensor(mu if mu is not None else [0.05] * asset_dim, dtype=self.dtype, device=self.device)
        if self.mu.numel() != asset_dim:
            raise ValueError("mu must have length equal to asset_dim.")

        self.sigma = torch.tensor(sigma if sigma is not None else [0.2] * asset_dim, dtype=self.dtype, device=self.device)
        if self.sigma.numel() != asset_dim:
            raise ValueError("sigma must have length equal to asset_dim.")

        if corr_matrix is None:
            corr_matrix = torch.eye(asset_dim, dtype=self.dtype, device=self.device)
        else:
            corr_matrix = corr_matrix.to(self.device).float()
            if corr_matrix.shape != (asset_dim, asset_dim):
                raise ValueError("corr_matrix must have shape (asset_dim, asset_dim).")

        self.corr_matrix = corr_matrix
        self.cholesky = torch.linalg.cholesky(self.corr_matrix)
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(self.seed)

    def _to_tensor(self, x: TensorLike) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device).float()
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.device).float()
        return torch.tensor(x, dtype=self.dtype, device=self.device)

    def _normalize_inputs(self, S0: TensorLike) -> torch.Tensor:
        S0 = self._to_tensor(S0)
        if S0.ndim == 1:
            S0 = S0.unsqueeze(0)
        if S0.shape[1] != self.asset_dim:
            raise ValueError("Initial asset prices S0 must have shape (asset_dim,) or (batch, asset_dim).")
        return S0

    def simulate_paths(
        self,
        S0: TensorLike,
        maturity: float,
        steps: int = 252,
        n_paths: int = 100000,
        antithetic: bool = False,
        variance_reduction: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Simulate correlated GBM paths using Euler-Maruyama.

        Args:
            S0: Initial asset prices of shape (asset_dim,) or (batch, asset_dim).
            maturity: Maturity time horizon.
            steps: Number of time steps.
            n_paths: Number of Monte Carlo simulation paths.
            antithetic: Use antithetic variates for variance reduction.
            variance_reduction: Optional hook to adjust increments or payoffs.

        Returns:
            Tensor of shape (n_paths, steps + 1, asset_dim).
        """
        S0 = self._normalize_inputs(S0)
        batch_shape = S0.shape[0]
        if batch_shape > 1:
            raise ValueError("S0 batch dimension must be 1 for path simulation.")

        dt = float(maturity) / float(steps)
        sqrt_dt = np.sqrt(dt)
        n_sim = n_paths * 2 if antithetic else n_paths

        normals = torch.randn((n_sim, steps, self.asset_dim), generator=self._rng, device=self.device, dtype=self.dtype)
        if variance_reduction is not None:
            normals = variance_reduction(normals)

        correlated = normals @ self.cholesky.T
        if antithetic:
            correlated = torch.cat([correlated, -correlated], dim=0)
            correlated = correlated[:n_paths]

        drift = (self.mu - 0.5 * self.sigma ** 2) * dt
        diffusion = self.sigma * sqrt_dt

        log_S = torch.log(S0).expand(n_paths, self.asset_dim).clone()
        paths = torch.empty((n_paths, steps + 1, self.asset_dim), device=self.device, dtype=self.dtype)
        paths[:, 0, :] = S0

        for t in range(steps):
            increment = drift + diffusion * correlated[:, t, :]
            log_S = log_S + increment
            paths[:, t + 1, :] = torch.exp(log_S)

        return paths

    def price_basket_option(
        self,
        S0: TensorLike,
        strike: TensorLike,
        maturity: float,
        basket_weights: Optional[Sequence[float]] = None,
        option_type: str = "call",
        steps: int = 252,
        n_paths: int = 100000,
        antithetic: bool = False,
        variance_reduction: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        confidence_level: float = 0.95,
    ) -> Dict[str, Union[float, torch.Tensor]]:
        """Price a basket option using Monte Carlo simulation.

        Args:
            S0: Initial asset prices.
            strike: Strike price or list of strikes.
            maturity: Option maturity.
            basket_weights: Weights for each asset in the basket.
            option_type: 'call' or 'put'.
            steps: Time discretization steps.
            n_paths: Number of simulated paths.
            antithetic: Use antithetic variates.
            variance_reduction: Optional hook for variance reduction.
            confidence_level: Confidence interval coverage.

        Returns:
            Dictionary with price, std_error, confidence_interval, runtime, and samples.
        """
        if basket_weights is None:
            basket_weights = [1.0 / self.asset_dim] * self.asset_dim
        weights = self._to_tensor(basket_weights)
        if weights.numel() != self.asset_dim:
            raise ValueError("basket_weights must have length equal to asset_dim.")

        strike_tensor = self._to_tensor(strike)
        runtime_start = time.perf_counter()
        paths = self.simulate_paths(
            S0,
            maturity,
            steps=steps,
            n_paths=n_paths,
            antithetic=antithetic,
            variance_reduction=variance_reduction,
        )
        terminal_prices = paths[:, -1, :]
        basket_values = terminal_prices @ weights

        if option_type.lower() == "call":
            payoff = torch.maximum(basket_values - strike_tensor, torch.zeros_like(basket_values))
        elif option_type.lower() == "put":
            payoff = torch.maximum(strike_tensor - basket_values, torch.zeros_like(basket_values))
        else:
            raise ValueError("option_type must be 'call' or 'put'.")

        if variance_reduction is not None:
            payoff = variance_reduction(payoff)

        discount = torch.exp(-self.risk_free_rate * maturity)
        price = discount * payoff.mean()
        std_error = discount * payoff.std(unbiased=True) / torch.sqrt(torch.tensor(float(n_paths), device=self.device))
        alpha = 1.0 - confidence_level
        z_score = float(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1.0 - alpha / 2)))
        ci_lower = price - z_score * std_error
        ci_upper = price + z_score * std_error

        runtime_end = time.perf_counter()
        runtime = runtime_end - runtime_start

        return {
            "price": price.item(),
            "std_error": std_error.item(),
            "confidence_interval": (ci_lower.item(), ci_upper.item()),
            "runtime": runtime,
            "payoff_samples": payoff.detach().cpu(),
        }
