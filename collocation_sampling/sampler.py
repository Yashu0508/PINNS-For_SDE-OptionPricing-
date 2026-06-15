"""Collocation sampling utilities for PINNs training."""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import qmc
from typing import Optional, Tuple


class CollocationSampler:
    """Generator for collocation points used in PINNs training."""

    def __init__(
        self,
        asset_dim: int,
        time_horizon: float = 1.0,
        interior_points: int = 50000,
        boundary_points: int = 10000,
        terminal_points: int = 10000,
        seed: int = 42,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialize the collocation sampler.

        Args:
            asset_dim: Number of underlying assets.
            time_horizon: Time horizon in years.
            interior_points: Number of interior collocation points.
            boundary_points: Number of boundary collocation points.
            terminal_points: Number of terminal payoff points.
            seed: Random seed for reproducibility.
            device: Torch device for tensor generation.
        """
        self.asset_dim = asset_dim
        self.time_horizon = time_horizon
        self.interior_points = interior_points
        self.boundary_points = boundary_points
        self.terminal_points = terminal_points
        self.seed = seed
        self.device = device

    def _make_qmc(self, n_points: int, dim: int) -> np.ndarray:
        """Generate Latin Hypercube samples in [0,1]^dim."""
        sampler = qmc.LatinHypercube(d=dim, seed=self.seed)
        return sampler.random(n=n_points)

    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        """Convert NumPy array to torch tensor on the configured device."""
        return torch.from_numpy(array.astype(np.float32)).to(self.device)

    def _batch_slice(self, total_points: int, batch_size: Optional[int], batch_index: int) -> Tuple[int, int]:
        """Calculate slice indices for batch generation."""
        if batch_size is None:
            return 0, total_points
        start = batch_index * batch_size
        end = min(start + batch_size, total_points)
        if start >= total_points:
            raise ValueError("Batch index out of range for requested sample size.")
        return start, end

    def sample_interior(self, batch_size: Optional[int] = None, batch_index: int = 0) -> torch.Tensor:
        """Generate interior collocation points for the PDE domain.

        Args:
            batch_size: Optional batch size for partial generation.
            batch_index: Batch index for sequential batch generation.

        Returns:
            Tensor of shape (n_points, asset_dim + 1) with normalized prices and time.
        """
        samples = self._make_qmc(self.interior_points, self.asset_dim + 1)
        prices = samples[:, : self.asset_dim]
        times = samples[:, -1:] * self.time_horizon
        points = np.concatenate([prices, times], axis=1)
        start, end = self._batch_slice(self.interior_points, batch_size, batch_index)
        return self._tensor(points[start:end])

    def sample_terminal(self, batch_size: Optional[int] = None, batch_index: int = 0) -> torch.Tensor:
        """Generate terminal payoff points at t = time_horizon.

        Args:
            batch_size: Optional batch size for partial generation.
            batch_index: Batch index for sequential batch generation.

        Returns:
            Tensor of shape (n_points, asset_dim + 1) with normalized prices and terminal time.
        """
        prices = self._make_qmc(self.terminal_points, self.asset_dim)
        times = np.full((self.terminal_points, 1), self.time_horizon, dtype=np.float32)
        points = np.concatenate([prices, times], axis=1)
        start, end = self._batch_slice(self.terminal_points, batch_size, batch_index)
        return self._tensor(points[start:end])

    def sample_boundary(self, batch_size: Optional[int] = None, batch_index: int = 0) -> torch.Tensor:
        """Generate boundary points on the unit hypercube for asset prices.

        Args:
            batch_size: Optional batch size for partial generation.
            batch_index: Batch index for sequential batch generation.

        Returns:
            Tensor of shape (n_points, asset_dim + 1) with boundary prices and time.
        """
        points_per_face = max(1, self.boundary_points // (2 * self.asset_dim))
        boundary_samples = []

        for dim_index in range(self.asset_dim):
            for fixed_value in (0.0, 1.0):
                face = self._make_qmc(points_per_face, self.asset_dim)
                face[:, dim_index] = fixed_value
                boundary_samples.append(face)

        raw_prices = np.vstack(boundary_samples)
        if raw_prices.shape[0] < self.boundary_points:
            extra = self._make_qmc(self.boundary_points - raw_prices.shape[0], self.asset_dim)
            raw_prices = np.vstack([raw_prices, extra])
        elif raw_prices.shape[0] > self.boundary_points:
            raw_prices = raw_prices[: self.boundary_points]

        times = np.random.default_rng(self.seed).uniform(0.0, self.time_horizon, size=(self.boundary_points, 1)).astype(np.float32)
        points = np.concatenate([raw_prices, times], axis=1)
        start, end = self._batch_slice(self.boundary_points, batch_size, batch_index)
        return self._tensor(points[start:end])

    def generate_samples(self, batch_size: Optional[int] = None, batch_index: int = 0) -> dict[str, torch.Tensor]:
        """Generate all collocation point sets in a single call."""
        return {
            "interior": self.sample_interior(batch_size=batch_size, batch_index=batch_index),
            "boundary": self.sample_boundary(batch_size=batch_size, batch_index=batch_index),
            "terminal": self.sample_terminal(batch_size=batch_size, batch_index=batch_index),
        }

    def plot_samples(self, samples: torch.Tensor, title: str = "Collocation Samples") -> None:
        """Visualize sampled points for up to 3 dimensions.

        Args:
            samples: Tensor of shape (n_points, asset_dim + 1).
            title: Title for the plot.
        """
        if samples.shape[1] < 2:
            raise ValueError("At least one asset dimension is required for plotting.")

        data = samples.cpu().numpy()
        dims = self.asset_dim
        fig = plt.figure(figsize=(8, 6))

        if dims == 1:
            plt.scatter(data[:, 0], data[:, 1], s=2, alpha=0.5)
            plt.xlabel("Normalized Price")
            plt.ylabel("Time")
        elif dims == 2:
            plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap="viridis", s=2, alpha=0.6)
            plt.xlabel("Price 1")
            plt.ylabel("Price 2")
            plt.colorbar(label="Time")
        else:
            plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap="viridis", s=2, alpha=0.6)
            plt.xlabel("Price 1")
            plt.ylabel("Price 2")
            plt.colorbar(label="Price 3")
            plt.title(f"{title} (showing first 3 asset dims)")

        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()