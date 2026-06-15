"""Visualization module for PINN option pricing experiments."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import torch
from typing import Any, Iterable, Optional, Sequence, Tuple, Union


NumericArray = Union[np.ndarray, torch.Tensor, Sequence[float]]


class Visualizer:
    """Publication-quality visualization utilities for PINN option pricing."""

    def __init__(self, style: str = "seaborn-darkgrid", figsize: Tuple[int, int] = (12, 8)) -> None:
        """Initialize the visualizer.

        Args:
            style: Matplotlib style string.
            figsize: Default figure size.
        """
        plt.style.use(style)
        self.figsize = figsize

    @staticmethod
    def _to_numpy(data: NumericArray) -> np.ndarray:
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        if isinstance(data, np.ndarray):
            return data
        return np.asarray(data, dtype=float)

    @staticmethod
    def _save_figure(fig: Any, save_path: Optional[str]) -> None:
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

    def plot_losses(
        self,
        train_losses: dict[str, Sequence[float]],
        val_losses: Optional[dict[str, Sequence[float]]] = None,
        title: str = "Training Loss Curves",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot training and validation loss curves.

        Args:
            train_losses: Dictionary of loss name to training values.
            val_losses: Optional dictionary of validation loss values.
            title: Plot title.
            save_path: Optional file path to export the figure.
        """
        plt.figure(figsize=self.figsize)
        for name, values in train_losses.items():
            plt.plot(values, label=f"Train {name}", linewidth=2)
        if val_losses is not None:
            for name, values in val_losses.items():
                plt.plot(values, label=f"Val {name}", linewidth=2, linestyle="--")

        plt.title(title, fontsize=18)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.yscale("log")
        plt.legend(fontsize=12)
        plt.grid(alpha=0.4)
        self._save_figure(plt, save_path)
        plt.show()

    def plot_price_surface(
        self,
        x_grid: NumericArray,
        y_grid: NumericArray,
        surface: NumericArray,
        x_label: str = "Asset 1 Price",
        y_label: str = "Asset 2 Price",
        z_label: str = "Option Price",
        title: str = "Option Price Surface",
        save_path: Optional[str] = None,
        interactive: bool = False,
    ) -> Optional[go.Figure]:
        """Plot a 3D surface of option prices.

        Args:
            x_grid: Meshgrid values for the first axis.
            y_grid: Meshgrid values for the second axis.
            surface: Surface values.
            x_label: Label for x axis.
            y_label: Label for y axis.
            z_label: Label for z axis.
            title: Plot title.
            save_path: Optional file path to export the figure.
            interactive: If True, return a Plotly figure.
        """
        x = self._to_numpy(x_grid)
        y = self._to_numpy(y_grid)
        z = self._to_numpy(surface)

        if interactive:
            fig = go.Figure(data=go.Surface(x=x, y=y, z=z, colorscale="viridis"))
            fig.update_layout(
                title=title,
                scene=dict(xaxis_title=x_label, yaxis_title=y_label, zaxis_title=z_label),
                template="plotly_white",
            )
            if save_path is not None:
                fig.write_image(save_path)
            fig.show()
            return fig

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, z, cmap="viridis", edgecolor="none", alpha=0.9)
        ax.set_title(title, fontsize=18)
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.set_zlabel(z_label, fontsize=14)
        plt.tight_layout()
        self._save_figure(fig, save_path)
        plt.show()
        return None

    def plot_delta_surface(
        self,
        x_grid: NumericArray,
        y_grid: NumericArray,
        delta_surface: NumericArray,
        x_label: str = "Asset 1 Price",
        y_label: str = "Asset 2 Price",
        title: str = "Delta Surface",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot a 2D delta surface for a two-asset system.

        Args:
            x_grid: Grid values for the first price axis.
            y_grid: Grid values for the second price axis.
            delta_surface: Surface values.
            x_label: X-axis label.
            y_label: Y-axis label.
            title: Plot title.
            save_path: Optional file path to export the figure.
        """
        x = self._to_numpy(x_grid)
        y = self._to_numpy(y_grid)
        z = self._to_numpy(delta_surface)

        plt.figure(figsize=self.figsize)
        contour = plt.contourf(x, y, z, levels=40, cmap="plasma")
        plt.title(title, fontsize=18)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        cbar = plt.colorbar(contour)
        cbar.set_label("Delta", fontsize=14)
        plt.tight_layout()
        self._save_figure(plt, save_path)
        plt.show()

    def plot_gamma_heatmap(
        self,
        gamma_matrix: NumericArray,
        asset_names: Optional[Sequence[str]] = None,
        title: str = "Gamma Heatmap",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot a gamma heatmap for multi-asset sensitivities.

        Args:
            gamma_matrix: Square matrix of gamma values.
            asset_names: Optional asset labels.
            title: Plot title.
            save_path: Optional file path to export the figure.
        """
        gamma = self._to_numpy(gamma_matrix)
        if gamma.ndim != 2 or gamma.shape[0] != gamma.shape[1]:
            raise ValueError("gamma_matrix must be a square 2D array.")

        labels = asset_names if asset_names is not None else [f"S{i+1}" for i in range(gamma.shape[0])]

        plt.figure(figsize=self.figsize)
        heatmap = plt.imshow(gamma, cmap="coolwarm", origin="lower")
        plt.title(title, fontsize=18)
        plt.xticks(ticks=np.arange(len(labels)), labels=labels, fontsize=12)
        plt.yticks(ticks=np.arange(len(labels)), labels=labels, fontsize=12)
        cbar = plt.colorbar(heatmap)
        cbar.set_label("Gamma", fontsize=14)
        plt.tight_layout()
        self._save_figure(plt, save_path)
        plt.show()

    def compare_models(
        self,
        x: NumericArray,
        pinn_values: NumericArray,
        benchmark_values: NumericArray,
        x_label: str = "Index",
        y_label: str = "Option Price",
        title: str = "PINN vs Benchmark Comparison",
        save_path: Optional[str] = None,
        interactive: bool = False,
    ) -> Optional[go.Figure]:
        """Compare PINN predictions against benchmark values.

        Args:
            x: Independent variable values.
            pinn_values: Predicted PINN values.
            benchmark_values: Benchmark values.
            x_label: X-axis label.
            y_label: Y-axis label.
            title: Plot title.
            save_path: Optional file path to export the figure.
            interactive: If True, return a Plotly figure.
        """
        x_vals = self._to_numpy(x)
        pinn_vals = self._to_numpy(pinn_values)
        benchmark_vals = self._to_numpy(benchmark_values)

        if interactive:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=pinn_vals, mode="lines+markers", name="PINN", line=dict(color="#1f77b4")))
            fig.add_trace(go.Scatter(x=x_vals, y=benchmark_vals, mode="lines+markers", name="Benchmark", line=dict(color="#ff7f0e")))
            fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, template="plotly_white")
            if save_path is not None:
                fig.write_image(save_path)
            fig.show()
            return fig

        plt.figure(figsize=self.figsize)
        plt.plot(x_vals, pinn_vals, label="PINN", linewidth=2)
        plt.plot(x_vals, benchmark_vals, label="Benchmark", linewidth=2, linestyle="--")
        plt.title(title, fontsize=18)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.4)
        self._save_figure(plt, save_path)
        plt.show()
        return None
