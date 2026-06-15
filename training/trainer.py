"""Training utilities for PINN models."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional

import torch
from torch import nn
from torch.optim import Adam, LBFGS, Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class PINNTrainer:
    """Trainer for high-dimensional option pricing PINNs."""

    def __init__(
        self,
        model: nn.Module,
        pde_module: Any,
        payoff_fn: Callable[[torch.Tensor], torch.Tensor],
        boundary_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        lambda_interior: float = 1.0,
        lambda_boundary: float = 10.0,
        lambda_terminal: float = 10.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        epochs: int = 50000,
        use_lbfgs: bool = False,
        lbfgs_max_iter: int = 20,
        clip_grad_norm: float = 1.0,
        patience: int = 200,
        log_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        mixed_precision: bool = True,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialize the PINN trainer.

        Args:
            model: PINN model to train.
            pde_module: PDE evaluator with residual computation.
            payoff_fn: Terminal payoff function.
            boundary_fn: Boundary value function.
            lambda_interior: Weight for the PDE residual loss.
            lambda_boundary: Weight for the boundary loss.
            lambda_terminal: Weight for the terminal payoff loss.
            learning_rate: Adam learning rate.
            weight_decay: Adam weight decay.
            epochs: Maximum number of training epochs.
            use_lbfgs: Whether to apply LBFGS refinement.
            lbfgs_max_iter: Maximum LBFGS iterations per refinement step.
            clip_grad_norm: Gradient clipping norm.
            patience: Early stopping patience.
            log_dir: TensorBoard log directory.
            checkpoint_dir: Directory to save checkpoints.
            mixed_precision: Enable AMP for CUDA training.
            device: Device for training.
        """
        self.device = device
        self.model = model.to(self.device)
        self.pde_module = pde_module
        self.payoff_fn = payoff_fn
        self.boundary_fn = boundary_fn
        self.lambda_interior = lambda_interior
        self.lambda_boundary = lambda_boundary
        self.lambda_terminal = lambda_terminal
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.use_lbfgs = use_lbfgs
        self.lbfgs_max_iter = lbfgs_max_iter
        self.clip_grad_norm = clip_grad_norm
        self.patience = patience
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.mixed_precision = mixed_precision and self.device.type == "cuda"

        self.optimizer: Optimizer = Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.lbfgs: Optional[LBFGS] = (
            LBFGS(
                self.model.parameters(),
                lr=1.0,
                max_iter=self.lbfgs_max_iter,
                history_size=10,
                tolerance_grad=1e-7,
                tolerance_change=1e-9,
                line_search_fn="strong_wolfe",
            )
            if self.use_lbfgs
            else None
        )
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        self.writer = SummaryWriter(log_dir) if self.log_dir else None
        self.best_loss = float("inf")
        self.early_stop_counter = 0
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        if self.checkpoint_dir is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _compute_pde_loss(self, x_interior: torch.Tensor) -> torch.Tensor:
        x_interior = x_interior.to(self.device).float()
        residual = self.pde_module.compute_pde_residual(self.model, x_interior)
        return torch.mean(residual ** 2)

    def _compute_boundary_loss(
        self,
        x_boundary: torch.Tensor,
        boundary_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_boundary = x_boundary.to(self.device).float()
        predictions = self.model(x_boundary)
        if boundary_values is None:
            if self.boundary_fn is None:
                raise ValueError("boundary_fn must be provided if boundary_values are not supplied.")
            boundary_values = self.boundary_fn(x_boundary)
        boundary_values = boundary_values.to(self.device).float()
        return torch.mean((predictions - boundary_values) ** 2)

    def _compute_terminal_loss(self, x_terminal: torch.Tensor) -> torch.Tensor:
        x_terminal = x_terminal.to(self.device).float()
        predictions = self.model(x_terminal)
        payoffs = self.payoff_fn(x_terminal).to(self.device).float()
        return torch.mean((predictions - payoffs) ** 2)

    def _compute_total_loss(self, training_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pde_loss = self._compute_pde_loss(training_data["interior"])
        boundary_loss = self._compute_boundary_loss(
            training_data["boundary"], training_data.get("boundary_values")
        )
        terminal_loss = self._compute_terminal_loss(training_data["terminal"])
        total_loss = (
            self.lambda_interior * pde_loss
            + self.lambda_boundary * boundary_loss
            + self.lambda_terminal * terminal_loss
        )
        return {
            "total_loss": total_loss,
            "pde_loss": pde_loss,
            "boundary_loss": boundary_loss,
            "terminal_loss": terminal_loss,
        }

    def _run_lbfgs(self, training_data: Dict[str, torch.Tensor]) -> None:
        if self.lbfgs is None:
            return

        def closure() -> torch.Tensor:
            self.lbfgs.zero_grad()
            losses = self._compute_total_loss(training_data)
            losses["total_loss"].backward()
            return losses["total_loss"]

        self.lbfgs.step(closure)

    def train(
        self,
        training_data: Dict[str, torch.Tensor],
        validation_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, list[float]]:
        """Train the PINN model.

        Args:
            training_data: Dictionary containing interior, boundary, terminal tensors and optional boundary values.
            validation_data: Optional validation dataset with the same keys.

        Returns:
            Training history dictionary.
        """
        history: Dict[str, list[float]] = {
            "total_loss": [],
            "pde_loss": [],
            "boundary_loss": [],
            "terminal_loss": [],
            "val_total_loss": [],
        }

        progress = tqdm(range(1, self.epochs + 1), desc="Training", unit="epoch")
        for epoch in progress:
            self.model.train()
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                losses = self._compute_total_loss(training_data)
            total_loss = losses["total_loss"]

            if self.scaler is not None:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()

            if self.use_lbfgs:
                self._run_lbfgs(training_data)

            val_metrics = None
            if validation_data is not None:
                val_metrics = self.validate(validation_data)
                history["val_total_loss"].append(val_metrics["total_loss"])

            history["total_loss"].append(total_loss.item())
            history["pde_loss"].append(losses["pde_loss"].item())
            history["boundary_loss"].append(losses["boundary_loss"].item())
            history["terminal_loss"].append(losses["terminal_loss"].item())

            if self.writer is not None:
                self.writer.add_scalar("loss/total", total_loss.item(), epoch)
                self.writer.add_scalar("loss/pde", losses["pde_loss"].item(), epoch)
                self.writer.add_scalar("loss/boundary", losses["boundary_loss"].item(), epoch)
                self.writer.add_scalar("loss/terminal", losses["terminal_loss"].item(), epoch)
                if val_metrics is not None:
                    self.writer.add_scalar("loss/val_total", val_metrics["total_loss"], epoch)

            monitor_loss = val_metrics["total_loss"] if val_metrics is not None else total_loss.item()
            if monitor_loss < self.best_loss:
                self.best_loss = monitor_loss
                self.early_stop_counter = 0
                if self.checkpoint_dir is not None:
                    self.save_checkpoint(os.path.join(self.checkpoint_dir, "best_checkpoint.pt"), epoch)
            else:
                self.early_stop_counter += 1

            progress.set_postfix(
                {
                    "total": total_loss.item(),
                    "pde": losses["pde_loss"].item(),
                    "boundary": losses["boundary_loss"].item(),
                    "terminal": losses["terminal_loss"].item(),
                    "val_total": val_metrics["total_loss"] if val_metrics is not None else None,
                }
            )

            if self.early_stop_counter >= self.patience:
                logging.info("Early stopping triggered at epoch %d.", epoch)
                break

        if self.writer is not None:
            self.writer.close()

        return history

    def validate(self, validation_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Validate the PINN model on a held-out dataset."""
        self.model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.mixed_precision):
            losses = self._compute_total_loss(validation_data)
        return {
            "total_loss": losses["total_loss"].item(),
            "pde_loss": losses["pde_loss"].item(),
            "boundary_loss": losses["boundary_loss"].item(),
            "terminal_loss": losses["terminal_loss"].item(),
        }

    def save_checkpoint(self, file_path: str, epoch: Optional[int] = None) -> None:
        """Save model and optimizer state to checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
        }
        if self.scaler is not None:
            checkpoint["scaler_state"] = self.scaler.state_dict()
        torch.save(checkpoint, file_path)

    def load_checkpoint(self, file_path: str) -> None:
        """Load model and optimizer state from checkpoint."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Checkpoint not found: {file_path}")

        checkpoint = torch.load(file_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        if self.scaler is not None and "scaler_state" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        logging.info("Loaded checkpoint from %s", file_path)
