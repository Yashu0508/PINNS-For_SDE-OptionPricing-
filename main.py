"""Experiment orchestration pipeline for PINN option pricing."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from calibration.calibrator import SDECalibrator
from collocation_sampling.sampler import CollocationSampler
from benchmarking.monte_carlo import MonteCarloPricer
from models.pinn_model import PINNModel
from pde_definitions.sde_pde import BlackScholesPDE
from training.trainer import PINNTrainer
from utils.config_utils import load_config
from utils.logging_utils import setup_logging
from utils.greeks_engine import GreeksEngine
from visualization.visualizer import Visualizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_experiment_dirs(base_path: Path, experiment_name: str) -> Dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = base_path / f"{experiment_name}_{timestamp}"
    paths = {
        "root": root,
        "logs": root / "logs",
        "checkpoints": root / "checkpoints",
        "figures": root / "figures",
        "artifacts": root / "artifacts",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def create_payoff_fn(
    strike: float,
    basket_weights: Sequence[float],
    option_type: str = "call",
) -> Any:
    def payoff(x: Tensor) -> Tensor:
        prices = x[:, :-1]
        weights_tensor = torch.tensor(basket_weights, dtype=torch.float32, device=prices.device)
        basket = prices @ weights_tensor
        strike_tensor = torch.tensor(strike, dtype=torch.float32, device=prices.device)
        if option_type.lower() == "call":
            return torch.clamp(basket - strike_tensor, min=0.0).unsqueeze(-1)
        return torch.clamp(strike_tensor - basket, min=0.0).unsqueeze(-1)

    return payoff


def create_boundary_fn(
    strike: float,
    basket_weights: Sequence[float],
    option_type: str = "call",
) -> Any:
    def boundary_value(x: Tensor) -> Tensor:
        return create_payoff_fn(strike, basket_weights, option_type)(x)

    return boundary_value


def save_config(config: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def evaluate_surface(
    model: PINNModel,
    asset_dim: int,
    points_per_axis: int = 50,
    time_value: float = 0.5,
) -> Dict[str, np.ndarray]:
    price_axis = np.linspace(0.0, 1.0, points_per_axis)
    x_grid, y_grid = np.meshgrid(price_axis, price_axis)
    if asset_dim < 2:
        raise ValueError("Surface evaluation requires at least 2 asset dimensions.")

    points = np.zeros((points_per_axis * points_per_axis, asset_dim + 1), dtype=np.float32)
    points[:, 0] = x_grid.ravel()
    points[:, 1] = y_grid.ravel()
    points[:, -1] = float(time_value)
    x_tensor = torch.from_numpy(points).to(model.device)
    with torch.no_grad():
        price_values = model(x_tensor).cpu().numpy().reshape(x_grid.shape)
    return {"x": x_grid, "y": y_grid, "z": price_values}


def run_experiment(config: Dict[str, Any], resume_checkpoint: Optional[str] = None) -> None:
    output_root = Path(config["experiment"]["output_root"])
    experiment_name = config["experiment"].get("name", "pinn_experiment")
    paths = build_experiment_dirs(output_root, experiment_name)

    setup_logging(config["experiment"].get("log_level", "INFO"))
    logger = logging.getLogger(__name__)
    logger.info("Starting experiment: %s", experiment_name)

    set_seed(config.get("seed", 42))

    save_config(config, paths["artifacts"] / "config.json")

    asset_dim = len(config["data"]["assets"])
    calibrator = SDECalibrator(
        file_path=config["data"]["csv_path"],
        assets=config["data"]["assets"],
        start_date=config["data"].get("start_date"),
        end_date=config["data"].get("end_date"),
    )
    calibrated_data = calibrator.load_data()
    returns = calibrator.compute_returns()
    drift = calibrator.estimate_drift()
    volatility = calibrator.estimate_volatility()
    covariance = calibrator.estimate_covariance()
    correlation = calibrator.estimate_correlation()

    logger.info("Loaded data with %d samples.", len(calibrated_data))
    logger.info("Estimated drift: %s", drift.to_dict())
    logger.info("Estimated volatility: %s", volatility.to_dict())
    np.savetxt(paths["artifacts"] / "correlation_matrix.csv", correlation.values, delimiter=",")

    corr_matrix = torch.tensor(config["pde"]["correlation"], dtype=torch.float32)
    sampler = CollocationSampler(
        asset_dim=asset_dim,
        time_horizon=config["pde"]["time_horizon"],
        interior_points=config["collocation"]["interior_points"],
        boundary_points=config["collocation"]["boundary_points"],
        terminal_points=config["collocation"]["terminal_points"],
        seed=config.get("seed", 42),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    training_data = sampler.generate_samples()
    training_data["boundary_values"] = create_boundary_fn(
        strike=config["pde"]["strike"],
        basket_weights=config["pde"]["basket_weights"],
        option_type=config["pde"]["option_type"],
    )(training_data["boundary"])

    model = PINNModel(
        input_dim=asset_dim + 1,
        hidden_layers=config["model"]["hidden_layers"],
        hidden_units=config["model"]["hidden_units"],
        output_dim=config["model"].get("output_dim", 1),
        activation=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    pde_module = BlackScholesPDE(
        asset_dim=asset_dim,
        risk_free_rate=config["pde"]["risk_free_rate"],
        sigma=config["pde"]["sigma"],
        corr_matrix=corr_matrix,
        device=model.device,
    )

    payoff_fn = create_payoff_fn(
        strike=config["pde"]["strike"],
        basket_weights=config["pde"]["basket_weights"],
        option_type=config["pde"]["option_type"],
    )

    trainer = PINNTrainer(
        model=model,
        pde_module=pde_module,
        payoff_fn=payoff_fn,
        boundary_fn=create_boundary_fn(
            strike=config["pde"]["strike"],
            basket_weights=config["pde"]["basket_weights"],
            option_type=config["pde"]["option_type"],
        ),
        lambda_interior=config["training"]["lambda_interior"],
        lambda_boundary=config["training"]["lambda_boundary"],
        lambda_terminal=config["training"]["lambda_terminal"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        epochs=config["training"]["epochs"],
        use_lbfgs=config["training"].get("use_lbfgs", False),
        patience=config["training"].get("patience", 200),
        log_dir=str(paths["logs"]),
        checkpoint_dir=str(paths["checkpoints"]),
        mixed_precision=torch.cuda.is_available(),
        device=model.device,
    )

    if resume_checkpoint is not None:
        trainer.load_checkpoint(resume_checkpoint)

    history = trainer.train(training_data)
    trainer.save_checkpoint(str(paths["checkpoints"] / "final_checkpoint.pt"))

    greeks_engine = GreeksEngine(model, device=model.device)
    sample_points = torch.tensor(training_data["terminal"][:1000].cpu().numpy(), device=model.device)
    delta = greeks_engine.compute_delta(sample_points)
    gamma = greeks_engine.compute_gamma(sample_points)

    pricer = MonteCarloPricer(
        asset_dim=asset_dim,
        risk_free_rate=config["pde"]["risk_free_rate"],
        mu=config["pde"].get("mu", [0.05] * asset_dim),
        sigma=config["pde"]["sigma"],
        corr_matrix=corr_matrix,
        seed=config.get("seed", 42),
        device=model.device,
    )
    mc_result = pricer.price_basket_option(
        S0=config["data"].get("initial_prices", [1.0] * asset_dim),
        strike=config["pde"]["strike"],
        maturity=config["pde"]["time_horizon"],
        basket_weights=config["pde"]["basket_weights"],
        option_type=config["pde"]["option_type"],
        steps=config["monte_carlo"]["steps"],
        n_paths=config["monte_carlo"]["n_paths"],
        antithetic=True,
        confidence_level=config["monte_carlo"].get("confidence_level", 0.95),
    )

    logger.info("Monte Carlo price: %.6f", mc_result["price"])
    benchmark_summary = {
        "price": mc_result["price"],
        "std_error": mc_result["std_error"],
        "confidence_interval": mc_result["confidence_interval"],
        "runtime": mc_result["runtime"],
    }
    with open(paths["artifacts"] / "benchmark.json", "w", encoding="utf-8") as fp:
        json.dump({"monte_carlo": benchmark_summary}, fp, indent=2)

    visualizer = Visualizer()
    visualizer.plot_losses(
        train_losses={"total_loss": history["total_loss"], "pde_loss": history["pde_loss"], "boundary_loss": history["boundary_loss"], "terminal_loss": history["terminal_loss"]},
        title="PINN Training Loss",
        save_path=str(paths["figures"] / "loss_curves.png"),
    )

    if asset_dim >= 2:
        surface = evaluate_surface(model, asset_dim)
        surface_points = np.zeros((surface["x"].size, asset_dim + 1), dtype=np.float32)
        surface_points[:, 0] = surface["x"].ravel()
        surface_points[:, 1] = surface["y"].ravel()
        surface_points[:, -1] = 0.5
        surface_tensor = torch.from_numpy(surface_points).to(model.device)
        surface_delta = greeks_engine.compute_delta(surface_tensor).cpu().numpy()[:, 0].reshape(surface["x"].shape)

        visualizer.plot_price_surface(
            x_grid=surface["x"],
            y_grid=surface["y"],
            surface=surface["z"],
            save_path=str(paths["figures"] / "price_surface.png"),
        )
        visualizer.plot_delta_surface(
            x_grid=surface["x"],
            y_grid=surface["y"],
            delta_surface=surface_delta,
            save_path=str(paths["figures"] / "delta_surface.png"),
        )

    visualizer.plot_gamma_heatmap(
        gamma_matrix=gamma.mean(dim=0).cpu().numpy(),
        title="Average Gamma Heatmap",
        save_path=str(paths["figures"] / "gamma_heatmap.png"),
    )

    x_line = np.linspace(0.0, 1.0, 50)
    if asset_dim >= 2:
        extra_dims = np.full((x_line.size, asset_dim - 2), 0.5, dtype=np.float32)
        x_eval = np.hstack([x_line[:, None], x_line[:, None], extra_dims, np.full((x_line.size, 1), 0.5, dtype=np.float32)])
    else:
        x_eval = np.stack([x_line, np.full_like(x_line, 0.5)], axis=1)
    x_tensor = torch.from_numpy(x_eval.astype(np.float32)).to(model.device)
    with torch.no_grad():
        pinn_values = model(x_tensor).cpu().numpy().flatten()

    visualizer.compare_models(
        x=x_line,
        pinn_values=pinn_values,
        benchmark_values=np.full_like(x_line, mc_result["price"]),
        title="PINN vs Monte Carlo Benchmark",
        save_path=str(paths["figures"] / "pinn_vs_mc.png"),
    )

    logger.info("Experiment complete. Artifacts saved to %s", paths["root"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PINN option pricing experiments.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_experiment(config, resume_checkpoint=args.resume)


if __name__ == "__main__":
    main()
    
