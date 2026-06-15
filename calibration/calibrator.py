"""Calibration utilities for SDE parameters."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any
import os

class SDECalibrator:
    """Calibrator for multi-asset Geometric Brownian Motion parameters."""

    def __init__(self, file_path: str, assets: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None) -> None:
        """Initialize the SDE calibrator.

        Args:
            file_path: Path to the CSV file containing OHLC data.
            assets: List of asset column names (adjusted close prices).
            start_date: Optional start date for data filtering (YYYY-MM-DD).
            end_date: Optional end date for data filtering (YYYY-MM-DD).
        """
        self.file_path = file_path
        self.assets = assets
        self.start_date = start_date
        self.end_date = end_date
        self.data: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Load OHLC adjusted close prices from CSV.

        Returns:
            DataFrame with adjusted close prices for specified assets.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If required columns are missing or data is invalid.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        try:
            self.data = pd.read_csv(self.file_path, index_col=0, parse_dates=True)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")

        # Filter by date
        if self.start_date:
            self.data = self.data[self.data.index >= pd.to_datetime(self.start_date)]
        if self.end_date:
            self.data = self.data[self.data.index <= pd.to_datetime(self.end_date)]

        # Check for required columns
        missing_assets = [asset for asset in self.assets if asset not in self.data.columns]
        if missing_assets:
            raise ValueError(f"Missing asset columns in data: {missing_assets}")

        self.data = self.data[self.assets].dropna()

        if self.data.empty:
            raise ValueError("No valid data after filtering.")

        return self.data

    def compute_returns(self) -> pd.DataFrame:
        """Compute log returns for all assets.

        Returns:
            DataFrame of log returns.
        """
        if self.data is None:
            self.load_data()

        self.returns = np.log(self.data / self.data.shift(1)).dropna()
        return self.returns

    def estimate_drift(self) -> pd.Series:
        """Estimate annualized drift (mean returns) for each asset.

        Returns:
            Series of annualized drift values.
        """
        if self.returns is None:
            self.compute_returns()

        return self.returns.mean() * 252  # Annualized assuming 252 trading days

    def estimate_volatility(self) -> pd.Series:
        """Estimate annualized volatility for each asset.

        Returns:
            Series of annualized volatility values.
        """
        if self.returns is None:
            self.compute_returns()

        return self.returns.std() * np.sqrt(252)

    def estimate_covariance(self) -> pd.DataFrame:
        """Estimate annualized covariance matrix.

        Returns:
            DataFrame of annualized covariance matrix.
        """
        if self.returns is None:
            self.compute_returns()

        return self.returns.cov() * 252

    def estimate_correlation(self) -> pd.DataFrame:
        """Estimate correlation matrix.

        Returns:
            DataFrame of correlation matrix.
        """
        cov = self.estimate_covariance()
        vol = self.estimate_volatility()
        return cov.div(vol, axis=0).div(vol, axis=1)

    def compute_rolling_volatility(self, window: int = 30) -> pd.DataFrame:
        """Compute rolling annualized volatility.

        Args:
            window: Rolling window size in days.

        Returns:
            DataFrame of rolling volatilities.
        """
        if self.returns is None:
            self.compute_returns()

        return self.returns.rolling(window).std() * np.sqrt(252)

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of all estimated parameters.

        Returns:
            Dictionary containing drift, volatility, covariance, correlation, and rolling volatility.
        """
        drift = self.estimate_drift()
        vol = self.estimate_volatility()
        cov = self.estimate_covariance()
        corr = self.estimate_correlation()
        rolling_vol = self.compute_rolling_volatility()

        return {
            'drift': drift,
            'volatility': vol,
            'covariance': cov,
            'correlation': corr,
            'rolling_volatility': rolling_vol
        }

    def plot_volatility(self, save_path: Optional[str] = None) -> None:
        """Plot annualized volatility for each asset.

        Args:
            save_path: Optional path to save the plot.
        """
        vol = self.estimate_volatility()
        plt.figure(figsize=(10, 6))
        vol.plot(kind='bar', color='skyblue')
        plt.title('Annualized Volatility')
        plt.ylabel('Volatility')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_correlation_heatmap(self, save_path: Optional[str] = None) -> None:
        """Plot correlation matrix heatmap.

        Args:
            save_path: Optional path to save the plot.
        """
        corr = self.estimate_correlation()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
        plt.title('Asset Correlation Matrix')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_rolling_volatility(self, window: int = 30, save_path: Optional[str] = None) -> None:
        """Plot rolling volatility over time.

        Args:
            window: Rolling window size.
            save_path: Optional path to save the plot.
        """
        rolling_vol = self.compute_rolling_volatility(window)
        plt.figure(figsize=(12, 8))
        for asset in self.assets:
            plt.plot(rolling_vol.index, rolling_vol[asset], label=asset)
        plt.title(f'Rolling Volatility (Window: {window} days)')
        plt.ylabel('Volatility')
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()