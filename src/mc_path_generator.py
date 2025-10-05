#!/usr/bin/env python3
"""
Monte Carlo Path Generator for Asset Returns

Generates multivariate Gaussian return paths at asset level, preserving correlations.
Enables multiple portfolios to backtest on identical market scenarios.

Design principles:
- Simple, clean interface
- Asset-level returns (not portfolio-level)
- Reusable paths across multiple portfolios
- Proper frequency scaling for period-level simulation
- Cacheable with pickle for reproducibility
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
import pickle
from pathlib import Path


class MCPathGenerator:
    """
    Generate multivariate Gaussian Monte Carlo paths for asset returns.

    Key features:
    - Preserves asset correlations via covariance matrix
    - Supports any time frequency (daily, weekly, biweekly, monthly, annual)
    - Generates N simulations × M periods × K assets
    - Reusable paths for comparing multiple portfolio strategies

    Example:
    --------
    >>> mean_returns = np.array([0.10, 0.06])  # Annualized
    >>> cov_matrix = np.array([[0.04, 0.01], [0.01, 0.02]])  # Annualized
    >>> generator = MCPathGenerator(['SPY', 'AGG'], mean_returns, cov_matrix, seed=42)
    >>>
    >>> # Generate 1000 simulations, 10 years × 26 biweekly periods
    >>> paths = generator.generate_paths(num_simulations=1000, total_periods=260,
    ...                                   periods_per_year=26)
    >>> # Shape: (1000, 260, 2) = (sims, periods, assets)
    >>>
    >>> # Get specific path as DataFrame for Portfolio.ingest_simulated_data()
    >>> path_df = generator.get_path_dataframe(simulation_idx=0,
    ...                                         start_date='2025-01-01',
    ...                                         frequency='biweekly')
    """

    def __init__(self,
                 tickers: List[str],
                 mean_returns: np.ndarray,
                 cov_matrix: np.ndarray,
                 seed: int = 42):
        """
        Initialize MC path generator with asset parameters.

        Parameters:
        -----------
        tickers : List[str]
            Asset ticker symbols
        mean_returns : np.ndarray
            Annualized mean returns per asset (length = num_assets)
        cov_matrix : np.ndarray
            Annualized covariance matrix (num_assets × num_assets)
        seed : int
            Random seed for reproducibility
        """
        self.tickers = tickers
        self.num_assets = len(tickers)
        self.mean_returns = mean_returns  # Annualized
        self.cov_matrix = cov_matrix      # Annualized
        self.seed = seed

        # Validation
        if len(mean_returns) != self.num_assets:
            raise ValueError(f"mean_returns length ({len(mean_returns)}) must match tickers ({self.num_assets})")
        if cov_matrix.shape != (self.num_assets, self.num_assets):
            raise ValueError(f"cov_matrix shape {cov_matrix.shape} must be ({self.num_assets}, {self.num_assets})")

        # Storage for generated paths
        self.paths: Optional[np.ndarray] = None  # Will be (num_sims, periods, assets)
        self.periods_per_year: Optional[int] = None
        self.num_simulations: Optional[int] = None
        self.total_periods: Optional[int] = None

        # Storage for time-varying parameters (optional)
        self.time_varying_mean: Optional[pd.DataFrame] = None
        self.time_varying_cov: Optional[dict] = None  # Dict of {date: cov_matrix}

    def generate_paths(self,
                      num_simulations: int,
                      total_periods: int,
                      periods_per_year: int = 1) -> np.ndarray:
        """
        Generate Monte Carlo return paths using multivariate Gaussian sampling.

        Parameters:
        -----------
        num_simulations : int
            Number of simulation paths to generate
        total_periods : int
            Total number of periods to simulate (e.g., 260 = 10 years × 26 biweekly)
        periods_per_year : int
            Number of periods per year (1=annual, 12=monthly, 26=biweekly, 52=weekly, 252=daily)
            Used to scale annualized parameters to period frequency

        Returns:
        --------
        np.ndarray
            Shape: (num_simulations, total_periods, num_assets)
            Asset returns for each period in each simulation

        Notes:
        ------
        - Returns are scaled: period_mean = annual_mean / periods_per_year
        - Covariance is scaled: period_cov = annual_cov / periods_per_year
        - This assumes i.i.d. normal returns (standard geometric Brownian motion)
        """
        np.random.seed(self.seed)

        # Scale annualized parameters to period frequency
        period_mean = self.mean_returns / periods_per_year
        period_cov = self.cov_matrix / periods_per_year

        # Generate all paths at once: (num_sims * total_periods, num_assets)
        total_samples = num_simulations * total_periods
        all_returns = np.random.multivariate_normal(
            mean=period_mean,
            cov=period_cov,
            size=total_samples
        )

        # Reshape to (num_simulations, total_periods, num_assets)
        paths = all_returns.reshape(num_simulations, total_periods, self.num_assets)

        # Store metadata
        self.paths = paths
        self.num_simulations = num_simulations
        self.total_periods = total_periods
        self.periods_per_year = periods_per_year

        return paths

    def generate_lifecycle_paths(self,
                                num_simulations: int,
                                accumulation_years: int,
                                accumulation_periods_per_year: int,
                                decumulation_years: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate CONTINUOUS lifecycle paths spanning accumulation and decumulation.

        The paths are CONTINUOUS - decumulation continues from last accumulation period.
        Handles frequency change: accumulation (biweekly) → decumulation (annual).

        Strategy:
        ---------
        1. Generate ONE continuous path at accumulation frequency
        2. Split into accumulation and decumulation portions
        3. For accumulation: use biweekly returns as-is
        4. For decumulation: aggregate biweekly returns to annual

        Parameters:
        -----------
        num_simulations : int
            Number of simulation paths
        accumulation_years : int
            Years in accumulation phase
        accumulation_periods_per_year : int
            Contribution frequency (e.g., 26 for biweekly)
        decumulation_years : int
            Years in decumulation phase

        Returns:
        --------
        tuple: (accumulation_paths, decumulation_paths)
            - accumulation_paths: (num_sims, acc_periods, num_assets)
            - decumulation_paths: (num_sims, dec_years, num_assets)
              Annual returns computed by compounding sub-annual periods

        Example:
        --------
        >>> # 9 years accumulation (biweekly) + 30 years decumulation (annual)
        >>> acc_paths, dec_paths = generator.generate_lifecycle_paths(
        ...     num_simulations=1000,
        ...     accumulation_years=9,
        ...     accumulation_periods_per_year=26,  # Biweekly
        ...     decumulation_years=30
        ... )
        >>> acc_paths.shape  # (1000, 234, 4) = 1000 sims × 9*26 periods × 4 assets
        >>> dec_paths.shape  # (1000, 30, 4) = 1000 sims × 30 years × 4 assets
        """
        np.random.seed(self.seed)

        # Calculate total periods needed at accumulation frequency
        acc_periods = accumulation_years * accumulation_periods_per_year
        dec_periods = decumulation_years * accumulation_periods_per_year  # Will aggregate later
        total_periods = acc_periods + dec_periods

        # Generate ONE continuous path at accumulation frequency
        period_mean = self.mean_returns / accumulation_periods_per_year
        period_cov = self.cov_matrix / accumulation_periods_per_year

        # Sample all periods at once for all simulations
        total_samples = num_simulations * total_periods
        all_returns = np.random.multivariate_normal(
            mean=period_mean,
            cov=period_cov,
            size=total_samples
        )

        # Reshape to (num_sims, total_periods, num_assets)
        continuous_paths = all_returns.reshape(num_simulations, total_periods, self.num_assets)

        # Split into accumulation and decumulation portions
        accumulation_paths = continuous_paths[:, :acc_periods, :]

        # Aggregate decumulation portion to annual returns
        # Input:  (num_sims, dec_periods, num_assets) where dec_periods = dec_years * periods_per_year
        # Output: (num_sims, dec_years, num_assets) with annual returns
        decumulation_portion = continuous_paths[:, acc_periods:, :]
        decumulation_paths = np.zeros((num_simulations, decumulation_years, self.num_assets))

        for sim in range(num_simulations):
            for year in range(decumulation_years):
                # Get periods for this year
                start_period = year * accumulation_periods_per_year
                end_period = (year + 1) * accumulation_periods_per_year

                # Get returns for this year's periods: (periods_per_year, num_assets)
                year_returns = decumulation_portion[sim, start_period:end_period, :]

                # Compound to get annual return for each asset
                # annual_return = (1+r1)*(1+r2)*...*(1+rN) - 1
                for asset in range(self.num_assets):
                    cumulative_return = np.prod(1 + year_returns[:, asset])
                    decumulation_paths[sim, year, asset] = cumulative_return - 1

        # Store metadata
        self.paths = continuous_paths
        self.num_simulations = num_simulations
        self.total_periods = total_periods
        self.periods_per_year = accumulation_periods_per_year

        print(f"  Generated continuous lifecycle paths:")
        print(f"    Accumulation: {accumulation_paths.shape} (at {accumulation_periods_per_year}/year frequency)")
        print(f"    Decumulation: {decumulation_paths.shape} (annual, aggregated from sub-annual)")
        print(f"    ✓ Paths are CONTINUOUS - dec starts from last acc period")

        return accumulation_paths, decumulation_paths

    def set_time_varying_parameters(self,
                                    mean_returns_ts: pd.DataFrame,
                                    cov_matrices_ts: Optional[pd.DataFrame] = None) -> None:
        """
        Set time-varying mean returns and covariance matrices.

        Enables regime-switching, adaptive estimation, or any time-dependent parameters.

        Parameters:
        -----------
        mean_returns_ts : pd.DataFrame
            Time series of annualized mean returns
            Index: DatetimeIndex
            Columns: Asset tickers
            Each row contains the mean returns to use for that period

        cov_matrices_ts : pd.DataFrame, optional
            Time series of covariance matrix elements (flattened)
            Index: DatetimeIndex (must match mean_returns_ts)
            Columns: Flattened covariance elements (e.g., 'SPY_SPY', 'SPY_AGG', 'AGG_AGG')
            If None, uses constant self.cov_matrix for all periods

        Example:
        --------
        >>> # Create time-varying means (e.g., regime switching)
        >>> dates = pd.date_range('2025-01-01', periods=1000, freq='D')
        >>> means = pd.DataFrame({
        ...     'SPY': [0.10 if i < 500 else 0.05 for i in range(1000)],  # Bull → Bear
        ...     'AGG': [0.04] * 1000
        ... }, index=dates)
        >>>
        >>> generator.set_time_varying_parameters(means)
        >>> paths = generator.generate_paths_time_varying(
        ...     num_simulations=1000,
        ...     start_date='2025-01-01',
        ...     total_periods=1000,
        ...     periods_per_year=252
        ... )
        """
        # Validate mean returns
        if not isinstance(mean_returns_ts.index, pd.DatetimeIndex):
            raise ValueError("mean_returns_ts index must be DatetimeIndex")

        if list(mean_returns_ts.columns) != self.tickers:
            raise ValueError(f"mean_returns_ts columns {list(mean_returns_ts.columns)} "
                           f"must match tickers {self.tickers}")

        self.time_varying_mean = mean_returns_ts

        # Process covariance matrices
        if cov_matrices_ts is not None:
            if not isinstance(cov_matrices_ts.index, pd.DatetimeIndex):
                raise ValueError("cov_matrices_ts index must be DatetimeIndex")

            # Reconstruct covariance matrices from flattened representation
            self.time_varying_cov = {}
            for date, row in cov_matrices_ts.iterrows():
                # Reconstruct matrix from flattened elements
                cov_matrix = np.zeros((self.num_assets, self.num_assets))
                idx = 0
                for i in range(self.num_assets):
                    for j in range(i, self.num_assets):
                        # Upper triangular + symmetric
                        col_name = f"{self.tickers[i]}_{self.tickers[j]}"
                        if col_name in row:
                            cov_matrix[i, j] = row[col_name]
                            cov_matrix[j, i] = row[col_name]  # Symmetric

                self.time_varying_cov[date] = cov_matrix
        else:
            # Use constant covariance
            self.time_varying_cov = None

        print(f"Set time-varying parameters:")
        print(f"  Mean returns: {len(mean_returns_ts)} periods")
        print(f"  Covariance: {'time-varying' if cov_matrices_ts is not None else 'constant'}")

    def generate_paths_time_varying(self,
                                    num_simulations: int,
                                    start_date: str,
                                    total_periods: int,
                                    periods_per_year: int = 1,
                                    frequency: str = 'D') -> np.ndarray:
        """
        Generate MC paths with TIME-VARYING mean returns and covariance matrices.

        Uses parameters from set_time_varying_parameters() to sample from
        different distributions at different times (e.g., regime switching).

        Parameters:
        -----------
        num_simulations : int
            Number of simulation paths
        start_date : str
            Starting date (format: 'YYYY-MM-DD')
        total_periods : int
            Total number of periods to simulate
        periods_per_year : int
            Number of periods per year (for scaling annualized params)
        frequency : str
            Pandas frequency for date generation ('D', 'W', 'M', 'biweekly', etc.)

        Returns:
        --------
        np.ndarray
            Shape: (num_simulations, total_periods, num_assets)
            Asset returns sampled from time-varying distributions

        Raises:
        -------
        ValueError
            If set_time_varying_parameters() hasn't been called

        Example:
        --------
        >>> # First set time-varying parameters
        >>> generator.set_time_varying_parameters(mean_ts, cov_ts)
        >>>
        >>> # Then generate paths
        >>> paths = generator.generate_paths_time_varying(
        ...     num_simulations=1000,
        ...     start_date='2025-01-01',
        ...     total_periods=1000,
        ...     periods_per_year=252,
        ...     frequency='D'
        ... )
        """
        if self.time_varying_mean is None:
            raise ValueError("Must call set_time_varying_parameters() first")

        np.random.seed(self.seed)

        # Generate date sequence
        start = pd.Timestamp(start_date)
        if frequency == 'biweekly':
            dates = [start + timedelta(days=14*i) for i in range(total_periods)]
        else:
            dates = pd.date_range(start=start, periods=total_periods, freq=frequency)

        # Initialize paths
        paths = np.zeros((num_simulations, total_periods, self.num_assets))

        # For each period, sample from the appropriate distribution
        for period_idx, date in enumerate(dates):
            # Get parameters for this date (use nearest available date)
            nearest_date_idx = self.time_varying_mean.index.get_indexer([date], method='nearest')[0]
            nearest_date = self.time_varying_mean.index[nearest_date_idx]

            # Get mean returns for this period (annualized)
            period_mean_annual = self.time_varying_mean.iloc[nearest_date_idx].values

            # Get covariance for this period (annualized)
            if self.time_varying_cov is not None:
                # Find nearest date in covariance dict
                cov_date = min(self.time_varying_cov.keys(), key=lambda d: abs((d - date).total_seconds()))
                period_cov_annual = self.time_varying_cov[cov_date]
            else:
                period_cov_annual = self.cov_matrix

            # Scale to period frequency
            period_mean = period_mean_annual / periods_per_year
            period_cov = period_cov_annual / periods_per_year

            # Sample for all simulations at this period
            period_returns = np.random.multivariate_normal(
                mean=period_mean,
                cov=period_cov,
                size=num_simulations
            )

            paths[:, period_idx, :] = period_returns

        # Store metadata
        self.paths = paths
        self.num_simulations = num_simulations
        self.total_periods = total_periods
        self.periods_per_year = periods_per_year

        print(f"Generated time-varying paths:")
        print(f"  Shape: {paths.shape}")
        print(f"  Periods: {total_periods} ({dates[0]} to {dates[-1]})")
        print(f"  Parameters varied across {len(self.time_varying_mean)} distinct periods")

        return paths

    def get_path_dataframe(self,
                          simulation_idx: int,
                          start_date: str,
                          frequency: str = 'D') -> pd.DataFrame:
        """
        Extract a single simulation path as a pandas DataFrame.

        Useful for feeding into Portfolio.ingest_simulated_data().

        Parameters:
        -----------
        simulation_idx : int
            Which simulation path to extract (0 to num_simulations-1)
        start_date : str
            Starting date for date index (format: 'YYYY-MM-DD')
        frequency : str
            Pandas frequency string for date index ('D', 'W', 'M', 'Q', 'Y')
            Or custom: 'biweekly' = 14 days

        Returns:
        --------
        pd.DataFrame
            Shape: (total_periods, num_assets)
            Columns: ticker symbols
            Index: DatetimeIndex with specified frequency

        Example:
        --------
        >>> path_df = generator.get_path_dataframe(0, '2025-01-01', 'biweekly')
        >>> portfolio.ingest_simulated_data(path_df)
        """
        if self.paths is None:
            raise ValueError("No paths generated yet. Call generate_paths() first.")
        if simulation_idx >= self.num_simulations:
            raise ValueError(f"simulation_idx {simulation_idx} >= num_simulations {self.num_simulations}")

        # Extract path
        path_returns = self.paths[simulation_idx, :, :]  # (total_periods, num_assets)

        # Create date index
        start = pd.Timestamp(start_date)

        if frequency == 'biweekly':
            # Custom: 14-day intervals
            dates = [start + timedelta(days=14*i) for i in range(self.total_periods)]
        else:
            # Use pandas frequency
            dates = pd.date_range(start=start, periods=self.total_periods, freq=frequency)

        # Create DataFrame
        df = pd.DataFrame(
            path_returns,
            index=dates,
            columns=self.tickers
        )

        return df

    def get_portfolio_returns(self,
                             simulation_idx: int,
                             weights: np.ndarray) -> np.ndarray:
        """
        Calculate portfolio-level returns for a specific simulation.

        Parameters:
        -----------
        simulation_idx : int
            Which simulation path to use
        weights : np.ndarray
            Portfolio weights (length = num_assets, must sum to 1)

        Returns:
        --------
        np.ndarray
            Portfolio returns for each period (length = total_periods)
        """
        if self.paths is None:
            raise ValueError("No paths generated yet. Call generate_paths() first.")
        if len(weights) != self.num_assets:
            raise ValueError(f"weights length ({len(weights)}) must match num_assets ({self.num_assets})")

        # Get asset returns for this simulation: (total_periods, num_assets)
        asset_returns = self.paths[simulation_idx, :, :]

        # Calculate weighted portfolio return for each period
        portfolio_returns = np.dot(asset_returns, weights)  # (total_periods,)

        return portfolio_returns

    def save_paths(self, filepath: str) -> None:
        """
        Save generated paths to disk using pickle.

        Parameters:
        -----------
        filepath : str
            Path to save file (e.g., 'mc_paths_1000x260.pkl')
        """
        if self.paths is None:
            raise ValueError("No paths to save. Call generate_paths() first.")

        data = {
            'paths': self.paths,
            'tickers': self.tickers,
            'mean_returns': self.mean_returns,
            'cov_matrix': self.cov_matrix,
            'seed': self.seed,
            'num_simulations': self.num_simulations,
            'total_periods': self.total_periods,
            'periods_per_year': self.periods_per_year
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"Saved {self.num_simulations} MC paths ({self.total_periods} periods) to {filepath}")

    @classmethod
    def load_paths(cls, filepath: str) -> 'MCPathGenerator':
        """
        Load previously generated paths from disk.

        Parameters:
        -----------
        filepath : str
            Path to saved file

        Returns:
        --------
        MCPathGenerator
            Generator instance with loaded paths
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Recreate instance
        generator = cls(
            tickers=data['tickers'],
            mean_returns=data['mean_returns'],
            cov_matrix=data['cov_matrix'],
            seed=data['seed']
        )

        # Restore paths
        generator.paths = data['paths']
        generator.num_simulations = data['num_simulations']
        generator.total_periods = data['total_periods']
        generator.periods_per_year = data['periods_per_year']

        print(f"Loaded {generator.num_simulations} MC paths ({generator.total_periods} periods) from {filepath}")
        return generator

    def get_summary_statistics(self) -> dict:
        """
        Calculate summary statistics of generated paths.

        Returns:
        --------
        dict
            Statistics including mean, std, correlations across all simulations
        """
        if self.paths is None:
            raise ValueError("No paths generated yet. Call generate_paths() first.")

        # Flatten all paths: (num_sims * total_periods, num_assets)
        all_returns = self.paths.reshape(-1, self.num_assets)

        # Calculate statistics
        empirical_mean = np.mean(all_returns, axis=0) * self.periods_per_year  # Annualize
        empirical_cov = np.cov(all_returns.T) * self.periods_per_year  # Annualize
        empirical_corr = np.corrcoef(all_returns.T)

        return {
            'num_simulations': self.num_simulations,
            'total_periods': self.total_periods,
            'periods_per_year': self.periods_per_year,
            'empirical_mean_returns': empirical_mean,
            'empirical_cov_matrix': empirical_cov,
            'empirical_correlation': empirical_corr,
            'theoretical_mean_returns': self.mean_returns,
            'theoretical_cov_matrix': self.cov_matrix,
            'mean_error': np.abs(empirical_mean - self.mean_returns),
            'cov_error': np.abs(empirical_cov - self.cov_matrix)
        }
