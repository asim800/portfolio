#!/usr/bin/env python3
"""
Monte Carlo Path Generator - SIMPLIFIED

ONE public method: generate_paths()
- All multivariate Gaussian sampling happens in ONE place
- Helper functions just parse and prepare parameters
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, List, Dict
from datetime import datetime, timedelta

import matplotlib.pyplot as plt

import ipdb


class MCPathGenerator:
    """
    Generate Monte Carlo paths using multivariate Gaussian sampling.

    ONE public method: generate_paths()
    - Handles constant or time-varying parameters
    - Handles regular paths or lifecycle paths
    - All sampling logic in ONE place
    """

    def __init__(self,
                 tickers: list,
                 mean_returns: Optional[np.ndarray] = None,
                 cov_matrix: Optional[np.ndarray] = None,
                 seed: Optional[int] = None):
        """
        Initialize with optional default constant parameters.

        Parameters:
        -----------
        tickers : list
            Asset ticker symbols
        mean_returns : np.ndarray, optional
            Default annual mean returns (length = num_assets)
            If None, must be provided to generate_paths()
        cov_matrix : np.ndarray, optional
            Default annual covariance matrix (num_assets x num_assets)
            If None, must be provided to generate_paths()
        seed : int, optional
            Random seed for reproducibility
        """
        self.tickers = tickers
        self.num_assets = len(tickers)
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.seed = seed
        self.cholesky = None
        

        # Store generated paths
        self.paths = None
        self.num_simulations = None
        self.total_periods = None
        self.periods_per_year = None

        self.lamda = 0 # 0.4
        self.delta = 0.1
        self.mu = -0.8

    def generate_paths(self,
                      num_simulations: int,
                      total_periods: Optional[int] = None,
                      periods_per_year: int = 1,
                      start_date: Optional[str] = None,
                      frequency: str = 'D',
                      mean_returns: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                      cov_matrices: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                      # Lifecycle parameters
                      accumulation_years: Optional[int] = None,
                      decumulation_years: Optional[int] = None,
                      # Time-varying parameter reindexing
                      reindex_method: str = 'ffill') -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        ONE method to generate ALL Monte Carlo paths.

        ALL multivariate Gaussian sampling happens here in ONE place.

        Basic usage (constant parameters):
            paths = gen.generate_paths(num_simulations=1000, total_periods=260, periods_per_year=26)

        Time-varying parameters:
            paths = gen.generate_paths(num_simulations=1000, total_periods=260,
                                      periods_per_year=26, mean_returns=df)

        Lifecycle simulation:
            acc, dec = gen.generate_paths(num_simulations=1000, accumulation_years=9,
                                         decumulation_years=30, periods_per_year=26)

        Time-varying with interpolation:
            acc, dec = gen.generate_paths(num_simulations=1000, accumulation_years=9,
                                         decumulation_years=30, periods_per_year=26,
                                         mean_returns=mean_df, reindex_method='interpolate')

        Parameters:
        -----------
        num_simulations : int
            Number of Monte Carlo paths

        total_periods : int, optional
            Total periods to simulate (required if not using lifecycle mode)

        periods_per_year : int
            Frequency (e.g., 26 for biweekly, 12 for monthly, 252 for daily)

        mean_returns : np.ndarray or pd.DataFrame, optional
            - None: Use default self.mean_returns
            - np.ndarray: Constant mean returns
            - pd.DataFrame: Time-varying mean returns (requires start_date)

        cov_matrices : np.ndarray or pd.DataFrame, optional
            - None: Use default self.cov_matrix
            - np.ndarray: Constant covariance
            - pd.DataFrame: Time-varying covariance (requires start_date)

        accumulation_years : int, optional
            If provided, enables lifecycle mode (returns tuple)

        decumulation_years : int, optional
            Required if accumulation_years is provided

        reindex_method : str, optional
            Method for reindexing time-varying parameters to simulation dates.
            - 'ffill' (default): Forward fill - use most recent past value
            - 'interpolate': Linear interpolation between known values
            Only applies when mean_returns or cov_matrices are DataFrames.

        Returns:
        --------
        np.ndarray OR tuple
            - Regular mode: (num_simulations, total_periods, num_assets)
            - Lifecycle mode: (accumulation_paths, decumulation_paths)
                accumulation_paths: (num_sims, acc_periods, num_assets)
                decumulation_paths: (num_sims, dec_years, num_assets)
        """
        # Set random seed
        np.random.seed(self.seed)



        # Determine mode: lifecycle or regular
        is_lifecycle = accumulation_years is not None

        if is_lifecycle:
            if decumulation_years is None:
                raise ValueError("decumulation_years required when using lifecycle mode")
            total_periods = (accumulation_years + decumulation_years) * periods_per_year
        else:
            if total_periods is None:
                raise ValueError("total_periods required when not using lifecycle mode")

        # Store configuration
        self.num_simulations = num_simulations
        self.total_periods = total_periods
        self.periods_per_year = periods_per_year

        # Parse parameters (helper functions just prepare data)
        mean_array, cov_array = self._parse_parameters(
            total_periods, periods_per_year, start_date, frequency,
            mean_returns, cov_matrices, reindex_method
        )


        # ====================================================================
        # SINGLE POINT: ALL MULTIVARIATE GAUSSIAN SAMPLING HAPPENS HERE
        # ====================================================================

        paths = np.zeros((num_simulations, total_periods, self.num_assets))

        for nmc in range(num_simulations):
          rj = self.lamda * (np.exp(self.mu + 0.5 * self.delta **2) - 1)
          # Time-varying: Sample period-by-period with varying parameters
          for period in range(total_periods):
              period_mean = mean_array[period]
              period_cov = cov_array[period]

              # SAMPLE: Multivariate Gaussian for this period
              paths[nmc, period, :] = np.random.multivariate_normal(
                  mean=period_mean,
                  cov=period_cov
              )

              self.cholesky = np.linalg.cholesky(period_cov)
              paths[nmc, period, :] = np.random.standard_normal(
                  size=(self.num_assets,)
              )

              paths[nmc, period, :] = np.dot(self.cholesky, paths[nmc, period, :])

        # ====================================================================
        # END OF SAMPLING - Everything after is just reshaping/splitting
        # ====================================================================
        # zz = paths.sum(1)
        # plt.hist(zz[:,1], 20)
        # plt.plot(zz[:,1])
        # plt.plot(paths[0])
        # Store paths
        self.paths = paths

        # Return based on mode
        if is_lifecycle:
            # Split into accumulation and decumulation (both period-level)
            acc_periods = accumulation_years * periods_per_year
            dec_periods = decumulation_years * periods_per_year

            accumulation_paths = paths[:, :acc_periods, :]
            decumulation_paths = paths[:, acc_periods:acc_periods + dec_periods, :]

            return accumulation_paths, decumulation_paths
        else:
            return paths

    def _parse_parameters(self,
                         total_periods: int,
                         periods_per_year: int,
                         start_date: Optional[str],
                         frequency: str,
                         mean_returns: Optional[Union[np.ndarray, pd.DataFrame]],
                         cov_matrices: Optional[Union[np.ndarray, pd.DataFrame]],
                         reindex_method: str = 'ffill') -> Tuple:
        """
        Helper: Parse and prepare parameters for sampling.

        Parameters:
        -----------
        reindex_method : str
            Method for reindexing time-varying parameters ('ffill' or 'interpolate')

        Returns:
        --------
        tuple: (mean_array, cov_array, is_time_varying)
            For constant: mean_array is 1D, cov_array is 2D
            For time-varying: mean_array is 2D (periods x assets), cov_array is 3D (periods x assets x assets)
        """
        # Determine if time-varying
        is_time_varying = isinstance(mean_returns, pd.DataFrame) or isinstance(cov_matrices, pd.DataFrame)

        if is_time_varying:
            # Time-varying mode
            if start_date is None:
                raise ValueError("start_date required for time-varying parameters")

            # Create date range starting at start_date
            dates = pd.date_range(start=pd.Timestamp(start_date), periods=total_periods, freq=frequency)
            # Ensure dates start exactly at start_date by applying offset
            offset = pd.Timestamp(start_date) - dates[0]
            dates = dates + offset

            # Parse mean returns
            if isinstance(mean_returns, pd.DataFrame):
                mean_array = self._parse_time_varying_mean(mean_returns, dates, periods_per_year, reindex_method)
            else:
                # Constant mean
                const_mean = mean_returns if mean_returns is not None else self.mean_returns
                mean_array = np.tile(const_mean / periods_per_year, (total_periods, 1))

            # Parse covariance
            if isinstance(cov_matrices, pd.DataFrame):
                cov_array = self._parse_time_varying_cov(cov_matrices, dates, periods_per_year, reindex_method)
            else:
                # Constant cov
                const_cov = cov_matrices if cov_matrices is not None else self.cov_matrix
                cov_array = np.tile(const_cov / periods_per_year, (total_periods, 1, 1))
        else:
            # Constant mode - broadcast to all periods
            const_mean = mean_returns if mean_returns is not None else self.mean_returns
            const_cov = cov_matrices if cov_matrices is not None else self.cov_matrix

            # Scale to period and broadcast to all periods
            # mean_array: (total_periods, num_assets)
            # cov_array: (total_periods, num_assets, num_assets)
            period_mean = const_mean / periods_per_year
            period_cov = const_cov / periods_per_year

            mean_array = np.tile(period_mean, (total_periods, 1))
            cov_array = np.tile(period_cov, (total_periods, 1, 1))

        return mean_array, cov_array

    def _parse_time_varying_mean(self,
                                 mean_df: pd.DataFrame,
                                 dates: pd.DatetimeIndex,
                                 periods_per_year: int,
                                 reindex_method: str = 'ffill') -> np.ndarray:
        """
        Parse time-varying mean DataFrame to array using pandas reindex.

        Parameters:
        -----------
        mean_df : pd.DataFrame
            Time-varying mean returns with DatetimeIndex
        dates : pd.DatetimeIndex
            Target dates for simulation
        periods_per_year : int
            Frequency for scaling
        reindex_method : str
            'ffill' (forward fill) or 'interpolate' (linear interpolation)

        Returns:
        --------
        np.ndarray
            Mean array scaled to period frequency, shape (len(dates), num_assets)
        """
        # Validate reindex_method
        if reindex_method not in ['ffill', 'interpolate']:
            raise ValueError(f"reindex_method must be 'ffill' or 'interpolate', got '{reindex_method}'")

        # Reindex using specified method
        if reindex_method == 'ffill':
            # Forward fill - use most recent past value
            reindexed_df = mean_df.reindex(dates, method='ffill')
        else:  # interpolate
            # Linear interpolate between known dates, then forward fill for dates outside range
            reindexed_df = mean_df.reindex(dates)
            reindexed_df.loc[mean_df.index[0]:mean_df.index[-1]] = reindexed_df.loc[mean_df.index[0]:mean_df.index[-1]].interpolate(method='linear')
            reindexed_df = reindexed_df.fillna(method='ffill')
        # Copy column names from mean_df to ensure consistent ticker order
        reindexed_df.columns = mean_df.columns

        # Check for NaN values (indicates dates before first parameter date or other issues)
        if reindexed_df.isna().any().any():
            raise ValueError(
                f"NaN values after reindexing with method='{reindex_method}'. "
                f"Ensure simulation start_date >= first parameter date. "
                f"First parameter date: {mean_df.index.min()}, "
                f"Simulation start: {dates.min()}"
            )

        # Extract values for the tickers and convert to numpy array
        mean_array = reindexed_df[self.tickers].values

        # Scale to period (convert annual returns to period returns)
        mean_array = mean_array / periods_per_year

        return mean_array

    def _parse_time_varying_cov(self,
                                cov_df: pd.DataFrame,
                                dates: pd.DatetimeIndex,
                                periods_per_year: int,
                                reindex_method: str = 'ffill') -> np.ndarray:
        """
        Parse time-varying covariance DataFrame to array using pandas reindex.
        NOTE: start_date must be >= first date in cov_df to avoid NaN

        Parameters:
        -----------
        cov_df : pd.DataFrame
            Time-varying covariance matrices with DatetimeIndex
        dates : pd.DatetimeIndex
            Target dates for simulation
        periods_per_year : int
            Frequency for scaling
        reindex_method : str
            'ffill' (forward fill) or 'interpolate' (linear interpolation)

        Returns:
        --------
        np.ndarray
            Covariance array scaled to period frequency, shape (len(dates), num_assets, num_assets)

        Note:
        -----
        Linear interpolation interpolates element-wise, which may not preserve the
        positive semi-definite property of covariance matrices in all cases. Use with
        caution when interpolating between significantly different covariance structures.
        """
        # Validate reindex_method
        if reindex_method not in ['ffill', 'interpolate']:
            raise ValueError(f"reindex_method must be 'ffill' or 'interpolate', got '{reindex_method}'")

        # ipdb.set_trace()
        reindexed_df = cov_df.reindex(dates, method='ffill')
        # Reindex using specified method
        if reindex_method == 'ffill':
            # Forward fill - use most recent past value
            reindexed_df = cov_df.reindex(dates, method='ffill')
        # else:  # interpolate
        #     # Linear interpolation between known values
        #     # Note: Need to handle edge cases where index errors could occur
        #     # Need special handling since cov_df elements are 2D arrays
        #     reindexed_df = cov_df.copy()
        #     # First reindex to get all dates, which will introduce NaN rows
        #     reindexed_df = reindexed_df.reindex(dates)
        #     # For each NaN row, interpolate the covariance matrix elements
        #     nan_indices = reindexed_df.isna().any(axis=1)
            # for idx in reindexed_df[nan_indices].index:
                # Find nearest non-NaN values before and after
                # prev_idx = reindexed_df.index[reindexed_df.index.get_loc(idx) - 1]
                # next_idx = reindexed_df.index[reindexed_df.index.get_loc(idx) + 1]
                # # Get matrices and weights for interpolation
                # prev_cov = reindexed_df.loc[prev_idx, 'cov_matrix']
                # next_cov = reindexed_df.loc[next_idx, 'cov_matrix']
                # # Calculate position between 0 and 1
                # weight = (idx - prev_idx) / (next_idx - prev_idx)
                # # Linear interpolation of matrices
                # interp_cov = prev_cov * (1 - weight) + next_cov * weight
                # reindexed_df.at[idx, 'cov_matrix'] = interp_cov
        # ipdb.set_trace()
        # Check for NaN values
        if reindexed_df.isna().any().any():
            raise ValueError(
                f"NaN values after reindexing with method='{reindex_method}'. "
                f"Ensure simulation start_date >= first parameter date. "
                f"First parameter date: {cov_df.index.min()}, "
                f"Simulation start: {dates.min()}"
            )

        cov_array = np.zeros((len(dates), self.num_assets, self.num_assets))

        # Check format: list-of-arrays or flattened
        if 'cov_matrix' in cov_df.columns:
            # List-of-arrays format - extract matrices directly
            for i, row in enumerate(reindexed_df.itertuples()):
                annual_cov = row.cov_matrix
                cov_array[i] = annual_cov / periods_per_year
        else:
            # Flattened format - reconstruct matrices from column pairs
            for i, row in enumerate(reindexed_df.itertuples()):
                annual_cov = np.zeros((self.num_assets, self.num_assets))
                for j in range(self.num_assets):
                    for k in range(j, self.num_assets):
                        col_name = f"{self.tickers[j]}_{self.tickers[k]}"
                        annual_cov[j, k] = getattr(row, col_name)
                        annual_cov[k, j] = getattr(row, col_name)

                cov_array[i] = annual_cov / periods_per_year

        return cov_array

    @staticmethod
    def calculate_portfolio_return(weights: np.ndarray, asset_returns: np.ndarray) -> np.ndarray:
        """
        Calculate portfolio return from weighted asset returns.

        This is the fundamental portfolio calculation used in both accumulation
        and decumulation simulations.

        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights (length = num_assets), should sum to 1
        asset_returns : np.ndarray
            Asset returns for one or more periods
            Shape: (num_assets,) for single period
                   (num_simulations, num_assets) for vectorized calculation

        Returns:
        --------
        np.ndarray
            Portfolio return(s) as weighted average of asset returns
            Shape matches input: scalar for single period, array for multiple simulations

        Example:
        --------
        >>> weights = np.array([0.6, 0.4])  # 60/40 portfolio
        >>> asset_returns = np.array([0.10, 0.05])  # SPY +10%, AGG +5%
        >>> MCPathGenerator.calculate_portfolio_return(weights, asset_returns)
        0.08  # Portfolio return = 0.6*0.10 + 0.4*0.05 = 8%
        """
        return np.dot(asset_returns, weights)

    def get_path_dataframe(self,
                          simulation_idx: int,
                          start_date: str,
                          frequency: str = 'D') -> pd.DataFrame:
        """
        Extract one simulation path as a pandas DataFrame.

        This converts MC simulation output to the format expected by Portfolio.ingest_simulated_data().

        Parameters:
        -----------
        simulation_idx : int
            Which simulation to extract (0-based index)
        start_date : str
            Starting date for the DataFrame index (format: 'YYYY-MM-DD')
        frequency : str
            Pandas frequency code for the date index
            - 'D' = Daily
            - 'W' = Weekly
            - '2W' = Biweekly
            - 'ME' = Month end
            - 'QE' = Quarter end
            - 'YE' = Year end

        Returns:
        --------
        pd.DataFrame
            Returns DataFrame with:
            - Index: DatetimeIndex starting at start_date
            - Columns: Asset tickers
            - Values: Period returns from the simulation
            Shape: (total_periods, num_assets)

        Example:
        --------
        >>> generator = MCPathGenerator(['SPY', 'AGG'], mean, cov, seed=42)
        >>> paths = generator.generate_paths(num_simulations=100, total_periods=260, periods_per_year=26)
        >>> returns_df = generator.get_path_dataframe(sim_idx=0, start_date='2025-01-01', frequency='2W')
        >>> returns_df.shape
        (260, 2)
        >>> returns_df.columns.tolist()
        ['SPY', 'AGG']
        """
        if self.paths is None:
            raise ValueError("No paths generated yet. Call generate_paths() first.")

        if simulation_idx < 0 or simulation_idx >= self.num_simulations:
            raise ValueError(
                f"simulation_idx must be in range [0, {self.num_simulations-1}], "
                f"got {simulation_idx}"
            )

        # Extract the specific simulation: (total_periods, num_assets)
        simulation_returns = self.paths[simulation_idx, :, :]

        # Create date range for index
        dates = pd.date_range(start=start_date, periods=self.total_periods, freq=frequency)

        # Create DataFrame with tickers as columns
        returns_df = pd.DataFrame(
            simulation_returns,
            index=dates,
            columns=self.tickers
        )

        return returns_df

    def get_multiple_path_dataframes(self,
                                    simulation_indices: Optional[List[int]] = None,
                                    start_date: str = None,
                                    frequency: str = 'D') -> Dict[int, pd.DataFrame]:
        """
        Extract multiple simulation paths as DataFrames.

        Useful for running the same portfolio strategy across multiple scenarios.

        Parameters:
        -----------
        simulation_indices : List[int], optional
            List of simulation indices to extract. If None, extracts all simulations.
        start_date : str
            Starting date for the DataFrame index (format: 'YYYY-MM-DD')
        frequency : str
            Pandas frequency code for the date index (default: 'D')

        Returns:
        --------
        Dict[int, pd.DataFrame]
            Dictionary mapping simulation_idx → returns DataFrame

        Example:
        --------
        >>> generator = MCPathGenerator(['SPY', 'AGG'], mean, cov, seed=42)
        >>> paths = generator.generate_paths(num_simulations=100, total_periods=260, periods_per_year=26)
        >>> dfs = generator.get_multiple_path_dataframes(
        ...     simulation_indices=[0, 1, 2],
        ...     start_date='2025-01-01',
        ...     frequency='2W'
        ... )
        >>> len(dfs)
        3
        >>> dfs[0].shape
        (260, 2)
        """
        if self.paths is None:
            raise ValueError("No paths generated yet. Call generate_paths() first.")

        if simulation_indices is None:
            simulation_indices = list(range(self.num_simulations))

        dataframes = {}
        for sim_idx in simulation_indices:
            dataframes[sim_idx] = self.get_path_dataframe(
                simulation_idx=sim_idx,
                start_date=start_date,
                frequency=frequency
            )

        return dataframes

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
            'cov_error': [np.abs(empirical_cov - cov_matrix) for cov_matrix in self.cov_matrix['cov_matrix']]
        }


# ============================================================================
# DEMO: Show the simple API
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("MC PATH GENERATOR - SIMPLIFIED DEMO")
    print("="*80)

    # Setup
    tickers = ['SPY', 'AGG']
    mean_returns = np.array([0.10, 0.04])
    cov_matrix = np.array([[0.04, 0.01], [0.01, 0.02]])

    generator = MCPathGenerator(tickers, mean_returns, cov_matrix, seed=42)

    print("\n1. Simple constant path (3 parameters):")
    paths = generator.generate_paths(
        num_simulations=100,
        total_periods=260,
        periods_per_year=26
    )
    print(f"   Shape: {paths.shape}")

    print("\n2. Lifecycle paths:")
    acc, dec = generator.generate_paths(
        num_simulations=100,
        accumulation_years=9,
        decumulation_years=30,
        periods_per_year=26
    )
    print(f"   Accumulation: {acc.shape}")
    print(f"   Decumulation: {dec.shape}")

    print("\n3. Time-varying mean:")
    dates = pd.date_range('2025-01-01', periods=260, freq='D')
    mean_df = pd.DataFrame({
        'SPY': [0.15 if i < 130 else 0.02 for i in range(260)],
        'AGG': [0.04] * 260
    }, index=dates)

    paths = generator.generate_paths(
        num_simulations=100,
        total_periods=260,
        periods_per_year=252,
        start_date='2025-01-01',
        frequency='D',
        mean_returns=mean_df
    )
    print(f"   Shape: {paths.shape}")
    print(f"   Bull avg: {paths[:, :130, 0].mean():.6f}")
    print(f"   Bear avg: {paths[:, 130:, 0].mean():.6f}")

    print("\n4. Reindex method comparison (ffill vs interpolate):")
    # Create sparse time-varying parameters (only 3 dates over full year)
    sparse_dates = pd.to_datetime(['2025-01-01', '2025-06-01', '2025-12-31'])
    sparse_mean = pd.DataFrame({
        'SPY': [0.08, 0.16, 0.10],  # Low → High → Medium (simulate regime shift)
        'AGG': [0.04, 0.04, 0.04]   # Constant (no change)
    }, index=sparse_dates)

    # Test ffill (step function - abrupt changes)
    paths_ffill = generator.generate_paths(
        num_simulations=1000, total_periods=365, periods_per_year=365,
        start_date='2025-01-01', frequency='D',
        mean_returns=sparse_mean, reindex_method='ffill'
    )

    # Test interpolate (smooth transition)
    paths_interp = generator.generate_paths(
        num_simulations=1000, total_periods=365, periods_per_year=365,
        start_date='2025-01-01', frequency='D',
        mean_returns=sparse_mean, reindex_method='interpolate'
    )

    print(f"   Sparse parameter dates: {sparse_dates.strftime('%Y-%m-%d').tolist()}")
    print(f"   SPY values at those dates: {sparse_mean['SPY'].tolist()}")
    print(f"\n   FFILL method (step function):")
    print(f"     Jan-May avg (days 0-150):  {paths_ffill[:, :150, 0].mean():.6f} (expects ~8%)")
    print(f"     Jun-Dec avg (days 150-365): {paths_ffill[:, 150:, 0].mean():.6f} (expects ~16% then 10%)")
    print(f"\n   INTERPOLATE method (smooth transition):")
    print(f"     Jan-May avg (days 0-150):  {paths_interp[:, :150, 0].mean():.6f} (gradual rise 8%→16%)")
    print(f"     Jun-Dec avg (days 150-365): {paths_interp[:, 150:, 0].mean():.6f} (gradual fall 16%→10%)")

    print("\n" + "="*80)
    print("ONE METHOD - ALL SAMPLING IN ONE PLACE")
    print("="*80)
