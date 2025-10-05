"""
UNIFIED MCPathGenerator - Best of Both Worlds

User's insight: Accept BOTH constant (scalar/array) AND time-varying (DataFrame)
parameters in the SAME method!

Key innovation:
- Simple API: generator.generate_paths(..., mean_returns=0.10)
- Advanced API: generator.generate_paths(..., mean_returns=mean_df)
- ONE method handles both cases
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union
from datetime import timedelta


class MCPathGenerator:
    """
    UNIFIED: Single generate_paths() method accepts BOTH formats.

    Usage:
    ------
    # Simple case - constant parameters (scalars or arrays)
    paths = generator.generate_paths(
        num_simulations=1000,
        total_periods=260,
        periods_per_year=26
    )  # Uses self.mean_returns and self.cov_matrix

    # Advanced case - time-varying parameters (DataFrames)
    paths = generator.generate_paths(
        num_simulations=1000,
        start_date='2025-01-01',
        total_periods=260,
        periods_per_year=26,
        mean_returns=mean_returns_df,  # DataFrame!
        cov_matrices=cov_matrices_df   # DataFrame!
    )
    """

    def __init__(self, tickers: List[str], mean_returns: np.ndarray,
                 cov_matrix: np.ndarray, seed: int = 42):
        self.tickers = tickers
        self.num_assets = len(tickers)
        self.mean_returns = mean_returns  # Default annualized
        self.cov_matrix = cov_matrix      # Default annualized
        self.seed = seed

        # Validation
        if len(mean_returns) != self.num_assets:
            raise ValueError(f"mean_returns length ({len(mean_returns)}) must match tickers ({self.num_assets})")
        if cov_matrix.shape != (self.num_assets, self.num_assets):
            raise ValueError(f"cov_matrix shape {cov_matrix.shape} must be ({self.num_assets}, {self.num_assets})")

        # Storage
        self.paths: Optional[np.ndarray] = None
        self.periods_per_year: Optional[int] = None
        self.num_simulations: Optional[int] = None
        self.total_periods: Optional[int] = None

    def generate_paths(self,
                      num_simulations: int,
                      total_periods: int,
                      periods_per_year: int = 1,
                      start_date: Optional[str] = None,
                      frequency: str = 'D',
                      mean_returns: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                      cov_matrices: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> np.ndarray:
        """
        UNIVERSAL method: Handles BOTH constant and time-varying parameters.

        Parameters:
        -----------
        num_simulations : int
            Number of simulation paths

        total_periods : int
            Total periods to simulate

        periods_per_year : int
            Periods per year (for scaling annualized parameters)

        start_date : str, optional
            Required if using time-varying parameters (DataFrame)
            Format: 'YYYY-MM-DD'

        frequency : str, optional
            Pandas frequency ('D', 'W', 'M', etc.)
            Only used with time-varying parameters

        mean_returns : np.ndarray or pd.DataFrame, optional
            - If None: Use self.mean_returns (constant)
            - If np.ndarray: Use as constant mean returns
            - If pd.DataFrame: Use as time-varying mean returns
                Index: DatetimeIndex
                Columns: Asset tickers

        cov_matrices : np.ndarray or pd.DataFrame, optional
            - If None: Use self.cov_matrix (constant)
            - If np.ndarray: Use as constant covariance
            - If pd.DataFrame: Use as time-varying covariance
                Index: DatetimeIndex
                Columns: Flattened cov elements ('SPY_SPY', 'SPY_AGG', etc.)

        Returns:
        --------
        np.ndarray
            Shape: (num_simulations, total_periods, num_assets)

        Examples:
        ---------
        # Simple case - use default constant parameters
        >>> paths = generator.generate_paths(
        ...     num_simulations=1000,
        ...     total_periods=260,
        ...     periods_per_year=26
        ... )

        # Advanced case - time-varying parameters
        >>> mean_df = pd.DataFrame({'SPY': [...], 'AGG': [...]}, index=dates)
        >>> paths = generator.generate_paths(
        ...     num_simulations=1000,
        ...     start_date='2025-01-01',
        ...     total_periods=260,
        ...     periods_per_year=26,
        ...     mean_returns=mean_df
        ... )
        """
        np.random.seed(self.seed)

        # Determine if time-varying or constant
        is_time_varying = isinstance(mean_returns, pd.DataFrame) or isinstance(cov_matrices, pd.DataFrame)

        if is_time_varying:
            # TIME-VARYING PATH
            return self._generate_time_varying(
                num_simulations, total_periods, periods_per_year,
                start_date, frequency, mean_returns, cov_matrices
            )
        else:
            # CONSTANT PATH (FAST)
            return self._generate_constant(
                num_simulations, total_periods, periods_per_year,
                mean_returns, cov_matrices
            )

    def _generate_constant(self,
                          num_simulations: int,
                          total_periods: int,
                          periods_per_year: int,
                          mean_returns: Optional[np.ndarray],
                          cov_matrix: Optional[np.ndarray]) -> np.ndarray:
        """
        Internal: Vectorized path generation for CONSTANT parameters.
        Fast path - single multivariate_normal call.
        """
        # Use provided or default parameters
        mean = mean_returns if mean_returns is not None else self.mean_returns
        cov = cov_matrix if cov_matrix is not None else self.cov_matrix

        # Scale to period frequency
        period_mean = mean / periods_per_year
        period_cov = cov / periods_per_year

        # Vectorized sampling (FAST!)
        total_samples = num_simulations * total_periods
        all_returns = np.random.multivariate_normal(
            mean=period_mean,
            cov=period_cov,
            size=total_samples
        )

        # Reshape
        paths = all_returns.reshape(num_simulations, total_periods, self.num_assets)

        # Store metadata
        self.paths = paths
        self.num_simulations = num_simulations
        self.total_periods = total_periods
        self.periods_per_year = periods_per_year

        return paths

    def _generate_time_varying(self,
                              num_simulations: int,
                              total_periods: int,
                              periods_per_year: int,
                              start_date: str,
                              frequency: str,
                              mean_returns_ts: Optional[pd.DataFrame],
                              cov_matrices_ts: Optional[pd.DataFrame]) -> np.ndarray:
        """
        Internal: Loop-based path generation for TIME-VARYING parameters.
        Slower but necessary for regime switching.
        """
        if start_date is None:
            raise ValueError("start_date required for time-varying parameters")

        # Generate date sequence
        start = pd.Timestamp(start_date)
        if frequency == 'biweekly':
            dates = [start + timedelta(days=14*i) for i in range(total_periods)]
        else:
            dates = pd.date_range(start=start, periods=total_periods, freq=frequency)

        # Prepare covariance dict if provided
        cov_dict = None
        if cov_matrices_ts is not None:
            cov_dict = {}

            # Check format: single 'cov_matrix' column (list of arrays) or flattened columns
            if 'cov_matrix' in cov_matrices_ts.columns:
                # List of arrays format (RECOMMENDED)
                for date, row in cov_matrices_ts.iterrows():
                    cov_dict[date] = row['cov_matrix']  # Already np.ndarray!
            else:
                # Flattened columns format (backward compatibility)
                for date, row in cov_matrices_ts.iterrows():
                    cov_matrix = np.zeros((self.num_assets, self.num_assets))
                    for i in range(self.num_assets):
                        for j in range(i, self.num_assets):
                            col_name = f"{self.tickers[i]}_{self.tickers[j]}"
                            cov_matrix[i, j] = row[col_name]
                            cov_matrix[j, i] = row[col_name]
                    cov_dict[date] = cov_matrix

        # Initialize paths
        paths = np.zeros((num_simulations, total_periods, self.num_assets))

        # Sample period-by-period
        for period_idx, date in enumerate(dates):
            # Get mean for this period
            if mean_returns_ts is not None:
                nearest_idx = mean_returns_ts.index.get_indexer([date], method='nearest')[0]
                period_mean_annual = mean_returns_ts.iloc[nearest_idx].values
            else:
                period_mean_annual = self.mean_returns

            # Get covariance for this period
            if cov_dict is not None:
                cov_date = min(cov_dict.keys(), key=lambda d: abs((d - date).total_seconds()))
                period_cov_annual = cov_dict[cov_date]
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

        return paths

    def generate_lifecycle_paths(self,
                                num_simulations: int,
                                accumulation_years: int,
                                accumulation_periods_per_year: int,
                                decumulation_years: int,
                                mean_returns: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                                cov_matrices: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                                start_date: str = '2025-01-01') -> tuple:
        """
        Generate continuous lifecycle paths.

        Now CALLS generate_paths() internally - no duplication!

        Parameters:
        -----------
        mean_returns : np.ndarray or pd.DataFrame, optional
            Constant (array) or time-varying (DataFrame) parameters
        """
        # Calculate total periods
        acc_periods = accumulation_years * accumulation_periods_per_year
        dec_periods = decumulation_years * accumulation_periods_per_year
        total_periods = acc_periods + dec_periods

        # CALL unified generate_paths() - works with both constant and time-varying!
        continuous_paths = self.generate_paths(
            num_simulations=num_simulations,
            total_periods=total_periods,
            periods_per_year=accumulation_periods_per_year,
            start_date=start_date,
            mean_returns=mean_returns,
            cov_matrices=cov_matrices
        )

        # Split into accumulation and decumulation
        accumulation_paths = continuous_paths[:, :acc_periods, :]
        decumulation_portion = continuous_paths[:, acc_periods:, :]

        # Compound decumulation to annual
        decumulation_paths = np.zeros((num_simulations, decumulation_years, self.num_assets))

        for sim in range(num_simulations):
            for year in range(decumulation_years):
                start_period = year * accumulation_periods_per_year
                end_period = (year + 1) * accumulation_periods_per_year
                year_returns = decumulation_portion[sim, start_period:end_period, :]

                for asset in range(self.num_assets):
                    cumulative_return = np.prod(1 + year_returns[:, asset])
                    decumulation_paths[sim, year, asset] = cumulative_return - 1

        print(f"  Generated continuous lifecycle paths:")
        print(f"    Accumulation: {accumulation_paths.shape}")
        print(f"    Decumulation: {decumulation_paths.shape}")

        return accumulation_paths, decumulation_paths

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


# ============================================================================
# DEMO: Show the beautiful unified API
# ============================================================================
if __name__ == "__main__":
    import time

    # Setup
    tickers = ['SPY', 'AGG', 'NVDA', 'GLD']
    mean_returns = np.array([0.10, 0.04, 0.30, 0.08])
    cov_matrix = np.array([
        [0.04, -0.01, 0.02, 0.01],
        [-0.01, 0.01, 0.00, 0.00],
        [0.02, 0.00, 0.16, 0.01],
        [0.01, 0.00, 0.01, 0.04]
    ])

    generator = MCPathGenerator(tickers, mean_returns, cov_matrix, seed=42)

    print("="*80)
    print("UNIFIED API DEMO - Simple and Advanced Use Cases")
    print("="*80)

    # ========================================================================
    # Use Case 1: SIMPLE - Constant parameters (most common)
    # ========================================================================
    print("\n[1] SIMPLE USE - Constant parameters")
    print("-" * 80)
    print("Code:")
    print("  paths = generator.generate_paths(")
    print("      num_simulations=1000,")
    print("      total_periods=260,")
    print("      periods_per_year=26")
    print("  )")
    print()

    start = time.time()
    paths_simple = generator.generate_paths(
        num_simulations=1000,
        total_periods=260,
        periods_per_year=26
    )
    time_simple = time.time() - start

    print(f"Result: {paths_simple.shape} in {time_simple:.4f} sec")
    print("✓ Simple API - just 3 parameters!")

    # ========================================================================
    # Use Case 2: ADVANCED - Time-varying parameters
    # ========================================================================
    print(f"\n[2] ADVANCED USE - Time-varying parameters")
    print("-" * 80)
    print("Code:")
    print("  # Create regime-switching parameters")
    print("  mean_df = pd.DataFrame({...}, index=dates)")
    print()
    print("  paths = generator.generate_paths(")
    print("      num_simulations=1000,")
    print("      start_date='2025-01-01',")
    print("      total_periods=260,")
    print("      periods_per_year=26,")
    print("      mean_returns=mean_df  # DataFrame!")
    print("  )")
    print()

    # Create time-varying parameters
    dates = pd.date_range('2025-01-01', periods=260, freq='D')
    mean_df = pd.DataFrame({
        'SPY': [0.15 if i < 130 else 0.02 for i in range(260)],  # Bull→Bear
        'AGG': [0.04] * 260,
        'NVDA': [0.30] * 260,
        'GLD': [0.08] * 260
    }, index=dates)

    start = time.time()
    paths_advanced = generator.generate_paths(
        num_simulations=1000,
        start_date='2025-01-01',
        total_periods=260,
        periods_per_year=26,
        mean_returns=mean_df  # DataFrame!
    )
    time_advanced = time.time() - start

    print(f"Result: {paths_advanced.shape} in {time_advanced:.4f} sec")
    print("✓ Same method handles time-varying!")

    # ========================================================================
    # Use Case 3: LIFECYCLE - Also works!
    # ========================================================================
    print(f"\n[3] LIFECYCLE USE - Continuous paths")
    print("-" * 80)
    print("Code:")
    print("  acc_paths, dec_paths = generator.generate_lifecycle_paths(")
    print("      num_simulations=1000,")
    print("      accumulation_years=9,")
    print("      accumulation_periods_per_year=26,")
    print("      decumulation_years=30")
    print("  )")
    print()

    start = time.time()
    acc_paths, dec_paths = generator.generate_lifecycle_paths(
        num_simulations=1000,
        accumulation_years=9,
        accumulation_periods_per_year=26,
        decumulation_years=30
    )
    time_lifecycle = time.time() - start

    print(f"Time: {time_lifecycle:.4f} sec")
    print("✓ Lifecycle also uses unified generate_paths()!")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\n✅ ONE method (generate_paths) handles:")
    print("   - Constant parameters (fast path)")
    print("   - Time-varying parameters (flexible path)")
    print("   - Lifecycle paths (composition)")
    print()
    print("✅ Simple API for simple cases:")
    print("   paths = generator.generate_paths(1000, 260, 26)")
    print()
    print("✅ Advanced API when you need it:")
    print("   paths = generator.generate_paths(..., mean_returns=df)")
    print()
    print("✅ No code duplication:")
    print("   - _generate_constant() for vectorized path")
    print("   - _generate_time_varying() for loop-based path")
    print("   - generate_paths() dispatches based on input type")
    print()
    print("="*80)
