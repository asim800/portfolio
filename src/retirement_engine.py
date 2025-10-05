#!/usr/bin/env python3
"""
Retirement Engine - Monte Carlo simulation for retirement planning.

Simple, modular implementation that reuses existing components.
"""

import numpy as np
import pandas as pd
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from retirement_config import RetirementConfig
from fin_data import FinData


@dataclass
class SimulationPath:
    """Results from a single simulation path"""
    years: List[int]  # Year numbers (0, 1, 2, ...)
    dates: List[datetime]  # Actual dates
    portfolio_values: List[float]  # Portfolio value at start of each year
    withdrawals: List[float]  # Withdrawal amount each year
    annual_returns: List[float]  # Portfolio return each year
    success: bool  # Did portfolio survive?
    depletion_year: Optional[int] = None  # Year when portfolio depleted (if applicable)
    final_value: float = 0.0  # Final portfolio value


class RetirementEngine:
    """
    Monte Carlo retirement simulation engine.

    Simple implementation focused on core functionality.
    Reuses FinData for data management and return sampling.
    """

    def __init__(self, config: RetirementConfig, fin_data: Optional[FinData] = None):
        """
        Initialize retirement engine.

        Parameters:
        -----------
        config : RetirementConfig
            Retirement configuration
        fin_data : Optional[FinData]
            FinData instance (created if not provided)
        """
        self.config = config

        # Initialize FinData if not provided
        if fin_data is None:
            self.fin_data = FinData(
                start_date=config.start_date,
                end_date=config.end_date,
                cache_dir='../data'
            )
            # Load historical data
            self.fin_data.get_returns_data(config.tickers)
        else:
            self.fin_data = fin_data

        # Portfolio weights as numpy array (aligned with tickers)
        self.weights = np.array([self.config.current_portfolio[t] for t in self.config.tickers])

        logging.info(f"RetirementEngine initialized: {config.num_years} years, "
                    f"{len(config.tickers)} assets")

    def run_single_path(self, annual_returns: pd.DataFrame) -> SimulationPath:
        """
        Simulate a single retirement path with given returns.

        Parameters:
        -----------
        annual_returns : pd.DataFrame
            Annual returns for each asset (rows=years, cols=tickers)

        Returns:
        --------
        SimulationPath with detailed tracking data
        """
        # Initialize tracking
        portfolio_value = self.config.initial_portfolio

        years = []
        dates = []
        portfolio_values = []
        withdrawals = []
        portfolio_returns = []

        start_date = datetime.strptime(self.config.start_date, '%Y-%m-%d')

        # Simulate each year
        for year_num in range(self.config.num_years):
            current_date = start_date + timedelta(days=year_num * 365)

            # Record start-of-year values
            years.append(year_num)
            dates.append(current_date)
            portfolio_values.append(portfolio_value)

            # Calculate portfolio return for this year
            # Use np.dot for simple weighted return calculation
            asset_returns = annual_returns.iloc[year_num][self.config.tickers].values
            portfolio_return = np.dot(self.weights, asset_returns)
            portfolio_returns.append(portfolio_return)

            # Apply return
            portfolio_value *= (1 + portfolio_return)

            # Calculate inflation-adjusted withdrawal
            inflation_adjusted_withdrawal = (
                self.config.annual_withdrawal *
                (1 + self.config.inflation_rate) ** year_num
            )
            withdrawals.append(inflation_adjusted_withdrawal)

            # Subtract withdrawal
            portfolio_value -= inflation_adjusted_withdrawal

            # Check for depletion
            if portfolio_value <= 0:
                return SimulationPath(
                    years=years,
                    dates=dates,
                    portfolio_values=portfolio_values,
                    withdrawals=withdrawals,
                    annual_returns=portfolio_returns,
                    success=False,
                    depletion_year=year_num,
                    final_value=0.0
                )

        # Successful completion
        return SimulationPath(
            years=years,
            dates=dates,
            portfolio_values=portfolio_values,
            withdrawals=withdrawals,
            annual_returns=portfolio_returns,
            success=True,
            depletion_year=None,
            final_value=portfolio_value
        )

    def run_monte_carlo(self,
                       num_simulations: int = 5000,
                       method: str = 'bootstrap',
                       seed: Optional[int] = None,
                       show_progress: bool = True) -> 'MonteCarloResults':
        """
        Run Monte Carlo retirement simulation.

        Parameters:
        -----------
        num_simulations : int
            Number of simulations to run
        method : str
            Return sampling method: 'bootstrap' or 'parametric'
        seed : Optional[int]
            Random seed for reproducibility
        show_progress : bool
            Whether to show progress bar (requires tqdm)

        Returns:
        --------
        MonteCarloResults with aggregated statistics
        """
        logging.info(f"Starting Monte Carlo simulation: {num_simulations} paths")

        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Storage for results
        all_paths = []
        final_values = []
        success_count = 0
        portfolio_values_matrix = []

        # Progress tracking
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(num_simulations), desc="Simulating")
            except ImportError:
                logging.warning("tqdm not available, progress bar disabled")
                iterator = range(num_simulations)
        else:
            iterator = range(num_simulations)

        # Run simulations
        for sim_num in iterator:
            # Generate random return path
            # Use different seed for each simulation if base seed provided
            sim_seed = seed + sim_num if seed is not None else None

            annual_returns = self.fin_data.sample_annual_returns(
                self.config.tickers,
                num_years=self.config.num_years,
                method=method,
                seed=sim_seed
            )

            # Run single path simulation
            path = self.run_single_path(annual_returns)

            # Store results
            all_paths.append(path)
            final_values.append(path.final_value)

            if path.success:
                success_count += 1

            # Store portfolio values for this path (pad with 0s if depleted)
            values = path.portfolio_values.copy()
            if len(values) < self.config.num_years:
                # Pad with zeros for depleted paths
                values.extend([0.0] * (self.config.num_years - len(values)))
            portfolio_values_matrix.append(values)

        # Calculate statistics
        final_values_array = np.array(final_values)
        portfolio_values_array = np.array(portfolio_values_matrix)

        success_rate = success_count / num_simulations

        percentiles = {
            '5th': np.percentile(final_values_array, 5),
            '25th': np.percentile(final_values_array, 25),
            '50th': np.percentile(final_values_array, 50),
            '75th': np.percentile(final_values_array, 75),
            '95th': np.percentile(final_values_array, 95)
        }

        results = MonteCarloResults(
            num_simulations=num_simulations,
            success_rate=success_rate,
            paths=all_paths,
            final_values=final_values_array,
            percentiles=percentiles,
            median_final_value=percentiles['50th'],
            mean_final_value=np.mean(final_values_array),
            std_final_value=np.std(final_values_array),
            portfolio_values_matrix=portfolio_values_array
        )

        logging.info(f"Monte Carlo complete: {success_rate:.1%} success rate, "
                    f"median final value: ${results.median_final_value:,.0f}")

        return results


@dataclass
class MonteCarloResults:
    """Aggregated results from Monte Carlo simulation"""
    num_simulations: int
    success_rate: float  # Fraction of simulations that succeeded
    paths: List[SimulationPath]  # All simulation paths
    final_values: np.ndarray  # Final portfolio values (0 for failed paths)

    # Percentile statistics
    percentiles: dict  # 5th, 25th, 50th, 75th, 95th percentiles
    median_final_value: float
    mean_final_value: float
    std_final_value: float

    # Portfolio values over time (for plotting)
    portfolio_values_matrix: np.ndarray  # Shape: (num_sims, num_years)

    def export_to_csv(self, output_dir: str):
        """Export results to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # 1. Summary statistics
        summary_df = pd.DataFrame({
            'Metric': ['Num_Simulations', 'Success_Rate', 'Median_Final_Value',
                       'Mean_Final_Value', 'Std_Dev_Final_Value',
                       'Percentile_5th', 'Percentile_25th', 'Percentile_50th',
                       'Percentile_75th', 'Percentile_95th'],
            'Value': [
                self.num_simulations,
                self.success_rate,
                self.median_final_value,
                self.mean_final_value,
                self.std_final_value,
                self.percentiles['5th'],
                self.percentiles['25th'],
                self.percentiles['50th'],
                self.percentiles['75th'],
                self.percentiles['95th']
            ]
        })
        summary_df.to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)

        # 2. All paths final values
        final_values_df = pd.DataFrame({
            'Simulation': range(self.num_simulations),
            'Final_Value': self.final_values,
            'Success': [p.success for p in self.paths]
        })
        final_values_df.to_csv(os.path.join(output_dir, 'final_values.csv'), index=False)

        # 3. Percentile paths over time
        percentile_data = {
            'Year': range(len(self.portfolio_values_matrix[0]))
        }
        for pct, label in [(5, '5th'), (25, '25th'), (50, '50th'), (75, '75th'), (95, '95th')]:
            percentile_data[f'{label}_percentile'] = np.percentile(
                self.portfolio_values_matrix, pct, axis=0
            )

        percentile_df = pd.DataFrame(percentile_data)
        percentile_df.to_csv(os.path.join(output_dir, 'percentile_paths.csv'), index=False)

        logging.info(f"Exported 3 CSV files to {output_dir}")
