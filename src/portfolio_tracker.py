#!/usr/bin/env python3
"""
Portfolio performance tracker using pandas DataFrames with datetime indices.
Clean interface for tracking portfolio weights, returns, and metrics over time.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, date
import logging

import ipdb

class PortfolioTracker:
    """
    Tracks portfolio performance using pandas DataFrames with datetime indices.
    Much cleaner than the old list-based approach.
    """

    def __init__(self, asset_names: List[str], portfolio_names: List[str]):
        """
        Initialize portfolio tracker.

        Parameters:
        -----------
        asset_names : List[str]
            Names of assets in portfolios
        portfolio_names : List[str]
            Names of portfolios to track
        """
        self.asset_names = asset_names
        self.portfolio_names = portfolio_names

        # Initialize tracking DataFrames with MultiIndex columns
        self._initialize_tracking_dataframes()

    def _initialize_tracking_dataframes(self) -> None:
        """Initialize the main tracking DataFrames."""

        # Portfolio weights over time (Date x Portfolio x Asset)
        weights_columns = pd.MultiIndex.from_product(
            [self.portfolio_names, self.asset_names],
            names=['Portfolio', 'Asset']
        )
        self.weights_df = pd.DataFrame(
            index=pd.DatetimeIndex([], name='Date'),
            columns=weights_columns
        )

        # Portfolio returns over time (Date x Portfolio)
        self.returns_df = pd.DataFrame(
            index=pd.DatetimeIndex([], name='Date'),
            columns=self.portfolio_names
        )

        # Cumulative returns over time (Date x Portfolio)
        self.cumulative_returns_df = pd.DataFrame(
            index=pd.DatetimeIndex([], name='Date'),
            columns=self.portfolio_names
        )

        # Portfolio values over time (Date x Portfolio) - starting from 100
        self.portfolio_values_df = pd.DataFrame(
            index=pd.DatetimeIndex([], name='Date'),
            columns=self.portfolio_names
        )

        # Period-level performance metrics (Date x Portfolio x Metric)
        metrics_columns = pd.MultiIndex.from_product(
            [self.portfolio_names, ['period_return', 'sharpe_ratio', 'volatility', 'beta', 'max_drawdown']],
            names=['Portfolio', 'Metric']
        )
        self.metrics_df = pd.DataFrame(
            index=pd.DatetimeIndex([], name='Date'),
            columns=metrics_columns
        )

        logging.info(f"PortfolioTracker initialized: {len(self.portfolio_names)} portfolios, "
                    f"{len(self.asset_names)} assets")

    def add_period_performance(self,
                             period_start: pd.Timestamp,
                             period_end: pd.Timestamp,
                             portfolio_weights: Dict[str, pd.Series],
                             portfolio_returns: Dict[str, float],
                             portfolio_metrics: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        """
        Add performance data for a specific period.

        Parameters:
        -----------
        period_start : pd.Timestamp
            Period start date
        period_end : pd.Timestamp
            Period end date (used as the index date)
        portfolio_weights : Dict[str, pd.Series]
            Portfolio weights for each portfolio as pandas Series
        portfolio_returns : Dict[str, float]
            Portfolio returns for each portfolio
        portfolio_metrics : Optional[Dict[str, Dict[str, float]]]
            Additional portfolio metrics for each portfolio
        """
        # Use period_end as the index date (when rebalancing occurs)
        index_date = period_end

        # Add weights - now works with pandas Series
        for portfolio_name, weights in portfolio_weights.items():
            if portfolio_name in self.portfolio_names:
                for asset_name in self.asset_names:
                    # Use pandas Series indexing instead of position-based indexing
                    weight_value = weights.get(asset_name, 0.0) if isinstance(weights, pd.Series) else weights[asset_name]
                    self.weights_df.loc[index_date, (portfolio_name, asset_name)] = weight_value

        # Add returns
        for portfolio_name, return_val in portfolio_returns.items():
            if portfolio_name in self.portfolio_names:
                self.returns_df.loc[index_date, portfolio_name] = return_val

        if portfolio_metrics:
            for portfolio_name, metrics in portfolio_metrics.items():
                if portfolio_name in self.portfolio_names:
                    for metric_name, metric_value in metrics.items():
                        if metric_name in ['period_return', 'sharpe_ratio', 'volatility', 'beta', 'max_drawdown']:
                            self.metrics_df.loc[index_date, (portfolio_name, metric_name)] = metric_value

        logging.debug(f"Added performance data for {len(portfolio_weights)} portfolios on {index_date}")

    def get_portfolio_weights_history(self, portfolio_name: str) -> pd.DataFrame:
        """
        Get weight history for a specific portfolio.

        Parameters:
        -----------
        portfolio_name : str
            Name of portfolio

        Returns:
        --------
        DataFrame with dates as index and assets as columns
        """
        if portfolio_name not in self.portfolio_names:
            raise ValueError(f"Portfolio '{portfolio_name}' not found")

        return self.weights_df[portfolio_name].copy()

    def get_portfolio_returns_history(self, portfolio_name: str) -> pd.Series:
        """
        Get returns history for a specific portfolio.

        Parameters:
        -----------
        portfolio_name : str
            Name of portfolio

        Returns:
        --------
        Series with dates as index and returns as values
        """
        if portfolio_name not in self.portfolio_names:
            raise ValueError(f"Portfolio '{portfolio_name}' not found")

        return self.returns_df[portfolio_name].copy()

    def get_portfolio_cumulative_returns(self, portfolio_name: str) -> pd.Series:
        """Get cumulative returns for a specific portfolio."""
        if portfolio_name not in self.portfolio_names:
            raise ValueError(f"Portfolio '{portfolio_name}' not found")

        return self.cumulative_returns_df[portfolio_name].copy()

    def get_all_portfolio_returns(self) -> pd.DataFrame:
        """Get returns for all portfolios."""
        return self.returns_df.copy()

    def get_all_cumulative_returns(self) -> pd.DataFrame:
        """Get cumulative returns for all portfolios."""
        return self.cumulative_returns_df.copy()

    def get_all_portfolio_values(self) -> pd.DataFrame:
        """Get portfolio values for all portfolios."""
        return self.portfolio_values_df.copy()

    def get_portfolio_summary_statistics(self) -> pd.DataFrame:
        """
        Calculate summary statistics for all portfolios.

        Returns:
        --------
        DataFrame with portfolios as index and metrics as columns
        """
        summary_stats = []

        for portfolio_name in self.portfolio_names:
            returns = self.get_portfolio_returns_history(portfolio_name).dropna()
            cumulative_returns = self.get_portfolio_cumulative_returns(portfolio_name).dropna()

            if len(returns) == 0:
                continue

            # Calculate statistics
            total_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0

            # Calculate the actual time span in years for proper annualization
            if len(cumulative_returns) > 1:
                start_date = cumulative_returns.index[0]
                end_date = cumulative_returns.index[-1]
                years = (end_date - start_date).days / 365.25
                annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            else:
                annual_return = 0

            # For volatility, we need to annualize period returns correctly
            # Since these are ~monthly returns, multiply by sqrt(12) to annualize
            periods_per_year = 12  # Approximately monthly rebalancing
            volatility = returns.std() * np.sqrt(periods_per_year)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0

            # Calculate max drawdown
            cumulative_values = (1 + returns).cumprod()
            peak = cumulative_values.cummax()
            drawdown = (cumulative_values - peak) / peak
            max_drawdown = drawdown.min()

            summary_stats.append({
                'Portfolio': portfolio_name,
                'Total_Return': total_return,
                'Annual_Return': annual_return,
                'Volatility': volatility,
                'Sharpe_Ratio': sharpe_ratio,
                'Max_Drawdown': max_drawdown,
                'Num_Periods': len(returns)
            })

        if not summary_stats:
            # Return empty DataFrame with proper columns if no data
            return pd.DataFrame(columns=['Total_Return', 'Annual_Return', 'Volatility',
                                       'Sharpe_Ratio', 'Max_Drawdown', 'Num_Periods'])

        return pd.DataFrame(summary_stats).set_index('Portfolio')

    def get_latest_weights(self, portfolio_name: str) -> pd.Series:
        """Get latest weights for a portfolio."""
        weights_history = self.get_portfolio_weights_history(portfolio_name)
        if weights_history.empty:
            return pd.Series(index=self.asset_names, dtype=float)

        return weights_history.iloc[-1]

    def get_weights_at_date(self, portfolio_name: str, date: pd.Timestamp) -> pd.Series:
        """Get portfolio weights at a specific date."""
        weights_history = self.get_portfolio_weights_history(portfolio_name)

        if date in weights_history.index:
            return weights_history.loc[date]
        else:
            # Find the most recent weights before this date
            available_dates = weights_history.index[weights_history.index <= date]
            if len(available_dates) > 0:
                return weights_history.loc[available_dates[-1]]
            else:
                return pd.Series(index=self.asset_names, dtype=float)

    def export_to_csv(self, output_dir: str) -> None:
        """Export all tracking data to CSV files."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Export weights
        weights_export = self.weights_df.copy()
        weights_export.to_csv(os.path.join(output_dir, 'portfolio_weights.csv'))

        # Export returns
        self.returns_df.to_csv(os.path.join(output_dir, 'portfolio_returns.csv'))

        # Export cumulative returns
        self.cumulative_returns_df.to_csv(os.path.join(output_dir, 'portfolio_cumulative_returns.csv'))

        # Export portfolio values
        self.portfolio_values_df.to_csv(os.path.join(output_dir, 'portfolio_values.csv'))

        # Export summary statistics
        summary_stats = self.get_portfolio_summary_statistics()
        summary_stats.to_csv(os.path.join(output_dir, 'portfolio_summary_statistics.csv'))

        logging.info(f"Portfolio tracking data exported to {output_dir}")

    def display_summary(self) -> None:
        """Display a summary of tracked portfolios."""
        logging.info("\n" + "="*60)
        logging.info("PORTFOLIO TRACKING SUMMARY")
        logging.info("="*60)

        summary_stats = self.get_portfolio_summary_statistics()

        if summary_stats.empty:
            logging.info("No portfolio performance data available to display")
            logging.info("="*60)
            return

        # Format for display
        display_df = summary_stats.copy()
        for col in ['Total_Return', 'Annual_Return', 'Volatility', 'Sharpe_Ratio', 'Max_Drawdown']:
            if col in display_df.columns:
                if col in ['Total_Return', 'Annual_Return', 'Volatility', 'Max_Drawdown']:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
                else:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")

        logging.info("\n" + display_df.to_string())
        logging.info("\n" + "="*60)

    def get_portfolio_summary_table(self, portfolio_name: str) -> pd.DataFrame:
        """
        Create a summary DataFrame for a specific portfolio with multi-index columns.

        Parameters:
        -----------
        portfolio_name : str
            Name of the portfolio

        Returns:
        --------
        DataFrame with:
        - Row index: Rebalance dates
        - Multi-index columns: Level 0 = Symbol, Level 1 = ['Weight', 'Cumulative_Return']
        """
        if portfolio_name not in self.portfolio_names:
            raise ValueError(f"Portfolio '{portfolio_name}' not found")

        # Get weights history (Date x Asset)
        weights_df = self.get_portfolio_weights_history(portfolio_name)

        # Get cumulative returns for this portfolio
        cumulative_returns_series = self.get_portfolio_cumulative_returns(portfolio_name)

        # Create the multi-index columns structure
        # Level 0: Asset symbols, Level 1: ['Weight', 'Cumulative_Return']
        column_tuples = []

        # Add weight columns for each asset
        for asset in self.asset_names:
            column_tuples.append((asset, 'Weight'))

        # Add cumulative return column for the portfolio
        column_tuples.append((portfolio_name, 'Cumulative_Return'))

        # Create multi-index columns
        multi_columns = pd.MultiIndex.from_tuples(column_tuples, names=['Symbol', 'Metric'])

        # Create the summary DataFrame
        summary_df = pd.DataFrame(index=weights_df.index, columns=multi_columns)

        # Fill in weights data
        for asset in self.asset_names:
            summary_df[(asset, 'Weight')] = weights_df[asset]

        # Fill in cumulative returns data
        summary_df[(portfolio_name, 'Cumulative_Return')] = cumulative_returns_series

        return summary_df

    def get_all_portfolio_summary_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Create summary tables for all portfolios.

        Returns:
        --------
        Dict[str, DataFrame] with portfolio names as keys and summary DataFrames as values
        """
        summary_tables = {}

        for portfolio_name in self.portfolio_names:
            try:
                summary_tables[portfolio_name] = self.get_portfolio_summary_table(portfolio_name)
            except Exception as e:
                logging.warning(f"Could not create summary table for {portfolio_name}: {e}")
                continue

        return summary_tables

    def export_portfolio_summary_tables(self, output_dir: str = "../results/rebalancing/") -> None:
        """
        Export summary tables for all portfolios as CSV files.

        Parameters:
        -----------
        output_dir : str
            Directory to save the summary tables
        """
        import os

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        summary_tables = self.get_all_portfolio_summary_tables()

        for portfolio_name, summary_df in summary_tables.items():
            if not summary_df.empty:
                # Save with multi-index columns
                filename = os.path.join(output_dir, f"portfolio_summary_table_{portfolio_name}.csv")
                summary_df.to_csv(filename)
                logging.info(f"Exported portfolio summary table for {portfolio_name} to {filename}")

        logging.info(f"Exported {len(summary_tables)} portfolio summary tables to {output_dir}")

    def print_portfolio_summary_tables(self, max_rows: int = 10) -> None:
        """
        Print portfolio summary tables to console with formatting.

        Parameters:
        -----------
        max_rows : int
            Maximum number of rows to display per table (default: 10)
        """
        summary_tables = self.get_all_portfolio_summary_tables()

        print("\n" + "="*80)
        print("PORTFOLIO SUMMARY TABLES")
        print("="*80)

        for i, (portfolio_name, summary_df) in enumerate(summary_tables.items()):
            if summary_df.empty:
                continue

            print(f"\n[{i+1}] {portfolio_name.upper()} PORTFOLIO")
            print("-" * 50)
            print(f"Shape: {summary_df.shape[0]} rebalance periods × {summary_df.shape[1]} metrics")
            print(f"Date Range: {summary_df.index[0].strftime('%Y-%m-%d')} to {summary_df.index[-1].strftime('%Y-%m-%d')}")

            # Display table with limited rows
            display_df = summary_df.head(max_rows) if len(summary_df) > max_rows else summary_df

            print(f"\nData Preview ({len(display_df)} of {len(summary_df)} periods):")
            print(display_df.to_string(float_format=lambda x: f"{x:.4f}"))

            if len(summary_df) > max_rows:
                print(f"\n... ({len(summary_df) - max_rows} more periods)")

            # Show summary statistics for weights
            weight_columns = [col for col in summary_df.columns if col[1] == 'Weight']
            if weight_columns:
                print(f"\nWeight Statistics:")
                weights_only = summary_df[weight_columns]
                print(f"  Average Weights:")
                for col, avg_weight in weights_only.mean().items():
                    asset_name = col[0]
                    print(f"    {asset_name}: {avg_weight:.4f}")
                print(f"  Weight Ranges:")
                for col, weight_range in (weights_only.max() - weights_only.min()).items():
                    asset_name = col[0]
                    print(f"    {asset_name}: {weight_range:.4f}")

        print(f"\n" + "="*80)
        print(f"Summary: {len(summary_tables)} portfolio summary tables displayed")
        print("="*80)