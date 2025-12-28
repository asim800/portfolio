#!/usr/bin/env python3
"""
Performance Engine for Portfolio Backtesting.
Simplified orchestration layer using Portfolio and PortfolioTracker.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import os

from src.config import RebalancingConfig
from src.metrics.tracker import PortfolioTracker
from src.engine.optimizer import PortfolioOptimizer
from src.data import FinData
from src.engine.portfolio import Portfolio
from src.visualization.backtest import RebalancingVisualizer
from src.engine.period_manager import PeriodManager

import ipdb


class PerformanceEngine:
    """
    Performance backtesting engine - simplified orchestration layer.

    Creates Portfolio instances, runs backtests via PortfolioTracker,
    and handles visualization and reporting.
    """

    def __init__(self, data: FinData, config: RebalancingConfig):
        """
        Initialize performance engine with data and configuration.

        Parameters:
        -----------
        data : FinData
            Financial data manager
        config : RebalancingConfig
            Configuration for optimization and rebalancing
        """
        self.data = data
        self.config = config

        # Core components
        self.tracker = PortfolioTracker()
        self.visualizer = RebalancingVisualizer(config)
        self.optimizer = PortfolioOptimizer(risk_free_rate=config.risk_free_rate)

        # Data storage
        self.returns_data: Optional[pd.DataFrame] = None
        self.asset_names: List[str] = []
        self.baseline_weights: Optional[pd.Series] = None
        self.period_manager: Optional[PeriodManager] = None

        # Portfolios
        self.portfolios: Dict[str, Portfolio] = {}

        logging.info("PerformanceEngine initialized (simplified architecture)")

    def load_data(self, returns_data: pd.DataFrame, baseline_weights: np.ndarray) -> None:
        """
        Load returns data and baseline weights.

        Parameters:
        -----------
        returns_data : pd.DataFrame
            Historical returns data (dates × assets)
        baseline_weights : np.ndarray
            Baseline portfolio weights
        """
        self.returns_data = returns_data.copy()
        self.asset_names = list(returns_data.columns)

        # Convert baseline weights to pandas Series
        self.baseline_weights = pd.Series(baseline_weights, index=self.asset_names)

        # Validate data
        if len(baseline_weights) != len(self.asset_names):
            raise ValueError(f"Baseline weights length ({len(baseline_weights)}) "
                           f"doesn't match number of assets ({len(self.asset_names)})")

        # Initialize period manager with frequency
        self.period_manager = PeriodManager(
            returns_data,
            frequency=self.config.get_rebalancing_frequency()
        )

        logging.info(f"Loaded data: {len(returns_data)} days, {len(self.asset_names)} assets")
        logging.info(f"Created {self.period_manager.num_periods} rebalancing periods")

        # Create portfolios
        self._create_portfolios()

    def _create_portfolios(self) -> None:
        """Create all configured portfolios using factory methods."""
        logging.info("Creating portfolios...")

        # Create rebalancing strategy portfolios
        for strategy_name in self.config.rebalancing_strategies:
            rebalance_frequency = self.config.get_rebalancing_period(strategy_name)

            if strategy_name == 'buy_and_hold':
                portfolio = Portfolio.create_buy_and_hold(
                    self.asset_names,
                    self.baseline_weights,
                    name=strategy_name
                )
            elif strategy_name == 'target_weight':
                portfolio = Portfolio.create_target_weight(
                    self.asset_names,
                    self.baseline_weights,
                    rebalance_frequency=rebalance_frequency,
                    name=strategy_name
                )
            elif strategy_name == 'equal_weight':
                portfolio = Portfolio.create_equal_weight(
                    self.asset_names,
                    rebalance_frequency=rebalance_frequency,
                    name=strategy_name
                )
            elif strategy_name == 'spy_only':
                portfolio = Portfolio.create_spy_only(
                    self.asset_names,
                    rebalance_frequency=rebalance_frequency,
                    name=strategy_name
                )
            else:
                logging.warning(f"Unknown rebalancing strategy: {strategy_name}, skipping")
                continue

            # Ingest data
            portfolio.returns_data = self.returns_data.copy()

            # Add to tracker
            self.portfolios[strategy_name] = portfolio
            self.tracker.add_portfolio(strategy_name, portfolio)
            logging.info(f"Created portfolio: {strategy_name}")

        # Create optimized portfolios
        for method in self.config.optimization_methods:
            portfolio = Portfolio.create_optimized(
                self.asset_names,
                self.baseline_weights,
                self.optimizer,
                method=method,
                rebalance_frequency=self.config.rebalancing_frequency,
                name=method
            )

            # Ingest data
            portfolio.returns_data = self.returns_data.copy()

            # Add to tracker
            self.portfolios[method] = portfolio
            self.tracker.add_portfolio(method, portfolio)
            logging.info(f"Created optimized portfolio: {method}")

        logging.info(f"Created {len(self.portfolios)} portfolios total")

    def run_backtest(self, start_date: str, end_date: str) -> None:
        """
        Run complete backtesting for all portfolios.

        Much simpler now - just delegates to PortfolioTracker which runs each Portfolio.

        Parameters:
        -----------
        start_date : str
            Start date (for logging, not used in new architecture)
        end_date : str
            End date (for logging, not used in new architecture)
        """
        if self.period_manager is None:
            raise RuntimeError("Must call load_data() before run_backtest()")

        logging.info("Starting performance backtesting...")
        logging.info(f"Running backtest from {start_date} to {end_date}")

        # Run backtest on all portfolios via tracker
        self.tracker.run_backtest(self.period_manager)

        logging.info("Backtesting completed successfully")

    # =========================================================================
    # PERFORMANCE ANALYSIS AND REPORTING
    # =========================================================================
    def get_performance_metrics(self) -> pd.DataFrame:
        """Get comprehensive performance metrics for all portfolios."""
        if self.tracker is None:
            return pd.DataFrame()

        return self.tracker.get_portfolio_summary_statistics()

    def get_portfolio_tables(self) -> Dict[str, pd.DataFrame]:
        """Get detailed performance tables for each portfolio."""
        if self.tracker is None:
            return {}

        tables = {}
        for portfolio_name in self.config.get_portfolio_names():
            # Get portfolio data
            cumulative_returns = self.tracker.get_portfolio_cumulative_returns(portfolio_name).dropna()
            returns = self.tracker.get_portfolio_returns_history(portfolio_name).dropna()
            # Calculate portfolio values from cumulative returns (starting from 100)
            values = cumulative_returns * 100

            if not cumulative_returns.empty:
                # Create detailed table
                table_data = pd.DataFrame({
                    'Date': cumulative_returns.index,
                    'Cumulative_Return': (cumulative_returns - 1) * 100,  # Convert to percentage
                    'Period_Return': returns * 100,  # Convert to percentage
                    'Portfolio_Value': values,
                    'Max_Drawdown': self._calculate_rolling_drawdown(cumulative_returns) * 100
                })
                table_data.set_index('Date', inplace=True)
                tables[portfolio_name] = table_data

        return tables

    def _calculate_rolling_drawdown(self, cumulative_returns: pd.Series) -> pd.Series:
        """Calculate rolling maximum drawdown."""
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown

    def compare_portfolios(self, portfolio1: str, portfolio2: str) -> pd.DataFrame:
        """Compare performance metrics between any two portfolios."""
        if self.tracker is None:
            return pd.DataFrame()

        # Get performance metrics
        metrics = self.get_performance_metrics()

        if portfolio1 not in metrics.index or portfolio2 not in metrics.index:
            logging.error(f"One or both portfolios not found: {portfolio1}, {portfolio2}")
            return pd.DataFrame()

        # Create comparison table
        comparison_data = []
        for metric in metrics.columns:
            value1 = metrics.loc[portfolio1, metric]
            value2 = metrics.loc[portfolio2, metric]
            difference = value2 - value1
            winner = portfolio2 if difference > 0 else portfolio1

            comparison_data.append({
                'Metric': metric,
                portfolio1: value1,
                portfolio2: value2,
                'Difference': difference,
                'Winner': winner
            })

        return pd.DataFrame(comparison_data)

    # Visualization Methods
    def plot_cumulative_returns(self, save_path: Optional[str] = None) -> None:
        """Plot cumulative returns for all portfolios."""
        if self.tracker is None:
            logging.error("No tracker available for plotting")
            return

        self.visualizer.plot_cumulative_returns(self.tracker, save_path)

    def plot_performance_summary(self, save_path: Optional[str] = None) -> None:
        """Plot performance metrics dashboard."""
        if self.tracker is None:
            logging.error("No tracker available for plotting")
            return

        self.visualizer.plot_performance_summary(self.tracker, save_path)

    def plot_max_drawdown_time_series(self, save_path: Optional[str] = None) -> None:
        """Plot max drawdown time series for all portfolios."""
        if self.tracker is None:
            logging.error("No tracker available for plotting")
            return

        self.visualizer.plot_max_drawdown_time_series(self.tracker, save_path)

    def plot_daily_cumulative_returns(self, save_path: Optional[str] = None) -> None:
        """Plot daily cumulative returns for all portfolios using underlying daily data."""
        if self.tracker is None:
            logging.error("No tracker available for plotting")
            return

        self.visualizer.plot_daily_cumulative_returns(self, save_path)

    def plot_portfolio_comparison(self, portfolio1: str, portfolio2: str,
                                save_path: Optional[str] = None) -> None:
        """Plot direct comparison between two portfolios."""
        if self.tracker is None:
            logging.error("No tracker available for plotting")
            return

        # Create focused comparison plot (to be added to visualizer)
        logging.info(f"Comparison plot for {portfolio1} vs {portfolio2} - to be implemented")

    def create_performance_tables(self, output_dir: Optional[str] = None) -> None:
        """Create formatted performance tables for each portfolio."""
        tables = self.get_portfolio_tables()

        if not tables:
            logging.warning("No portfolio tables to create")
            return

        output_dir = output_dir or self.config.results_directory
        os.makedirs(output_dir, exist_ok=True)

        for portfolio_name, table in tables.items():
            filename = f'portfolio_analysis_{portfolio_name}.csv'
            filepath = os.path.join(output_dir, filename)
            table.to_csv(filepath)
            logging.info(f"Saved performance table: {filepath}")

    def export_results(self, output_dir: Optional[str] = None) -> None:
        """Export all results to CSV files."""
        if self.tracker is None:
            logging.error("No tracker available for export")
            return

        output_dir = output_dir or self.config.results_directory
        self.tracker.export_to_csv(output_dir)
        self.tracker.export_portfolio_summary_tables(output_dir)
        logging.info(f"Exported all results to: {output_dir}")

    def print_portfolio_summary_tables(self, max_rows: int = 10) -> None:
        """Print portfolio summary tables to console."""
        if self.tracker is None:
            logging.error("No tracker available for printing")
            return

        self.tracker.print_portfolio_summary_tables(max_rows)

    def get_results(self) -> Optional[PortfolioTracker]:
        """Return the portfolio tracker with all results."""
        return self.tracker