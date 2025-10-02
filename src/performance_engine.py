#!/usr/bin/env python3
"""
Performance Engine for Portfolio Backtesting.
Clean architecture replacing RebalancingEngine with proper separation of concerns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, date, timedelta
import logging
import os

from config import RebalancingConfig
from portfolio_tracker import PortfolioTracker
from portfolio_optimizer import PortfolioOptimizer
from fin_data import FinData
from rebalancing_strategies import RebalancingStrategies
from rebalancing_visualization import RebalancingVisualizer
from period_manager import PeriodManager

import ipdb

class PerformanceEngine:
    """
    Performance backtesting engine with clean architecture.

    Handles static and dynamic portfolio backtesting with unified data storage,
    performance calculations, and visualization capabilities.
    """

    def __init__(self, data: FinData, optimizer: PortfolioOptimizer, config: RebalancingConfig):
        """Initialize performance engine with data, optimizer and configuration."""
        self.data = data
        self.optimizer = optimizer
        self.config = config

        # Core components
        self.tracker: Optional[PortfolioTracker] = None
        self.visualizer = RebalancingVisualizer(config)
        self.rebalancing_factory = RebalancingStrategies()

        # Data storage
        self.returns_data: Optional[pd.DataFrame] = None
        self.asset_names: List[str] = []
        self.baseline_weights: Optional[np.ndarray] = None
        self.period_manager: Optional[PeriodManager] = None

        # Strategy management
        self.active_strategies: Dict[str, Any] = {}
        self.static_strategies: Dict[str, np.ndarray] = {}
        self.dynamic_strategies: Dict[str, dict] = {}

        # Track actual portfolio weights for buy-and-hold drift calculation
        self.last_period_weights: Dict[str, np.ndarray] = {}  # {strategy: weights}

        logging.info("PerformanceEngine initialized with clean architecture")

    def load_data(self, returns_data: pd.DataFrame, baseline_weights: np.ndarray) -> None:
        """Load returns data and baseline weights."""
        self.returns_data = returns_data.copy()
        self.asset_names = list(returns_data.columns)
        self.baseline_weights = baseline_weights.copy()

        # Validate data
        if len(baseline_weights) != len(self.asset_names):
            raise ValueError(f"Baseline weights length ({len(baseline_weights)}) "
                           f"doesn't match number of assets ({len(self.asset_names)})")

        # Initialize period manager
        self.period_manager = PeriodManager(
            returns_data,
            rebalancing_period_days=self.config.rebalancing_period_days
        )

        # Initialize portfolio tracker
        portfolio_names = self.config.get_portfolio_names()
        self.tracker = PortfolioTracker(
            asset_names=self.asset_names,
            portfolio_names=portfolio_names
        )

        logging.info(f"Loaded data: {len(returns_data)} days, {len(self.asset_names)} assets")
        logging.info(f"Created {self.period_manager.num_periods} rebalancing periods")

    def add_static_strategy(self, strategy_name: str, weights: Optional[np.ndarray] = None) -> None:
        """Add a static portfolio strategy."""
        if weights is None:
            if strategy_name == "equal_weight":
                weights = np.ones(len(self.asset_names)) / len(self.asset_names)
            elif strategy_name == "spy_only":
                # Find SPY index
                spy_idx = None
                for i, asset in enumerate(self.asset_names):
                    if asset.upper() == 'SPY':
                        spy_idx = i
                        break
                if spy_idx is not None:
                    weights = np.zeros(len(self.asset_names))
                    weights[spy_idx] = 1.0
                else:
                    logging.warning("SPY not found in assets, using equal weights")
                    weights = np.ones(len(self.asset_names)) / len(self.asset_names)
            else:
                weights = self.baseline_weights.copy()

        self.static_strategies[strategy_name] = weights.copy()
        logging.info(f"Added static strategy '{strategy_name}'")

    def add_dynamic_strategy(self, optimization_method: str, rebalance_days: int = 30) -> None:
        """Add a dynamic optimization strategy."""
        self.dynamic_strategies[optimization_method] = {
            'method': optimization_method,
            'rebalance_days': rebalance_days
        }
        logging.info(f"Added dynamic strategy '{optimization_method}' with {rebalance_days}-day rebalancing")

    def run_backtest(self, start_date: str, end_date: str) -> None:
        """Run complete backtesting for all strategies."""
        if self.tracker is None:
            raise RuntimeError("Must call load_data() before run_backtest()")

        logging.info("Starting performance backtesting...")

        # Initialize rebalancing strategies from config
        self._initialize_rebalancing_strategies()

        # Run backtest across all periods
        cumulative_returns = {name: 1.0 for name in self.config.get_portfolio_names()}

        for period_num, period_data, period_info in self.period_manager.iter_periods():
            start_date = period_info['period_start'].date()
            end_date = period_info['period_end'].date()

            logging.info(f"Processing period {period_num}: {start_date} to {end_date} "
                        f"({period_info['trading_days']} trading days)")

            # Get current weights for all strategies
            current_weights = self._get_period_weights(period_num, period_data)

            # Calculate performance for each strategy
            portfolio_weights = {}
            portfolio_returns = {}

            for portfolio_name, weights in current_weights.items():
                # Calculate portfolio performance
                performance = self._calculate_portfolio_performance(
                    period_data, weights, portfolio_name
                )


                # Store results
                portfolio_weights[portfolio_name] = pd.Series(weights, index=self.asset_names, name='weights')
                portfolio_returns[portfolio_name] = performance['period_return']

                # Update cumulative returns
                cumulative_returns[portfolio_name] *= (1 + performance['period_return'])
 
                logging.debug(f"{portfolio_name}: {performance['period_return']:.2%} return")

            # ipdb.set_trace()

            # Add to tracker
            if portfolio_weights and portfolio_returns:
                self.tracker.add_period_performance(
                    period_start=pd.Timestamp(start_date),
                    period_end=pd.Timestamp(end_date),
                    portfolio_weights=portfolio_weights,
                    portfolio_returns=portfolio_returns
                )

        self.tracker.cumulative_returns_df = (1+self.tracker.returns_df).cumprod()

        logging.info("Backtesting completed successfully")

    def _initialize_rebalancing_strategies(self) -> None:
        """Initialize rebalancing strategies from config."""
        for strategy_name in self.config.rebalancing_strategies:
            period = self.config.get_rebalancing_period(strategy_name)

            if strategy_name == 'buy_and_hold':
                # Buy and hold never rebalances
                strategy = self.rebalancing_factory.create_strategy(
                    'buy_and_hold',
                    asset_names=self.asset_names,
                    rebalancing_period_days=float('inf')
                )
            else:
                strategy = self.rebalancing_factory.create_strategy(
                    strategy_name,
                    asset_names=self.asset_names,
                    rebalancing_period_days=period
                )

            self.active_strategies[strategy_name] = strategy
            logging.info(f"Initialized {strategy_name} strategy with {period}-day period")

    def _get_period_weights(self, period_num: int, period_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get portfolio weights for current period."""
        current_weights = {}

        # Get weights from rebalancing strategies
        for strategy_name, strategy in self.active_strategies.items():
            # Get period start and end dates
            period_start = period_data.index[0].date()
            period_end = period_data.index[-1].date()

            # Handle weight calculation based on strategy type
            if period_num == 0:
                # First period - all strategies start with baseline weights
                current_period_weights = self.baseline_weights.copy()
            elif strategy_name == 'buy_and_hold':
                # For buy-and-hold, calculate drifted weights from previous period
                if strategy_name in self.last_period_weights:
                    # Get the previous period's data and calculate drift
                    prev_period_data = self._get_previous_period_data(period_num - 1)
                    if prev_period_data is not None and not prev_period_data.empty:
                        # Calculate how weights drifted based on returns
                        prev_weights = self.last_period_weights[strategy_name]
                        period_returns = prev_period_data.sum()  # Sum of daily returns for the period

                        # Calculate new weights after drift: w_new = w_old * (1 + r) / sum(w_old * (1 + r))
                        asset_values = prev_weights * (1 + period_returns)
                        current_period_weights = asset_values / asset_values.sum()

                        logging.debug(f"BuyAndHold drift: prev_weights={prev_weights} -> drifted_weights={current_period_weights}")
                    else:
                        current_period_weights = self.baseline_weights.copy()
                else:
                    current_period_weights = self.baseline_weights.copy()
            else:
                # Other strategies use baseline weights or their own logic
                current_period_weights = self.baseline_weights.copy()

            # Calculate target weights using the strategy
            if strategy_name == 'spy_only':
                # SpyOnlyStrategy needs period returns as a series
                period_returns_series = period_data.mean()  # Use mean returns as proxy
                weights = strategy.calculate_target_weights(
                    current_weights=current_period_weights,
                    period_returns=period_returns_series,
                    current_date=period_end,
                    baseline_weights=self.baseline_weights
                )
            else:
                weights = strategy.calculate_target_weights(
                    current_weights=current_period_weights,
                    period_returns=period_data.mean(),  # Use mean returns as proxy
                    current_date=period_end,
                    target_weights=self.baseline_weights
                )

            current_weights[strategy_name] = weights

            # Store weights for next period (especially important for buy-and-hold drift)
            self.last_period_weights[strategy_name] = weights.copy()

        # Get weights from optimization strategies
        optimization_success = {}
        if period_num >= self.config.min_history_periods:
            # Sufficient history for optimization
            for method in self.config.optimization_methods:
                try:
                    # Get historical data for optimization (expanding window)
                    lookback_data = self._get_lookback_data(period_num)

                    if len(lookback_data) >= self.config.min_history_periods:
                        # Calculate mean returns and covariance
                        mean_returns = lookback_data.mean() * 252  # Annualize
                        cov_matrix = self.data.get_covariance_matrix(
                            lookback_data, method=self.config.covariance_method
                        )

                        # Optimize portfolio
                        result = self.optimizer.optimize(
                            method=method,
                            mean_returns=mean_returns,
                            cov_matrix=cov_matrix,
                            risk_aversion=self.config.risk_aversion,
                            min_weight=self.config.min_weight,
                            max_weight=self.config.max_weight
                        )

                        if result['success']:
                            current_weights[method] = result['weights'].values
                            optimization_success[method] = True
                            logging.info(f"Successfully optimized {method} portfolio for period {period_num}")
                        else:
                            current_weights[method] = self.baseline_weights.copy()
                            optimization_success[method] = False
                            logging.warning(f"Optimization failed for {method}, using baseline weights")
                    else:
                        current_weights[method] = self.baseline_weights.copy()
                        optimization_success[method] = False

                except Exception as e:
                    logging.error(f"Error optimizing {method} portfolio: {str(e)}")
                    current_weights[method] = self.baseline_weights.copy()
                    optimization_success[method] = False
        else:
            # Use baseline weights for early periods
            logging.info(f"Using baseline weights for optimization methods in period {period_num} (insufficient history)")
            for method in self.config.optimization_methods:
                current_weights[method] = self.baseline_weights.copy()
                optimization_success[method] = True

        return current_weights

    def _get_previous_period_data(self, period_num: int) -> Optional[pd.DataFrame]:
        """Get data from a specific previous period for drift calculations."""
        if self.period_manager is None or period_num < 0:
            return None

        try:
            # Get the period data for the specified period
            period_data = self.period_manager.get_period_data(period_num)
            return period_data
        except (IndexError, ValueError):
            return None

    def _get_lookback_data(self, period_num: int) -> pd.DataFrame:
        """Get lookback data for optimization (expanding window)."""
        if period_num <= 0:
            # For first period, use minimal data
            return self.returns_data.iloc[:30]  # Use first 30 days

        # Get the end date of the previous period
        period_info = self.period_manager.get_period_info(period_num - 1)
        end_date = period_info['period_end']

        if self.config.use_expanding_window:
            # Use all data from start up to previous period end
            lookback_data = self.returns_data[self.returns_data.index <= end_date]
        else:
            # Use rolling window - approximate 6 months back
            window_days = self.config.rolling_window_periods * 20  # Approx 20 trading days per month
            start_date = end_date - pd.Timedelta(days=window_days)
            lookback_data = self.returns_data[
                (self.returns_data.index >= start_date) &
                (self.returns_data.index <= end_date)
            ]

        return lookback_data

    def _calculate_portfolio_performance(self, period_data: pd.DataFrame,
                                       weights: np.ndarray, portfolio_name: str) -> Dict[str, Any]:
        """Calculate portfolio performance for a period."""
        if period_data.empty:
            return {
                'period_return': 0.0,
                'daily_returns': pd.Series([], dtype=float),
                'notes': 'No trading data for period'
            }

        # Calculate daily portfolio returns
        daily_portfolio_returns = period_data.dot(weights)

        # Calculate period return (cumulative)
        period_return = (1 + daily_portfolio_returns).prod() - 1

        return {
            'period_return': period_return,
            'daily_returns': daily_portfolio_returns,
            'notes': f'Period return calculated for {portfolio_name}'
        }

    # Performance Analysis Methods
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