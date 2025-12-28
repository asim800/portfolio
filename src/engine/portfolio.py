#!/usr/bin/env python3
"""
Portfolio Class - Single portfolio managing weights, returns, and metrics over time.

Supports both static portfolios (buy-and-hold, target weight, equal weight) and
dynamic portfolios (optimization-based rebalancing).
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import logging
import os

from src.strategies.allocation import (
    AllocationStrategy, StaticAllocation, EqualWeight,
    SingleAsset, OptimizedAllocation
)
from src.strategies.rebalancing import (
    RebalancingTrigger, Never, Periodic, Threshold,
    EventDriven, Combined
)
from src.engine.optimizer import PortfolioOptimizer
from src.data import FinData


class Portfolio:
    """
    Single portfolio managing weights, returns, and performance metrics over time.

    Designed with simplicity and clarity in mind:
    - Uses pandas throughout for labeled, debuggable data
    - Handles both static and dynamic rebalancing strategies
    - Tracks complete history for analysis
    - Supports real and simulated data ingestion
    """

    def __init__(self,
                 asset_names: List[str],
                 initial_weights: pd.Series,
                 allocation_strategy: AllocationStrategy,
                 rebalancing_trigger: RebalancingTrigger,
                 name: str = "portfolio"):
        """
        Initialize portfolio with allocation strategy and rebalancing trigger.

        Note: Typically use factory methods instead of calling this directly.

        Parameters:
        -----------
        asset_names : List[str]
            List of asset tickers
        initial_weights : pd.Series
            Initial portfolio weights (must have asset_names as index)
        allocation_strategy : AllocationStrategy
            Strategy defining WHAT weights to use (e.g., StaticAllocation, OptimizedAllocation)
        rebalancing_trigger : RebalancingTrigger
            Trigger defining WHEN to rebalance (e.g., Never, Periodic, Threshold)
        name : str
            Portfolio name for identification
        """
        self.name = name
        self.asset_names = asset_names
        self.allocation_strategy = allocation_strategy
        self.rebalancing_trigger = rebalancing_trigger

        # Validate initial weights
        if not isinstance(initial_weights, pd.Series):
            raise ValueError("initial_weights must be a pandas Series")
        if not all(asset in initial_weights.index for asset in asset_names):
            raise ValueError("initial_weights must have all asset_names in index")

        # Normalize weights to sum to 1
        initial_weights = initial_weights / initial_weights.sum()

        # History tracking (all pandas DataFrames/Series with datetime index)
        self.weights_history = pd.DataFrame(columns=asset_names)  # dates × assets
        self.returns_history = pd.Series(dtype=float, name='portfolio_return')  # portfolio returns
        self.portfolio_values = pd.Series(dtype=float, name='portfolio_value')  # cumulative value
        self.metrics_history = pd.DataFrame()  # dates × metrics

        # Data source (returns data for backtesting)
        self.returns_data: Optional[pd.DataFrame] = None

        # Current state
        self.current_weights = initial_weights.copy()
        self.current_value = 100.0  # Start with portfolio value of 100

        # Store initial weights
        self._add_weights(pd.Timestamp.now(), initial_weights)

        logging.info(f"Portfolio '{self.name}' initialized with {len(asset_names)} assets")

    # =========================================================================
    # FACTORY METHODS - User-friendly portfolio creation
    # =========================================================================

    @classmethod
    def create_buy_and_hold(cls,
                           asset_names: List[str],
                           initial_weights: pd.Series,
                           name: str = "buy_and_hold") -> 'Portfolio':
        """
        Create a buy-and-hold portfolio that never rebalances.

        Weights drift naturally with market performance - share counts stay fixed.

        Parameters:
        -----------
        asset_names : List[str]
            List of asset tickers
        initial_weights : pd.Series
            Initial portfolio allocation (asset → weight)
        name : str
            Portfolio name

        Returns:
        --------
        Portfolio instance with buy-and-hold strategy

        Example:
        --------
        >>> weights = pd.Series([0.6, 0.4], index=['SPY', 'AGG'])
        >>> portfolio = Portfolio.create_buy_and_hold(['SPY', 'AGG'], weights)
        """
        allocation = StaticAllocation(initial_weights.values, name="static")
        trigger = Never()
        return cls(asset_names, initial_weights, allocation, trigger, name=name)

    @classmethod
    def create_target_weight(cls,
                            asset_names: List[str],
                            target_weights: pd.Series,
                            rebalance_frequency: str = 'ME',
                            name: str = "target_weight") -> 'Portfolio':
        """
        Create a portfolio that rebalances to target weights periodically.

        Parameters:
        -----------
        asset_names : List[str]
            List of asset tickers
        target_weights : pd.Series
            Target portfolio allocation to maintain
        rebalance_frequency : str
            Rebalancing frequency ('D', 'W', '2W', 'ME', 'QE', 'YE')
            Default: 'ME' (month-end)
        name : str
            Portfolio name

        Returns:
        --------
        Portfolio instance with target weight strategy

        Example:
        --------
        >>> weights = pd.Series([0.6, 0.4], index=['SPY', 'AGG'])
        >>> portfolio = Portfolio.create_target_weight(
        ...     ['SPY', 'AGG'], weights, rebalance_frequency='ME'
        ... )
        """
        allocation = StaticAllocation(target_weights.values, name="target")
        trigger = Periodic(rebalance_frequency)
        return cls(asset_names, target_weights, allocation, trigger, name=name)

    @classmethod
    def create_equal_weight(cls,
                           asset_names: List[str],
                           rebalance_frequency: str = 'ME',
                           name: str = "equal_weight") -> 'Portfolio':
        """
        Create an equal-weight portfolio (1/N allocation).

        Rebalances to equal weights across all assets periodically.

        Parameters:
        -----------
        asset_names : List[str]
            List of asset tickers
        rebalance_frequency : str
            Rebalancing frequency ('D', 'W', '2W', 'ME', 'QE', 'YE')
            Default: 'ME' (month-end)
        name : str
            Portfolio name

        Returns:
        --------
        Portfolio instance with equal weight strategy

        Example:
        --------
        >>> portfolio = Portfolio.create_equal_weight(['SPY', 'AGG', 'BIL'])
        """
        initial_weights = pd.Series(
            np.ones(len(asset_names)) / len(asset_names),
            index=asset_names
        )
        allocation = EqualWeight(name="equal_weight")
        trigger = Periodic(rebalance_frequency)
        return cls(asset_names, initial_weights, allocation, trigger, name=name)

    @classmethod
    def create_spy_only(cls,
                       asset_names: List[str],
                       rebalance_frequency: str = 'ME',
                       name: str = "spy_only") -> 'Portfolio':
        """
        Create a portfolio that allocates 100% to SPY.

        Useful as a market benchmark for comparison.

        Parameters:
        -----------
        asset_names : List[str]
            List of asset tickers (must include 'SPY')
        rebalance_frequency : str
            Rebalancing frequency ('D', 'W', '2W', 'ME', 'QE', 'YE')
            Default: 'ME' (month-end)
        name : str
            Portfolio name

        Returns:
        --------
        Portfolio instance with SPY-only allocation

        Example:
        --------
        >>> portfolio = Portfolio.create_spy_only(['SPY', 'AGG', 'BIL'])
        """
        # Find SPY index
        spy_idx = None
        for i, asset in enumerate(asset_names):
            if 'SPY' in asset.upper():
                spy_idx = i
                break

        if spy_idx is None:
            raise ValueError("SPY not found in asset_names")

        initial_weights = pd.Series(np.zeros(len(asset_names)), index=asset_names)
        initial_weights.iloc[spy_idx] = 1.0

        allocation = SingleAsset(spy_idx, len(asset_names), name="SPY_only")
        trigger = Periodic(rebalance_frequency)
        return cls(asset_names, initial_weights, allocation, trigger, name=name)

    @classmethod
    def create_optimized(cls,
                        asset_names: List[str],
                        initial_weights: pd.Series,
                        optimizer: PortfolioOptimizer,
                        method: str = 'mean_variance',
                        rebalance_frequency: str = 'ME',
                        name: Optional[str] = None) -> 'Portfolio':
        """
        Create a portfolio that uses optimization to rebalance periodically.

        Parameters:
        -----------
        asset_names : List[str]
            List of asset tickers
        initial_weights : pd.Series
            Initial portfolio allocation (used until first optimization)
        optimizer : PortfolioOptimizer
            Optimizer instance to use for rebalancing
        method : str
            Optimization method ('mean_variance', 'robust_mean_variance', etc.)
        rebalance_frequency : str
            Rebalancing frequency ('D', 'W', '2W', 'ME', 'QE', 'YE')
            Default: 'ME' (month-end)
        name : str, optional
            Portfolio name (defaults to 'optimized_{method}')

        Returns:
        --------
        Portfolio instance with optimization-based rebalancing

        Example:
        --------
        >>> optimizer = PortfolioOptimizer(risk_free_rate=0.02)
        >>> weights = pd.Series([0.6, 0.4], index=['SPY', 'AGG'])
        >>> portfolio = Portfolio.create_optimized(
        ...     ['SPY', 'AGG'], weights, optimizer,
        ...     method='mean_variance', rebalance_frequency='ME'
        ... )
        """
        if name is None:
            name = f"optimized_{method}"

        allocation = OptimizedAllocation(optimizer, method, name=f"opt_{method}")
        trigger = Periodic(rebalance_frequency)

        return cls(asset_names, initial_weights, allocation, trigger, name=name)

    # =========================================================================
    # DATA INGESTION
    # =========================================================================

    def ingest_real_data(self,
                        fin_data: FinData,
                        tickers: List[str],
                        start_date: str,
                        end_date: str) -> None:
        """
        Ingest real market data from FinData object.

        Parameters:
        -----------
        fin_data : FinData
            FinData instance with market data
        tickers : List[str]
            List of tickers to load
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        """
        # Load returns data
        self.returns_data = fin_data.get_returns_data(tickers)

        # Filter by date range
        self.returns_data = self.returns_data[
            (self.returns_data.index >= start_date) &
            (self.returns_data.index <= end_date)
        ]

        logging.info(f"{self.name}: Ingested {len(self.returns_data)} days of real market data")

    def ingest_simulated_data(self, mc_returns: pd.DataFrame) -> None:
        """
        Ingest simulated returns data (e.g., from Monte Carlo simulation).

        Parameters:
        -----------
        mc_returns : pd.DataFrame
            Simulated returns data (dates × assets)
            Must have same assets as self.asset_names
        """
        # Validate columns match
        if not all(asset in mc_returns.columns for asset in self.asset_names):
            raise ValueError("mc_returns must contain all assets in portfolio")

        self.returns_data = mc_returns[self.asset_names].copy()

        logging.info(f"{self.name}: Ingested {len(self.returns_data)} days of simulated data")

    # =========================================================================
    # BACKTESTING AND REBALANCING
    # =========================================================================

    def run_backtest(self, period_manager) -> None:
        """
        Run backtest across all rebalancing periods using new architecture.

        Parameters:
        -----------
        period_manager : PeriodManager
            Period manager with rebalancing schedule
        """
        if self.returns_data is None:
            raise RuntimeError("Must ingest data before running backtest")

        logging.info(f"{self.name}: Starting backtest with {period_manager.num_periods} periods")

        # Reset history for clean backtest
        self.weights_history = pd.DataFrame(columns=self.asset_names)
        self.returns_history = pd.Series(dtype=float, name='portfolio_return')
        self.portfolio_values = pd.Series(dtype=float, name='portfolio_value')

        # Start with initial value
        current_value = 100.0
        current_weights = self.current_weights.copy()

        # Track for buy-and-hold drift
        last_period_weights = current_weights.copy()

        for period_num, period_data, period_info in period_manager.iter_periods():
            period_start = period_info['period_start']
            period_end = period_info['period_end']

            # Check if we should rebalance at period start
            should_rebalance = False
            if period_num == 0:
                # First period - use initial weights (no rebalancing check)
                rebalanced_weights = current_weights.copy()
            else:
                # Check trigger for rebalancing
                should_rebalance = self.rebalancing_trigger.should_rebalance(
                    current_date=period_end.date(),
                    current_weights=current_weights.values,
                    target_weights=self.current_weights.values,
                    returns_data=self.returns_data.loc[:period_end],
                    portfolio_values=self.portfolio_values
                )

                if should_rebalance:
                    # Get lookback data for optimization strategies
                    lookback_data = period_manager.get_expanding_window_data(period_num)

                    # Calculate new weights from allocation strategy
                    new_weights_array = self.allocation_strategy.calculate_weights(
                        current_weights=current_weights.values,
                        lookback_data=lookback_data
                    )
                    rebalanced_weights = pd.Series(new_weights_array, index=self.asset_names)

                    # Record rebalancing event
                    self.rebalancing_trigger.record_rebalance(period_end.date())

                    logging.debug(f"{self.name} Period {period_num}: Rebalanced from {current_weights.values} to {rebalanced_weights.values}")
                else:
                    # No rebalancing - weights drift naturally with market
                    if period_num > 0:
                        prev_period_data = period_manager.get_period_data(period_num - 1)
                        if not prev_period_data.empty:
                            # Calculate cumulative return for each asset over previous period
                            asset_returns = (1 + prev_period_data).prod() - 1

                            # Calculate new weights after drift: w_new = w_old * (1 + r) / sum(w_old * (1 + r))
                            asset_values = last_period_weights * (1 + asset_returns)
                            rebalanced_weights = asset_values / asset_values.sum()
                        else:
                            rebalanced_weights = last_period_weights.copy()
                    else:
                        rebalanced_weights = current_weights.copy()

            # Record weights at period start
            self._add_weights(period_end, rebalanced_weights)

            # Calculate portfolio return for the period
            period_return = self.calculate_period_return(period_data, rebalanced_weights)

            # Update portfolio value
            current_value *= (1 + period_return)

            # Record performance
            self.returns_history.loc[period_end] = period_return
            self.portfolio_values.loc[period_end] = current_value

            # Update state for next period
            current_weights = rebalanced_weights.copy()
            last_period_weights = rebalanced_weights.copy()

            logging.debug(f"{self.name} Period {period_num}: return={period_return:.2%}, value={current_value:.2f}")

        logging.info(f"{self.name}: Backtest complete. Final value: {current_value:.2f}")

    def calculate_period_return(self,
                               period_returns: pd.DataFrame,
                               weights: pd.Series) -> float:
        """
        Calculate portfolio return for a period given returns and weights.

        Parameters:
        -----------
        period_returns : pd.DataFrame
            Daily returns for the period (dates × assets)
        weights : pd.Series
            Portfolio weights to use

        Returns:
        --------
        float : Cumulative portfolio return for the period
        """
        if period_returns.empty:
            return 0.0

        # Calculate daily portfolio returns
        daily_portfolio_returns = period_returns.dot(weights)

        # Compound to get period return
        period_return = (1 + daily_portfolio_returns).prod() - 1

        return period_return

    # =========================================================================
    # METRICS AND REPORTING
    # =========================================================================

    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Calculate summary statistics for the portfolio.

        Returns:
        --------
        pd.DataFrame with performance metrics
        """
        if len(self.returns_history) == 0:
            return pd.DataFrame()

        # Calculate cumulative returns
        cumulative_returns = (1 + self.returns_history).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0

        # Annualize returns (assuming ~monthly periods)
        if len(cumulative_returns) > 1:
            start_date = cumulative_returns.index[0]
            end_date = cumulative_returns.index[-1]
            years = (end_date - start_date).days / 365.25
            annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        else:
            annual_return = 0

        # Volatility (annualized, assuming ~monthly returns)
        periods_per_year = 12
        volatility = self.returns_history.std() * np.sqrt(periods_per_year)

        # Sharpe ratio
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

        # Max drawdown
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()

        summary = pd.DataFrame({
            'Portfolio': [self.name],
            'Total_Return': [total_return],
            'Annual_Return': [annual_return],
            'Volatility': [volatility],
            'Sharpe_Ratio': [sharpe_ratio],
            'Max_Drawdown': [max_drawdown],
            'Num_Periods': [len(self.returns_history)]
        })

        return summary.set_index('Portfolio')

    def get_current_weights(self) -> pd.Series:
        """Get current portfolio weights."""
        return self.current_weights.copy()

    def get_weights_at_date(self, target_date: pd.Timestamp) -> pd.Series:
        """
        Get portfolio weights at a specific date.

        Parameters:
        -----------
        target_date : pd.Timestamp
            Date to get weights for

        Returns:
        --------
        pd.Series : Portfolio weights at that date (or most recent before)
        """
        if target_date in self.weights_history.index:
            return self.weights_history.loc[target_date]

        # Find most recent weights before this date
        available_dates = self.weights_history.index[self.weights_history.index <= target_date]
        if len(available_dates) > 0:
            return self.weights_history.loc[available_dates[-1]]

        return pd.Series(0.0, index=self.asset_names)

    def export_to_csv(self, output_dir: str) -> None:
        """
        Export portfolio data to CSV files.

        Parameters:
        -----------
        output_dir : str
            Directory to save CSV files
        """
        os.makedirs(output_dir, exist_ok=True)

        # Export weights history
        weights_file = os.path.join(output_dir, f'{self.name}_weights.csv')
        self.weights_history.to_csv(weights_file)

        # Export returns history
        returns_file = os.path.join(output_dir, f'{self.name}_returns.csv')
        self.returns_history.to_csv(returns_file)

        # Export portfolio values
        values_file = os.path.join(output_dir, f'{self.name}_values.csv')
        self.portfolio_values.to_csv(values_file)

        # Export summary statistics
        summary = self.get_summary_statistics()
        summary_file = os.path.join(output_dir, f'{self.name}_summary.csv')
        summary.to_csv(summary_file)

        logging.info(f"{self.name}: Exported results to {output_dir}")

    # =========================================================================
    # INTERNAL HELPER METHODS
    # =========================================================================

    def _add_weights(self, date: pd.Timestamp, weights: pd.Series) -> None:
        """Record weights at a specific date."""
        self.weights_history.loc[date] = weights
        self.current_weights = weights.copy()
