#!/usr/bin/env python3
"""
Dynamic portfolio rebalancing engine.
Implements periodic rebalancing with configurable optimization strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, date, timedelta
import logging
import os

from config import RebalancingConfig
from performance_tracker import PerformanceTracker
from portfolio_optimizer import PortfolioOptimizer
from fin_data import FinData
from rebalancing_strategies import RebalancingStrategies


class RebalancingEngine:
    """
    Main engine for dynamic portfolio rebalancing.
    
    Handles data splitting, optimization, and performance tracking
    across multiple rebalancing periods.
    """
    
    def __init__(self, config: RebalancingConfig):
        """Initialize rebalancing engine with configuration."""
        self.config = config
        self.performance_tracker = PerformanceTracker(config)
        
        # Initialize portfolio optimizer and data handler
        self.portfolio_optimizer = PortfolioOptimizer(risk_free_rate=config.risk_free_rate)
        self.fin_data = FinData(config.start_date, config.end_date)
        
        # Initialize rebalancing strategies
        self.rebalancing_factory = RebalancingStrategies()
        self.active_strategies: Dict[str, Any] = {}  # Will store strategy instances
        
        # Data storage
        self.returns_data: Optional[pd.DataFrame] = None
        self.asset_names: List[str] = []
        self.baseline_weights: Optional[np.ndarray] = None
        
        # Period management
        self.period_dates: List[Tuple[date, date]] = []
        self.current_period = 0
        
        # Portfolio weights history
        self.weights_history: Dict[str, List[np.ndarray]] = {}
        
        logging.info(f"RebalancingEngine initialized with {config.rebalancing_period_days}-day periods")
    
    def load_data(self, returns_data: pd.DataFrame, baseline_weights: np.ndarray) -> None:
        """
        Load returns data and baseline weights for backtesting.
        
        Parameters:
        -----------
        returns_data : pd.DataFrame
            Daily returns data with assets as columns and dates as index
        baseline_weights : np.ndarray
            Baseline portfolio weights
        """
        self.returns_data = returns_data.copy()
        self.asset_names = list(returns_data.columns)
        self.baseline_weights = baseline_weights.copy()
        
        # Validate data
        if len(baseline_weights) != len(self.asset_names):
            raise ValueError(f"Baseline weights length ({len(baseline_weights)}) "
                           f"doesn't match number of assets ({len(self.asset_names)})")
        
        # Get all portfolio names from configuration
        portfolio_names = self.config.get_portfolio_names()
        
        # Initialize weights history for all portfolio types
        for portfolio_name in portfolio_names:
            self.weights_history[portfolio_name] = []
        
        # Initialize rebalancing strategies
        self._initialize_rebalancing_strategies(baseline_weights)
        
        logging.info(f"Loaded data: {len(returns_data)} days, {len(self.asset_names)} assets")
        logging.info(f"Data range: {returns_data.index[0]} to {returns_data.index[-1]}")
    
    def _initialize_rebalancing_strategies(self, baseline_weights: np.ndarray) -> None:
        """Initialize rebalancing strategy instances."""
        for strategy_name in self.config.rebalancing_strategies:
            period = self.config.get_rebalancing_period(strategy_name)
            
            if strategy_name == 'target_weight':
                # Target weight strategy uses the baseline weights as targets
                strategy = self.rebalancing_factory.create_strategy(
                    strategy_name,
                    rebalancing_period_days=period,
                    target_weights=baseline_weights
                )
            else:
                # Other strategies don't need specific target weights
                strategy = self.rebalancing_factory.create_strategy(
                    strategy_name,
                    rebalancing_period_days=period
                )
            
            self.active_strategies[strategy_name] = strategy
            logging.info(f"Initialized {strategy_name} strategy with {period}-day period")
    
    def split_data_into_periods(self) -> List[Tuple[date, date]]:
        """
        Split the data timeline into rebalancing periods.
        
        Returns:
        --------
        List of (start_date, end_date) tuples for each period
        """
        if self.returns_data is None:
            raise ValueError("Data must be loaded before splitting into periods")
        
        start_date = self.returns_data.index[0].date()
        end_date = self.returns_data.index[-1].date()
        
        periods = []
        current_start = start_date
        
        while current_start <= end_date:
            current_end = current_start + timedelta(days=self.config.rebalancing_period_days - 1)
            
            # Don't go beyond the available data
            if current_end > end_date:
                current_end = end_date
            
            periods.append((current_start, current_end))
            
            # Move to next period
            current_start = current_end + timedelta(days=1)
            
            # Break if we've reached the end
            if current_start > end_date:
                break
        
        self.period_dates = periods
        logging.info(f"Split data into {len(periods)} periods of ~{self.config.rebalancing_period_days} days")
        
        return periods
    
    def get_expanding_window_data(self, current_period: int) -> pd.DataFrame:
        """
        Get expanding window data for optimization.
        
        Parameters:
        -----------
        current_period : int
            Current period number (0-based)
            
        Returns:
        --------
        DataFrame with returns data from start through end of previous period
        """
        if current_period == 0:
            return pd.DataFrame()  # No historical data for first period
        
        # Get data from start through end of previous period
        end_date = self.period_dates[current_period - 1][1]
        
        # Convert to datetime for pandas indexing
        end_datetime = pd.Timestamp(end_date)
        
        # Get all data up to (and including) end_date
        historical_data = self.returns_data[self.returns_data.index <= end_datetime]
        
        return historical_data
    
    def optimize_portfolio(self, 
                         historical_data: pd.DataFrame, 
                         method: str) -> Tuple[np.ndarray, bool]:
        """
        Optimize portfolio weights using specified method.
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            Historical returns data for optimization
        method : str
            Optimization method ('vanilla' or 'robust')
            
        Returns:
        --------
        Tuple of (weights, success_flag)
        """
        if historical_data.empty:
            logging.warning(f"No historical data available for {method} optimization")
            return np.zeros(len(self.asset_names)), False
        
        if len(historical_data) < 5:  # Minimum data requirement
            logging.warning(f"Insufficient data for {method} optimization ({len(historical_data)} days)")
            return np.zeros(len(self.asset_names)), False
        
        try:
            # Calculate mean returns
            mean_returns = historical_data.mean().values
            
            # Calculate covariance matrix using configured method
            cov_matrix = self.fin_data.get_covariance_matrix(
                historical_data,
                method=self.config.covariance_method,
                **self.config.get_covariance_params()
            )
            
            # Normalize method name (handle legacy names)
            normalized_method = self.config.normalize_method_name(method)
            
            # Get optimization parameters
            opt_params = self.config.get_optimization_params(method)
            
            # Run optimization using new PortfolioOptimizer
            result = self.portfolio_optimizer.optimize(
                method=normalized_method,
                mean_returns=mean_returns,
                cov_matrix=cov_matrix,
                **opt_params
            )
            
            if result.get('status') == 'optimal':
                return result['weights'], True
            else:
                logging.warning(f"{method} optimization failed with status: {result.get('status', 'unknown')}")
                return np.zeros(len(self.asset_names)), False
                
        except Exception as e:
            logging.error(f"Error in {method} optimization: {e}")
            return np.zeros(len(self.asset_names)), False
    
    def get_period_data(self, period_number: int) -> pd.DataFrame:
        """Get returns data for a specific period."""
        start_date, end_date = self.period_dates[period_number]
        
        # Convert to datetime for pandas indexing
        start_datetime = pd.Timestamp(start_date)
        end_datetime = pd.Timestamp(end_date)
        
        # Get period data
        period_mask = (self.returns_data.index >= start_datetime) & (self.returns_data.index <= end_datetime)
        period_data = self.returns_data[period_mask]
        
        return period_data
    
    def calculate_portfolio_performance(self, 
                                      weights: np.ndarray, 
                                      period_data: pd.DataFrame,
                                      portfolio_name: str) -> Dict[str, Any]:
        """
        Calculate performance metrics for a portfolio over a period.
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        period_data : pd.DataFrame
            Returns data for the period
        portfolio_name : str
            Name of the portfolio
            
        Returns:
        --------
        Dictionary with performance metrics
        """
        if period_data.empty:
            return {
                'weights': weights,
                'total_weight': np.sum(weights),
                'optimization_successful': False,
                'notes': 'No period data available'
            }
        
        # Calculate portfolio returns
        portfolio_returns = (period_data * weights).sum(axis=1)
        
        # Get market proxy for beta calculation
        market_returns = self._get_market_proxy_returns(period_data)
        
        # Calculate metrics
        metrics = self.performance_tracker.calculate_period_metrics(
            portfolio_returns=portfolio_returns,
            market_returns=market_returns,
            risk_free_rate=self.config.risk_free_rate
        )
        
        # Add portfolio-specific info
        metrics.update({
            'weights': weights,
            'total_weight': np.sum(weights),
            'optimization_successful': True,
            'notes': ''
        })
        
        return metrics
    
    def _get_market_proxy_returns(self, period_data: pd.DataFrame) -> pd.Series:
        """Get market proxy returns for beta calculation."""
        # Try to find SPY first
        if 'SPY' in period_data.columns:
            return period_data['SPY']
        elif 'SPX' in period_data.columns:
            return period_data['SPX']
        else:
            # Use equal-weighted portfolio as proxy
            equal_weights = np.ones(len(period_data.columns)) / len(period_data.columns)
            return (period_data * equal_weights).sum(axis=1)
    
    def calculate_mixed_portfolio_weights(self, equity_weights: np.ndarray, cash_percentage: float) -> np.ndarray:
        """
        Calculate mixed portfolio weights combining cash and equity weights.
        
        Parameters:
        -----------
        equity_weights : np.ndarray
            Pure equity portfolio weights
        cash_percentage : float
            Target cash allocation (e.g., 0.4 for 40% cash)
            
        Returns:
        --------
        Mixed portfolio weights where equity portion is scaled by (1 - cash_percentage)
        Note: Cash component is tracked separately, not in these weights
        """
        equity_percentage = 1.0 - cash_percentage
        return equity_weights * equity_percentage
    
    def calculate_mixed_portfolio_performance(self, 
                                            equity_weights: np.ndarray,
                                            period_data: pd.DataFrame,
                                            cash_percentage: float,
                                            cash_return: float,
                                            portfolio_name: str) -> Dict[str, Any]:
        """
        Calculate performance for mixed cash/equity portfolio.
        
        Parameters:
        -----------
        equity_weights : np.ndarray
            Scaled equity weights (already adjusted for cash percentage)
        period_data : pd.DataFrame
            Returns data for the period
        cash_percentage : float
            Actual cash allocation percentage
        cash_return : float
            Return on cash for this period
        portfolio_name : str
            Name of the portfolio
            
        Returns:
        --------
        Dictionary with performance metrics
        """
        if period_data.empty:
            return {
                'weights': equity_weights,
                'total_weight': np.sum(equity_weights) + cash_percentage,
                'optimization_successful': False,
                'notes': 'No period data available',
                'cash_percentage': cash_percentage,
                'equity_percentage': 1.0 - cash_percentage
            }
        
        # Calculate equity portfolio returns
        equity_returns = (period_data * equity_weights).sum(axis=1)
        
        # Calculate blended portfolio returns (equity + cash)
        blended_returns = equity_returns + cash_return * cash_percentage
        
        # Get market proxy for beta calculation
        market_returns = self._get_market_proxy_returns(period_data)
        
        # Calculate metrics using blended returns
        metrics = self.performance_tracker.calculate_period_metrics(
            portfolio_returns=blended_returns,
            market_returns=market_returns,
            risk_free_rate=self.config.risk_free_rate
        )
        
        # Add mixed portfolio specific info
        metrics.update({
            'weights': equity_weights,
            'total_weight': np.sum(equity_weights) + cash_percentage,
            'optimization_successful': True,
            'notes': f'Mixed portfolio: {cash_percentage:.1%} cash, {1-cash_percentage:.1%} equity',
            'cash_percentage': cash_percentage,
            'equity_percentage': 1.0 - cash_percentage
        })
        
        return metrics
    
    def run_backtest(self) -> PerformanceTracker:
        """
        Run the complete rebalancing backtest.
        
        Returns:
        --------
        PerformanceTracker with all results
        """
        if self.returns_data is None:
            raise ValueError("Data must be loaded before running backtest")
        
        # Split data into periods
        self.split_data_into_periods()
        
        if len(self.period_dates) < self.config.min_history_periods + 1:
            raise ValueError(f"Insufficient data: need at least {self.config.min_history_periods + 1} periods, "
                           f"got {len(self.period_dates)}")
        
        logging.info(f"Starting backtest with {len(self.period_dates)} periods")
        
        # Initialize cumulative returns tracking for all portfolio types
        portfolio_names = self.config.get_portfolio_names()
        cumulative_returns = {name: 1.0 for name in portfolio_names}
        self.cumulative_returns_tracker = cumulative_returns  # Store for table display
        
        # Run through each period
        for period_num in range(len(self.period_dates)):
            start_date, end_date = self.period_dates[period_num]
            
            logging.info(f"Processing period {period_num}: {start_date} to {end_date}")
            
            # Get period data for performance calculation
            period_data = self.get_period_data(period_num)
            
            if period_data.empty:
                logging.warning(f"No data available for period {period_num}")
                continue
            
            # Determine weights to use for this period
            current_weights = {}
            optimization_success = {}
            
            # Execute rebalancing strategies
            for strategy_name, strategy in self.active_strategies.items():
                # Special handling for buy_and_hold to track actual weight drift
                if strategy_name == 'buy_and_hold':
                    current_weights[strategy_name] = self._calculate_buy_and_hold_weights(period_num, period_data)
                    optimization_success[strategy_name] = True
                    
                    # Log that buy_and_hold doesn't rebalance (except first period setup)
                    if period_num == 0:
                        logging.info(f"{strategy_name} established initial position")
                    else:
                        logging.info(f"{strategy_name} weights drifted naturally (no rebalancing)")
                else:
                    # Get current weights (from previous period or baseline)
                    if period_num == 0:
                        # First period - use baseline weights
                        prev_weights = self.baseline_weights.copy()
                    else:
                        # Get weights from previous period (after any portfolio drift)
                        prev_weights = self.weights_history[strategy_name][-1].copy()
                    
                    # Get period returns for strategy decision-making
                    period_returns = period_data.mean() if not period_data.empty else pd.Series()
                    
                    # Execute rebalancing strategy
                    strategy_weights, rebalancing_info = strategy.execute_rebalancing(
                        current_weights=prev_weights,
                        period_returns=period_returns,
                        current_date=end_date,
                        baseline_weights=self.baseline_weights
                    )
                    
                    current_weights[strategy_name] = strategy_weights
                    optimization_success[strategy_name] = True  # Strategies always succeed
                    
                    # Log rebalancing activity
                    if rebalancing_info['rebalanced']:
                        logging.info(f"{strategy_name} rebalanced on {rebalancing_info['rebalance_date']}, "
                                   f"turnover: {rebalancing_info['total_turnover']:.3f}")
            
            # Add legacy baseline portfolios if configured
            if self.config.include_baseline and 'buy_and_hold' not in self.config.rebalancing_strategies:
                current_weights['static_baseline'] = self.baseline_weights
                optimization_success['static_baseline'] = True
            
            if self.config.include_rebalanced_baseline:
                current_weights['rebalanced_baseline'] = self.baseline_weights  # Legacy behavior
                optimization_success['rebalanced_baseline'] = True
            
            # Handle optimization methods  
            if period_num < self.config.min_history_periods:
                # Use baseline weights for optimization methods during initial periods
                logging.info(f"Using baseline weights for optimization methods in period {period_num} (insufficient history)")
                for method in self.config.optimization_methods:
                    current_weights[method] = self.baseline_weights
                    optimization_success[method] = True
            else:
                # Get expanding window data for optimization
                historical_data = self.get_expanding_window_data(period_num)
                
                # Run optimization for each method
                for method in self.config.optimization_methods:
                    weights, success = self.optimize_portfolio(historical_data, method)
                    current_weights[method] = weights
                    optimization_success[method] = success
                    
                    if success:
                        logging.info(f"Successfully optimized {method} portfolio for period {period_num}")
                    else:
                        logging.warning(f"Failed to optimize {method} portfolio for period {period_num}, using baseline")
                        current_weights[method] = self.baseline_weights
            
            # Update portfolio analysis tables for this period
            self._update_portfolio_analysis_tables(period_num, start_date, end_date, current_weights)
            
            # Calculate cash return for the period (needed for mixed portfolios)
            period_length_days = (end_date - start_date).days + 1
            daily_cash_rate = self.config.cash_interest_rate / 365.0
            period_cash_return = daily_cash_rate * period_length_days
            
            # Add mixed portfolios if enabled
            if self.config.include_mixed_portfolios:
                
                for method in self.config.optimization_methods:
                    if method in current_weights:
                        # Get equity weights and create mixed portfolio
                        equity_weights = current_weights[method]
                        mixed_weights = self.calculate_mixed_portfolio_weights(
                            equity_weights, 
                            self.config.mixed_cash_percentage
                        )
                        
                        mixed_portfolio_name = f'mixed_{method}'
                        current_weights[mixed_portfolio_name] = mixed_weights
                        
                        # Mixed portfolio inherits optimization success from base method
                        optimization_success[mixed_portfolio_name] = optimization_success.get(method, False)
            
            # Calculate performance for each portfolio
            period_performances = {}
            
            for portfolio_name, weights in current_weights.items():
                # Store weights
                self.weights_history[portfolio_name].append(weights.copy())
                
                # Calculate performance based on portfolio type
                if portfolio_name.startswith('mixed_'):
                    # Use special calculation for mixed portfolios
                    performance = self.calculate_mixed_portfolio_performance(
                        equity_weights=weights,
                        period_data=period_data,
                        cash_percentage=self.config.mixed_cash_percentage,
                        cash_return=period_cash_return,
                        portfolio_name=portfolio_name
                    )
                else:
                    # Regular portfolio calculation
                    performance = self.calculate_portfolio_performance(
                        weights=weights,
                        period_data=period_data,
                        portfolio_name=portfolio_name
                    )
                
                # Update cumulative returns
                period_return = performance['period_return']
                cumulative_returns[portfolio_name] *= (1 + period_return)
                performance['cumulative_return'] = cumulative_returns[portfolio_name] - 1
                
                # Add optimization status for non-baseline portfolios
                baseline_names = ['static_baseline', 'rebalanced_baseline'] + list(self.config.rebalancing_strategies)
                if portfolio_name not in baseline_names:
                    performance['optimization_successful'] = optimization_success.get(portfolio_name, False)
                
                period_performances[portfolio_name] = performance
            
            # Add to performance tracker
            self.performance_tracker.add_period_performance(
                period_number=period_num,
                start_date=start_date,
                end_date=end_date,
                portfolio_performances=period_performances
            )
        
        logging.info("Backtest completed successfully")
        
        # Display full portfolio analysis tables
        self.display_full_portfolio_tables()
        
        # Generate final summaries
        self.performance_tracker.get_performance_summary()
        self.performance_tracker.get_weights_summary(self.asset_names)
        self.performance_tracker.get_cumulative_returns_series()
        
        return self.performance_tracker
    
    def save_results(self) -> None:
        """Save all backtest results."""
        if self.config.save_results:
            os.makedirs(self.config.results_directory, exist_ok=True)
            self.performance_tracker.save_results(self.config.results_directory)
            logging.info(f"Results saved to {self.config.results_directory}")
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all portfolios."""
        return self.performance_tracker.get_portfolio_statistics()
    
    def _update_portfolio_analysis_tables(self, period_num: int, start_date: date, end_date: date, 
                                         current_weights: Dict[str, np.ndarray]) -> None:
        """
        Update portfolio analysis tables with data for the current period.
        Creates/updates pandas DataFrames for each portfolio with MultiIndex columns.
        
        Parameters:
        -----------
        period_num : int
            Period number
        start_date : date
            Period start date
        end_date : date
            Period end date
        current_weights : Dict[str, np.ndarray]
            Portfolio weights for each portfolio
        """
        if not current_weights or not self.asset_names:
            logging.info("No weights data available for portfolio table")
            return
        
        import pandas as pd
        
        # Initialize portfolio tables if not exists
        if not hasattr(self, 'portfolio_analysis_tables'):
            self.portfolio_analysis_tables = {}
        
        # Get period data for returns calculation
        period_data = self.get_period_data(period_num)
        
        if period_data.empty:
            logging.info(f"No return data available for period {period_num}")
            return
        
        # Calculate period returns for each asset (total return over the period)
        if not period_data.empty:
            # Calculate total period return (not daily average)
            period_returns = ((1 + period_data).prod() - 1).values
        else:
            period_returns = np.zeros(len(self.asset_names))
        
        # Update tables for each portfolio
        for portfolio_name, weights in current_weights.items():
            # Calculate portfolio metrics
            portfolio_return = np.sum(weights * period_returns)
            
            # Get cumulative return
            if portfolio_name in self.cumulative_returns_tracker:
                cumulative_return = self.cumulative_returns_tracker[portfolio_name] - 1
            else:
                cumulative_return = 0.0
            
            portfolio_value = self.cumulative_returns_tracker.get(portfolio_name, 1.0) * 100
            
            # Create row data for this period
            row_data = {}
            
            # Add symbol data with MultiIndex structure
            for i, symbol in enumerate(self.asset_names):
                row_data[(symbol, 'Weight')] = weights[i]
                row_data[(symbol, 'Period_Return')] = period_returns[i]
            
            # Add portfolio-level metrics
            row_data[('Portfolio', 'Period_Return')] = portfolio_return
            row_data[('Portfolio', 'Cumulative_Return')] = cumulative_return
            row_data[('Portfolio', 'Portfolio_Value')] = portfolio_value
            
            # Create/update DataFrame for this portfolio
            if portfolio_name not in self.portfolio_analysis_tables:
                # Create new DataFrame
                self.portfolio_analysis_tables[portfolio_name] = pd.DataFrame(
                    index=pd.DatetimeIndex([], name='Date'),
                    columns=pd.MultiIndex.from_tuples(
                        [(symbol, metric) for symbol in self.asset_names for metric in ['Weight', 'Period_Return']] +
                        [('Portfolio', 'Period_Return'), ('Portfolio', 'Cumulative_Return'), ('Portfolio', 'Portfolio_Value')]
                    )
                )
            
            # Add this period's data
            self.portfolio_analysis_tables[portfolio_name].loc[pd.to_datetime(start_date)] = row_data
        
        # Display the updated tables
        self._display_portfolio_analysis_tables(period_num, start_date, end_date)
    
    def _display_portfolio_analysis_tables(self, period_num: int, start_date: date, end_date: date) -> None:
        """Display the portfolio analysis tables in a readable format."""
        if not hasattr(self, 'portfolio_analysis_tables'):
            return
        
        logging.info(f"\n=== Portfolio Analysis Tables - Period {period_num} ({start_date} to {end_date}) ===")
        
        for portfolio_name, df in self.portfolio_analysis_tables.items():
            logging.info(f"\n{portfolio_name.upper()} Portfolio:")
            
            if df.empty:
                logging.info("  No data available")
                continue
            
            # Get the latest row (current period)
            latest_data = df.iloc[-1]
            
            # Display summary metrics first
            portfolio_return = latest_data[('Portfolio', 'Period_Return')]
            cumulative_return = latest_data[('Portfolio', 'Cumulative_Return')]
            portfolio_value = latest_data[('Portfolio', 'Portfolio_Value')]
            
            logging.info(f"  Period Return: {portfolio_return:.2%} | Cumulative: {cumulative_return:.2%} | Value: {portfolio_value:.1f}")
            
            # Display significant holdings (weight > 0.5%)
            logging.info("  Holdings (Weight | Period Return):")
            holdings_display = []
            
            for symbol in self.asset_names:
                weight = latest_data[(symbol, 'Weight')]
                period_ret = latest_data[(symbol, 'Period_Return')]
                
                if weight > 0.005:  # Show weights > 0.5%
                    holdings_display.append(f"    {symbol}: {weight:.1%} | {period_ret:.2%}")
            
            # Display in chunks for readability
            for i in range(0, len(holdings_display), 3):
                chunk = holdings_display[i:i+3]
                for holding in chunk:
                    logging.info(holding)
        
        logging.info("")  # Empty line for readability
    
    def _calculate_buy_and_hold_weights(self, period_num: int, period_data: pd.DataFrame) -> np.ndarray:
        """
        Calculate buy_and_hold weights based on actual share counts and price evolution.
        
        For buy_and_hold:
        - Share counts remain constant (no buying/selling)
        - Dollar values change as prices evolve
        - Percentage weights drift naturally
        
        Parameters:
        -----------
        period_num : int
            Current period number
        period_data : pd.DataFrame
            Price/return data for the period
            
        Returns:
        --------
        np.ndarray of current market-value weights after drift
        """
        # Initialize share tracking on first period
        if not hasattr(self, 'buy_and_hold_shares'):
            # Calculate initial share counts based on baseline weights
            # Assume $100 initial portfolio value
            initial_portfolio_value = 100.0
            initial_dollar_allocations = self.baseline_weights * initial_portfolio_value
            
            # For simplicity, assume we can buy fractional shares
            # Share count = dollar allocation / initial "price" (normalized to 1.0)
            self.buy_and_hold_shares = initial_dollar_allocations.copy()  # Dollar amounts become "share counts"
            self.buy_and_hold_cumulative_returns = np.ones(len(self.asset_names))  # Track cumulative price changes
            
            logging.info(f"Buy-and-hold: Established initial position with baseline weights")
            return self.baseline_weights.copy()
        
        # Calculate period returns for this period
        if not period_data.empty:
            # Calculate total period return for each asset
            period_returns = ((1 + period_data).prod() - 1).values
        else:
            period_returns = np.zeros(len(self.asset_names))
        
        # Update cumulative returns (represents cumulative price changes)
        self.buy_and_hold_cumulative_returns *= (1 + period_returns)
        
        # Calculate current dollar values: shares * cumulative price change
        current_dollar_values = self.buy_and_hold_shares * self.buy_and_hold_cumulative_returns
        
        # Calculate current portfolio weights
        total_portfolio_value = np.sum(current_dollar_values)
        current_weights = current_dollar_values / total_portfolio_value if total_portfolio_value > 0 else self.baseline_weights
        
        # Log the weight drift
        significant_drift = np.max(np.abs(current_weights - self.baseline_weights)) > 0.01  # 1% drift threshold
        if significant_drift:
            # Find assets with biggest changes
            weight_changes = current_weights - self.baseline_weights
            max_gain_idx = np.argmax(weight_changes)
            max_loss_idx = np.argmin(weight_changes)
            
            logging.info(f"Buy-and-hold weight drift: "
                        f"{self.asset_names[max_gain_idx]} +{weight_changes[max_gain_idx]:.1%}, "
                        f"{self.asset_names[max_loss_idx]} {weight_changes[max_loss_idx]:.1%}")
        
        return current_weights
    
    def display_full_portfolio_tables(self) -> None:
        """Display the full portfolio analysis tables and optionally save to CSV."""
        if not hasattr(self, 'portfolio_analysis_tables'):
            logging.info("No portfolio analysis tables available")
            return
        
        import pandas as pd
        
        logging.info("\n" + "="*80)
        logging.info("COMPLETE PORTFOLIO ANALYSIS TABLES")
        logging.info("="*80)
        
        for portfolio_name, df in self.portfolio_analysis_tables.items():
            logging.info(f"\n{portfolio_name.upper()} PORTFOLIO:")
            logging.info("-" * 50)
            
            if df.empty:
                logging.info("No data available")
                continue
            
            # Format DataFrame for display
            display_df = df.copy()
            
            # Format percentage columns
            for col in display_df.columns:
                if col[1] in ['Weight', 'Period_Return', 'Cumulative_Return']:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
                elif col[1] == 'Portfolio_Value':
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}")
            
            # Display the formatted table
            logging.info("\n" + display_df.to_string())
            
            # Save to CSV if configured
            if self.config.save_results:
                import os
                os.makedirs(self.config.results_directory, exist_ok=True)
                csv_path = os.path.join(self.config.results_directory, f'portfolio_analysis_{portfolio_name}.csv')
                df.to_csv(csv_path)
                logging.info(f"Table saved to: {csv_path}")
        
        logging.info("\n" + "="*80)
        
        # Save portfolio analysis tables to pickle files
        self._save_portfolio_tables_pickle()
    
    def _save_portfolio_tables_pickle(self) -> None:
        """Save portfolio analysis tables to pickle files with ticker hash."""
        if not hasattr(self, 'portfolio_analysis_tables') or not self.portfolio_analysis_tables:
            return
        
        import hashlib
        import pickle
        
        # Generate ticker hash (same method as FinData)
        sorted_tickers = sorted(self.asset_names)
        ticker_string = ','.join(sorted_tickers)
        ticker_hash = hashlib.md5(ticker_string.encode()).hexdigest()[:8]
        
        # Save each portfolio table to pickle
        if self.config.save_results:
            import os
            os.makedirs(self.config.results_directory, exist_ok=True)
            
            for portfolio_name, df in self.portfolio_analysis_tables.items():
                pickle_filename = f'portfolio_tables_{portfolio_name}_{self.config.start_date}_{self.config.end_date}_{ticker_hash}.pkl'
                pickle_path = os.path.join(self.config.results_directory, pickle_filename)
                
                try:
                    with open(pickle_path, 'wb') as f:
                        pickle.dump(df, f)
                    logging.info(f"Portfolio table saved to pickle: {pickle_path}")
                except Exception as e:
                    logging.warning(f"Failed to save portfolio table pickle: {e}")
            
            # Also save the complete dictionary
            all_tables_filename = f'all_portfolio_tables_{self.config.start_date}_{self.config.end_date}_{ticker_hash}.pkl'
            all_tables_path = os.path.join(self.config.results_directory, all_tables_filename)
            
            try:
                with open(all_tables_path, 'wb') as f:
                    pickle.dump(self.portfolio_analysis_tables, f)
                logging.info(f"All portfolio tables saved to pickle: {all_tables_path}")
            except Exception as e:
                logging.warning(f"Failed to save all portfolio tables pickle: {e}")
    
    def get_portfolio_analysis_tables(self) -> Dict[str, 'pd.DataFrame']:
        """
        Return the portfolio analysis tables dictionary for external use.
        
        Returns:
        --------
        Dict with portfolio names as keys and DataFrames as values
        """
        if hasattr(self, 'portfolio_analysis_tables'):
            return self.portfolio_analysis_tables
        else:
            return {}
    
    def _print_portfolio_comparison_summary(self, current_weights: Dict[str, np.ndarray], 
                                          period_returns: np.ndarray, period_num: int) -> None:
        """Print a summary comparison of all portfolios for this period."""
        import pandas as pd
        
        logging.info("\nPortfolio Comparison Summary:")
        
        comparison_data = {
            'Portfolio': [],
            'Top 3 Holdings': [],
            'Period Return': [],
            'Diversification': []
        }
        
        for portfolio_name, weights in current_weights.items():
            portfolio_return = np.sum(weights * period_returns)
            
            # Get top 3 holdings
            top_indices = np.argsort(weights)[-3:][::-1]  # Top 3 in descending order
            top_holdings = []
            for idx in top_indices:
                if weights[idx] > 0.001:
                    top_holdings.append(f"{self.asset_names[idx]}({weights[idx]:.1%})")
            
            # Calculate diversification (Herfindahl index)
            herfindahl = np.sum(weights ** 2)
            diversification = 1 / herfindahl if herfindahl > 0 else 0
            
            comparison_data['Portfolio'].append(portfolio_name)
            comparison_data['Top 3 Holdings'].append(", ".join(top_holdings[:3]))
            comparison_data['Period Return'].append(f"{portfolio_return:.2%}")
            comparison_data['Diversification'].append(f"{diversification:.1f}")
        
        df = pd.DataFrame(comparison_data)
        logging.info("\n" + df.to_string(index=False))