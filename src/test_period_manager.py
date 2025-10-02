#!/usr/bin/env python3
"""
Test script for the new PeriodManager and PortfolioTracker implementations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
import logging

from period_manager import PeriodManager
from portfolio_tracker import PortfolioTracker

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def create_sample_data():
    """Create sample returns data for testing."""
    # Create date range
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')

    # Filter to weekdays only (simulate trading days)
    dates = dates[dates.weekday < 5]

    # Create sample assets
    assets = ['SPY', 'BIL', 'MSFT', 'NVDA']

    # Generate random returns
    np.random.seed(42)  # For reproducible results
    returns_data = pd.DataFrame(
        np.random.normal(0.0005, 0.02, (len(dates), len(assets))),
        index=dates,
        columns=assets
    )

    return returns_data

def test_period_manager():
    """Test the PeriodManager functionality."""
    print("\n" + "="*50)
    print("TESTING PERIOD MANAGER")
    print("="*50)

    # Create sample data
    returns_data = create_sample_data()
    print(f"Created sample data: {len(returns_data)} days, {len(returns_data.columns)} assets")
    print(f"Date range: {returns_data.index[0].date()} to {returns_data.index[-1].date()}")

    # Initialize period manager
    period_manager = PeriodManager(returns_data, rebalancing_period_days=30)

    # Test basic functionality
    print(f"\nTotal periods: {period_manager.num_periods}")

    # Show periods summary
    periods_summary = period_manager.get_periods_summary()
    print("\nFirst 5 periods:")
    print(periods_summary.head().to_string())

    # Test getting period data
    print("\n" + "-"*40)
    print("Testing period data retrieval:")

    for period_num in range(min(3, period_manager.num_periods)):
        period_data = period_manager.get_period_data(period_num)
        period_info = period_manager.get_period_info(period_num)

        print(f"\nPeriod {period_num}:")
        print(f"  Date range: {period_info['period_start'].date()} to {period_info['period_end'].date()}")
        print(f"  Trading days: {period_info['trading_days']}")
        print(f"  Actual data shape: {period_data.shape}")

    # Test expanding window
    print("\n" + "-"*40)
    print("Testing expanding window data:")

    for period_num in range(min(4, period_manager.num_periods)):
        expanding_data = period_manager.get_expanding_window_data(period_num)
        print(f"Period {period_num}: expanding window has {len(expanding_data)} days")

    return period_manager

def test_portfolio_tracker():
    """Test the PortfolioTracker functionality."""
    print("\n" + "="*50)
    print("TESTING PORTFOLIO TRACKER")
    print("="*50)

    # Create sample setup
    asset_names = ['SPY', 'BIL', 'MSFT', 'NVDA']
    portfolio_names = ['buy_and_hold', 'vanilla', 'robust', 'mixed_vanilla']

    # Initialize tracker
    tracker = PortfolioTracker(asset_names, portfolio_names)

    # Simulate adding some performance data
    dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='30D')

    for i, date in enumerate(dates):
        # Generate sample weights and returns
        weights = {}
        returns = {}

        for portfolio_name in portfolio_names:
            # Random weights that sum to 1
            portfolio_weights = np.random.dirichlet(np.ones(len(asset_names)))
            weights[portfolio_name] = portfolio_weights

            # Random returns
            returns[portfolio_name] = np.random.normal(0.02, 0.05)

        # Add to tracker
        tracker.add_period_performance(
            period_start=date - pd.Timedelta(days=30),
            period_end=date,
            portfolio_weights=weights,
            portfolio_returns=returns
        )

    # Test retrieval methods
    print("\nTesting data retrieval:")

    # Get returns for one portfolio
    buy_hold_returns = tracker.get_portfolio_returns_history('buy_and_hold')
    print(f"Buy-and-hold returns shape: {buy_hold_returns.shape}")
    print(f"Returns: {buy_hold_returns.values}")

    # Get all cumulative returns
    all_cumulative = tracker.get_all_cumulative_returns()
    print(f"\nAll cumulative returns shape: {all_cumulative.shape}")
    print("Final cumulative returns:")
    print(all_cumulative.iloc[-1].to_string())

    # Get summary statistics
    summary_stats = tracker.get_portfolio_summary_statistics()
    print("\nSummary statistics:")
    print(summary_stats.to_string())

    # Display full summary
    tracker.display_summary()

    return tracker

def test_integration():
    """Test PeriodManager and PortfolioTracker working together."""
    print("\n" + "="*50)
    print("TESTING INTEGRATION")
    print("="*50)

    # Create sample data
    returns_data = create_sample_data()

    # Initialize components
    period_manager = PeriodManager(returns_data, rebalancing_period_days=30)
    asset_names = list(returns_data.columns)
    portfolio_names = ['baseline', 'optimized']
    tracker = PortfolioTracker(asset_names, portfolio_names)

    # Simulate a mini backtest
    baseline_weights = np.array([0.25, 0.25, 0.25, 0.25])  # Equal weights

    print(f"Running mini backtest with {period_manager.num_periods} periods")

    for period_num, period_data, period_info in period_manager.iter_periods():
        if period_num >= 5:  # Only test first 5 periods
            break

        print(f"\nProcessing period {period_num}: {period_info['period_start'].date()} to {period_info['period_end'].date()}")

        # Calculate period returns for each asset
        if not period_data.empty:
            asset_returns = ((1 + period_data).prod() - 1).values
            print(f"  Asset returns: {asset_returns}")

            # Portfolio performance
            baseline_return = np.sum(baseline_weights * asset_returns)
            optimized_weights = np.random.dirichlet(np.ones(len(asset_names)))  # Random optimization
            optimized_return = np.sum(optimized_weights * asset_returns)

            print(f"  Baseline return: {baseline_return:.4f}")
            print(f"  Optimized return: {optimized_return:.4f}")

            # Add to tracker
            tracker.add_period_performance(
                period_start=period_info['period_start'],
                period_end=period_info['period_end'],
                portfolio_weights={
                    'baseline': baseline_weights,
                    'optimized': optimized_weights
                },
                portfolio_returns={
                    'baseline': baseline_return,
                    'optimized': optimized_return
                }
            )

    # Show final results
    print("\nFinal Results:")
    tracker.display_summary()

if __name__ == "__main__":
    try:
        # Test individual components
        period_manager = test_period_manager()
        tracker = test_portfolio_tracker()

        # Test integration
        test_integration()

        print("\n" + "="*50)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*50)

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()