#!/usr/bin/env python3
"""
Example: Using the New Portfolio Architecture

This script demonstrates how to use the refactored portfolio system
with the new Portfolio class and simplified API.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the new architecture
from portfolio import Portfolio
from portfolio_tracker import PortfolioTracker
from portfolio_optimizer import PortfolioOptimizer
from fin_data import FinData
from period_manager import PeriodManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def example_simple_usage():
    """
    Example 1: Simple portfolio comparison with simulated data.

    Shows the cleanest way to use the new architecture.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Portfolio Comparison (Simulated Data)")
    print("="*70 + "\n")

    # Step 1: Define assets and baseline weights
    asset_names = ['SPY', 'AGG', 'BIL']
    baseline_weights = pd.Series([0.6, 0.3, 0.1], index=asset_names)

    print(f"Assets: {asset_names}")
    print(f"Baseline weights: {baseline_weights.to_dict()}\n")

    # Step 2: Create simulated returns data (1 year of daily data)
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    returns = pd.DataFrame({
        'SPY': np.random.randn(len(dates)) * 0.01 + 0.0003,  # ~7.5% annual, 10% vol
        'AGG': np.random.randn(len(dates)) * 0.005 + 0.0001,  # ~2.5% annual, 5% vol
        'BIL': np.random.randn(len(dates)) * 0.001 + 0.00015  # ~4% annual, 1% vol
    }, index=dates)

    print(f"Created {len(returns)} days of simulated returns\n")

    # Step 3: Create portfolios using factory methods
    print("Creating portfolios...")

    # Buy-and-hold portfolio (weights drift with market)
    buy_hold = Portfolio.create_buy_and_hold(
        asset_names,
        baseline_weights,
        name='buy_and_hold'
    )

    # Target weight portfolio (rebalances to baseline every 30 days)
    target_wt = Portfolio.create_target_weight(
        asset_names,
        baseline_weights,
        rebalance_days=30,
        name='target_weight'
    )

    # Equal weight portfolio (1/3 each, rebalanced monthly)
    equal_wt = Portfolio.create_equal_weight(
        asset_names,
        rebalance_days=30,
        name='equal_weight'
    )

    # SPY-only portfolio (100% SPY, market benchmark)
    spy_only = Portfolio.create_spy_only(
        asset_names,
        rebalance_days=30,
        name='spy_benchmark'
    )

    print(f"✓ Created 4 portfolios\n")

    # Step 4: Ingest data into all portfolios
    print("Ingesting returns data...")
    for portfolio in [buy_hold, target_wt, equal_wt, spy_only]:
        portfolio.ingest_simulated_data(returns)
    print(f"✓ Data loaded into all portfolios\n")

    # Step 5: Create PortfolioTracker and add portfolios
    print("Setting up tracker...")
    tracker = PortfolioTracker()
    tracker.add_portfolio('buy_and_hold', buy_hold)
    tracker.add_portfolio('target_weight', target_wt)
    tracker.add_portfolio('equal_weight', equal_wt)
    tracker.add_portfolio('spy_benchmark', spy_only)
    print(f"✓ Added {len(tracker.portfolios)} portfolios to tracker\n")

    # Step 6: Create period manager and run backtest
    print("Running backtest...")
    period_manager = PeriodManager(returns, frequency="ME")
    print(f"  Rebalancing periods: {period_manager.num_periods}")

    tracker.run_backtest(period_manager)
    print(f"✓ Backtest complete\n")

    # Step 7: Get and display results
    print("="*70)
    print("RESULTS")
    print("="*70 + "\n")

    summary = tracker.get_portfolio_summary_statistics()

    # Format for display
    display_df = summary.copy()
    display_df['Total_Return'] = display_df['Total_Return'].apply(lambda x: f"{x:.2%}")
    display_df['Annual_Return'] = display_df['Annual_Return'].apply(lambda x: f"{x:.2%}")
    display_df['Volatility'] = display_df['Volatility'].apply(lambda x: f"{x:.2%}")
    display_df['Sharpe_Ratio'] = display_df['Sharpe_Ratio'].apply(lambda x: f"{x:.3f}")
    display_df['Max_Drawdown'] = display_df['Max_Drawdown'].apply(lambda x: f"{x:.2%}")

    print(display_df.to_string())
    print("\n")


def example_with_optimization():
    """
    Example 2: Portfolio comparison including optimized portfolios.

    Shows how to use PortfolioOptimizer with the new architecture.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Portfolio Comparison with Optimization")
    print("="*70 + "\n")

    # Step 1: Set up data
    asset_names = ['SPY', 'AGG', 'BIL', 'GLD']
    baseline_weights = pd.Series([0.5, 0.3, 0.1, 0.1], index=asset_names)

    print(f"Assets: {asset_names}")
    print(f"Baseline weights: {baseline_weights.to_dict()}\n")

    # Create simulated returns (1 year)
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')

    # More realistic correlation structure
    n_days = len(dates)
    spy_returns = np.random.randn(n_days) * 0.01 + 0.0003
    agg_returns = -0.3 * spy_returns + np.random.randn(n_days) * 0.005 + 0.0001  # Negative correlation with stocks
    bil_returns = np.random.randn(n_days) * 0.001 + 0.00015
    gld_returns = 0.2 * spy_returns + np.random.randn(n_days) * 0.012 + 0.0002  # Slight positive correlation

    returns = pd.DataFrame({
        'SPY': spy_returns,
        'AGG': agg_returns,
        'BIL': bil_returns,
        'GLD': gld_returns
    }, index=dates)

    print(f"Created {len(returns)} days of simulated returns\n")

    # Step 2: Create optimizer
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    print("✓ Created PortfolioOptimizer\n")

    # Step 3: Create portfolios
    print("Creating portfolios...")

    # Baseline portfolios
    buy_hold = Portfolio.create_buy_and_hold(asset_names, baseline_weights)

    # Optimized portfolios
    mean_var = Portfolio.create_optimized(
        asset_names,
        baseline_weights,
        optimizer,
        method='mean_variance',
        rebalance_days=30,
        name='mean_variance'
    )

    robust = Portfolio.create_optimized(
        asset_names,
        baseline_weights,
        optimizer,
        method='robust_mean_variance',
        rebalance_days=30,
        name='robust_mv'
    )

    min_var = Portfolio.create_optimized(
        asset_names,
        baseline_weights,
        optimizer,
        method='min_variance',
        rebalance_days=30,
        name='min_variance'
    )

    portfolios = {
        'buy_and_hold': buy_hold,
        'mean_variance': mean_var,
        'robust_mv': robust,
        'min_variance': min_var
    }

    print(f"✓ Created {len(portfolios)} portfolios\n")

    # Step 4: Ingest data
    print("Ingesting data...")
    for portfolio in portfolios.values():
        portfolio.ingest_simulated_data(returns)
    print("✓ Data loaded\n")

    # Step 5: Run backtest
    print("Running backtest...")
    tracker = PortfolioTracker()
    for name, portfolio in portfolios.items():
        tracker.add_portfolio(name, portfolio)

    period_manager = PeriodManager(returns, frequency="ME")
    tracker.run_backtest(period_manager)
    print("✓ Backtest complete\n")

    # Step 6: Display results
    print("="*70)
    print("RESULTS: Optimized vs Baseline")
    print("="*70 + "\n")

    summary = tracker.get_portfolio_summary_statistics()

    # Sort by Sharpe Ratio
    summary_sorted = summary.sort_values('Sharpe_Ratio', ascending=False)

    # Format for display
    display_df = summary_sorted.copy()
    display_df['Total_Return'] = display_df['Total_Return'].apply(lambda x: f"{x:.2%}")
    display_df['Annual_Return'] = display_df['Annual_Return'].apply(lambda x: f"{x:.2%}")
    display_df['Volatility'] = display_df['Volatility'].apply(lambda x: f"{x:.2%}")
    display_df['Sharpe_Ratio'] = display_df['Sharpe_Ratio'].apply(lambda x: f"{x:.3f}")
    display_df['Max_Drawdown'] = display_df['Max_Drawdown'].apply(lambda x: f"{x:.2%}")

    print(display_df.to_string())
    print("\n")


def example_real_data():
    """
    Example 3: Using real market data from FinData.

    Shows the complete workflow with actual market data.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Real Market Data Workflow")
    print("="*70 + "\n")

    # Step 1: Initialize FinData
    print("Loading real market data...")
    fin_data = FinData(
        start_date='2024-01-01',
        end_date='2024-12-31',
        cache_dir='../data'
    )

    # Step 2: Load tickers and weights
    ticker_file = '../tickers.txt'
    try:
        tickers_df = fin_data.load_tickers(ticker_file)
        asset_names = tickers_df['Symbol'].tolist()

        print(f"✓ Loaded {len(asset_names)} tickers: {asset_names}\n")

        # Step 3: Get price data and returns
        print("Downloading price data (may take a moment)...")
        price_data = fin_data.get_price_data(asset_names)
        returns = fin_data.get_returns_data(asset_names)

        print(f"✓ Downloaded {len(returns)} days of returns")
        print(f"  Date range: {returns.index[0].date()} to {returns.index[-1].date()}\n")

        # Step 4: Get baseline weights
        baseline_weights = fin_data.get_baseline_weights(asset_names)
        baseline_weights_series = pd.Series(baseline_weights, index=asset_names)

        print(f"Baseline weights:")
        for ticker, weight in baseline_weights_series.items():
            print(f"  {ticker}: {weight:.2%}")
        print()

        # Step 5: Create portfolios
        print("Creating portfolios...")

        buy_hold = Portfolio.create_buy_and_hold(asset_names, baseline_weights_series)
        buy_hold.returns_data = returns.copy()

        target_wt = Portfolio.create_target_weight(
            asset_names,
            baseline_weights_series,
            rebalance_days=30
        )
        target_wt.returns_data = returns.copy()

        # Create optimizer and optimized portfolio
        optimizer = PortfolioOptimizer(risk_free_rate=0.02)
        mean_var = Portfolio.create_optimized(
            asset_names,
            baseline_weights_series,
            optimizer,
            method='mean_variance',
            rebalance_days=30
        )
        mean_var.returns_data = returns.copy()

        print(f"✓ Created 3 portfolios\n")

        # Step 6: Run backtest
        print("Running backtest on real data...")
        tracker = PortfolioTracker()
        tracker.add_portfolio('buy_and_hold', buy_hold)
        tracker.add_portfolio('target_weight', target_wt)
        tracker.add_portfolio('mean_variance', mean_var)

        period_manager = PeriodManager(returns, frequency="ME")
        print(f"  Rebalancing periods: {period_manager.num_periods}")

        tracker.run_backtest(period_manager)
        print("✓ Backtest complete\n")

        # Step 7: Display results
        print("="*70)
        print("RESULTS: Real Market Performance (2024)")
        print("="*70 + "\n")

        summary = tracker.get_portfolio_summary_statistics()

        # Format for display
        display_df = summary.copy()
        display_df['Total_Return'] = display_df['Total_Return'].apply(lambda x: f"{x:.2%}")
        display_df['Annual_Return'] = display_df['Annual_Return'].apply(lambda x: f"{x:.2%}")
        display_df['Volatility'] = display_df['Volatility'].apply(lambda x: f"{x:.2%}")
        display_df['Sharpe_Ratio'] = display_df['Sharpe_Ratio'].apply(lambda x: f"{x:.3f}")
        display_df['Max_Drawdown'] = display_df['Max_Drawdown'].apply(lambda x: f"{x:.2%}")

        print(display_df.to_string())
        print("\n")

        # Step 8: Export results (optional)
        print("Exporting results...")
        import os
        output_dir = '../results/example_run'
        os.makedirs(output_dir, exist_ok=True)

        tracker.export_to_csv(output_dir)
        print(f"✓ Results exported to {output_dir}\n")

    except FileNotFoundError:
        print(f"✗ Ticker file not found: {ticker_file}")
        print("  Please create a ticker file or use simulated data examples\n")
    except Exception as e:
        print(f"✗ Error with real data: {e}")
        print("  Falling back to simulated data examples\n")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("NEW PORTFOLIO ARCHITECTURE - USAGE EXAMPLES")
    print("="*70)

    # Example 1: Simple comparison with simulated data
    example_simple_usage()

    # Example 2: With optimization
    example_with_optimization()

    # Example 3: Real market data (if available)
    try:
        example_real_data()
    except Exception as e:
        print(f"\nSkipping real data example: {e}")

    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70)
    print("\nKey takeaways:")
    print("1. Use Portfolio.create_*() factory methods for clarity")
    print("2. PortfolioTracker manages multiple portfolios")
    print("3. PeriodManager handles rebalancing schedule")
    print("4. All data is pandas (labeled and debuggable)")
    print("5. Optimizer uses numpy (fast and standard)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
