#!/usr/bin/env python3
"""
Simple test: Buy-and-hold backtest followed by MC simulation for retirement.

Workflow:
1. Load system config (2024-01-01 to 2025-09-19, weekly tracking)
2. Load portfolio from tickers.txt (BIL, MSFT, NVDA, SPY with 25% each)
3. Run backtest (buy-and-hold, weekly tracking)
4. Estimate mean/variance from backtest results
5. Run Monte Carlo simulation for:
   - Accumulation phase: 2025-09-19 to 2035-01-01 (10 years)
   - Decumulation phase: 2035-01-01 to 2065-01-01 (30 years with withdrawals)
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from system_config import SystemConfig
from portfolio_config import PortfolioConfig
from fin_data import FinData
from portfolio import Portfolio
from period_manager import PeriodManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

def main():
    print("=" * 80)
    print("SIMPLE BUY-AND-HOLD TEST WITH MC SIMULATION")
    print("=" * 80)

    # Step 1: Load system configuration
    print("\n[1/5] Loading system configuration...")
    config = SystemConfig.from_json('../configs/test_simple_buyhold.json')

    print(f"  Backtest period: {config.start_date} to {config.end_date}")
    print(f"  Retirement date: {config.retirement_date}")
    print(f"  Simulation horizon: {config.simulation_horizon_years} years")
    print(f"  Accumulation years: {config.get_accumulation_years():.1f}")
    print(f"  Decumulation years: {config.get_decumulation_years():.1f}")

    # Step 2: Load portfolio data from tickers.txt
    print("\n[2/5] Loading portfolio data...")

    # Read tickers file
    tickers_df = pd.read_csv(config.ticker_file)
    tickers = tickers_df['Symbol'].tolist()
    weights = dict(zip(tickers_df['Symbol'], tickers_df['Weight']))

    print(f"  Assets: {tickers}")
    print(f"  Weights: {weights}")

    # Download data
    fin_data = FinData(start_date=config.start_date, end_date=config.end_date)
    fin_data.fetch_ticker_data(tickers)

    returns_data = fin_data.get_returns_data(tickers)
    print(f"  Downloaded {len(returns_data)} days of data")

    # Step 3: Run backtest (weekly tracking, buy-and-hold)
    print("\n[3/5] Running backtest (buy-and-hold, weekly tracking)...")

    # Create portfolio
    portfolio = Portfolio.create_buy_and_hold(
        asset_names=tickers,
        initial_weights=pd.Series(weights),
        name='buy_and_hold_test'
    )

    # Ingest data
    portfolio.ingest_real_data(fin_data, tickers, config.start_date, config.end_date)

    # Create period manager (weekly periods)
    period_manager = PeriodManager(returns_data, frequency='W')
    print(f"  Created {period_manager.num_periods} weekly periods")

    # Run backtest
    portfolio.run_backtest(period_manager)

    # Get results
    summary = portfolio.get_summary_statistics()
    print(f"\n  Backtest Results:")
    print(f"    Total Return: {summary.loc[portfolio.name, 'Total_Return']:.2%}")
    print(f"    Annual Return: {summary.loc[portfolio.name, 'Annual_Return']:.2%}")
    print(f"    Volatility: {summary.loc[portfolio.name, 'Volatility']:.2%}")
    print(f"    Sharpe Ratio: {summary.loc[portfolio.name, 'Sharpe_Ratio']:.2f}")
    print(f"    Max Drawdown: {summary.loc[portfolio.name, 'Max_Drawdown']:.2%}")

    # Step 4: Estimate mean/variance from backtest
    print("\n[4/5] Estimating mean and covariance from backtest...")

    # Calculate annualized statistics
    mean_returns = returns_data.mean() * 252  # Annualized mean
    cov_matrix = returns_data.cov() * 252     # Annualized covariance

    print(f"  Annualized Mean Returns:")
    for ticker in tickers:
        print(f"    {ticker}: {mean_returns[ticker]:.2%}")

    print(f"\n  Annualized Volatility:")
    for ticker in tickers:
        vol = np.sqrt(cov_matrix.loc[ticker, ticker])
        print(f"    {ticker}: {vol:.2%}")

    # Portfolio statistics
    weights_array = np.array([weights[t] for t in tickers])
    portfolio_mean = np.dot(weights_array, mean_returns.values)
    portfolio_var = np.dot(weights_array, np.dot(cov_matrix.values, weights_array))
    portfolio_vol = np.sqrt(portfolio_var)

    print(f"\n  Portfolio Statistics (from estimates):")
    print(f"    Expected Return: {portfolio_mean:.2%}")
    print(f"    Volatility: {portfolio_vol:.2%}")
    print(f"    Sharpe Ratio: {(portfolio_mean - config.risk_free_rate) / portfolio_vol:.2f}")

    # Step 5: Run Monte Carlo simulation
    print("\n[5/5] Running Monte Carlo simulation...")
    print("  (This would use the estimated mean/variance for both phases)")
    print(f"  - Accumulation: {config.get_accumulation_years():.1f} years")
    print(f"  - Decumulation: {config.get_decumulation_years():.1f} years")

    withdrawal_config = config.get_withdrawal_config()
    if withdrawal_config:
        print(f"  - Withdrawal strategy: {withdrawal_config['strategy']}")
        print(f"  - Annual withdrawal: ${withdrawal_config['annual_amount']:,}")
        print(f"  - Inflation rate: {withdrawal_config['inflation_rate']:.1%}")

    print("\n  NOTE: Full MC simulation implementation pending.")
    print("  Would sample from estimated distribution for future returns.")

    # Summary
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"✓ Backtest completed: {period_manager.num_periods} weekly periods")
    print(f"✓ Mean/variance estimated from {len(returns_data)} days of data")
    print(f"✓ Ready for MC simulation with:")
    print(f"  - Portfolio mean: {portfolio_mean:.2%}")
    print(f"  - Portfolio vol: {portfolio_vol:.2%}")
    print(f"  - Accumulation: {config.get_accumulation_years():.1f} years")
    print(f"  - Decumulation: {config.get_decumulation_years():.1f} years")
    print("=" * 80)

if __name__ == '__main__':
    main()
