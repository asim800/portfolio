#!/usr/bin/env python3
"""
Visualize Monte Carlo simulation for full retirement lifecycle:
- Accumulation phase: Portfolio grows with potential contributions
- Decumulation phase: Portfolio withdrawals with spending strategy
"""

import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta

from system_config import SystemConfig
from fin_data import FinData
from mc_path_generator import MCPathGenerator

import ipdb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def run_accumulation_mc(
    initial_value: float,
    weights: np.ndarray,
    asset_returns_paths: np.ndarray,
    years: int,
    contributions_per_year: int = 1,
    contribution_amount: float = 0.0,
    employer_match_rate: float = 0.0,
    employer_match_cap: float = None
) -> np.ndarray:
    """
    Run Monte Carlo for accumulation phase with periodic contributions.

    Uses pre-generated asset-level return paths from MCPathGenerator,
    enabling portfolio comparison on identical market scenarios.

    Parameters:
    -----------
    initial_value : float
        Starting portfolio value
    weights : np.ndarray
        Portfolio weights (must sum to 1, length = num_assets)
    asset_returns_paths : np.ndarray
        Pre-generated asset returns from MCPathGenerator
        Shape: (num_simulations, total_periods, num_assets)
    years : int
        Number of years to simulate
    contributions_per_year : int
        Number of contributions per year (weekly=52, biweekly=26, monthly=12, annual=1)
    contribution_amount : float
        Contribution amount per period (e.g., $1000 per paycheck)
    employer_match_rate : float
        Employer match as fraction of employee contribution (e.g., 0.5 = 50% match)
    employer_match_cap : float
        Maximum employer match per year (None = unlimited)

    Returns:
    --------
    np.ndarray: (num_simulations, years+1) array of portfolio values at year boundaries
    """
    num_simulations = asset_returns_paths.shape[0]
    total_periods = asset_returns_paths.shape[1]

    # Validate
    expected_periods = years * contributions_per_year
    if total_periods != expected_periods:
        raise ValueError(f"asset_returns_paths has {total_periods} periods, expected {expected_periods} "
                        f"(years={years} × contributions_per_year={contributions_per_year})")

    # Initialize results (save values at year boundaries for output)
    values = np.zeros((num_simulations, years + 1))
    values[:, 0] = initial_value

    # Run simulations
    for sim in range(num_simulations):
        portfolio_value = initial_value
        employer_match_ytd = 0.0  # Track employer match for annual cap

        for period in range(1, total_periods + 1):
            # Add employee contribution
            if contribution_amount > 0:
                portfolio_value += contribution_amount

                # Calculate employer match (respecting annual cap)
                period_match = contribution_amount * employer_match_rate

                # Check if we've hit the annual cap
                if employer_match_cap is not None:
                    remaining_match_capacity = employer_match_cap - employer_match_ytd
                    period_match = min(period_match, remaining_match_capacity)
                    employer_match_ytd += period_match

                portfolio_value += period_match

            # Get asset returns for this period: (num_assets,)
            asset_returns = asset_returns_paths[sim, period - 1, :]

            # Calculate portfolio return as weighted average
            portfolio_return = np.dot(weights, asset_returns)

            # Apply return
            portfolio_value *= (1 + portfolio_return)

            # Save value at year boundaries
            if period % contributions_per_year == 0:
                year_idx = period // contributions_per_year
                values[sim, year_idx] = portfolio_value
                employer_match_ytd = 0.0  # Reset annual match tracker

    return values

def run_decumulation_mc(
    initial_values: np.ndarray,
    weights: np.ndarray,
    asset_returns_paths: np.ndarray,
    annual_withdrawal: float,
    inflation_rate: float,
    years: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Monte Carlo for decumulation phase with withdrawals.

    Uses pre-generated asset-level return paths from MCPathGenerator,
    enabling portfolio comparison on identical market scenarios.

    Parameters:
    -----------
    initial_values : np.ndarray
        Starting portfolio values from accumulation phase (one per simulation)
    weights : np.ndarray
        Portfolio weights (must sum to 1, length = num_assets)
    asset_returns_paths : np.ndarray
        Pre-generated annual asset returns from MCPathGenerator
        Shape: (num_simulations, years, num_assets)
    annual_withdrawal : float
        Initial annual withdrawal amount (before inflation)
    inflation_rate : float
        Annual inflation rate (e.g., 0.03 for 3%)
    years : int
        Number of years to simulate

    Returns:
    --------
    tuple: (portfolio_values, success_flags)
        - portfolio_values: (num_simulations, years+1) array
        - success_flags: (num_simulations,) boolean array (True = survived full period)
    """
    num_simulations = len(initial_values)

    # Validate
    if asset_returns_paths.shape[0] != num_simulations:
        raise ValueError(f"asset_returns_paths has {asset_returns_paths.shape[0]} simulations, "
                        f"expected {num_simulations}")
    if asset_returns_paths.shape[1] != years:
        raise ValueError(f"asset_returns_paths has {asset_returns_paths.shape[1]} years, expected {years}")

    # Initialize results
    values = np.zeros((num_simulations, years + 1))
    values[:, 0] = initial_values
    success = np.ones(num_simulations, dtype=bool)

    # Run simulations
    for sim in range(num_simulations):
        for year in range(1, years + 1):
            # Get asset returns for this year: (num_assets,)
            asset_returns = asset_returns_paths[sim, year - 1, :]

            # Calculate portfolio return as weighted average
            portfolio_return = np.dot(weights, asset_returns)

            # Apply return
            portfolio_value = values[sim, year - 1] * (1 + portfolio_return)

            # Calculate inflation-adjusted withdrawal
            withdrawal = annual_withdrawal * ((1 + inflation_rate) ** (year - 1))

            # Subtract withdrawal
            portfolio_value -= withdrawal

            # Check for depletion
            if portfolio_value <= 0:
                portfolio_value = 0
                success[sim] = False

            values[sim, year] = portfolio_value

    return values, success

def plot_lifecycle_mc(
    accumulation_values: np.ndarray,
    decumulation_values: np.ndarray,
    success_rate: float,
    config: SystemConfig,
    output_path: str = None
):
    """
    Create fan chart showing full lifecycle with accumulation and decumulation.
    """
    acc_years = accumulation_values.shape[1] - 1
    dec_years = decumulation_values.shape[1] - 1
    total_years = acc_years + dec_years

    # Calculate percentiles for accumulation
    acc_percentiles = {
        '5th': np.percentile(accumulation_values, 5, axis=0),
        '25th': np.percentile(accumulation_values, 25, axis=0),
        '50th': np.percentile(accumulation_values, 50, axis=0),
        '75th': np.percentile(accumulation_values, 75, axis=0),
        '95th': np.percentile(accumulation_values, 95, axis=0)
    }

    # Calculate percentiles for decumulation
    dec_percentiles = {
        '5th': np.percentile(decumulation_values, 5, axis=0),
        '25th': np.percentile(decumulation_values, 25, axis=0),
        '50th': np.percentile(decumulation_values, 50, axis=0),
        '75th': np.percentile(decumulation_values, 75, axis=0),
        '95th': np.percentile(decumulation_values, 95, axis=0)
    }

    # Create time axes
    start_date = datetime.strptime(config.end_date, '%Y-%m-%d')
    retirement_date = datetime.strptime(config.retirement_date, '%Y-%m-%d')

    acc_dates = [start_date + relativedelta(years=i) for i in range(acc_years + 1)]
    dec_dates = [retirement_date + relativedelta(years=i) for i in range(dec_years + 1)]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot accumulation phase
    ax.fill_between(acc_dates, acc_percentiles['5th'], acc_percentiles['95th'],
                     alpha=0.2, color='green', label='5th-95th percentile (Accumulation)')
    ax.fill_between(acc_dates, acc_percentiles['25th'], acc_percentiles['75th'],
                     alpha=0.3, color='green')
    ax.plot(acc_dates, acc_percentiles['50th'], 'g-', linewidth=2, label='Median (Accumulation)')

    # Plot decumulation phase
    ax.fill_between(dec_dates, dec_percentiles['5th'], dec_percentiles['95th'],
                     alpha=0.2, color='blue', label='5th-95th percentile (Decumulation)')
    ax.fill_between(dec_dates, dec_percentiles['25th'], dec_percentiles['75th'],
                     alpha=0.3, color='blue')
    ax.plot(dec_dates, dec_percentiles['50th'], 'b-', linewidth=2, label='Median (Decumulation)')

    # Mark retirement date
    ax.axvline(retirement_date, color='red', linestyle='--', linewidth=2,
               label=f'Retirement ({retirement_date.strftime("%Y-%m-%d")})')

    # Formatting
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
    ax.set_title(f'Monte Carlo Retirement Lifecycle Simulation\n'
                 f'Success Rate: {success_rate:.1%} | {accumulation_values.shape[0]} Simulations',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    plt.show(block=False)
    return fig

def plot_spaghetti_log(
    accumulation_values: np.ndarray,
    decumulation_values: np.ndarray,
    success_rate: float,
    config: SystemConfig,
    num_paths: int = 100,
    output_path: str = None
):
    """
    Create spaghetti plot with individual MC paths on log10 y-axis.

    Shows raw simulation paths to visualize the full range of outcomes.
    """
    acc_years = accumulation_values.shape[1] - 1
    dec_years = decumulation_values.shape[1] - 1
    num_sims = accumulation_values.shape[0]

    # Randomly sample paths to plot (avoid overcrowding)
    if num_sims > num_paths:
        np.random.seed(42)
        sample_idx = np.random.choice(num_sims, num_paths, replace=False)
    else:
        sample_idx = np.arange(num_sims)

    # Create time axes
    start_date = datetime.strptime(config.end_date, '%Y-%m-%d')
    retirement_date = datetime.strptime(config.retirement_date, '%Y-%m-%d')

    acc_dates = [start_date + relativedelta(years=i) for i in range(acc_years + 1)]
    dec_dates = [retirement_date + relativedelta(years=i) for i in range(dec_years + 1)]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot individual accumulation paths
    for idx in sample_idx:
        ax.plot(acc_dates, accumulation_values[idx, :],
                color='green', alpha=0.1, linewidth=0.5)

    # Plot individual decumulation paths
    for idx in sample_idx:
        ax.plot(dec_dates, decumulation_values[idx, :],
                color='blue', alpha=0.1, linewidth=0.5)

    # Calculate and plot median paths for reference
    acc_median = np.median(accumulation_values, axis=0)
    dec_median = np.median(decumulation_values, axis=0)

    ax.plot(acc_dates, acc_median, 'g-', linewidth=3,
            label='Median (Accumulation)', zorder=10)
    ax.plot(dec_dates, dec_median, 'b-', linewidth=3,
            label='Median (Decumulation)', zorder=10)

    # Mark retirement date
    ax.axvline(retirement_date, color='red', linestyle='--', linewidth=2,
               label=f'Retirement ({retirement_date.strftime("%Y-%m-%d")})', zorder=5)

    # Set log scale
    ax.set_yscale('log')

    # Formatting
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Portfolio Value ($) - Log Scale', fontsize=12, fontweight='bold')
    ax.set_title(f'Monte Carlo Paths - Log10 Scale\n'
                 f'Showing {len(sample_idx)}/{num_sims} Simulation Paths | Success Rate: {success_rate:.1%}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    # Format y-axis with nice log scale labels
    from matplotlib.ticker import FuncFormatter
    def log_formatter(x, pos):
        if x >= 1e9:
            return f'${x/1e9:.1f}B'
        elif x >= 1e6:
            return f'${x/1e6:.1f}M'
        elif x >= 1e3:
            return f'${x/1e3:.0f}K'
        else:
            return f'${x:.0f}'

    ax.yaxis.set_major_formatter(FuncFormatter(log_formatter))

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    plt.show(block=False)
    return fig

def main():
    print("=" * 80)
    print("MONTE CARLO LIFECYCLE VISUALIZATION")
    print("=" * 80)

    # Load configuration
    print("\n[1/5] Loading configuration...")
    config = SystemConfig.from_json('../configs/test_simple_buyhold.json')

    print(f"  Backtest: {config.start_date} to {config.end_date}")
    print(f"  Retirement: {config.retirement_date}")
    print(f"  Accumulation: {config.get_accumulation_years():.1f} years")
    print(f"  Decumulation: {config.get_decumulation_years():.1f} years")

    # Load historical data and estimate parameters
    print("\n[2/5] Loading data and estimating parameters...")

    tickers_df = pd.read_csv(config.ticker_file)
    tickers = tickers_df['Symbol'].tolist()
    weights_dict = dict(zip(tickers_df['Symbol'], tickers_df['Weight']))
    weights = np.array([weights_dict[t] for t in tickers])

    fin_data = FinData(start_date=config.start_date, end_date=config.end_date)
    fin_data.fetch_ticker_data(tickers)
    returns_data = fin_data.get_returns_data(tickers)

    # Estimate parameters
    mean_returns = returns_data.mean().values * 252
    cov_matrix = returns_data.cov().values * 252

    portfolio_mean = np.dot(weights, mean_returns)
    portfolio_std = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

    print(f"  Estimated portfolio return: {portfolio_mean:.2%}")
    print(f"  Estimated portfolio volatility: {portfolio_std:.2%}")

    # Generate MC paths using MCPathGenerator
    print("\n[3/5] Generating Monte Carlo paths...")

    # Setup parameters
    initial_value = 100_000  # Starting amount
    acc_years = int(config.get_accumulation_years())
    dec_years = int(config.get_decumulation_years())
    num_sims = 1000

    # Get contribution configuration
    contribution_config = config.get_contribution_config()
    if contribution_config:
        contribution_amount = contribution_config['amount']
        contributions_per_year = contribution_config['contributions_per_year']
        employer_match_rate = contribution_config['employer_match_rate']
        employer_match_cap = contribution_config['employer_match_cap']
        print(f"  Contributions: ${contribution_amount:,} {contribution_config['frequency']}")
        print(f"  Annual contribution: ${contribution_config['annual_contribution']:,}")
        if employer_match_rate > 0:
            print(f"  Employer match: {employer_match_rate:.0%}" +
                  (f" (max ${employer_match_cap:,}/year)" if employer_match_cap else ""))
    else:
        contribution_amount = 0
        contributions_per_year = 1
        employer_match_rate = 0
        employer_match_cap = None
        print(f"  No contributions configured")

    # Create path generator
    path_generator = MCPathGenerator(
        tickers=tickers,
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        seed=42
    )

    # Generate CONTINUOUS lifecycle paths (accumulation → decumulation)
    print(f"  Generating {num_sims} continuous lifecycle paths...")
    print(f"    Accumulation: {acc_years} years × {contributions_per_year} periods/year")
    print(f"    Decumulation: {dec_years} years (annual)")

    acc_paths, dec_paths = path_generator.generate_lifecycle_paths(
        num_simulations=num_sims,
        accumulation_years=acc_years,
        accumulation_periods_per_year=contributions_per_year,
        decumulation_years=dec_years
    )

    # Run accumulation simulation
    print("\n[4/6] Running accumulation phase simulation...")
    accumulation_values = run_accumulation_mc(
        initial_value=initial_value,
        weights=weights,
        asset_returns_paths=acc_paths,
        years=acc_years,
        contributions_per_year=contributions_per_year,
        contribution_amount=contribution_amount,
        employer_match_rate=employer_match_rate,
        employer_match_cap=employer_match_cap
    )

    final_acc_values = accumulation_values[:, -1]
    print(f"  Simulations: {num_sims}")
    print(f"  Starting value: ${initial_value:,}")
    print(f"  Final values (percentiles):")
    print(f"    5th:  ${np.percentile(final_acc_values, 5):,.0f}")
    print(f"    50th: ${np.percentile(final_acc_values, 50):,.0f}")
    print(f"    95th: ${np.percentile(final_acc_values, 95):,.0f}")

    # Run decumulation simulation
    print("\n[5/6] Running decumulation phase simulation...")

    withdrawal_config = config.get_withdrawal_config()

    decumulation_values, success = run_decumulation_mc(
        initial_values=final_acc_values,
        weights=weights,
        asset_returns_paths=dec_paths,
        annual_withdrawal=withdrawal_config['annual_amount'],
        inflation_rate=withdrawal_config['inflation_rate'],
        years=dec_years
    )

    success_rate = success.mean()

    print(f"  Annual withdrawal: ${withdrawal_config['annual_amount']:,}")
    print(f"  Inflation rate: {withdrawal_config['inflation_rate']:.1%}")
    print(f"  Success rate: {success_rate:.1%}")
    print(f"  Final values (percentiles):")
    print(f"    5th:  ${np.percentile(decumulation_values[:, -1], 5):,.0f}")
    print(f"    50th: ${np.percentile(decumulation_values[:, -1], 50):,.0f}")
    print(f"    95th: ${np.percentile(decumulation_values[:, -1], 95):,.0f}")

    # Create visualizations
    print("\n[6/6] Creating visualizations...")

    # Fan chart
    fan_chart_path = '../plots/test/mc_lifecycle_fan_chart.png'
    plot_lifecycle_mc(
        accumulation_values=accumulation_values,
        decumulation_values=decumulation_values,
        success_rate=success_rate,
        config=config,
        output_path=fan_chart_path
    )

    # Spaghetti plot with log scale
    spaghetti_path = '../plots/test/mc_lifecycle_spaghetti_log.png'
    plot_spaghetti_log(
        accumulation_values=accumulation_values,
        decumulation_values=decumulation_values,
        success_rate=success_rate,
        config=config,
        num_paths=200,  # Show 200 paths
        output_path=spaghetti_path
    )

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"✓ Accumulation: {acc_years} years, median final value ${np.median(final_acc_values):,.0f}")
    print(f"✓ Decumulation: {dec_years} years, {success_rate:.1%} success rate")
    print(f"✓ Fan chart saved to: {fan_chart_path}")
    print(f"✓ Spaghetti plot (log scale) saved to: {spaghetti_path}")
    print("=" * 80)

    # Keep plot open
    input("\nPress Enter to close...")

if __name__ == '__main__':
    main()
