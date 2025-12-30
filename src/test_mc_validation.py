#!/usr/bin/env python3
"""
Monte Carlo Lifecycle Validation Test

Validates the complete Monte Carlo lifecycle simulation system with:
- Time-varying parameters (regime shift at retirement)
- Continuous accumulation → decumulation paths
- Asset-level return path generation with correlation preservation
- Period-level simulation with contributions and withdrawals

This test uses controlled simulated data with known parameters to validate
the MC path generation without Yahoo Finance dependency.

Usage:
    uv run python test_mc_validation.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
from system_config import SystemConfig
from fin_data import FinData
import mc_path_generator, visualize_mc_lifecycle
importlib.reload(mc_path_generator)
importlib.reload(visualize_mc_lifecycle)
from mc_path_generator import MCPathGenerator
from visualize_mc_lifecycle import run_accumulation_mc, run_decumulation_mc
from visualize_covariance_matrices import plot_covariance_evolution
from portfolio import Portfolio
import simulated_data_params as sim_params

import ipdb
np.set_printoptions(linewidth=100)


# ============================================================================
# Helper Functions
# ============================================================================

def create_lifecycle_visualization(acc_paths, dec_paths, acc_values, dec_values,
                                   retirement_period, weights, tickers, mean_ts, dates, retirement_date, config):
    """Create 2x2 visualization for lifecycle simulation."""

    print("\nCreating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # Plot 1: Sample accumulation paths
    ax = axes[0, 0]
    num_paths = min(20, acc_values.shape[0])  # Don't exceed available simulations
    for i in range(num_paths):
        ax.plot(acc_values[i, :], alpha=0.4, linewidth=0.8, color='green')

    ax.set_title(f'Accumulation Paths ({num_paths} simulations)', fontsize=13, fontweight='bold')
    ax.set_xlabel(f'Period ({config.simulation_frequency})', fontsize=11)
    ax.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))

    # Plot 2: Time-varying mean returns (full lifecycle)
    ax = axes[0, 1]

    # Reindex mean_ts to full date range for visualization
    # This shows the complete time-varying parameters over accumulation and decumulation
    mean_ts_full = mean_ts.reindex(dates, method='ffill')

    # Plot each ticker's time-varying mean return
    for ticker in tickers:
        ax.plot(mean_ts_full.index, mean_ts_full[ticker], label=ticker, linewidth=2, alpha=0.8)

    # Mark the regime shift points (sparse input dates)
    for shift_date in mean_ts.index:
        ax.axvline(shift_date, color='gray', linestyle=':', linewidth=1, alpha=0.4)

    # Highlight retirement date
    ax.axvline(retirement_date, color='red', linestyle='--',
               linewidth=2, label='Retirement', alpha=0.6)

    # Add shaded regions for accumulation vs decumulation
    ax.axvspan(dates[0], retirement_date, alpha=0.05, color='green', label='Accumulation')
    ax.axvspan(retirement_date, dates[-1], alpha=0.05, color='red', label='Decumulation')

    ax.set_title('Time-Varying Mean Returns (Annual) - Full Lifecycle', fontsize=13, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Expected Return', fontsize=11)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

    # Plot 3: Accumulation distribution
    ax = axes[1, 0]
    final_acc = acc_values[:, -1]
    ax.hist(final_acc, bins=50, alpha=0.7, color='green', edgecolor='black')
    for p, color in [(5, 'red'), (50, 'blue'), (95, 'orange')]:
        val = np.percentile(final_acc, p)
        ax.axvline(val, color=color, linestyle='--', linewidth=2,
                   label=f'{p}th: ${val/1e6:.2f}M')
    ax.set_title('Final Accumulation Value Distribution', fontsize=13, fontweight='bold')
    ax.set_xlabel('Final Value ($)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

    # Plot 4: Complete lifecycle spaghetti plot (accumulation → decumulation)
    ax = axes[1, 1]
    num_spaghetti = min(50, acc_values.shape[0])

    # Combine accumulation and decumulation into one continuous path
    acc_periods = acc_values.shape[1]  # Number of accumulation values (includes t=0)
    dec_periods = dec_values.shape[1]  # Number of decumulation values (includes t=0)

    # Create date arrays for x-axis
    # acc_values has shape (sims, 417) → need 417 dates
    # dec_values has shape (sims, 1041) → need 1041 dates
    acc_dates = dates[:acc_periods]  # First acc_periods dates
    dec_dates = dates[acc_periods-1:acc_periods-1 + dec_periods]  # Overlapping at retirement

    for i in range(num_spaghetti):
        # Plot accumulation phase (green)
        ax.plot(acc_dates, acc_values[i, :], color='green', alpha=0.15, linewidth=0.8)

        # Plot decumulation phase (red)
        ax.plot(dec_dates, dec_values[i, :], color='red', alpha=0.15, linewidth=0.8)

    # Add vertical line at retirement
    ax.axvline(retirement_date, color='blue', linestyle='--', linewidth=2,
               label='Retirement', alpha=0.7)

    ax.set_title(f'Complete Lifecycle Paths ({num_spaghetti} simulations)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale to see full range
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))

    # Format x-axis dates
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))  # Show every 2 years
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add phase labels using date positions
    acc_mid_date = acc_dates[len(acc_dates)//2]
    dec_mid_date = dec_dates[len(dec_dates)//2]
    ax.text(acc_mid_date, ax.get_ylim()[1]*0.8, 'Accumulation',
            ha='center', fontsize=10, color='green', weight='bold', alpha=0.7)
    ax.text(dec_mid_date, ax.get_ylim()[1]*0.8, 'Decumulation',
            ha='center', fontsize=10, color='red', weight='bold', alpha=0.7)

    plt.tight_layout()

    import os
    os.makedirs('../plots/test', exist_ok=True)
    filepath = '../plots/test/mc_validation_lifecycle.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")


# ============================================================================
# Main Validation Script
# ============================================================================

print("=" * 80)
print("MC PATH GENERATION VALIDATION - LIFECYCLE WITH REGIME SHIFT")
print("=" * 80)

# Step 1: Load config and data
print("\n[1/6] Loading configuration and data...")
config = SystemConfig.from_json('../configs/test_simple_buyhold.json')
# Resolve ticker file path relative to project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ticker_file = config.ticker_file if os.path.isabs(config.ticker_file) else os.path.join(PROJECT_ROOT, config.ticker_file)
tickers_df = pd.read_csv(ticker_file)
tickers = tickers_df['Symbol'].tolist()
weights_dict = dict(zip(tickers_df['Symbol'], tickers_df['Weight']))
weights_df = pd.Series(weights_dict)
weights = np.array([weights_dict[t] for t in tickers])

# Get settings from config
periods_per_year = config.frequency_to_periods_per_year(config.simulation_frequency)
simulation_frequency = config.get_simulation_pandas_frequency()

print(f"  ✓ Simulation frequency: {config.simulation_frequency} -> {simulation_frequency} -> {periods_per_year} periods/year")
print(f"  ✓ Number of simulations: {config.num_mc_simulations}")
print(f"  ✓ Use simulated data: {config.use_simulated_data}")
print(f"  ✓ Reindex method: {config.mc_reindex_method}")

# Load or generate data based on config
if config.use_simulated_data:
    print("  Using simulated data for parameter estimation...")
    returns_data = sim_params.create_simulated_returns_data(
        tickers=tickers,
        num_days=sim_params.NUM_DAYS,
        seed=sim_params.RANDOM_SEED
    )
    print(f"  ✓ Generated {len(returns_data)} days of simulated returns")
else:
    # Use historical data
    fin_data = FinData(start_date=config.start_date, end_date=config.end_date)
    fin_data.fetch_ticker_data(tickers)
    returns_data = fin_data.get_returns_data(tickers)

print(f"  ✓ Loaded {len(tickers)} tickers: {tickers}")
print(f"  ✓ Returns data: {len(returns_data)} days")

# Step 2: Create time-varying parameters (regime shift at retirement)
print("\n[2/6] Creating time-varying parameters...")

# Get parameters from simulated_data_params module
mean_returns_acc, cov_matrix_acc = sim_params.get_accumulation_params(regularize=True)
mean_returns_dec, cov_matrix_dec = sim_params.get_decumulation_params(regularize=True)

# Create DataFrames for validation
cov_matrix_acc_df = pd.DataFrame(cov_matrix_acc, columns=tickers, index=tickers)
cov_matrix_dec_df = pd.DataFrame(cov_matrix_dec, columns=tickers, index=tickers)

# Align dates to frequency
mc_start_date = SystemConfig.align_date_to_frequency(
    config.get_mc_start_date(),
    config.get_contribution_pandas_frequency()
)
retirement_date_config = SystemConfig.align_date_to_frequency(
    config.retirement_date,
    config.get_contribution_pandas_frequency()
)


## Create time-varying mean and covariance data from files
# Read mean returns from file (2 rows: accumulation, decumulation)
# Resolve paths relative to project root
mean_file = config.simulated_mean_returns_file if os.path.isabs(config.simulated_mean_returns_file) else os.path.join(PROJECT_ROOT, config.simulated_mean_returns_file)
cov_file = config.simulated_cov_matrices_file if os.path.isabs(config.simulated_cov_matrices_file) else os.path.join(PROJECT_ROOT, config.simulated_cov_matrices_file)
mean_returns_df = pd.read_csv(mean_file, index_col=0)

# Load covariance matrices from file as 2D array (2 regimes × n_assets × n_assets flattened)
cov_matrices_2d = np.loadtxt(cov_file)

# Get asset count from mean returns
n_assets = mean_returns_df.shape[1]

# Reshape 2D array into 3D array (2 regimes, n_assets, n_assets)
n_regimes = 2
cov_matrices = cov_matrices_2d.reshape(n_regimes, n_assets, n_assets)

# Create time-varying DataFrames using actual dates
# We'll create one entry for start of accumulation and one for start of decumulation
regime_dates = [
    mc_start_date,           # Accumulation regime starts at MC start
    retirement_date_config   # Decumulation regime starts at retirement
]

# Create time-varying mean returns DataFrame with dates
mean_ts = pd.DataFrame(
    mean_returns_df.values,
    index=pd.DatetimeIndex(regime_dates),
    columns=mean_returns_df.columns
)

# Create time-varying covariance DataFrame with dates
cov_data = [{'cov_matrix': cov} for cov in cov_matrices]
cov_ts = pd.DataFrame(cov_data, index=pd.DatetimeIndex(regime_dates))





# # Step 3: Setup lifecycle simulation
# print("\n[3/6] Setting up lifecycle simulation...")
acc_years = int(config.get_accumulation_years())
dec_years = int(config.get_decumulation_years())

acc_periods = acc_years * periods_per_year
dec_periods = dec_years * periods_per_year
total_periods = acc_periods + dec_periods

# print(f"  Accumulation: {acc_years} years = {acc_periods} periods")
# print(f"  Decumulation: {dec_years} years = {dec_periods} periods")
# print(f"  Total: {total_periods} periods ({total_periods / periods_per_year:.1f} years)")

# Create dates for visualization (includes t=0, so total_periods + 1)
# IMPORTANT: Use contribution frequency for dates, not simulation frequency
# The simulation functions output values at contribution/withdrawal checkpoints
visualization_frequency = config.get_contribution_pandas_frequency()
visualization_periods_per_year = config.frequency_to_periods_per_year(config.contribution_frequency)
acc_vis_periods = acc_years * visualization_periods_per_year
dec_vis_periods = dec_years * visualization_periods_per_year
total_vis_periods = acc_vis_periods + dec_vis_periods

dates = pd.date_range(start=mc_start_date, periods=total_vis_periods + 1,
                      freq=visualization_frequency)
retirement_date = dates[acc_vis_periods]

print(f"  MC start date: {mc_start_date}")
# print(f"  First simulation date: {dates[0].date()}")
print(f"  Retirement date: {retirement_date.date()} (period {acc_periods})")


# Step 5: Generate paths
print("\n[5/6] Generating lifecycle paths...")

# Initialize generator
path_generator = MCPathGenerator(tickers, seed=sim_params.RANDOM_SEED)

# Generate paths with TIME-VARYING parameters
acc_paths, dec_paths = path_generator.generate_paths(
    num_simulations=config.num_mc_simulations,
    accumulation_years=acc_years,
    decumulation_years=dec_years,
    periods_per_year=periods_per_year,
    start_date=mc_start_date,
    frequency=simulation_frequency,
    mean_returns=mean_ts,
    cov_matrices=cov_ts,
    reindex_method=config.mc_reindex_method
)

print(f"  ✓ Accumulation paths: {acc_paths.shape}")
print(f"  ✓ Decumulation paths: {dec_paths.shape}")

# Verify regime shift impact (period-level returns)
acc_spy_avg = acc_paths[:, :, tickers.index('SPY')].mean()
dec_spy_avg = dec_paths[:, :, tickers.index('SPY')].mean()
print(f"  ✓ SPY average return (period-level):")
print(f"    Accumulation: {acc_spy_avg:.6f}")
print(f"    Decumulation: {dec_spy_avg:.6f}")

# Step 6: Run simulations
print("\n[6/6] Running accumulation and decumulation simulations...")

contribution_config = config.get_contribution_config()
withdrawal_config = config.get_withdrawal_config()

# Handle case where contribution_config is None (no contributions)
if contribution_config is None:
    contribution_config = {
        'amount': 0.0,
        'frequency': 'none',
        'employer_match_rate': 0.0,
        'employer_match_cap': None
    }

# Run accumulation
print(f"  Accumulation:")
print(f"    Initial: ${config.initial_portfolio_value:,.0f}")
if contribution_config['amount'] > 0:
    print(f"    Contribution: ${contribution_config['amount']:,.0f} {contribution_config['frequency']}")
else:
    print(f"    Contribution: None (buy-and-hold only)")

accumulation_values = run_accumulation_mc(
    initial_value=config.initial_portfolio_value,
    weights=weights,
    asset_returns_paths=acc_paths,
    asset_returns_frequency=config.pandas_frequency_to_periods_per_year(simulation_frequency),
    years=acc_years,
    contributions_per_year=config.frequency_to_periods_per_year(config.contribution_frequency),
    contribution_amount=contribution_config['amount'],
    employer_match_rate=contribution_config['employer_match_rate'],
    employer_match_cap=contribution_config['employer_match_cap']
)

final_acc_values = accumulation_values[:, -1]
print(f"    ✓ Final values (percentiles):")
print(f"      5th:  ${np.percentile(final_acc_values, 5):,.0f}")
print(f"      50th: ${np.percentile(final_acc_values, 50):,.0f}")
print(f"      95th: ${np.percentile(final_acc_values, 95):,.0f}")

# Handle case where withdrawal_config is None (no withdrawals)
if withdrawal_config is None:
    withdrawal_config = {
        'annual_amount': 0.0,
        'inflation_rate': 0.0,
        'frequency': 'none',
        'withdrawals_per_year': 1
    }

# Run decumulation
print(f"  Decumulation:")
if withdrawal_config['annual_amount'] > 0:
    print(f"    Withdrawal: ${withdrawal_config['annual_amount']:,.0f}/year ({withdrawal_config['frequency']})")
    print(f"    Inflation: {withdrawal_config['inflation_rate']:.1%}")
else:
    print(f"    Withdrawal: None (portfolio growth only)")

decumulation_values, success = run_decumulation_mc(
    initial_values=final_acc_values,
    weights=weights,
    asset_returns_paths=dec_paths,
    asset_returns_frequency=config.pandas_frequency_to_periods_per_year(simulation_frequency),
    annual_withdrawal=withdrawal_config['annual_amount'],
    inflation_rate=withdrawal_config['inflation_rate'],
    years=dec_years,
    withdrawals_per_year=config.frequency_to_periods_per_year(config.withdrawal_frequency)
)

success_rate = success.mean()
print(f"    ✓ Success rate: {success_rate:.1%}")
print(f"    ✓ Final values (percentiles):")
print(f"      5th:  ${np.percentile(decumulation_values[:, -1], 5):,.0f}")
print(f"      50th: ${np.percentile(decumulation_values[:, -1], 50):,.0f}")
print(f"      95th: ${np.percentile(decumulation_values[:, -1], 95):,.0f}")

# Visualization 1: Lifecycle paths
create_lifecycle_visualization(
    acc_paths, dec_paths, accumulation_values, decumulation_values,
    acc_periods, weights, tickers, mean_ts, dates, retirement_date, config
)

# Visualization 2: Covariance matrix evolution
print("\nCreating covariance matrix visualization...")
plot_covariance_evolution(
    cov_matrix_acc=cov_matrix_acc_df,
    cov_matrix_dec=cov_matrix_dec_df,
    acc_label='Accumulation',
    dec_label='Decumulation',
    output_path='../plots/test/mc_validation_covariance.png'
)

print("\n" + "=" * 80)
print("ALL VALIDATIONS PASSED ✓")
print("=" * 80)
print("\nKey Results:")
print(f"  ✓ Regime shift at retirement (period {acc_periods})")
print(f"  ✓ Accumulation median: ${np.percentile(final_acc_values, 50):,.0f}")
print(f"  ✓ Decumulation success: {success_rate:.1%}")
print(f"  ✓ reindex() with {config.mc_reindex_method} works correctly")
print("\nVisualizations:")
print(f"  ✓ Lifecycle: ../plots/test/mc_validation_lifecycle.png")
print(f"  ✓ Covariance: ../plots/test/mc_validation_covariance.png")
print("=" * 80)

'''
grep -l "print" $(find . -type f) | xargs ls -lt

please read claude.md file and also read test_mc_validation.py - it contains Monte Carlo simulation paths. We also have PortfolioOrchestrator object in main.py that uses real financial data from Yahoo finance to rebalance specified portfolios. I want to integrate MC paths generated in test_mc_validation.py into main.py and to enable it to use either simulated MC data or real financial data for Portfolio performance tracking. 
please read relevant files and propose a step by step task list with test to make sure that we haven't broken anything

'''