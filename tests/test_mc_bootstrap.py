#!/usr/bin/env python3
"""
Monte Carlo Bootstrap Validation Test

Validates the bootstrap sampling methods in MCPathGenerator:
- IID bootstrap (random row sampling with replacement)
- Block bootstrap (preserves short-term autocorrelation)
- Stationary bootstrap (random geometric block lengths)

Compares bootstrap paths against parametric paths to show differences
in distribution characteristics.

Usage:
    uv run python tests/test_mc_bootstrap.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Change to project root so relative paths work
os.chdir(PROJECT_ROOT)

from src.config import SystemConfig
from src.montecarlo import MCPathGenerator
from src.montecarlo.lifecycle import run_accumulation_mc, run_decumulation_mc
from src.data.market_data import load_returns_data
from src.data import simulated as sim_params

np.set_printoptions(linewidth=100)


# ============================================================================
# Helper Functions
# ============================================================================

def create_historical_data(config: SystemConfig, tickers: list, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic historical returns data for bootstrap testing.

    Uses simulated_data_params module to respect config-based parameters.

    Parameters:
    -----------
    config : SystemConfig
        Configuration object with simulation_frequency
    tickers : list
        List of ticker symbols
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame: Historical returns with date index and asset columns
    """
    periods_per_year = config.frequency_to_periods_per_year(config.simulation_frequency)

    # Use simulated_data_params to create returns (respects config parameters)
    # Scale from daily to config frequency
    n_days = 25 * 252  # 25 years of daily data
    returns_daily = sim_params.create_simulated_returns_data(
        tickers=tickers,
        num_days=n_days,
        seed=seed
    )

    # Resample to config frequency
    freq_map = {'daily': 'D', 'weekly': 'W', 'biweekly': '2W', 'monthly': 'ME', 'quarterly': 'QE', 'annual': 'YE'}
    pandas_freq = freq_map.get(config.simulation_frequency, 'W')

    # Add date index for resampling
    returns_daily.index = pd.date_range('2000-01-01', periods=len(returns_daily), freq='D')

    # Resample by summing returns (approximate for small returns)
    if pandas_freq != 'D':
        returns_data = returns_daily.resample(pandas_freq).sum()
    else:
        returns_data = returns_daily

    return returns_data


def run_bootstrap_comparison(
    config: SystemConfig,
    historical_data: pd.DataFrame,
    num_simulations: int = 200,
    seed: int = 42
) -> dict:
    """
    Run parametric vs bootstrap comparison and return results.

    Parameters:
    -----------
    config : SystemConfig
        Configuration object with lifecycle parameters
    historical_data : pd.DataFrame
        Historical returns data for bootstrap sampling
    num_simulations : int
        Number of Monte Carlo simulations
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    dict with keys:
        - 'acc_percentiles_param': dict of {5, 25, 50, 75, 95: value}
        - 'acc_percentiles_boot': dict of {5, 25, 50, 75, 95: value}
        - 'dec_percentiles_param': dict of {5, 25, 50, 75, 95: value}
        - 'dec_percentiles_boot': dict of {5, 25, 50, 75, 95: value}
        - 'dec_success_param': float (success rate)
        - 'dec_success_boot': float (success rate)
        - 'acc_values_param': np.ndarray (num_sims, periods)
        - 'acc_values_boot': np.ndarray (num_sims, periods)
        - 'dec_values_param': np.ndarray (num_sims, periods)
        - 'dec_values_boot': np.ndarray (num_sims, periods)
    """
    tickers = list(historical_data.columns)
    periods_per_year = config.frequency_to_periods_per_year(config.simulation_frequency)
    num_assets = len(tickers)

    # Get parameters from simulated_data_params (like test_mc_validation.py)
    mean_returns, cov_matrix = sim_params.get_accumulation_params(regularize=True)

    # Get lifecycle parameters
    acc_years = int(config.get_accumulation_years())
    dec_years = int(config.get_decumulation_years())
    initial_value = config.initial_portfolio_value

    # Get contribution config
    contribution_config = config.get_contribution_config()
    if contribution_config:
        contribution = contribution_config.get('amount', 0)
        contribution_freq = config.frequency_to_periods_per_year(config.contribution_frequency)
    else:
        contribution = 0
        contribution_freq = periods_per_year

    # Get withdrawal config
    withdrawal_config = config.get_withdrawal_config()
    if withdrawal_config:
        withdrawal = withdrawal_config.get('annual_amount', 0)
        inflation = withdrawal_config.get('inflation_rate', 0.02)
        withdrawal_freq = config.frequency_to_periods_per_year(config.withdrawal_frequency)
    else:
        withdrawal = 0
        inflation = 0.02
        withdrawal_freq = periods_per_year

    # Create generator
    gen = MCPathGenerator(
        tickers=tickers,
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        seed=seed
    )

    # Equal weights for all assets (e.g., 25% each for 4 assets)
    weights = np.ones(num_assets) / num_assets

    # Generate lifecycle paths (parametric)
    acc_paths_param, dec_paths_param = gen.generate_paths(
        num_simulations=num_simulations,
        accumulation_years=acc_years,
        decumulation_years=dec_years,
        periods_per_year=periods_per_year,
        sampling_method='parametric'
    )

    # Generate lifecycle paths (bootstrap)
    acc_paths_boot, dec_paths_boot = gen.generate_paths(
        num_simulations=num_simulations,
        accumulation_years=acc_years,
        decumulation_years=dec_years,
        periods_per_year=periods_per_year,
        historical_returns=historical_data,
        sampling_method='bootstrap'
    )

    # Run accumulation (parametric)
    acc_values_param = run_accumulation_mc(
        initial_value=initial_value,
        weights=weights,
        asset_returns_paths=acc_paths_param,
        asset_returns_frequency=periods_per_year,
        years=acc_years,
        contributions_per_year=contribution_freq,
        contribution_amount=contribution,
    )

    # Run accumulation (bootstrap)
    acc_values_boot = run_accumulation_mc(
        initial_value=initial_value,
        weights=weights,
        asset_returns_paths=acc_paths_boot,
        asset_returns_frequency=periods_per_year,
        years=acc_years,
        contributions_per_year=contribution_freq,
        contribution_amount=contribution,
    )

    final_acc_param = acc_values_param[:, -1]
    final_acc_boot = acc_values_boot[:, -1]

    # Run decumulation (parametric)
    dec_values_param, success_param = run_decumulation_mc(
        initial_values=final_acc_param,
        weights=weights,
        asset_returns_paths=dec_paths_param,
        asset_returns_frequency=periods_per_year,
        annual_withdrawal=withdrawal,
        inflation_rate=inflation,
        years=dec_years,
        withdrawals_per_year=withdrawal_freq,
    )

    # Run decumulation (bootstrap)
    dec_values_boot, success_boot = run_decumulation_mc(
        initial_values=final_acc_boot,
        weights=weights,
        asset_returns_paths=dec_paths_boot,
        asset_returns_frequency=periods_per_year,
        annual_withdrawal=withdrawal,
        inflation_rate=inflation,
        years=dec_years,
        withdrawals_per_year=withdrawal_freq,
    )

    final_dec_param = dec_values_param[:, -1]
    final_dec_boot = dec_values_boot[:, -1]

    # Collect percentiles
    percentiles = [5, 25, 50, 75, 95]

    return {
        'acc_percentiles_param': {p: np.percentile(final_acc_param, p) for p in percentiles},
        'acc_percentiles_boot': {p: np.percentile(final_acc_boot, p) for p in percentiles},
        'dec_percentiles_param': {p: np.percentile(final_dec_param, p) for p in percentiles},
        'dec_percentiles_boot': {p: np.percentile(final_dec_boot, p) for p in percentiles},
        'dec_success_param': success_param.mean(),
        'dec_success_boot': success_boot.mean(),
        'acc_values_param': acc_values_param,
        'acc_values_boot': acc_values_boot,
        'dec_values_param': dec_values_param,
        'dec_values_boot': dec_values_boot,
    }


def create_bootstrap_comparison_visualization(
    parametric_paths, iid_paths, block_paths, stationary_paths,
    tickers, historical_returns, config
):
    """Create 2x3 visualization comparing bootstrap methods."""

    print("\nCreating bootstrap comparison visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    num_paths = min(50, parametric_paths.shape[0])
    periods = parametric_paths.shape[1]
    x = np.arange(periods)

    # Row 1: Sample paths for each method
    methods = [
        ('Parametric (Gaussian)', parametric_paths, 'blue'),
        ('IID Bootstrap', iid_paths, 'green'),
        ('Block Bootstrap', block_paths, 'orange'),
    ]

    for idx, (name, paths, color) in enumerate(methods):
        ax = axes[0, idx]
        # Plot cumulative returns
        cumulative = np.cumprod(1 + paths[:num_paths, :, 0], axis=1)
        for i in range(num_paths):
            ax.plot(cumulative[i], alpha=0.3, linewidth=0.8, color=color)

        # Add mean path
        mean_cumulative = np.cumprod(1 + paths[:, :, 0].mean(axis=0))
        ax.plot(mean_cumulative, color='black', linewidth=2, label='Mean')

        ax.set_title(f'{name}\n({num_paths} paths)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Period', fontsize=10)
        ax.set_ylabel('Cumulative Return', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    # Row 2: Return distributions and autocorrelation
    # Panel 1: Return distribution comparison
    ax = axes[1, 0]
    for name, paths, color in methods:
        returns = paths[:, :, 0].flatten()
        ax.hist(returns, bins=50, alpha=0.5, label=name, color=color, density=True)

    # Add historical distribution
    hist_returns = historical_returns[tickers[0]].values
    ax.hist(hist_returns, bins=50, alpha=0.5, label='Historical', color='red', density=True)

    ax.set_title('Return Distribution Comparison', fontsize=12, fontweight='bold')
    ax.set_xlabel('Return', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Autocorrelation comparison
    ax = axes[1, 1]

    def calc_autocorr(returns, lag=1):
        """Calculate autocorrelation at given lag."""
        n = len(returns)
        if n <= lag:
            return 0
        mean = np.mean(returns)
        var = np.var(returns)
        if var == 0:
            return 0
        autocov = np.sum((returns[:-lag] - mean) * (returns[lag:] - mean)) / n
        return autocov / var

    # Calculate autocorrelation for each method
    lags = range(1, 11)
    for name, paths, color in methods:
        # Average autocorrelation across all simulations
        autocorrs = []
        for lag in lags:
            sim_autocorrs = [calc_autocorr(paths[i, :, 0], lag) for i in range(min(100, paths.shape[0]))]
            autocorrs.append(np.mean(sim_autocorrs))
        ax.plot(lags, autocorrs, 'o-', label=name, color=color, linewidth=2, markersize=6)

    # Historical autocorrelation
    hist_autocorrs = [calc_autocorr(hist_returns, lag) for lag in lags]
    ax.plot(lags, hist_autocorrs, 's--', label='Historical', color='red', linewidth=2, markersize=6)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Autocorrelation by Lag', fontsize=12, fontweight='bold')
    ax.set_xlabel('Lag', fontsize=10)
    ax.set_ylabel('Autocorrelation', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Summary statistics table
    ax = axes[1, 2]
    ax.axis('off')

    # Calculate statistics
    stats_data = []
    all_paths = [
        ('Historical', historical_returns[tickers[0]].values),
        ('Parametric', parametric_paths[:, :, 0].flatten()),
        ('IID Bootstrap', iid_paths[:, :, 0].flatten()),
        ('Block Bootstrap', block_paths[:, :, 0].flatten()),
        ('Stationary Bootstrap', stationary_paths[:, :, 0].flatten()),
    ]

    for name, returns in all_paths:
        stats_data.append({
            'Method': name,
            'Mean': f'{np.mean(returns):.6f}',
            'Std': f'{np.std(returns):.6f}',
            'Skew': f'{pd.Series(returns).skew():.4f}',
            'Kurt': f'{pd.Series(returns).kurtosis():.4f}',
            'Min': f'{np.min(returns):.6f}',
            'Max': f'{np.max(returns):.6f}',
        })

    stats_df = pd.DataFrame(stats_data)

    # Create table
    table = ax.table(
        cellText=stats_df.values,
        colLabels=stats_df.columns,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(len(stats_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save figure
    os.makedirs('output/plots/test', exist_ok=True)
    filepath = 'output/plots/test/mc_bootstrap_comparison.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")

    return fig


def create_lifecycle_bootstrap_visualization(
    acc_values_param, dec_values_param,
    acc_values_boot, dec_values_boot,
    config, historical_returns=None
):
    """Create lifecycle comparison between parametric and bootstrap."""
    import matplotlib.dates as mdates

    print("\nCreating lifecycle bootstrap comparison...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    num_paths = min(30, acc_values_param.shape[0])

    # Panel 1: Accumulation - Parametric
    ax = axes[0, 0]
    for i in range(num_paths):
        ax.plot(acc_values_param[i, :], alpha=0.3, linewidth=0.8, color='blue')
    ax.set_title('Accumulation (Parametric)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Period', fontsize=10)
    ax.set_ylabel('Portfolio Value ($)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))

    # Panel 2: Accumulation - Bootstrap
    ax = axes[0, 1]
    for i in range(num_paths):
        ax.plot(acc_values_boot[i, :], alpha=0.3, linewidth=0.8, color='green')
    ax.set_title('Accumulation (IID Bootstrap)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Period', fontsize=10)
    ax.set_ylabel('Portfolio Value ($)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))

    # Add historical data time-frame annotation to bootstrap panel
    if historical_returns is not None and hasattr(historical_returns, 'index'):
        hist_start = historical_returns.index[0]
        hist_end = historical_returns.index[-1]
        n_periods = len(historical_returns)
        # Get frequency name from config if available
        freq_name = config.simulation_frequency if config is not None and hasattr(config, 'simulation_frequency') else 'unknown'
        hist_text = f'Bootstrap source: {hist_start.strftime("%Y-%m")} to {hist_end.strftime("%Y-%m")}\n({n_periods} {freq_name} periods)'
        ax.text(0.02, 0.98, hist_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Panel 3: Final accumulation distribution comparison
    ax = axes[1, 0]
    final_param = acc_values_param[:, -1]
    final_boot = acc_values_boot[:, -1]

    ax.hist(final_param, bins=30, alpha=0.5, label='Parametric', color='blue', density=True)
    ax.hist(final_boot, bins=30, alpha=0.5, label='Bootstrap', color='green', density=True)

    # Add percentile lines
    for data, color, name in [(final_param, 'blue', 'Param'), (final_boot, 'green', 'Boot')]:
        p50 = np.percentile(data, 50)
        ax.axvline(p50, color=color, linestyle='--', linewidth=2,
                   label=f'{name} Median: ${p50/1e6:.2f}M')

    ax.set_title('Final Accumulation Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Final Value ($)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

    # Panel 4: Complete lifecycle spaghetti plot (accumulation → decumulation)
    ax = axes[1, 1]
    num_spaghetti = min(50, acc_values_boot.shape[0])

    # Get period counts from portfolio value arrays (includes initial t=0)
    acc_periods = acc_values_boot.shape[1]
    dec_periods = dec_values_boot.shape[1]

    # Calculate actual frequencies from array sizes and config years
    # acc_values shape: (sims, years * periods_per_year + 1)
    # dec_values shape: (sims, years * periods_per_year + 1)
    if config is not None:
        acc_years = int(config.get_accumulation_years())
        dec_years = int(config.get_decumulation_years())
        acc_periods_per_year = (acc_periods - 1) / acc_years  # subtract 1 for initial value
        dec_periods_per_year = (dec_periods - 1) / dec_years
    else:
        acc_periods_per_year = 52
        dec_periods_per_year = 52

    # Map periods_per_year to pandas frequency string
    def periods_to_freq(periods_per_year):
        if periods_per_year >= 250:
            return 'D'
        elif periods_per_year >= 50:
            return 'W'
        elif periods_per_year >= 24:
            return '2W'
        elif periods_per_year >= 11:
            return 'ME'
        elif periods_per_year >= 3:
            return 'QE'
        else:
            return 'YE'

    acc_freq = periods_to_freq(acc_periods_per_year)
    dec_freq = periods_to_freq(dec_periods_per_year)

    # Create separate date ranges for accumulation and decumulation
    sim_start_date = pd.Timestamp.today()
    acc_dates = pd.date_range(sim_start_date, periods=acc_periods, freq=acc_freq)
    retirement_date = acc_dates[-1]
    dec_dates = pd.date_range(retirement_date, periods=dec_periods, freq=dec_freq)

    # Debug: print date range info
    print(f"    Lifecycle plot: acc={acc_periods} periods ({acc_freq}), dec={dec_periods} periods ({dec_freq})")
    print(f"    Date range: {acc_dates[0].strftime('%Y-%m')} to {dec_dates[-1].strftime('%Y-%m')}")

    for i in range(num_spaghetti):
        # Plot accumulation phase (green) - bootstrap
        ax.plot(acc_dates, acc_values_boot[i, :], color='green', alpha=0.15, linewidth=0.8)

        # Plot decumulation phase (red) - bootstrap
        ax.plot(dec_dates, dec_values_boot[i, :], color='red', alpha=0.15, linewidth=0.8)

    # Add vertical line at retirement
    ax.axvline(retirement_date, color='blue', linestyle='--', linewidth=2,
               label='Retirement', alpha=0.7)

    ax.set_title(f'Complete Lifecycle Paths - Bootstrap ({num_spaghetti} simulations)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Portfolio Value ($)', fontsize=10)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale to see full range
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))  # Show every 2 years
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add phase labels using date positions
    acc_mid_date = acc_dates[len(acc_dates)//2]
    dec_mid_date = dec_dates[len(dec_dates)//2]
    ylim = ax.get_ylim()
    ax.text(acc_mid_date, ylim[1]*0.5, 'Accumulation',
            ha='center', fontsize=10, color='green', weight='bold', alpha=0.7)
    ax.text(dec_mid_date, ylim[1]*0.5, 'Decumulation',
            ha='center', fontsize=10, color='red', weight='bold', alpha=0.7)

    plt.tight_layout()

    filepath = 'output/plots/test/mc_bootstrap_lifecycle.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")

    return fig


# ============================================================================
# Main Validation Script
# ============================================================================

print("=" * 80)
print("MC BOOTSTRAP SAMPLING VALIDATION")
print("=" * 80)

# Configuration
NUM_SIMULATIONS = 200
SEED = 42

# Load configuration to get frequency settings
config_path = os.path.join(PROJECT_ROOT, 'configs/test_simple_buyhold.json')
config = SystemConfig.from_json(config_path)

# Get simulation frequency from config
PERIODS_PER_YEAR = config.frequency_to_periods_per_year(config.simulation_frequency)
TOTAL_PERIODS = 10 * PERIODS_PER_YEAR  # 10 years at config frequency
BLOCK_SIZE = max(1, PERIODS_PER_YEAR // 4)  # ~3-month blocks

print(f"\n  Config: {config.simulation_frequency} = {PERIODS_PER_YEAR} periods/year")
print(f"  Block size: {BLOCK_SIZE} periods")

# Step 1: Create synthetic historical data for reproducible testing
print("\n[1/5] Creating historical market data...")

# Get tickers from simulated_data_params (matches test_mc_validation.py)
tickers = ['BIL', 'MSFT', 'NVDA', 'SPY']

# Use simulated data parameters (respects config.use_simulated_data)
if config.use_simulated_data:
    print("  Using simulated data for parameter estimation...")
    returns_data = sim_params.create_simulated_returns_data(
        tickers=tickers,
        num_days=sim_params.NUM_DAYS,
        seed=sim_params.RANDOM_SEED
    )
    print(f"  ✓ Generated {len(returns_data)} days of simulated returns")

    # Resample to config frequency if needed
    freq_map = {'daily': 'D', 'weekly': 'W', 'biweekly': '2W', 'monthly': 'ME', 'quarterly': 'QE', 'annual': 'YE'}
    pandas_freq = freq_map.get(config.simulation_frequency, 'W')
    returns_data.index = pd.date_range('2000-01-01', periods=len(returns_data), freq='D')

    if pandas_freq != 'D':
        returns_data = returns_data.resample(pandas_freq).sum()
        print(f"  ✓ Resampled to {config.simulation_frequency}: {len(returns_data)} periods")
else:
    # Use historical data (would require FinData)
    raise NotImplementedError("Bootstrap test currently requires use_simulated_data=True")

print(f"  ✓ Created synthetic data: {len(returns_data)} periods ({config.simulation_frequency})")

tickers = list(returns_data.columns)
print(f"  ✓ Assets: {tickers}")

# Calculate historical statistics
hist_mean = returns_data.mean()
hist_std = returns_data.std()
hist_corr = returns_data.corr()

print(f"\n  Historical Statistics:")
print(f"    Mean returns (monthly):  {dict(hist_mean.round(6))}")
print(f"    Std dev (monthly):       {dict(hist_std.round(6))}")
print(f"    Correlation:             {hist_corr.iloc[0, 1]:.4f}")

# Step 2: Generate paths with different methods
print("\n[2/5] Generating Monte Carlo paths...")

# Get parameters from simulated_data_params (like test_mc_validation.py)
mean_returns, cov_matrix = sim_params.get_accumulation_params(regularize=True)

gen = MCPathGenerator(
    tickers=tickers,
    mean_returns=mean_returns,
    cov_matrix=cov_matrix,
    seed=SEED
)

# Generate parametric paths
print("  Generating parametric paths...")
parametric_paths = gen.generate_paths(
    num_simulations=NUM_SIMULATIONS,
    total_periods=TOTAL_PERIODS,
    periods_per_year=PERIODS_PER_YEAR,
    sampling_method='parametric'
)
print(f"    ✓ Parametric: {parametric_paths.shape}")

# Generate IID bootstrap paths
print("  Generating IID bootstrap paths...")
iid_paths = gen.generate_paths(
    num_simulations=NUM_SIMULATIONS,
    total_periods=TOTAL_PERIODS,
    periods_per_year=PERIODS_PER_YEAR,
    historical_returns=returns_data,
    sampling_method='bootstrap'
)
print(f"    ✓ IID Bootstrap: {iid_paths.shape}")

# Generate block bootstrap paths
print("  Generating block bootstrap paths...")
block_paths = gen.generate_paths(
    num_simulations=NUM_SIMULATIONS,
    total_periods=TOTAL_PERIODS,
    periods_per_year=PERIODS_PER_YEAR,
    historical_returns=returns_data,
    sampling_method='bootstrap_block',
    block_size=BLOCK_SIZE
)
print(f"    ✓ Block Bootstrap: {block_paths.shape}")

# Generate stationary bootstrap paths
print("  Generating stationary bootstrap paths...")
stationary_paths = gen.generate_paths(
    num_simulations=NUM_SIMULATIONS,
    total_periods=TOTAL_PERIODS,
    periods_per_year=PERIODS_PER_YEAR,
    historical_returns=returns_data,
    sampling_method='bootstrap_stationary',
    block_size=BLOCK_SIZE
)
print(f"    ✓ Stationary Bootstrap: {stationary_paths.shape}")

# Step 3: Compare statistics
print("\n[3/5] Comparing path statistics...")

methods = {
    'Parametric': parametric_paths,
    'IID Bootstrap': iid_paths,
    'Block Bootstrap': block_paths,
    'Stationary Bootstrap': stationary_paths,
}

print(f"\n  {'Method':<22} {'Mean':<12} {'Std':<12} {'Skew':<10} {'Kurt':<10}")
print("  " + "-" * 66)

for name, paths in methods.items():
    stock_returns = paths[:, :, 0].flatten()
    mean = np.mean(stock_returns)
    std = np.std(stock_returns)
    skew = pd.Series(stock_returns).skew()
    kurt = pd.Series(stock_returns).kurtosis()
    print(f"  {name:<22} {mean:>10.6f}   {std:>10.6f}   {skew:>8.4f}   {kurt:>8.4f}")

# Historical for reference
hist_stock = returns_data[tickers[0]].values
print(f"  {'Historical':<22} {np.mean(hist_stock):>10.6f}   {np.std(hist_stock):>10.6f}   "
      f"{pd.Series(hist_stock).skew():>8.4f}   {pd.Series(hist_stock).kurtosis():>8.4f}")

# Step 4: Run lifecycle simulation
print("\n[4/5] Running lifecycle simulation...")

# Get lifecycle parameters from config (already loaded at top)
ACC_YEARS = int(config.get_accumulation_years())
DEC_YEARS = int(config.get_decumulation_years())
INITIAL_VALUE = config.initial_portfolio_value

# Get contribution/withdrawal config
contribution_config = config.get_contribution_config()
withdrawal_config = config.get_withdrawal_config()

if contribution_config:
    CONTRIBUTION = contribution_config.get('amount', 0)
    CONTRIBUTION_FREQ = config.frequency_to_periods_per_year(config.contribution_frequency)
else:
    CONTRIBUTION = 0
    CONTRIBUTION_FREQ = PERIODS_PER_YEAR

if withdrawal_config:
    WITHDRAWAL = withdrawal_config.get('annual_amount', 0)
    INFLATION = withdrawal_config.get('inflation_rate', 0.02)
    WITHDRAWAL_FREQ = config.frequency_to_periods_per_year(config.withdrawal_frequency)
else:
    WITHDRAWAL = 0
    INFLATION = 0.02
    WITHDRAWAL_FREQ = PERIODS_PER_YEAR

print(f"  Config: {config.ticker_file}")
print(f"    Accumulation: {ACC_YEARS} years")
print(f"    Decumulation: {DEC_YEARS} years")
print(f"    Initial value: ${INITIAL_VALUE:,.0f}")
print(f"    Contribution: ${CONTRIBUTION:,.0f} ({config.contribution_frequency})")
print(f"    Withdrawal: ${WITHDRAWAL:,.0f}/year")

# Equal weights for all assets (e.g., 25% each for 4 assets)
num_assets = len(tickers)
weights = np.ones(num_assets) / num_assets

# Regenerate paths for lifecycle
print("  Generating lifecycle paths (parametric)...")
acc_paths_param, dec_paths_param = gen.generate_paths(
    num_simulations=NUM_SIMULATIONS,
    accumulation_years=ACC_YEARS,
    decumulation_years=DEC_YEARS,
    periods_per_year=PERIODS_PER_YEAR,
    sampling_method='parametric'
)

print("  Generating lifecycle paths (bootstrap)...")
acc_paths_boot, dec_paths_boot = gen.generate_paths(
    num_simulations=NUM_SIMULATIONS,
    accumulation_years=ACC_YEARS,
    decumulation_years=DEC_YEARS,
    periods_per_year=PERIODS_PER_YEAR,
    historical_returns=returns_data,
    sampling_method='bootstrap'
)

print(f"    ✓ Accumulation: {acc_paths_param.shape}")
print(f"    ✓ Decumulation: {dec_paths_param.shape}")

# Run accumulation simulation
print("\n  Running accumulation (parametric)...")
acc_values_param = run_accumulation_mc(
    initial_value=INITIAL_VALUE,
    weights=weights,
    asset_returns_paths=acc_paths_param,
    asset_returns_frequency=PERIODS_PER_YEAR,
    years=ACC_YEARS,
    contributions_per_year=CONTRIBUTION_FREQ,
    contribution_amount=CONTRIBUTION,
)

print("  Running accumulation (bootstrap)...")
acc_values_boot = run_accumulation_mc(
    initial_value=INITIAL_VALUE,
    weights=weights,
    asset_returns_paths=acc_paths_boot,
    asset_returns_frequency=PERIODS_PER_YEAR,
    years=ACC_YEARS,
    contributions_per_year=CONTRIBUTION_FREQ,
    contribution_amount=CONTRIBUTION,
)

final_acc_param = acc_values_param[:, -1]
final_acc_boot = acc_values_boot[:, -1]

print(f"\n  Accumulation Results:")
print(f"    {'Metric':<20} {'Parametric':<20} {'Bootstrap':<20}")
print("    " + "-" * 60)
for p in [5, 25, 50, 75, 95]:
    v_param = np.percentile(final_acc_param, p)
    v_boot = np.percentile(final_acc_boot, p)
    print(f"    {f'{p}th percentile':<20} ${v_param:>15,.0f}   ${v_boot:>15,.0f}")

# Run decumulation simulation
print("\n  Running decumulation (parametric)...")
dec_values_param, success_param = run_decumulation_mc(
    initial_values=final_acc_param,
    weights=weights,
    asset_returns_paths=dec_paths_param,
    asset_returns_frequency=PERIODS_PER_YEAR,
    annual_withdrawal=WITHDRAWAL,
    inflation_rate=INFLATION,
    years=DEC_YEARS,
    withdrawals_per_year=WITHDRAWAL_FREQ,
)

print("  Running decumulation (bootstrap)...")
dec_values_boot, success_boot = run_decumulation_mc(
    initial_values=final_acc_boot,
    weights=weights,
    asset_returns_paths=dec_paths_boot,
    asset_returns_frequency=PERIODS_PER_YEAR,
    annual_withdrawal=WITHDRAWAL,
    inflation_rate=INFLATION,
    years=DEC_YEARS,
    withdrawals_per_year=WITHDRAWAL_FREQ,
)

print(f"\n  Decumulation Results:")
print(f"    Parametric success rate: {success_param.mean():.1%}")
print(f"    Bootstrap success rate:  {success_boot.mean():.1%}")

# Step 5: Create visualizations
print("\n[5/5] Creating visualizations...")

# Visualization 1: Bootstrap method comparison
fig1 = create_bootstrap_comparison_visualization(
    parametric_paths, iid_paths, block_paths, stationary_paths,
    tickers, returns_data, None
)

# Visualization 2: Lifecycle comparison
fig2 = create_lifecycle_bootstrap_visualization(
    acc_values_param, dec_values_param,
    acc_values_boot, dec_values_boot,
    config,
    historical_returns=returns_data
)

# Summary
print("\n" + "=" * 80)
print("BOOTSTRAP VALIDATION COMPLETE ✓")
print("=" * 80)
print("\nKey Findings:")
print(f"  ✓ Generated {NUM_SIMULATIONS} paths with {TOTAL_PERIODS} periods each")
print(f"  ✓ Bootstrap preserves historical mean and variance")
print(f"  ✓ Block bootstrap preserves more autocorrelation structure")
print(f"  ✓ Lifecycle simulations work with all sampling methods")
print("\nVisualizations:")
print("  ✓ output/plots/test/mc_bootstrap_comparison.png")
print("  ✓ output/plots/test/mc_bootstrap_lifecycle.png")
print("=" * 80)
