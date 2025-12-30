#!/usr/bin/env python3
"""
Production Monte Carlo Simulator.

Runs lifecycle Monte Carlo simulations with:
- Bootstrap sampling using Yahoo Finance historical data
- Parametric sampling using sim_params mean/covariance
- Parameter sweep support

If historical data is available: runs BOTH parametric AND bootstrap.
If historical data is unavailable: runs PARAMETRIC ONLY.

Usage:
    # Basic run with Yahoo Finance data
    uv run python src/run_mc.py --config configs/test_simple_buyhold.json

    # Run parameter sweep
    uv run python src/run_mc.py --config configs/test_simple_buyhold.json --sweep
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import SystemConfig
from src.montecarlo import MCPathGenerator
from src.montecarlo.lifecycle import run_accumulation_mc, run_decumulation_mc
from src.data.market_data import load_returns_data

import ipdb

# ============================================================================
# Parameter Sweep Registry
# ============================================================================

SWEEP_PARAMS = {
    'initial_portfolio_value': {
        'type': 'currency',
        'default_range': (500_000, 2_000_000, 500_000),
        'description': 'Initial portfolio value at MC start',
    },
    'annual_withdrawal_amount': {
        'type': 'currency',
        'default_range': (20_000, 80_000, 10_000),
        'description': 'Annual withdrawal amount during decumulation',
    },
    'contribution_amount': {
        'type': 'currency',
        'default_range': (0, 5_000, 500),
        'description': 'Per-period contribution amount',
    },
    'inflation_rate': {
        'type': 'percentage',
        'default_range': (0.01, 0.06, 0.01),
        'description': 'Annual inflation rate',
    },
    'retirement_date': {
        'type': 'date',
        'default_range': ('2027-01-01', '2039-01-01', '1Y'),
        'description': 'Retirement date',
    },
    'simulation_horizon_years': {
        'type': 'numeric',
        'default_range': (15, 30, 5),
        'description': 'Years in decumulation phase',
    },
    'employer_match_rate': {
        'type': 'percentage',
        'default_range': (0.0, 0.10, 0.02),
        'description': 'Employer match rate',
    },
}


def _get_param_type(param_name: str) -> str:
    """Get parameter type from registry."""
    return SWEEP_PARAMS.get(param_name, {}).get('type', 'numeric')


def _get_default_range(param_name: str) -> tuple:
    """Get default sweep range from registry."""
    param_info = SWEEP_PARAMS.get(param_name)
    if param_info:
        return param_info['default_range']
    return None


# ============================================================================
# Helper Functions
# ============================================================================

def _get_tickers_from_config(config: SystemConfig) -> list:
    """
    Get ticker list from config file.

    Reads ticker_file from config and extracts Symbol column.

    Raises:
        FileNotFoundError: If ticker file doesn't exist
    """
    ticker_file = config.ticker_file
    if not os.path.isabs(ticker_file):
        ticker_file = os.path.join(PROJECT_ROOT, ticker_file)

    if not os.path.exists(ticker_file):
        raise FileNotFoundError(f"Ticker file not found: {ticker_file}")

    tickers_df = pd.read_csv(ticker_file, skipinitialspace=True)
    tickers_df.columns = tickers_df.columns.str.strip()

    if 'ticker' in tickers_df.columns:
        tickers_df = tickers_df.rename(columns={'ticker': 'Symbol'})

    return tickers_df['Symbol'].tolist()


def _config_freq_to_yf(simulation_frequency: str) -> str:
    """Convert config frequency to Yahoo Finance frequency."""
    freq_map = {
        'daily': 'D',
        'weekly': 'W',
        'biweekly': 'W',  # YF doesn't have biweekly, use weekly
        'monthly': 'M',
        'quarterly': 'M',  # YF doesn't have quarterly, use monthly
        'annual': 'M'  # Use monthly for annual
    }
    return freq_map.get(simulation_frequency, 'W')


def load_bootstrap_data(config: SystemConfig) -> tuple:
    """
    Load historical returns for bootstrap sampling.

    Uses load_returns_data() from market_data.py to fetch Yahoo Finance data.
    Returns (None, None) if data is unavailable.

    Parameters:
    -----------
    config : SystemConfig
        Configuration object

    Returns:
    --------
    tuple: (returns_df, tickers) or (None, None) if unavailable
    """
    tickers = _get_tickers_from_config(config)
    ticker_dict = {t: t for t in tickers}

    try:
        print(f"  Fetching historical data from Yahoo Finance...")
        print(f"    Tickers: {tickers}")
        print(f"    Period: {config.start_date} to {config.end_date}")

        returns_df, _ = load_returns_data(
            mode='yahoo',
            tickers=ticker_dict,
            frequency=_config_freq_to_yf(config.simulation_frequency),
            start_date=config.start_date,
            end_date=config.end_date,
            use_cache=True
        )

        print(f"    Loaded {len(returns_df)} periods ({config.simulation_frequency})")
        return returns_df, tickers

    except Exception as e:
        print(f"  Warning: Could not load historical data ({e})")
        return None, None


# ============================================================================
# Core Simulation Functions
# ============================================================================

def run_mc_simulation(
    config: SystemConfig,
    historical_returns: Optional[pd.DataFrame] = None,
    num_simulations: int = None,
    seed: int = 42,
    verbose: bool = True
) -> dict:
    """
    Run Monte Carlo simulation.

    If historical_returns is provided: runs BOTH parametric AND bootstrap.
    If historical_returns is None: runs PARAMETRIC ONLY using sim_params.

    Parameters:
    -----------
    config : SystemConfig
        Configuration object with lifecycle parameters
    historical_returns : pd.DataFrame, optional
        Historical returns data for bootstrap sampling. If None, runs parametric only.
    num_simulations : int, optional
        Number of MC simulations (overrides config)
    seed : int
        Random seed for reproducibility
    verbose : bool
        Print progress messages

    Returns:
    --------
    dict with keys:
        - 'acc_percentiles_param': dict of {5, 25, 50, 75, 95: value}
        - 'acc_percentiles_boot': dict or None (if no historical data)
        - 'dec_percentiles_param': dict of {5, 25, 50, 75, 95: value}
        - 'dec_percentiles_boot': dict or None
        - 'dec_success_param': float (success rate)
        - 'dec_success_boot': float or None
        - 'acc_values_param': np.ndarray (num_sims, periods)
        - 'acc_values_boot': np.ndarray or None
        - 'dec_values_param': np.ndarray (num_sims, periods)
        - 'dec_values_boot': np.ndarray or None
        - 'has_bootstrap': bool
    """
    # Load tickers and weights from config
    ticker_file = config.ticker_file
    if not os.path.isabs(ticker_file):
        ticker_file = os.path.join(PROJECT_ROOT, ticker_file)
    tickers_df = pd.read_csv(ticker_file)
    tickers = tickers_df['Symbol'].tolist()
    weights_dict = dict(zip(tickers_df['Symbol'], tickers_df['Weight']))
    weights = np.array([weights_dict[t] for t in tickers])

    has_bootstrap = historical_returns is not None
    num_sims = num_simulations or config.num_mc_simulations
    periods_per_year = config.frequency_to_periods_per_year(config.simulation_frequency)
    simulation_frequency = config.get_simulation_pandas_frequency()

    # Load time-varying parameters from files (matching test_mc_validation.py)
    mean_file = config.simulated_mean_returns_file
    cov_file = config.simulated_cov_matrices_file
    if not os.path.isabs(mean_file):
        mean_file = os.path.join(PROJECT_ROOT, mean_file)
    if not os.path.isabs(cov_file):
        cov_file = os.path.join(PROJECT_ROOT, cov_file)
    mean_returns_df = pd.read_csv(mean_file, index_col=0)
    cov_matrices_2d = np.loadtxt(cov_file)
    n_assets = mean_returns_df.shape[1]
    cov_matrices = cov_matrices_2d.reshape(2, n_assets, n_assets)

    # Align dates to frequency
    mc_start_date = SystemConfig.align_date_to_frequency(
        config.get_mc_start_date(),
        config.get_contribution_pandas_frequency()
    )
    retirement_date_config = SystemConfig.align_date_to_frequency(
        config.retirement_date,
        config.get_contribution_pandas_frequency()
    )

    # Create time-varying DataFrames with regime dates
    regime_dates = [mc_start_date, retirement_date_config]
    mean_ts = pd.DataFrame(
        mean_returns_df.values,
        index=pd.DatetimeIndex(regime_dates),
        columns=mean_returns_df.columns
    )
    cov_ts = pd.DataFrame(
        [{'cov_matrix': cov} for cov in cov_matrices],
        index=pd.DatetimeIndex(regime_dates)
    )

    if verbose:
        print(f"    Using time-varying parameters from files")
        print(f"    Regime dates: {regime_dates}")

    num_assets = len(tickers)

    if verbose:
        print(f"  Running {num_sims} simulations...")
        print(f"    Assets: {tickers}")
        print(f"    Frequency: {config.simulation_frequency} ({periods_per_year}/year)")
        print(f"    Mode: {'Parametric + Bootstrap' if has_bootstrap else 'Parametric only'}")

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

    if verbose:
        print(f"    Accumulation: {acc_years} years, initial ${initial_value:,.0f}")
        print(f"    Decumulation: {dec_years} years, withdrawal ${withdrawal:,.0f}/year")

    # Create generator (matching test_mc_validation.py)
    gen = MCPathGenerator(tickers, seed=seed)

    # Generate lifecycle paths with TIME-VARYING parameters
    if verbose:
        print("    Generating parametric paths with time-varying params...")
    # ipdb.set_trace()
    acc_paths_param, dec_paths_param = gen.generate_paths(
        num_simulations=num_sims,
        accumulation_years=acc_years,
        decumulation_years=dec_years,
        periods_per_year=periods_per_year,
        start_date=mc_start_date,
        frequency=simulation_frequency,
        mean_returns=mean_ts,
        cov_matrices=cov_ts,
        reindex_method=config.mc_reindex_method
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

    final_acc_param = acc_values_param[:, -1]

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

    final_dec_param = dec_values_param[:, -1]

    # Bootstrap (only if historical data available)
    if has_bootstrap:
        if verbose:
            print("    Generating bootstrap paths...")
        # ipdb.set_trace()
        # historical_returns = 0.5 * historical_returns
        acc_paths_boot, dec_paths_boot = gen.generate_paths(
            num_simulations=num_sims,
            accumulation_years=acc_years,
            decumulation_years=dec_years,
            periods_per_year=periods_per_year,
            start_date=mc_start_date,
            frequency=simulation_frequency,
            historical_returns=historical_returns,
            sampling_method='bootstrap',
            reindex_method=config.mc_reindex_method
        )

        acc_values_boot = run_accumulation_mc(
            initial_value=initial_value,
            weights=weights,
            asset_returns_paths=acc_paths_boot,
            asset_returns_frequency=periods_per_year,
            years=acc_years,
            contributions_per_year=contribution_freq,
            contribution_amount=contribution,
        )

        final_acc_boot = acc_values_boot[:, -1]

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

        final_dec_boot = dec_values_boot[:, -1]
    else:
        acc_values_boot = None
        dec_values_boot = None
        final_acc_boot = None
        final_dec_boot = None
        success_boot = None

    # Collect percentiles
    percentiles = [5, 25, 50, 75, 95]

    # ipdb.set_trace()
    result = {
        'acc_percentiles_param': {p: np.percentile(final_acc_param, p) for p in percentiles},
        'dec_percentiles_param': {p: np.percentile(final_dec_param, p) for p in percentiles},
        'dec_success_param': success_param.mean(),
        'acc_values_param': acc_values_param,
        'dec_values_param': dec_values_param,
        'has_bootstrap': has_bootstrap,
    }

    if has_bootstrap:
        result['acc_percentiles_boot'] = {p: np.percentile(final_acc_boot, p) for p in percentiles}
        result['dec_percentiles_boot'] = {p: np.percentile(final_dec_boot, p) for p in percentiles}
        result['dec_success_boot'] = success_boot.mean()
        result['acc_values_boot'] = acc_values_boot
        result['dec_values_boot'] = dec_values_boot
    else:
        result['acc_percentiles_boot'] = None
        result['dec_percentiles_boot'] = None
        result['dec_success_boot'] = None
        result['acc_values_boot'] = None
        result['dec_values_boot'] = None

    return result


# ============================================================================
# Parameter Sweep
# ============================================================================

def _is_date_param(param_name: str) -> bool:
    """Check if parameter is a date type."""
    return _get_param_type(param_name) == 'date'


def _format_param_value(value, param_name: str) -> str:
    """
    Format parameter value for display using registry type.

    Uses SWEEP_PARAMS registry to determine formatting:
    - 'currency': $1.5M, $500K, $1,000
    - 'percentage': 3.0%
    - 'date': YYYY-MM-DD
    - 'numeric': 1,000.00
    """
    param_type = _get_param_type(param_name)

    if param_type == 'date':
        return str(value)[:10]  # YYYY-MM-DD
    elif param_type == 'currency':
        if isinstance(value, (int, float)):
            if abs(value) >= 1e6:
                return f"${value/1e6:.1f}M"
            elif abs(value) >= 1000:
                return f"${value/1e3:.0f}K"
            return f"${value:,.0f}"
        return str(value)
    elif param_type == 'percentage':
        if isinstance(value, (int, float)):
            return f"{value:.1%}"
        return str(value)
    else:  # 'numeric' or unknown
        if isinstance(value, (int, float)):
            if value == int(value):
                return f"{int(value):,}"
            return f"{value:,.2f}"
        return str(value)


def run_parameter_sweep(
    config_path: str,
    param_name: str,
    param_values: list,
    num_simulations: int = None,
    seed: int = 42,
    verbose: bool = True,
    skip_bootstrap: bool = False
) -> pd.DataFrame:
    """
    Run Monte Carlo simulations across parameter values.

    Parameters:
    -----------
    config_path : str
        Path to base JSON config file
    param_name : str
        Parameter to sweep (e.g., 'initial_portfolio_value', 'retirement_date')
    param_values : list
        List of values to test (numbers or date strings like '2030-01-01')
    num_simulations : int, optional
        Number of MC simulations per run
    seed : int
        Random seed for reproducibility
    verbose : bool
        Print progress updates
    skip_bootstrap : bool
        If True, skip bootstrap sampling (parametric only)

    Returns:
    --------
    pd.DataFrame with columns for parameter value and percentiles
    """
    results = []
    is_date = _is_date_param(param_name)

    # Load historical data once (use first config for data loading)
    base_config = SystemConfig.from_json(config_path)
    if skip_bootstrap:
        historical_returns, tickers = None, None
    else:
        historical_returns, tickers = load_bootstrap_data(base_config)
    has_bootstrap = historical_returns is not None
    data_source = "yahoo_finance" if has_bootstrap else "parametric_only"

    for i, value in enumerate(param_values):
        if verbose:
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(param_values)}] Running {param_name} = {_format_param_value(value, param_name)}")
            print(f"{'='*60}")

        # Load fresh config and modify parameter
        config = SystemConfig.from_json(config_path)
        setattr(config, param_name, value)

        # Run simulation
        result = run_mc_simulation(
            config, historical_returns,
            num_simulations=num_simulations,
            seed=seed,
            verbose=verbose
        )

        # Flatten results into row
        row = {
            param_name: value,
            'data_source': data_source
        }

        for pct in [5, 25, 50, 75, 95]:
            row[f'acc_p{pct}_param'] = result['acc_percentiles_param'][pct]
            row[f'dec_p{pct}_param'] = result['dec_percentiles_param'][pct]
            if has_bootstrap:
                row[f'acc_p{pct}_boot'] = result['acc_percentiles_boot'][pct]
                row[f'dec_p{pct}_boot'] = result['dec_percentiles_boot'][pct]

        row['dec_success_param'] = result['dec_success_param']
        if has_bootstrap:
            row['dec_success_boot'] = result['dec_success_boot']

        results.append(row)

        if verbose:
            msg = f"  Accumulation P50: Param=${result['acc_percentiles_param'][50]:,.0f}"
            if has_bootstrap:
                msg += f", Boot=${result['acc_percentiles_boot'][50]:,.0f}"
            print(msg)

            msg = f"  Dec Success: Param={result['dec_success_param']:.1%}"
            if has_bootstrap:
                msg += f", Boot={result['dec_success_boot']:.1%}"
            print(msg)

    return pd.DataFrame(results)


def run_parameter_grid_sweep(
    config_path: str,
    param1_name: str,
    param1_values: list,
    param2_name: str,
    param2_values: list,
    num_simulations: int = None,
    seed: int = 42,
    verbose: bool = True,
    skip_bootstrap: bool = False
) -> pd.DataFrame:
    """
    Run Monte Carlo simulations across a 2D grid of parameter values.

    Parameters:
    -----------
    config_path : str
        Path to base JSON config file
    param1_name : str
        First parameter to sweep (rows in grid)
    param1_values : list
        List of values for first parameter
    param2_name : str
        Second parameter to sweep (columns in grid)
    param2_values : list
        List of values for second parameter
    num_simulations : int, optional
        Number of MC simulations per run
    seed : int
        Random seed for reproducibility
    verbose : bool
        Print progress updates
    skip_bootstrap : bool
        If True, skip bootstrap sampling (parametric only)

    Returns:
    --------
    pd.DataFrame with columns for both parameter values and all metrics
    """
    results = []
    total_runs = len(param1_values) * len(param2_values)

    # Load historical data once (use first config for data loading)
    base_config = SystemConfig.from_json(config_path)
    if skip_bootstrap:
        historical_returns, tickers = None, None
    else:
        historical_returns, tickers = load_bootstrap_data(base_config)
    has_bootstrap = historical_returns is not None
    data_source = "yahoo_finance" if has_bootstrap else "parametric_only"

    run_num = 0
    for v1 in param1_values:
        for v2 in param2_values:
            run_num += 1
            if verbose:
                print(f"\n{'='*60}")
                print(f"[{run_num}/{total_runs}] {param1_name}={_format_param_value(v1, param1_name)}, "
                      f"{param2_name}={_format_param_value(v2, param2_name)}")
                print(f"{'='*60}")

            # Load fresh config and modify both parameters
            config = SystemConfig.from_json(config_path)
            setattr(config, param1_name, v1)
            setattr(config, param2_name, v2)

            # Run simulation
            result = run_mc_simulation(
                config, historical_returns,
                num_simulations=num_simulations,
                seed=seed,
                verbose=verbose
            )

            # Flatten results into row
            row = {
                param1_name: v1,
                param2_name: v2,
                'data_source': data_source
            }

            for pct in [5, 25, 50, 75, 95]:
                row[f'acc_p{pct}_param'] = result['acc_percentiles_param'][pct]
                row[f'dec_p{pct}_param'] = result['dec_percentiles_param'][pct]
                if has_bootstrap:
                    row[f'acc_p{pct}_boot'] = result['acc_percentiles_boot'][pct]
                    row[f'dec_p{pct}_boot'] = result['dec_percentiles_boot'][pct]

            row['dec_success_param'] = result['dec_success_param']
            if has_bootstrap:
                row['dec_success_boot'] = result['dec_success_boot']

            results.append(row)

            if verbose:
                msg = f"  Dec Success: Param={result['dec_success_param']:.1%}"
                if has_bootstrap:
                    msg += f", Boot={result['dec_success_boot']:.1%}"
                print(msg)

    return pd.DataFrame(results)


# ============================================================================
# Visualization
# ============================================================================

def create_single_run_visualization(
    result: dict,
    config: SystemConfig,
    output_path: str = None
) -> plt.Figure:
    """
    Create visualization for a single MC run.

    Parameters:
    -----------
    result : dict
        Results from run_mc_simulation()
    config : SystemConfig
        Configuration object
    output_path : str, optional
        Path to save the figure

    Returns:
    --------
    plt.Figure
    """
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    acc_values_param = result['acc_values_param']
    acc_values_boot = result['acc_values_boot']
    dec_values_param = result['dec_values_param']
    dec_values_boot = result['dec_values_boot']

    num_paths = min(30, acc_values_param.shape[0])

    # Colors
    param_color = '#2E86AB'  # Blue for parametric
    boot_color = '#A23B72'   # Magenta for bootstrap

    # Panel 1: Accumulation Paths - Parametric + Bootstrap overlaid
    ax = axes[0, 0]
    for i in range(num_paths):
        ax.plot(acc_values_param[i, :], alpha=0.3, linewidth=0.8, color=param_color)
    if acc_values_boot is not None:
        for i in range(num_paths):
            ax.plot(acc_values_boot[i, :], alpha=0.3, linewidth=0.8, color=boot_color)
        ax.set_title('Accumulation (Parametric + Bootstrap)', fontsize=12, fontweight='bold')
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=param_color, lw=2, alpha=0.7, label='Parametric'),
            Line2D([0], [0], color=boot_color, lw=2, alpha=0.7, label='Bootstrap'),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc='upper left')
    else:
        ax.set_title('Accumulation (Parametric only)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Period', fontsize=10)
    ax.set_ylabel('Portfolio Value ($)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))

    # Panel 2: Decumulation Paths - Parametric + Bootstrap overlaid
    ax = axes[0, 1]
    for i in range(num_paths):
        ax.plot(dec_values_param[i, :], alpha=0.3, linewidth=0.8, color=param_color)
    if dec_values_boot is not None:
        for i in range(num_paths):
            ax.plot(dec_values_boot[i, :], alpha=0.3, linewidth=0.8, color=boot_color)
        ax.set_title('Decumulation (Parametric + Bootstrap)', fontsize=12, fontweight='bold')
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=param_color, lw=2, alpha=0.7, label='Parametric'),
            Line2D([0], [0], color=boot_color, lw=2, alpha=0.7, label='Bootstrap'),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc='upper right')
    else:
        ax.set_title('Decumulation (Parametric only)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Period', fontsize=10)
    ax.set_ylabel('Portfolio Value ($)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))

    # Panel 3: Time-varying mean returns (from test_mc_validation.py)
    ax = axes[0, 2]

    # Load time-varying parameters for visualization
    mean_file = config.simulated_mean_returns_file
    if not os.path.isabs(mean_file):
        mean_file = os.path.join(PROJECT_ROOT, mean_file)
    mean_returns_df = pd.read_csv(mean_file, index_col=0)

    # Get contribution frequency for date generation
    contribution_freq = config.get_contribution_pandas_frequency()
    mc_start_date = pd.Timestamp(SystemConfig.align_date_to_frequency(
        config.get_mc_start_date(),
        contribution_freq
    ))
    retirement_date_config = pd.Timestamp(SystemConfig.align_date_to_frequency(
        config.retirement_date,
        contribution_freq
    ))

    # Create regime dates and time-varying DataFrame
    regime_dates = [mc_start_date, retirement_date_config]
    tickers_list = mean_returns_df.columns.tolist()
    mean_ts = pd.DataFrame(
        mean_returns_df.values,
        index=pd.DatetimeIndex(regime_dates),
        columns=tickers_list
    )

    # Create full date range for visualization
    acc_years = int(config.get_accumulation_years())
    dec_years = int(config.get_decumulation_years())
    periods_per_year = config.frequency_to_periods_per_year(config.simulation_frequency)
    total_periods = (acc_years + dec_years) * periods_per_year
    dates = pd.date_range(start=mc_start_date, periods=total_periods + 1, freq=contribution_freq)

    # Reindex mean_ts to full date range for visualization
    mean_ts_full = mean_ts.reindex(dates, method='ffill')

    # Plot each ticker's time-varying mean return
    for ticker in tickers_list:
        ax.plot(mean_ts_full.index, mean_ts_full[ticker], label=ticker, linewidth=2, alpha=0.8)

    # Mark the regime shift points
    for shift_date in mean_ts.index:
        ax.axvline(shift_date, color='gray', linestyle=':', linewidth=1, alpha=0.4)

    # Highlight retirement date
    ax.axvline(retirement_date_config, color='red', linestyle='--',
               linewidth=2, label='Retirement', alpha=0.6)

    # Add shaded regions for accumulation vs decumulation
    ax.axvspan(dates[0], retirement_date_config, alpha=0.05, color='green')
    ax.axvspan(retirement_date_config, dates[-1], alpha=0.05, color='red')

    ax.set_title('Time-Varying Mean Returns (Annual)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Expected Return', fontsize=10)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))  # Show every 5 years
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Panel 4: Final accumulation distribution comparison
    ax = axes[1, 0]
    final_param = acc_values_param[:, -1]

    ax.hist(final_param, bins=30, alpha=0.5, label='Parametric', color=param_color, density=True)
    if acc_values_boot is not None:
        final_boot = acc_values_boot[:, -1]
        ax.hist(final_boot, bins=30, alpha=0.5, label='Bootstrap', color=boot_color, density=True)

    # Add percentile lines
    p50_param = np.percentile(final_param, 50)
    ax.axvline(p50_param, color=param_color, linestyle='--', linewidth=2,
               label=f'Param Median: ${p50_param/1e6:.2f}M')
    if acc_values_boot is not None:
        p50_boot = np.percentile(final_boot, 50)
        ax.axvline(p50_boot, color=boot_color, linestyle='--', linewidth=2,
                   label=f'Boot Median: ${p50_boot/1e6:.2f}M')

    ax.set_title('Final Accumulation Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Final Value ($)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

    # -------------------------------------------------------------------------
    # Shared setup for lifecycle plots (Panels 5 and 6)
    # -------------------------------------------------------------------------
    num_spaghetti = min(50, acc_values_param.shape[0])

    # Get actual periods from value arrays (authoritative source)
    acc_periods = acc_values_param.shape[1]
    dec_periods = dec_values_param.shape[1]

    # Get contribution frequency for date generation (reuse from Panel 3)
    # Generate accumulation dates (starting from mc_start_date)
    acc_dates = pd.date_range(start=mc_start_date, periods=acc_periods, freq=contribution_freq)
    retirement_date = acc_dates[-1]

    # Generate decumulation dates (starting from retirement, overlapping one point)
    dec_dates = pd.date_range(start=retirement_date, periods=dec_periods, freq=contribution_freq)

    # Colors by phase (green=accumulation, red=decumulation)
    acc_color = 'green'             # Accumulation
    dec_color = '#E60026'           # Strong red (decumulation)

    # -------------------------------------------------------------------------
    # Panel 5: Bootstrap Lifecycle Paths
    # -------------------------------------------------------------------------
    ax = axes[1, 1]

    # Calculate mid-dates for phase labels
    acc_mid_date = acc_dates[len(acc_dates)//2]
    dec_mid_date = dec_dates[len(dec_dates)//2]

    if acc_values_boot is not None:
        boot_acc_color = 'blue'         # Bootstrap accumulation
        boot_dec_color = '#0072B2'      # Contrasting blue (decumulation bootstrap)

        # Plot bootstrap paths
        for i in range(num_spaghetti):
            ax.plot(acc_dates, acc_values_boot[i, :], color=boot_acc_color, alpha=0.15, linewidth=0.8)
            ax.plot(dec_dates, dec_values_boot[i, :], color=boot_dec_color, alpha=0.15, linewidth=0.8)

        # Add vertical line at retirement
        ax.axvline(retirement_date, color='black', linestyle='--', linewidth=2, alpha=0.7)

        # Legend for bootstrap
        legend_elements_boot = [
            Line2D([0], [0], color=boot_acc_color, lw=2, alpha=0.7, label='Accumulation'),
            Line2D([0], [0], color=boot_dec_color, lw=2, alpha=0.7, label='Decumulation'),
            Line2D([0], [0], color='black', lw=2, linestyle='--', alpha=0.7, label='Retirement'),
        ]
        ax.legend(handles=legend_elements_boot, fontsize=7, loc='lower left', frameon=True)

        ax.set_title(f'Bootstrap Lifecycle ({num_spaghetti} sims)', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No Bootstrap Data', ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('Bootstrap Lifecycle (N/A)', fontsize=12, fontweight='bold')

    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Portfolio Value ($)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale to see full range
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add phase labels (after setting scale so ylim is correct)
    if acc_values_boot is not None:
        ylim = ax.get_ylim()
        ax.text(acc_mid_date, ylim[1]*0.5, 'Acc',
                ha='center', fontsize=9, color='darkblue', weight='bold', alpha=0.7)
        ax.text(dec_mid_date, ylim[1]*0.5, 'Dec',
                ha='center', fontsize=9, color='#0072B2', weight='bold', alpha=0.7)

    # -------------------------------------------------------------------------
    # Panel 6: Parametric Lifecycle Paths
    # -------------------------------------------------------------------------
    ax = axes[1, 2]

    # Plot parametric paths (solid lines)
    for i in range(num_spaghetti):
        ax.plot(acc_dates, acc_values_param[i, :], color=acc_color, alpha=0.15, linewidth=0.8)
        ax.plot(dec_dates, dec_values_param[i, :], color=dec_color, alpha=0.15, linewidth=0.8)

    # Add vertical line at retirement
    ax.axvline(retirement_date, color='black', linestyle='--', linewidth=2, alpha=0.7)

    # Legend for parametric
    legend_elements_param = [
        Line2D([0], [0], color=acc_color, lw=2, alpha=0.7, label='Accumulation'),
        Line2D([0], [0], color=dec_color, lw=2, alpha=0.7, label='Decumulation'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', alpha=0.7, label='Retirement'),
    ]
    ax.legend(handles=legend_elements_param, fontsize=7, loc='lower left', frameon=True)

    ax.set_title(f'Parametric Lifecycle ({num_spaghetti} sims)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Portfolio Value ($)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale to see full range
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add phase labels
    ylim = ax.get_ylim()
    ax.text(acc_mid_date, ylim[1]*0.5, 'Acc',
            ha='center', fontsize=9, color='darkgreen', weight='bold', alpha=0.7)
    ax.text(dec_mid_date, ylim[1]*0.5, 'Dec',
            ha='center', fontsize=9, color='darkred', weight='bold', alpha=0.7)

    plt.suptitle(f'Monte Carlo Lifecycle Simulation\nInitial: ${config.initial_portfolio_value:,.0f}, '
                 f'Withdrawal: ${config.annual_withdrawal_amount:,.0f}/year',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig


def _create_sweep_figure(
    df: pd.DataFrame,
    param_name: str,
    method: str,
    color: str,
    output_path: str = None,
    config_info: dict = None
) -> plt.Figure:
    """
    Create visualization for a single sampling method (parametric or bootstrap).

    Parameters:
    -----------
    df : pd.DataFrame
        Results from run_parameter_sweep()
    param_name : str
        Name of the swept parameter
    method : str
        'param' or 'boot'
    color : str
        Color for the plots
    output_path : str, optional
        Path to save the figure
    config_info : dict, optional
        Configuration parameters to display on the figure

    Returns:
    --------
    plt.Figure
    """
    import matplotlib.dates as mdates

    method_label = 'Parametric' if method == 'param' else 'Bootstrap'
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))  # Taller for config text

    is_date = _is_date_param(param_name)
    x_label = param_name.replace('_', ' ').title()

    # Handle date vs numeric x-axis
    if is_date:
        # Convert date strings to datetime for plotting
        x = pd.to_datetime(df[param_name])
        x_numeric = mdates.date2num(x)
        use_dates = True
    else:
        x = df[param_name].values
        x_numeric = x
        use_dates = False

    # Format x-axis based on parameter type
    if is_date:
        x_formatter = mdates.DateFormatter('%Y')
    elif 'value' in param_name.lower() or 'amount' in param_name.lower():
        x_formatter = plt.FuncFormatter(lambda v, p: f'${v/1e6:.1f}M' if v >= 1e6 else f'${v/1e3:.0f}K')
    else:
        x_formatter = plt.FuncFormatter(lambda v, p: f'{v:,.0f}')

    # Panel 1: Accumulation percentiles (fan chart)
    ax = axes[0, 0]
    ax.fill_between(x, df[f'acc_p5_{method}'], df[f'acc_p95_{method}'],
                    alpha=0.2, color=color, label='5-95%')
    ax.fill_between(x, df[f'acc_p25_{method}'], df[f'acc_p75_{method}'],
                    alpha=0.3, color=color, label='25-75%')
    ax.plot(x, df[f'acc_p50_{method}'], '-', color=color, linewidth=2,
            label='Median')

    ax.set_title(f'Final Accumulation Value ({method_label})', fontsize=12, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel('Portfolio Value at Retirement ($)', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    if use_dates:
        ax.xaxis.set_major_formatter(x_formatter)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.xaxis.set_major_formatter(x_formatter)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, p: f'${v/1e6:.1f}M' if v >= 1e6 else f'${v/1e3:.0f}K'))

    # Panel 2: Decumulation percentiles (fan chart)
    ax = axes[0, 1]
    ax.fill_between(x, df[f'dec_p5_{method}'], df[f'dec_p95_{method}'],
                    alpha=0.2, color=color, label='5-95%')
    ax.fill_between(x, df[f'dec_p25_{method}'], df[f'dec_p75_{method}'],
                    alpha=0.3, color=color, label='25-75%')
    ax.plot(x, df[f'dec_p50_{method}'], '-', color=color, linewidth=2,
            label='Median')

    ax.set_title(f'Final Decumulation Value ({method_label})', fontsize=12, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel('Final Portfolio Value ($)', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    if use_dates:
        ax.xaxis.set_major_formatter(x_formatter)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.xaxis.set_major_formatter(x_formatter)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, p: f'${v/1e6:.1f}M' if v >= 1e6 else f'${v/1e3:.0f}K'))

    # Panel 3: Success rates
    ax = axes[1, 0]
    if use_dates:
        # For dates, calculate bar width in days (about 200 days = ~0.6 year)
        width = 200 if len(x) > 1 else 100
    else:
        width = (x[1] - x[0]) * 0.6 if len(x) > 1 else x[0] * 0.2
    ax.bar(x, df[f'dec_success_{method}'] * 100, width,
           color=color, alpha=0.8)

    ax.set_title(f'Decumulation Success Rate ({method_label})', fontsize=12, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel('Success Rate (%)', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    if use_dates:
        ax.xaxis.set_major_formatter(x_formatter)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.xaxis.set_major_formatter(x_formatter)
    ax.set_ylim(0, 105)

    # Panel 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')

    table_data = []
    for _, row in df.iterrows():
        param_val = row[param_name]
        param_str = _format_param_value(param_val, param_name)

        table_data.append([
            param_str,
            f"${row[f'acc_p50_{method}']/1e6:.2f}M",
            f"${row[f'dec_p50_{method}']/1e6:.2f}M",
            f"{row[f'dec_success_{method}']:.1%}",
        ])

    columns = [x_label, 'Acc Median', 'Dec Median', 'Success']

    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor(color)
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title(f'Summary ({method_label})', fontsize=12, fontweight='bold', pad=20)

    # Add config info text box at the bottom (usetex=False to avoid LaTeX parsing of $)
    if config_info:
        config_text = (
            f"Sweep: {config_info.get('sweep_param', param_name)}  |  "
            f"Tickers: {config_info.get('ticker_file', 'N/A')}  |  "
            f"Mean: {config_info.get('mean_returns_file', 'N/A')}  |  "
            f"Cov: {config_info.get('cov_matrices_file', 'N/A')}\n"
            f"Initial: ${config_info.get('initial_value', 0):,.0f}  |  "
            f"Retirement: {config_info.get('retirement_date', 'N/A')}  |  "
            f"Dec End: {config_info.get('dec_end_date', 'N/A')}  |  "
            f"Freq: {config_info.get('simulation_frequency', 'N/A')}  |  "
            f"Contribution: ${config_info.get('contribution_amount', 0):,.0f} ({config_info.get('contribution_frequency', 'N/A')})\n"
            f"Match: {config_info.get('employer_match_rate', 0):.0%}  |  "
            f"Withdrawal: ${config_info.get('annual_withdrawal', 0):,.0f}/yr  |  "
            f"Inflation: {config_info.get('inflation_rate', 0):.1%}  |  "
            f"Simulations: {config_info.get('num_simulations', 0):,}  |  "
            f"Seed: {config_info.get('seed', 42)}  |  "
            f"Data: {config_info.get('data_source', 'N/A')}"
        )
        fig.text(0.5, -0.02, config_text, ha='center', va='top', fontsize=8,
                 fontfamily='monospace', usetex=False,
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.suptitle(f'Parameter Sweep: {param_name.replace("_", " ").title()} ({method_label})',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])  # Leave room for config text at bottom

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig


def create_sweep_visualization(
    df: pd.DataFrame,
    param_name: str,
    output_path: str = None,
    has_bootstrap: bool = True,
    config_info: dict = None
) -> list:
    """
    Create separate visualizations for parametric and bootstrap sweep results.

    Parameters:
    -----------
    df : pd.DataFrame
        Results from run_parameter_sweep()
    param_name : str
        Name of the swept parameter
    output_path : str, optional
        Base path to save figures (will create _parametric.png and _bootstrap.png)
    has_bootstrap : bool
        Whether bootstrap data is available
    config_info : dict, optional
        Configuration parameters to display on the figures

    Returns:
    --------
    list of plt.Figure
    """
    figures = []

    # Colors
    param_color = '#2E86AB'  # Blue for parametric
    boot_color = '#A23B72'   # Magenta for bootstrap

    # Create parametric figure
    if output_path:
        base_path = output_path.replace('.png', '')
        param_path = f"{base_path}_parametric.png"
    else:
        param_path = None

    fig_param = _create_sweep_figure(df, param_name, 'param', param_color, param_path, config_info)
    figures.append(fig_param)

    # Create bootstrap figure (if data available)
    if has_bootstrap:
        if output_path:
            boot_path = f"{base_path}_bootstrap.png"
        else:
            boot_path = None

        fig_boot = _create_sweep_figure(df, param_name, 'boot', boot_color, boot_path, config_info)
        figures.append(fig_boot)

    return figures


def create_grid_sweep_visualization(
    df: pd.DataFrame,
    param1_name: str,
    param2_name: str,
    output_path: str = None,
    has_bootstrap: bool = True,
    config_info: dict = None
) -> list:
    """
    Create heatmap visualizations for 2D parameter grid sweep results.

    Creates separate figures for parametric and bootstrap showing:
    - Heatmap of decumulation success rates
    - Heatmap of median accumulation values
    - Heatmap of median decumulation values

    Parameters:
    -----------
    df : pd.DataFrame
        Results from run_parameter_grid_sweep()
    param1_name : str
        First parameter name (rows)
    param2_name : str
        Second parameter name (columns)
    output_path : str, optional
        Base path to save figures
    has_bootstrap : bool
        Whether bootstrap data is available
    config_info : dict, optional
        Configuration parameters to display

    Returns:
    --------
    list of plt.Figure
    """
    figures = []

    def _create_grid_figure(method: str, color_map: str, output_file: str = None):
        """Create a grid heatmap figure for one method (param or boot)."""
        method_label = 'Parametric' if method == 'param' else 'Bootstrap'

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Get unique values for axes
        param1_values = sorted(df[param1_name].unique())
        param2_values = sorted(df[param2_name].unique())

        # Create pivot tables
        success_pivot = df.pivot(
            index=param1_name, columns=param2_name,
            values=f'dec_success_{method}'
        )
        acc_pivot = df.pivot(
            index=param1_name, columns=param2_name,
            values=f'acc_p50_{method}'
        )
        dec_pivot = df.pivot(
            index=param1_name, columns=param2_name,
            values=f'dec_p50_{method}'
        )

        # Format tick labels
        param1_labels = [_format_param_value(v, param1_name) for v in param1_values]
        param2_labels = [_format_param_value(v, param2_name) for v in param2_values]

        # Panel 1: Success Rate Heatmap
        ax = axes[0]
        im = ax.imshow(success_pivot.values * 100, cmap='RdYlGn', aspect='auto',
                       vmin=0, vmax=100)
        ax.set_xticks(range(len(param2_labels)))
        ax.set_xticklabels(param2_labels, rotation=45, ha='right')
        ax.set_yticks(range(len(param1_labels)))
        ax.set_yticklabels(param1_labels)
        ax.set_xlabel(param2_name.replace('_', ' ').title())
        ax.set_ylabel(param1_name.replace('_', ' ').title())
        ax.set_title('Decumulation Success Rate (%)', fontweight='bold')

        # Add text annotations
        for i in range(len(param1_values)):
            for j in range(len(param2_values)):
                val = success_pivot.values[i, j] * 100
                color = 'white' if val < 50 else 'black'
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                        color=color, fontsize=8, fontweight='bold')

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Success Rate (%)')

        # Panel 2: Accumulation Median Heatmap
        ax = axes[1]
        im = ax.imshow(acc_pivot.values / 1e6, cmap='Blues', aspect='auto')
        ax.set_xticks(range(len(param2_labels)))
        ax.set_xticklabels(param2_labels, rotation=45, ha='right')
        ax.set_yticks(range(len(param1_labels)))
        ax.set_yticklabels(param1_labels)
        ax.set_xlabel(param2_name.replace('_', ' ').title())
        ax.set_ylabel(param1_name.replace('_', ' ').title())
        ax.set_title('Accumulation Median ($M)', fontweight='bold')

        # Add text annotations
        for i in range(len(param1_values)):
            for j in range(len(param2_values)):
                val = acc_pivot.values[i, j] / 1e6
                ax.text(j, i, f'${val:.1f}M', ha='center', va='center',
                        color='white' if val > acc_pivot.values.mean() / 1e6 else 'black',
                        fontsize=7)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Portfolio Value ($M)')

        # Panel 3: Decumulation Median Heatmap
        ax = axes[2]
        im = ax.imshow(dec_pivot.values / 1e6, cmap='Purples', aspect='auto')
        ax.set_xticks(range(len(param2_labels)))
        ax.set_xticklabels(param2_labels, rotation=45, ha='right')
        ax.set_yticks(range(len(param1_labels)))
        ax.set_yticklabels(param1_labels)
        ax.set_xlabel(param2_name.replace('_', ' ').title())
        ax.set_ylabel(param1_name.replace('_', ' ').title())
        ax.set_title('Decumulation Median ($M)', fontweight='bold')

        # Add text annotations
        for i in range(len(param1_values)):
            for j in range(len(param2_values)):
                val = dec_pivot.values[i, j] / 1e6
                ax.text(j, i, f'${val:.1f}M', ha='center', va='center',
                        color='white' if val > dec_pivot.values.mean() / 1e6 else 'black',
                        fontsize=7)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Portfolio Value ($M)')

        # Add config info at bottom
        if config_info:
            config_text = (
                f"Grid: {param1_name} x {param2_name}  |  "
                f"Simulations: {config_info.get('num_simulations', 0):,}  |  "
                f"Seed: {config_info.get('seed', 42)}  |  "
                f"Data: {config_info.get('data_source', 'N/A')}"
            )
            fig.text(0.5, -0.02, config_text, ha='center', va='top', fontsize=8,
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

        plt.suptitle(
            f'Grid Sweep: {param1_name.replace("_", " ").title()} vs '
            f'{param2_name.replace("_", " ").title()} ({method_label})',
            fontsize=14, fontweight='bold', y=1.02
        )
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"  Saved: {output_file}")

        return fig

    # Create parametric figure
    if output_path:
        base_path = output_path.replace('.png', '')
        param_path = f"{base_path}_parametric.png"
    else:
        param_path = None

    fig_param = _create_grid_figure('param', 'Blues', param_path)
    figures.append(fig_param)

    # Create bootstrap figure (if available)
    if has_bootstrap:
        if output_path:
            boot_path = f"{base_path}_bootstrap.png"
        else:
            boot_path = None

        fig_boot = _create_grid_figure('boot', 'Purples', boot_path)
        figures.append(fig_boot)

    return figures


# ============================================================================
# CLI Interface
# ============================================================================

def _generate_sweep_values(param_name: str, start, end, step) -> list:
    """
    Generate sweep values for a parameter using registry type.

    Parameters:
    -----------
    param_name : str
        Parameter name (must be in SWEEP_PARAMS registry)
    start, end, step : various
        Range specification (None to use defaults from registry)

    Returns:
    --------
    list of parameter values
    """
    param_type = _get_param_type(param_name)
    default_range = _get_default_range(param_name)

    if param_type == 'date':
        # Date sweep
        start_date = pd.Timestamp(start) if start else pd.Timestamp(default_range[0])
        end_date = pd.Timestamp(end) if end else pd.Timestamp(default_range[1])
        step_str = step if step else default_range[2]

        # Parse step (e.g., "1Y", "6M", "1M")
        if step_str.endswith('Y'):
            freq = 'YS'  # Year start
            multiplier = int(step_str[:-1]) if len(step_str) > 1 else 1
        elif step_str.endswith('M'):
            freq = 'MS'  # Month start
            multiplier = int(step_str[:-1]) if len(step_str) > 1 else 1
        else:
            freq = 'YS'
            multiplier = 1

        all_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        if multiplier > 1:
            all_dates = all_dates[::multiplier]
        return [d.strftime('%Y-%m-%d') for d in all_dates]

    else:
        # Numeric sweep (currency, percentage, numeric)
        if default_range:
            start_val = float(start) if start is not None else default_range[0]
            end_val = float(end) if end is not None else default_range[1]
            step_val = float(step) if step is not None else default_range[2]
        else:
            # Fallback for unknown params
            start_val = float(start) if start is not None else 0
            end_val = float(end) if end is not None else 100
            step_val = float(step) if step is not None else 10

        return np.arange(start_val, end_val + step_val / 2, step_val).tolist()


def main():
    # Build list of available sweep params for help text
    sweep_params_list = ', '.join(SWEEP_PARAMS.keys())

    parser = argparse.ArgumentParser(
        description='Monte Carlo Lifecycle Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Basic run with Yahoo Finance data
  uv run python src/run_mc.py --config configs/test_simple_buyhold.json

  # Single parameter sweep (uses default range from registry)
  uv run python src/run_mc.py --config configs/test_simple_buyhold.json \\
      --sweep --sweep-param inflation_rate

  # Single parameter sweep with custom range
  uv run python src/run_mc.py --config configs/test_simple_buyhold.json \\
      --sweep --sweep-param annual_withdrawal_amount \\
      --sweep-start 30000 --sweep-end 60000 --sweep-step 10000

  # 2D Grid sweep (all combinations of two parameters)
  uv run python src/run_mc.py --config configs/test_simple_buyhold.json \\
      --sweep --sweep-param annual_withdrawal_amount \\
      --sweep-param2 inflation_rate

  # 2D Grid sweep with custom ranges
  uv run python src/run_mc.py --config configs/test_simple_buyhold.json \\
      --sweep --sweep-param contribution_amount --sweep-start 0 --sweep-end 2000 --sweep-step 500 \\
      --sweep-param2 inflation_rate --sweep2-start 0.02 --sweep2-end 0.05 --sweep2-step 0.01

  # Date sweep (retirement date)
  uv run python src/run_mc.py --config configs/test_simple_buyhold.json \\
      --sweep --sweep-param retirement_date \\
      --sweep-start 2030-01-01 --sweep-end 2040-01-01 --sweep-step 2Y

Available sweep parameters: {sweep_params_list}
        """
    )

    # Required
    parser.add_argument('--config', '-c', required=True,
                        help='Path to JSON config file')

    # Simulation parameters
    parser.add_argument('--sims', '-n', type=int, default=None,
                        help='Number of MC simulations (overrides config)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    # Parameter sweep (first param)
    parser.add_argument('--sweep', action='store_true',
                        help='Run parameter sweep')
    parser.add_argument('--sweep-param', type=str, default=None,
                        help=f'First parameter to sweep. Available: {sweep_params_list}')
    parser.add_argument('--sweep-start', type=str, default=None,
                        help='First param start value (uses registry default if omitted)')
    parser.add_argument('--sweep-end', type=str, default=None,
                        help='First param end value (uses registry default if omitted)')
    parser.add_argument('--sweep-step', type=str, default=None,
                        help='First param step size (number, "1Y", or "1M")')

    # Second parameter (for 2D grid sweep)
    parser.add_argument('--sweep-param2', type=str, default=None,
                        help='Second parameter for 2D grid sweep')
    parser.add_argument('--sweep2-start', type=str, default=None,
                        help='Second param start value')
    parser.add_argument('--sweep2-end', type=str, default=None,
                        help='Second param end value')
    parser.add_argument('--sweep2-step', type=str, default=None,
                        help='Second param step size')

    # Output
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory (default: output/mc)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating visualization')
    parser.add_argument('--no-bootstrap', action='store_true',
                        help='Run parametric only, skip bootstrap sampling')

    args = parser.parse_args()

    # Banner
    print("=" * 80)
    print("MONTE CARLO LIFECYCLE SIMULATOR")
    print("=" * 80)

    # Load config
    config = SystemConfig.from_json(args.config)
    print(f"\nConfig: {args.config}")
    print(f"  Initial value: ${config.initial_portfolio_value:,.0f}")
    print(f"  Accumulation: {config.get_accumulation_years():.0f} years")
    print(f"  Decumulation: {config.get_decumulation_years():.0f} years")
    print(f"  Withdrawal: ${config.annual_withdrawal_amount:,.0f}/year")

    # Set output directory
    output_dir = Path(args.output) if args.output else Path('output/mc')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine mode
    if args.sweep:
        # Parameter sweep mode
        is_grid_sweep = args.sweep_param2 is not None

        if is_grid_sweep:
            print("\n[MODE] 2D Grid Parameter Sweep")
        else:
            print("\n[MODE] Single Parameter Sweep")

        # Get first parameter
        if args.sweep_param:
            param1_name = args.sweep_param
            # Warn if unknown param (but still allow it)
            if param1_name not in SWEEP_PARAMS:
                print(f"  Warning: '{param1_name}' not in registry, using defaults")
            param1_values = _generate_sweep_values(
                param1_name, args.sweep_start, args.sweep_end, args.sweep_step
            )
        elif hasattr(config, 'sweep_params') and config.sweep_params:
            # Use config sweep_params (first one)
            sweep_config = config.sweep_params[0]
            param1_name = sweep_config['name']
            param1_values = np.arange(
                sweep_config['start'],
                sweep_config['end'] + sweep_config['step'] / 2,
                sweep_config['step']
            ).tolist()
        else:
            # Default sweep
            param1_name = 'initial_portfolio_value'
            param1_values = _generate_sweep_values(param1_name, None, None, None)

        # Get second parameter (for grid sweep)
        if is_grid_sweep:
            param2_name = args.sweep_param2
            if param2_name not in SWEEP_PARAMS:
                print(f"  Warning: '{param2_name}' not in registry, using defaults")
            param2_values = _generate_sweep_values(
                param2_name, args.sweep2_start, args.sweep2_end, args.sweep2_step
            )

        # Display sweep info
        first_val1 = _format_param_value(param1_values[0], param1_name)
        last_val1 = _format_param_value(param1_values[-1], param1_name)
        print(f"  Parameter 1: {param1_name}")
        print(f"    Values: {len(param1_values)} ({first_val1} to {last_val1})")

        if is_grid_sweep:
            first_val2 = _format_param_value(param2_values[0], param2_name)
            last_val2 = _format_param_value(param2_values[-1], param2_name)
            print(f"  Parameter 2: {param2_name}")
            print(f"    Values: {len(param2_values)} ({first_val2} to {last_val2})")
            print(f"  Total runs: {len(param1_values) * len(param2_values)}")

        # Run sweep
        sweep_dir = output_dir / 'sweep'
        sweep_dir.mkdir(parents=True, exist_ok=True)

        if is_grid_sweep:
            # 2D Grid sweep
            df = run_parameter_grid_sweep(
                config_path=args.config,
                param1_name=param1_name,
                param1_values=param1_values,
                param2_name=param2_name,
                param2_values=param2_values,
                num_simulations=args.sims,
                seed=args.seed,
                verbose=True,
                skip_bootstrap=args.no_bootstrap
            )
            csv_path = sweep_dir / f'{param1_name}_{param2_name}_grid.csv'
        else:
            # Single param sweep
            df = run_parameter_sweep(
                config_path=args.config,
                param_name=param1_name,
                param_values=param1_values,
                num_simulations=args.sims,
                seed=args.seed,
                verbose=True,
                skip_bootstrap=args.no_bootstrap
            )
            csv_path = sweep_dir / f'{param1_name}_sweep.csv'

        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")

        # Check if bootstrap columns exist in results
        has_bootstrap = 'acc_p50_boot' in df.columns

        # Print summary
        print("\n" + "=" * 80)
        if is_grid_sweep:
            print("GRID SWEEP RESULTS")
        else:
            print("PARAMETER SWEEP RESULTS")
        print("=" * 80)

        # Print config context
        retirement_dt = pd.Timestamp(config.retirement_date)
        dec_end_date = retirement_dt + pd.DateOffset(years=int(config.get_decumulation_years()))
        dec_end_str = dec_end_date.strftime('%b %Y')

        print("\nConfiguration:")
        print(f"  Ticker file:        {config.ticker_file}")
        print(f"  Initial value:      ${config.initial_portfolio_value:,.0f}")
        print(f"  Retirement date:    {config.retirement_date}")
        print(f"  Decumulation end:   {dec_end_str}")
        print(f"  Contribution:       ${config.contribution_amount:,.0f} ({config.contribution_frequency})")
        print(f"  Annual withdrawal:  ${config.annual_withdrawal_amount:,.0f} ({config.withdrawal_frequency})")
        print(f"  Inflation rate:     {config.inflation_rate:.1%}")
        print(f"  MC simulations:     {args.sims or config.num_mc_simulations:,}")
        print(f"  Data source:        {'Yahoo Finance + Parametric' if has_bootstrap else 'Parametric only'}")

        if is_grid_sweep:
            # Grid sweep summary - show success rate matrix
            print(f"\nGrid: {param1_name} x {param2_name}")
            print("\nSuccess Rate (Parametric):")

            # Create pivot table for display
            success_pivot = df.pivot(
                index=param1_name, columns=param2_name, values='dec_success_param'
            )

            # Print header row
            header = f"{'':>12}"
            for v2 in sorted(df[param2_name].unique()):
                header += f"{_format_param_value(v2, param2_name):>10}"
            print(header)

            # Print data rows
            for v1 in sorted(df[param1_name].unique()):
                row = f"{_format_param_value(v1, param1_name):>12}"
                for v2 in sorted(df[param2_name].unique()):
                    val = success_pivot.loc[v1, v2] * 100
                    row += f"{val:>9.0f}%"
                print(row)

        else:
            # Single param sweep summary
            print(f"\nSweep Parameter: {param1_name.replace('_', ' ').title()}")

            is_date = _is_date_param(param1_name)
            param_headers = [_format_param_value(val, param1_name) for val in df[param1_name].values]
            col_width = 14 if not is_date else 12

            # Print header
            param_label = param1_name.replace('_', ' ').title()
            print(f"\n{param_label + ':':<25}" + "".join(f"{h:>{col_width}}" for h in param_headers))
            print("-" * (25 + col_width * len(param_headers)))

            # Success rates
            print("\nSuccess Rate:")
            row = f"{'  Parametric':<25}"
            for _, r in df.iterrows():
                row += f"{r['dec_success_param']*100:>{col_width}.1f}%"
            print(row)

            if has_bootstrap:
                row = f"{'  Bootstrap':<25}"
                for _, r in df.iterrows():
                    row += f"{r['dec_success_boot']*100:>{col_width}.1f}%"
                print(row)

            # Accumulation median
            print("\nAccumulation Median:")
            row = f"{'  Parametric':<25}"
            for _, r in df.iterrows():
                val = r['acc_p50_param']
                row += f"${val/1e6:>{col_width-1}.2f}M"
            print(row)

        # Create visualization
        if not args.no_plot:
            config_info = {
                'num_simulations': args.sims or config.num_mc_simulations,
                'seed': args.seed,
                'data_source': 'Yahoo Finance + Parametric' if has_bootstrap else 'Parametric only',
            }

            if is_grid_sweep:
                plot_path = sweep_dir / f'{param1_name}_{param2_name}_grid.png'
                figures = create_grid_sweep_visualization(
                    df, param1_name, param2_name, str(plot_path), has_bootstrap, config_info
                )
            else:
                config_info.update({
                    'sweep_param': param1_name.replace('_', ' ').title(),
                    'ticker_file': config.ticker_file,
                    'mean_returns_file': config.simulated_mean_returns_file,
                    'cov_matrices_file': config.simulated_cov_matrices_file,
                    'initial_value': config.initial_portfolio_value,
                    'retirement_date': config.retirement_date,
                    'dec_end_date': dec_end_str,
                    'simulation_frequency': config.simulation_frequency,
                    'contribution_amount': config.contribution_amount,
                    'contribution_frequency': config.contribution_frequency,
                    'employer_match_rate': config.employer_match_rate,
                    'annual_withdrawal': config.annual_withdrawal_amount,
                    'inflation_rate': config.inflation_rate,
                })
                plot_path = sweep_dir / f'{param1_name}_sweep.png'
                figures = create_sweep_visualization(
                    df, param1_name, str(plot_path), has_bootstrap, config_info
                )

            for fig in figures:
                plt.close(fig)

    else:
        # Single run mode
        print("\n[MODE] Single Run")

        # Load historical data
        print("\n[1/3] Loading historical data...")
        if args.no_bootstrap:
            print("  Skipping bootstrap (--no-bootstrap flag)")
            historical_returns, tickers = None, None
        else:
            historical_returns, tickers = load_bootstrap_data(config)
        has_bootstrap = historical_returns is not None
        data_source = "yahoo_finance" if has_bootstrap else "parametric_only"

        if has_bootstrap:
            print(f"  Data source: {data_source}")
            print(f"  Periods: {len(historical_returns)}")
        else:
            print(f"  Data source: {data_source} (no historical data available)")

        # Run simulation
        print("\n[2/3] Running Monte Carlo simulation...")
        result = run_mc_simulation(
            config, historical_returns,
            num_simulations=args.sims,
            seed=args.seed,
            verbose=True
        )

        # Print results
        print("\n" + "=" * 80)
        print("SIMULATION RESULTS")
        print("=" * 80)

        print(f"\nAccumulation (end of {config.get_accumulation_years():.0f} years):")
        if has_bootstrap:
            print(f"  {'Percentile':<15} {'Parametric':<20} {'Bootstrap':<20}")
            print("  " + "-" * 55)
            for p in [5, 25, 50, 75, 95]:
                v_param = result['acc_percentiles_param'][p]
                v_boot = result['acc_percentiles_boot'][p]
                print(f"  {f'{p}th':<15} ${v_param:>15,.0f}   ${v_boot:>15,.0f}")
        else:
            print(f"  {'Percentile':<15} {'Parametric':<20}")
            print("  " + "-" * 35)
            for p in [5, 25, 50, 75, 95]:
                v_param = result['acc_percentiles_param'][p]
                print(f"  {f'{p}th':<15} ${v_param:>15,.0f}")

        print(f"\nDecumulation (after {config.get_decumulation_years():.0f} years):")
        if has_bootstrap:
            print(f"  Success Rate: Parametric={result['dec_success_param']:.1%}, "
                  f"Bootstrap={result['dec_success_boot']:.1%}")
            print(f"  {'Percentile':<15} {'Parametric':<20} {'Bootstrap':<20}")
            print("  " + "-" * 55)
            for p in [5, 25, 50, 75, 95]:
                v_param = result['dec_percentiles_param'][p]
                v_boot = result['dec_percentiles_boot'][p]
                print(f"  {f'{p}th':<15} ${v_param:>15,.0f}   ${v_boot:>15,.0f}")
        else:
            print(f"  Success Rate: Parametric={result['dec_success_param']:.1%}")
            print(f"  {'Percentile':<15} {'Parametric':<20}")
            print("  " + "-" * 35)
            for p in [5, 25, 50, 75, 95]:
                v_param = result['dec_percentiles_param'][p]
                print(f"  {f'{p}th':<15} ${v_param:>15,.0f}")

        # Save results to CSV
        results_data = {
            'metric': ['acc_p5', 'acc_p25', 'acc_p50', 'acc_p75', 'acc_p95',
                       'dec_success', 'dec_p5', 'dec_p25', 'dec_p50', 'dec_p75', 'dec_p95'],
            'parametric': [
                result['acc_percentiles_param'][5],
                result['acc_percentiles_param'][25],
                result['acc_percentiles_param'][50],
                result['acc_percentiles_param'][75],
                result['acc_percentiles_param'][95],
                result['dec_success_param'],
                result['dec_percentiles_param'][5],
                result['dec_percentiles_param'][25],
                result['dec_percentiles_param'][50],
                result['dec_percentiles_param'][75],
                result['dec_percentiles_param'][95],
            ],
        }
        if has_bootstrap:
            results_data['bootstrap'] = [
                result['acc_percentiles_boot'][5],
                result['acc_percentiles_boot'][25],
                result['acc_percentiles_boot'][50],
                result['acc_percentiles_boot'][75],
                result['acc_percentiles_boot'][95],
                result['dec_success_boot'],
                result['dec_percentiles_boot'][5],
                result['dec_percentiles_boot'][25],
                result['dec_percentiles_boot'][50],
                result['dec_percentiles_boot'][75],
                result['dec_percentiles_boot'][95],
            ]
        results_df = pd.DataFrame(results_data)
        csv_path = output_dir / 'mc_results.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")

        # Create visualization
        if not args.no_plot:
            print("\n[3/3] Creating visualization...")
            plot_path = output_dir / 'mc_lifecycle.png'
            fig = create_single_run_visualization(
                result, config, str(plot_path)
            )
            plt.close(fig)

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
