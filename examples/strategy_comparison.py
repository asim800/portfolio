"""
Universal Portfolio + Safe Haven Comparison

Main entry point for running portfolio strategy comparisons.
This module orchestrates the loading of market data, running comparisons,
and generating visualizations.
"""

import os
import sys
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('error', category=RuntimeWarning, module='numpy.linalg')

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.universal import *
from src.strategies.registry import (
    compare_portfolios,
    create_portfolio_registry,
    get_default_colors,
    simulate_safe_haven_returns,
    simulate_extended_returns,
)
from src.visualization.comparison import (
    create_strategy_visualization,
    create_ten_more_visualization,
    create_comparison_visualization,
    create_asset_correlation_heatmap,
    create_portfolio_correlation_heatmap,
)
from src.data.market_data import fetch_yahoo_finance_data, load_returns_data


# Output folder for saved figures (use consolidated output directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTFOLDER = os.path.join(PROJECT_ROOT, 'output', 'plots')


if __name__ == "__main__":
    # Create output folder if it doesn't exist
    os.makedirs(OUTFOLDER, exist_ok=True)

    # Create all 15 portfolios using registry
    registry = create_portfolio_registry()

    # Create all 15 strategies at once
    all_strategies = {
        name: cls(**kwargs)
        for name, (cls, kwargs, _) in registry.items()
        if name.startswith(tuple(f'{i}.' for i in range(1, 16)))
    }

    n_period = 100

    # ============================================================================
    # Configuration: Market Data Source
    # ============================================================================
    DATA_MODE = 'yahoo'  # Options: 'yahoo' (Yahoo Finance), 'csv' (local file)
    DATA_FREQUENCY = 'M'  # Options: 'D' (daily), 'W' (weekly), 'M' (monthly)
    CACHE_FILE = 'market_data_cache'
    RETURNS_CSV = 'returns.csv'  # Used when DATA_MODE='csv'

    # Yahoo Finance ticker symbols (can be customized)
    TICKERS = {
        'bonds': 'TLT',              # 20+ Year Treasury Bond ETF
        'stocks': 'SPY',             # S&P 500 ETF
        'tail_hedge_vxx': 'VXX',     # VIX Short-Term Futures ETN
        'tail_hedge_vixy': 'VIXY',   # VIX Short-Term Futures ETF
        'tail_hedge_tail': 'TAIL',   # Cambria Tail Risk ETF
        'tail_hedge_managed_fut': 'DBMF',
        'commodities': 'DBC',        # Invesco DB Commodity Index Tracking Fund
        'gold': 'GLD'                # SPDR Gold Shares
    }

    # ============================================================================
    # Load Market Data
    # ============================================================================
    print("\n" + "="*80)
    print("LOADING MARKET DATA")
    print("="*80)

    try:
        if DATA_MODE == 'yahoo':
            print(f"\nFetching historical market data from Yahoo Finance...")
            print(f"Frequency: {DATA_FREQUENCY} ({'Monthly' if DATA_FREQUENCY == 'M' else 'Weekly' if DATA_FREQUENCY == 'W' else 'Daily'})")
            returns_data, price_df = load_returns_data(
                mode='yahoo',
                frequency=DATA_FREQUENCY,
                cache_file=CACHE_FILE,
                use_cache=True,
                tickers=TICKERS
            )
        else:
            print(f"\nLoading returns data from CSV file: {RETURNS_CSV}")
            returns_data, price_df = load_returns_data(mode='csv', csv_file=RETURNS_CSV)

        # Print data summary
        print(f"\n{'='*80}")
        print("DATA SUMMARY")
        print(f"{'='*80}")
        print(f"  Source: {DATA_MODE.upper()}")
        print(f"  Periods: {len(returns_data)}")
        print(f"  Assets: {', '.join(returns_data.columns)}")
        print(f"  Date range: {returns_data.index[0].date()} to {returns_data.index[-1].date()}")
        print(f"{'='*80}")

    except Exception as e:
        print(f"\nError loading data: {e}")
        print("Falling back to simulated data...")
        returns_data = None  # Will trigger simulation in compare_portfolios()

    # Run unified comparison for all 15 portfolios
    print("\n" + "="*80)
    print("UNIFIED PORTFOLIO COMPARISON - ALL 15 STRATEGIES")
    print("="*80)
    results_df, all_strats = compare_portfolios(
        strategies=all_strategies,
        returns_data=returns_data,
        n_periods=n_period,
        seed=42,
        n_display=15,  # Show all 15 portfolios
        rank_by='final_wealth',
        title="COMPARING ALL 15 UNIVERSAL PORTFOLIO + SAFE HAVEN APPROACHES",
        include_baselines=True
    )

    print("\nAll 15 portfolios compared!")
    print("\nThese approaches explore:")
    print("  - Multi-asset allocation (1, 8)")
    print("  - Ergodicity adjustments (2, 6, 9)")
    print("  - Safe haven strategies (3, 8, 11)")
    print("  - Hierarchical methods (4, 7, 13, 14, 15)")
    print("  - Drawdown control (5, 10)")
    print("  - Volatility management (12)")

    # Extract top 5 ranked strategies for visualization
    top_5_names = results_df.head(5)['Strategy'].tolist()
    top_5_strategies = {name: all_strats[name] for name in top_5_names
                        if name in all_strats and not name.startswith('[*]') and 'Standard UP' not in name}

    baseline_up = all_strats.get('Standard UP (2-asset)')
    buy_hold = all_strats.get('[*] Buy & Hold 60/40 (Benchmark)')

    # Generate returns for visualization
    viz_returns_data = simulate_safe_haven_returns(n_period, seed=42)

    print(f"\nCreating visualization for top 5 ranked strategies...")
    print(f"Top 5: {', '.join([name.split('.')[0] for name in top_5_names[:5]])}")
    fig = create_comparison_visualization(top_5_strategies, baseline_up.wealth_history, buy_hold, viz_returns_data)
    fig.savefig(os.path.join(OUTFOLDER, 'safe_haven_approaches_comparison.png'),
                dpi=300, bbox_inches='tight')
    print("Saved: safe_haven_approaches_comparison.png")

    # Extract top 10 from portfolios 6-15 for second visualization
    portfolios_6_15_names = [name for name in results_df['Strategy'].tolist()
                             if any(name.startswith(f'{i}.') for i in range(6, 16))]
    top_10_from_6_15 = portfolios_6_15_names[:10]
    strategies_6_15 = {name: all_strats[name] for name in top_10_from_6_15 if name in all_strats}

    # Generate returns for visualization
    returns_data_ten = simulate_extended_returns(n_period, seed=42)

    print(f"\nCreating visualization for top portfolios from 6-15...")
    print(f"Included: {', '.join([name.split('.')[0] for name in top_10_from_6_15])}")
    fig = create_ten_more_visualization(strategies_6_15, returns_data_ten)
    fig.savefig(os.path.join(OUTFOLDER, 'ten_more_approaches_comparison.png'),
                dpi=300, bbox_inches='tight')
    print("Saved: ten_more_approaches_comparison.png")

    # ============================================================================
    # Create Asset Correlation and Covariance Heatmaps
    # ============================================================================
    print("\n" + "="*80)
    print("ASSET CORRELATION AND COVARIANCE ANALYSIS")
    print("="*80)

    # Calculate and print correlation matrix
    asset_returns_df = returns_data.copy()
    asset_returns_df.columns = [col.replace('_', ' ').title() for col in asset_returns_df.columns]

    # Convert to percentage returns
    returns_pct = (asset_returns_df - 1.0) * 100

    # Calculate correlation and covariance
    asset_correlation = returns_pct.corr()
    asset_covariance = returns_pct.cov()

    print("\nAsset Correlation Matrix:")
    print("-" * 80)
    print(asset_correlation.to_string(float_format=lambda x: f'{x:7.3f}'))

    print("\n\nAsset Covariance Matrix (% squared units):")
    print("-" * 80)
    print(asset_covariance.to_string(float_format=lambda x: f'{x:7.3f}'))
    print("-" * 80)

    # Create visualization
    fig_asset_heatmap = create_asset_correlation_heatmap(
        returns_data,
        title="Asset Return Correlation and Covariance Analysis"
    )
    fig_asset_heatmap.savefig(os.path.join(OUTFOLDER, 'asset_correlation_covariance_heatmap.png'),
                              dpi=300, bbox_inches='tight')
    print("\nSaved: asset_correlation_covariance_heatmap.png")

    # ============================================================================
    # Create Portfolio Correlation and Covariance Heatmaps
    # ============================================================================
    print("\n" + "="*80)
    print("PORTFOLIO CORRELATION AND COVARIANCE ANALYSIS (TOP 10)")
    print("="*80)

    # Use top 10 portfolios for cleaner visualization
    top_10_names = results_df.head(10)['Strategy'].tolist()
    top_10_strategies = {name: all_strats[name] for name in top_10_names
                        if name in all_strats and not name.startswith('[*]')}

    # Add baselines for comparison
    if baseline_up and hasattr(baseline_up, 'wealth_history'):
        top_10_strategies['Standard UP (2-asset)'] = baseline_up
    if buy_hold and hasattr(buy_hold, 'wealth_history'):
        top_10_strategies['[*] Buy & Hold 60/40 (Benchmark)'] = buy_hold

    # Calculate portfolio returns for correlation/covariance
    portfolio_returns = {}
    for name, strategy in top_10_strategies.items():
        if hasattr(strategy, 'wealth_history') and len(strategy.wealth_history) > 1:
            wealth = np.array(strategy.wealth_history)
            returns = wealth[1:] / wealth[:-1]
            returns_pct = (returns - 1.0) * 100
            # Clean up name for display
            display_name = name.split('. ', 1)[1] if '. ' in name else name
            # Truncate if too long
            display_name = display_name[:25] + '...' if len(display_name) > 25 else display_name
            portfolio_returns[display_name] = returns_pct

    portfolio_returns_df = pd.DataFrame(portfolio_returns)
    portfolio_correlation = portfolio_returns_df.corr()
    portfolio_covariance = portfolio_returns_df.cov()

    print("\nPortfolio Correlation Matrix:")
    print("-" * 80)
    print(portfolio_correlation.to_string(float_format=lambda x: f'{x:7.3f}'))

    print("\n\nPortfolio Covariance Matrix (% squared units):")
    print("-" * 80)
    print(portfolio_covariance.to_string(float_format=lambda x: f'{x:7.3f}'))
    print("-" * 80)

    # Create visualization
    fig_portfolio_heatmap = create_portfolio_correlation_heatmap(
        top_10_strategies,
        title="Portfolio Return Correlation and Covariance Analysis (Top 10)"
    )
    fig_portfolio_heatmap.savefig(os.path.join(OUTFOLDER, 'portfolio_correlation_covariance_heatmap.png'),
                                   dpi=300, bbox_inches='tight')
    print("\nSaved: portfolio_correlation_covariance_heatmap.png")

    print("\n" + "="*80)
    print("All comparisons complete!")
    print("="*80)
