#!/usr/bin/env python3
"""
Strategy Registry Module.

Provides a registry pattern for dynamically instantiating portfolio strategies.
Extracted from ufsrc/src/metrics.py.
"""

from typing import Dict, Tuple, Type, Any
import ipdb

from .universal import (
    ThreeAssetUniversalPortfolio,
    ErgodicUniversalPortfolio,
    ConditionalSafeHavenPortfolio,
    UniversalPortfolioBasic,
    BuyAndHold,
    HierarchicalUniversalPortfolio,
    DrawdownAwareUniversalPortfolio,
    TimeVaryingErgodicityUP,
    MultiTimeframeHierarchical,
    MultiSafeHavenUP,
    KellyUniversalPortfolio,
    AsymmetricLossAverseUP,
    SequentialThresholdUP,
    VolatilityScaledUP,
    MomentumEnhancedHierarchical,
    ThreeLevelHierarchical,
    DynamicGranularityUP,
)


def create_portfolio_registry() -> Dict[str, Tuple[Type, Dict[str, Any], int]]:
    """
    Create a registry mapping portfolio names to (class, init_params, required_assets).

    Returns:
        Dict with structure: {
            'portfolio_name': (StrategyClass, init_kwargs, n_required_assets)
        }

    Example:
        >>> registry = create_portfolio_registry()
        >>> name = '1. Three-Asset UP'
        >>> cls, params, n_assets = registry[name]
        >>> strategy = cls(**params)
    """
    registry = {
        # Portfolios 1-5 (various asset requirements)
        '1. Three-Asset UP': (ThreeAssetUniversalPortfolio, {'n_portfolios': 21}, 3),
        '2. Ergodicity-Adjusted UP': (ErgodicUniversalPortfolio, {'n_portfolios': 101, 'ergodicity_factor': 0.5}, 2),
        '3. Conditional Safe Haven': (ConditionalSafeHavenPortfolio, {'n_portfolios': 101}, 2),
        '4. Hierarchical UP': (HierarchicalUniversalPortfolio, {'n_portfolios_risky': 51, 'n_portfolios_hedge': 21}, 3),
        '5. Drawdown-Aware UP': (DrawdownAwareUniversalPortfolio, {'n_portfolios': 101, 'drawdown_penalty': 1.0}, 2),

        # Portfolios 6-10
        '6. Time-Varying Ergodicity': (TimeVaryingErgodicityUP, {'n_portfolios': 101, 'base_ergodicity': 0.5}, 2),
        '7. Multi-Timeframe Hierarchical': (MultiTimeframeHierarchical, {'n_portfolios_risky': 51, 'n_portfolios_hedge': 21, 'hedge_rebalance_freq': 5}, 3),
        '8. Multi-Safe-Haven UP': (MultiSafeHavenUP, {'n_portfolios': 15}, 4),
        '9. Kelly-Criterion UP': (KellyUniversalPortfolio, {'n_portfolios': 101, 'kelly_fraction': 0.5}, 2),
        '10. Asymmetric Loss-Averse': (AsymmetricLossAverseUP, {'n_portfolios': 101, 'loss_penalty': 1.5}, 2),

        # Portfolios 11-15
        '11. Sequential Threshold UP': (SequentialThresholdUP, {'n_portfolios': 101, 'wealth_threshold': 2.0, 'hedge_pct': 0.15}, 3),
        '12. Volatility-Scaled UP': (VolatilityScaledUP, {'n_portfolios': 101, 'target_vol': 0.12, 'vol_window': 20}, 2),
        '13. Momentum-Enhanced Hierarchical': (MomentumEnhancedHierarchical, {'n_portfolios_risky': 51, 'n_portfolios_hedge': 21, 'momentum_window': 10}, 3),
        '14. Three-Level Hierarchical': (ThreeLevelHierarchical, {'n_level1': 21, 'n_level2': 11, 'n_level3': 11}, 5),
        '15. Dynamic Granularity UP': (DynamicGranularityUP, {'min_portfolios': 21, 'max_portfolios': 101, 'vol_threshold': 0.15}, 2),

        # Baselines
        '[*] Buy & Hold 60/40 (Benchmark)': (BuyAndHold, {'initial_stock_pct': 0.6}, 2),
        'Standard UP (2-asset)': (UniversalPortfolioBasic, {'n_portfolios': 101}, 2),
    }

    return registry


def get_strategy_class(strategy_name: str) -> Type:
    """
    Get the strategy class by name.

    Parameters:
    -----------
    strategy_name : str
        Name of the strategy from the registry

    Returns:
    --------
    Type: The strategy class
    """
    registry = create_portfolio_registry()
    if strategy_name not in registry:
        available = list(registry.keys())
        raise ValueError(f"Unknown strategy: '{strategy_name}'. Available: {available}")
    return registry[strategy_name][0]


def get_strategy_params(strategy_name: str) -> Dict[str, Any]:
    """
    Get default initialization parameters for a strategy.

    Parameters:
    -----------
    strategy_name : str
        Name of the strategy from the registry

    Returns:
    --------
    Dict: Default initialization parameters
    """
    registry = create_portfolio_registry()
    if strategy_name not in registry:
        raise ValueError(f"Unknown strategy: '{strategy_name}'")
    return registry[strategy_name][1]


def get_required_assets(strategy_name: str) -> int:
    """
    Get the number of required assets for a strategy.

    Parameters:
    -----------
    strategy_name : str
        Name of the strategy from the registry

    Returns:
    --------
    int: Number of required assets (2, 3, 4, or 5)
    """
    registry = create_portfolio_registry()
    if strategy_name not in registry:
        raise ValueError(f"Unknown strategy: '{strategy_name}'")
    return registry[strategy_name][2]


def instantiate_strategy(strategy_name: str, **override_params):
    """
    Instantiate a strategy by name with optional parameter overrides.

    Parameters:
    -----------
    strategy_name : str
        Name of the strategy from the registry
    **override_params : dict
        Parameters to override defaults

    Returns:
    --------
    Strategy instance

    Example:
        >>> strategy = instantiate_strategy('1. Three-Asset UP', n_portfolios=31)
    """
    registry = create_portfolio_registry()
    if strategy_name not in registry:
        raise ValueError(f"Unknown strategy: '{strategy_name}'")

    cls, default_params, _ = registry[strategy_name]
    params = {**default_params, **override_params}
    return cls(**params)


def list_strategies(include_baselines: bool = False) -> list:
    """
    List all available strategy names.

    Parameters:
    -----------
    include_baselines : bool
        Whether to include baseline strategies

    Returns:
    --------
    list: List of strategy names
    """
    registry = create_portfolio_registry()
    if include_baselines:
        return list(registry.keys())
    else:
        return [k for k in registry.keys()
                if not k.startswith('[*]') and k != 'Standard UP (2-asset)']


# ============================================================================
# Simulation Functions
# ============================================================================

def simulate_safe_haven_returns(n_periods=100, seed=42):
    """
    Simulate returns for stocks, bonds, and tail hedge.

    Tail hedge characteristics (inspired by put options / VIX):
    - Loses 5-10% in normal times (decay)
    - Gains 200-500% during crashes
    - Negative correlation with stocks during drawdowns

    Args:
        n_periods: Number of periods to simulate
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Shape (n_periods, 3) with columns [bonds, stocks, tail_hedge]
                   Values are price multipliers (e.g., 1.02 = +2%)
    """
    import numpy as np
    np.random.seed(seed)

    returns = np.zeros((n_periods, 3))

    # Bonds: steady 2% with low volatility
    returns[:, 0] = 1.02

    # Stocks: mean 5%, vol 15%, with occasional crashes
    for t in range(n_periods):
        if np.random.random() < 0.08:  # 8% chance of crash
            returns[t, 1] = np.random.uniform(0.7, 0.9)  # -10% to -30%
        else:
            returns[t, 1] = np.random.lognormal(np.log(1.05), 0.15)

    # Tail hedge:
    # - Decays 8% in normal times
    # - Explodes during crashes (negative correlation with stocks)
    for t in range(n_periods):
        if returns[t, 1] < 0.95:  # Stock down
            # Hedge pays off inversely proportional to stock loss
            stock_loss = 1 - returns[t, 1]
            hedge_gain = 1 + (stock_loss * 8)  # 8x leverage on losses
            returns[t, 2] = min(hedge_gain, 6.0)  # Cap at 500% gain
        else:
            # Normal decay
            returns[t, 2] = np.random.uniform(0.90, 0.95)  # -5% to -10%

    return returns


def simulate_extended_returns(n_periods=100, seed=42):
    """
    Simulate returns including commodities and gold.

    Args:
        n_periods: Number of periods to simulate
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Shape (n_periods, 5) with columns
                   [bonds, stocks, tail_hedge, commodities, gold]
                   Values are price multipliers (e.g., 1.02 = +2%)
    """
    import numpy as np
    np.random.seed(seed)
    returns = np.zeros((n_periods, 5))

    for t in range(n_periods):
        # Bonds: steady 2%
        returns[t, 0] = 1.02

        # Stocks: crashes or normal
        if np.random.random() < 0.08:
            returns[t, 1] = np.random.uniform(0.7, 0.9)
        else:
            returns[t, 1] = np.random.lognormal(np.log(1.05), 0.15)

        # Tail hedge: inverse to stocks
        if returns[t, 1] < 0.95:
            stock_loss = 1 - returns[t, 1]
            returns[t, 2] = min(1 + (stock_loss * 8), 6.0)
        else:
            returns[t, 2] = np.random.uniform(0.90, 0.95)

        # Commodities: moderate volatility
        returns[t, 3] = np.random.lognormal(np.log(1.03), 0.18)

        # Gold: safe haven, slight negative correlation with stocks
        if returns[t, 1] < 0.95:
            returns[t, 4] = np.random.uniform(1.02, 1.08)
        else:
            returns[t, 4] = np.random.uniform(0.98, 1.03)

    return returns


# ============================================================================
# Performance Metrics
# ============================================================================

def calculate_max_drawdown(wealth_history):
    """Calculate maximum drawdown percentage."""
    import numpy as np
    wealth = np.array(wealth_history)
    running_max = np.maximum.accumulate(wealth)
    drawdown = (running_max - wealth) / running_max
    return np.max(drawdown) * 100


def calculate_sharpe(wealth_history, periods_per_year=None, risk_free_rate=0.0):
    """Calculate Sharpe ratio."""
    import numpy as np
    if len(wealth_history) < 2:
        return 0

    # Calculate log returns
    returns = np.diff(np.log(wealth_history))

    if np.std(returns, ddof=1) == 0:
        return 0

    # Per-period risk-free rate
    rf_per_period = risk_free_rate / (periods_per_year if periods_per_year else 1)

    # Calculate per-period Sharpe ratio
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    sharpe = (mean_return - rf_per_period) / std_return

    # Annualize if requested
    if periods_per_year is not None:
        sharpe = sharpe * np.sqrt(periods_per_year)

    return sharpe


def calculate_portfolio_metrics(wealth_history):
    """Calculate comprehensive metrics for a portfolio."""
    import numpy as np
    if len(wealth_history) < 2:
        return {
            'final_wealth': wealth_history[0] if len(wealth_history) > 0 else 0,
            'mean_return': 0,
            'std_return': 0,
            'sharpe': 0,
            'max_dd': 0
        }

    returns = np.diff(np.log(wealth_history))
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    sharpe = calculate_sharpe(wealth_history)
    max_dd = calculate_max_drawdown(wealth_history)
    final_wealth = wealth_history[-1]

    return {
        'final_wealth': final_wealth,
        'mean_return': mean_return,
        'std_return': std_return,
        'sharpe': sharpe,
        'max_dd': max_dd
    }


# ============================================================================
# Portfolio Comparison
# ============================================================================

def get_default_colors():
    """Returns complete color mapping for all 15 portfolios + baselines."""
    colors = {
        '1. Three-Asset UP': 'purple',
        '2. Ergodicity-Adjusted UP': 'blue',
        '3. Conditional Safe Haven': 'green',
        '4. Hierarchical UP': 'red',
        '5. Drawdown-Aware UP': 'orange',
        '6. Time-Varying Ergodicity': 'purple',
        '7. Multi-Timeframe Hierarchical': 'red',
        '8. Multi-Safe-Haven UP': 'green',
        '9. Kelly-Criterion UP': 'blue',
        '10. Asymmetric Loss-Averse': 'orange',
        '11. Sequential Threshold UP': 'brown',
        '12. Volatility-Scaled UP': 'pink',
        '13. Momentum-Enhanced Hierarchical': 'darkgreen',
        '14. Three-Level Hierarchical': 'gold',
        '15. Dynamic Granularity UP': 'cyan',
        '[*] Buy & Hold 60/40 (Benchmark)': 'black',
        'Standard UP (2-asset)': 'gray',
    }
    return colors


def determine_required_assets(strategy_name):
    """Determine number of assets required for a strategy based on its name."""
    registry = create_portfolio_registry()
    if strategy_name in registry:
        return registry[strategy_name][2]

    multi_asset_map = {
        'Three-Asset UP': 3,
        'Hierarchical UP': 3,
        'Multi-Timeframe Hierarchical': 3,
        'Multi-Safe-Haven UP': 4,
        'Momentum-Enhanced': 3,
        'Three-Level Hierarchical': 5,
        'Sequential Threshold': 3,
    }
    base_name = strategy_name.split('. ', 1)[-1] if '. ' in strategy_name else strategy_name
    return multi_asset_map.get(base_name, 2)


def _get_strategy_required_assets(strategy):
    """
    Get the required assets for a strategy from its class attribute.

    Falls back to ['stocks', 'bonds'] if not defined.
    """
    if hasattr(strategy.__class__, 'REQUIRED_ASSETS'):
        return strategy.__class__.REQUIRED_ASSETS
    return ['stocks', 'bonds']


def _filter_data_for_strategy(price_multipliers_df, required_assets, strategy_name):
    """
    Filter DataFrame to drop rows with NaN in required asset columns.

    Returns filtered DataFrame and prints diagnostic if rows were dropped.
    """
    import numpy as np

    original_len = len(price_multipliers_df)

    # Check which required assets exist in the data
    available_assets = [a for a in required_assets if a in price_multipliers_df.columns]
    missing_assets = [a for a in required_assets if a not in price_multipliers_df.columns]

    if missing_assets:
        raise KeyError(
            f"Strategy '{strategy_name}' requires assets {missing_assets} "
            f"but they are not in the data. Available: {list(price_multipliers_df.columns)}"
        )

    # Drop rows where any required asset has NaN
    mask = price_multipliers_df[available_assets].notna().all(axis=1)
    filtered_df = price_multipliers_df[mask].copy()

    dropped_count = original_len - len(filtered_df)
    if dropped_count > 0:
        # Find which assets had NaN values
        nan_counts = {}
        for asset in available_assets:
            nan_count = price_multipliers_df[asset].isna().sum()
            if nan_count > 0:
                nan_counts[asset] = nan_count

        print(f"  [{strategy_name}] Dropped {dropped_count}/{original_len} rows with NaN in: {nan_counts}")

    return filtered_df


def compare_portfolios(
    strategies,
    returns_data=None,
    n_periods=100,
    seed=42,
    n_display=5,
    rank_by='final_wealth',
    include_baselines=True,
    title="Portfolio Comparison"
):
    """
    Unified portfolio comparison function.

    Each strategy is simulated with data filtered to remove NaN values only in
    columns that the strategy actually uses. This allows strategies with fewer
    asset requirements to use more of the available data.

    Args:
        strategies: Dict mapping name -> strategy instance
        returns_data: Optional pandas DataFrame with returns
        n_periods: Number of periods to simulate (only if returns_data is None)
        seed: Random seed for simulation
        n_display: Number of top portfolios to display
        rank_by: Metric to rank by ('final_wealth', 'mean_return', 'std_return', 'sharpe', 'max_dd')
        include_baselines: Whether to include Buy & Hold and Standard UP baselines
        title: Title for output table

    Returns:
        tuple: (results_df, all_strategies_dict)
    """
    import pandas as pd
    import numpy as np

    # Handle returns_data
    if returns_data is None:
        max_assets = max([determine_required_assets(name) for name in strategies.keys()])
        if max_assets <= 3:
            returns_array = simulate_safe_haven_returns(n_periods, seed)
            asset_names = ['bonds', 'stocks', 'tail_hedge']
        else:
            returns_array = simulate_extended_returns(n_periods, seed)
            asset_names = ['bonds', 'stocks', 'tail_hedge', 'commodities', 'gold']
        returns_data = pd.DataFrame(returns_array, columns=asset_names)
        price_multipliers_full = returns_data
    else:
        if not isinstance(returns_data, pd.DataFrame):
            raise ValueError("returns_data must be a pandas DataFrame")

        asset_names = list(returns_data.columns)
        price_multipliers_full = returns_data + 1.0

    # Create baselines if requested
    baselines = {}
    if include_baselines:
        buy_hold = BuyAndHold(initial_stock_pct=0.6)
        standard_up = UniversalPortfolioBasic(n_portfolios=101)
        baselines = {
            '[*] Buy & Hold 60/40 (Benchmark)': buy_hold,
            'Standard UP (2-asset)': standard_up
        }

    # Combine all strategies for unified processing
    all_strategies = {**baselines, **strategies}

    # Check for NaN in data and print summary
    nan_summary = price_multipliers_full.isna().sum()
    if nan_summary.any():
        print("\nNaN values detected in input data:")
        for col, count in nan_summary.items():
            if count > 0:
                print(f"  {col}: {count} NaN values")
        print("\nFiltering data per-strategy based on required assets...")

    # Run simulation for each strategy with its own filtered data
    for name, strategy in all_strategies.items():
        required_assets = _get_strategy_required_assets(strategy)

        # Filter data for this strategy (drops rows with NaN in required columns only)
        try:
            filtered_price_multipliers = _filter_data_for_strategy(
                price_multipliers_full, required_assets, name
            )
        except KeyError as e:
            print(f"  [{name}] Skipped: {e}")
            continue

        # Run simulation for this strategy
        for t in range(len(filtered_price_multipliers)):
            price_mult_dict = filtered_price_multipliers.iloc[t].to_dict()

            try:
                strategy.update(price_mult_dict)
            except KeyError as e:
                raise KeyError(
                    f"Strategy '{name}' requires asset {e} but available assets are: {list(price_mult_dict.keys())}. "
                    f"Strategy may require more assets than provided in the data."
                ) from e

    # Collect metrics
    results = []
    for name, strategy in all_strategies.items():
        if hasattr(strategy, 'wealth_history') and strategy.wealth_history:
            metrics = calculate_portfolio_metrics(strategy.wealth_history)
            metrics['strategy'] = name
            results.append(metrics)

    # Sort by specified metric
    reverse_sort = (rank_by != 'max_dd')
    results.sort(key=lambda x: x[rank_by], reverse=reverse_sort)
    results = results[:n_display]

    # Create DataFrame
    df = pd.DataFrame(results)
    df = df[['strategy', 'final_wealth', 'mean_return', 'std_return', 'sharpe', 'max_dd']]
    df.columns = ['Strategy', 'Final Wealth', 'Mean Return', 'Std Dev', 'Sharpe', 'Max DD']

    # Print results
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(df.to_string(index=False))
    print("-" * 80)

    return df, all_strategies
