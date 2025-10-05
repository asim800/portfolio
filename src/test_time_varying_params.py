#!/usr/bin/env python3
"""
Test and demonstrate time-varying parameter support in MCPathGenerator.

This shows how to use time-varying mean returns and covariance matrices
for regime-switching, adaptive estimation, or evolving market conditions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mc_path_generator import MCPathGenerator

def test_time_varying_mean_only():
    """Test 1: Time-varying mean returns with constant covariance"""
    print("="*80)
    print("TEST 1: TIME-VARYING MEAN RETURNS (Regime Switching)")
    print("="*80)

    # Setup
    tickers = ['SPY', 'AGG']
    # Initial constant params (not used for time-varying)
    mean_returns = np.array([0.10, 0.04])
    cov_matrix = np.array([[0.04, 0.01], [0.01, 0.02]])

    generator = MCPathGenerator(tickers, mean_returns, cov_matrix, seed=42)

    # Create time-varying mean returns (regime switch at period 500)
    dates = pd.date_range('2025-01-01', periods=1000, freq='D')

    # Bull market (first 500 days) → Bear market (last 500 days)
    mean_ts = pd.DataFrame({
        'SPY': [0.15 if i < 500 else 0.02 for i in range(1000)],  # 15% → 2%
        'AGG': [0.04 if i < 500 else 0.06 for i in range(1000)]   # 4% → 6% (flight to safety)
    }, index=dates)

    print(f"\nMean returns time series:")
    print(f"  Shape: {mean_ts.shape}")
    print(f"  Bull market (days 0-499):")
    print(f"    SPY: {mean_ts.iloc[0]['SPY']:.2%}, AGG: {mean_ts.iloc[0]['AGG']:.2%}")
    print(f"  Bear market (days 500-999):")
    print(f"    SPY: {mean_ts.iloc[500]['SPY']:.2%}, AGG: {mean_ts.iloc[500]['AGG']:.2%}")

    # Set time-varying parameters
    generator.set_time_varying_parameters(mean_ts)

    # Generate paths
    paths = generator.generate_paths_time_varying(
        num_simulations=1000,
        start_date='2025-01-01',
        total_periods=1000,
        periods_per_year=252,  # Daily
        frequency='D'
    )

    print(f"\nGenerated paths shape: {paths.shape}")

    # Analyze regime impact
    bull_returns = paths[:, :500, 0].mean(axis=1)  # SPY in bull market
    bear_returns = paths[:, 500:, 0].mean(axis=1)  # SPY in bear market

    print(f"\nRegime impact on SPY:")
    print(f"  Bull market avg return: {bull_returns.mean():.6f} (expected: {0.15/252:.6f})")
    print(f"  Bear market avg return: {bear_returns.mean():.6f} (expected: {0.02/252:.6f})")

    # Verify the regime shift worked
    assert bull_returns.mean() > bear_returns.mean(), "Bull returns should be higher than bear"
    print(f"  ✓ Regime shift verified!")

    return paths, mean_ts


def test_time_varying_mean_and_cov():
    """Test 2: Both mean returns AND covariance varying"""
    print("\n" + "="*80)
    print("TEST 2: TIME-VARYING MEAN + COVARIANCE (Volatility Regimes)")
    print("="*80)

    # Setup
    tickers = ['SPY', 'AGG']
    mean_returns = np.array([0.10, 0.04])
    cov_matrix = np.array([[0.04, 0.01], [0.01, 0.02]])

    generator = MCPathGenerator(tickers, mean_returns, cov_matrix, seed=42)

    # Create time-varying parameters
    dates = pd.date_range('2025-01-01', periods=500, freq='D')

    # Mean returns (slight variation)
    mean_ts = pd.DataFrame({
        'SPY': [0.10] * 500,
        'AGG': [0.04] * 500
    }, index=dates)

    # Covariance: Low vol (first 250 days) → High vol (last 250 days)
    cov_ts_data = []
    for i in range(500):
        if i < 250:
            # Low volatility regime
            spy_var = 0.02  # Low vol
            agg_var = 0.01
            correlation = 0.3
        else:
            # High volatility regime
            spy_var = 0.06  # High vol
            agg_var = 0.02
            correlation = 0.6  # Higher correlation in crisis

        spy_std = np.sqrt(spy_var)
        agg_std = np.sqrt(agg_var)
        spy_agg_cov = correlation * spy_std * agg_std

        cov_ts_data.append({
            'SPY_SPY': spy_var,
            'SPY_AGG': spy_agg_cov,
            'AGG_SPY': spy_agg_cov,
            'AGG_AGG': agg_var
        })

    cov_ts = pd.DataFrame(cov_ts_data, index=dates)

    print(f"\nCovariance time series:")
    print(f"  Shape: {cov_ts.shape}")
    print(f"  Low vol regime (days 0-249):")
    print(f"    SPY variance: {cov_ts.iloc[0]['SPY_SPY']:.4f}")
    print(f"  High vol regime (days 250-499):")
    print(f"    SPY variance: {cov_ts.iloc[250]['SPY_SPY']:.4f}")

    # Set parameters
    generator.set_time_varying_parameters(mean_ts, cov_ts)

    # Generate paths
    paths = generator.generate_paths_time_varying(
        num_simulations=1000,
        start_date='2025-01-01',
        total_periods=500,
        periods_per_year=252,
        frequency='D'
    )

    # Analyze volatility regimes
    low_vol_std = np.std(paths[:, :250, 0], axis=1).mean()  # SPY in low vol
    high_vol_std = np.std(paths[:, 250:, 0], axis=1).mean()  # SPY in high vol

    print(f"\nVolatility regime impact:")
    print(f"  Low vol period std: {low_vol_std:.6f}")
    print(f"  High vol period std: {high_vol_std:.6f}")
    print(f"  Ratio: {high_vol_std/low_vol_std:.2f}x")

    assert high_vol_std > low_vol_std, "High vol should be higher than low vol"
    print(f"  ✓ Volatility shift verified!")

    return paths, mean_ts, cov_ts


def test_expanding_window_estimation():
    """Test 3: Adaptive/expanding window estimation"""
    print("\n" + "="*80)
    print("TEST 3: EXPANDING WINDOW ESTIMATION (Adaptive Parameters)")
    print("="*80)

    # Simulate expanding window mean/cov estimation
    tickers = ['SPY', 'AGG']
    mean_returns = np.array([0.10, 0.04])
    cov_matrix = np.array([[0.04, 0.01], [0.01, 0.02]])

    generator = MCPathGenerator(tickers, mean_returns, cov_matrix, seed=42)

    # Create fake historical returns for expanding window
    np.random.seed(42)
    n_historical = 1000
    historical_returns = pd.DataFrame({
        'SPY': np.random.normal(0.10/252, 0.02, n_historical),
        'AGG': np.random.normal(0.04/252, 0.01, n_historical)
    }, index=pd.date_range('2020-01-01', periods=n_historical, freq='D'))

    # Calculate expanding window estimates
    expanding_means = []
    dates_list = []
    min_window = 30  # Minimum 30 days

    for i in range(min_window, len(historical_returns), 10):  # Every 10 days
        window_data = historical_returns.iloc[:i]
        mean_est = window_data.mean() * 252  # Annualize
        expanding_means.append(mean_est)
        dates_list.append(historical_returns.index[i])

    mean_ts = pd.DataFrame(expanding_means, index=dates_list)
    mean_ts.columns = tickers

    print(f"\nExpanding window estimation:")
    print(f"  Historical data: {len(historical_returns)} days")
    print(f"  Estimates: {len(mean_ts)} points")
    print(f"  First estimate (30-day window):")
    print(f"    SPY: {mean_ts.iloc[0]['SPY']:.2%}, AGG: {mean_ts.iloc[0]['AGG']:.2%}")
    print(f"  Final estimate (full window):")
    print(f"    SPY: {mean_ts.iloc[-1]['SPY']:.2%}, AGG: {mean_ts.iloc[-1]['AGG']:.2%}")

    # Set and generate
    generator.set_time_varying_parameters(mean_ts)

    paths = generator.generate_paths_time_varying(
        num_simulations=100,
        start_date=dates_list[0].strftime('%Y-%m-%d'),
        total_periods=len(mean_ts),
        periods_per_year=252,
        frequency='D'
    )

    print(f"\nGenerated paths with expanding window:")
    print(f"  Shape: {paths.shape}")
    print(f"  ✓ Successfully used adaptive parameter estimation!")

    return paths, mean_ts


def visualize_results(paths1, mean_ts1, paths2=None):
    """Create visualization of time-varying parameter effects"""
    print("\n" + "="*80)
    print("Creating visualization...")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Sample paths with regime change
    ax = axes[0, 0]
    for i in range(10):
        cumulative = np.cumprod(1 + paths1[i, :, 0])
        ax.plot(cumulative, alpha=0.5, linewidth=0.8)
    ax.axvline(500, color='red', linestyle='--', label='Regime Change', linewidth=2)
    ax.set_title('Sample SPY Paths (Bull → Bear Regime)')
    ax.set_xlabel('Day')
    ax.set_ylabel('Cumulative Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Mean returns over time
    ax = axes[0, 1]
    ax.plot(mean_ts1.index, mean_ts1['SPY'], label='SPY', linewidth=2)
    ax.plot(mean_ts1.index, mean_ts1['AGG'], label='AGG', linewidth=2)
    ax.set_title('Time-Varying Mean Returns (Annualized)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean Return')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Distribution comparison (bull vs bear)
    ax = axes[1, 0]
    bull_final = np.mean(paths1[:, :500, 0], axis=1)
    bear_final = np.mean(paths1[:, 500:, 0], axis=1)
    ax.hist(bull_final, bins=30, alpha=0.5, label='Bull Period', density=True)
    ax.hist(bear_final, bins=30, alpha=0.5, label='Bear Period', density=True)
    ax.set_title('Return Distribution by Regime')
    ax.set_xlabel('Mean Period Return')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Volatility over time (rolling)
    ax = axes[1, 1]
    rolling_vol = []
    window = 50
    for i in range(window, paths1.shape[1]):
        vol = np.std(paths1[:, i-window:i, 0])
        rolling_vol.append(vol)
    ax.plot(range(window, paths1.shape[1]), rolling_vol, linewidth=2)
    if paths1.shape[1] > 500:
        ax.axvline(500, color='red', linestyle='--', label='Regime Change', linewidth=2)
    ax.set_title(f'Rolling Volatility ({window}-day window)')
    ax.set_xlabel('Day')
    ax.set_ylabel('Volatility')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../plots/test/time_varying_params_demo.png', dpi=150)
    print(f"  Saved: ../plots/test/time_varying_params_demo.png")

    return fig


def main():
    print("\n" + "="*80)
    print("TIME-VARYING PARAMETER MONTE CARLO - COMPREHENSIVE TEST")
    print("="*80)

    # Test 1: Regime switching (mean only)
    paths1, mean_ts1 = test_time_varying_mean_only()

    # Test 2: Volatility regimes (mean + cov)
    paths2, mean_ts2, cov_ts2 = test_time_varying_mean_and_cov()

    # Test 3: Adaptive estimation
    paths3, mean_ts3 = test_expanding_window_estimation()

    # Visualize
    visualize_results(paths1, mean_ts1, paths2)

    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)
    print("\nKey features demonstrated:")
    print("  ✓ Time-varying mean returns (regime switching)")
    print("  ✓ Time-varying covariance (volatility regimes)")
    print("  ✓ Expanding window estimation (adaptive parameters)")
    print("  ✓ Automatic date matching and parameter lookup")
    print("  ✓ Proper frequency scaling (annual → period)")
    print("="*80)


if __name__ == '__main__':
    main()
