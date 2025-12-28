#!/usr/bin/env python3
"""
Simple Example: Run Portfolio with MC Simulated Data (NEW ARCHITECTURE)

This script demonstrates the new architecture with:
- AllocationStrategy (WHAT weights to use)
- RebalancingTrigger (WHEN to rebalance)
- MCPathGenerator integration

Usage:
    uv run python example_simple_mc_portfolio.py
"""

import sys
import os
import numpy as np
import pandas as pd
import logging

# Add src3 to path
sys.path.insert(0, os.path.dirname(__file__))

from mc_path_generator import MCPathGenerator
from portfolio import Portfolio
from allocation_strategies import StaticAllocation, EqualWeight, OptimizedAllocation
from rebalancing_triggers import Never, Periodic, Threshold
from portfolio_optimizer import PortfolioOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

def main():
    print("\n" + "="*80)
    print("SIMPLE EXAMPLE: Portfolio with MC Simulated Data (NEW ARCHITECTURE)")
    print("="*80)

    # ========================================================================
    # Step 1: Define portfolio assets
    # ========================================================================
    print("\n1. Defining portfolio assets...")

    tickers = ['SPY', 'AGG', 'NVDA', 'GLD']
    weights = [0.4, 0.3, 0.2, 0.1]  # 40% SPY, 30% AGG, 20% NVDA, 10% GLD

    print(f"   ✓ Tickers: {tickers}")
    print(f"   ✓ Weights: {weights}")

    # ========================================================================
    # Step 2: Define MC simulation parameters
    # ========================================================================
    print("\n2. Setting up MC parameters...")

    # Expected annual returns (based on historical averages or your assumptions)
    mean_returns = np.array([
        0.10,   # SPY: 10% expected annual return
        0.04,   # AGG: 4% expected annual return
        0.20,   # NVDA: 20% expected annual return (higher risk/reward)
        0.06    # GLD: 6% expected annual return
    ])

    # Annual covariance matrix (volatility and correlations)
    # Diagonal = variance (volatility^2), off-diagonal = covariance
    cov_matrix = np.array([
        [0.04,   0.01,   0.02,   0.005],  # SPY: 20% volatility
        [0.01,   0.02,   0.005,  0.01],   # AGG: 14% volatility
        [0.02,   0.005,  0.08,   0.01],   # NVDA: 28% volatility
        [0.005,  0.01,   0.01,   0.03]    # GLD: 17% volatility
    ])

    print(f"   ✓ Mean returns: {mean_returns}")
    print(f"   ✓ Volatilities: {np.sqrt(np.diag(cov_matrix))}")

    # ========================================================================
    # Step 3: Generate MC paths
    # ========================================================================
    print("\n3. Generating Monte Carlo paths...")

    generator = MCPathGenerator(
        tickers=tickers,
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        seed=42  # For reproducibility
    )

    # Generate 1 simulation, 10 years of biweekly data
    num_simulations = 1
    total_periods = 260  # 10 years × 26 biweekly periods
    periods_per_year = 26

    paths = generator.generate_paths(
        num_simulations=num_simulations,
        total_periods=total_periods,
        periods_per_year=periods_per_year
    )

    print(f"   ✓ Generated {num_simulations} simulations")
    print(f"   ✓ Shape: {paths.shape}")
    print(f"   ✓ Coverage: 10 years of biweekly returns")

    # Convert to DataFrame for Portfolio ingestion
    mc_returns_df = generator.get_path_dataframe(
        simulation_idx=0,
        start_date='2025-01-01',
        frequency='2W'  # Biweekly
    )

    print(f"   ✓ Returns DataFrame shape: {mc_returns_df.shape}")
    print(f"   ✓ Date range: {mc_returns_df.index[0]} to {mc_returns_df.index[-1]}")

    # ========================================================================
    # Step 4: Create portfolios with NEW architecture
    # ========================================================================
    print("\n4. Creating portfolios with NEW architecture...")

    # Create initial weights as pd.Series
    initial_weights = pd.Series(weights, index=tickers)

    # Portfolio 1: Buy & Hold (40/30/20/10)
    print("\n   Creating Portfolio 1: Buy & Hold (40/30/20/10)")
    portfolio1 = Portfolio(
        asset_names=tickers,
        initial_weights=initial_weights,
        allocation_strategy=StaticAllocation(weights, name="40/30/20/10"),
        rebalancing_trigger=Never(),
        name="buy_and_hold_40_30_20_10"
    )

    # Portfolio 2: Equal Weight (25/25/25/25) with Monthly Rebalancing
    print("   Creating Portfolio 2: Equal Weight (25/25/25/25) - Monthly")
    portfolio2 = Portfolio(
        asset_names=tickers,
        initial_weights=pd.Series([0.25, 0.25, 0.25, 0.25], index=tickers),
        allocation_strategy=EqualWeight(name="equal_weight"),
        rebalancing_trigger=Periodic('ME'),  # Month-end
        name="equal_weight_monthly"
    )

    # Portfolio 3: Target Weight (40/30/20/10) with Quarterly Rebalancing
    print("   Creating Portfolio 3: Target Weight (40/30/20/10) - Quarterly")
    portfolio3 = Portfolio(
        asset_names=tickers,
        initial_weights=initial_weights,
        allocation_strategy=StaticAllocation(weights, name="target_40/30/20/10"),
        rebalancing_trigger=Periodic('QE'),  # Quarter-end
        name="target_weight_quarterly"
    )

    # Portfolio 4: Target Weight with Threshold Rebalancing (5% drift)
    print("   Creating Portfolio 4: Target Weight (40/30/20/10) - Threshold (5%)")
    portfolio4 = Portfolio(
        asset_names=tickers,
        initial_weights=initial_weights,
        allocation_strategy=StaticAllocation(weights, name="target_40/30/20/10"),
        rebalancing_trigger=Threshold(drift_threshold=0.05),
        name="target_weight_threshold_5pct"
    )

    portfolios = [portfolio1, portfolio2, portfolio3, portfolio4]

    print(f"\n   ✓ Created {len(portfolios)} portfolios")

    # ========================================================================
    # Step 5: Ingest MC data into all portfolios
    # ========================================================================
    print("\n5. Ingesting MC data into portfolios...")

    for portfolio in portfolios:
        portfolio.ingest_simulated_data(mc_returns_df)
        print(f"   ✓ {portfolio.name}: {len(portfolio.returns_data)} periods")

    # ========================================================================
    # Step 6: Run backtests (SIMPLIFIED - no PeriodManager)
    # ========================================================================
    print("\n6. Running simplified backtests...")
    print("   NOTE: This example uses a simplified backtest without PeriodManager")
    print("   For full backtest with period management, see test_mc_portfolio_integration.py")

    # Simple daily backtest simulation
    for portfolio in portfolios:
        print(f"\n   Running {portfolio.name}...")

        # Reset tracking
        portfolio.returns_history = pd.Series(dtype=float)
        portfolio.portfolio_values = pd.Series(dtype=float)
        portfolio.weights_history = pd.DataFrame(columns=tickers)

        current_value = 100.0
        current_weights = portfolio.current_weights.copy()

        # Iterate through each period
        for i, (date, returns_row) in enumerate(mc_returns_df.iterrows()):
            # Check if should rebalance
            should_rebalance = portfolio.rebalancing_trigger.should_rebalance(
                current_date=date.date(),
                current_weights=current_weights.values,
                target_weights=initial_weights.values
            )

            if should_rebalance:
                # Get new weights from allocation strategy
                new_weights = portfolio.allocation_strategy.calculate_weights(
                    current_weights=current_weights.values
                )
                current_weights = pd.Series(new_weights, index=tickers)
                portfolio.rebalancing_trigger.record_rebalance(date.date())

            # Calculate portfolio return
            portfolio_return = np.dot(current_weights, returns_row)

            # Update value
            current_value *= (1 + portfolio_return)

            # Update weights (drift)
            asset_values = current_weights * (1 + returns_row)
            current_weights = asset_values / asset_values.sum()

            # Record
            portfolio.returns_history.loc[date] = portfolio_return
            portfolio.portfolio_values.loc[date] = current_value
            portfolio.weights_history.loc[date] = current_weights

        print(f"      Final value: ${current_value:,.2f}")
        print(f"      Total return: {(current_value/100 - 1)*100:.2f}%")

    # ========================================================================
    # Step 7: Compare results
    # ========================================================================
    print("\n7. Comparison Results:")
    print("="*80)

    results = []
    for portfolio in portfolios:
        if len(portfolio.portfolio_values) > 0:
            final_value = portfolio.portfolio_values.iloc[-1]
            total_return = (final_value / 100 - 1) * 100

            # Count rebalances
            rebalance_count = len(portfolio.rebalancing_trigger.last_rebalance_dates) if hasattr(portfolio.rebalancing_trigger, 'last_rebalance_dates') else 0

            results.append({
                'Portfolio': portfolio.name,
                'Final_Value': final_value,
                'Total_Return_%': total_return,
                'Rebalances': rebalance_count
            })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    print("\n" + "="*80)
    print("DONE! New architecture working successfully.")
    print("="*80)

    print("\nKey Takeaways:")
    print("   ✅ AllocationStrategy defines WHAT weights to use")
    print("   ✅ RebalancingTrigger defines WHEN to rebalance")
    print("   ✅ Portfolio class integrates both cleanly")
    print("   ✅ MCPathGenerator provides simulated data")
    print("   ✅ All components work together seamlessly")

    return 0


if __name__ == "__main__":
    sys.exit(main())
