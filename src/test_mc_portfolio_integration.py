#!/usr/bin/env python3
"""
Phase 3: MC → Portfolio Integration Test (NEW ARCHITECTURE)

Tests the complete integration of Monte Carlo path generation with Portfolio system.
Uses the new architecture with AllocationStrategy + RebalancingTrigger.
"""

import sys
import os
import numpy as np
import pandas as pd
import logging

# Add src3 directory to path - all imports from src3 only
current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

# Import everything from src3
from mc_path_generator import MCPathGenerator
from portfolio import Portfolio
from allocation_strategies import StaticAllocation, EqualWeight
from rebalancing_triggers import Never, Periodic

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Suppress info logs for cleaner test output
    format='%(levelname)s - %(message)s'
)


def test_basic_mc_integration():
    """Test basic MC data flow through portfolio system."""
    print("\n" + "="*80)
    print("TEST 1: Basic MC → Portfolio Integration")
    print("="*80)

    # Setup test parameters
    tickers = ['SPY', 'AGG', 'NVDA', 'GLD']
    mean_returns = np.array([0.10, 0.04, 0.20, 0.06])
    cov_matrix = np.array([
        [0.04, 0.01, 0.02, 0.005],
        [0.01, 0.02, 0.005, 0.01],
        [0.02, 0.005, 0.08, 0.01],
        [0.005, 0.01, 0.01, 0.03]
    ])

    # Generate MC paths
    print("\n1. Generating MC paths...")
    generator = MCPathGenerator(tickers, mean_returns, cov_matrix, seed=42)
    paths = generator.generate_paths(
        num_simulations=10,
        total_periods=260,  # 10 years of biweekly data
        periods_per_year=26
    )
    print(f"   ✓ Generated paths: {paths.shape}")

    # Convert to DataFrame
    print("\n2. Converting to DataFrame...")
    mc_returns_df = generator.get_path_dataframe(
        simulation_idx=0,
        start_date='2025-01-01',
        frequency='2W'
    )
    print(f"   ✓ DataFrame shape: {mc_returns_df.shape}")

    # Create portfolio
    print("\n3. Creating buy-and-hold portfolio...")
    initial_weights = pd.Series([0.4, 0.3, 0.2, 0.1], index=tickers)
    portfolio = Portfolio(
        asset_names=tickers,
        initial_weights=initial_weights,
        allocation_strategy=StaticAllocation([0.4, 0.3, 0.2, 0.1], name="40/30/20/10"),
        rebalancing_trigger=Never(),
        name="buy_and_hold"
    )
    print(f"   ✓ Portfolio created: {portfolio.name}")

    # Ingest MC data
    print("\n4. Ingesting MC data...")
    portfolio.ingest_simulated_data(mc_returns_df)
    print(f"   ✓ Data ingested: {len(portfolio.returns_data)} periods")

    # Verify data
    print("\n5. Verifying integration...")
    assert portfolio.returns_data is not None, "returns_data is None"
    assert portfolio.returns_data.shape == (260, 4), f"Wrong shape: {portfolio.returns_data.shape}"
    assert list(portfolio.returns_data.columns) == tickers, "Columns mismatch"
    assert isinstance(portfolio.returns_data.index, pd.DatetimeIndex), "Index not DatetimeIndex"
    print(f"   ✓ All validations passed")

    print("\n" + "="*80)
    print("TEST 1: PASSED ✓")
    print("="*80)

    return True


def test_portfolio_backtest_with_mc():
    """Test portfolio backtest using MC data (simplified without PeriodManager)."""
    print("\n" + "="*80)
    print("TEST 2: Portfolio Backtest with MC Data")
    print("="*80)

    # Setup
    tickers = ['SPY', 'AGG']
    mean_returns = np.array([0.10, 0.04])
    cov_matrix = np.array([[0.04, 0.01], [0.01, 0.02]])

    # Generate MC paths
    print("\n1. Generating MC paths...")
    generator = MCPathGenerator(tickers, mean_returns, cov_matrix, seed=42)
    paths = generator.generate_paths(
        num_simulations=1,
        total_periods=52,  # 1 year of weekly data
        periods_per_year=52
    )
    print(f"   ✓ Generated paths: {paths.shape}")

    # Convert to DataFrame
    mc_returns_df = generator.get_path_dataframe(
        simulation_idx=0,
        start_date='2025-01-01',
        frequency='W'
    )

    # Create two portfolios for comparison
    print("\n2. Creating portfolios...")
    initial_weights = pd.Series([0.6, 0.4], index=tickers)

    # Buy & hold
    portfolio_bh = Portfolio(
        asset_names=tickers,
        initial_weights=initial_weights,
        allocation_strategy=StaticAllocation([0.6, 0.4], name="60/40"),
        rebalancing_trigger=Never(),
        name="buy_and_hold"
    )

    # Monthly rebalancing
    portfolio_monthly = Portfolio(
        asset_names=tickers,
        initial_weights=initial_weights,
        allocation_strategy=StaticAllocation([0.6, 0.4], name="60/40"),
        rebalancing_trigger=Periodic('ME'),
        name="monthly_rebalance"
    )

    portfolios = [portfolio_bh, portfolio_monthly]
    print(f"   ✓ Created {len(portfolios)} portfolios")

    # Ingest data
    print("\n3. Ingesting MC data...")
    for portfolio in portfolios:
        portfolio.ingest_simulated_data(mc_returns_df)
    print(f"   ✓ Data ingested into all portfolios")

    # Run simplified backtest
    print("\n4. Running simplified backtests...")
    for portfolio in portfolios:
        current_value = 100.0
        current_weights = portfolio.current_weights.copy()

        for date, returns_row in mc_returns_df.iterrows():
            # Check rebalancing
            should_rebalance = portfolio.rebalancing_trigger.should_rebalance(
                current_date=date.date(),
                current_weights=current_weights.values,
                target_weights=initial_weights.values
            )

            if should_rebalance:
                new_weights = portfolio.allocation_strategy.calculate_weights(
                    current_weights=current_weights.values
                )
                current_weights = pd.Series(new_weights, index=tickers)
                portfolio.rebalancing_trigger.record_rebalance(date.date())

            # Calculate return
            portfolio_return = np.dot(current_weights, returns_row)
            current_value *= (1 + portfolio_return)

            # Update weights (drift)
            asset_values = current_weights * (1 + returns_row)
            current_weights = asset_values / asset_values.sum()

            # Record
            portfolio.returns_history.loc[date] = portfolio_return
            portfolio.portfolio_values.loc[date] = current_value

        print(f"   {portfolio.name}: Final value = ${current_value:.2f}")

    # Verify results
    print("\n5. Verifying results...")
    for portfolio in portfolios:
        assert len(portfolio.returns_history) == 52, f"{portfolio.name}: Wrong history length"
        assert len(portfolio.portfolio_values) == 52, f"{portfolio.name}: Wrong values length"
        assert portfolio.portfolio_values.iloc[-1] > 0, f"{portfolio.name}: Final value <= 0"
    print(f"   ✓ All portfolios have valid results")

    print("\n" + "="*80)
    print("TEST 2: PASSED ✓")
    print("="*80)

    return True


def test_multiple_simulations():
    """Test running multiple MC simulations through portfolio system."""
    print("\n" + "="*80)
    print("TEST 3: Multiple Simulations")
    print("="*80)

    # Setup
    tickers = ['SPY', 'AGG']
    mean_returns = np.array([0.10, 0.04])
    cov_matrix = np.array([[0.04, 0.01], [0.01, 0.02]])

    # Generate MC paths
    print("\n1. Generating 3 MC simulations...")
    generator = MCPathGenerator(tickers, mean_returns, cov_matrix, seed=42)
    paths = generator.generate_paths(
        num_simulations=3,
        total_periods=26,  # 6 months of biweekly data
        periods_per_year=26
    )
    print(f"   ✓ Generated paths: {paths.shape}")

    # Run each simulation
    print("\n2. Running analysis for each simulation...")
    results = []

    for sim_idx in range(3):
        # Get DataFrame for this simulation
        mc_returns_df = generator.get_path_dataframe(
            simulation_idx=sim_idx,
            start_date='2025-01-01',
            frequency='2W'
        )

        # Create portfolio
        initial_weights = pd.Series([0.6, 0.4], index=tickers)
        portfolio = Portfolio(
            asset_names=tickers,
            initial_weights=initial_weights,
            allocation_strategy=StaticAllocation([0.6, 0.4], name="60/40"),
            rebalancing_trigger=Never(),
            name=f"buy_hold_sim_{sim_idx}"
        )

        # Ingest data
        portfolio.ingest_simulated_data(mc_returns_df)

        # Run simple backtest
        current_value = 100.0
        current_weights = portfolio.current_weights.copy()

        for date, returns_row in mc_returns_df.iterrows():
            portfolio_return = np.dot(current_weights, returns_row)
            current_value *= (1 + portfolio_return)

            # Update weights (drift)
            asset_values = current_weights * (1 + returns_row)
            current_weights = asset_values / asset_values.sum()

        results.append({
            'sim_idx': sim_idx,
            'final_value': current_value,
            'mean_return': mc_returns_df.mean().mean()
        })

        print(f"   Simulation {sim_idx}: final_value=${current_value:.2f}")

    # Verify all simulations ran
    print("\n3. Verifying all simulations...")
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    for r in results:
        assert r['final_value'] > 0, f"Sim {r['sim_idx']}: Invalid final value"
    print("   ✓ All simulations completed successfully")

    print("\n" + "="*80)
    print("TEST 3: PASSED ✓")
    print("="*80)

    return True


def test_allocation_strategies_with_mc():
    """Test different allocation strategies with MC data."""
    print("\n" + "="*80)
    print("TEST 4: Different Allocation Strategies with MC Data")
    print("="*80)

    # Setup
    tickers = ['SPY', 'AGG', 'NVDA', 'GLD']
    mean_returns = np.array([0.10, 0.04, 0.20, 0.06])
    cov_matrix = np.array([
        [0.04, 0.01, 0.02, 0.005],
        [0.01, 0.02, 0.005, 0.01],
        [0.02, 0.005, 0.08, 0.01],
        [0.005, 0.01, 0.01, 0.03]
    ])

    # Generate MC paths
    print("\n1. Generating MC paths...")
    generator = MCPathGenerator(tickers, mean_returns, cov_matrix, seed=42)
    paths = generator.generate_paths(
        num_simulations=1,
        total_periods=260,
        periods_per_year=26
    )
    mc_returns_df = generator.get_path_dataframe(
        simulation_idx=0,
        start_date='2025-01-01',
        frequency='2W'
    )
    print(f"   ✓ Generated {len(mc_returns_df)} periods")

    # Create portfolios with different strategies
    print("\n2. Creating portfolios with different strategies...")

    portfolios = []

    # Static allocation
    portfolios.append(Portfolio(
        asset_names=tickers,
        initial_weights=pd.Series([0.4, 0.3, 0.2, 0.1], index=tickers),
        allocation_strategy=StaticAllocation([0.4, 0.3, 0.2, 0.1], name="static"),
        rebalancing_trigger=Periodic('ME'),
        name="static_monthly"
    ))

    # Equal weight
    portfolios.append(Portfolio(
        asset_names=tickers,
        initial_weights=pd.Series([0.25, 0.25, 0.25, 0.25], index=tickers),
        allocation_strategy=EqualWeight(name="equal"),
        rebalancing_trigger=Periodic('ME'),
        name="equal_weight_monthly"
    ))

    print(f"   ✓ Created {len(portfolios)} portfolios")

    # Ingest data and run backtests
    print("\n3. Running backtests...")
    for portfolio in portfolios:
        portfolio.ingest_simulated_data(mc_returns_df)

        current_value = 100.0
        current_weights = portfolio.current_weights.copy()
        target_weights = portfolio.current_weights.copy()

        for date, returns_row in mc_returns_df.iterrows():
            should_rebalance = portfolio.rebalancing_trigger.should_rebalance(
                current_date=date.date(),
                current_weights=current_weights.values,
                target_weights=target_weights.values
            )

            if should_rebalance:
                new_weights = portfolio.allocation_strategy.calculate_weights(
                    current_weights=current_weights.values
                )
                current_weights = pd.Series(new_weights, index=tickers)
                portfolio.rebalancing_trigger.record_rebalance(date.date())

            portfolio_return = np.dot(current_weights, returns_row)
            current_value *= (1 + portfolio_return)

            asset_values = current_weights * (1 + returns_row)
            current_weights = asset_values / asset_values.sum()

            portfolio.portfolio_values.loc[date] = current_value

        print(f"   {portfolio.name}: ${current_value:.2f}")

    # Verify
    print("\n4. Verifying results...")
    for portfolio in portfolios:
        assert len(portfolio.portfolio_values) > 0, f"{portfolio.name}: No values"
        assert portfolio.portfolio_values.iloc[-1] > 0, f"{portfolio.name}: Invalid final value"
    print("   ✓ All strategies produced valid results")

    print("\n" + "="*80)
    print("TEST 4: PASSED ✓")
    print("="*80)

    return True


def main():
    """Run all integration tests."""
    print("\n" + "="*80)
    print("PHASE 3: MC → PORTFOLIO INTEGRATION TESTS (NEW ARCHITECTURE)")
    print("="*80)

    tests = [
        ("Basic MC Integration", test_basic_mc_integration),
        ("Portfolio Backtest with MC", test_portfolio_backtest_with_mc),
        ("Multiple Simulations", test_multiple_simulations),
        ("Different Allocation Strategies", test_allocation_strategies_with_mc)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {test_name} FAILED with exception:")
            print(f"  {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("PHASE 3 TEST RESULTS (NEW ARCHITECTURE)")
    print("="*80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✓ ALL PHASE 3 TESTS PASSED - NEW ARCHITECTURE INTEGRATION WORKING!")
        print("="*80)
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
