#!/usr/bin/env python3
"""
Basic tests for the Portfolio class.
Validates core functionality of the refactored architecture.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from portfolio import Portfolio
from portfolio_optimizer import PortfolioOptimizer
from period_manager import PeriodManager


def test_portfolio_creation():
    """Test Portfolio factory methods."""
    print("=" * 60)
    print("TEST: Portfolio Creation")
    print("=" * 60)

    asset_names = ['SPY', 'AGG', 'BIL']
    weights = pd.Series([0.6, 0.3, 0.1], index=asset_names)

    # Test buy-and-hold creation
    try:
        buy_hold = Portfolio.create_buy_and_hold(asset_names, weights)
        print(f"✓ Buy-and-hold portfolio created: {buy_hold.name}")
        assert buy_hold.name == 'buy_and_hold'
        assert len(buy_hold.asset_names) == 3
    except Exception as e:
        print(f"✗ Buy-and-hold creation failed: {e}")
        return False

    # Test target weight creation
    try:
        target_wt = Portfolio.create_target_weight(asset_names, weights, rebalance_days=30)
        print(f"✓ Target weight portfolio created: {target_wt.name}")
        assert target_wt.name == 'target_weight'
    except Exception as e:
        print(f"✗ Target weight creation failed: {e}")
        return False

    # Test equal weight creation
    try:
        equal_wt = Portfolio.create_equal_weight(asset_names, rebalance_days=30)
        print(f"✓ Equal weight portfolio created: {equal_wt.name}")
        assert equal_wt.name == 'equal_weight'
        # Check weights are equal
        current_weights = equal_wt.get_current_weights()
        assert np.allclose(current_weights.values, 1/3), "Weights should be equal"
    except Exception as e:
        print(f"✗ Equal weight creation failed: {e}")
        return False

    # Test optimized portfolio creation
    try:
        optimizer = PortfolioOptimizer(risk_free_rate=0.02)
        optimized = Portfolio.create_optimized(
            asset_names, weights, optimizer,
            method='mean_variance', rebalance_days=30
        )
        print(f"✓ Optimized portfolio created: {optimized.name}")
        assert optimized.name == 'optimized_mean_variance'
        assert optimized.optimizer is not None
    except Exception as e:
        print(f"✗ Optimized creation failed: {e}")
        return False

    print("\n✓ All portfolio creation tests passed!")
    return True


def test_portfolio_data_ingestion():
    """Test data ingestion."""
    print("\n" + "=" * 60)
    print("TEST: Portfolio Data Ingestion")
    print("=" * 60)

    asset_names = ['SPY', 'AGG']
    weights = pd.Series([0.6, 0.4], index=asset_names)

    # Create simulated returns data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
    returns = pd.DataFrame(
        np.random.randn(len(dates), 2) * 0.01,  # 1% daily volatility
        index=dates,
        columns=asset_names
    )

    portfolio = Portfolio.create_buy_and_hold(asset_names, weights)

    try:
        portfolio.ingest_simulated_data(returns)
        print(f"✓ Simulated data ingested: {len(portfolio.returns_data)} days")
        assert portfolio.returns_data is not None
        assert len(portfolio.returns_data) == len(returns)
    except Exception as e:
        print(f"✗ Data ingestion failed: {e}")
        return False

    print("\n✓ Data ingestion test passed!")
    return True


def test_portfolio_backtest():
    """Test portfolio backtesting."""
    print("\n" + "=" * 60)
    print("TEST: Portfolio Backtest")
    print("=" * 60)

    asset_names = ['SPY', 'AGG']
    weights = pd.Series([0.6, 0.4], index=asset_names)

    # Create simulated returns data (90 days)
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
    returns = pd.DataFrame(
        np.random.randn(len(dates), 2) * 0.01,
        index=dates,
        columns=asset_names
    )

    # Create portfolio and ingest data
    portfolio = Portfolio.create_buy_and_hold(asset_names, weights)
    portfolio.ingest_simulated_data(returns)

    # Create period manager (30-day periods)
    try:
        period_manager = PeriodManager(returns, frequency="ME")
        print(f"✓ Period manager created: {period_manager.num_periods} periods")
    except Exception as e:
        print(f"✗ Period manager creation failed: {e}")
        return False

    # Run backtest
    try:
        portfolio.run_backtest(period_manager)
        print(f"✓ Backtest completed")
        print(f"  Returns history length: {len(portfolio.returns_history)}")
        print(f"  Portfolio values length: {len(portfolio.portfolio_values)}")
        print(f"  Final value: {portfolio.portfolio_values.iloc[-1]:.2f}")

        assert len(portfolio.returns_history) > 0, "Should have returns"
        assert len(portfolio.portfolio_values) > 0, "Should have values"

    except Exception as e:
        print(f"✗ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✓ Backtest test passed!")
    return True


def test_portfolio_metrics():
    """Test portfolio metrics calculation."""
    print("\n" + "=" * 60)
    print("TEST: Portfolio Metrics")
    print("=" * 60)

    asset_names = ['SPY', 'AGG']
    weights = pd.Series([0.6, 0.4], index=asset_names)

    # Create simulated returns data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
    returns = pd.DataFrame(
        np.random.randn(len(dates), 2) * 0.01,
        index=dates,
        columns=asset_names
    )

    # Create and run portfolio
    portfolio = Portfolio.create_target_weight(asset_names, weights, rebalance_days=30)
    portfolio.ingest_simulated_data(returns)

    period_manager = PeriodManager(returns, frequency="ME")
    portfolio.run_backtest(period_manager)

    # Get summary statistics
    try:
        summary = portfolio.get_summary_statistics()
        print(f"✓ Summary statistics calculated")
        print(f"\nMetrics:")
        print(summary.to_string())

        assert 'Total_Return' in summary.columns
        assert 'Annual_Return' in summary.columns
        assert 'Sharpe_Ratio' in summary.columns
        assert 'Max_Drawdown' in summary.columns

    except Exception as e:
        print(f"✗ Metrics calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✓ Metrics test passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PORTFOLIO CLASS VALIDATION TESTS")
    print("=" * 60 + "\n")

    tests = [
        test_portfolio_creation,
        test_portfolio_data_ingestion,
        test_portfolio_backtest,
        test_portfolio_metrics,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
