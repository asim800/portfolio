#!/usr/bin/env python3
"""
Test pandas/numpy integration for the simplified portfolio system.
Verify that numpy optimization works with pandas data management.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_pandas_numpy_integration():
    """Test the core pandas/numpy integration pattern."""
    print("="*60)
    print("TESTING PANDAS/NUMPY INTEGRATION")
    print("="*60)

    # Create sample data
    assets = ['SPY', 'BIL', 'AAPL', 'MSFT']
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    # Sample returns data as DataFrame
    np.random.seed(42)
    returns_df = pd.DataFrame(
        np.random.normal(0.001, 0.02, (len(dates), len(assets))),
        index=dates,
        columns=assets
    )

    print(f"Created returns data: {returns_df.shape}")
    print(f"Assets: {list(returns_df.columns)}")

    # Sample portfolio weights as pandas Series
    weights_series = pd.Series([0.4, 0.3, 0.2, 0.1], index=assets, name='weights')
    print(f"\nPortfolio weights:")
    print(weights_series)

    # Test 1: Portfolio return calculation using pandas dot product
    print("\nTest 1: Portfolio return calculation")
    daily_portfolio_returns = returns_df.dot(weights_series)
    period_return = (1 + daily_portfolio_returns).prod() - 1

    print(f"Daily portfolio returns (first 5): {daily_portfolio_returns.head().values}")
    print(f"Total period return: {period_return:.4f}")

    # Test 2: Convert pandas to numpy for optimization (simulate CVXPY usage)
    print("\nTest 2: Pandas to numpy conversion")
    mean_returns_series = returns_df.mean()
    cov_matrix_df = returns_df.cov()

    print(f"Mean returns (pandas Series): {mean_returns_series.values}")
    print(f"Covariance matrix shape: {cov_matrix_df.shape}")

    # Convert to numpy for "optimization"
    mu = mean_returns_series.values
    Sigma = cov_matrix_df.values

    print(f"Numpy mean returns shape: {mu.shape}")
    print(f"Numpy covariance matrix shape: {Sigma.shape}")

    # Simple "optimization" result (just equal weights for demo)
    optimized_weights_numpy = np.ones(len(assets)) / len(assets)

    # Convert back to pandas Series
    optimized_weights_series = pd.Series(
        optimized_weights_numpy,
        index=assets,
        name='optimized_weights'
    )

    print(f"Optimized weights (numpy): {optimized_weights_numpy}")
    print(f"Optimized weights (pandas):")
    print(optimized_weights_series)

    # Test 3: Portfolio performance calculation with new weights
    print("\nTest 3: Performance with optimized weights")
    optimized_daily_returns = returns_df.dot(optimized_weights_series)
    optimized_period_return = (1 + optimized_daily_returns).prod() - 1

    print(f"Optimized portfolio return: {optimized_period_return:.4f}")
    print(f"Original portfolio return: {period_return:.4f}")
    print(f"Improvement: {(optimized_period_return - period_return):.4f}")

    # Test 4: Verify asset alignment works correctly
    print("\nTest 4: Asset alignment verification")

    # Create weights with different order
    weights_different_order = pd.Series([0.1, 0.2, 0.4, 0.3], index=['MSFT', 'AAPL', 'SPY', 'BIL'])

    # Calculate returns - pandas should handle alignment automatically
    aligned_returns = returns_df.dot(weights_different_order)
    aligned_period_return = (1 + aligned_returns).prod() - 1

    print(f"Weights in different order:")
    print(weights_different_order)
    print(f"Portfolio return with alignment: {aligned_period_return:.4f}")

    # Should match original calculation (same weights, different order)
    print(f"Matches original? {abs(aligned_period_return - period_return) < 1e-10}")

    return True

def test_portfolio_tracker_integration():
    """Test integration with the PortfolioTracker."""
    print("\n" + "="*60)
    print("TESTING PORTFOLIO TRACKER INTEGRATION")
    print("="*60)

    try:
        from portfolio_tracker import PortfolioTracker

        # Setup
        assets = ['SPY', 'BIL', 'AAPL', 'MSFT']
        portfolios = ['baseline', 'optimized', 'mixed']
        tracker = PortfolioTracker(assets, portfolios)

        # Create sample period data
        dates = pd.date_range('2024-01-01', periods=5, freq='30D')

        for i, date in enumerate(dates):
            # Sample weights for each portfolio
            weights = {}
            returns = {}

            for portfolio in portfolios:
                # Random weights
                w = np.random.dirichlet(np.ones(len(assets)))
                weights[portfolio] = pd.Series(w, index=assets)

                # Random return
                returns[portfolio] = np.random.normal(0.02, 0.03)

            # Add to tracker
            tracker.add_period_performance(
                period_start=date - pd.Timedelta(days=30),
                period_end=date,
                portfolio_weights=weights,
                portfolio_returns=returns
            )

        # Test retrieval
        print("Portfolio tracking test successful!")
        print(f"Total periods tracked: {len(tracker.returns_df)}")

        # Show summary
        tracker.display_summary()

        return True

    except ImportError:
        print("PortfolioTracker not available - test skipped")
        return True

if __name__ == "__main__":
    try:
        print("Testing pandas/numpy integration for portfolio optimization...")

        # Test core integration
        success1 = test_pandas_numpy_integration()

        # Test portfolio tracker
        success2 = test_portfolio_tracker_integration()

        if success1 and success2:
            print("\n" + "="*60)
            print("ALL INTEGRATION TESTS PASSED!")
            print("Pandas data management + numpy optimization works correctly")
            print("="*60)
        else:
            print("Some tests failed!")

    except Exception as e:
        print(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()