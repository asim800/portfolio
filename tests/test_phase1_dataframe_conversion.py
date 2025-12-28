#!/usr/bin/env python3
"""
Phase 1 Test: DataFrame Conversion Methods in MCPathGenerator

Tests the new get_path_dataframe() and get_multiple_path_dataframes() methods.
"""

import numpy as np
import pandas as pd
import sys
from mc_path_generator import MCPathGenerator


def test_single_path_dataframe():
    """Test get_path_dataframe() method."""
    print("\n" + "="*80)
    print("TEST 1: Single Path DataFrame Conversion")
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

    # Create generator and generate paths
    generator = MCPathGenerator(tickers, mean_returns, cov_matrix, seed=42)
    paths = generator.generate_paths(
        num_simulations=100,
        total_periods=260,
        periods_per_year=26
    )

    print(f"✓ Generated paths: {paths.shape}")
    print(f"  Expected: (100, 260, 4)")
    assert paths.shape == (100, 260, 4), "Path shape mismatch"

    # Convert to DataFrame
    returns_df = generator.get_path_dataframe(
        simulation_idx=0,
        start_date='2025-01-01',
        frequency='2W'
    )

    print(f"\n✓ DataFrame conversion successful")
    print(f"  Shape: {returns_df.shape}")
    print(f"  Expected: (260, 4)")
    assert returns_df.shape == (260, 4), "DataFrame shape mismatch"

    print(f"\n✓ Column names correct")
    print(f"  Columns: {returns_df.columns.tolist()}")
    print(f"  Expected: {tickers}")
    assert returns_df.columns.tolist() == tickers, "Column names mismatch"

    print(f"\n✓ Index is DatetimeIndex")
    print(f"  Type: {type(returns_df.index)}")
    assert isinstance(returns_df.index, pd.DatetimeIndex), "Index must be DatetimeIndex"

    print(f"\n✓ Date range correct")
    print(f"  Start: {returns_df.index[0].date()}")
    print(f"  End: {returns_df.index[-1].date()}")
    print(f"  Frequency: {returns_df.index.freq}")

    # Verify data values match original paths
    print(f"\n✓ Values match original paths")
    original_values = paths[0, :, :]
    df_values = returns_df.values
    assert np.allclose(original_values, df_values), "Values don't match original paths"
    print(f"  Max difference: {np.max(np.abs(original_values - df_values))}")

    print("\n" + "="*80)
    print("TEST 1: PASSED ✓")
    print("="*80)

    return True


def test_multiple_path_dataframes():
    """Test get_multiple_path_dataframes() method."""
    print("\n" + "="*80)
    print("TEST 2: Multiple Path DataFrames Conversion")
    print("="*80)

    # Setup
    tickers = ['SPY', 'AGG']
    mean_returns = np.array([0.10, 0.04])
    cov_matrix = np.array([[0.04, 0.01], [0.01, 0.02]])

    # Create generator and generate paths
    generator = MCPathGenerator(tickers, mean_returns, cov_matrix, seed=42)
    paths = generator.generate_paths(
        num_simulations=10,
        total_periods=52,
        periods_per_year=52
    )

    print(f"✓ Generated paths: {paths.shape}")

    # Convert first 5 simulations to DataFrames
    sim_indices = [0, 1, 2, 3, 4]
    dfs = generator.get_multiple_path_dataframes(
        simulation_indices=sim_indices,
        start_date='2025-01-01',
        frequency='W'
    )

    print(f"\n✓ Multiple DataFrame conversion successful")
    print(f"  Number of DataFrames: {len(dfs)}")
    print(f"  Expected: {len(sim_indices)}")
    assert len(dfs) == len(sim_indices), "Wrong number of DataFrames"

    # Check each DataFrame
    for idx in sim_indices:
        df = dfs[idx]
        print(f"\n✓ Simulation {idx}:")
        print(f"  Shape: {df.shape}")
        assert df.shape == (52, 2), f"DataFrame {idx} has wrong shape"

        # Verify values match
        original_values = paths[idx, :, :]
        df_values = df.values
        assert np.allclose(original_values, df_values), f"Values don't match for simulation {idx}"
        print(f"  Values match original: ✓")

    print("\n" + "="*80)
    print("TEST 2: PASSED ✓")
    print("="*80)

    return True


def test_all_simulations_conversion():
    """Test converting all simulations when simulation_indices is None."""
    print("\n" + "="*80)
    print("TEST 3: All Simulations Conversion")
    print("="*80)

    # Setup
    tickers = ['SPY', 'AGG']
    mean_returns = np.array([0.10, 0.04])
    cov_matrix = np.array([[0.04, 0.01], [0.01, 0.02]])

    num_sims = 5
    generator = MCPathGenerator(tickers, mean_returns, cov_matrix, seed=42)
    paths = generator.generate_paths(
        num_simulations=num_sims,
        total_periods=26,
        periods_per_year=26
    )

    # Convert all simulations (None means all)
    dfs = generator.get_multiple_path_dataframes(
        simulation_indices=None,
        start_date='2025-01-01',
        frequency='2W'
    )

    print(f"✓ All simulations converted")
    print(f"  Number of DataFrames: {len(dfs)}")
    print(f"  Expected: {num_sims}")
    assert len(dfs) == num_sims, "Should convert all simulations when indices is None"

    print("\n" + "="*80)
    print("TEST 3: PASSED ✓")
    print("="*80)

    return True


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\n" + "="*80)
    print("TEST 4: Error Handling")
    print("="*80)

    # Setup
    tickers = ['SPY', 'AGG']
    mean_returns = np.array([0.10, 0.04])
    cov_matrix = np.array([[0.04, 0.01], [0.01, 0.02]])

    generator = MCPathGenerator(tickers, mean_returns, cov_matrix, seed=42)

    # Test 1: Call before generating paths
    print("\n✓ Test: Call before generate_paths()")
    try:
        generator.get_path_dataframe(0, '2025-01-01', 'D')
        print("  ERROR: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  Correctly raised ValueError: {str(e)}")

    # Generate paths
    paths = generator.generate_paths(
        num_simulations=10,
        total_periods=52,
        periods_per_year=52
    )

    # Test 2: Invalid simulation index (negative)
    print("\n✓ Test: Negative simulation index")
    try:
        generator.get_path_dataframe(-1, '2025-01-01', 'D')
        print("  ERROR: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  Correctly raised ValueError: {str(e)}")

    # Test 3: Invalid simulation index (too large)
    print("\n✓ Test: Simulation index out of range")
    try:
        generator.get_path_dataframe(100, '2025-01-01', 'D')
        print("  ERROR: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  Correctly raised ValueError: {str(e)}")

    print("\n" + "="*80)
    print("TEST 4: PASSED ✓")
    print("="*80)

    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PHASE 1 TESTS: DataFrame Conversion Methods")
    print("="*80)

    tests = [
        ("Single Path DataFrame", test_single_path_dataframe),
        ("Multiple Path DataFrames", test_multiple_path_dataframes),
        ("All Simulations Conversion", test_all_simulations_conversion),
        ("Error Handling", test_error_handling)
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
    print("PHASE 1 TEST RESULTS")
    print("="*80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✓ ALL PHASE 1 TESTS PASSED")
        print("="*80)
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
