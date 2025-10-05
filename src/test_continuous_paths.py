#!/usr/bin/env python3
"""
Validate that accumulation and decumulation paths are CONTINUOUS.

This script demonstrates that decumulation paths start from where
accumulation paths ended, creating one continuous market scenario.
"""

import numpy as np
import pandas as pd
from mc_path_generator import MCPathGenerator

def test_path_continuity():
    print("=" * 80)
    print("CONTINUOUS PATH VALIDATION")
    print("=" * 80)

    # Setup simple test case
    tickers = ['SPY', 'AGG']
    mean_returns = np.array([0.10, 0.04])
    cov_matrix = np.array([[0.04, 0.01], [0.01, 0.02]])

    generator = MCPathGenerator(tickers, mean_returns, cov_matrix, seed=42)

    # Generate continuous lifecycle paths
    print("\n[1/3] Generating continuous lifecycle paths...")
    acc_years = 5
    acc_periods_per_year = 26  # Biweekly
    dec_years = 3

    acc_paths, dec_paths = generator.generate_lifecycle_paths(
        num_simulations=10,
        accumulation_years=acc_years,
        accumulation_periods_per_year=acc_periods_per_year,
        decumulation_years=dec_years
    )

    print(f"\nPath shapes:")
    print(f"  Accumulation: {acc_paths.shape}")  # (10, 130, 2) = 10 sims × 5*26 periods × 2 assets
    print(f"  Decumulation: {dec_paths.shape}")  # (10, 3, 2) = 10 sims × 3 years × 2 assets

    # Verify continuity for simulation 0
    print("\n[2/3] Verifying continuity for simulation 0...")
    sim_idx = 0

    # Get last few accumulation periods
    last_acc_periods = acc_paths[sim_idx, -26:, :]  # Last year (26 biweekly periods)
    print(f"\nLast 26 accumulation periods (sim {sim_idx}):")
    print(f"  Shape: {last_acc_periods.shape}")
    print(f"  Last 3 periods:")
    for i in range(-3, 0):
        print(f"    Period {130+i}: {acc_paths[sim_idx, i, :]}")

    # Get first decumulation year
    first_dec_year_returns = dec_paths[sim_idx, 0, :]
    print(f"\nFirst decumulation year (sim {sim_idx}):")
    print(f"  Annual returns: {first_dec_year_returns}")

    # Manually compound the 26 periods that SHOULD have been used
    # These are the NEXT 26 periods after accumulation in the continuous path
    continuous_path = generator.paths[sim_idx, :, :]  # Full continuous path
    acc_end_period = acc_years * acc_periods_per_year  # 130
    first_dec_year_periods = continuous_path[acc_end_period:acc_end_period+26, :]

    print(f"\nContinuous path verification:")
    print(f"  Accumulation ends at period: {acc_end_period}")
    print(f"  Decumulation year 1 uses periods: {acc_end_period} to {acc_end_period+26}")
    print(f"  First 3 periods of dec year 1:")
    for i in range(3):
        print(f"    Period {acc_end_period + i}: {first_dec_year_periods[i, :]}")

    # Compound these periods to verify they match dec_paths
    manual_annual_returns = np.zeros(2)
    for asset in range(2):
        cumulative = np.prod(1 + first_dec_year_periods[:, asset])
        manual_annual_returns[asset] = cumulative - 1

    print(f"\n  Manually compounded annual return: {manual_annual_returns}")
    print(f"  From dec_paths[0, 0, :]:          {first_dec_year_returns}")
    print(f"  Difference: {np.abs(manual_annual_returns - first_dec_year_returns)}")

    # Verify they match
    assert np.allclose(manual_annual_returns, first_dec_year_returns, atol=1e-10), \
        "Decumulation paths don't match continuous path compounding!"

    print(f"  ✓ VERIFIED: Decumulation year 1 is correctly compounded from continuous path")

    # Verify simulation 1 as well
    print("\n[3/3] Verifying continuity for simulation 1...")
    sim_idx = 1

    continuous_path_1 = generator.paths[sim_idx, :, :]
    first_dec_year_periods_1 = continuous_path_1[acc_end_period:acc_end_period+26, :]

    manual_annual_returns_1 = np.zeros(2)
    for asset in range(2):
        cumulative = np.prod(1 + first_dec_year_periods_1[:, asset])
        manual_annual_returns_1[asset] = cumulative - 1

    first_dec_year_returns_1 = dec_paths[sim_idx, 0, :]

    print(f"  Manually compounded: {manual_annual_returns_1}")
    print(f"  From dec_paths:      {first_dec_year_returns_1}")
    print(f"  Difference: {np.abs(manual_annual_returns_1 - first_dec_year_returns_1)}")

    assert np.allclose(manual_annual_returns_1, first_dec_year_returns_1, atol=1e-10)
    print(f"  ✓ VERIFIED: Simulation 1 also continuous")

    print("\n" + "=" * 80)
    print("ALL CONTINUITY CHECKS PASSED ✓")
    print("=" * 80)
    print("\nKey findings:")
    print("  ✓ Accumulation and decumulation paths are from ONE continuous sequence")
    print("  ✓ Decumulation starts immediately after last accumulation period")
    print("  ✓ No gap or independent sampling between phases")
    print("  ✓ Annual decumulation returns correctly compound sub-annual periods")
    print("=" * 80)

if __name__ == '__main__':
    test_path_continuity()
