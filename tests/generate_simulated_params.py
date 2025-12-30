#!/usr/bin/env python3
"""
Generate and save simulated data parameters to files.

This script creates the CSV and text files containing mean returns and
covariance matrices for Monte Carlo validation testing.

Usage:
    uv run python tests/generate_simulated_params.py
"""

import sys
import os

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from src.config import SystemConfig
from src.data import simulated as sim_params


def main():
    """Generate and save parameter files based on config."""
    print("=" * 80)
    print("GENERATING SIMULATED DATA PARAMETER FILES")
    print("=" * 80)

    # Load configuration
    config = SystemConfig.from_json(os.path.join(PROJECT_ROOT, 'configs/test_simple_buyhold.json'))

    # Load tickers
    tickers_df = pd.read_csv(config.ticker_file)
    tickers = tickers_df['Symbol'].tolist()

    print(f"\nTickers: {tickers}")
    print(f"Output files:")
    print(f"  - Mean returns: {config.simulated_mean_returns_file}")
    print(f"  - Covariance matrices: {config.simulated_cov_matrices_file}")

    # Save parameters
    print("\nGenerating files...")
    sim_params.save_all_parameters(
        mean_csv_path=config.simulated_mean_returns_file,
        cov_txt_path=config.simulated_cov_matrices_file,
        tickers=tickers
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    # Verify by loading back
    print("\nVerifying files...")
    mean_acc, mean_dec, cov_acc, cov_dec = sim_params.load_all_parameters(
        mean_csv_path=config.simulated_mean_returns_file,
        cov_txt_path=config.simulated_cov_matrices_file,
        n_assets=len(tickers)
    )

    print(f"✓ Mean returns loaded: acc shape={mean_acc.shape}, dec shape={mean_dec.shape}")
    print(f"✓ Covariance matrices loaded: acc shape={cov_acc.shape}, dec shape={cov_dec.shape}")
    print("\nAccumulation mean returns:")
    print(f"  {dict(zip(tickers, mean_acc))}")
    print("\nDecumulation mean returns:")
    print(f"  {dict(zip(tickers, mean_dec))}")


if __name__ == '__main__':
    main()
