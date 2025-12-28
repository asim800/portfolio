#!/usr/bin/env python3
"""
Generate and save simulated data parameters to files.

This script creates the CSV and text files containing mean returns and
covariance matrices for Monte Carlo validation testing.

Usage:
    uv run python generate_simulated_params.py
"""

import pandas as pd
from system_config import SystemConfig
import simulated_data_params as sim_params


def main():
    """Generate and save parameter files based on config."""
    print("=" * 80)
    print("GENERATING SIMULATED DATA PARAMETER FILES")
    print("=" * 80)

    # Load configuration
    config = SystemConfig.from_json('../configs/test_simple_buyhold.json')

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
