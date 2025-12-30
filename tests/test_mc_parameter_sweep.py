#!/usr/bin/env python3
"""
Parameter Sweep for Monte Carlo Bootstrap Validation.

Runs test_mc_bootstrap simulations across a range of parameter values
and collects accumulation/decumulation percentiles for both parametric
and bootstrap methods.

Usage:
    # Default sweep (initial_portfolio_value $500K-$5M)
    uv run python tests/test_mc_parameter_sweep.py

    # Custom sweep
    uv run python tests/test_mc_parameter_sweep.py \
        --param initial_portfolio_value \
        --start 500000 \
        --end 5000000 \
        --step 500000

    # Sweep withdrawal amount
    uv run python tests/test_mc_parameter_sweep.py \
        --param annual_withdrawal_amount \
        --start 20000 \
        --end 80000 \
        --step 10000
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Change to project root so relative paths work
os.chdir(PROJECT_ROOT)

from src.config import SystemConfig
from tests.test_mc_bootstrap import run_bootstrap_comparison, create_historical_data


def run_parameter_sweep(
    config_path: str,
    param_name: str,
    param_values: list,
    num_simulations: int = 200,
    seed: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run Monte Carlo simulations across parameter values.

    Parameters:
    -----------
    config_path : str
        Path to base JSON config file
    param_name : str
        Parameter to sweep (e.g., 'initial_portfolio_value')
    param_values : list
        List of values to test
    num_simulations : int
        Number of MC simulations per run
    seed : int
        Random seed for reproducibility
    verbose : bool
        Print progress updates

    Returns:
    --------
    pd.DataFrame with columns:
        - {param_name}: The swept parameter value
        - acc_p5_param, acc_p25_param, acc_p50_param, acc_p75_param, acc_p95_param
        - acc_p5_boot, acc_p25_boot, acc_p50_boot, acc_p75_boot, acc_p95_boot
        - dec_success_param, dec_success_boot
        - dec_p5_param, dec_p25_param, dec_p50_param, dec_p75_param, dec_p95_param
        - dec_p5_boot, dec_p25_boot, dec_p50_boot, dec_p75_boot, dec_p95_boot
    """
    results = []

    for i, value in enumerate(param_values):
        if verbose:
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(param_values)}] Running {param_name} = {value:,.0f}")
            print(f"{'='*60}")

        # Load fresh config and modify parameter
        config = SystemConfig.from_json(config_path)
        setattr(config, param_name, value)

        # Get tickers from simulated_data_params (matches test_mc_validation.py)
        tickers = ['BIL', 'MSFT', 'NVDA', 'SPY']

        # Generate historical data (use same seed for consistency)
        historical_data = create_historical_data(config, tickers=tickers, seed=seed)

        # Run simulation
        result = run_bootstrap_comparison(config, historical_data, num_simulations, seed)

        # Flatten results into row
        row = {param_name: value}

        for pct in [5, 25, 50, 75, 95]:
            row[f'acc_p{pct}_param'] = result['acc_percentiles_param'][pct]
            row[f'acc_p{pct}_boot'] = result['acc_percentiles_boot'][pct]
            row[f'dec_p{pct}_param'] = result['dec_percentiles_param'][pct]
            row[f'dec_p{pct}_boot'] = result['dec_percentiles_boot'][pct]

        row['dec_success_param'] = result['dec_success_param']
        row['dec_success_boot'] = result['dec_success_boot']

        results.append(row)

        if verbose:
            print(f"  Accumulation P50: Param=${result['acc_percentiles_param'][50]:,.0f}, "
                  f"Boot=${result['acc_percentiles_boot'][50]:,.0f}")
            print(f"  Dec Success: Param={result['dec_success_param']:.1%}, "
                  f"Boot={result['dec_success_boot']:.1%}")

    return pd.DataFrame(results)


def create_sweep_visualization(
    df: pd.DataFrame,
    param_name: str,
    output_path: str = None
) -> plt.Figure:
    """
    Create visualization of parameter sweep results.

    Parameters:
    -----------
    df : pd.DataFrame
        Results from run_parameter_sweep()
    param_name : str
        Name of the swept parameter
    output_path : str, optional
        Path to save the figure

    Returns:
    --------
    plt.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = df[param_name].values

    # Format x-axis label based on parameter
    if 'value' in param_name.lower() or 'amount' in param_name.lower():
        x_label = param_name.replace('_', ' ').title()
        x_formatter = lambda v, p: f'${v/1e6:.1f}M' if v >= 1e6 else f'${v/1e3:.0f}K'
    else:
        x_label = param_name.replace('_', ' ').title()
        x_formatter = lambda v, p: f'{v:,.0f}'

    # Colors
    param_color = '#2E86AB'  # Blue for parametric
    boot_color = '#A23B72'   # Magenta for bootstrap

    # Panel 1: Accumulation percentiles (fan chart)
    ax = axes[0, 0]

    # Parametric fan
    ax.fill_between(x, df['acc_p5_param'], df['acc_p95_param'],
                    alpha=0.2, color=param_color, label='Parametric 5-95%')
    ax.fill_between(x, df['acc_p25_param'], df['acc_p75_param'],
                    alpha=0.3, color=param_color)
    ax.plot(x, df['acc_p50_param'], '-', color=param_color, linewidth=2,
            label='Parametric Median')

    # Bootstrap fan
    ax.fill_between(x, df['acc_p5_boot'], df['acc_p95_boot'],
                    alpha=0.2, color=boot_color, label='Bootstrap 5-95%')
    ax.fill_between(x, df['acc_p25_boot'], df['acc_p75_boot'],
                    alpha=0.3, color=boot_color)
    ax.plot(x, df['acc_p50_boot'], '--', color=boot_color, linewidth=2,
            label='Bootstrap Median')

    ax.set_title('Final Accumulation Value by Parameter', fontsize=12, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel('Portfolio Value at Retirement ($)', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(x_formatter))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, p: f'${v/1e6:.1f}M' if v >= 1e6 else f'${v/1e3:.0f}K'))

    # Panel 2: Decumulation percentiles (fan chart)
    ax = axes[0, 1]

    # Parametric fan
    ax.fill_between(x, df['dec_p5_param'], df['dec_p95_param'],
                    alpha=0.2, color=param_color)
    ax.fill_between(x, df['dec_p25_param'], df['dec_p75_param'],
                    alpha=0.3, color=param_color)
    ax.plot(x, df['dec_p50_param'], '-', color=param_color, linewidth=2,
            label='Parametric Median')

    # Bootstrap fan
    ax.fill_between(x, df['dec_p5_boot'], df['dec_p95_boot'],
                    alpha=0.2, color=boot_color)
    ax.fill_between(x, df['dec_p25_boot'], df['dec_p75_boot'],
                    alpha=0.3, color=boot_color)
    ax.plot(x, df['dec_p50_boot'], '--', color=boot_color, linewidth=2,
            label='Bootstrap Median')

    ax.set_title('Final Decumulation Value by Parameter', fontsize=12, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel('Final Portfolio Value ($)', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(x_formatter))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, p: f'${v/1e6:.1f}M' if v >= 1e6 else f'${v/1e3:.0f}K'))

    # Panel 3: Success rates comparison
    ax = axes[1, 0]

    width = (x[1] - x[0]) * 0.35 if len(x) > 1 else x[0] * 0.1
    ax.bar(x - width/2, df['dec_success_param'] * 100, width,
           label='Parametric', color=param_color, alpha=0.8)
    ax.bar(x + width/2, df['dec_success_boot'] * 100, width,
           label='Bootstrap', color=boot_color, alpha=0.8)

    ax.set_title('Decumulation Success Rate by Parameter', fontsize=12, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel('Success Rate (%)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(x_formatter))
    ax.set_ylim(0, 105)

    # Panel 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')

    # Create summary table
    table_data = []
    for _, row in df.iterrows():
        param_val = row[param_name]
        if param_val >= 1e6:
            param_str = f'${param_val/1e6:.1f}M'
        else:
            param_str = f'${param_val/1e3:.0f}K'

        table_data.append([
            param_str,
            f"${row['acc_p50_param']/1e6:.2f}M",
            f"${row['acc_p50_boot']/1e6:.2f}M",
            f"{row['dec_success_param']:.1%}",
            f"{row['dec_success_boot']:.1%}",
        ])

    columns = [x_label, 'Acc Med (P)', 'Acc Med (B)', 'Success (P)', 'Success (B)']

    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Summary Results', fontsize=12, fontweight='bold', pad=20)

    plt.suptitle(f'Parameter Sweep: {param_name.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Run Monte Carlo parameter sweep',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sweep initial portfolio value
  python tests/test_mc_parameter_sweep.py --param initial_portfolio_value \\
      --start 500000 --end 5000000 --step 500000

  # Sweep withdrawal amount
  python tests/test_mc_parameter_sweep.py --param annual_withdrawal_amount \\
      --start 20000 --end 80000 --step 10000

  # Custom number of simulations
  python tests/test_mc_parameter_sweep.py --param initial_portfolio_value \\
      --start 500000 --end 2000000 --step 500000 --sims 500
        """
    )

    parser.add_argument('--param', type=str, default='initial_portfolio_value',
                        help='Parameter to sweep (default: initial_portfolio_value)')
    parser.add_argument('--start', type=float, default=500_000,
                        help='Start value (default: 500000)')
    parser.add_argument('--end', type=float, default=5_000_000,
                        help='End value (default: 5000000)')
    parser.add_argument('--step', type=float, default=500_000,
                        help='Step size (default: 500000)')
    parser.add_argument('--sims', type=int, default=200,
                        help='Number of MC simulations (default: 200)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--config', type=str, default='configs/test_simple_buyhold.json',
                        help='Config file path (default: configs/test_simple_buyhold.json)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: output/sweep/{param}_sweep.csv)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating visualization')

    args = parser.parse_args()

    # Generate parameter values
    param_values = np.arange(args.start, args.end + args.step/2, args.step).tolist()

    print("=" * 80)
    print("MONTE CARLO PARAMETER SWEEP")
    print("=" * 80)
    print(f"  Parameter: {args.param}")
    print(f"  Range: {args.start:,.0f} to {args.end:,.0f} (step: {args.step:,.0f})")
    print(f"  Values: {len(param_values)}")
    print(f"  Simulations per value: {args.sims}")
    print(f"  Config: {args.config}")
    print("=" * 80)

    # Run parameter sweep
    df = run_parameter_sweep(
        config_path=args.config,
        param_name=args.param,
        param_values=param_values,
        num_simulations=args.sims,
        seed=args.seed,
        verbose=True
    )

    # Save results
    output_dir = Path('output/sweep')
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        csv_path = args.output
    else:
        csv_path = output_dir / f'{args.param}_sweep.csv'

    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("PARAMETER SWEEP RESULTS")
    print("=" * 80)
    print(df.to_string(index=False, float_format=lambda x: f'{x:,.0f}' if abs(x) > 100 else f'{x:.3f}'))

    # Create visualization
    if not args.no_plot:
        plot_path = output_dir / f'{args.param}_sweep.png'
        fig = create_sweep_visualization(df, args.param, str(plot_path))
        plt.close(fig)

    print("\n" + "=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
