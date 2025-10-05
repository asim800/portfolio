#!/usr/bin/env python3
"""
Quick demo of the retirement Monte Carlo engine.
Run this to see the engine in action!
"""

import logging
from retirement_config import RetirementConfig
from retirement_engine import RetirementEngine
from retirement_visualization import RetirementVisualizer

# Setup logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

def main():
    print("="*60)
    print("RETIREMENT MONTE CARLO SIMULATION - DEMO")
    print("="*60)

    # Configure retirement scenario
    config = RetirementConfig(
        initial_portfolio=1_000_000,
        start_date='2024-01-01',
        end_date='2054-01-01',  # 30 years
        annual_withdrawal=40_000,  # 4% withdrawal rate
        current_portfolio={'SPY': 0.6, 'AGG': 0.4},  # 60/40 portfolio
        inflation_rate=0.03
    )

    print(f"\nConfiguration:")
    print(f"  Initial Portfolio:   ${config.initial_portfolio:,}")
    print(f"  Annual Withdrawal:   ${config.annual_withdrawal:,}")
    print(f"  Withdrawal Rate:     {config.withdrawal_rate:.1%}")
    print(f"  Time Horizon:        {config.num_years} years")
    print(f"  Portfolio:           {config.current_portfolio}")
    print(f"  Inflation:           {config.inflation_rate:.1%}")

    # Create engine
    print(f"\nInitializing retirement engine...")
    engine = RetirementEngine(config)

    # Run Monte Carlo simulation
    print(f"\nRunning Monte Carlo simulation with 1000 paths...")
    print("(This will take ~10-15 seconds)")
    results = engine.run_monte_carlo(
        num_simulations=1000,
        method='bootstrap',
        seed=42  # For reproducibility
    )

    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nSuccess Rate:         {results.success_rate:.1%}")
    print(f"  ({int(results.success_rate * 1000)}/1000 simulations succeeded)")

    print(f"\nFinal Portfolio Values:")
    print(f"  Mean:                ${results.mean_final_value:,.0f}")
    print(f"  Median:              ${results.median_final_value:,.0f}")
    print(f"  Std Dev:             ${results.std_final_value:,.0f}")

    print(f"\nPercentiles:")
    print(f"  95th (best case):    ${results.percentiles['95th']:,.0f}")
    print(f"  75th:                ${results.percentiles['75th']:,.0f}")
    print(f"  50th (median):       ${results.percentiles['50th']:,.0f}")
    print(f"  25th:                ${results.percentiles['25th']:,.0f}")
    print(f"  5th (worst case):    ${results.percentiles['5th']:,.0f}")

    # Export to CSV
    print(f"\nExporting results to CSV...")
    output_dir = '../results/retirement/demo'
    results.export_to_csv(output_dir)
    print(f"  Exported to: {output_dir}")

    # Analysis
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    if results.success_rate >= 0.90:
        print("✓ VERY SAFE - High probability of success")
    elif results.success_rate >= 0.80:
        print("✓ RELATIVELY SAFE - Good chance of success")
    elif results.success_rate >= 0.70:
        print("⚠ MODERATE RISK - Consider reducing withdrawals")
    else:
        print("✗ HIGH RISK - Need to adjust plan")

    print(f"\nWith a {config.withdrawal_rate:.1%} withdrawal rate from a "
          f"{config.current_portfolio['SPY']:.0%}/{config.current_portfolio['AGG']:.0%} portfolio:")
    print(f"  • {results.success_rate:.0%} of simulations lasted the full {config.num_years} years")
    print(f"  • Median ending balance: ${results.median_final_value/1e6:.1f}M")

    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    visualizer = RetirementVisualizer(config)
    plot_dir = '../plots/retirement/demo'
    visualizer.plot_all(results, output_dir=plot_dir, show=False)  # Don't block

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"  CSV files:  {output_dir}")
    print(f"  Plots:      {plot_dir}")
    print("\nNext steps:")
    print("  1. Review the plots that just opened")
    print("  2. Check the CSV files for detailed data")
    print("  3. Try different scenarios by modifying the config")
    print("="*60)

if __name__ == '__main__':
    main()
