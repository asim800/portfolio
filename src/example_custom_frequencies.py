#!/usr/bin/env python3
"""
Example: Using custom rebalancing frequencies

Demonstrates the flexibility of pandas native frequency support:
- Standard frequencies: 'ME', 'Q', 'W'
- Custom multiples: '2W', '21D', '3ME'
- Business day frequencies: 'BM', 'BQ'
"""

import pandas as pd
import numpy as np
from period_manager import PeriodManager
from config import RebalancingConfig

# Generate sample returns data
dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
returns = pd.DataFrame(
    np.random.randn(len(dates), 5) * 0.01,
    index=dates,
    columns=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA']
)

print("=" * 80)
print("CUSTOM FREQUENCY EXAMPLES")
print("=" * 80)

# Example 1: Standard monthly rebalancing (month-end)
print("\n1. Monthly Rebalancing (ME)")
print("-" * 80)
pm_monthly = PeriodManager(returns, frequency='ME')
print(f"Number of periods: {pm_monthly.num_periods}")
print(f"First 3 periods:\n{pm_monthly.get_periods_summary().head(3)}")

# Example 2: Bi-weekly rebalancing
print("\n2. Bi-weekly Rebalancing (2W)")
print("-" * 80)
pm_biweekly = PeriodManager(returns, frequency='2W')
print(f"Number of periods: {pm_biweekly.num_periods}")
print(f"First 3 periods:\n{pm_biweekly.get_periods_summary().head(3)}")

# Example 3: Every 21 calendar days
print("\n3. Every 21 Days (21D)")
print("-" * 80)
pm_21days = PeriodManager(returns, frequency='21D')
print(f"Number of periods: {pm_21days.num_periods}")
print(f"First 3 periods:\n{pm_21days.get_periods_summary().head(3)}")

# Example 4: Quarterly rebalancing using custom multiple
print("\n4. Quarterly Rebalancing (3ME)")
print("-" * 80)
pm_quarterly = PeriodManager(returns, frequency='3ME')
print(f"Number of periods: {pm_quarterly.num_periods}")
print(f"All periods:\n{pm_quarterly.get_periods_summary()}")

# Example 5: Business month-end (excludes weekends)
print("\n5. Business Month-End (BME)")
print("-" * 80)
pm_bm = PeriodManager(returns, frequency='BME')
print(f"Number of periods: {pm_bm.num_periods}")
print(f"First 3 periods:\n{pm_bm.get_periods_summary().head(3)}")

# Example 6: Semi-annual rebalancing
print("\n6. Semi-Annual Rebalancing (6ME)")
print("-" * 80)
pm_semiannual = PeriodManager(returns, frequency='6ME')
print(f"Number of periods: {pm_semiannual.num_periods}")
print(f"All periods:\n{pm_semiannual.get_periods_summary()}")

# Example 7: Using custom frequencies in RebalancingConfig
print("\n7. Custom Frequencies in Config")
print("-" * 80)
config = RebalancingConfig(
    rebalancing_frequency='2W',  # Bi-weekly for optimized portfolios
    rebalancing_strategy_frequencies={
        'target_weight': 'ME',   # Monthly for target weight
        'equal_weight': '21D',   # Every 21 days for equal weight
    }
)
print(f"Main rebalancing frequency: {config.get_rebalancing_frequency()}")
print(f"Target weight frequency: {config.get_rebalancing_period('target_weight')}")
print(f"Equal weight frequency: {config.get_rebalancing_period('equal_weight')}")

print("\n" + "=" * 80)
print("✓ Custom frequencies work seamlessly with pandas resample()!")
print("=" * 80)
