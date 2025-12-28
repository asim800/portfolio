# Time-Varying Parameters - Quick Start Guide

## Overview

MCPathGenerator supports time-varying mean returns and covariance matrices for Monte Carlo simulations. This enables modeling:

1. **Regime shifts** (e.g., aggressive → conservative at retirement)
2. **Rolling window estimation** (parameters updated at rebalancing frequency)
3. **Market cycles** (bull → bear transitions)

## Format: How to Pass Time-Varying Parameters

### Mean Returns (DataFrame)

```python
import pandas as pd

# Create dates for your simulation
dates = pd.date_range(start='2025-01-01', periods=1014, freq='2W')  # Biweekly

# Create time-varying mean returns
mean_ts = pd.DataFrame(index=dates)
mean_ts['SPY'] = [0.10 if i < 234 else 0.07 for i in range(1014)]  # 10% → 7% at period 234
mean_ts['AGG'] = [0.04] * 1014  # Constant 4%
```

**Requirements:**
- Index: `pd.DatetimeIndex`
- Columns: Asset tickers (must match generator tickers)
- Values: **Annual** returns (will be automatically scaled to period returns)

### Covariance Matrices (DataFrame with list-of-arrays)

**RECOMMENDED FORMAT:**

```python
# Create time-varying covariance matrices
cov_data = []
for i in range(total_periods):
    if i < retirement_period:
        # Accumulation: Higher volatility
        spy_vol = 0.20  # 20% annual volatility
        agg_vol = 0.05
        correlation = 0.2
    else:
        # Decumulation: Lower volatility
        spy_vol = 0.18  # 18% annual volatility
        agg_vol = 0.05
        correlation = 0.3

    # Build covariance matrix
    spy_var = spy_vol ** 2
    agg_var = agg_vol ** 2
    spy_agg_cov = correlation * spy_vol * agg_vol

    cov_matrix = np.array([
        [spy_var, spy_agg_cov],
        [spy_agg_cov, agg_var]
    ])

    cov_data.append({'cov_matrix': cov_matrix})

cov_ts = pd.DataFrame(cov_data, index=dates)
```

**Requirements:**
- Index: `pd.DatetimeIndex` (same as mean_ts)
- Single column: `'cov_matrix'` containing `np.ndarray` objects
- Values: **Annual** covariance matrices (will be automatically scaled to period covariance)

### Generate Paths

```python
from mc_path_generator import MCPathGenerator

generator = MCPathGenerator(tickers, mean_returns_default, cov_matrix_default, seed=42)

paths = generator.generate_paths(
    num_simulations=500,
    total_periods=1014,
    periods_per_year=26,
    start_date='2025-01-01',  # REQUIRED for time-varying
    frequency='2W',           # Pandas frequency string
    mean_returns=mean_ts,     # DataFrame triggers time-varying mode
    cov_matrices=cov_ts       # DataFrame with list-of-arrays
)
```

**Result:** `paths` shape is `(500, 1014, 2)` - 500 simulations × 1014 periods × 2 assets

## Scenario 1: Regime Shift at Retirement

**Use case:** Switch from aggressive (80/20) to conservative (40/60) at retirement.

```python
# Timeline
acc_years = 9
dec_years = 30
periods_per_year = 26
total_periods = (acc_years + dec_years) * periods_per_year
retirement_period = acc_years * periods_per_year  # Period 234

dates = pd.date_range(start='2025-01-01', periods=total_periods, freq='2W')

# Time-varying mean
mean_ts = pd.DataFrame(index=dates)
mean_ts['SPY'] = [0.10 if i < retirement_period else 0.07 for i in range(total_periods)]
mean_ts['AGG'] = [0.04] * total_periods

# Time-varying covariance
cov_data = []
for i in range(total_periods):
    if i < retirement_period:
        spy_vol, agg_vol, corr = 0.20, 0.05, 0.2  # Aggressive
    else:
        spy_vol, agg_vol, corr = 0.18, 0.05, 0.3  # Conservative

    spy_var = spy_vol ** 2
    agg_var = agg_vol ** 2
    cov = corr * spy_vol * agg_vol

    cov_data.append({'cov_matrix': np.array([[spy_var, cov], [cov, agg_var]])})

cov_ts = pd.DataFrame(cov_data, index=dates)

# Generate
paths = generator.generate_paths(
    num_simulations=500,
    total_periods=total_periods,
    periods_per_year=26,
    start_date='2025-01-01',
    frequency='2W',
    mean_returns=mean_ts,
    cov_matrices=cov_ts
)
```

**Visualization:** See `../plots/test/retirement_regime_shift.png`

## Scenario 2: Rolling Window Estimation

**Use case:** Update mean/cov every 6 months using trailing 252-day window.

```python
from fin_data import FinData

# Get historical data
fin_data = FinData(start_date='2020-01-01', end_date='2024-12-31')
fin_data.fetch_ticker_data(['SPY', 'AGG'])
returns_data = fin_data.get_returns_data(['SPY', 'AGG'])

# Simulation setup
total_periods = 130  # 5 years × 26 periods/year
window_size = 252    # 1-year rolling window
dates_sim = pd.date_range(start='2025-01-01', periods=total_periods, freq='2W')

# Calculate rolling estimates
mean_estimates = []
cov_estimates = []

for i in range(total_periods):
    # Extract historical window (cycling through data)
    historical_idx = (i * 14) % (len(returns_data) - window_size)
    window_data = returns_data.iloc[historical_idx:historical_idx + window_size]

    # Calculate estimates
    mean_annual = window_data.mean() * 252  # Annualize
    cov_annual = window_data.cov() * 252    # Annualize

    mean_estimates.append(mean_annual)
    cov_estimates.append({'cov_matrix': cov_annual.values})

# Create DataFrames
mean_ts = pd.DataFrame(mean_estimates, index=dates_sim, columns=['SPY', 'AGG'])
cov_ts = pd.DataFrame(cov_estimates, index=dates_sim)

# Generate
paths = generator.generate_paths(
    num_simulations=500,
    total_periods=total_periods,
    periods_per_year=26,
    start_date='2025-01-01',
    frequency='2W',
    mean_returns=mean_ts,
    cov_matrices=cov_ts
)
```

**Visualization:** See `../plots/test/rolling_window_estimation.png`

## Test Script

**Run comprehensive test:**
```bash
uv run python test_retirement_time_varying.py
```

**Output:**
- Console validation with statistics
- `../plots/test/retirement_regime_shift.png` - 4-panel visualization
- `../plots/test/rolling_window_estimation.png` - 4-panel visualization

**Plots include:**
1. Sample paths with regime markers
2. Time-varying mean returns
3. Return distributions by phase
4. Time-varying volatility

## Key Points

✅ **Annual values:** Pass annual returns and covariance - automatic scaling to periods
✅ **DatetimeIndex required:** Both mean_ts and cov_ts must have matching DatetimeIndex
✅ **List-of-arrays format:** Use `{'cov_matrix': np.ndarray}` for each row
✅ **Automatic detection:** Passing DataFrames automatically triggers time-varying mode
✅ **Nearest neighbor matching:** Dates in simulation are matched to nearest date in your DataFrames

## Common Patterns

### Pattern 1: Step change at specific date
```python
mean_ts['SPY'] = [0.10 if date < retirement_date else 0.07 for date in dates]
```

### Pattern 2: Gradual transition
```python
# Gradually reduce from 10% to 7% over 5 years
transition_periods = 5 * 26
mean_ts['SPY'] = [
    0.10 - (0.03 * min(i, transition_periods) / transition_periods)
    for i in range(total_periods)
]
```

### Pattern 3: Rolling window
```python
for i, date in enumerate(dates):
    window = historical_data.loc[date - pd.Timedelta(days=252):date]
    mean_estimates.append(window.mean() * 252)
```

### Pattern 4: CAPE-based expected returns
```python
# Lower expected returns when CAPE is high
for date in dates:
    cape_ratio = get_cape(date)
    expected_return = base_return * (median_cape / cape_ratio)
    mean_estimates.append(expected_return)
```

## See Also

- Full examples: `test_retirement_time_varying.py`
- Original guide: `TIME_VARYING_PARAMS_GUIDE.md`
- Basic time-varying: `test_time_varying_params.py`
