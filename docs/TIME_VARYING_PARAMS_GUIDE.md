# Time-Varying Parameters Guide for MCPathGenerator

## Overview

MCPathGenerator now supports **time-varying mean returns and covariance matrices** via pandas DataFrames. This enables:

- **Regime switching**: Different market regimes (bull/bear, low/high volatility)
- **Adaptive estimation**: Rolling or expanding window parameter updates
- **Structural changes**: Modeling long-term shifts in market dynamics
- **Scenario analysis**: What-if scenarios with changing conditions

## Quick Start

### Basic Example: Regime Switching

```python
from mc_path_generator import MCPathGenerator
import pandas as pd
import numpy as np

# 1. Create generator with baseline parameters
tickers = ['SPY', 'AGG']
mean_returns = np.array([0.10, 0.04])  # Used as fallback
cov_matrix = np.array([[0.04, 0.01], [0.01, 0.02]])

generator = MCPathGenerator(tickers, mean_returns, cov_matrix, seed=42)

# 2. Define time-varying mean returns
dates = pd.date_range('2025-01-01', periods=1000, freq='D')
mean_ts = pd.DataFrame({
    'SPY': [0.15 if i < 500 else 0.02 for i in range(1000)],  # Bull → Bear
    'AGG': [0.04 if i < 500 else 0.06 for i in range(1000)]   # Flight to safety
}, index=dates)

# 3. Set time-varying parameters
generator.set_time_varying_parameters(mean_ts)

# 4. Generate paths
paths = generator.generate_paths_time_varying(
    num_simulations=1000,
    start_date='2025-01-01',
    total_periods=1000,
    periods_per_year=252,
    frequency='D'
)

# paths.shape = (1000, 1000, 2)
```

## API Reference

### 1. `set_time_varying_parameters(mean_returns_ts, cov_matrices_ts=None)`

**Purpose**: Configure time-varying parameters for path generation.

**Parameters**:
- `mean_returns_ts`: `pd.DataFrame`
  - **Index**: DatetimeIndex (dates when parameters apply)
  - **Columns**: Asset tickers (must match `self.tickers`)
  - **Values**: Annualized mean returns

- `cov_matrices_ts`: `pd.DataFrame`, optional
  - **Index**: DatetimeIndex (must align with `mean_returns_ts`)
  - **Columns**: Flattened covariance elements (`'SPY_SPY'`, `'SPY_AGG'`, etc.)
  - **Values**: Annualized covariance values
  - If `None`, uses constant `self.cov_matrix`

**Example**:
```python
# Mean returns time series
mean_ts = pd.DataFrame({
    'SPY': [0.12, 0.10, 0.08, ...],  # Varying over time
    'AGG': [0.04, 0.04, 0.05, ...]
}, index=pd.date_range('2025-01-01', periods=N, freq='D'))

# Covariance time series (flattened)
cov_ts = pd.DataFrame({
    'SPY_SPY': [0.04, 0.05, 0.06, ...],  # SPY variance
    'SPY_AGG': [0.01, 0.01, 0.02, ...],  # SPY-AGG covariance
    'AGG_AGG': [0.02, 0.02, 0.02, ...]   # AGG variance
}, index=pd.date_range('2025-01-01', periods=N, freq='D'))

generator.set_time_varying_parameters(mean_ts, cov_ts)
```

### 2. `generate_paths_time_varying(num_simulations, start_date, total_periods, periods_per_year, frequency)`

**Purpose**: Generate MC paths using time-varying parameters.

**Parameters**:
- `num_simulations`: int - Number of paths to generate
- `start_date`: str - Starting date ('YYYY-MM-DD')
- `total_periods`: int - Number of periods to simulate
- `periods_per_year`: int - Frequency (1=annual, 12=monthly, 252=daily)
- `frequency`: str - Pandas frequency ('D', 'W', 'M', 'biweekly', etc.)

**Returns**: `np.ndarray` of shape `(num_simulations, total_periods, num_assets)`

**How it works**:
1. For each period, looks up parameters for that date (nearest neighbor)
2. Scales annual params to period frequency: `period_mean = annual_mean / periods_per_year`
3. Samples from multivariate normal with those parameters
4. Repeats for all periods (each period can have different distribution)

**Example**:
```python
paths = generator.generate_paths_time_varying(
    num_simulations=1000,
    start_date='2025-01-01',
    total_periods=252,      # 1 year of daily data
    periods_per_year=252,
    frequency='D'
)
```

## Use Cases

### Use Case 1: Regime Switching (2-State Model)

Model bull and bear markets with different return characteristics.

```python
# Define regimes
dates = pd.date_range('2025-01-01', periods=1000, freq='D')

# Bull: high returns, low vol
# Bear: low returns, high vol
mean_ts = pd.DataFrame({
    'SPY': [0.15 if i % 200 < 100 else 0.02 for i in range(1000)],
    'AGG': [0.04 if i % 200 < 100 else 0.06 for i in range(1000)]
}, index=dates)

cov_ts = pd.DataFrame({
    'SPY_SPY': [0.02 if i % 200 < 100 else 0.06 for i in range(1000)],
    'SPY_AGG': [0.005 if i % 200 < 100 else 0.02 for i in range(1000)],
    'AGG_AGG': [0.01 if i % 200 < 100 else 0.02 for i in range(1000)]
}, index=dates)

generator.set_time_varying_parameters(mean_ts, cov_ts)
paths = generator.generate_paths_time_varying(1000, '2025-01-01', 1000, 252, 'D')
```

### Use Case 2: Expanding Window Estimation

Adaptive parameter estimation using historical data.

```python
import pandas as pd

# Load historical returns
historical = pd.read_csv('historical_returns.csv', index_col=0, parse_dates=True)

# Calculate expanding window estimates
mean_estimates = []
dates_list = []
min_window = 60  # Minimum 60 days

for i in range(min_window, len(historical), 5):  # Update every 5 days
    window = historical.iloc[:i]
    mean_est = window.mean() * 252  # Annualize
    mean_estimates.append(mean_est)
    dates_list.append(historical.index[i])

mean_ts = pd.DataFrame(mean_estimates, index=dates_list)
mean_ts.columns = tickers

generator.set_time_varying_parameters(mean_ts)
paths = generator.generate_paths_time_varying(1000, dates_list[0], len(mean_ts), 252, 'D')
```

### Use Case 3: CAPE-Based Expected Returns

Use valuation metrics to adjust expected returns.

```python
# Load CAPE ratio time series
cape = pd.read_csv('shiller_cape.csv', index_col=0, parse_dates=True)

# Convert CAPE to expected return (Shiller formula or custom)
def cape_to_expected_return(cape_value):
    """Higher CAPE → lower expected returns"""
    return max(0.01, 0.20 - 0.005 * cape_value)

mean_ts = pd.DataFrame({
    'SPY': cape['CAPE'].apply(cape_to_expected_return),
    'AGG': [0.04] * len(cape)  # Constant for bonds
}, index=cape.index)

generator.set_time_varying_parameters(mean_ts)
paths = generator.generate_paths_time_varying(1000, cape.index[0], len(cape), 12, 'M')
```

### Use Case 4: Structural Break

Model a one-time permanent shift in market conditions.

```python
dates = pd.date_range('2025-01-01', periods=500, freq='D')
breakpoint = 250

mean_ts = pd.DataFrame({
    'SPY': [0.12 if i < breakpoint else 0.08 for i in range(500)],  # Permanent shift down
    'AGG': [0.04] * 500
}, index=dates)

# Correlation also changes (decoupling)
cov_ts = pd.DataFrame({
    'SPY_SPY': [0.04] * 500,
    'SPY_AGG': [0.01 if i < breakpoint else 0.005 for i in range(500)],  # Lower correlation
    'AGG_AGG': [0.02] * 500
}, index=dates)

generator.set_time_varying_parameters(mean_ts, cov_ts)
paths = generator.generate_paths_time_varying(1000, '2025-01-01', 500, 252, 'D')
```

## Creating Covariance Time Series

Covariance matrices must be **symmetric** and **positive definite**. The flattened format only needs upper triangular elements.

### Method 1: From Correlation + Volatility

```python
dates = pd.date_range('2025-01-01', periods=N, freq='D')

# Define time-varying volatilities
spy_vol = [0.15 if i < N//2 else 0.25 for i in range(N)]  # Vol regime change
agg_vol = [0.08] * N

# Define time-varying correlation
corr = [0.3 if i < N//2 else 0.6 for i in range(N)]  # Correlation increases

# Construct covariance elements
cov_data = []
for i in range(N):
    spy_var = spy_vol[i] ** 2
    agg_var = agg_vol[i] ** 2
    spy_agg_cov = corr[i] * spy_vol[i] * agg_vol[i]

    cov_data.append({
        'SPY_SPY': spy_var,
        'SPY_AGG': spy_agg_cov,
        'AGG_AGG': agg_var
    })

cov_ts = pd.DataFrame(cov_data, index=dates)
```

### Method 2: From Historical Rolling Windows

```python
# Calculate rolling covariance
window = 60
historical_returns = pd.read_csv('returns.csv', index_col=0, parse_dates=True)

cov_list = []
dates_list = []

for i in range(window, len(historical_returns)):
    window_data = historical_returns.iloc[i-window:i]
    cov_matrix = window_data.cov() * 252  # Annualize

    # Flatten
    cov_dict = {}
    for asset1 in tickers:
        for asset2 in tickers:
            cov_dict[f"{asset1}_{asset2}"] = cov_matrix.loc[asset1, asset2]

    cov_list.append(cov_dict)
    dates_list.append(historical_returns.index[i])

cov_ts = pd.DataFrame(cov_list, index=dates_list)
```

## Integration with Lifecycle Simulation

You can combine time-varying parameters with continuous lifecycle paths:

```python
# Option 1: Generate separate time-varying paths for each phase
# (currently requires manual combination - future enhancement)

# Accumulation phase
acc_mean_ts = pd.DataFrame(...)  # Your accumulation period params
generator.set_time_varying_parameters(acc_mean_ts)
acc_paths = generator.generate_paths_time_varying(
    1000, '2025-01-01', 234, 26, 'biweekly'
)

# Decumulation phase
dec_mean_ts = pd.DataFrame(...)  # Your decumulation period params
generator.set_time_varying_parameters(dec_mean_ts)
dec_paths = generator.generate_paths_time_varying(
    1000, '2035-01-01', 30, 1, 'Y'
)

# Run simulations
acc_values = run_accumulation_mc(initial_value, weights, acc_paths, ...)
dec_values, success = run_decumulation_mc(final_values, weights, dec_paths, ...)
```

**Note**: Future enhancement will add `generate_lifecycle_paths_time_varying()` to handle this automatically with continuous paths.

## Performance Considerations

Time-varying parameter generation is slower than constant parameters because:
- Each period requires a separate parameter lookup
- Each period requires a separate random sampling call

**Optimization tips**:
1. **Reduce parameter update frequency**: Update every N periods instead of every period
2. **Pre-compute parameters**: Calculate all parameters before the loop
3. **Use constant covariance**: Set `cov_matrices_ts=None` if only mean varies

**Example**: Update parameters every 10 days instead of daily
```python
# Instead of 1000 daily parameters
dates_sparse = pd.date_range('2025-01-01', periods=100, freq='10D')
mean_ts_sparse = pd.DataFrame({...}, index=dates_sparse)

# Automatic nearest-neighbor lookup fills gaps
paths = generator.generate_paths_time_varying(
    1000, '2025-01-01', 1000, 252, 'D'
)
# Days 0-9 use first parameter, 10-19 use second, etc.
```

## Testing and Validation

**Verify regime shifts worked**:
```python
# Generate paths with regime at period 500
paths = generator.generate_paths_time_varying(...)

# Compare before/after means
regime1_mean = paths[:, :500, 0].mean()
regime2_mean = paths[:, 500:, 0].mean()

print(f"Regime 1 mean: {regime1_mean}")
print(f"Regime 2 mean: {regime2_mean}")
# Should match your specified means (scaled to period)
```

**Verify volatility changes**:
```python
regime1_std = paths[:, :500, 0].std()
regime2_std = paths[:, 500:, 0].std()

print(f"Regime 1 volatility: {regime1_std}")
print(f"Regime 2 volatility: {regime2_std}")
# Should reflect your covariance specification
```

## Complete Example

See [test_time_varying_params.py](test_time_varying_params.py) for three comprehensive examples:
1. **Regime switching** (mean only)
2. **Volatility regimes** (mean + covariance)
3. **Adaptive estimation** (expanding window)

Run the test:
```bash
uv run python test_time_varying_params.py
```

Output includes:
- Validation that regime shifts work correctly
- Visualization showing parameter evolution
- Statistical verification of volatility changes

## Common Pitfalls

1. **Forgetting to annualize**: Parameters in DataFrames must be annualized, not period-level
   ```python
   # WRONG
   mean_ts = pd.DataFrame({'SPY': [0.0004] * N})  # Daily return

   # CORRECT
   mean_ts = pd.DataFrame({'SPY': [0.10] * N})  # Annual return (scales internally)
   ```

2. **Index mismatch**: Covariance index must match mean returns index
   ```python
   # WRONG
   mean_ts.index = pd.date_range('2025-01-01', periods=1000, freq='D')
   cov_ts.index = pd.date_range('2025-01-02', periods=1000, freq='D')  # Off by 1!

   # CORRECT
   dates = pd.date_range('2025-01-01', periods=1000, freq='D')
   mean_ts.index = dates
   cov_ts.index = dates
   ```

3. **Non-symmetric covariance**: Must have both SPY_AGG and AGG_SPY with same value
   ```python
   # CORRECT
   cov_ts['SPY_AGG'] = spy_agg_cov
   cov_ts['AGG_SPY'] = spy_agg_cov  # Symmetric
   ```

4. **Using wrong frequency**: Match `frequency` parameter with your date generation
   ```python
   # WRONG
   dates = pd.date_range(..., freq='M')  # Monthly dates
   paths = generator.generate_paths_time_varying(..., frequency='D')  # Daily frequency

   # CORRECT
   dates = pd.date_range(..., freq='M')
   paths = generator.generate_paths_time_varying(..., frequency='M')
   ```

## References

- **Implementation**: [mc_path_generator.py](mc_path_generator.py) (lines 253-436)
- **Test/Example**: [test_time_varying_params.py](test_time_varying_params.py)
- **Visualization**: `../plots/test/time_varying_params_demo.png`

---

**Status**: ✅ Complete and tested
**Added**: 2025-10-04
**Use Cases**: Regime switching, adaptive estimation, structural breaks, CAPE-based forecasting
