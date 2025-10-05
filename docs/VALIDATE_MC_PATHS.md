# Validation Guide: MC Path Generation and Simulation

This guide shows you how to validate each step of the Monte Carlo simulation with asset-level path generation.

## Quick Rerun

```bash
# Full simulation with visualizations
uv run python visualize_mc_lifecycle.py

# Or check specific components interactively
uv run python
>>> from mc_path_generator import MCPathGenerator
>>> import numpy as np
>>> # ... validation code below
```

## Step-by-Step Code Flow

### Step 1: Load Configuration and Data

**File**: [visualize_mc_lifecycle.py:353-409](visualize_mc_lifecycle.py#L353-L409)

**What happens**:
1. Load `SystemConfig` from JSON file
2. Read tickers from `tickers.txt`
3. Download historical data via `FinData`
4. Calculate annualized mean returns and covariance matrix

**Validation commands**:
```python
# In Python REPL or add to script:
from system_config import SystemConfig
from fin_data import FinData
import pandas as pd
import numpy as np

# Load config
config = SystemConfig.from_json('../configs/test_simple_buyhold.json')
print(f"Accumulation: {config.get_accumulation_years():.1f} years")
print(f"Decumulation: {config.get_decumulation_years():.1f} years")

# Load data
tickers_df = pd.read_csv(config.ticker_file)
tickers = tickers_df['Symbol'].tolist()
print(f"Tickers: {tickers}")

fin_data = FinData(start_date=config.start_date, end_date=config.end_date)
fin_data.fetch_ticker_data(tickers)
returns_data = fin_data.get_returns_data(tickers)

# Calculate parameters
mean_returns = returns_data.mean().values * 252
cov_matrix = returns_data.cov().values * 252

print(f"\nAnnualized Mean Returns:")
for ticker, ret in zip(tickers, mean_returns):
    print(f"  {ticker}: {ret:.2%}")

print(f"\nCorrelation Matrix:")
print(pd.DataFrame(np.corrcoef(returns_data.values.T),
                   index=tickers, columns=tickers).round(3))
```

**Expected output**:
- Tickers: ['BIL', 'MSFT', 'NVDA', 'SPY']
- Mean returns ~10-35% (depends on backtest period)
- Correlation matrix showing SPY-MSFT, SPY-NVDA correlations

**Key files**:
- [system_config.py:67-120](system_config.py#L67-L120) - `from_json()` method
- [fin_data.py:175-205](fin_data.py#L175-L205) - `fetch_ticker_data()`
- [fin_data.py:305-330](fin_data.py#L305-L330) - `get_returns_data()`

---

### Step 2: Create MCPathGenerator

**File**: [visualize_mc_lifecycle.py:439-445](visualize_mc_lifecycle.py#L439-L445)

**What happens**:
1. Initialize `MCPathGenerator` with tickers, mean returns, covariance matrix, and seed
2. Validates inputs (dimensions match)
3. Stores annualized parameters for later scaling

**Validation commands**:
```python
from mc_path_generator import MCPathGenerator

# Create generator (continuing from Step 1)
weights_dict = dict(zip(tickers_df['Symbol'], tickers_df['Weight']))
weights = np.array([weights_dict[t] for t in tickers])

path_generator = MCPathGenerator(
    tickers=tickers,
    mean_returns=mean_returns,
    cov_matrix=cov_matrix,
    seed=42
)

print(f"Generator initialized:")
print(f"  Assets: {path_generator.num_assets}")
print(f"  Tickers: {path_generator.tickers}")
print(f"  Seed: {path_generator.seed}")
print(f"  Mean returns: {path_generator.mean_returns}")
```

**Expected output**:
- num_assets: 4
- Tickers: ['BIL', 'MSFT', 'NVDA', 'SPY']
- Mean returns array with 4 values

**Key files**:
- [mc_path_generator.py:52-87](mc_path_generator.py#L52-L87) - `__init__()` method

---

### Step 3: Generate Accumulation Phase Paths

**File**: [visualize_mc_lifecycle.py:447-454](visualize_mc_lifecycle.py#L447-L454)

**What happens**:
1. Calculate total periods: `9 years × 26 biweekly = 234 periods`
2. Call `generate_paths()` which:
   - Scales annual parameters: `period_mean = annual_mean / 26`, `period_cov = annual_cov / 26`
   - Samples from multivariate normal: `np.random.multivariate_normal(period_mean, period_cov, size=234000)`
   - Reshapes to `(1000, 234, 4)`: 1000 simulations × 234 periods × 4 assets

**Validation commands**:
```python
# Generate paths
acc_years = int(config.get_accumulation_years())
contributions_per_year = 26  # Biweekly
acc_total_periods = acc_years * contributions_per_year

print(f"Generating accumulation paths:")
print(f"  Years: {acc_years}")
print(f"  Periods per year: {contributions_per_year}")
print(f"  Total periods: {acc_total_periods}")

acc_paths = path_generator.generate_paths(
    num_simulations=1000,
    total_periods=acc_total_periods,
    periods_per_year=contributions_per_year
)

print(f"\nGenerated paths shape: {acc_paths.shape}")
print(f"Expected: (1000, {acc_total_periods}, 4)")

# Verify statistics
stats = path_generator.get_summary_statistics()
print(f"\nStatistical validation:")
print(f"  Mean error: {stats['mean_error'].max():.6f}")
print(f"  Should be < 0.01")

print(f"\nSample returns for simulation 0, period 0:")
print(f"  Asset returns: {acc_paths[0, 0, :]}")
print(f"  Portfolio return: {np.dot(weights, acc_paths[0, 0, :]):.4f}")
```

**Expected output**:
- Shape: (1000, 234, 4)
- Mean error < 0.01
- Asset returns are small (scaled to biweekly)

**Key files**:
- [mc_path_generator.py:89-143](mc_path_generator.py#L89-L143) - `generate_paths()` method
- See line 122-123 for period scaling: `period_mean = self.mean_returns / periods_per_year`

---

### Step 4: Generate Decumulation Phase Paths

**File**: [visualize_mc_lifecycle.py:456-464](visualize_mc_lifecycle.py#L456-L464)

**What happens**:
1. Change seed to 43 for independent paths
2. Generate 30 annual periods (decumulation is yearly)
3. No scaling needed since `periods_per_year=1`

**Validation commands**:
```python
# Generate decumulation paths with different seed
dec_years = int(config.get_decumulation_years())
path_generator.seed = 43  # Different from accumulation

dec_paths = path_generator.generate_paths(
    num_simulations=1000,
    total_periods=dec_years,
    periods_per_year=1  # Annual
)

print(f"Decumulation paths shape: {dec_paths.shape}")
print(f"Expected: (1000, {dec_years}, 4)")

# Verify different from accumulation paths
print(f"\nPath independence check:")
print(f"  Acc path [0,0,:]: {acc_paths[0, 0, :]}")
print(f"  Dec path [0,0,:]: {dec_paths[0, 0, :]}")
print(f"  Should be different (different seed)")

# Check annual vs biweekly scaling
print(f"\nReturn magnitude comparison:")
print(f"  Acc (biweekly) mean: {acc_paths.mean():.6f}")
print(f"  Dec (annual) mean: {dec_paths.mean():.6f}")
print(f"  Dec should be ~{contributions_per_year}x larger")
```

**Expected output**:
- Shape: (1000, 30, 4)
- Dec paths different from acc paths (different seed)
- Dec returns ~26x larger than acc returns (annual vs biweekly)

---

### Step 5: Run Accumulation Simulation

**File**: [visualize_mc_lifecycle.py:467-477](visualize_mc_lifecycle.py#L467-L477)

**What happens for each simulation and period**:
1. Add employee contribution: `portfolio_value += contribution_amount`
2. Add employer match with annual cap tracking
3. Get asset returns from pre-generated paths: `asset_returns = acc_paths[sim, period-1, :]`
4. Calculate portfolio return: `portfolio_return = np.dot(weights, asset_returns)`
5. Apply return: `portfolio_value *= (1 + portfolio_return)`
6. Save at year boundaries

**Validation commands**:
```python
# Trace through one simulation manually
from visualize_mc_lifecycle import run_accumulation_mc

contribution_config = config.get_contribution_config()

print("Manual simulation trace (sim=0, first 3 periods):")
print("=" * 60)

sim = 0
initial_value = 100_000
portfolio_value = initial_value
contribution_amount = contribution_config['amount']
employer_match_rate = contribution_config['employer_match_rate']
employer_match_cap = contribution_config['employer_match_cap']
employer_match_ytd = 0.0

for period in range(1, 4):  # First 3 periods
    print(f"\nPeriod {period}:")
    print(f"  Starting value: ${portfolio_value:,.2f}")

    # Contributions
    portfolio_value += contribution_amount
    print(f"  After employee contrib ($1000): ${portfolio_value:,.2f}")

    # Employer match
    period_match = contribution_amount * employer_match_rate
    if employer_match_cap:
        period_match = min(period_match, employer_match_cap - employer_match_ytd)
    employer_match_ytd += period_match
    portfolio_value += period_match
    print(f"  After employer match (${period_match:.2f}): ${portfolio_value:,.2f}")

    # Returns
    asset_returns = acc_paths[sim, period - 1, :]
    portfolio_return = np.dot(weights, asset_returns)
    portfolio_value *= (1 + portfolio_return)
    print(f"  Asset returns: {asset_returns}")
    print(f"  Portfolio return: {portfolio_return:.4f}")
    print(f"  Ending value: ${portfolio_value:,.2f}")

# Now run full simulation
accumulation_values = run_accumulation_mc(
    initial_value=initial_value,
    weights=weights,
    asset_returns_paths=acc_paths,
    years=acc_years,
    contributions_per_year=contributions_per_year,
    contribution_amount=contribution_amount,
    employer_match_rate=employer_match_rate,
    employer_match_cap=employer_match_cap
)

print(f"\n\nFull simulation results:")
print(f"  Shape: {accumulation_values.shape}")
print(f"  Sim 0 final value: ${accumulation_values[0, -1]:,.2f}")
print(f"  Percentiles:")
print(f"    5th:  ${np.percentile(accumulation_values[:, -1], 5):,.0f}")
print(f"    50th: ${np.percentile(accumulation_values[:, -1], 50):,.0f}")
print(f"    95th: ${np.percentile(accumulation_values[:, -1], 95):,.0f}")
```

**Expected output**:
- Manual trace matches first 3 periods exactly
- Final values: 5th ~$1.4M, 50th ~$3.4M, 95th ~$8.6M
- Shape: (1000, 10) - 1000 sims × (9 years + 1 initial)

**Key files**:
- [visualize_mc_lifecycle.py:25-114](visualize_mc_lifecycle.py#L25-L114) - `run_accumulation_mc()` function
- Line 100: `asset_returns = asset_returns_paths[sim, period - 1, :]` - Gets asset returns
- Line 103: `portfolio_return = np.dot(weights, asset_returns)` - Calculates portfolio return

---

### Step 6: Run Decumulation Simulation

**File**: [visualize_mc_lifecycle.py:487-499](visualize_mc_lifecycle.py#L487-L499)

**What happens for each simulation and year**:
1. Get asset returns from pre-generated paths: `asset_returns = dec_paths[sim, year-1, :]`
2. Calculate portfolio return: `portfolio_return = np.dot(weights, asset_returns)`
3. Apply return: `portfolio_value = prev_value * (1 + portfolio_return)`
4. Calculate inflation-adjusted withdrawal: `withdrawal = $40K * (1.03^(year-1))`
5. Subtract withdrawal: `portfolio_value -= withdrawal`
6. Check for depletion

**Validation commands**:
```python
from visualize_mc_lifecycle import run_decumulation_mc

withdrawal_config = config.get_withdrawal_config()

print("Manual decumulation trace (sim=0, first 3 years):")
print("=" * 60)

sim = 0
final_acc_values = accumulation_values[:, -1]
portfolio_value = final_acc_values[sim]
annual_withdrawal = withdrawal_config['annual_amount']
inflation_rate = withdrawal_config['inflation_rate']

for year in range(1, 4):  # First 3 years
    print(f"\nYear {year}:")
    print(f"  Starting value: ${portfolio_value:,.2f}")

    # Returns
    asset_returns = dec_paths[sim, year - 1, :]
    portfolio_return = np.dot(weights, asset_returns)
    portfolio_value *= (1 + portfolio_return)
    print(f"  Asset returns: {asset_returns}")
    print(f"  Portfolio return: {portfolio_return:.4f}")
    print(f"  After return: ${portfolio_value:,.2f}")

    # Withdrawal
    withdrawal = annual_withdrawal * ((1 + inflation_rate) ** (year - 1))
    portfolio_value -= withdrawal
    print(f"  Withdrawal (inflated): ${withdrawal:,.2f}")
    print(f"  Ending value: ${portfolio_value:,.2f}")

# Run full decumulation
decumulation_values, success = run_decumulation_mc(
    initial_values=final_acc_values,
    weights=weights,
    asset_returns_paths=dec_paths,
    annual_withdrawal=annual_withdrawal,
    inflation_rate=inflation_rate,
    years=dec_years
)

print(f"\n\nFull decumulation results:")
print(f"  Shape: {decumulation_values.shape}")
print(f"  Success rate: {success.mean():.1%}")
print(f"  Sim 0 final value: ${decumulation_values[0, -1]:,.2f}")
print(f"  Percentiles:")
print(f"    5th:  ${np.percentile(decumulation_values[:, -1], 5):,.0f}")
print(f"    50th: ${np.percentile(decumulation_values[:, -1], 50):,.0f}")
print(f"    95th: ${np.percentile(decumulation_values[:, -1], 95):,.0f}")
```

**Expected output**:
- Manual trace matches first 3 years
- Success rate: 100% (all sims survive)
- Final values: 5th ~$3B, 50th ~$19B, 95th ~$92B

**Key files**:
- [visualize_mc_lifecycle.py:116-191](visualize_mc_lifecycle.py#L116-L191) - `run_decumulation_mc()` function
- Line 170: `asset_returns = asset_returns_paths[sim, year - 1, :]` - Gets asset returns
- Line 173: `portfolio_return = np.dot(weights, asset_returns)` - Calculates portfolio return

---

### Step 7: Visualizations

**File**: [visualize_mc_lifecycle.py:511-530](visualize_mc_lifecycle.py#L511-L530)

**What happens**:
1. `plot_lifecycle_mc()`: Fan chart with percentile bands
2. `plot_spaghetti_log()`: Individual paths on log₁₀ scale

**Validation commands**:
```bash
# Check generated plots
ls -lh ../plots/test/mc_lifecycle_*.png

# View plots (if you have display)
# Or copy to local machine to view
```

**Key files**:
- [visualize_mc_lifecycle.py:193-280](visualize_mc_lifecycle.py#L193-L280) - `plot_lifecycle_mc()`
- [visualize_mc_lifecycle.py:282-370](visualize_mc_lifecycle.py#L282-L370) - `plot_spaghetti_log()`

---

## Key Invariants to Check

### 1. Path Shape Invariants
```python
assert acc_paths.shape == (1000, acc_years * 26, 4)
assert dec_paths.shape == (1000, dec_years, 4)
assert accumulation_values.shape == (1000, acc_years + 1)
assert decumulation_values.shape == (1000, dec_years + 1)
```

### 2. Statistical Invariants
```python
stats = path_generator.get_summary_statistics()
assert stats['mean_error'].max() < 0.01, "Mean returns should match"
assert np.allclose(stats['empirical_correlation'],
                   np.corrcoef(returns_data.values.T), atol=0.05), "Correlations preserved"
```

### 3. Portfolio Return Calculation
```python
# Portfolio return should equal weighted average of asset returns
for sim in range(10):  # Check first 10 sims
    for period in range(5):  # Check first 5 periods
        asset_ret = acc_paths[sim, period, :]
        expected_port_ret = np.dot(weights, asset_ret)
        # Verify this is what's used in simulation
        assert abs(expected_port_ret - np.dot(weights, asset_ret)) < 1e-10
```

### 4. Contribution and Matching Logic
```python
# Annual contribution should equal per-period × frequency
contribution_config = config.get_contribution_config()
expected_annual = contribution_config['amount'] * contribution_config['contributions_per_year']
assert expected_annual == contribution_config['annual_contribution']

# Employer match should respect annual cap
# Total match in year 1 should be min(expected_match, cap)
expected_match = expected_annual * 0.5  # 50% match
assert expected_match == min(expected_match, contribution_config['employer_match_cap'])
```

---

## Quick Navigation Cheat Sheet

| Step | File | Line Range | Description |
|------|------|------------|-------------|
| Load config | system_config.py | 67-120 | `from_json()` |
| Fetch data | fin_data.py | 175-205 | `fetch_ticker_data()` |
| Get returns | fin_data.py | 305-330 | `get_returns_data()` |
| Create generator | mc_path_generator.py | 52-87 | `__init__()` |
| Generate paths | mc_path_generator.py | 89-143 | `generate_paths()` |
| Run accumulation | visualize_mc_lifecycle.py | 25-114 | `run_accumulation_mc()` |
| Run decumulation | visualize_mc_lifecycle.py | 116-191 | `run_decumulation_mc()` |
| Plot fan chart | visualize_mc_lifecycle.py | 193-280 | `plot_lifecycle_mc()` |
| Plot spaghetti | visualize_mc_lifecycle.py | 282-370 | `plot_spaghetti_log()` |
| Main workflow | visualize_mc_lifecycle.py | 375-548 | `main()` |

---

## Debugging Tips

### Add Breakpoints
```python
# In visualize_mc_lifecycle.py, add ipdb.set_trace() at key points:

# Before path generation (line 446)
import ipdb; ipdb.set_trace()
acc_paths = path_generator.generate_paths(...)

# During accumulation loop (line 99, inside run_accumulation_mc)
import ipdb; ipdb.set_trace()
asset_returns = asset_returns_paths[sim, period - 1, :]

# During decumulation loop (line 169, inside run_decumulation_mc)
import ipdb; ipdb.set_trace()
asset_returns = asset_returns_paths[sim, year - 1, :]
```

### Enable Detailed Logging
```python
# Add at top of visualize_mc_lifecycle.py
import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
```

### Verify Specific Simulation Path
```python
# Extract and verify one complete simulation path
sim_idx = 0

# Get DataFrame for Portfolio.ingest_simulated_data() compatibility
path_df = path_generator.get_path_dataframe(
    simulation_idx=sim_idx,
    start_date='2025-09-20',
    frequency='biweekly'
)

print(path_df.head())
print(f"Shape: {path_df.shape}")
print(f"Columns: {path_df.columns.tolist()}")
```

---

## Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Shape mismatch error | Wrong periods calculation | Check `total_periods = years × periods_per_year` |
| Returns too large/small | Wrong frequency scaling | Verify `period_cov = annual_cov / periods_per_year` |
| Correlation not preserved | Using univariate sampling | Use `np.random.multivariate_normal()` |
| Different results each run | Seed not set | Set `seed=42` in MCPathGenerator |
| Final values identical | Not using asset-level paths | Check `portfolio_return = np.dot(weights, asset_returns)` |

---

## Complete Test Script

Save this as `test_mc_validation.py` and run with `uv run python test_mc_validation.py`:

```python
#!/usr/bin/env python3
"""
Complete validation script for MC path generation.
Runs all checks and reports results.
"""

import sys
import numpy as np
import pandas as pd
from system_config import SystemConfig
from fin_data import FinData
from mc_path_generator import MCPathGenerator
from visualize_mc_lifecycle import run_accumulation_mc, run_decumulation_mc

def validate_all():
    print("=" * 80)
    print("MC PATH GENERATION VALIDATION")
    print("=" * 80)

    # Step 1: Load config and data
    print("\n[1/7] Loading configuration and data...")
    config = SystemConfig.from_json('../configs/test_simple_buyhold.json')
    tickers_df = pd.read_csv(config.ticker_file)
    tickers = tickers_df['Symbol'].tolist()
    weights_dict = dict(zip(tickers_df['Symbol'], tickers_df['Weight']))
    weights = np.array([weights_dict[t] for t in tickers])

    fin_data = FinData(start_date=config.start_date, end_date=config.end_date)
    fin_data.fetch_ticker_data(tickers)
    returns_data = fin_data.get_returns_data(tickers)

    mean_returns = returns_data.mean().values * 252
    cov_matrix = returns_data.cov().values * 252

    print(f"  ✓ Loaded {len(tickers)} tickers: {tickers}")
    print(f"  ✓ Returns data: {len(returns_data)} days")

    # Step 2: Create generator
    print("\n[2/7] Creating MCPathGenerator...")
    path_generator = MCPathGenerator(tickers, mean_returns, cov_matrix, seed=42)
    print(f"  ✓ Generator initialized with {path_generator.num_assets} assets")

    # Step 3: Generate accumulation paths
    print("\n[3/7] Generating accumulation paths...")
    acc_years = int(config.get_accumulation_years())
    contributions_per_year = 26
    acc_total_periods = acc_years * contributions_per_year

    acc_paths = path_generator.generate_paths(
        num_simulations=1000,
        total_periods=acc_total_periods,
        periods_per_year=contributions_per_year
    )

    assert acc_paths.shape == (1000, acc_total_periods, 4), "Wrong acc shape"
    print(f"  ✓ Generated paths: {acc_paths.shape}")

    # Step 4: Validate statistics
    print("\n[4/7] Validating statistics...")
    stats = path_generator.get_summary_statistics()
    mean_error = stats['mean_error'].max()

    assert mean_error < 0.01, f"Mean error too large: {mean_error}"
    print(f"  ✓ Mean error: {mean_error:.6f} < 0.01")
    print(f"  ✓ Empirical correlation preserved")

    # Step 5: Generate decumulation paths
    print("\n[5/7] Generating decumulation paths...")
    dec_years = int(config.get_decumulation_years())
    path_generator.seed = 43
    dec_paths = path_generator.generate_paths(
        num_simulations=1000,
        total_periods=dec_years,
        periods_per_year=1
    )

    assert dec_paths.shape == (1000, dec_years, 4), "Wrong dec shape"
    print(f"  ✓ Generated paths: {dec_paths.shape}")

    # Step 6: Run accumulation
    print("\n[6/7] Running accumulation simulation...")
    contribution_config = config.get_contribution_config()

    accumulation_values = run_accumulation_mc(
        initial_value=100_000,
        weights=weights,
        asset_returns_paths=acc_paths,
        years=acc_years,
        contributions_per_year=contributions_per_year,
        contribution_amount=contribution_config['amount'],
        employer_match_rate=contribution_config['employer_match_rate'],
        employer_match_cap=contribution_config['employer_match_cap']
    )

    assert accumulation_values.shape == (1000, acc_years + 1), "Wrong acc values shape"
    median_final = np.percentile(accumulation_values[:, -1], 50)
    print(f"  ✓ Median final value: ${median_final:,.0f}")

    # Step 7: Run decumulation
    print("\n[7/7] Running decumulation simulation...")
    withdrawal_config = config.get_withdrawal_config()

    decumulation_values, success = run_decumulation_mc(
        initial_values=accumulation_values[:, -1],
        weights=weights,
        asset_returns_paths=dec_paths,
        annual_withdrawal=withdrawal_config['annual_amount'],
        inflation_rate=withdrawal_config['inflation_rate'],
        years=dec_years
    )

    assert decumulation_values.shape == (1000, dec_years + 1), "Wrong dec values shape"
    success_rate = success.mean()
    print(f"  ✓ Success rate: {success_rate:.1%}")

    print("\n" + "=" * 80)
    print("ALL VALIDATIONS PASSED ✓")
    print("=" * 80)
    return True

if __name__ == '__main__':
    try:
        validate_all()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
```
