# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview - What This System Does

This is a comprehensive **portfolio lifecycle simulation and analysis system** that helps answer critical financial planning questions:

**For investors**: "If I invest $100K today with $1K biweekly contributions for 9 years, then retire and withdraw $40K/year for 30 years, what's the probability my money lasts?"

**For portfolio managers**: "How does a 60/40 SPY/AGG portfolio compare to optimized rebalancing strategies over different market regimes?"

### Three Main Capabilities

1. **Historical Backtesting** (Legacy System)
   - Tests portfolio strategies on actual historical data (e.g., 2024-2025)
   - Compares vanilla mean-variance optimization vs. robust optimization vs. buy-and-hold
   - Generates performance metrics (Sharpe ratio, drawdown, beta) and visualizations
   - **Entry point**: `uv run main.py`

2. **Monte Carlo Lifecycle Simulation** (Current Focus)
   - Simulates thousands of possible future market scenarios
   - Models complete lifecycle: **accumulation** (working years with contributions) → **decumulation** (retirement with withdrawals)
   - Accounts for employer matching, inflation, sequence-of-returns risk
   - Answers: "What's the probability of success?" and "What could go wrong?"
   - **Entry point**: `uv run python visualize_mc_lifecycle.py`

3. **Advanced Monte Carlo Features** (Recent Additions)
   - **Time-varying parameters**: Model regime changes (bull→bear markets), volatility cycles, adaptive strategies
   - **Continuous lifecycle paths**: Ensures accumulation and retirement phases use the same market scenario (no artificial discontinuity)
   - **Asset-level correlation**: Preserves correlation structure between assets (e.g., SPY and bonds)
   - **Entry point**: See test scripts and ../docs/TIME_VARYING_PARAMS_GUIDE.md

### Technology Stack

- **Python 3.12** with `uv` package manager (fast, modern dependency management)
- **cvxpy**: Convex optimization for portfolio weight optimization
- **yfinance**: Yahoo Finance data download (with pickle caching to avoid rate limits)
- **pandas/numpy**: Data manipulation and vectorized numerical computing
- **matplotlib**: Professional financial visualizations (fan charts, percentile bands)
- **pytest**: Comprehensive test suite (61+ tests for Monte Carlo system alone)

## Getting Started for New Engineers

### TL;DR - What You Need to Know

**Purpose**: Monte Carlo lifecycle simulation for retirement planning (accumulation + decumulation with contributions/withdrawals)

**Key Files**:
- **Entry point**: `visualize_mc_lifecycle.py` - Run this to see everything in action
- **Path generation**: `mc_path_generator.py:MCPathGenerator` - Creates correlated asset return paths
- **Configuration**: `test_simple_buyhold.json` + `system_config.py:SystemConfig` - All parameters
- **Validation**: Run `uv run python test_mc_validation.py` to verify system integrity

**Critical Concepts**:
1. **Asset-level paths** (not portfolio-level) - Enables comparing multiple portfolios on same market scenarios
2. **Continuous lifecycle paths** - Accumulation and decumulation use ONE continuous random sequence (no gap)
3. **Period-level simulation** - Simulate at contribution frequency (biweekly/monthly), not annual
4. **Time-varying parameters** - Mean/covariance can change over time (regime switching, adaptive estimation)

**Common Gotchas**:
- Covariance scales **linearly** with time: `cov_period = cov_annual / periods_per_year` (NOT sqrt!)
- Compound sub-annual returns: `(1+r1)*(1+r2)*...-1` (NOT `r1+r2+...`)
- Employer match cap is **annual**, reset each year
- Always use continuous paths via `generate_lifecycle_paths()` (NOT `generate_paths()` twice)

**Output**: Percentile fan charts showing range of possible outcomes, success rate, final value distributions

**Tests**: 61+ passing tests validate all components. Run `uv run python test_mc_validation.py` to verify.

### Quick Start (5 minutes)

1. **Setup environment**:
   ```bash
   cd /path/to/fin/port/src
   uv sync  # Installs all dependencies in .venv/
   ```

2. **Run a simple test** to verify everything works:
   ```bash
   uv run python test_mc_validation.py
   # Should output: "ALL VALIDATIONS PASSED ✓"
   ```

3. **Run a lifecycle simulation**:
   ```bash
   uv run python visualize_mc_lifecycle.py
   # Generates plots in ../plots/test/
   # Shows: accumulation (9 years) → decumulation (30 years)
   # Success rate, percentiles, final portfolio values
   ```

4. **Examine output**:
   - Open `../plots/test/mc_lifecycle_fan_chart.png` - shows percentile bands over time
   - Open `../plots/test/mc_lifecycle_spaghetti_log.png` - shows individual simulation paths

### Understanding the Codebase (15 minutes)

**Start here** to understand the system architecture:

1. **Read the configuration** (easiest entry point):
   - Open [test_simple_buyhold.json](test_simple_buyhold.json) - see what parameters control the simulation
   - Key fields: `start_date`, `retirement_date`, `contribution_amount`, `annual_withdrawal_amount`

2. **Trace a simulation** (follow the data flow):
   - Start: [visualize_mc_lifecycle.py:main()](visualize_mc_lifecycle.py) - entry point
   - Load config: `SystemConfig.from_json()` - reads JSON and validates parameters
   - Generate paths: `MCPathGenerator.generate_lifecycle_paths()` - creates random asset return paths
   - Run simulation: `run_accumulation_mc()` → `run_decumulation_mc()` - dollar values over time
   - Visualize: `plot_lifecycle_mc()` - create fan charts

3. **Read the validation guide** for step-by-step walkthrough:
   - [../docs/VALIDATE_MC_PATHS.md](../docs/VALIDATE_MC_PATHS.md) - explains each step with validation commands
   - [../docs/MC_QUICK_REFERENCE.md](../docs/MC_QUICK_REFERENCE.md) - cheat sheet for common tasks

### Key Architectural Decisions (Why Things Are This Way)

**1. Why asset-level paths instead of portfolio-level?**
- **Old approach**: Generated portfolio returns directly (single random stream)
- **New approach**: Generate correlated asset returns, then calculate portfolio return = dot(weights, asset_returns)
- **Why**: Enables comparing different portfolios (60/40 vs. 70/30) on identical market scenarios
- **File**: [mc_path_generator.py](mc_path_generator.py)

**2. Why continuous lifecycle paths?**
- **Problem**: Original implementation used independent random seeds for accumulation and decumulation
- **Issue**: Retirement could start in a bull market even if accumulation ended in a bear market (unrealistic)
- **Solution**: Generate ONE continuous path, split into accumulation (periods 0-233) and decumulation (periods 234-1013)
- **Validation**: [test_continuous_paths.py](test_continuous_paths.py) confirms zero-error continuity

**3. Why period-level simulation instead of annual?**
- **Problem**: Annual simulation incorrectly models dollar-cost averaging with contributions
- **Issue**: Contributing $26K once per year ≠ contributing $1K biweekly (timing matters!)
- **Solution**: Simulate at contribution frequency (biweekly = 26 periods/year)
- **Key formula**: `period_mean = annual_mean / periods_per_year`, `period_cov = annual_cov / periods_per_year`
- **Note**: This was a critical bug fix - see commit history

**4. Why time-varying parameters?**
- **Limitation**: Constant mean/cov assumes markets never change (unrealistic for 30+ year simulations)
- **Solution**: Allow parameters to vary by date via pandas DataFrame
- **Use cases**: Bull→bear regime shifts, CAPE-based expected returns, adaptive estimation
- **File**: [mc_path_generator.py:set_time_varying_parameters()](mc_path_generator.py)
- **Guide**: [../docs/TIME_VARYING_PARAMS_GUIDE.md](../docs/TIME_VARYING_PARAMS_GUIDE.md)

**5. Why flattened covariance matrices in DataFrames?**
- **Problem**: Can't store 2D numpy arrays directly in DataFrame columns
- **Solution**: Flatten to columns like `SPY_SPY`, `SPY_AGG`, `AGG_AGG` (symmetric matrix)
- **Reconstruction**: See `set_time_varying_parameters()` - rebuilds symmetric matrix from flattened elements

### Common Gotchas and Edge Cases

**1. Frequency scaling is NOT sqrt scaling for covariance**
```python
# ❌ WRONG (common mistake)
period_cov = annual_cov / np.sqrt(periods_per_year)

# ✅ CORRECT
period_cov = annual_cov / periods_per_year
```
Covariance scales linearly with time, not by square root.

**2. Annual returns from sub-annual periods require compounding**
```python
# ❌ WRONG
annual_return = np.sum(biweekly_returns)  # Additive (incorrect)

# ✅ CORRECT
annual_return = np.prod(1 + biweekly_returns) - 1  # Multiplicative
```

**3. Employer match cap is annual, not per-period**
```python
# Track year-to-date match, reset each January
if employer_match_cap is not None:
    period_match = min(period_match, employer_match_cap - ytd_match)
```
See `run_accumulation_mc()` for implementation.

**4. Nearest-neighbor date matching for time-varying parameters**
```python
# If parameter dates don't exactly match simulation dates, use nearest
nearest_idx = time_varying_mean.index.get_indexer([date], method='nearest')[0]
```
This is intentional - allows flexible parameter specification.

**5. Pickle cache files can become stale**
```python
# Located in ../data/
# If yfinance data structure changes, delete cache files and re-download
rm ../data/*.pkl
```

### Data Flow Diagram

**Monte Carlo Lifecycle Simulation - Complete Flow**:
```
1. Configuration
   test_simple_buyhold.json
   ↓
   SystemConfig.from_json()
   → Validates all parameters
   → Returns: backtest dates, retirement date, contribution/withdrawal config

2. Historical Data (for parameter estimation)
   FinData('2024-01-01', '2025-09-19')
   ↓
   Downloads SPY, AGG, NVDA, GLD from Yahoo Finance
   ↓
   Calculates daily returns
   ↓
   Estimates: mean_returns (annualized), cov_matrix (annualized)

3. Path Generation
   MCPathGenerator(tickers, mean_returns, cov_matrix, seed=42)
   ↓
   generate_lifecycle_paths(num_sims=1000, acc_years=9, acc_periods_per_year=26, dec_years=30)
   ↓
   Returns:
   - acc_paths: (1000, 234, 4) = 1000 simulations × 234 biweekly periods × 4 assets
   - dec_paths: (1000, 30, 4) = 1000 simulations × 30 annual periods × 4 assets
   Note: Paths are CONTINUOUS (no gap between accumulation and decumulation)

4. Accumulation Simulation
   run_accumulation_mc(
     initial_value=$100K,
     weights=[0.25, 0.25, 0.25, 0.25],  # Equal-weight portfolio
     asset_returns_paths=acc_paths,
     contribution_amount=$1000,
     employer_match_rate=0.5,
     employer_match_cap=$10K
   )
   ↓
   For each period (0→233):
     - portfolio_return = dot(weights, asset_returns[period])
     - portfolio_value *= (1 + portfolio_return)
     - portfolio_value += contribution + employer_match
     - Track YTD match for annual cap
   ↓
   Returns: acc_values (1000, 234) = portfolio value at each period for each simulation

5. Decumulation Simulation
   run_decumulation_mc(
     initial_values=acc_values[:, -1],  # Final accumulation values (1000,)
     weights=[0.25, 0.25, 0.25, 0.25],
     asset_returns_paths=dec_paths,
     annual_withdrawal=$40K,
     inflation_rate=0.03
   )
   ↓
   For each year (0→29):
     - portfolio_return = dot(weights, asset_returns[year])
     - portfolio_value *= (1 + portfolio_return)
     - withdrawal = $40K × (1.03)^year  # Inflation-adjusted
     - portfolio_value -= withdrawal
     - If portfolio_value <= 0: FAILED
   ↓
   Returns:
   - dec_values (1000, 30) = portfolio value each year for each simulation
   - success (1000,) = boolean array (True = survived 30 years)

6. Analysis & Visualization
   - Calculate percentiles (5th, 25th, 50th, 75th, 95th) for each period
   - Success rate = % of simulations with portfolio > 0 at end
   - Generate fan chart (percentile bands over time)
   - Generate spaghetti plot (individual paths on log scale)
   - Export to CSV (optional)
```

### Troubleshooting Guide

**Problem**: `test_mc_validation.py` fails with "Mean error too large"
- **Cause**: Random sampling variation (should be rare with 10000 simulations)
- **Fix**: Re-run the test (different random seed may pass)
- **Check**: If consistently fails, verify `periods_per_year` scaling is correct

**Problem**: Discontinuity visible in fan chart at accumulation→decumulation transition
- **Cause**: Using `generate_paths()` twice instead of `generate_lifecycle_paths()`
- **Fix**: Replace with `generate_lifecycle_paths()` to ensure continuous paths
- **Verify**: Run `test_continuous_paths.py` - should show zero error

**Problem**: Success rate is 0% or 100% (unrealistic)
- **Cause**: Incorrect parameter estimation or simulation error
- **Debug**:
  ```python
  # Check estimated parameters
  print(f"Mean return: {mean_returns}")  # Should be reasonable (e.g., 0.08-0.15)
  print(f"Volatility: {np.sqrt(np.diag(cov_matrix))}")  # Should be 0.10-0.25

  # Check withdrawal rate
  withdrawal_rate = annual_withdrawal / median_final_accumulation
  print(f"Withdrawal rate: {withdrawal_rate:.2%}")  # Should be 3-5%
  ```

**Problem**: `visualize_mc_lifecycle.py` shows astronomical final values (billions)
- **Cause**: Likely using wrong frequency scaling or compounding error
- **Check**:
  - Verify `periods_per_year` matches `contribution_frequency` (biweekly = 26, not 12)
  - Check that returns are being compounded correctly: `(1+r1)*(1+r2)...-1` not `r1+r2+...`
  - Verify mean returns are annualized (not daily or monthly)

**Problem**: `YahooFinanceDataException` or empty data
- **Cause**: Yahoo Finance API rate limiting or ticker delisted
- **Fix**: Delete pickle cache and retry: `rm ../data/*.pkl`
- **Alternative**: Use different date range or tickers

**Problem**: Time-varying parameters not affecting results
- **Cause**: Dates in DataFrame don't overlap with simulation dates
- **Debug**:
  ```python
  print(f"Parameter dates: {mean_ts.index.min()} to {mean_ts.index.max()}")
  print(f"Simulation dates: {start_date} to {end_date}")
  # Should overlap!
  ```

**Problem**: Employer match exceeds cap during simulation
- **Cause**: Not resetting YTD match each year
- **Fix**: Check `run_accumulation_mc()` has year boundary detection:
  ```python
  if period_idx % periods_per_year == 0:
      ytd_match = 0  # Reset each year
  ```

### Development Guidelines

Please keep code well structured, clear and modular. Use libraries and frequently used components rather than custom solutions. Keep in mind that I may need to go through the code to understand what is going on and hand off code to other engineers.

**Code Quality Principles**:
1. **One function, one purpose** - Functions should be <50 lines when possible
2. **Extensive inline comments** - Explain "why", not just "what"
3. **Use numpy vectorization** - Avoid Python loops for performance (except where clarity matters more)
4. **Validate inputs** - Check shapes, ranges, data types at function entry
5. **Write tests first** - See test_*.py files for examples
6. **Document with examples** - Every feature should have a working example

**Performance Considerations**:
- 1000 simulations × 30 years: ~1 second (fast)
- 10000 simulations × 30 years: ~5 seconds (acceptable)
- Time-varying parameters: ~2x slower due to per-period parameter lookup
- Bottleneck: `np.random.multivariate_normal()` - use numpy's random generator, not Python's

### Common Modification Patterns

**Pattern 1: Add a new portfolio strategy**
```python
# Example: Add a 70/30 portfolio to compare against 60/40

# In visualize_mc_lifecycle.py or create new script:
weights_6040 = np.array([0.6, 0.4, 0.0, 0.0])  # 60% SPY, 40% AGG
weights_7030 = np.array([0.7, 0.3, 0.0, 0.0])  # 70% SPY, 30% AGG

# Generate paths ONCE
acc_paths, dec_paths = path_generator.generate_lifecycle_paths(...)

# Run BOTH portfolios on SAME paths (identical market scenarios)
acc_values_6040 = run_accumulation_mc(..., weights=weights_6040, asset_returns_paths=acc_paths)
acc_values_7030 = run_accumulation_mc(..., weights=weights_7030, asset_returns_paths=acc_paths)

# Compare median outcomes, success rates, etc.
```

**Pattern 2: Change contribution frequency**
```python
# Change from biweekly to monthly contributions

# In test_simple_buyhold.json:
{
  "contribution_frequency": "monthly",  // Changed from "biweekly"
  "contribution_amount": 2173  // $26K/12 months instead of $26K/26 periods
}

# SystemConfig automatically handles:
# - periods_per_year = 12 (instead of 26)
# - Frequency scaling in MCPathGenerator
```

**Pattern 3: Add Social Security income during decumulation**
```python
# Modify run_decumulation_mc() to add income stream

# In visualize_mc_lifecycle.py:
def run_decumulation_mc(..., social_security_start_year=10, social_security_amount=20000):
    for year in range(years):
        # ... existing code ...

        # Add Social Security (starts at year 10, for example)
        if year >= social_security_start_year:
            portfolio_value += social_security_amount * (1 + inflation_rate) ** year

        # ... rest of code ...
```

**Pattern 4: Test different withdrawal strategies**
```python
# Add variable withdrawal (% of portfolio) instead of fixed dollar amount

def run_decumulation_mc_variable(initial_values, weights, asset_returns_paths,
                                 withdrawal_rate=0.04, inflation_rate=0.03):
    """Variable withdrawal: withdraw X% of current portfolio each year"""
    for year in range(years):
        portfolio_return = np.dot(weights, asset_returns_paths[:, year, :])
        portfolio_values[:, year] = portfolio_values[:, year-1] * (1 + portfolio_return)

        # Withdraw percentage of CURRENT portfolio (not inflation-adjusted)
        withdrawal = portfolio_values[:, year] * withdrawal_rate
        portfolio_values[:, year] -= withdrawal
```

**Pattern 5: Add multiple regime scenarios**
```python
# Test bull/normal/bear market scenarios

import pandas as pd

# Bull scenario: high returns
bull_mean = pd.DataFrame({
    'SPY': [0.15] * 1000, 'AGG': [0.04] * 1000
}, index=pd.date_range('2025-01-01', periods=1000, freq='D'))

# Bear scenario: low returns
bear_mean = pd.DataFrame({
    'SPY': [0.02] * 1000, 'AGG': [0.06] * 1000
}, index=pd.date_range('2025-01-01', periods=1000, freq='D'))

# Run simulations for each scenario
for scenario_name, mean_ts in [('Bull', bull_mean), ('Bear', bear_mean)]:
    path_generator.set_time_varying_parameters(mean_ts)
    paths = path_generator.generate_paths_time_varying(...)
    # ... run simulation and save results with scenario_name ...
```

**Pattern 6: Export results for external analysis**
```python
# Save all simulation paths to CSV for Excel/R/Python analysis

import pandas as pd

# Export accumulation paths
acc_df = pd.DataFrame(
    acc_values.T,  # Transpose to (periods × simulations)
    columns=[f'sim_{i}' for i in range(num_sims)]
)
acc_df.to_csv('../results/test/accumulation_all_paths.csv')

# Export summary statistics
summary = pd.DataFrame({
    'metric': ['mean', 'median', 'std', '5th_pct', '95th_pct'],
    'accumulation_final': [
        acc_values[:, -1].mean(),
        np.median(acc_values[:, -1]),
        acc_values[:, -1].std(),
        np.percentile(acc_values[:, -1], 5),
        np.percentile(acc_values[:, -1], 95)
    ]
})
summary.to_csv('../results/test/summary_statistics.csv')
```

### Where to Find Things (Quick Reference)

**Need to modify...**:
- **Contribution logic** → `visualize_mc_lifecycle.py:run_accumulation_mc()` (lines ~150-200)
- **Withdrawal logic** → `visualize_mc_lifecycle.py:run_decumulation_mc()` (lines ~250-300)
- **Path generation** → `mc_path_generator.py:MCPathGenerator` (lines ~1-450)
- **Configuration options** → `system_config.py:SystemConfig` (lines ~1-150)
- **Visualization styles** → `visualize_mc_lifecycle.py:plot_lifecycle_mc()` (lines ~400-500)
- **Time-varying parameters** → `mc_path_generator.py:set_time_varying_parameters()` (lines ~250-330)
- **Portfolio optimization** → `main.py` (legacy system, separate from Monte Carlo)

**Need to understand...**:
- **How continuous paths work** → Read `test_continuous_paths.py` + `../docs/CONTINUOUS_PATHS_SUMMARY.md`
- **How time-varying works** → Read `../docs/TIME_VARYING_PARAMS_GUIDE.md` + `test_time_varying_params.py`
- **How to validate changes** → Read `../docs/VALIDATE_MC_PATHS.md` + run `test_mc_validation.py`
- **Overall architecture** → Start with this section, then trace data flow diagram above

### Mathematical Formulas and Key Algorithms

**Multivariate Gaussian Sampling** (Core of MCPathGenerator):
```python
# Sample correlated asset returns
returns = np.random.multivariate_normal(mean=μ, cov=Σ, size=N)

# Where:
# μ = [μ_SPY, μ_AGG, μ_NVDA, μ_GLD]  # Mean returns (annualized)
# Σ = [[σ²_SPY,    ρ_SPY_AGG*σ_SPY*σ_AGG, ...],
#      [ρ_SPY_AGG*σ_SPY*σ_AGG, σ²_AGG, ...],
#      [...]]  # Covariance matrix (annualized)
# N = number of samples

# Key property: Preserves correlation structure
# E.g., if SPY and AGG have ρ = -0.2, sampled returns will also have ρ ≈ -0.2
```

**Frequency Scaling** (Daily → Biweekly → Annual):
```python
# IMPORTANT: Covariance scales LINEARLY with time (not sqrt!)

# Annual parameters
μ_annual = 0.10  # 10% annual return
σ²_annual = 0.04  # 20% annual volatility (σ = 0.20)

# Convert to biweekly (26 periods per year)
μ_period = μ_annual / 26 = 0.00385  # 0.385% per biweekly period
σ²_period = σ²_annual / 26 = 0.00154  # Variance scales linearly

# Common mistake: σ_period = σ_annual / sqrt(26) = 0.0392
# This is correct for STANDARD DEVIATION, but we work with COVARIANCE MATRICES
# Covariance formula: Cov(X+Y) = Cov(X) + Cov(Y) for independent X,Y
# Therefore: Cov_annual = Cov_period * periods_per_year
```

**Portfolio Return Calculation**:
```python
# Given: asset weights w = [w_SPY, w_AGG, w_NVDA, w_GLD]
#        asset returns r = [r_SPY, r_AGG, r_NVDA, r_GLD]

# Portfolio return (for each period):
r_portfolio = np.dot(w, r)  # Weighted average
            = w_SPY * r_SPY + w_AGG * r_AGG + w_NVDA * r_NVDA + w_GLD * r_GLD

# Portfolio value update:
V_new = V_old * (1 + r_portfolio)

# NOT: V_new = V_old + V_old * r_portfolio  # Same result, but less clear
```

**Compounding Sub-Annual to Annual Returns**:
```python
# Given: biweekly returns r = [r₀, r₁, r₂, ..., r₂₅]  # 26 periods

# WRONG (arithmetic mean):
r_annual = np.mean(r)  # Incorrect!

# CORRECT (geometric mean / compounding):
r_annual = np.prod(1 + r) - 1
         = (1+r₀) * (1+r₁) * ... * (1+r₂₅) - 1

# Example: Two periods with +10% and -10%
# Wrong: (0.10 + -0.10) / 2 = 0% (suggests no change)
# Right: (1.10) * (0.90) - 1 = -0.01 = -1% (actually lost 1%)
```

**Inflation-Adjusted Withdrawals**:
```python
# Initial withdrawal: W₀ = $40,000
# Inflation rate: i = 3% = 0.03

# Withdrawal in year t:
W_t = W₀ * (1 + i)^t

# Year 0: $40,000
# Year 1: $40,000 * 1.03 = $41,200
# Year 10: $40,000 * 1.03^10 = $53,755
# Year 30: $40,000 * 1.03^30 = $97,090
```

**Employer Matching with Annual Cap**:
```python
# Per-period contribution: C = $1,000 (biweekly)
# Match rate: m = 50% = 0.5
# Annual cap: CAP = $10,000

# Each period:
period_match = C * m  # $500 per period
ytd_match += period_match

# Enforce annual cap:
if ytd_match > CAP:
    period_match = max(0, CAP - (ytd_match - period_match))

# Reset at year boundary:
if period_idx % periods_per_year == 0:
    ytd_match = 0

# Example: 26 biweekly periods
# Periods 0-19: $500 match each = $10,000 total (cap reached)
# Periods 20-25: $0 match (cap exceeded)
# Next year: Reset ytd_match, start over
```

**Success Rate Calculation**:
```python
# After running N simulations:
final_values = dec_values[:, -1]  # Final portfolio value for each simulation

# Portfolio "succeeded" if value > 0 at end
success = final_values > 0  # Boolean array

# Success rate (as percentage):
success_rate = np.sum(success) / N * 100

# Example: 950 out of 1000 simulations end with value > 0
# Success rate = 950 / 1000 * 100 = 95%
```

**Percentile Calculation**:
```python
# Get percentile bands for fan chart
percentiles = [5, 25, 50, 75, 95]

for period in range(num_periods):
    period_values = all_simulations[:, period]  # All sim values at this period
    percentile_values[period] = np.percentile(period_values, percentiles)

# Result: percentile_values[period] = [p5, p25, p50, p75, p95]
# p5 = 5th percentile (95% of outcomes are better)
# p50 = median
# p95 = 95th percentile (95% of outcomes are worse)
```

**Continuous Lifecycle Path Generation**:
```python
# Key insight: Generate ONE long path, then split

# Total periods needed:
acc_periods = acc_years * acc_periods_per_year  # e.g., 9 * 26 = 234
dec_periods = dec_years * dec_periods_per_year  # e.g., 30 * 26 = 780
total_periods = acc_periods + dec_periods  # 234 + 780 = 1014

# Generate entire path at accumulation frequency:
continuous_path = np.random.multivariate_normal(μ, Σ, size=total_periods)

# Split into accumulation (first 234 periods) and decumulation (last 780 periods):
acc_portion = continuous_path[:234, :]
dec_portion = continuous_path[234:, :]

# For decumulation, compound to annual returns:
for year in range(30):
    year_periods = dec_portion[year*26:(year+1)*26, :]  # Get 26 biweekly returns
    annual_return = np.prod(1 + year_periods, axis=0) - 1
    dec_paths[year, :] = annual_return

# Result: Decumulation year 0 starts EXACTLY where accumulation ended
# No discontinuity, same market scenario
```

## Development Commands

### Environment Setup
- Uses uv package manager with Python 3.12
- Virtual environment is managed in `.venv/`
- Install dependencies: `uv sync`
- Activate environment: `source .venv/bin/activate` or use `uv run`

### Running the Application
- Main entry point: `uv run main.py`
- Run with custom ticker file: `uv run main.py --file path/to/tickers.txt`
- Run single ticker analysis: `uv run main.py --ticker AAPL`
- Run dynamic rebalancing: Automatically runs when portfolio optimization is executed

### Project Structure
```
/
├── src/                           # Main source code
│   ├── main.py                   # Application entry point with portfolio optimization
│   ├── config.py                 # Configuration system for rebalancing parameters
│   ├── rebalancing_engine.py     # Core dynamic rebalancing logic
│   ├── performance_tracker.py    # Performance metrics collection and analysis
│   ├── rebalancing_visualization.py  # Visualization system for results
│   ├── performance_analysis.py   # Risk metrics and performance calculations
│   ├── pyproject.toml            # Project configuration and dependencies
│   └── .venv/                    # Virtual environment
├── data/                         # Data storage directory (pickle cache files)
├── logs/                         # Application logs
├── plots/rebalancing/            # Generated rebalancing plots and visualizations
├── results/rebalancing/          # CSV results and performance summaries
├── docs/                         # Documentation
└── tickers.txt                   # Default ticker file (format: SYMBOL, WEIGHT)
```

## Architecture Details

### Core Modules

#### Main Application (main.py)
- Command-line application with argparse for ticker file or single ticker analysis
- Implements modular portfolio optimization functions using cvxpy:
  - `optimize_portfolio_vanilla()` - Mean-variance optimization
  - `optimize_portfolio_robust()` - Robust optimization with uncertainty sets
- Integrates dynamic rebalancing system with configurable parameters
- Uses pickle-based data caching to avoid repeated API calls
- Generates comprehensive visualizations and performance analysis

#### Configuration System (config.py)
- `RebalancingConfig` dataclass with comprehensive parameters
- Default 30-day rebalancing periods with expanding window analysis
- Mixed portfolio configuration (40% cash default, 3% interest rate)
- Configurable optimization methods, risk parameters, and output settings
- Validation for all configuration parameters

#### Rebalancing Engine (rebalancing_engine.py)
- `RebalancingEngine` class orchestrating dynamic portfolio rebalancing
- Implements expanding window optimization with fallback strategies
- Supports multiple portfolio types: baseline, vanilla, robust, mixed portfolios
- Mixed portfolios combine optimized equity weights with cash component
- Tracks performance across calendar day periods with comprehensive metrics

#### Portfolio Tracking (portfolio_tracker.py)
- `PortfolioTracker` class for collecting and analyzing portfolio metrics using pandas DataFrames
- Calculates comprehensive metrics: Sharpe ratio, beta, drawdown, Calmar ratio
- Supports mixed portfolio tracking with cash/equity allocation percentages
- Exports results to CSV files for analysis

#### Visualization System (rebalancing_visualization.py)
- `RebalancingVisualizer` for creating comprehensive performance charts
- Dynamic rebalancing comparison plots with datetime x-axis
- Visual markers for rebalancing events and portfolio comparison
- Distinct styling for different portfolio types (solid/dashed lines)
- Mixed portfolio labels showing cash allocation percentages

### Portfolio Types

The system supports 5 different portfolio types:

1. **Baseline Portfolio**: Buy-and-hold with fixed weights from tickers.txt
2. **Vanilla Portfolio**: Mean-variance optimization (rebalanced every 30 days)
3. **Robust Portfolio**: Robust optimization with uncertainty sets (rebalanced every 30 days)
4. **Mixed Vanilla Portfolio**: 40% cash + 60% vanilla optimized equity (rebalanced every 30 days)
5. **Mixed Robust Portfolio**: 40% cash + 60% robust optimized equity (rebalanced every 30 days)

### Data Flow
1. Reads ticker symbols and weights from CSV file (format: Symbol,Weight with headers)
2. Downloads historical daily data from Yahoo Finance (cached with pickle)
3. Constructs multi-index DataFrame with proper ticker structure
4. Splits timeline into 30-day rebalancing periods
5. For each period:
   - Gets expanding window of historical data
   - Optimizes vanilla and robust portfolios using cvxpy
   - Creates mixed portfolios with cash component
   - Calculates performance metrics for all portfolios
   - Tracks cumulative returns and risk metrics
6. Generates visualization and exports results to CSV

### Key Dependencies
- **cvxpy**: Convex optimization for portfolio optimization
- **yfinance**: Yahoo Finance data download
- **pandas/numpy**: Data manipulation and numerical computing
- **matplotlib**: Plotting and visualization
- **ipdb**: Interactive debugging

### Configuration
- Default date range: 2024-01-01 to 2024-12-31
- Default ticker file: `../tickers.txt` (with headers: Symbol,Weight)
- Rebalancing period: 30 calendar days
- Mixed portfolio cash allocation: 40%
- Cash interest rate: 3% annually
- Results saved to: `../results/rebalancing/`
- Plots saved to: `../plots/rebalancing/`

### Key Features
- **Dynamic Rebalancing**: Portfolios rebalance every 30 days using expanding window data
- **Multiple Optimization Methods**: Vanilla mean-variance and robust optimization
- **Mixed Portfolios**: Conservative cash/equity blend with configurable allocation
- **Performance Tracking**: Comprehensive metrics including Sharpe ratio, beta, drawdown
- **Visualization**: Interactive plots with rebalancing markers and datetime axes
- **Data Caching**: Pickle files avoid repeated Yahoo Finance API calls
- **Configurable Plotting**: Non-blocking plots for simultaneous viewing

## Notes for Development
- All optimization uses cvxpy with proper error handling and fallback strategies
- Mixed portfolios inherit equity weights from base optimization methods
- Cash component grows at configured interest rate between rebalancing periods
- Performance analysis includes market beta calculation using SPY/equal-weight proxy
- Visualization uses distinct colors and line styles for easy portfolio identification

---

## Portfolio Lifecycle Simulation System (UPDATED)

### Overview
An integrated portfolio lifecycle system combining **accumulation** (wealth building with periodic contributions) and **decumulation** (retirement withdrawals) phases. Built on the modular portfolio optimization framework with Monte Carlo simulation capabilities.

**Current Status**: ✅ Accumulation + Decumulation Monte Carlo Complete
- Fully integrated with SystemConfig architecture
- Period-level simulation (daily, weekly, biweekly, monthly, quarterly, annual)
- Employer matching with annual caps
- Inflation-adjusted withdrawals
- Professional visualizations with fan charts and spaghetti plots

### Design Principles (User Requirements)
- "Keep code simple and modular"
- "Use libraries and frequently used components rather than custom solutions"
- "Well structured, clear and simple so code is not a black box"
- Each file has ONE clear purpose
- Each function does ONE thing
- Extensive inline documentation

### Running Lifecycle Simulations

#### Quick Start
```bash
# Run lifecycle simulation (backtest + accumulation MC + decumulation MC)
uv run python visualize_mc_lifecycle.py

# Uses configuration from test_simple_buyhold.json:
# - Backtest: 2024-01-01 to 2025-09-19
# - Accumulation: 2025-09-19 to 2035-01-01 (9.3 years)
# - Decumulation: 2035-01-01 to 2065-01-01 (30 years)
# - Contributions: $1,000 biweekly + 50% employer match (max $10K/year)
# - Withdrawals: $40,000/year with 3% inflation
```

#### Output
- **Console**: Success rate, percentiles, final portfolio values for both phases
- **Plots**: `../plots/test/` (configurable in JSON)
  - `mc_lifecycle_fan_chart.png` - Percentile bands (5th, 25th, 50th, 75th, 95th) showing accumulation → decumulation transition
  - `mc_lifecycle_spaghetti_log.png` - Individual simulation paths on log₁₀ scale (useful for wide outcome ranges)
- **Future**: CSV exports for detailed analysis (not yet implemented)

### Lifecycle System Architecture

#### File Structure
```
src/
├── system_config.py              # Global configuration (retirement dates, contributions, withdrawals)
├── portfolio_config.py           # Per-portfolio configuration (rebalancing, optimization)
├── comparison_config.py          # Experimental comparison configuration
├── mc_path_generator.py          # Asset-level multivariate Gaussian MC path generator
├── visualize_mc_lifecycle.py     # Monte Carlo lifecycle simulation + visualization
├── test_simple_buyhold.json      # Example configuration file
├── test_simple_buyhold_mc.py     # Test script for backtest + MC workflow
├── fin_data.py                   # Market data fetching with pickle caching
├── portfolio.py                  # Core portfolio class with tracking
├── ../docs/RUN_MC_TEST.md                # Validation walkthrough guide
├── ../docs/VALIDATE_MC_PATHS.md          # Step-by-step MC validation guide
├── ../docs/MC_QUICK_REFERENCE.md         # Quick reference cheat sheet
├── ../docs/TIME_VARYING_PARAMS_GUIDE.md  # Time-varying parameters complete guide
├── test_mc_validation.py         # Automated validation script
├── test_continuous_paths.py      # Continuous lifecycle paths test
└── test_time_varying_params.py   # Time-varying parameters test

plots/test/                       # Generated MC visualizations
results/test/                     # Backtest results (CSV)
data/                             # Cached MC paths (optional, for reuse)
```

#### Core Components

**1. system_config.py** - SystemConfig Class
```python
@dataclass
class SystemConfig:
    """Global configuration for entire backtesting/simulation system"""

    # Backtest period
    start_date: str                          # e.g., '2024-01-01'
    end_date: str                            # e.g., '2025-09-19'
    ticker_file: str = '../tickers.txt'

    # Retirement/Lifecycle dates
    retirement_date: Optional[str] = None    # End of accumulation phase
    simulation_horizon_date: Optional[str] = None  # End of decumulation
    simulation_horizon_years: Optional[int] = None # Alternative to horizon_date

    # Periodic contributions (accumulation phase)
    contribution_amount: Optional[float] = None    # Per-period amount (e.g., $1000 biweekly)
    contribution_frequency: str = 'biweekly'       # 'weekly', 'biweekly', 'monthly', 'annual'
    employer_match_rate: float = 0.0              # 0.5 = 50% match
    employer_match_cap: Optional[float] = None    # Annual cap (e.g., $10,000)

    # Withdrawal strategy (decumulation phase)
    withdrawal_strategy: Optional[str] = None     # None = no decumulation
    annual_withdrawal_amount: Optional[float] = None
    withdrawal_percentage: Optional[float] = None
    inflation_rate: float = 0.03

    # Output settings
    save_plots: bool = True
    plots_directory: str = '../plots/test'
    save_results: bool = True
    results_directory: str = '../results/test'
```

**Key Methods**:
- `get_accumulation_years()`: Calculate years between end_date and retirement_date
- `get_decumulation_years()`: Calculate years from retirement to simulation horizon
- `get_contribution_config()`: Return dict with contribution details or None
- `get_withdrawal_config()`: Return dict with withdrawal strategy or None
- `from_json(path)`: Load configuration from JSON file (filters '_comment' fields)

**2. mc_path_generator.py** - Asset-Level MC Path Generator
```python
class MCPathGenerator:
    """
    Generate multivariate Gaussian Monte Carlo paths for asset returns.

    Preserves asset correlations and enables portfolio comparison on identical scenarios.
    Supports both constant and time-varying parameters.
    """

    def __init__(self, tickers: List[str], mean_returns: np.ndarray,
                 cov_matrix: np.ndarray, seed: int = 42):
        """Initialize with annualized mean returns and covariance matrix"""

    def generate_paths(self, num_simulations: int, total_periods: int,
                      periods_per_year: int = 1) -> np.ndarray:
        """
        Generate asset-level return paths using multivariate normal sampling.

        Returns: (num_simulations, total_periods, num_assets) array

        Key scaling:
        - period_mean = annual_mean / periods_per_year
        - period_cov = annual_cov / periods_per_year
        """

    def generate_lifecycle_paths(self, num_simulations: int, accumulation_years: int,
                                 accumulation_periods_per_year: int,
                                 decumulation_years: int) -> tuple:
        """
        Generate CONTINUOUS lifecycle paths spanning accumulation and decumulation.

        Returns: (acc_paths, dec_paths)
        - acc_paths: (num_simulations, acc_periods, num_assets)
        - dec_paths: (num_simulations, dec_years, num_assets) - annual returns

        Key features:
        - ONE continuous random path (no gap, no re-seeding)
        - Decumulation starts from last accumulation period
        - Decumulation returns compound sub-annual periods to annual
        """

    def set_time_varying_parameters(self, mean_returns_ts: pd.DataFrame,
                                    cov_matrices_ts: Optional[pd.DataFrame] = None):
        """
        Set time-varying mean returns and covariance matrices.

        Args:
        - mean_returns_ts: DataFrame with DatetimeIndex, columns = asset tickers
                          Values are annualized mean returns
        - cov_matrices_ts: DataFrame with DatetimeIndex, columns = flattened cov elements
                          (e.g., 'SPY_SPY', 'SPY_AGG', 'AGG_AGG')

        Enables: regime switching, adaptive estimation, structural breaks
        """

    def generate_paths_time_varying(self, num_simulations: int, start_date: str,
                                    total_periods: int, periods_per_year: int,
                                    frequency: str) -> np.ndarray:
        """
        Generate MC paths with TIME-VARYING distributions.

        For each period:
        1. Generate date using pd.date_range(start_date, periods, freq)
        2. Lookup nearest parameters from time_varying_mean/cov
        3. Scale to period frequency
        4. Sample from multivariate normal

        Returns: (num_simulations, total_periods, num_assets) array
        """

    def get_path_dataframe(self, simulation_idx: int, start_date: str,
                          frequency: str) -> pd.DataFrame:
        """Extract one simulation as DataFrame for Portfolio.ingest_simulated_data()"""

    def get_portfolio_returns(self, simulation_idx: int,
                             weights: np.ndarray) -> np.ndarray:
        """Calculate portfolio returns for specific simulation"""

    def save_paths(self, filepath: str):
        """Save paths to pickle for reuse"""

    @classmethod
    def load_paths(cls, filepath: str) -> 'MCPathGenerator':
        """Load previously generated paths"""

    def get_summary_statistics(self) -> dict:
        """Verify empirical mean/cov match theoretical values"""
```

**3. visualize_mc_lifecycle.py** - Monte Carlo Simulation Engine
```python
def run_accumulation_mc(
    initial_value: float,
    weights: np.ndarray,             # Portfolio weights
    asset_returns_paths: np.ndarray, # Pre-generated from MCPathGenerator
    years: int,
    contributions_per_year: int = 1,
    contribution_amount: float = 0.0,
    employer_match_rate: float = 0.0,
    employer_match_cap: float = None
) -> np.ndarray:
    """
    Run accumulation MC using pre-generated asset-level return paths.

    Key changes from previous version:
    - Uses asset_returns_paths instead of sampling portfolio return
    - Calculates portfolio_return = np.dot(weights, asset_returns)
    - Enables multiple portfolios to use same paths
    """

def run_decumulation_mc(
    initial_values: np.ndarray,
    weights: np.ndarray,
    asset_returns_paths: np.ndarray,  # Pre-generated from MCPathGenerator
    annual_withdrawal: float,
    inflation_rate: float,
    years: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run decumulation MC using pre-generated asset-level return paths.

    Returns:
    - values: (num_simulations × years) portfolio values
    - success: (num_simulations,) boolean array (True = survived full period)
    """

def plot_lifecycle_mc(...):
    """Fan chart with percentile bands (5th, 25th, 50th, 75th, 95th)"""

def plot_spaghetti_log(...):
    """Individual simulation paths on log₁₀ y-axis"""
```

**Workflow with MCPathGenerator (Continuous Paths):**
```python
# 1. Create path generator with asset-level parameters
path_generator = MCPathGenerator(tickers, mean_returns, cov_matrix, seed=42)

# 2. Generate CONTINUOUS lifecycle paths (accumulation → decumulation)
#    ONE continuous random sequence split into two phases
acc_paths, dec_paths = path_generator.generate_lifecycle_paths(
    num_simulations=1000,
    accumulation_years=9,
    accumulation_periods_per_year=26,  # Biweekly
    decumulation_years=30
)
# acc_paths: (1000, 234, 4) = 9 years × 26 biweekly periods
# dec_paths: (1000, 30, 4) = 30 annual returns (compounded from biweekly)
# ✓ Decumulation starts from last accumulation period (continuous)

# 3. Run simulations with continuous paths
acc_values = run_accumulation_mc(initial_value, weights, acc_paths, ...)
dec_values, success = run_decumulation_mc(final_values, weights, dec_paths, ...)

# 4. FUTURE: Multiple portfolios on same paths
portfolio_A.ingest_simulated_data(path_generator.get_path_dataframe(sim_idx=0, ...))
portfolio_B.ingest_simulated_data(path_generator.get_path_dataframe(sim_idx=0, ...))
```

**Alternative: Separate Phase Paths (for independent scenarios)**
```python
# If you need independent accumulation and decumulation scenarios:
acc_paths = path_generator.generate_paths(
    num_simulations=1000, total_periods=234, periods_per_year=26
)

path_generator.seed = 43  # Different seed
dec_paths = path_generator.generate_paths(
    num_simulations=1000, total_periods=30, periods_per_year=1
)
# ⚠️ These paths are INDEPENDENT (not continuous)
```

**Time-Varying Parameters (Advanced)**
```python
# Use time-varying mean returns and covariance for regime modeling

# Example 1: Regime switching (bull → bear market)
import pandas as pd

dates = pd.date_range('2025-01-01', periods=1000, freq='D')
mean_ts = pd.DataFrame({
    'SPY': [0.15 if i < 500 else 0.02 for i in range(1000)],  # 15% → 2%
    'AGG': [0.04 if i < 500 else 0.06 for i in range(1000)]   # 4% → 6%
}, index=dates)

path_generator.set_time_varying_parameters(mean_ts)
paths = path_generator.generate_paths_time_varying(
    num_simulations=1000,
    start_date='2025-01-01',
    total_periods=1000,
    periods_per_year=252,  # Daily
    frequency='D'
)
# ✓ First 500 days: bull market returns
# ✓ Last 500 days: bear market returns

# Example 2: Volatility regimes (low vol → high vol)
cov_ts = pd.DataFrame([{
    'SPY_SPY': 0.02 if i < 250 else 0.06,  # Variance increases 3x
    'SPY_AGG': 0.0,  # Zero correlation
    'AGG_AGG': 0.01
} for i in range(500)], index=dates[:500])

path_generator.set_time_varying_parameters(mean_ts[:500], cov_ts)
paths = path_generator.generate_paths_time_varying(
    num_simulations=1000,
    start_date='2025-01-01',
    total_periods=500,
    periods_per_year=252,
    frequency='D'
)
# ✓ First 250 days: low volatility (σ² = 0.02)
# ✓ Last 250 days: high volatility (σ² = 0.06)

# Example 3: Expanding window estimation (adaptive parameters)
from fin_data import FinData

fin_data = FinData(start_date='2020-01-01', end_date='2025-12-31')
dates = pd.date_range('2025-01-01', periods=500, freq='D')

mean_list = []
cov_list = []
for i, date in enumerate(dates):
    # Use expanding window: first 30 days, then 31, 32, ..., 529 days
    window_data = fin_data.returns.iloc[:30+i]
    mean_list.append(window_data.mean() * 252)
    cov_list.append(window_data.cov() * 252)

mean_ts = pd.DataFrame(mean_list, index=dates, columns=['SPY', 'AGG'])
# Flatten covariance matrices
cov_flattened = []
for cov in cov_list:
    cov_flattened.append({
        'SPY_SPY': cov.loc['SPY', 'SPY'],
        'SPY_AGG': cov.loc['SPY', 'AGG'],
        'AGG_AGG': cov.loc['AGG', 'AGG']
    })
cov_ts = pd.DataFrame(cov_flattened, index=dates)

path_generator.set_time_varying_parameters(mean_ts, cov_ts)
paths = path_generator.generate_paths_time_varying(
    num_simulations=1000,
    start_date='2025-01-01',
    total_periods=500,
    periods_per_year=252,
    frequency='D'
)
# ✓ Parameters update every day based on expanding historical window
# ✓ Simulates adaptive estimation strategy

# See ../docs/TIME_VARYING_PARAMS_GUIDE.md for complete documentation
```

**3. test_simple_buyhold.json** - Example Configuration
```json
{
  "_comment": "Simple buy-and-hold test: backtest 2024-2025, then MC sim",

  "start_date": "2024-01-01",
  "end_date": "2025-09-19",
  "ticker_file": "../tickers.txt",

  "retirement_date": "2035-01-01",
  "simulation_horizon_years": 30,

  "contribution_amount": 1000,
  "contribution_frequency": "biweekly",
  "employer_match_rate": 0.5,
  "employer_match_cap": 10000,

  "withdrawal_strategy": "constant_inflation_adjusted",
  "annual_withdrawal_amount": 40000,
  "inflation_rate": 0.03,

  "save_plots": true,
  "plots_directory": "../plots/test",
  "save_results": true,
  "results_directory": "../results/test"
}
```
**Note**: Fields starting with `_` (like `_comment`) are automatically filtered during loading.

### Key Concepts

**Monte Carlo Simulation**: Run thousands of scenarios with random return paths sampled from **multivariate Gaussian distribution** at asset level (using historical mean returns and covariance matrix) to estimate probability distributions of outcomes.

**Asset-Level Path Generation**: MCPathGenerator uses `np.random.multivariate_normal()` to sample correlated asset returns, preserving the correlation structure between assets. This enables:
- Accurate correlation modeling (e.g., SPY and NVDA correlation preserved)
- Portfolio comparison on identical market scenarios
- Reusable paths for different portfolio strategies

**Period-Level Simulation**: Critical for accurate dollar-cost averaging modeling. Contributions happen at configured frequency (biweekly, monthly, etc.), so returns are sampled at that frequency:
- `period_mean = annual_mean / periods_per_year`
- `period_cov = annual_cov / periods_per_year` (NOT `annual_std / sqrt(periods_per_year)` for covariance!)
- Portfolio return each period: `portfolio_return = np.dot(weights, asset_returns)`

**Employer Matching**: Employer contributions calculated per period with year-to-date tracking to enforce annual cap:
```python
period_match = contribution_amount * employer_match_rate
if employer_match_cap is not None:
    period_match = min(period_match, employer_match_cap - ytd_match)
```

**Percentile Analysis**:
- 5th percentile = worst-case scenario (95% of outcomes better than this)
- 50th percentile = median outcome
- 95th percentile = best-case scenario (95% of outcomes worse than this)

**Success Rate**: Percentage of simulations where portfolio survived the full decumulation period without depletion.

**Inflation-Adjusted Withdrawals**: Each year's withdrawal = initial_withdrawal × (1 + inflation_rate)^year

**Time-Varying Parameters**: MCPathGenerator supports time-varying mean returns and covariance matrices via pandas DataFrames with DatetimeIndex. This enables:
- **Regime switching**: Different return distributions for bull/bear markets
- **Adaptive estimation**: Rolling or expanding window parameter updates
- **Structural breaks**: One-time permanent shifts in market characteristics
- **Scenario analysis**: CAPE-based expected returns, volatility regimes, market cycles
- **Integration with forecasting models**: Use external models to predict future parameters

### Interpreting Results

**Accumulation Phase**:
- **Median Final Value**: Typical outcome after years of contributions + market growth
- **5th/95th Percentile Range**: Shows market risk impact on final accumulation
- **Contribution Impact**: Compare results with/without contributions to see dollar-cost averaging benefit

**Decumulation Phase**:
- **Success Rate**: Percentage of paths that survive full retirement period
  - ≥90%: Very safe, high probability of success
  - 80-90%: Relatively safe, good chance of success
  - 70-80%: Moderate risk, consider reducing withdrawals
  - <70%: High risk, need to adjust plan
- **Median Final Value**: If significantly > 0, indicates conservative plan with room to increase spending
- **5th Percentile Path**: "Bad luck" scenario - if this stays > 0, plan is robust to poor markets

### Workflow Integration

**Complete Workflow**:
1. **Backtest**: Historical performance using actual portfolio system (e.g., 2024-2025)
2. **Estimate Parameters**: Calculate mean returns and covariance from backtest period
3. **Accumulation MC**: Simulate wealth building with periodic contributions
4. **Decumulation MC**: Simulate retirement withdrawals starting from accumulation outcomes
5. **Visualization**: Generate fan charts and log-scale path plots

**Validation and Testing**:
- **Complete validation**: See [../docs/VALIDATE_MC_PATHS.md](../docs/VALIDATE_MC_PATHS.md) for step-by-step validation guide
- **Automated validation**: Run `uv run python test_mc_validation.py` to validate all MC components
- **Quick reference**: See [../docs/MC_QUICK_REFERENCE.md](../docs/MC_QUICK_REFERENCE.md) for common tasks and debugging
- **Continuous paths**: `test_continuous_paths.py` verifies zero-error continuity between accumulation and decumulation
- **Time-varying params**: `test_time_varying_params.py` validates regime switching and adaptive estimation

---

## Lifecycle System - Implementation Status

### ✅ Completed Features

**Phase 1: Core Monte Carlo Framework**
- [x] SystemConfig with retirement dates, contributions, withdrawals
- [x] Period-level accumulation simulation (biweekly, monthly, etc.)
- [x] Employer matching with annual cap tracking
- [x] Decumulation with inflation-adjusted withdrawals
- [x] Fan chart visualization (percentile bands)
- [x] Spaghetti plot with log₁₀ scale
- [x] Integration with existing portfolio backtest system
- [x] JSON configuration with comment filtering
- [x] **MCPathGenerator** - Asset-level multivariate Gaussian path generation
- [x] **Continuous lifecycle paths** - Accumulation → decumulation continuity
- [x] **Time-varying parameters** - Regime switching, adaptive estimation, structural breaks
- [x] Reusable paths enabling portfolio comparison on identical scenarios
- [x] **Comprehensive testing** - test_mc_validation.py, test_continuous_paths.py, test_time_varying_params.py
- [x] **Documentation** - ../docs/VALIDATE_MC_PATHS.md, ../docs/MC_QUICK_REFERENCE.md, ../docs/TIME_VARYING_PARAMS_GUIDE.md

**Recent Updates (2025-10-04/05)**:
1. **Period-level iteration fix**: Changed from annual to period-level iteration for accurate dollar-cost averaging
2. **Asset-level path generation** (MAJOR):
   - Created `MCPathGenerator` class using `np.random.multivariate_normal()`
   - Preserves asset correlations via covariance matrix
   - Generates reusable paths: (num_simulations, total_periods, num_assets)
   - Refactored `run_accumulation_mc()` and `run_decumulation_mc()` to use pre-generated paths
   - Enables multiple portfolios to backtest on same market scenarios
   - Portfolio returns calculated as: `portfolio_return = np.dot(weights, asset_returns)`
3. **Continuous lifecycle paths** (CRITICAL FIX):
   - Added `generate_lifecycle_paths()` method to MCPathGenerator
   - Generates ONE continuous random path spanning both accumulation and decumulation
   - Decumulation paths start from last accumulation period (no gap, no re-seeding)
   - Annual decumulation returns compound sub-annual periods: `(1+r1)*(1+r2)*...(1+r26) - 1`
   - Validated: test_continuous_paths.py confirms zero-error continuity
4. **Time-varying parameters** (NEW):
   - Added support for time-varying mean returns and covariance matrices via pandas DataFrames
   - Methods: `set_time_varying_parameters(mean_ts, cov_ts)` and `generate_paths_time_varying(...)`
   - Enables regime switching (bull/bear markets), adaptive estimation, structural breaks
   - Nearest-neighbor date matching for parameter lookup
   - Validated: test_time_varying_params.py confirms regime transitions and volatility changes
   - Documentation: ../docs/TIME_VARYING_PARAMS_GUIDE.md with complete examples
5. **Validation documentation**:
   - Created ../docs/VALIDATE_MC_PATHS.md - Step-by-step validation walkthrough
   - Created test_mc_validation.py - Automated validation script
   - Created ../docs/MC_QUICK_REFERENCE.md - Quick reference cheat sheet

### 🚧 Future Work

### Phase 2: Advanced Withdrawal Strategies

Current implementation uses simple 4% rule with inflation adjustments. Add adaptive strategies:

**1. Guyton-Klinger Method**
- Dynamic adjustments with guardrails
- Increase spending in good years, cut in bad years
- Upper/lower bounds to prevent excessive changes

**2. Variable Percentage Withdrawal (VPW)**
- Withdrawal rate varies by age and portfolio value
- More conservative as you age
- Adjusts to market performance

**3. Floor-Ceiling Strategy**
- Minimum floor (essential expenses)
- Discretionary ceiling (nice-to-haves)
- Cut discretionary first in downturns

**Implementation Approach**:
- Extend `run_decumulation_mc()` to support strategy parameter
- Create `withdrawal_strategies.py` module with strategy classes
- Each strategy implements `calculate_withdrawal(year, portfolio_value, prev_withdrawal, config) -> float`
- Add strategy comparison visualizations

### Phase 3: Financial Events Timeline

Add support for time-varying cash flows:

**Events to Support**:
- Mortgage payments and payoff date
- Social Security commencement (age 62, 67, or 70)
- Pension income
- One-time expenses (home repairs, medical, travel)
- Part-time work income
- Healthcare costs before Medicare

**Implementation Approach**:
- Create `FinancialEvent` dataclass with date, amount, type, frequency
- Add `events: List[FinancialEvent]` to SystemConfig
- Modify simulation loops to apply events at appropriate dates
- Track net cash flow separately from portfolio returns

### Phase 4: CSV Export and Reporting

Add comprehensive data export capabilities:

**Features**:
- Export all simulation paths to CSV
- Summary statistics (mean, median, percentiles by year)
- Success rate analysis
- Contribution vs. growth breakdown (accumulation)
- Withdrawal vs. returns breakdown (decumulation)

**Implementation Approach**:
- Add export functions to visualize_mc_lifecycle.py
- Create `results/test/accumulation_paths.csv` and `decumulation_paths.csv`
- Generate summary statistics CSV files
- Optional: Excel workbook with multiple sheets

### Phase 5: Regime-Switching Monte Carlo

Integrate with existing RSMC.py for market regime modeling:

**Features**:
- Market regime detection (bull/bear/normal or CAPE-based)
- Regime-specific return parameters
- Transition probability matrices
- More realistic long-term simulations

**Implementation Approach**:
- Extend `fin_data.py` with regime detection methods
- Add `regime_params` and `transition_matrix` to config
- Create `sample_regime_switching_returns()` method
- Option to use regime-switching or vanilla bootstrap
- Compare regime-aware vs. IID assumptions

**Note**: RSMC.py already exists with regime infrastructure - focus on integration.

### Phase 6: Dynamic Asset Allocation

Enable portfolio rebalancing during accumulation/decumulation:

**Features**:
- Glide path strategies (aggressive → conservative over time)
- Target-date fund behavior
- Dynamic rebalancing based on market conditions
- Integration with existing portfolio optimization

**Implementation Approach**:
- Add `rebalancing_strategy` to SystemConfig
- Support periodic rebalancing (annual, quarterly, threshold-based)
- Use existing Portfolio class for optimization
- Track transaction costs if desired

### Phase 7: Integration Testing & Historical Validation

Validate against known studies and historical periods:

**Tests**:
- Trinity Study validation (4% rule, various allocations)
- Historical sequence-of-returns testing (1929, 1966, 2000, 2008 retirements)
- Different contribution scenarios
- Performance optimization for large simulations (10K+ paths)

**Implementation Approach**:
- Create integration test suite
- Add historical period configs (Great Depression, stagflation, dot-com, 2008)
- Compare success rates to published studies
- Document validation results and edge cases

---

## Code Quality Standards

When working on lifecycle simulation code:

1. **Simplicity First**: Prefer clear code over clever abstractions
2. **One Purpose per File**: Each module has a single, clear responsibility
3. **Standard Libraries**: Use pandas, numpy, matplotlib - avoid custom frameworks
4. **Extensive Documentation**: Inline comments explaining "why", not just "what"
5. **Modular Design**: Components work independently and compose cleanly
6. **Clear Examples**: Working configurations and test scripts
7. **Performance**: Optimize for large simulations (vectorize operations, use numpy efficiently)