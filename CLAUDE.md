# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

A **portfolio lifecycle simulation system** for retirement planning that answers:
- "If I invest $100K with $1K biweekly contributions for 9 years, then withdraw $40K/year for 20 years, what's the probability my money lasts?"

### Main Entry Points

| Script | Purpose |
|--------|---------|
| `uv run python src/run_mc.py --config configs/test_simple_buyhold.json` | **Production MC simulator** - parametric + bootstrap sampling |
| `uv run python src/test_mc_validation.py` | Validation with known parameters |
| `uv run python src/main.py` | Legacy backtesting system |

### Quick Start

```bash
cd /path/to/fin/port
uv sync  # Install dependencies

# Run Monte Carlo simulation
uv run python src/run_mc.py --config configs/test_simple_buyhold.json

# Output: output/mc/mc_lifecycle.png, output/mc/mc_results.csv
```

---

## Project Structure

```
port/
├── src/
│   ├── run_mc.py                 # Production MC simulator (parametric + bootstrap)
│   ├── test_mc_validation.py     # Validation script with known parameters
│   ├── main.py                   # Legacy backtesting entry point
│   │
│   ├── montecarlo/               # Monte Carlo simulation
│   │   ├── __init__.py          # Exports MCPathGenerator
│   │   ├── path_generator.py    # MCPathGenerator - generates asset return paths
│   │   ├── lifecycle.py         # run_accumulation_mc(), run_decumulation_mc()
│   │   └── bootstrap.py         # Bootstrap sampling functions
│   │
│   ├── config/                   # Configuration
│   │   ├── system_config.py     # SystemConfig dataclass (dates, contributions, withdrawals)
│   │   └── rebalancing_config.py
│   │
│   ├── data/                     # Data loading
│   │   ├── market_data.py       # load_returns_data(), fetch_yahoo_finance_data()
│   │   └── covariance.py
│   │
│   ├── simulated_data_params.py  # Known test parameters (mean returns, covariance)
│   ├── mc_path_generator.py      # Legacy path generator (use montecarlo/ instead)
│   ├── fin_data.py               # FinData class for Yahoo Finance with caching
│   └── system_config.py          # Legacy config (use config/ instead)
│
├── configs/
│   ├── test_simple_buyhold.json  # Example configuration
│   └── data/
│       ├── simulated_mean_returns.csv
│       └── simulated_cov_matrices.txt
│
├── tickers.txt                   # Ticker symbols and weights
├── output/                       # Generated outputs
│   ├── mc/                       # MC simulation results
│   └── cache/                    # Data cache files
└── tests/                        # Test files
```

---

## Key Concepts

### Two Sampling Methods

1. **Parametric**: Sample from multivariate Gaussian using mean/covariance from config files
2. **Bootstrap**: Resample from historical Yahoo Finance data (preserves actual market behavior)

### Lifecycle Phases

1. **Accumulation**: Working years with periodic contributions + employer matching
2. **Decumulation**: Retirement with inflation-adjusted withdrawals

### Five Important Dates

| Date | Purpose |
|------|---------|
| `start_date` | Historical data start (for bootstrap) |
| `end_date` | Historical data end |
| `mc_start_date` | MC simulation start (accumulation begins) |
| `retirement_date` | End of accumulation / start of decumulation |
| `simulation_horizon_date` | End of simulation |

Timeline: `[start_date --- historical data --- end_date] → [mc_start_date --- accumulation --- retirement_date --- decumulation --- horizon]`

---

## Configuration (configs/test_simple_buyhold.json)

```json
{
  "start_date": "2005-01-01",
  "end_date": "2025-09-19",
  "mc_start_date": "2025-10-01",
  "ticker_file": "tickers.txt",

  "retirement_date": "2034-01-01",
  "simulation_horizon_years": 20,
  "initial_portfolio_value": 1000000,
  "simulation_frequency": "weekly",

  "contribution_amount": 0,
  "contribution_frequency": "biweekly",
  "employer_match_rate": 0.0,

  "withdrawal_strategy": "constant_inflation_adjusted",
  "annual_withdrawal_amount": 40000,
  "withdrawal_frequency": "biweekly",
  "inflation_rate": 0.03,

  "num_mc_simulations": 100,
  "simulated_mean_returns_file": "configs/data/simulated_mean_returns.csv",
  "simulated_cov_matrices_file": "configs/data/simulated_cov_matrices.txt"
}
```

---

## Core Components

### MCPathGenerator (src/montecarlo/path_generator.py)

Generates correlated asset return paths using multivariate Gaussian sampling.

```python
from src.montecarlo import MCPathGenerator

# Create generator
gen = MCPathGenerator(tickers=['SPY', 'AGG', 'NVDA', 'GLD'], seed=42)

# Generate paths with time-varying parameters
acc_paths, dec_paths = gen.generate_paths(
    num_simulations=100,
    accumulation_years=9,
    decumulation_years=20,
    periods_per_year=52,  # weekly
    start_date=mc_start_date,
    frequency='W',
    mean_returns=mean_ts,      # pd.DataFrame with regime dates
    cov_matrices=cov_ts,       # pd.DataFrame with cov_matrix column
    sampling_method='parametric'  # or 'bootstrap'
)
```

### Lifecycle Functions (src/montecarlo/lifecycle.py)

```python
from src.montecarlo.lifecycle import run_accumulation_mc, run_decumulation_mc

# Run accumulation (returns portfolio values at each checkpoint)
acc_values = run_accumulation_mc(
    initial_value=1_000_000,
    weights=np.array([0.6, 0.4, 0.0, 0.0]),
    asset_returns_paths=acc_paths,
    asset_returns_frequency=52,
    years=9,
    contributions_per_year=26,
    contribution_amount=1000
)

# Run decumulation (returns values and success flags)
dec_values, success = run_decumulation_mc(
    initial_values=acc_values[:, -1],
    weights=weights,
    asset_returns_paths=dec_paths,
    asset_returns_frequency=52,
    annual_withdrawal=40000,
    inflation_rate=0.03,
    years=20,
    withdrawals_per_year=26
)
```

---

## Common Gotchas

### 1. Covariance Scales Linearly (NOT sqrt)
```python
# ❌ WRONG
period_cov = annual_cov / np.sqrt(periods_per_year)

# ✅ CORRECT
period_cov = annual_cov / periods_per_year
```

### 2. Compound Returns (NOT sum)
```python
# ❌ WRONG
annual_return = np.sum(weekly_returns)

# ✅ CORRECT
annual_return = np.prod(1 + weekly_returns) - 1
```

### 3. Bootstrap Frequency Mismatch
Historical data may be at different frequency than simulation. The system resamples using geometric compounding:
```python
resampled = (1 + daily_returns).resample('W').prod() - 1
```

### 4. Stale Cache Files
```bash
# Clear cache if data looks wrong
rm output/cache/market_data_cache_*.csv
rm output/cache/*.pkl
```

---

## Data Flow

```
run_mc.py
    │
    ├── Load config (SystemConfig.from_json)
    │
    ├── Load historical data (load_returns_data)
    │   ├── Yahoo Finance → returns_df (for bootstrap)
    │   └── tickers.txt → tickers, weights
    │
    ├── Load time-varying params
    │   ├── simulated_mean_returns.csv → mean_ts (DataFrame)
    │   └── simulated_cov_matrices.txt → cov_ts (DataFrame)
    │
    ├── MCPathGenerator.generate_paths()
    │   ├── Parametric: sample from multivariate Gaussian
    │   └── Bootstrap: resample from historical returns
    │
    ├── run_accumulation_mc() → acc_values (num_sims, periods)
    │
    ├── run_decumulation_mc() → dec_values, success
    │
    └── Visualization → output/mc/mc_lifecycle.png
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| "Mean error too large" in validation | Random variation | Re-run with different seed |
| Bootstrap shows billions | Frequency mismatch | Check `_resample_returns_to_frequency()` is called |
| Success rate 0% or 100% | Bad parameters | Check mean returns are reasonable (0.05-0.15) |
| Wrong columns in data | Stale cache | Delete `output/cache/*.csv` and `.pkl` files |
| Yahoo Finance errors | Rate limiting | Delete cache, retry later |

---

## Development Guidelines

1. **Simplicity First**: Clear code over clever abstractions
2. **Configuration-Driven**: All settings in `SystemConfig` and JSON
3. **Vectorized Operations**: Use numpy broadcasting for performance
4. **Extensive Comments**: Explain "why", not just "what"
5. **Don't remove `import ipdb`**: User needs it for debugging

---

## Key Files Reference

| Need to modify... | File |
|-------------------|------|
| MC path generation | `src/montecarlo/path_generator.py` |
| Accumulation/decumulation logic | `src/montecarlo/lifecycle.py` |
| Bootstrap sampling | `src/montecarlo/bootstrap.py` |
| Configuration loading | `src/config/system_config.py` |
| Production simulator | `src/run_mc.py` |
| Market data loading | `src/data/market_data.py` |
| Test parameters | `src/simulated_data_params.py` |
| Validation script | `src/test_mc_validation.py` |
