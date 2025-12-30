# Monte Carlo Simulator Guide

Production Monte Carlo simulator for lifecycle portfolio simulations.

## Quick Start

```bash
# Single run with Yahoo Finance data
uv run python src/run_mc.py --config configs/test_simple_buyhold.json

# Parameter sweep
uv run python src/run_mc.py --config configs/test_simple_buyhold.json --sweep
```

## Overview

`run_mc.py` runs Monte Carlo simulations for retirement lifecycle planning with two sampling methods:

| Method | Data Source | Use Case |
|--------|-------------|----------|
| **Parametric** | Mean/covariance from `sim_params` or historical | Gaussian returns |
| **Bootstrap** | Historical returns (Yahoo Finance) | Empirical distribution |

### Simplified Data Flow

```
run_mc.py
    │
    ├── Try load_returns_data(mode='yahoo')
    │   │
    │   ├── SUCCESS → historical_returns available
    │   │             tickers from tickers.txt
    │   │             Run BOTH parametric AND bootstrap
    │   │
    │   └── FAIL → No historical data
    │              tickers = sim_params.tickers
    │              Run PARAMETRIC ONLY
    │
    └── MCPathGenerator → run_accumulation_mc() / run_decumulation_mc()
```

**Key design decision**: Bootstrap with simulated Gaussian data is redundant (same as parametric), so when Yahoo Finance is unavailable, only parametric sampling runs.

## Data Sources

### Yahoo Finance (Default)

Fetches real market data for tickers in `tickers.txt`:

```bash
uv run python src/run_mc.py -c configs/test_simple_buyhold.json
```

Output:
```
[1/3] Loading historical data...
  Fetching historical data from Yahoo Finance...
    Tickers: ['SPY', 'AGG', 'NVDA', 'GLD']
    Period: 2024-01-01 to 2025-09-19
    Loaded 89 periods (weekly)
  Data source: yahoo_finance
```

### Parametric Only (Fallback)

If Yahoo Finance fails, runs parametric-only using `sim_params`:

```
[1/3] Loading historical data...
  Warning: Could not load historical data (...)
  Data source: parametric_only (no historical data available)
```

### Using CSV Data

For offline use, place CSV files in `output/cache/`:

```
output/cache/
  market_data_cache_returns.csv   # Returns data
  market_data_cache_prices.csv    # Price data
```

**CSV Format:**
```csv
Date,SPY,AGG,NVDA,GLD
2024-01-08,0.012,-0.003,0.045,0.002
2024-01-15,0.008,0.001,0.023,-0.001
```

**Important**: Current caching uses filename-only keys (not ticker-aware). To force fresh data, set `use_cache=False` in `load_bootstrap_data()` or delete cache files.

## Mean/Covariance Selection

| Scenario | Tickers Used | Mean/Cov Source |
|----------|--------------|-----------------|
| Yahoo available, tickers match sim_params | From tickers.txt | sim_params |
| Yahoo available, tickers differ | From tickers.txt | Historical data |
| Yahoo unavailable | sim_params.tickers | sim_params |

When tickers match `sim_params.tickers` (`['BIL', 'MSFT', 'NVDA', 'SPY']`), the known mean/cov from `simulated_data_params.py` is used for parametric sampling.

## CLI Reference

```
usage: run_mc.py [-h] --config CONFIG [--sims SIMS] [--seed SEED]
                 [--sweep] [--sweep-param SWEEP_PARAM]
                 [--sweep-start SWEEP_START] [--sweep-end SWEEP_END]
                 [--sweep-step SWEEP_STEP] [--output OUTPUT] [--no-plot]

Options:
  --config, -c CONFIG     Path to JSON config file (required)
  --sims, -n SIMS         Number of MC simulations (overrides config)
  --seed SEED             Random seed (default: 42)
  --sweep                 Run parameter sweep from config sweep_params
  --sweep-param PARAM     Parameter to sweep (overrides config)
  --sweep-start VALUE     Sweep start value
  --sweep-end VALUE       Sweep end value
  --sweep-step VALUE      Sweep step size
  --output, -o DIR        Output directory (default: output/mc)
  --no-plot               Skip generating visualization
```

## Usage Examples

### Single Run

```bash
# Basic run with Yahoo Finance
uv run python src/run_mc.py --config configs/test_simple_buyhold.json

# With more simulations
uv run python src/run_mc.py -c configs/test_simple_buyhold.json --sims 500

# With custom seed
uv run python src/run_mc.py -c configs/test_simple_buyhold.json --seed 123

# Skip visualization
uv run python src/run_mc.py -c configs/test_simple_buyhold.json --no-plot
```

### Parameter Sweep

```bash
# Sweep from config file sweep_params
uv run python src/run_mc.py -c configs/test_simple_buyhold.json --sweep

# Custom sweep: initial portfolio value $500K to $2M
uv run python src/run_mc.py -c configs/test_simple_buyhold.json \
    --sweep --sweep-param initial_portfolio_value \
    --sweep-start 500000 --sweep-end 2000000 --sweep-step 500000

# Custom sweep: withdrawal amount $20K to $80K
uv run python src/run_mc.py -c configs/test_simple_buyhold.json \
    --sweep --sweep-param annual_withdrawal_amount \
    --sweep-start 20000 --sweep-end 80000 --sweep-step 10000
```

## Config File

### Required Settings

```json
{
  "start_date": "2024-01-01",
  "end_date": "2025-09-19",
  "ticker_file": "tickers.txt",

  "retirement_date": "2034-01-01",
  "decumulation_horizon_years": 20,
  "initial_portfolio_value": 1000000,
  "simulation_frequency": "weekly",

  "withdrawal_strategy": "constant_inflation_adjusted",
  "annual_withdrawal_amount": 40000,

  "num_mc_simulations": 100
}
```

### Sweep Parameters (Optional)

```json
{
  "sweep_params": [
    {
      "name": "initial_portfolio_value",
      "start": 500000,
      "end": 2000000,
      "step": 500000
    }
  ]
}
```

### Sweepable Parameters

| Parameter | Description |
|-----------|-------------|
| `initial_portfolio_value` | Starting portfolio value |
| `annual_withdrawal_amount` | Annual withdrawal in decumulation |
| `contribution_amount` | Per-period contribution in accumulation |
| `inflation_rate` | Annual inflation rate |

## Output Files

### Single Run

```
output/mc/
  mc_results.csv          # Percentiles and success rates
  mc_lifecycle.png        # Lifecycle visualization (4 panels)
```

**mc_results.csv columns:**
- `metric`: acc_p5, acc_p25, acc_p50, acc_p75, acc_p95, dec_success, dec_p5, ...
- `parametric`: Parametric method values
- `bootstrap`: Bootstrap method values (if historical data available)

### Parameter Sweep

```
output/mc/sweep/
  {param}_sweep.csv       # Sweep results
  {param}_sweep.png       # Fan chart visualization
```

## Example Output

### With Yahoo Finance (Parametric + Bootstrap)

```
================================================================================
MONTE CARLO LIFECYCLE SIMULATOR
================================================================================

Config: configs/test_simple_buyhold.json
  Initial value: $1,000,000
  Accumulation: 8 years
  Decumulation: 20 years
  Withdrawal: $40,000/year

[MODE] Single Run

[1/3] Loading historical data...
  Fetching historical data from Yahoo Finance...
    Tickers: ['SPY', 'AGG', 'NVDA', 'GLD']
    Period: 2024-01-01 to 2025-09-19
    Loaded 89 periods (weekly)
  Data source: yahoo_finance
  Periods: 89

[2/3] Running Monte Carlo simulation...
    Using historical mean/cov (tickers differ from sim_params)
  Running 100 simulations...
    Mode: Parametric + Bootstrap
    Generating parametric paths...
    Generating bootstrap paths...

================================================================================
SIMULATION RESULTS
================================================================================

Accumulation (end of 8 years):
  Percentile      Parametric           Bootstrap
  -------------------------------------------------------
  5th             $        970,549   $      8,975,773
  25th            $      1,019,129   $     12,917,462
  50th            $      1,045,329   $     16,864,276
  75th            $      1,097,633   $     23,891,919
  95th            $      1,176,131   $     35,651,227

Decumulation (after 20 years):
  Success Rate: Parametric=72.0%, Bootstrap=100.0%
```

### Without Yahoo Finance (Parametric Only)

```
[1/3] Loading historical data...
  Warning: Could not load historical data (...)
  Data source: parametric_only (no historical data available)

[2/3] Running Monte Carlo simulation...
    Using sim_params (no historical data)
  Running 100 simulations...
    Mode: Parametric only

Accumulation (end of 8 years):
  Percentile      Parametric
  -----------------------------------
  5th             $        872,546
  25th            $        938,347
  50th            $        983,872
  75th            $      1,043,441
  95th            $      1,127,445

Decumulation (after 20 years):
  Success Rate: Parametric=15.0%
```

## Troubleshooting

### Yahoo Finance Fails

The simulator automatically falls back to parametric-only mode:
```
Warning: Could not load historical data (...)
Data source: parametric_only
```

This is expected when:
- No internet connection
- Yahoo Finance API rate limiting
- Invalid ticker symbols

### Stale Cache Data

**Symptom**: Wrong column names in loaded data
```
tickers: ['bonds', 'stocks', ...] != expected: ['SPY', 'AGG', ...]
```

**Solution**: Delete cache files
```bash
rm output/cache/market_data_cache_*.csv
rm output/cache/*.pkl
```

### Ticker Mismatch Warning

```
Using historical mean/cov (tickers: ['SPY', 'AGG', 'NVDA', 'GLD'] != sim_params: ['BIL', 'MSFT', 'NVDA', 'SPY'])
```

This is informational - the system uses historical statistics when tickers don't match sim_params. To use sim_params mean/cov, update `tickers.txt` to match.

### Missing Config Parameters

Ensure your config file has:
- `retirement_date` - End of accumulation phase
- `decumulation_horizon_years` or `simulation_horizon_date` - Length of decumulation
- `initial_portfolio_value` - Starting portfolio value
- `annual_withdrawal_amount` - For decumulation simulation

## Related Files

| File | Purpose |
|------|---------|
| `src/run_mc.py` | Main Monte Carlo simulator |
| `src/simulated_data_params.py` | Mean/covariance for parametric MC |
| `src/data/market_data.py` | Yahoo Finance data loading |
| `src/montecarlo/path_generator.py` | MCPathGenerator with bootstrap |
| `src/montecarlo/lifecycle.py` | Accumulation/decumulation simulation |
| `tickers.txt` | Ticker symbols and weights |
| `configs/test_simple_buyhold.json` | Example configuration |

## Architecture

### Core Functions

**`load_bootstrap_data(config)`**
- Loads historical returns from Yahoo Finance
- Uses `load_returns_data()` from `market_data.py`
- Returns `(returns_df, tickers)` or `(None, None)` if unavailable

**`run_mc_simulation(config, historical_returns, ...)`**
- Main simulation entry point
- If `historical_returns` is None: parametric only
- If `historical_returns` provided: parametric + bootstrap

**`run_parameter_sweep(config_path, param_name, param_values, ...)`**
- Runs multiple simulations across parameter values
- Loads historical data once, reuses for all runs

### Data Flow

```
1. Load config from JSON
2. Load historical data (Yahoo Finance via load_bootstrap_data)
3. Determine tickers and mean/cov:
   - If Yahoo available + tickers match sim_params → use sim_params mean/cov
   - If Yahoo available + tickers differ → use historical mean/cov
   - If Yahoo unavailable → use sim_params tickers and mean/cov
4. Create MCPathGenerator
5. Generate parametric paths (always)
6. Generate bootstrap paths (only if historical data available)
7. Run accumulation/decumulation simulations
8. Calculate percentiles and success rates
9. Generate visualization
10. Save results to CSV
```
