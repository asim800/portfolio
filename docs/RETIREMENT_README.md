# Retirement Monte Carlo Simulation - Code Guide

## Overview

This retirement planning system uses **Monte Carlo simulation** to estimate the probability that your portfolio will last throughout retirement. It's built with **simple, modular code** using standard Python libraries.

## How It Works (Simple Explanation)

1. **You provide**: Starting portfolio, withdrawal amount, time horizon, asset allocation
2. **The system runs 1000s of simulations**: Each simulation randomly samples historical market returns
3. **You get results**: Success rate, percentile outcomes, visualizations

Think of it like a weather forecast: instead of predicting one future, it shows you a range of possible futures based on historical data.

## Code Structure

### 📁 Core Files (in order of execution)

```
src/
├── retirement_config.py          # What you want to simulate
├── fin_data.py                   # Gets historical market data
├── retirement_engine.py          # Runs the simulations
├── retirement_visualization.py   # Creates plots
└── demo_retirement.py            # Example usage
```

---

## File-by-File Explanation

### 1. `retirement_config.py` - CONFIGURATION

**What it does**: Stores all your simulation parameters in one place.

**Key class**: `RetirementConfig`

**Simple example**:
```python
from retirement_config import RetirementConfig

config = RetirementConfig(
    initial_portfolio=1_000_000,      # Starting with $1M
    annual_withdrawal=40_000,          # Withdraw $40k/year
    start_date='2024-01-01',
    end_date='2054-01-01',            # 30-year retirement
    current_portfolio={'SPY': 0.6, 'AGG': 0.4},  # 60% stocks, 40% bonds
    inflation_rate=0.03                # 3% inflation per year
)
```

**What gets validated**:
- Dates are in correct order
- Portfolio weights sum to 100%
- Withdrawal rate is reasonable
- Inflation rate is between 0-20%

**Output**:
```python
config.num_years        # Calculated: 30
config.withdrawal_rate  # Calculated: 4.0% (40k/1M)
config.tickers          # ['SPY', 'AGG']
```

---

### 2. `fin_data.py` - HISTORICAL DATA

**What it does**: Downloads historical stock/bond returns from Yahoo Finance and generates random return scenarios.

**Key methods**:

```python
from fin_data import FinData

# Initialize with date range
fin_data = FinData(
    start_date='2020-01-01',
    end_date='2024-01-01',
    cache_dir='../data'  # Saves downloaded data here
)

# Get historical daily returns
returns = fin_data.get_returns_data(['SPY', 'AGG'])
# Returns: DataFrame with daily % changes for SPY and AGG

# Sample random future returns (for Monte Carlo)
random_returns = fin_data.sample_annual_returns(
    tickers=['SPY', 'AGG'],
    num_years=30,
    method='bootstrap',  # Randomly resample from history
    seed=42             # For reproducibility
)
# Returns: DataFrame with 30 rows (years) of simulated annual returns
```

**Two sampling methods**:
1. **Bootstrap** (default): Randomly picks actual historical days
   - Pro: Preserves real market behavior (fat tails, crashes)
   - Con: Limited by historical data

2. **Parametric**: Fits a normal distribution to historical data
   - Pro: Can generate scenarios beyond historical range
   - Con: Assumes normal distribution (may miss extreme events)

---

### 3. `retirement_engine.py` - SIMULATION ENGINE

**What it does**: The core logic. Simulates retirement portfolios year-by-year.

#### Class: `RetirementEngine`

**Initialization**:
```python
from retirement_engine import RetirementEngine

engine = RetirementEngine(config)
# Automatically loads historical data for your tickers
```

**Method 1: Single Path Simulation**

Simulates ONE retirement scenario with given returns:

```python
# Provide specific returns (e.g., from historical data or custom scenario)
annual_returns = pd.DataFrame({
    'SPY': [0.08, 0.10, -0.05, 0.12, ...],  # 30 years of returns
    'AGG': [0.03, 0.02, 0.04, 0.01, ...]
})

result = engine.run_single_path(annual_returns)
```

**Output**: `SimulationPath` object:
```python
result.success              # True/False - did portfolio survive?
result.final_value          # Ending portfolio value
result.depletion_year       # Year it ran out (if failed)
result.portfolio_values     # [1000000, 1050000, 1100000, ...]
result.withdrawals          # [40000, 41200, 42436, ...]  # Inflation-adjusted
result.annual_returns       # [0.068, 0.072, -0.01, ...]  # Portfolio returns
```

**How it works (year-by-year)**:
```python
For each year:
    1. Record starting portfolio value
    2. Calculate portfolio return = (SPY return × 60%) + (AGG return × 40%)
    3. Grow portfolio: value = value × (1 + return)
    4. Calculate withdrawal with inflation: withdrawal × (1.03 ^ year)
    5. Subtract withdrawal: value = value - withdrawal
    6. Check if depleted: if value ≤ 0, mark as failed
```

**Method 2: Monte Carlo Simulation**

Runs thousands of simulations with random return paths:

```python
results = engine.run_monte_carlo(
    num_simulations=5000,      # Run 5000 different scenarios
    method='bootstrap',         # How to generate random returns
    seed=42,                   # For reproducibility
    show_progress=True         # Show progress bar
)
```

**Output**: `MonteCarloResults` object:
```python
results.success_rate           # 0.95 = 95% of paths succeeded
results.median_final_value     # $5,200,000 (typical outcome)
results.mean_final_value       # $6,800,000 (average, skewed by big wins)
results.percentiles           # {'5th': $800k, '25th': $2.5M, ..., '95th': $15M}
results.paths                 # List of all 5000 SimulationPath objects
results.portfolio_values_matrix  # Shape: (5000, 30) - values over time
```

**How Monte Carlo works**:
```python
For each of 5000 simulations:
    1. Generate random 30-year return sequence (bootstrap from history)
    2. Run single_path simulation with those returns
    3. Record: success/failure, final value, year-by-year values

Then calculate statistics:
    - Success rate = (# succeeded) / 5000
    - Percentiles = sort final values, pick 5th, 25th, 50th, 75th, 95th
    - Portfolio matrix = stack all paths for plotting
```

---

### 4. `retirement_visualization.py` - PLOTS

**What it does**: Creates 3 clear, professional plots using standard matplotlib.

#### Class: `RetirementVisualizer`

```python
from retirement_visualization import RetirementVisualizer

visualizer = RetirementVisualizer(config)
```

**Plot 1: Fan Chart** (Most Important!)

Shows the range of possible outcomes:

```python
visualizer.plot_fan_chart(results, save_path='fan_chart.png', show=True)
```

**What you see**:
- **Light blue area**: 90% of outcomes fall here (5th-95th percentile)
- **Dark blue area**: 50% of outcomes (25th-75th percentile)
- **Blue line**: Median outcome (50th percentile)
- **Red dashed line**: Zero (depletion level)
- **Gray dotted line**: Starting portfolio ($1M)

**How to read it**:
- If blue areas stay above red line → portfolio survives
- If areas touch red line → some simulations depleted
- Wider spread = more uncertainty

**Plot 2: Summary Dashboard**

4-panel overview:

```python
visualizer.plot_summary_dashboard(results, save_path='dashboard.png')
```

**Panels**:
1. **Success Rate**: Big number (95%) with color coding
   - Green: ≥90% (safe)
   - Orange: 70-90% (moderate risk)
   - Red: <70% (high risk)

2. **Final Value Distribution**: Histogram showing how final values are distributed

3. **Percentile Table**: Quick reference for best/worst/median cases

4. **Configuration Summary**: Your inputs and key results

**Plot 3: Sample Paths**

Spaghetti plot showing individual simulations:

```python
visualizer.plot_sample_paths(results, num_paths=100)
```

**What you see**:
- **Green lines**: Successful paths
- **Red lines**: Failed paths (depleted)
- **Blue thick line**: Median path

**Convenience method - Create all 3 at once**:

```python
visualizer.plot_all(
    results,
    output_dir='../plots/retirement/',
    show=True  # Display plots
)
# Creates: fan_chart.png, summary_dashboard.png, sample_paths.png
```

---

### 5. `demo_retirement.py` - EXAMPLE USAGE

**What it does**: Complete end-to-end example showing how to use everything.

**Run it**:
```bash
uv run python demo_retirement.py
```

**What it does**:
1. Creates a standard retirement scenario ($ 1M portfolio, $40k/year withdrawal)
2. Runs 1000 Monte Carlo simulations
3. Displays results to console
4. Exports CSV files with detailed data
5. Creates 3 visualization plots
6. Interprets results (safe/risky)

**Modify it to try different scenarios**:

```python
# Scenario 1: Conservative (3% withdrawal)
config = RetirementConfig(
    initial_portfolio=1_000_000,
    annual_withdrawal=30_000,  # Changed from 40k
    ...
)

# Scenario 2: Aggressive (80% stocks)
config = RetirementConfig(
    ...
    current_portfolio={'SPY': 0.8, 'AGG': 0.2},  # 80/20 instead of 60/40
)

# Scenario 3: Longer retirement (40 years)
config = RetirementConfig(
    start_date='2024-01-01',
    end_date='2064-01-01',  # Changed from 2054
    ...
)
```

---

## Data Flow Diagram

```
┌─────────────────────┐
│  RetirementConfig   │ ← You define: portfolio, withdrawal, dates
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│      FinData        │ ← Downloads historical data from Yahoo Finance
│                     │   Caches to ../data/
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  RetirementEngine   │ ← Runs simulations:
│                     │   • Samples random returns (bootstrap)
│  run_monte_carlo()  │   • Simulates year-by-year portfolio
│                     │   • Tracks withdrawals with inflation
└──────────┬──────────┘   • Detects depletion
           │
           ↓
┌─────────────────────┐
│ MonteCarloResults   │ ← Statistics:
│                     │   • Success rate
│                     │   • Percentiles
│                     │   • All simulation paths
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│RetirementVisualizer │ ← Creates plots:
│                     │   • Fan chart (percentile bands)
│ plot_all()          │   • Summary dashboard
│                     │   • Sample paths
└─────────────────────┘
```

---

## Key Concepts

### Monte Carlo Simulation

**Why use it?**
The future is uncertain. Instead of assuming one return sequence, we simulate thousands of possible futures based on historical data.

**How it works**:
1. **Historical data**: We have 20+ years of SPY and AGG returns
2. **Bootstrap sampling**: Randomly pick actual historical days to build a 30-year future
3. **Repeat 1000s of times**: Each repetition uses different random samples
4. **Analyze distribution**: See how often portfolio survives, typical outcomes, worst cases

**Example**:
- Simulation #1: Might hit a crash in year 5, then recover
- Simulation #2: Might have steady 7% growth whole time
- Simulation #3: Might hit TWO crashes (2008-like and COVID-like)
- ...
- Simulation #5000: Different random sequence

Then we count: How many survived? What's the median ending value?

### Bootstrap vs Parametric Sampling

**Bootstrap** (recommended):
```python
# Actual historical daily returns
history = [-2%, +1%, +3%, -1%, +2%, ...]

# Randomly pick days with replacement
year_1 = [+3%, -1%, +2%, +1%, -2%, ...]  # Random picks from history
year_2 = [+1%, +2%, -1%, +3%, -2%, ...]  # Different random picks
```
- **Pro**: Uses real market behavior (captures crashes, volatility clustering)
- **Con**: Can't create scenarios worse than history

**Parametric**:
```python
# Fit normal distribution to history
mean = 0.08
std = 0.18

# Generate from normal distribution
year_1 = random.normal(mean, std)  # e.g., [+0.05, +0.12, -0.03, ...]
```
- **Pro**: Can generate any scenario
- **Con**: Assumes normal distribution (markets aren't perfectly normal)

### Withdrawal with Inflation

```python
Year 0: Withdraw $40,000
Year 1: Withdraw $40,000 × 1.03 = $41,200  (3% inflation)
Year 2: Withdraw $40,000 × 1.03² = $42,436
Year 3: Withdraw $40,000 × 1.03³ = $43,709
...
Year 30: Withdraw $40,000 × 1.03³⁰ = $97,103
```

This maintains purchasing power - $40k in year 30 buys the same as $40k in year 0.

### Portfolio Return Calculation

With 60% SPY, 40% AGG:

```python
SPY return for year = 10%
AGG return for year = 3%

Portfolio return = (0.6 × 10%) + (0.4 × 3%) = 6% + 1.2% = 7.2%
```

This is a **weighted average** based on your allocation.

---

## Interpreting Results

### Success Rate

| Rate | Interpretation | Action |
|------|---------------|--------|
| ≥95% | Very safe | On track, could potentially increase withdrawals |
| 85-95% | Relatively safe | Good plan, monitor regularly |
| 70-85% | Moderate risk | Consider reducing withdrawals or working longer |
| <70% | High risk | Need significant plan adjustment |

### Percentiles

- **5th percentile**: Worst 5% of outcomes (very pessimistic scenario)
- **25th percentile**: Worse than average, but not terrible
- **50th percentile (median)**: Typical outcome - half better, half worse
- **75th percentile**: Better than average
- **95th percentile**: Best 5% of outcomes (very optimistic scenario)

**Example**:
```
5th:  $800,000    ← In worst 5% cases, you end with $800k
25th: $2,500,000
50th: $5,200,000  ← Most likely outcome
75th: $9,100,000
95th: $18,000,000 ← In best 5% cases, you end with $18M
```

Even in the 5th percentile (pessimistic), portfolio survived with $800k. This means 95% of simulations ended with MORE than $800k.

### Fan Chart Reading

```
Portfolio Value
    │
$5M ┤     ╱────────────────────
    │   ╱  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  ← 95th percentile
$4M ┤  ╱  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    │ ╱ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← 75th percentile
$3M ┤╱ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    │ ████████████████████████▓   ← Median (50th)
$2M ┤ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
$1M ┤ ░░░░░░░░░░░░░░░░░░░░░░░  ← 25th percentile
    │ ░░░░░░░░░░░░░░░░░░░░░░   ← 5th percentile
 $0 ┤──────────────────────────  ← Depletion
    └──────────────────────────→
    0      10     20     30 years
```

- **Dark area** (50% of outcomes): Most likely range
- **Light area** (90% of outcomes): Nearly all scenarios fall here
- **If areas stay well above $0**: Safe plan
- **If touching $0**: Some scenarios deplete

---

## Files Generated

### CSV Files (in `../results/retirement/`)

1. **summary_statistics.csv**: One-row summary
   - Num simulations, success rate, median, mean, std dev, percentiles

2. **final_values.csv**: Final portfolio value for each simulation
   - Columns: Simulation number, final value, success (True/False)

3. **percentile_paths.csv**: Percentile values for each year
   - Columns: Year, 5th, 25th, 50th, 75th, 95th percentile values

### Plot Files (in `../plots/retirement/`)

1. **fan_chart.png**: Main visualization (percentile bands)
2. **summary_dashboard.png**: 4-panel overview
3. **sample_paths.png**: Individual simulation paths

---

## Common Modifications

### Change Withdrawal Amount

```python
config = RetirementConfig(
    initial_portfolio=1_000_000,
    annual_withdrawal=50_000,  # Try $50k instead of $40k
    ...
)
```

### Change Portfolio Allocation

```python
# More aggressive (80% stocks)
current_portfolio={'SPY': 0.8, 'AGG': 0.2}

# More conservative (40% stocks)
current_portfolio={'SPY': 0.4, 'AGG': 0.6}

# All stocks (risky!)
current_portfolio={'SPY': 1.0}
```

### Change Time Horizon

```python
# 40-year retirement
config = RetirementConfig(
    start_date='2024-01-01',
    end_date='2064-01-01',  # 40 years instead of 30
    ...
)
```

### Add More Assets

```python
# Include gold and international stocks
current_portfolio={
    'SPY': 0.50,   # US stocks
    'VXUS': 0.20,  # International stocks
    'AGG': 0.25,   # Bonds
    'GLD': 0.05    # Gold
}
```

### More Simulations (Higher Accuracy)

```python
results = engine.run_monte_carlo(
    num_simulations=10000,  # 10x more simulations
    ...
)
# Takes longer but gives smoother percentile curves
```

---

## Testing

All code is thoroughly tested:

```bash
# Test configuration
uv run pytest test_retirement_config.py -v

# Test return sampling
uv run pytest test_fin_data_sampling.py -v

# Test simulation engine
uv run pytest test_retirement_engine.py -v
```

**Total: 61 tests, all passing ✅**

---

## Performance

- **1000 simulations**: ~15 seconds
- **5000 simulations**: ~60 seconds (1 minute)
- **10000 simulations**: ~2 minutes

Runs faster on second execution due to data caching.

---

## Next Steps (Future Enhancements)

1. **Advanced Withdrawal Strategies**:
   - Guyton-Klinger (dynamic adjustments)
   - Variable Percentage Withdrawal (VPW)
   - Floor-ceiling strategies

2. **Accumulation Phase**:
   - Simulate saving period before retirement
   - Portfolio optimization during accumulation

3. **Financial Events**:
   - Mortgages, Social Security, pensions
   - One-time expenses (healthcare, travel)

4. **Regime-Switching Monte Carlo**:
   - Market regimes (bull/bear/normal)
   - Different return distributions per regime

---

## Questions?

**Q: Why does my success rate change if I run twice without a seed?**
A: Each run uses different random samples. Set `seed=42` for reproducible results.

**Q: What's the difference between median and mean?**
A: Median = middle value (half higher, half lower). Mean = average (can be skewed by extreme outcomes). Median is usually more meaningful for planning.

**Q: Can I use real historical sequences instead of random?**
A: Yes! Just provide specific returns to `run_single_path()`. This lets you test "what if 2008 happens in year 5?"

**Q: How do I know if my plan is safe?**
A: Look at success rate (aim for >90%) and 5th percentile (worst-case scenario). If 5th percentile is still positive, you can survive even bad luck.

**Q: Should I use bootstrap or parametric?**
A: Bootstrap (default) is recommended - it uses real market behavior including crashes.
