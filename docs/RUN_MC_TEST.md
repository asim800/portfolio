# Monte Carlo Lifecycle Test - Walkthrough

## Quick Start - Rerun the Test

```bash
cd /home/saahmed1/coding/python/fin/port/src
uv run python visualize_mc_lifecycle.py
```

Or with cleaner output:
```bash
uv run python visualize_mc_lifecycle.py 2>&1 | grep -v "FutureWarning\|INFO -\|completed"
```

## Configuration File

**Location**: [configs/test_simple_buyhold.json](../configs/test_simple_buyhold.json)

**Key Settings**:
```json
{
  "start_date": "2024-01-01",           // Backtest start
  "end_date": "2025-09-19",             // Backtest end
  "retirement_date": "2035-01-01",      // Retirement begins
  "simulation_horizon_years": 30,       // 30-year retirement

  "contribution_amount": 1000,          // $1K biweekly
  "contribution_frequency": "biweekly", // 26 times/year
  "employer_match_rate": 0.5,           // 50% match
  "employer_match_cap": 10000,          // Max $10K/year

  "withdrawal_strategy": "constant_inflation_adjusted",
  "annual_withdrawal_amount": 40000,    // $40K/year
  "inflation_rate": 0.03                // 3% inflation
}
```

## Code Flow - Step by Step

### Step 1: Main Entry Point
**File**: [visualize_mc_lifecycle.py:301](visualize_mc_lifecycle.py#L301)
```python
def main():
    # Loads configuration and runs full simulation
```

### Step 2: Load System Configuration
**File**: [visualize_mc_lifecycle.py:306](visualize_mc_lifecycle.py#L306)
```python
config = SystemConfig.from_json('../configs/test_simple_buyhold.json')
```

**Configuration Class**: [system_config.py:18](system_config.py#L18)
- Defines all global settings
- Validates parameters in `__post_init__()` at line 96
- Utility methods:
  - `get_accumulation_years()` - line 233
  - `get_decumulation_years()` - line 260
  - `get_contribution_config()` - line 306
  - `get_withdrawal_config()` - line 332

### Step 3: Load Historical Data
**File**: [visualize_mc_lifecycle.py:318](visualize_mc_lifecycle.py#L318)
```python
# Read tickers from file
tickers_df = pd.read_csv(config.ticker_file)

# Download historical data
fin_data = FinData(start_date=config.start_date, end_date=config.end_date)
fin_data.fetch_ticker_data(tickers)
returns_data = fin_data.get_returns_data(tickers)
```

**Data Source**: [tickers.txt](../tickers.txt)
- Format: `Symbol,Weight`
- Current: BIL (25%), MSFT (25%), NVDA (25%), SPY (25%)

### Step 4: Estimate Parameters
**File**: [visualize_mc_lifecycle.py:327](visualize_mc_lifecycle.py#L327)
```python
# Calculate annualized statistics from historical returns
mean_returns = returns_data.mean().values * 252  # Daily → Annual
cov_matrix = returns_data.cov().values * 252     # Daily → Annual
```

**Key Calculation**:
```python
portfolio_mean = np.dot(weights, mean_returns)        # Expected return
portfolio_std = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))  # Volatility
```

### Step 5: Run Accumulation Monte Carlo
**File**: [visualize_mc_lifecycle.py:365](visualize_mc_lifecycle.py#L365)

**Get Contribution Config**:
```python
contribution_config = config.get_contribution_config()
# Returns:
# {
#   'amount': 1000,
#   'frequency': 'biweekly',
#   'contributions_per_year': 26,
#   'annual_contribution': 26000,
#   'employer_match_rate': 0.5,
#   'employer_match_cap': 10000
# }
```

**Run Simulation**: [visualize_mc_lifecycle.py:390](visualize_mc_lifecycle.py#L390)
```python
accumulation_values = run_accumulation_mc(
    initial_value=100_000,
    mean_returns=mean_returns,
    cov_matrix=cov_matrix,
    weights=weights,
    years=9,                            # 2025-09-19 to 2035-01-01
    num_simulations=1000,
    annual_contribution=36000,          # $26K + $10K match
    employer_match_rate=0.5,
    employer_match_cap=10000,
    seed=42
)
```

**MC Accumulation Function**: [visualize_mc_lifecycle.py:23](visualize_mc_lifecycle.py#L23)

**Key Logic** (per year, per simulation):
```python
# Line 66: Start with previous year's balance
portfolio_value = values[sim, year - 1]

# Line 69: Add contributions (beginning of year)
if annual_contribution > 0:
    employer_match = annual_contribution * employer_match_rate
    if employer_match_cap:
        employer_match = min(employer_match, employer_match_cap)
    total_contribution = annual_contribution + employer_match
    portfolio_value += total_contribution

# Line 79: Sample random return
annual_return = np.random.normal(portfolio_mean, portfolio_std)

# Line 82: Apply return
portfolio_value *= (1 + annual_return)
```

### Step 6: Run Decumulation Monte Carlo
**File**: [visualize_mc_lifecycle.py:411](visualize_mc_lifecycle.py#L411)

```python
withdrawal_config = config.get_withdrawal_config()
# Returns:
# {
#   'strategy': 'constant_inflation_adjusted',
#   'annual_amount': 40000,
#   'inflation_rate': 0.03
# }

decumulation_values, success = run_decumulation_mc(
    initial_values=final_acc_values,     # Start with ending accumulation values
    mean_returns=mean_returns,
    cov_matrix=cov_matrix,
    weights=weights,
    annual_withdrawal=40000,
    inflation_rate=0.03,
    years=30
)
```

**MC Decumulation Function**: [visualize_mc_lifecycle.py:88](visualize_mc_lifecycle.py#L88)

**Key Logic** (per year, per simulation):
```python
# Line 120: Sample return
annual_return = np.random.normal(portfolio_mean, portfolio_std)

# Line 123: Apply return
portfolio_value = values[sim, year - 1] * (1 + annual_return)

# Line 126: Calculate inflation-adjusted withdrawal
withdrawal = annual_withdrawal * ((1 + inflation_rate) ** (year - 1))

# Line 129: Subtract withdrawal
portfolio_value -= withdrawal

# Line 132: Check for depletion
if portfolio_value <= 0:
    portfolio_value = 0
    success[sim] = False
```

### Step 7: Create Visualizations
**File**: [visualize_mc_lifecycle.py:430](visualize_mc_lifecycle.py#L430)

**Fan Chart**: [visualize_mc_lifecycle.py:120](visualize_mc_lifecycle.py#L120)
- Shows percentile bands (5th, 25th, 50th, 75th, 95th)
- Green: Accumulation, Blue: Decumulation
- Output: `../plots/test/mc_lifecycle_fan_chart.png`

**Spaghetti Plot (Log Scale)**: [visualize_mc_lifecycle.py:205](visualize_mc_lifecycle.py#L205)
- Shows 200 individual simulation paths
- Log10 y-axis for wide range visualization
- Output: `../plots/test/mc_lifecycle_spaghetti_log.png`

## Output Files

**Generated Visualizations**:
```
/home/saahmed1/coding/python/fin/port/plots/test/
├── mc_lifecycle_fan_chart.png        # Percentile bands
└── mc_lifecycle_spaghetti_log.png    # Individual paths (log scale)
```

## Key Validation Points

### 1. Configuration Loading
**Validate**: [system_config.py:96-190](system_config.py#L96)
- Check dates are valid
- Check contribution/withdrawal parameters are positive
- Check frequencies are valid

### 2. Contribution Calculation
**Validate**: [system_config.py:316](system_config.py#L316)
```python
freq_map = {'weekly': 52, 'biweekly': 26, 'monthly': 12, 'annual': 1}
contributions_per_year = freq_map[contribution_frequency]
annual_contribution = contribution_amount * contributions_per_year
```

Expected: $1,000 × 26 = $26,000/year

### 3. Employer Match Calculation
**Validate**: [visualize_mc_lifecycle.py:71-73](visualize_mc_lifecycle.py#L71)
```python
employer_match = annual_contribution * employer_match_rate  # $26K × 0.5 = $13K
employer_match = min(employer_match, employer_match_cap)    # min($13K, $10K) = $10K
total_contribution = annual_contribution + employer_match    # $26K + $10K = $36K
```

Expected: $36,000/year total

### 4. Accumulation Results
**Expected median after 9 years**:
- Starting: $100,000
- Contributions: $36,000/year × 9 years = $324,000
- Returns: ~34.84% annually on growing balance
- **Median final: ~$3.1M**

### 5. Decumulation Success Rate
**Validate**: [visualize_mc_lifecycle.py:419](visualize_mc_lifecycle.py#L419)
```python
success_rate = success.mean()
```

Expected: 100% (withdrawing $40K/year from $3M+ portfolio is very safe)

## Quick Debug Commands

```bash
# Check configuration loads correctly
uv run python -c "from system_config import SystemConfig; c = SystemConfig.from_json('../configs/test_simple_buyhold.json'); print(c.get_contribution_config())"

# Check contribution calculation
uv run python -c "from system_config import SystemConfig; c = SystemConfig.from_json('../configs/test_simple_buyhold.json'); cc = c.get_contribution_config(); print(f'Annual: \${cc[\"annual_contribution\"]:,}')"

# Check accumulation/decumulation years
uv run python -c "from system_config import SystemConfig; c = SystemConfig.from_json('../configs/test_simple_buyhold.json'); print(f'Acc: {c.get_accumulation_years():.1f}y, Dec: {c.get_decumulation_years():.1f}y')"

# Run just the accumulation MC
uv run python -c "
import numpy as np
from visualize_mc_lifecycle import run_accumulation_mc
np.random.seed(42)
values = run_accumulation_mc(
    initial_value=100_000,
    mean_returns=np.array([0.05, 0.22, 0.90, 0.22]),
    cov_matrix=np.eye(4) * 0.01,
    weights=np.array([0.25, 0.25, 0.25, 0.25]),
    years=9,
    num_simulations=10,
    annual_contribution=36000,
    employer_match_rate=0.5,
    employer_match_cap=10000
)
print(f'Median final: \${np.median(values[:, -1]):,.0f}')
"
```

## Modify and Test

**Try different contribution amounts**:
Edit [configs/test_simple_buyhold.json](../configs/test_simple_buyhold.json):
```json
"contribution_amount": 2000,  // Double to $2K biweekly
```

**Try monthly contributions**:
```json
"contribution_amount": 4000,
"contribution_frequency": "monthly",
```

**Remove employer match**:
```json
"employer_match_rate": 0.0,
```

Then rerun:
```bash
uv run python visualize_mc_lifecycle.py
```
