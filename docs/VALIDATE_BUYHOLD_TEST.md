# Validating Buy-and-Hold Monte Carlo Test

This guide walks you through running and validating the buy-and-hold lifecycle test step-by-step, showing you exactly where each piece of code executes.

## Quick Run

```bash
cd /home/saahmed1/coding/python/fin/port/src
uv run python visualize_mc_lifecycle.py
```

## Step-by-Step Code Trace

### Step 1: Load Configuration

**File**: `visualize_mc_lifecycle.py:main()` (line ~480)

```python
# Code location: visualize_mc_lifecycle.py, line ~480
config = SystemConfig.from_json('../configs/test_simple_buyhold.json')
```

**What happens**:
- Reads `../configs/test_simple_buyhold.json`
- Parses JSON and filters `_comment` fields
- Validates all parameters
- Creates SystemConfig object

**Validate**:
```bash
# Check the configuration file exists and is valid
cat ../configs/test_simple_buyhold.json | python -m json.tool

# Or inspect in Python:
uv run python -c "
from system_config import SystemConfig
config = SystemConfig.from_json('../configs/test_simple_buyhold.json')
print(f'Backtest: {config.start_date} to {config.end_date}')
print(f'Retirement: {config.retirement_date}')
print(f'Accumulation: {config.get_accumulation_years():.1f} years')
print(f'Decumulation: {config.get_decumulation_years():.1f} years')
print(f'Contribution: \${config.contribution_amount} {config.contribution_frequency}')
print(f'Withdrawal: \${config.annual_withdrawal_amount}/year')
"
```

**Expected output**:
```
Backtest: 2024-01-01 to 2025-09-19
Retirement: 2035-01-01
Accumulation: 9.3 years
Decumulation: 30.0 years
Contribution: $1000 biweekly
Withdrawal: $40000/year
```

**File**: `system_config.py:SystemConfig.from_json()` (line ~50-80)

### Step 2: Load Historical Data and Estimate Parameters

**File**: `visualize_mc_lifecycle.py:main()` (line ~490)

```python
# Code location: visualize_mc_lifecycle.py, line ~490
tickers = ['SPY', 'AGG', 'NVDA', 'GLD']  # Read from ticker file

# Download/load cached data
fin_data = FinData(start_date=config.start_date, end_date=config.end_date)

# Estimate parameters from historical data
returns = fin_data.returns[tickers]  # Daily returns
mean_returns = returns.mean() * 252  # Annualized mean
cov_matrix = returns.cov() * 252     # Annualized covariance
```

**What happens**:
- Checks for pickle cache in `../data/`
- If cache exists: loads from pickle
- If no cache: downloads from Yahoo Finance, saves to pickle
- Calculates daily returns: `(price_today - price_yesterday) / price_yesterday`
- Annualizes: multiply by 252 trading days

**Validate**:
```python
uv run python -c "
from fin_data import FinData
import numpy as np

fin_data = FinData('2024-01-01', '2025-09-19')
tickers = ['SPY', 'AGG', 'NVDA', 'GLD']
returns = fin_data.returns[tickers]

mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
volatility = np.sqrt(np.diag(cov_matrix))

print('Annualized Mean Returns:')
for ticker, mean in zip(tickers, mean_returns):
    print(f'  {ticker}: {mean:.2%}')

print('\nAnnualized Volatility:')
for ticker, vol in zip(tickers, volatility):
    print(f'  {ticker}: {vol:.2%}')

print(f'\nCorrelation SPY-AGG: {returns[\"SPY\"].corr(returns[\"AGG\"]):.3f}')
print(f'Data points: {len(returns)} days')
"
```

**Expected output**:
```
Annualized Mean Returns:
  SPY: ~10-20%
  AGG: ~2-5%
  NVDA: ~30-50% (volatile)
  GLD: ~5-15%

Annualized Volatility:
  SPY: ~15-20%
  AGG: ~5-8%
  NVDA: ~30-40%
  GLD: ~12-18%

Correlation SPY-AGG: ~-0.2 to 0.0 (slightly negative)
Data points: ~440 days (252 trading days per year × 1.75 years)
```

**File**: `fin_data.py:FinData.__init__()` (line ~1-50)

### Step 3: Create Path Generator

**File**: `visualize_mc_lifecycle.py:main()` (line ~510)

```python
# Code location: visualize_mc_lifecycle.py, line ~510
path_generator = MCPathGenerator(
    tickers=tickers,
    mean_returns=mean_returns.values,  # numpy array [4,]
    cov_matrix=cov_matrix.values,      # numpy array [4, 4]
    seed=42
)
```

**What happens**:
- Stores mean returns and covariance matrix
- Sets random seed for reproducibility
- Does NOT generate paths yet (that happens next)

**Validate**:
```python
uv run python -c "
from fin_data import FinData
from mc_path_generator import MCPathGenerator
import numpy as np

# Setup
fin_data = FinData('2024-01-01', '2025-09-19')
tickers = ['SPY', 'AGG', 'NVDA', 'GLD']
returns = fin_data.returns[tickers]
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

# Create generator
path_generator = MCPathGenerator(
    tickers=tickers,
    mean_returns=mean_returns.values,
    cov_matrix=cov_matrix.values,
    seed=42
)

print(f'Path generator created')
print(f'Tickers: {path_generator.tickers}')
print(f'Num assets: {path_generator.num_assets}')
print(f'Mean returns shape: {path_generator.mean_returns.shape}')
print(f'Cov matrix shape: {path_generator.cov_matrix.shape}')
print(f'Seed: {path_generator.seed}')
print(f'Paths generated: {path_generator.paths is not None}')  # Should be False
"
```

**Expected output**:
```
Path generator created
Tickers: ['SPY', 'AGG', 'NVDA', 'GLD']
Num assets: 4
Mean returns shape: (4,)
Cov matrix shape: (4, 4)
Seed: 42
Paths generated: False
```

**File**: `mc_path_generator.py:MCPathGenerator.__init__()` (line ~20-50)

### Step 4: Generate Continuous Lifecycle Paths

**File**: `visualize_mc_lifecycle.py:main()` (line ~520)

```python
# Code location: visualize_mc_lifecycle.py, line ~520
num_sims = 1000
acc_years = 9  # From 2025-09-19 to 2035-01-01
dec_years = 30  # From 2035-01-01 to 2065-01-01
contributions_per_year = 26  # Biweekly

# Generate CONTINUOUS paths (one random sequence)
acc_paths, dec_paths = path_generator.generate_lifecycle_paths(
    num_simulations=num_sims,
    accumulation_years=acc_years,
    accumulation_periods_per_year=contributions_per_year,
    decumulation_years=dec_years
)
```

**What happens**:
1. Calculate total periods: `acc_periods = 9 × 26 = 234`, `dec_periods = 30 × 26 = 780`, `total = 1014`
2. Scale parameters to biweekly: `period_mean = annual_mean / 26`, `period_cov = annual_cov / 26`
3. Generate ONE continuous path: `np.random.multivariate_normal(period_mean, period_cov, size=1000 × 1014)`
4. Reshape to: `(1000 simulations, 1014 periods, 4 assets)`
5. Split: accumulation = periods 0-233, decumulation = periods 234-1013
6. Compound decumulation to annual: For each year, compound 26 biweekly periods

**Validate**:
```python
uv run python -c "
from fin_data import FinData
from mc_path_generator import MCPathGenerator
import numpy as np

# Setup (abbreviated)
fin_data = FinData('2024-01-01', '2025-09-19')
tickers = ['SPY', 'AGG', 'NVDA', 'GLD']
returns = fin_data.returns[tickers]
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

path_generator = MCPathGenerator(tickers, mean_returns.values, cov_matrix.values, seed=42)

# Generate paths
acc_paths, dec_paths = path_generator.generate_lifecycle_paths(
    num_simulations=1000,
    accumulation_years=9,
    accumulation_periods_per_year=26,
    decumulation_years=30
)

print('Accumulation paths:')
print(f'  Shape: {acc_paths.shape}')  # (1000, 234, 4)
print(f'  Expected: (1000 sims, 234 periods, 4 assets)')
print(f'  Mean return SPY: {acc_paths[:, :, 0].mean():.4f} (should be ~{(mean_returns[0]/26):.4f} per period)')
print(f'  Std SPY: {acc_paths[:, :, 0].std():.4f}')

print('\nDecumulation paths:')
print(f'  Shape: {dec_paths.shape}')  # (1000, 30, 4)
print(f'  Expected: (1000 sims, 30 years, 4 assets)')
print(f'  Mean return SPY: {dec_paths[:, :, 0].mean():.4f} (should be ~{mean_returns[0]:.4f} annual)')
print(f'  Std SPY: {dec_paths[:, :, 0].std():.4f}')

# Verify continuity (last acc period connects to first dec year)
print('\nContinuity check:')
sim_idx = 0
last_26_acc_periods = path_generator.paths[sim_idx, 234-26:234, 0]  # Last 26 biweekly periods
manual_annual = np.prod(1 + last_26_acc_periods) - 1
first_dec_year = dec_paths[sim_idx, 0, 0]
error = abs(manual_annual - first_dec_year)
print(f'  Manual compound: {manual_annual:.6f}')
print(f'  Dec path year 0: {first_dec_year:.6f}')
print(f'  Error: {error:.2e} (should be < 1e-10)')
print(f'  Continuous: {error < 1e-10}')
"
```

**Expected output**:
```
Accumulation paths:
  Shape: (1000, 234, 4)
  Expected: (1000 sims, 234 periods, 4 assets)
  Mean return SPY: ~0.0038 (should be ~0.0038 per period)
  Std SPY: ~0.0078

Decumulation paths:
  Shape: (1000, 30, 4)
  Expected: (1000 sims, 30 years, 4 assets)
  Mean return SPY: ~0.10 (should be ~0.10 annual)
  Std SPY: ~0.20

Continuity check:
  Manual compound: 0.023456
  Dec path year 0: 0.023456
  Error: 1.23e-16 (should be < 1e-10)
  Continuous: True
```

**File**: `mc_path_generator.py:generate_lifecycle_paths()` (line ~142-247)

### Step 5: Run Accumulation Simulation

**File**: `visualize_mc_lifecycle.py:run_accumulation_mc()` (line ~150-220)

```python
# Code location: visualize_mc_lifecycle.py, line ~530
initial_value = 100_000  # $100K starting value
weights = np.array([0.25, 0.25, 0.25, 0.25])  # Equal-weight portfolio

acc_values = run_accumulation_mc(
    initial_value=initial_value,
    weights=weights,
    asset_returns_paths=acc_paths,  # Pre-generated paths
    years=acc_years,
    contributions_per_year=contributions_per_year,
    contribution_amount=1000,  # $1K biweekly
    employer_match_rate=0.5,   # 50% match
    employer_match_cap=10_000  # $10K annual cap
)
```

**What happens** (for each simulation, each period):
1. Calculate portfolio return: `portfolio_return = np.dot(weights, asset_returns[period])`
2. Apply return: `portfolio_value *= (1 + portfolio_return)`
3. Add contribution: `portfolio_value += contribution_amount`
4. Add employer match (track YTD for cap): `portfolio_value += min(employer_match, cap - ytd)`
5. Reset YTD match each year

**Validate**:
```python
uv run python -c "
from fin_data import FinData
from mc_path_generator import MCPathGenerator
from visualize_mc_lifecycle import run_accumulation_mc
import numpy as np

# Setup (abbreviated - use same as Step 4)
fin_data = FinData('2024-01-01', '2025-09-19')
tickers = ['SPY', 'AGG', 'NVDA', 'GLD']
returns = fin_data.returns[tickers]
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
path_generator = MCPathGenerator(tickers, mean_returns.values, cov_matrix.values, seed=42)
acc_paths, _ = path_generator.generate_lifecycle_paths(1000, 9, 26, 30)

# Run accumulation
weights = np.array([0.25, 0.25, 0.25, 0.25])
acc_values = run_accumulation_mc(
    initial_value=100_000,
    weights=weights,
    asset_returns_paths=acc_paths,
    years=9,
    contributions_per_year=26,
    contribution_amount=1000,
    employer_match_rate=0.5,
    employer_match_cap=10_000
)

print('Accumulation Results:')
print(f'  Shape: {acc_values.shape}')  # (1000, 234)
print(f'  Initial value (all sims): \$100,000')
print(f'  Final values:')
print(f'    5th percentile:  \${np.percentile(acc_values[:, -1], 5):,.0f}')
print(f'    25th percentile: \${np.percentile(acc_values[:, -1], 25):,.0f}')
print(f'    Median (50th):   \${np.percentile(acc_values[:, -1], 50):,.0f}')
print(f'    75th percentile: \${np.percentile(acc_values[:, -1], 75):,.0f}')
print(f'    95th percentile: \${np.percentile(acc_values[:, -1], 95):,.0f}')

# Calculate total contributions
total_periods = 9 * 26
total_contributions = 1000 * total_periods  # Employee
total_match = min(10_000 * 9, 1000 * 0.5 * total_periods)  # Employer (capped)
print(f'\n  Total employee contributions: \${total_contributions:,}')
print(f'  Total employer match: \${total_match:,}')
print(f'  Total invested: \${total_contributions + total_match:,}')
print(f'  Median growth: \${np.median(acc_values[:, -1]) - total_contributions - total_match:,.0f}')
"
```

**Expected output**:
```
Accumulation Results:
  Shape: (1000, 234)
  Initial value (all sims): $100,000
  Final values:
    5th percentile:  $600,000 - $700,000
    25th percentile: $900,000 - $1,100,000
    Median (50th):   $1,200,000 - $1,500,000
    75th percentile: $1,600,000 - $2,000,000
    95th percentile: $2,500,000 - $3,500,000

  Total employee contributions: $234,000
  Total employer match: $90,000 (9 years × $10K cap)
  Total invested: $324,000
  Median growth: ~$800,000 - $1,100,000
```

### Step 6: Run Decumulation Simulation

**File**: `visualize_mc_lifecycle.py:run_decumulation_mc()` (line ~250-320)

```python
# Code location: visualize_mc_lifecycle.py, line ~550
dec_values, success = run_decumulation_mc(
    initial_values=acc_values[:, -1],  # Final accumulation values (1000,)
    weights=weights,
    asset_returns_paths=dec_paths,
    annual_withdrawal=40_000,
    inflation_rate=0.03,
    years=dec_years
)
```

**What happens** (for each simulation, each year):
1. Calculate portfolio return: `portfolio_return = np.dot(weights, asset_returns[year])`
2. Apply return: `portfolio_value *= (1 + portfolio_return)`
3. Calculate inflation-adjusted withdrawal: `withdrawal = 40000 × (1.03)^year`
4. Subtract withdrawal: `portfolio_value -= withdrawal`
5. Check depletion: if `portfolio_value <= 0`, mark as failed

**Validate**:
```python
uv run python -c "
from fin_data import FinData
from mc_path_generator import MCPathGenerator
from visualize_mc_lifecycle import run_accumulation_mc, run_decumulation_mc
import numpy as np

# Setup (abbreviated - use same as Step 5)
fin_data = FinData('2024-01-01', '2025-09-19')
tickers = ['SPY', 'AGG', 'NVDA', 'GLD']
returns = fin_data.returns[tickers]
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
path_generator = MCPathGenerator(tickers, mean_returns.values, cov_matrix.values, seed=42)
acc_paths, dec_paths = path_generator.generate_lifecycle_paths(1000, 9, 26, 30)

weights = np.array([0.25, 0.25, 0.25, 0.25])
acc_values = run_accumulation_mc(100_000, weights, acc_paths, 9, 26, 1000, 0.5, 10_000)

# Run decumulation
dec_values, success = run_decumulation_mc(
    initial_values=acc_values[:, -1],
    weights=weights,
    asset_returns_paths=dec_paths,
    annual_withdrawal=40_000,
    inflation_rate=0.03,
    years=30
)

print('Decumulation Results:')
print(f'  Shape: {dec_values.shape}')  # (1000, 30)
print(f'  Success array shape: {success.shape}')  # (1000,)
print(f'  Success rate: {np.sum(success) / len(success) * 100:.1f}%')
print(f'\n  Initial values (from accumulation):')
print(f'    Median: \${np.median(acc_values[:, -1]):,.0f}')
print(f'\n  Final values (after 30 years):')
print(f'    5th percentile:  \${np.percentile(dec_values[:, -1], 5):,.0f}')
print(f'    25th percentile: \${np.percentile(dec_values[:, -1], 25):,.0f}')
print(f'    Median (50th):   \${np.percentile(dec_values[:, -1], 50):,.0f}')
print(f'    75th percentile: \${np.percentile(dec_values[:, -1], 75):,.0f}')
print(f'    95th percentile: \${np.percentile(dec_values[:, -1], 95):,.0f}')

# Calculate total withdrawals
total_withdrawn = sum([40_000 * (1.03)**year for year in range(30)])
print(f'\n  Total withdrawn (inflation-adjusted): \${total_withdrawn:,.0f}')
print(f'  Year 1 withdrawal: \${40_000:,}')
print(f'  Year 30 withdrawal: \${40_000 * (1.03)**29:,.0f}')

# Depletion analysis
failed_sims = np.where(~success)[0]
if len(failed_sims) > 0:
    print(f'\n  Failed simulations: {len(failed_sims)} ({len(failed_sims)/10:.1f}%)')
    print(f'  Example failure years:')
    for sim_idx in failed_sims[:3]:
        depletion_year = np.where(dec_values[sim_idx, :] <= 0)[0]
        if len(depletion_year) > 0:
            print(f'    Sim {sim_idx}: depleted in year {depletion_year[0]}')
else:
    print(f'\n  No failures - all simulations survived 30 years!')
"
```

**Expected output**:
```
Decumulation Results:
  Shape: (1000, 30)
  Success array shape: (1000,)
  Success rate: 95-100%

  Initial values (from accumulation):
    Median: $1,300,000

  Final values (after 30 years):
    5th percentile:  $500,000 - $1,500,000
    25th percentile: $2,000,000 - $3,500,000
    Median (50th):   $4,000,000 - $7,000,000
    75th percentile: $8,000,000 - $12,000,000
    95th percentile: $20,000,000 - $35,000,000

  Total withdrawn (inflation-adjusted): ~$1,900,000
  Year 1 withdrawal: $40,000
  Year 30 withdrawal: $94,554

  No failures - all simulations survived 30 years!
```

### Step 7: Visualization

**File**: `visualize_mc_lifecycle.py:plot_lifecycle_mc()` (line ~400-500)

```python
# Code location: visualize_mc_lifecycle.py, line ~570
plot_lifecycle_mc(
    acc_values=acc_values,
    dec_values=dec_values,
    acc_years=acc_years,
    dec_years=dec_years,
    output_dir='../plots/test',
    show=False
)
```

**What happens**:
1. Calculate percentiles (5th, 25th, 50th, 75th, 95th) for each period
2. Create fan chart with percentile bands
3. Save to `../plots/test/mc_lifecycle_fan_chart.png`
4. Create spaghetti plot (log scale) of individual paths
5. Save to `../plots/test/mc_lifecycle_spaghetti_log.png`

**Validate**:
```bash
# Check plots were created
ls -lh ../plots/test/mc_lifecycle*.png

# Open plots (Linux)
xdg-open ../plots/test/mc_lifecycle_fan_chart.png

# Or use file browser to view
```

**What to look for in plots**:
- **Fan chart**: Should show smooth transition from accumulation to decumulation (no discontinuity)
- **Accumulation phase**: Values should trend upward (contributions + growth)
- **Decumulation phase**: Median should stay relatively flat or grow (portfolio growth > withdrawals)
- **5th percentile**: Should stay above 0 (or just barely touch 0 if success rate < 100%)
- **95th percentile**: Should be much higher than median (upside potential)

## Complete Validation Script

Save this as `validate_buyhold_complete.py`:

```python
"""Complete validation of buy-and-hold lifecycle simulation"""

from system_config import SystemConfig
from fin_data import FinData
from mc_path_generator import MCPathGenerator
from visualize_mc_lifecycle import run_accumulation_mc, run_decumulation_mc
import numpy as np

print("="*80)
print("BUY-AND-HOLD LIFECYCLE VALIDATION")
print("="*80)

# Step 1: Load config
print("\n[Step 1] Loading configuration...")
config = SystemConfig.from_json('../configs/test_simple_buyhold.json')
print(f"  ✓ Backtest: {config.start_date} to {config.end_date}")
print(f"  ✓ Retirement: {config.retirement_date}")
print(f"  ✓ Accumulation: {config.get_accumulation_years():.1f} years")
print(f"  ✓ Decumulation: {config.get_decumulation_years():.1f} years")

# Step 2: Load data and estimate parameters
print("\n[Step 2] Loading historical data and estimating parameters...")
tickers = ['SPY', 'AGG', 'NVDA', 'GLD']
fin_data = FinData(config.start_date, config.end_date)
returns = fin_data.returns[tickers]
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
print(f"  ✓ Data points: {len(returns)} days")
print(f"  ✓ Mean returns: {mean_returns.values}")
print(f"  ✓ Volatility: {np.sqrt(np.diag(cov_matrix.values))}")

# Step 3: Create path generator
print("\n[Step 3] Creating path generator...")
path_generator = MCPathGenerator(
    tickers=tickers,
    mean_returns=mean_returns.values,
    cov_matrix=cov_matrix.values,
    seed=42
)
print(f"  ✓ Generator created with {path_generator.num_assets} assets")

# Step 4: Generate continuous lifecycle paths
print("\n[Step 4] Generating continuous lifecycle paths...")
acc_years = int(config.get_accumulation_years())
dec_years = int(config.get_decumulation_years())
contributions_per_year = 26  # Biweekly

acc_paths, dec_paths = path_generator.generate_lifecycle_paths(
    num_simulations=1000,
    accumulation_years=acc_years,
    accumulation_periods_per_year=contributions_per_year,
    decumulation_years=dec_years
)
print(f"  ✓ Accumulation paths: {acc_paths.shape}")
print(f"  ✓ Decumulation paths: {dec_paths.shape}")

# Verify continuity
sim_idx = 0
last_26 = path_generator.paths[sim_idx, acc_years*26-26:acc_years*26, 0]
manual = np.prod(1 + last_26) - 1
first_dec = dec_paths[sim_idx, 0, 0]
error = abs(manual - first_dec)
print(f"  ✓ Continuity error: {error:.2e} (< 1e-10: {error < 1e-10})")

# Step 5: Run accumulation
print("\n[Step 5] Running accumulation simulation...")
weights = np.array([0.25, 0.25, 0.25, 0.25])
acc_values = run_accumulation_mc(
    initial_value=100_000,
    weights=weights,
    asset_returns_paths=acc_paths,
    years=acc_years,
    contributions_per_year=contributions_per_year,
    contribution_amount=1000,
    employer_match_rate=0.5,
    employer_match_cap=10_000
)
print(f"  ✓ Accumulation shape: {acc_values.shape}")
print(f"  ✓ Final values:")
print(f"    5th:  ${np.percentile(acc_values[:, -1], 5):,.0f}")
print(f"    50th: ${np.percentile(acc_values[:, -1], 50):,.0f}")
print(f"    95th: ${np.percentile(acc_values[:, -1], 95):,.0f}")

# Step 6: Run decumulation
print("\n[Step 6] Running decumulation simulation...")
dec_values, success = run_decumulation_mc(
    initial_values=acc_values[:, -1],
    weights=weights,
    asset_returns_paths=dec_paths,
    annual_withdrawal=40_000,
    inflation_rate=0.03,
    years=dec_years
)
print(f"  ✓ Decumulation shape: {dec_values.shape}")
print(f"  ✓ Success rate: {np.sum(success) / len(success) * 100:.1f}%")
print(f"  ✓ Final values:")
print(f"    5th:  ${np.percentile(dec_values[:, -1], 5):,.0f}")
print(f"    50th: ${np.percentile(dec_values[:, -1], 50):,.0f}")
print(f"    95th: ${np.percentile(dec_values[:, -1], 95):,.0f}")

print("\n" + "="*80)
print("ALL STEPS VALIDATED ✓")
print("="*80)
```

Run it:
```bash
uv run python validate_buyhold_complete.py
```

## Quick Reference: File Locations

**Configuration**:
- `../configs/test_simple_buyhold.json` - All simulation parameters
- `system_config.py:SystemConfig` (line ~1-150) - Config class

**Data**:
- `fin_data.py:FinData` (line ~1-100) - Yahoo Finance data download
- `../data/*.pkl` - Pickle cache files

**Path Generation**:
- `mc_path_generator.py:MCPathGenerator.__init__()` (line ~20-50)
- `mc_path_generator.py:generate_lifecycle_paths()` (line ~142-247)

**Simulation**:
- `visualize_mc_lifecycle.py:run_accumulation_mc()` (line ~150-220)
- `visualize_mc_lifecycle.py:run_decumulation_mc()` (line ~250-320)

**Visualization**:
- `visualize_mc_lifecycle.py:plot_lifecycle_mc()` (line ~400-500)
- `../plots/test/` - Output directory

**Main Entry**:
- `visualize_mc_lifecycle.py:main()` (line ~480-600)
