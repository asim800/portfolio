# MC Path Generation - Quick Reference Card

## 🚀 Quick Commands

```bash
# Run full simulation with visualizations
uv run python visualize_mc_lifecycle.py

# Run validation with detailed output
uv run python test_mc_validation.py

# Interactive testing
uv run python
>>> from mc_path_generator import MCPathGenerator
>>> import numpy as np
>>> # Test code here...
```

## 📁 File Map (What's Where)

| File | Purpose | Key Lines |
|------|---------|-----------|
| **mc_path_generator.py** | Asset-level path generation | 89-143: `generate_paths()` |
| **visualize_mc_lifecycle.py** | MC simulation + viz | 25-114: accumulation, 116-191: decumulation |
| **system_config.py** | Configuration | 67-120: `from_json()` |
| **fin_data.py** | Market data fetching | 175-205: `fetch_ticker_data()` |
| **VALIDATE_MC_PATHS.md** | Step-by-step guide | Full validation walkthrough |
| **test_mc_validation.py** | Automated validation | Complete test suite |

## 🔑 Key Concepts (The Important Stuff)

### Asset-Level Path Generation (The Big Change)
```python
# OLD (univariate, portfolio-level):
period_return = np.random.normal(portfolio_mean, portfolio_std)

# NEW (multivariate, asset-level):
asset_returns = np.random.multivariate_normal(period_mean, period_cov, size=N)
portfolio_return = np.dot(weights, asset_returns)
```

### Why Multivariate Matters
- ✅ Preserves correlations (e.g., MSFT-NVDA: 0.73)
- ✅ Enables portfolio comparison on same scenarios
- ✅ More realistic market modeling

### Return Scaling (Critical for Accuracy)
```python
# For biweekly (26 periods/year):
period_mean = annual_mean / 26
period_cov = annual_cov / 26      # NOT annual_std / sqrt(26) for covariance!
period_std = annual_std / sqrt(26) # Only for std deviation
```

### Portfolio Return Calculation
```python
# Each period:
asset_returns = paths[sim, period, :]  # Shape: (num_assets,)
portfolio_return = np.dot(weights, asset_returns)  # Weighted average
```

## 🎯 Common Debugging Scenarios

### "My results are different each run"
```python
# Check seed is set
path_generator = MCPathGenerator(..., seed=42)  # ← Must set seed!
```

### "Returns are too large/small"
```python
# Check frequency scaling
periods_per_year = 26  # For biweekly
period_cov = annual_cov / periods_per_year  # Should be DIVIDED by 26
```

### "Correlations not preserved"
```python
# Verify using multivariate normal (not univariate)
stats = path_generator.get_summary_statistics()
print(stats['empirical_correlation'])  # Should match historical
```

### "Shape mismatch error"
```python
# Verify periods calculation
total_periods = years * periods_per_year  # e.g., 9 × 26 = 234
assert paths.shape == (num_sims, total_periods, num_assets)
```

## 📊 Expected Output Ranges

### Path Shapes
```python
acc_paths:  (1000, 234, 4)  # 1000 sims × 234 biweekly periods × 4 assets
dec_paths:  (1000, 30, 4)   # 1000 sims × 30 annual periods × 4 assets
acc_values: (1000, 10)      # 1000 sims × 10 year boundaries
dec_values: (1000, 31)      # 1000 sims × 31 year boundaries
```

### Statistical Validation
```python
mean_error < 0.01           # Empirical mean matches theoretical
correlation_error < 0.05    # Empirical correlation matches historical
```

### Simulation Results (for test config)
```
Accumulation (9 years, $1K biweekly, 50% match):
  5th percentile:  ~$1.4M
  50th percentile: ~$3.4M
  95th percentile: ~$8.6M

Decumulation (30 years, $40K/year withdrawal):
  Success rate: 100%
  Median final: ~$19B (very high due to strong returns)
```

## 🔍 Quick Validation Checks

### 1. Path Generation
```python
path_generator = MCPathGenerator(tickers, mean_returns, cov_matrix, seed=42)
paths = path_generator.generate_paths(1000, 234, periods_per_year=26)

# Verify shape
assert paths.shape == (1000, 234, 4)

# Verify statistics
stats = path_generator.get_summary_statistics()
assert stats['mean_error'].max() < 0.01
```

### 2. Portfolio Return Calculation
```python
# Extract one period
sim = 0
period = 0
asset_returns = paths[sim, period, :]

# Calculate portfolio return
portfolio_return = np.dot(weights, asset_returns)

# Verify it's used in simulation
# (check run_accumulation_mc line 103)
```

### 3. Accumulation Simulation
```python
acc_values = run_accumulation_mc(
    initial_value=100_000,
    weights=weights,
    asset_returns_paths=paths,
    ...
)

# Verify final values are reasonable
median_final = np.percentile(acc_values[:, -1], 50)
assert median_final > 1_000_000  # Should grow significantly
```

## 🛠️ Adding Breakpoints for Investigation

```python
# In visualize_mc_lifecycle.py

# Before path generation (main function, ~line 450)
import ipdb; ipdb.set_trace()
acc_paths = path_generator.generate_paths(...)

# Inside accumulation loop (run_accumulation_mc, ~line 99)
import ipdb; ipdb.set_trace()
asset_returns = asset_returns_paths[sim, period - 1, :]
portfolio_return = np.dot(weights, asset_returns)

# Inside decumulation loop (run_decumulation_mc, ~line 170)
import ipdb; ipdb.set_trace()
asset_returns = asset_returns_paths[sim, year - 1, :]
```

## 📈 Path Through Code (Main Workflow)

```
main() [visualize_mc_lifecycle.py:375]
  ↓
[1] Load config & data
  → SystemConfig.from_json() [system_config.py:67]
  → FinData.fetch_ticker_data() [fin_data.py:175]
  → FinData.get_returns_data() [fin_data.py:305]
  ↓
[2] Create MCPathGenerator
  → MCPathGenerator.__init__() [mc_path_generator.py:52]
  ↓
[3] Generate accumulation paths
  → MCPathGenerator.generate_paths() [mc_path_generator.py:89]
      - Scales: period_cov = annual_cov / periods_per_year
      - Samples: np.random.multivariate_normal()
      - Reshapes: (N*M, K) → (N, M, K)
  ↓
[4] Generate decumulation paths
  → MCPathGenerator.generate_paths() [mc_path_generator.py:89]
      (with different seed and periods_per_year=1)
  ↓
[5] Run accumulation simulation
  → run_accumulation_mc() [visualize_mc_lifecycle.py:25]
      For each sim, each period:
        - Add contributions + employer match
        - Get asset_returns from paths
        - Calculate portfolio_return = dot(weights, asset_returns)
        - Apply return to portfolio value
  ↓
[6] Run decumulation simulation
  → run_decumulation_mc() [visualize_mc_lifecycle.py:116]
      For each sim, each year:
        - Get asset_returns from paths
        - Calculate portfolio_return = dot(weights, asset_returns)
        - Apply return to portfolio value
        - Subtract inflation-adjusted withdrawal
  ↓
[7] Create visualizations
  → plot_lifecycle_mc() [visualize_mc_lifecycle.py:193]
  → plot_spaghetti_log() [visualize_mc_lifecycle.py:282]
```

## 💡 Key Invariants (Must Always Be True)

```python
# 1. Path dimensions
assert acc_paths.shape[0] == num_simulations
assert acc_paths.shape[1] == years * periods_per_year
assert acc_paths.shape[2] == num_assets

# 2. Portfolio return is weighted average
portfolio_return = np.dot(weights, asset_returns)
assert abs(portfolio_return - np.sum(weights * asset_returns)) < 1e-10

# 3. Weights sum to 1
assert abs(weights.sum() - 1.0) < 1e-10

# 4. Statistical validity
stats = path_generator.get_summary_statistics()
assert stats['mean_error'].max() < 0.01

# 5. Correlation preserved
empirical_corr = stats['empirical_correlation']
theoretical_corr = np.corrcoef(returns_data.values.T)
assert np.allclose(empirical_corr, theoretical_corr, atol=0.05)
```

## 🔗 Related Documentation

- **Full validation guide**: [VALIDATE_MC_PATHS.md](VALIDATE_MC_PATHS.md)
- **Test validation script**: [test_mc_validation.py](test_mc_validation.py)
- **Main documentation**: [CLAUDE.md](CLAUDE.md) (lines 247-362)
- **Example config**: [../configs/test_simple_buyhold.json](../configs/test_simple_buyhold.json)

## 📞 Quick Help

**Q: How do I verify correlations are preserved?**
```python
stats = path_generator.get_summary_statistics()
print(stats['empirical_correlation'])
```

**Q: How do I test one simulation path?**
```python
sim_idx = 0
path_df = path_generator.get_path_dataframe(sim_idx, '2025-01-01', 'biweekly')
print(path_df.head())
```

**Q: How do I compare two portfolios on same paths?**
```python
# Run portfolio A
acc_values_A = run_accumulation_mc(initial_value, weights_A, acc_paths, ...)

# Run portfolio B on SAME paths
acc_values_B = run_accumulation_mc(initial_value, weights_B, acc_paths, ...)

# Compare results
print(f"Portfolio A median: ${np.median(acc_values_A[:, -1]):,.0f}")
print(f"Portfolio B median: ${np.median(acc_values_B[:, -1]):,.0f}")
```

**Q: How do I save/load paths for reuse?**
```python
# Save
path_generator.save_paths('../data/mc_paths_1000x234.pkl')

# Load
loaded_gen = MCPathGenerator.load_paths('../data/mc_paths_1000x234.pkl')
paths = loaded_gen.paths  # (1000, 234, 4)
```
