# Quick Start Guide - New Portfolio Architecture

## Overview

The refactored portfolio system uses a clean, modular architecture with:
- **Portfolio class**: Single portfolio with weights/returns history
- **PortfolioTracker**: Manages multiple portfolios for comparison
- **PortfolioOptimizer**: Pure numpy optimization (no pandas)
- **Factory methods**: Clear, self-documenting portfolio creation

## Basic Usage

### 1. Simple Portfolio Backtest

```python
from portfolio import Portfolio
from portfolio_tracker import PortfolioTracker
from period_manager import PeriodManager
import pandas as pd

# Define assets and weights
asset_names = ['SPY', 'AGG', 'BIL']
weights = pd.Series([0.6, 0.3, 0.1], index=asset_names)

# Create portfolios using factory methods
buy_hold = Portfolio.create_buy_and_hold(asset_names, weights)
target_wt = Portfolio.create_target_weight(asset_names, weights, rebalance_days=30)

# Load your returns data (pandas DataFrame)
# returns = pd.DataFrame(...)  # dates × assets
buy_hold.ingest_simulated_data(returns)
target_wt.ingest_simulated_data(returns)

# Set up tracker and run backtest
tracker = PortfolioTracker()
tracker.add_portfolio('buy_and_hold', buy_hold)
tracker.add_portfolio('target_weight', target_wt)

period_manager = PeriodManager(returns, rebalancing_period_days=30)
tracker.run_backtest(period_manager)

# Get results
summary = tracker.get_portfolio_summary_statistics()
print(summary)
```

### 2. With Optimization

```python
from portfolio_optimizer import PortfolioOptimizer

# Create optimizer
optimizer = PortfolioOptimizer(risk_free_rate=0.02)

# Create optimized portfolio
mean_var = Portfolio.create_optimized(
    asset_names,
    weights,
    optimizer,
    method='mean_variance',
    rebalance_days=30
)

# Load data and run backtest
mean_var.ingest_simulated_data(returns)
tracker.add_portfolio('mean_variance', mean_var)
tracker.run_backtest(period_manager)
```

### 3. With Real Market Data

```python
from fin_data import FinData

# Initialize FinData
fin_data = FinData(
    start_date='2024-01-01',
    end_date='2024-12-31',
    cache_dir='../data'
)

# Load tickers from file
tickers_df = fin_data.load_tickers('../tickers.txt')
asset_names = tickers_df['Symbol'].tolist()

# Get market data
returns = fin_data.get_returns_data(asset_names)
baseline_weights = fin_data.get_baseline_weights(asset_names)
baseline_weights_series = pd.Series(baseline_weights, index=asset_names)

# Create portfolio
portfolio = Portfolio.create_buy_and_hold(asset_names, baseline_weights_series)
portfolio.returns_data = returns.copy()  # Direct assignment for real data

# Run backtest
tracker = PortfolioTracker()
tracker.add_portfolio('buy_and_hold', portfolio)

period_manager = PeriodManager(returns, rebalancing_period_days=30)
tracker.run_backtest(period_manager)
```

### 4. Complete Workflow (Using PerformanceEngine)

```python
from main import PortfolioOrchestrator
from config import RebalancingConfig

# Create config
config = RebalancingConfig(
    start_date='2024-01-01',
    end_date='2024-12-31',
    rebalancing_strategies=['buy_and_hold', 'target_weight'],
    optimization_methods=['mean_variance', 'robust_mean_variance'],
    rebalancing_period_days=30
)

# Run full analysis
orchestrator = PortfolioOrchestrator(config)
orchestrator.run_full_analysis('../tickers.txt')
```

## Factory Methods

### Available Portfolio Types

| Method | Description | Rebalancing |
|--------|-------------|-------------|
| `create_buy_and_hold()` | Buy and hold, weights drift | Never |
| `create_target_weight()` | Rebalance to target weights | Every N days |
| `create_equal_weight()` | 1/N allocation | Every N days |
| `create_spy_only()` | 100% SPY (benchmark) | Every N days |
| `create_optimized()` | Uses optimizer | Every N days |

### Examples

```python
# Buy-and-hold (weights drift with market)
buy_hold = Portfolio.create_buy_and_hold(
    asset_names=['SPY', 'AGG'],
    initial_weights=pd.Series([0.6, 0.4], index=['SPY', 'AGG'])
)

# Target weight (rebalance to baseline)
target = Portfolio.create_target_weight(
    asset_names=['SPY', 'AGG'],
    target_weights=pd.Series([0.6, 0.4], index=['SPY', 'AGG']),
    rebalance_days=30
)

# Equal weight (1/N)
equal = Portfolio.create_equal_weight(
    asset_names=['SPY', 'AGG', 'BIL'],
    rebalance_days=30
)

# Optimized (mean-variance, robust, etc.)
optimizer = PortfolioOptimizer(risk_free_rate=0.02)
optimized = Portfolio.create_optimized(
    asset_names=['SPY', 'AGG'],
    initial_weights=pd.Series([0.6, 0.4], index=['SPY', 'AGG']),
    optimizer=optimizer,
    method='mean_variance',
    rebalance_days=30
)
```

## Data Ingestion

### Option 1: Simulated Data

```python
import numpy as np

# Create simulated returns
dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
returns = pd.DataFrame({
    'SPY': np.random.randn(len(dates)) * 0.01,
    'AGG': np.random.randn(len(dates)) * 0.005
}, index=dates)

portfolio.ingest_simulated_data(returns)
```

### Option 2: Real Data (via FinData)

```python
fin_data = FinData('2024-01-01', '2024-12-31')
returns = fin_data.get_returns_data(['SPY', 'AGG'])

portfolio.returns_data = returns.copy()  # Direct assignment
```

## Optimization Methods

Available optimization methods:
- `mean_variance`: Classic Markowitz mean-variance
- `robust_mean_variance`: Robust optimization with uncertainty sets
- `min_variance`: Minimum variance portfolio
- `max_sharpe`: Maximum Sharpe ratio
- `risk_parity`: Equal risk contribution
- `max_diversification`: Maximum diversification ratio
- `hierarchical_risk_parity`: HRP allocation
- `black_litterman`: Black-Litterman with views

### Using Optimizer Directly

```python
optimizer = PortfolioOptimizer(risk_free_rate=0.02)

# Calculate statistics (pandas)
mean_returns = returns.mean() * 252  # Annualize
cov_matrix = returns.cov() * 252

# Optimize (convert to numpy at boundary)
result = optimizer.optimize(
    method='mean_variance',
    mean_returns=mean_returns.values,  # pandas → numpy
    cov_matrix=cov_matrix.values       # pandas → numpy
)

if result['status'] == 'optimal':
    optimal_weights = result['weights']  # numpy array
    print(f"Return: {result['expected_return']:.2%}")
    print(f"Volatility: {result['volatility']:.2%}")
    print(f"Sharpe: {result['sharpe_ratio']:.3f}")
```

## Results and Analysis

### Summary Statistics

```python
# Get summary for all portfolios
summary = tracker.get_portfolio_summary_statistics()

# Columns: Total_Return, Annual_Return, Volatility,
#          Sharpe_Ratio, Max_Drawdown, Num_Periods
print(summary)
```

### Individual Portfolio Metrics

```python
# Get specific portfolio data
portfolio = tracker.portfolios['buy_and_hold']

# Returns history
returns_history = portfolio.returns_history

# Portfolio values over time
values = portfolio.portfolio_values

# Current weights
current_weights = portfolio.get_current_weights()

# Summary statistics
summary = portfolio.get_summary_statistics()
```

### Export Results

```python
# Export all tracker data
tracker.export_to_csv('../results/my_backtest')

# Export individual portfolio
portfolio.export_to_csv('../results/my_portfolio')

# Saved files:
# - {name}_weights.csv
# - {name}_returns.csv
# - {name}_values.csv
# - {name}_summary.csv
```

## Key Design Principles

### 1. Pandas in Portfolio, Numpy in Optimizer

```python
# Portfolio uses pandas (labeled, debuggable)
weights_history = portfolio.weights_history  # DataFrame
returns = portfolio.returns_history          # Series

# Optimizer uses numpy (fast, standard)
result = optimizer.optimize(
    mean_returns=mu_array,   # numpy
    cov_matrix=sigma_array   # numpy
)
```

### 2. Clear Conversions at Boundary

```python
# Always convert explicitly
mean_returns = returns_df.mean()        # pandas Series
result = optimizer.optimize(
    mean_returns=mean_returns.values    # → numpy
)

# Convert back with labels
optimal_weights = pd.Series(
    result['weights'],  # numpy
    index=asset_names   # add labels
)
```

### 3. Factory Methods for Clarity

```python
# ✓ Good: Self-documenting
portfolio = Portfolio.create_buy_and_hold(assets, weights)

# ✗ Avoid: Complex constructors
portfolio = Portfolio(assets, weights, BuyAndHoldStrategy(), ...)
```

## Common Patterns

### Compare Multiple Strategies

```python
strategies = {
    'buy_hold': Portfolio.create_buy_and_hold(assets, weights),
    'equal_wt': Portfolio.create_equal_weight(assets),
    'spy_only': Portfolio.create_spy_only(assets),
    'optimized': Portfolio.create_optimized(assets, weights, optimizer, 'mean_variance')
}

tracker = PortfolioTracker()
for name, portfolio in strategies.items():
    portfolio.ingest_simulated_data(returns)
    tracker.add_portfolio(name, portfolio)

tracker.run_backtest(period_manager)
summary = tracker.get_portfolio_summary_statistics()
```

### Compare Optimization Methods

```python
optimizer = PortfolioOptimizer(risk_free_rate=0.02)
methods = ['mean_variance', 'robust_mean_variance', 'min_variance', 'max_sharpe']

tracker = PortfolioTracker()
for method in methods:
    portfolio = Portfolio.create_optimized(assets, weights, optimizer, method)
    portfolio.ingest_simulated_data(returns)
    tracker.add_portfolio(method, portfolio)

tracker.run_backtest(period_manager)
```

### Analyze Different Rebalancing Frequencies

```python
for freq in [30, 60, 90]:
    portfolio = Portfolio.create_target_weight(
        assets, weights, rebalance_days=freq,
        name=f'rebal_{freq}d'
    )
    portfolio.ingest_simulated_data(returns)
    tracker.add_portfolio(f'rebal_{freq}d', portfolio)
```

## Troubleshooting

### Issue: "Must call load_data before run_backtest"
**Solution**: Ensure portfolio has data before backtesting
```python
portfolio.ingest_simulated_data(returns)  # or
portfolio.returns_data = returns.copy()
```

### Issue: Optimization fails silently
**Solution**: Check result status
```python
result = optimizer.optimize(...)
if result['status'] != 'optimal':
    print(f"Failed: {result.get('message', 'unknown')}")
```

### Issue: Weights don't sum to 1
**Solution**: Portfolio automatically normalizes, but check input
```python
weights = pd.Series([0.6, 0.4], index=['SPY', 'AGG'])
weights = weights / weights.sum()  # Normalize
```

### Issue: Missing asset in returns data
**Solution**: Ensure returns DataFrame has all assets
```python
assert all(asset in returns.columns for asset in asset_names)
```

## Running Examples

```bash
# Run validation tests
uv run python test_portfolio.py

# Run usage examples
uv run python example_new_architecture.py

# Run full analysis with config
uv run python main.py
```

## Next Steps

1. **Explore optimization methods**: Try different methods and compare
2. **Tune rebalancing frequency**: Test 30, 60, 90 day periods
3. **Add constraints**: Use min_weight, max_weight parameters
4. **Create custom strategies**: Extend BaseRebalancingStrategy
5. **Integrate with retirement simulation**: Use Portfolio class for Monte Carlo

## Additional Resources

- **Full Documentation**: See CLAUDE.md for complete system overview
- **Retirement Simulation**: See RETIREMENT_README.md for Monte Carlo features
- **API Reference**: Check docstrings in each module
- **Examples**: See example_new_architecture.py for working code
