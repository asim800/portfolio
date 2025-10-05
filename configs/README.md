# Configuration System

The portfolio backtesting framework uses a **three-tier JSON configuration system**:

1. **System Config** - Global backtest settings (one file)
2. **Portfolio Configs** - Individual portfolio definitions (one file per portfolio)
3. **Comparison Configs** - Experiment definitions (one file per comparison)

---

## Directory Structure

```
configs/
├── system_config.json          # Global system settings
├── portfolios/                 # Portfolio definitions
│   ├── buy_and_hold_baseline.json
│   ├── target_weight_monthly.json
│   ├── optimized_mean_variance.json
│   ├── optimized_robust.json
│   ├── spy_benchmark.json
│   └── conservative_60_40.json
└── comparisons/                # Comparison experiments
    ├── active_vs_passive.json
    └── conservative_strategies.json
```

---

## 1. System Config (`system_config.json`)

**Purpose**: Global settings that apply to ALL portfolios in a backtest.

**Contains**:
- Data environment (dates, tickers, risk-free rate)
- Backtest engine settings (expanding window, min history)
- Performance metrics to track
- Output directories

**Example**:
```json
{
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "ticker_file": "../tickers.txt",
  "risk_free_rate": 0.02,

  "min_history_periods": 2,
  "use_expanding_window": true,
  "rolling_window_periods": 6,

  "metrics_to_track": [
    "returns", "volatility", "sharpe_ratio",
    "max_drawdown", "calmar_ratio", "beta"
  ],

  "save_plots": true,
  "plots_directory": "../plots/rebalancing",
  "save_results": true,
  "results_directory": "../results/rebalancing"
}
```

---

## 2. Portfolio Config (`portfolios/*.json`)

**Purpose**: Define a single portfolio's strategy, constraints, and parameters.

**Contains**:
- Portfolio identity (name, description)
- Strategy type (buy_and_hold, target_weight, optimized, etc.)
- Rebalancing frequency (pandas frequency string: 'ME', 'Q', '2W', etc.)
- Optimization method and parameters (if optimized)
- Constraints (min/max weights, long-only, etc.)
- Covariance calculation method
- Transaction costs
- Cash allocation (for mixed portfolios)

**Strategy Types**:
- `buy_and_hold` - Static weights, never rebalances
- `target_weight` - Rebalance to fixed target weights
- `equal_weight` - Rebalance to equal weights
- `spy_only` - 100% SPY allocation (benchmark)
- `optimized` - Uses optimization algorithm

**Optimization Methods** (for `strategy_type: "optimized"`):
- `mean_variance` - Classic Markowitz mean-variance
- `robust_mean_variance` - Robust optimization with uncertainty
- `risk_parity` - Equal risk contribution
- `min_variance` - Minimum variance
- `max_sharpe` - Maximum Sharpe ratio

**Example: Buy-and-Hold Baseline**
```json
{
  "name": "buy_and_hold_baseline",
  "description": "Static buy-and-hold baseline - never rebalances",

  "strategy_type": "buy_and_hold",
  "rebalancing_frequency": "ME",

  "initial_weights": {
    "AAPL": 0.2,
    "GOOGL": 0.2,
    "MSFT": 0.2,
    "AMZN": 0.2,
    "NVDA": 0.2
  },

  "optimization_method": null,
  "covariance_method": "sample",

  "long_only": true,
  "min_weight": 0.001,
  "max_weight": 0.4,

  "transaction_costs": 0.0,
  "cash_percentage": 0.0
}
```

**Example: Optimized Mean-Variance**
```json
{
  "name": "optimized_mean_variance",
  "description": "Mean-variance optimization rebalanced monthly",

  "strategy_type": "optimized",
  "rebalancing_frequency": "ME",

  "optimization_method": "mean_variance",
  "risk_aversion": 1.0,

  "covariance_method": "sample",

  "long_only": true,
  "min_weight": 0.001,
  "max_weight": 0.4,
  "max_concentration": 0.6,

  "transaction_costs": 0.0,
  "cash_percentage": 0.0
}
```

**Example: Conservative Mixed Portfolio**
```json
{
  "name": "conservative_60_40",
  "description": "40% cash, 60% optimized equity",

  "strategy_type": "optimized",
  "rebalancing_frequency": "ME",

  "optimization_method": "mean_variance",
  "risk_aversion": 2.0,

  "covariance_method": "sample",

  "long_only": true,
  "min_weight": 0.001,
  "max_weight": 0.25,

  "transaction_costs": 0.0,

  "cash_percentage": 0.4,
  "cash_interest_rate": 0.03
}
```

---

## 3. Comparison Config (`comparisons/*.json`)

**Purpose**: Define an experiment comparing multiple portfolios.

**Contains**:
- Which portfolios to include (list of portfolio JSON files)
- Which pairs to compare directly
- Output settings (subdirectory, file prefix)
- Analysis options (statistical tests, rolling metrics)
- Visualization options (what plots to generate)

**Example: Active vs Passive**
```json
{
  "name": "active_vs_passive",
  "description": "Compare optimization strategies against baselines",

  "portfolios": [
    "portfolios/buy_and_hold_baseline.json",
    "portfolios/spy_benchmark.json",
    "portfolios/optimized_mean_variance.json",
    "portfolios/optimized_robust.json"
  ],

  "comparison_pairs": [
    ["buy_and_hold_baseline", "optimized_mean_variance"],
    ["spy_benchmark", "optimized_mean_variance"]
  ],

  "output_prefix": "active_vs_passive",
  "output_subdirectory": "active_vs_passive",

  "include_statistical_tests": true,
  "include_rolling_metrics": false,

  "plot_cumulative_returns": true,
  "plot_drawdowns": true,
  "plot_rebalancing_events": true
}
```

---

## Rebalancing Frequencies

Portfolios support **any pandas resample frequency**:

**Standard Frequencies**:
- `'D'` - Daily
- `'W'` - Weekly
- `'ME'` - Month-end (recommended for monthly)
- `'MS'` - Month-start
- `'Q'` - Quarter-end
- `'A'` - Year-end

**Custom Multiples**:
- `'2W'` - Bi-weekly (every 2 weeks)
- `'21D'` - Every 21 days
- `'3ME'` - Quarterly (every 3 months)
- `'6ME'` - Semi-annual (every 6 months)

**Business Day Frequencies**:
- `'B'` - Business day
- `'BME'` - Business month-end (excludes weekends/holidays)
- `'BQE'` - Business quarter-end

---

## Loading Configurations

**Python Usage**:

```python
from system_config import load_system_config
from portfolio_config import load_portfolio_config
from comparison_config import load_comparison_config

# Load system config
system = load_system_config('configs/system_config.json')

# Load individual portfolio
portfolio = load_portfolio_config('configs/portfolios/buy_and_hold_baseline.json')

# Load comparison experiment
comparison = load_comparison_config('configs/comparisons/active_vs_passive.json')
```

---

## Benefits of This Architecture

1. **Separation of Concerns**:
   - System settings separate from portfolio settings
   - Experiments defined independently

2. **Flexibility**:
   - Easy to create new portfolios without code changes
   - Mix and match portfolios in different comparisons
   - Change global settings without touching portfolios

3. **Clarity**:
   - Each portfolio is self-contained in one JSON file
   - Easy to understand what each portfolio does
   - No hidden defaults or inheritance complexity

4. **Version Control Friendly**:
   - Track portfolio configurations in git
   - Easy to see what changed between experiments
   - Reproduce past results by checking out old configs

5. **Modularity**:
   - Portfolios don't know about each other
   - Comparisons compose portfolios together
   - Easy to add new portfolios or experiments
