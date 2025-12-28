# Rebalancing Strategies vs Optimization Methods - Explained

## TL;DR - Quick Answer

**`rebalancing_strategies`** = **STATIC** portfolio strategies (use fixed rules)
- Examples: Buy & hold, equal weight, 60/40 target allocation
- **No optimization** - just follow simple rules

**`optimization_methods`** = **DYNAMIC** portfolio strategies (use math to optimize)
- Examples: Mean-variance, robust optimization, risk parity
- **Uses optimization** - calculates best weights based on data

---

## The Confusion

Both parameters create portfolios that rebalance over time, but they use **completely different approaches**:

```python
config = create_custom_config(
    rebalancing_strategies=['buy_and_hold', 'target_weight'],  # STATIC rules
    optimization_methods=['mean_variance', 'robust_mean_variance']  # DYNAMIC math
)
```

This creates **4 portfolios total**:
1. Buy & hold (static)
2. Target weight (static)
3. Mean-variance (optimized)
4. Robust mean-variance (optimized)

---

## Detailed Comparison

### Rebalancing Strategies (Static/Rule-Based)

**Definition:** Portfolios that follow **fixed, predetermined rules** without looking at returns/correlations.

| Strategy | Rule | Rebalancing Logic |
|----------|------|-------------------|
| `buy_and_hold` | Keep original weights, let them drift | **Never rebalances** - share counts stay fixed |
| `target_weight` | Always return to original weights | Rebalances to 40% SPY, 30% AGG, 20% NVDA, 10% GLD every period |
| `equal_weight` | Always equal allocation | Rebalances to 25% each asset every period |
| `spy_only` | 100% SPY | Rebalances to 100% SPY (benchmark) |

**Key characteristics:**
- ✅ Simple, easy to understand
- ✅ No data/calculation required
- ✅ Predictable behavior
- ❌ Doesn't adapt to market conditions
- ❌ Ignores correlations and expected returns

**Example - Target Weight Portfolio:**
```python
# Every 90 days, reset to original weights
weights = [0.4, 0.3, 0.2, 0.1]  # Always the same, no matter what
```

---

### Optimization Methods (Dynamic/Math-Based)

**Definition:** Portfolios that **calculate optimal weights** using optimization algorithms based on historical data.

| Method | Optimization Goal | Mathematical Approach |
|--------|------------------|----------------------|
| `mean_variance` | Maximize return per unit of risk | Solves: max(return - risk_aversion × variance) |
| `robust_mean_variance` | Like mean-variance but conservative | Accounts for parameter uncertainty |
| `risk_parity` | Equal risk contribution from each asset | Allocates based on volatility |
| `min_variance` | Minimize portfolio volatility | Finds lowest-risk combination |
| `max_sharpe` | Maximize risk-adjusted return | Maximize (return - rf) / volatility |

**Key characteristics:**
- ✅ Adapts to market conditions
- ✅ Uses correlations intelligently
- ✅ Can outperform static strategies
- ❌ Complex calculations required
- ❌ Requires sufficient historical data
- ❌ Can be sensitive to estimation errors

**Example - Mean Variance Portfolio:**
```python
# Every 90 days, recalculate optimal weights
mean_returns = historical_data.mean()  # Estimate expected returns
cov_matrix = historical_data.cov()     # Estimate covariances

# Solve optimization problem
weights = optimize(
    objective=maximize(return - risk_aversion * variance),
    constraints=[weights.sum() == 1, weights >= 0]
)
# Result: weights = [0.35, 0.15, 0.42, 0.08]  # Changes every period!
```

---

## When to Use Each

### Use Rebalancing Strategies When:

✅ You want **benchmarks** to compare against optimization
✅ You want **simple, explainable** portfolios
✅ You have **limited data** (not enough for reliable optimization)
✅ You want to test if **active management adds value** (compare to buy & hold)

**Common use case:**
```python
# "Does optimization beat a simple 60/40 portfolio?"
rebalancing_strategies=['target_weight'],  # 60/40 benchmark
optimization_methods=['mean_variance']     # Test if this beats it
```

### Use Optimization Methods When:

✅ You want to **actively manage** the portfolio
✅ You have **sufficient historical data** (at least 2+ periods)
✅ You want to **adapt** to changing correlations
✅ You want to **maximize risk-adjusted returns**

**Common use case:**
```python
# "Which optimization approach works best?"
optimization_methods=['mean_variance', 'robust_mean_variance', 'risk_parity']
```

---

## Real-World Example

**Scenario:** You have $100K to invest in 4 assets (SPY, AGG, NVDA, GLD).

### Option 1: Rebalancing Strategy (Static)
```python
# 60/40 stock/bond allocation, rebalanced quarterly
rebalancing_strategies=['target_weight']

# Result over 10 years:
# Q1: 60% SPY, 40% AGG  (initial)
# Q2: 60% SPY, 40% AGG  (rebalanced back)
# Q3: 60% SPY, 40% AGG  (rebalanced back)
# ...always the same weights every quarter
```

**Pros:** Simple, predictable, low turnover
**Cons:** Doesn't adapt if bonds start outperforming stocks

### Option 2: Optimization Method (Dynamic)
```python
# Mean-variance optimization, rebalanced quarterly
optimization_methods=['mean_variance']

# Result over 10 years:
# Q1: 55% SPY, 45% AGG  (stocks look risky, favor bonds)
# Q2: 70% SPY, 30% AGG  (stocks recovering, increase allocation)
# Q3: 50% SPY, 50% AGG  (high volatility, reduce stocks)
# ...weights change based on market conditions
```

**Pros:** Adapts to markets, potentially higher returns
**Cons:** More complex, higher turnover, requires data

---

## How They Work Together in the System

When you specify both:
```python
config = create_custom_config(
    rebalancing_strategies=['buy_and_hold', 'target_weight'],
    optimization_methods=['mean_variance', 'robust_mean_variance']
)
```

The system creates **4 independent portfolios** and runs them **side-by-side**:

1. **Buy & Hold Portfolio**
   - Strategy: buy_and_hold
   - Rebalances: Never
   - Weights: Drift with market performance

2. **Target Weight Portfolio**
   - Strategy: target_weight
   - Rebalances: Every 90 days
   - Weights: Reset to [0.4, 0.3, 0.2, 0.1] each time

3. **Mean Variance Portfolio**
   - Method: mean_variance
   - Rebalances: Every 90 days
   - Weights: Optimized each time based on expanding window data

4. **Robust Mean Variance Portfolio**
   - Method: robust_mean_variance
   - Rebalances: Every 90 days
   - Weights: Conservatively optimized each time

**At the end:** You get performance comparison showing which approach worked best!

---

## Code Architecture

### Where They're Used

**Rebalancing Strategies** → `rebalancing_strategies.py`
```python
# Simple rule-based classes
class BuyAndHoldStrategy(BaseRebalancingStrategy):
    def calculate_target_weights(self, current_weights, ...):
        return current_weights  # Never change!

class TargetWeightStrategy(BaseRebalancingStrategy):
    def calculate_target_weights(self, current_weights, ...):
        return self.target_weights  # Always same!
```

**Optimization Methods** → `portfolio_optimizer.py`
```python
# Complex optimization using cvxpy
class PortfolioOptimizer:
    def optimize(self, method, mean_returns, cov_matrix, ...):
        if method == 'mean_variance':
            # Solve: maximize return - risk_aversion * variance
            weights = cp.Variable(n)
            objective = mean_returns @ weights - risk_aversion * cp.quad_form(weights, cov_matrix)
            problem = cp.Problem(cp.Maximize(objective), constraints)
            problem.solve()
            return weights.value  # Different every time!
```

---

## Performance Comparison Example

**Typical Output:**

| Portfolio | Total Return | Sharpe Ratio | Max Drawdown | Turnover |
|-----------|--------------|--------------|--------------|----------|
| buy_and_hold | +85% | 0.92 | -22% | 0% |
| target_weight | +88% | 0.95 | -20% | 12% |
| mean_variance | +102% | 1.15 | -18% | 25% |
| robust_mean_variance | +95% | 1.08 | -16% | 20% |

**Insights:**
- **Buy & hold** underperformed but had zero turnover
- **Target weight** beat buy & hold with modest rebalancing
- **Mean variance** had highest return but also highest turnover
- **Robust mean variance** balanced return and stability

---

## Common Patterns

### Pattern 1: Benchmark Comparison
```python
# Test if optimization adds value over passive
rebalancing_strategies=['buy_and_hold'],  # Passive benchmark
optimization_methods=['mean_variance']    # Active strategy
```

### Pattern 2: Strategy Tournament
```python
# Test multiple static strategies
rebalancing_strategies=['buy_and_hold', 'target_weight', 'equal_weight', 'spy_only'],
optimization_methods=[]  # None
```

### Pattern 3: Optimization Comparison
```python
# Test multiple optimization approaches
rebalancing_strategies=[],  # None
optimization_methods=['mean_variance', 'robust_mean_variance', 'min_variance', 'max_sharpe']
```

### Pattern 4: Full Comparison (like our example)
```python
# Test everything!
rebalancing_strategies=['buy_and_hold', 'target_weight'],
optimization_methods=['mean_variance', 'robust_mean_variance']
# Creates 4 portfolios total
```

---

## FAQ

**Q: Can I use only one of them?**
A: Yes! Either works alone:
```python
# Just static strategies
config = create_custom_config(
    rebalancing_strategies=['buy_and_hold', 'equal_weight'],
    optimization_methods=[]  # Empty list
)

# Just optimization
config = create_custom_config(
    rebalancing_strategies=[],  # Empty list
    optimization_methods=['mean_variance']
)
```

**Q: Why does comparison_mode require 2+ optimization methods?**
A: The `comparison_mode=True` flag enables **optimization method comparison**, so you need at least 2 to compare. It doesn't count rebalancing strategies.

**Q: Which is better - static or optimized?**
A: **It depends!**
- Static strategies are more robust to estimation errors
- Optimized strategies can outperform in ideal conditions
- **Best practice:** Test both and see what works for your data

**Q: Can I mix them?**
A: Yes, that's exactly what the example does! Run both types side-by-side and compare performance.

---

## Summary Table

| Aspect | Rebalancing Strategies | Optimization Methods |
|--------|----------------------|---------------------|
| **Approach** | Rule-based | Math-based |
| **Weights** | Fixed or predetermined | Calculated each period |
| **Data needed** | None | Historical returns + covariances |
| **Complexity** | Low | High |
| **Turnover** | Low to medium | Medium to high |
| **Adaptability** | None | High |
| **Robustness** | High | Depends on data quality |
| **Use case** | Benchmarks, simple strategies | Active management |
| **Examples** | Buy & hold, 60/40 | Mean-variance, risk parity |

---

## Bottom Line

**Think of it this way:**

- **`rebalancing_strategies`** = What would a **human investor** do with simple rules?
  - "I'll keep 60% stocks and 40% bonds"
  - "I'll just buy and hold forever"

- **`optimization_methods`** = What would a **quantitative analyst** do with math?
  - "Let me calculate the optimal weights to maximize Sharpe ratio"
  - "Let me minimize variance subject to return constraints"

Both are useful! The power is in **comparing them** to see if the extra complexity of optimization is worth it.
