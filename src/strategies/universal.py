"""
Universal Portfolio Strategies with Safe Haven Investing

This module contains all portfolio strategy classes that combine Universal Portfolio
theory with Safe Haven investing approaches, as described by Mark Spitznagel.

Core Insight from Spitznagel:
- Carrying a "drag" asset that loses money most of the time
- But provides massive payoff during crashes
- Actually IMPROVES geometric mean return (the real compounding rate)
- This is about path optimization, not just final wealth

Perfect synergy with Universal Portfolio:
- Both focus on geometric mean / log wealth
- Both are path-dependent
- Both account for volatility drag
"""

import numpy as np
from scipy.stats import norm


__all__ = [
    'ThreeAssetUniversalPortfolio',
    'ErgodicUniversalPortfolio',
    'ConditionalSafeHavenPortfolio',
    'UniversalPortfolioBasic',
    'BuyAndHold',
    'HierarchicalUniversalPortfolio',
    'DrawdownAwareUniversalPortfolio',
    'TimeVaryingErgodicityUP',
    'MultiTimeframeHierarchical',
    'MultiSafeHavenUP',
    'KellyUniversalPortfolio',
    'AsymmetricLossAverseUP',
    'SequentialThresholdUP',
    'VolatilityScaledUP',
    'MomentumEnhancedHierarchical',
    'ThreeLevelHierarchical',
    'DynamicGranularityUP',
]


# ============================================================================
# APPROACH 1: Three-Asset Universal Portfolio
# Add tail-risk hedge as third asset alongside stocks and bonds
# ============================================================================

class ThreeAssetUniversalPortfolio:
    """
    Universal Portfolio with Stocks, Bonds, and Tail Hedge

    The tail hedge:
    - Decays slowly in normal times (costs money)
    - Explodes during crashes (insurance payoff)
    """

    # Asset order for portfolio matrix columns
    ASSET_ORDER = ['stocks', 'bonds', 'tail_hedge']

    def __init__(self, n_portfolios=21):
        """
        Initialize with 3D simplex of portfolios
        Each portfolio: (stock%, bond%, hedge%)
        """
        self.portfolios = self._generate_3d_simplex(n_portfolios)
        self.n_portfolios = len(self.portfolios)
        self.log_weights = np.zeros(self.n_portfolios)
        self.wealth_history = []
        self.allocation_history = []

    def _generate_3d_simplex(self, n):
        """Generate portfolios on 3D simplex: x+y+z=1, x,y,z >= 0"""
        portfolios = []
        step = 1.0 / (n - 1)
        for i in range(n):
            for j in range(n - i):
                stock = i * step
                bond = j * step
                hedge = 1.0 - stock - bond
                if hedge >= -1e-10:  # numerical tolerance
                    portfolios.append([max(0, stock), max(0, bond), max(0, hedge)])
        return np.array(portfolios)

    def get_allocation(self):
        """Get current portfolio allocation"""
        weights = np.exp(self.log_weights - np.max(self.log_weights))
        weights = weights / np.sum(weights)
        allocation = np.sum(weights[:, np.newaxis] * self.portfolios, axis=0)
        return allocation

    def update(self, price_multipliers):
        """
        Update with price multipliers dict.

        Args:
            price_multipliers: Dict with 'stocks', 'bonds', 'tail_hedge' keys
                              Values are price multipliers (e.g., 1.02 = +2%)
        """
        # Convert dict to array in correct order for portfolio matrix multiplication
        mult_array = np.array([price_multipliers[asset] for asset in self.ASSET_ORDER])

        # Calculate return for each portfolio (weighted sum of multipliers)
        portfolio_returns = np.sum(self.portfolios * mult_array, axis=1)

        # Clip to prevent numerical issues
        portfolio_returns = np.clip(portfolio_returns, 0.01, 10.0)

        # Multiplicative update
        self.log_weights += np.log(portfolio_returns)

        # Numerical stability: normalize log weights
        self.log_weights = self.log_weights - np.mean(self.log_weights)

        # Calculate universal portfolio return
        weights = np.exp(self.log_weights - np.max(self.log_weights))
        weights = weights / np.sum(weights)
        universal_return = np.sum(weights * portfolio_returns)

        # Track history
        wealth = 1.0 if len(self.wealth_history) == 0 else self.wealth_history[-1] * universal_return
        self.wealth_history.append(wealth)
        self.allocation_history.append(self.get_allocation())

        return universal_return


# ============================================================================
# APPROACH 2: Ergodicity-Adjusted Universal Portfolio
# Explicitly penalize portfolios with high volatility
# ============================================================================

class ErgodicUniversalPortfolio:
    """
    Universal Portfolio that explicitly optimizes for geometric mean

    Key insight from ergodicity economics:
    - Arithmetic mean ≠ time average (what you actually experience)
    - High volatility reduces geometric mean
    - Should penalize risky paths even if they have high final wealth
    """

    def __init__(self, n_portfolios=101, ergodicity_factor=0.5):
        """
        ergodicity_factor: how much to penalize volatility (0 to 1)
        0 = standard Universal Portfolio
        1 = heavily penalize volatile portfolios
        """
        self.n_portfolios = n_portfolios
        self.portfolios = np.linspace(0, 1, n_portfolios)
        self.ergodicity_factor = ergodicity_factor

        # Track cumulative return AND volatility
        self.log_wealth = np.zeros(n_portfolios)
        self.squared_log_returns = np.zeros(n_portfolios)
        self.periods = 0

        self.wealth_history = []
        self.allocation_history = []

    def get_allocation(self):
        """Get allocation based on ergodicity-adjusted weights"""
        # Standard weight: e^(log wealth)
        # Ergodicity adjustment: e^(log wealth - λ*variance)
        variance = self.squared_log_returns / max(1, self.periods) - (self.log_wealth / max(1, self.periods))**2
        adjusted_log_wealth = self.log_wealth - self.ergodicity_factor * variance * self.periods

        weights = np.exp(adjusted_log_wealth - np.max(adjusted_log_wealth))
        weights = weights / np.sum(weights)

        return np.sum(weights * self.portfolios)

    def update(self, price_multipliers):
        """
        Update with price multipliers dict.

        Args:
            price_multipliers: Dict with 'bonds' and 'stocks' keys
                              Values are price multipliers (e.g., 1.02 = +2%)
        """
        bond_mult = price_multipliers['bonds']
        stock_mult = price_multipliers['stocks']

        # Calculate portfolio multipliers (weighted combination of asset multipliers)
        portfolio_returns = (1 - self.portfolios) * bond_mult + self.portfolios * stock_mult
        log_returns = np.log(portfolio_returns)

        # Update statistics
        self.log_wealth += log_returns
        self.squared_log_returns += log_returns ** 2
        self.periods += 1

        # Get current allocation and calculate universal portfolio multiplier
        allocation = self.get_allocation()
        universal_multiplier = (1 - allocation) * bond_mult + allocation * stock_mult

        # Track history
        wealth = 1.0 if len(self.wealth_history) == 0 else self.wealth_history[-1]
        self.wealth_history.append(wealth * universal_multiplier)
        self.allocation_history.append(allocation)

        return universal_multiplier


# ============================================================================
# APPROACH 3: Conditional Safe Haven Universal Portfolio
# Shift allocation based on market regime detection
# ============================================================================

class ConditionalSafeHavenPortfolio:
    """
    Uses Universal Portfolio normally, but shifts to safe haven during crisis

    Market regimes:
    - Normal: Use Universal Portfolio with stocks/bonds
    - Elevated risk: Increase bond/hedge allocation
    - Crisis: Go heavy into safe haven
    """

    def __init__(self, n_portfolios=101, lookback=20):
        self.normal_portfolio = UniversalPortfolioBasic(n_portfolios)
        self.safe_portfolio = UniversalPortfolioBasic(n_portfolios)  # Conservative
        self.lookback = lookback

        self.returns_buffer = []
        self.wealth_history = []
        self.allocation_history = []
        self.regime_history = []

    def detect_regime(self):
        """
        Detect market regime based on recent volatility and drawdown
        Returns: 'normal', 'elevated', or 'crisis'
        """
        if len(self.returns_buffer) < self.lookback:
            return 'normal'

        recent_mults = self.returns_buffer[-self.lookback:]
        stock_mults = np.array([m['stocks'] for m in recent_mults])

        # Calculate realized volatility
        volatility = np.std(np.log(stock_mults))

        # Calculate drawdown
        cumulative = np.cumprod(stock_mults)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        max_drawdown = np.max(drawdown)

        # Regime classification
        if max_drawdown > 0.15 or volatility > 0.2:
            return 'crisis'
        elif max_drawdown > 0.08 or volatility > 0.12:
            return 'elevated'
        else:
            return 'normal'

    def update(self, price_multipliers):
        """
        Update both portfolios and blend based on regime.

        Args:
            price_multipliers: Dict with 'bonds' and 'stocks' keys
                              Values are price multipliers (e.g., 1.02 = +2%)
        """
        bond_mult = price_multipliers['bonds']
        stock_mult = price_multipliers['stocks']
        self.returns_buffer.append(price_multipliers)

        # Update both portfolios
        self.normal_portfolio.update(price_multipliers)
        self.safe_portfolio.update(price_multipliers)

        # Detect regime
        regime = self.detect_regime()
        self.regime_history.append(regime)

        # Blend allocations based on regime
        normal_alloc = self.normal_portfolio.get_allocation()

        if regime == 'crisis':
            # Go conservative: reduce stock allocation by 50%
            allocation = normal_alloc * 0.5
        elif regime == 'elevated':
            # Moderate reduction: 75% of normal
            allocation = normal_alloc * 0.75
        else:
            # Normal times: full allocation
            allocation = normal_alloc

        # Calculate return
        universal_return = (1 - allocation) * bond_mult + allocation * stock_mult

        # Track history
        wealth = 1.0 if len(self.wealth_history) == 0 else self.wealth_history[-1]
        self.wealth_history.append(wealth * universal_return)
        self.allocation_history.append(allocation)

        return universal_return


class UniversalPortfolioBasic:
    """Basic Universal Portfolio for 2-asset allocation (bonds/stocks)"""
    def __init__(self, n_portfolios=101):
        self.portfolios = np.linspace(0, 1, n_portfolios)
        self.log_weights = np.zeros(n_portfolios)
        self.wealth_history = []
        self.allocation_history = []

    def get_allocation(self):
        weights = np.exp(self.log_weights - np.max(self.log_weights))
        weights = weights / np.sum(weights)
        return np.sum(weights * self.portfolios)

    def update(self, price_multipliers):
        """
        Update with price multipliers dict.

        Args:
            price_multipliers: Dict with 'bonds' and 'stocks' keys
                              Values are price multipliers (e.g., 1.02 = +2%)
        """
        bond_mult = price_multipliers['bonds']
        stock_mult = price_multipliers['stocks']
        portfolio_returns = (1 - self.portfolios) * bond_mult + self.portfolios * stock_mult
        self.log_weights += np.log(portfolio_returns)

        # Track wealth and allocation
        allocation = self.get_allocation()
        universal_return = (1 - allocation) * bond_mult + allocation * stock_mult
        wealth = 1.0 if len(self.wealth_history) == 0 else self.wealth_history[-1]
        self.wealth_history.append(wealth * universal_return)
        self.allocation_history.append(allocation)


# ============================================================================
# Buy and Hold Benchmark
# ============================================================================

class BuyAndHold:
    """
    Simple buy and hold strategy with fixed initial allocation
    No rebalancing - just let the portfolio drift with market returns

    This is what most individual investors actually do!
    """

    def __init__(self, initial_stock_pct=0.6):
        """
        Args:
            initial_stock_pct: Initial allocation to stocks (default 60%)
        """
        self.stock_pct = initial_stock_pct
        self.bond_pct = 1 - initial_stock_pct
        self.wealth_history = []
        self.allocation_history = []

    def update(self, price_multipliers):
        """
        Update wealth without rebalancing. Allocation drifts with relative performance.

        Args:
            price_multipliers: Dict with 'bonds' and 'stocks' keys
                              Values are price multipliers (e.g., 1.02 = +2%)
        """
        bond_mult = price_multipliers['bonds']
        stock_mult = price_multipliers['stocks']

        # Calculate new values
        new_stock_value = self.stock_pct * stock_mult
        new_bond_value = self.bond_pct * bond_mult
        total_value = new_stock_value + new_bond_value

        # Update allocations (they drift)
        self.stock_pct = new_stock_value / total_value
        self.bond_pct = new_bond_value / total_value

        # Calculate return
        portfolio_multiplier = new_stock_value + new_bond_value

        # Track history
        wealth = 1.0 if len(self.wealth_history) == 0 else self.wealth_history[-1]
        self.wealth_history.append(wealth * portfolio_multiplier)
        self.allocation_history.append(self.stock_pct)

        return portfolio_multiplier


# ============================================================================
# APPROACH 4: Hierarchical Universal Portfolio
# Two-level optimization: risky assets, then risk/safe-haven split
# ============================================================================

class HierarchicalUniversalPortfolio:
    """
    Level 1: Universal Portfolio over risky assets (stocks, commodities, etc.)
    Level 2: Universal Portfolio for allocation between risky bucket and safe haven

    This mirrors Spitznagel's approach:
    - Optimize growth assets separately
    - Then optimize insurance allocation
    """

    def __init__(self, n_portfolios_risky=51, n_portfolios_hedge=21):
        # Level 1: Allocation within risky assets (stocks vs bonds)
        self.risky_portfolio = UniversalPortfolioBasic(n_portfolios_risky)

        # Level 2: Allocation between risky bucket and tail hedge
        self.hedge_portfolios = np.linspace(0, 0.3, n_portfolios_hedge)  # 0-30% hedge
        self.hedge_log_weights = np.zeros(n_portfolios_hedge)

        self.wealth_history = []
        self.allocation_history = []
        self.hedge_allocation_history = []

    def update(self, price_multipliers):
        """
        Update with price multipliers dict.

        Args:
            price_multipliers: Dict with 'bonds', 'stocks', 'tail_hedge' keys
                              Values are price multipliers (e.g., 1.02 = +2%)
        """
        bond_mult = price_multipliers['bonds']
        stock_mult = price_multipliers['stocks']
        hedge_mult = price_multipliers['tail_hedge']

        # Level 1: Update risky portfolio allocation
        self.risky_portfolio.update({'bonds': bond_mult, 'stocks': stock_mult})
        risky_allocation = self.risky_portfolio.get_allocation()

        # Calculate risky bucket return
        risky_return = (1 - risky_allocation) * bond_mult + risky_allocation * stock_mult

        # Level 2: Determine hedge allocation
        # Each meta-portfolio: (1-hedge_pct)*risky + hedge_pct*hedge
        hedge_portfolio_returns = (1 - self.hedge_portfolios) * risky_return + self.hedge_portfolios * hedge_mult

        # Update hedge weights
        self.hedge_log_weights += np.log(hedge_portfolio_returns)

        # Get optimal hedge allocation
        hedge_weights = np.exp(self.hedge_log_weights - np.max(self.hedge_log_weights))
        hedge_weights = hedge_weights / np.sum(hedge_weights)
        optimal_hedge_pct = np.sum(hedge_weights * self.hedge_portfolios)

        # Calculate final return
        universal_return = (1 - optimal_hedge_pct) * risky_return + optimal_hedge_pct * hedge_mult

        # Track history
        wealth = 1.0 if len(self.wealth_history) == 0 else self.wealth_history[-1]
        self.wealth_history.append(wealth * universal_return)
        self.allocation_history.append(risky_allocation)
        self.hedge_allocation_history.append(optimal_hedge_pct)

        return universal_return


# ============================================================================
# APPROACH 5: Drawdown-Aware Universal Portfolio
# Explicitly track and minimize drawdowns using safe haven
# ============================================================================

class DrawdownAwareUniversalPortfolio:
    """
    Universal Portfolio that explicitly considers drawdown in weight updates

    Key idea:
    - Not just maximize growth
    - Also minimize maximum drawdown
    - Penalize portfolios that had severe drawdowns even if they recovered
    """

    def __init__(self, n_portfolios=101, drawdown_penalty=1.0):
        self.portfolios = np.linspace(0, 1, n_portfolios)
        self.n_portfolios = n_portfolios
        self.drawdown_penalty = drawdown_penalty

        # Track cumulative wealth and max drawdown for each portfolio
        self.cumulative_wealth = np.ones(n_portfolios)
        self.max_wealth = np.ones(n_portfolios)
        self.max_drawdown = np.zeros(n_portfolios)

        self.wealth_history = []
        self.allocation_history = []
        self.drawdown_history = []

    def update(self, price_multipliers):
        """
        Update with drawdown-adjusted weights.

        Args:
            price_multipliers: Dict with 'bonds' and 'stocks' keys
                              Values are price multipliers (e.g., 1.02 = +2%)
        """
        bond_mult = price_multipliers['bonds']
        stock_mult = price_multipliers['stocks']

        # Calculate returns for each portfolio
        portfolio_returns = (1 - self.portfolios) * bond_mult + self.portfolios * stock_mult

        # Update cumulative wealth
        self.cumulative_wealth *= portfolio_returns

        # Update max wealth and drawdown
        self.max_wealth = np.maximum(self.max_wealth, self.cumulative_wealth)
        current_drawdown = (self.max_wealth - self.cumulative_wealth) / self.max_wealth
        self.max_drawdown = np.maximum(self.max_drawdown, current_drawdown)

        # Adjust weights: log(wealth) - penalty * max_drawdown
        adjusted_log_wealth = np.log(self.cumulative_wealth) - self.drawdown_penalty * self.max_drawdown

        # Calculate allocation
        weights = np.exp(adjusted_log_wealth - np.max(adjusted_log_wealth))
        weights = weights / np.sum(weights)
        allocation = np.sum(weights * self.portfolios)

        # Calculate universal portfolio return
        universal_return = (1 - allocation) * bond_mult + allocation * stock_mult

        # Track our own drawdown
        wealth = 1.0 if len(self.wealth_history) == 0 else self.wealth_history[-1]
        new_wealth = wealth * universal_return
        max_wealth_so_far = new_wealth if len(self.wealth_history) == 0 else max(max(self.wealth_history), new_wealth)
        our_drawdown = (max_wealth_so_far - new_wealth) / max_wealth_so_far if max_wealth_so_far > 0 else 0

        # Track history
        self.wealth_history.append(new_wealth)
        self.allocation_history.append(allocation)
        self.drawdown_history.append(our_drawdown)

        return universal_return


# ============================================================================
# APPROACH 6: Time-Varying Ergodicity Universal Portfolio
# Adjust volatility penalty based on recent market conditions
# ============================================================================

class TimeVaryingErgodicityUP:
    """
    Dynamically adjust ergodicity penalty based on market volatility

    High volatility → increase penalty (more conservative)
    Low volatility → decrease penalty (more aggressive)
    """

    def __init__(self, n_portfolios=101, base_ergodicity=0.5, volatility_window=20):
        self.n_portfolios = n_portfolios
        self.portfolios = np.linspace(0, 1, n_portfolios)
        self.base_ergodicity = base_ergodicity
        self.volatility_window = volatility_window

        self.log_wealth = np.zeros(n_portfolios)
        self.squared_log_returns = np.zeros(n_portfolios)
        self.periods = 0

        self.returns_buffer = []
        self.wealth_history = []
        self.allocation_history = []
        self.ergodicity_factor_history = []

    def get_current_volatility(self):
        """Calculate recent market volatility"""
        if len(self.returns_buffer) < 2:
            return 0.15  # Default

        recent = self.returns_buffer[-self.volatility_window:]
        stock_mults = [r['stocks'] for r in recent]
        return np.std(np.log(stock_mults))

    def get_ergodicity_factor(self):
        """Adjust ergodicity based on volatility"""
        vol = self.get_current_volatility()

        # Scale: low vol (0.05) → factor 0.3, high vol (0.30) → factor 1.5
        factor = self.base_ergodicity * (1 + 2 * (vol - 0.15))
        return np.clip(factor, 0.1, 2.0)

    def update(self, price_multipliers):
        """
        Update with price multipliers dict.

        Args:
            price_multipliers: Dict with 'bonds' and 'stocks' keys
                              Values are price multipliers (e.g., 1.02 = +2%)
        """
        bond_mult = price_multipliers['bonds']
        stock_mult = price_multipliers['stocks']
        self.returns_buffer.append(price_multipliers)

        # Calculate returns for each portfolio
        portfolio_returns = (1 - self.portfolios) * bond_mult + self.portfolios * stock_mult
        log_returns = np.log(portfolio_returns)

        # Update statistics
        self.log_wealth += log_returns
        self.squared_log_returns += log_returns ** 2
        self.periods += 1

        # Dynamic ergodicity factor
        ergodicity_factor = self.get_ergodicity_factor()
        self.ergodicity_factor_history.append(ergodicity_factor)

        # Calculate variance
        variance = self.squared_log_returns / max(1, self.periods) - (self.log_wealth / max(1, self.periods))**2
        adjusted_log_wealth = self.log_wealth - ergodicity_factor * variance * self.periods

        # Get allocation
        weights = np.exp(adjusted_log_wealth - np.max(adjusted_log_wealth))
        weights = weights / np.sum(weights)
        allocation = np.sum(weights * self.portfolios)

        # Calculate return
        universal_return = (1 - allocation) * bond_mult + allocation * stock_mult

        # Track
        wealth = 1.0 if len(self.wealth_history) == 0 else self.wealth_history[-1]
        self.wealth_history.append(wealth * universal_return)
        self.allocation_history.append(allocation)

        return universal_return


# ============================================================================
# APPROACH 7: Multi-Timeframe Hierarchical
# Different rebalancing frequencies for different levels
# ============================================================================

class MultiTimeframeHierarchical:
    """
    Hierarchical with different rebalancing frequencies

    Level 1 (Growth): Rebalance frequently (every period)
    Level 2 (Hedge): Rebalance slowly (every N periods)

    Rationale: Growth assets need quick adaptation,
               Insurance decisions are more strategic
    """

    def __init__(self, n_portfolios_risky=51, n_portfolios_hedge=21, hedge_rebalance_freq=5):
        self.risky_portfolios = np.linspace(0, 1, n_portfolios_risky)
        self.risky_weights = np.ones(n_portfolios_risky)

        self.hedge_portfolios = np.linspace(0, 0.3, n_portfolios_hedge)
        self.hedge_weights = np.ones(n_portfolios_hedge)
        self.hedge_rebalance_freq = hedge_rebalance_freq

        self.period_count = 0
        self.last_hedge_allocation = 0.15  # Start at 15%

        self.wealth_history = []
        self.allocation_history = []
        self.hedge_allocation_history = []

    def update(self, price_multipliers):
        """
        Update with price multipliers dict.

        Args:
            price_multipliers: Dict with 'bonds', 'stocks', 'tail_hedge' keys
                              Values are price multipliers (e.g., 1.02 = +2%)
        """
        bond_mult = price_multipliers['bonds']
        stock_mult = price_multipliers['stocks']
        hedge_mult = price_multipliers['tail_hedge']

        # Level 1: ALWAYS rebalance (every period)
        portfolio_returns = (1 - self.risky_portfolios) * bond_mult + self.risky_portfolios * stock_mult
        self.risky_weights *= portfolio_returns

        weights_norm = self.risky_weights / np.sum(self.risky_weights)
        risky_allocation = np.sum(weights_norm * self.risky_portfolios)
        risky_return = (1 - risky_allocation) * bond_mult + risky_allocation * stock_mult

        # Level 2: Rebalance only every N periods
        self.period_count += 1

        if self.period_count % self.hedge_rebalance_freq == 0:
            # Time to rebalance hedge allocation
            meta_returns = (1 - self.hedge_portfolios) * risky_return + self.hedge_portfolios * hedge_mult
            self.hedge_weights *= meta_returns

            hedge_weights_norm = self.hedge_weights / np.sum(self.hedge_weights)
            self.last_hedge_allocation = np.sum(hedge_weights_norm * self.hedge_portfolios)

        # Use last hedge allocation (updated or carried forward)
        hedge_pct = self.last_hedge_allocation

        # Final return
        final_return = (1 - hedge_pct) * risky_return + hedge_pct * hedge_mult

        # Track
        wealth = 1.0 if len(self.wealth_history) == 0 else self.wealth_history[-1]
        self.wealth_history.append(wealth * final_return)
        self.allocation_history.append(risky_allocation)
        self.hedge_allocation_history.append(hedge_pct)

        return final_return


# ============================================================================
# APPROACH 8: Multi-Safe-Haven Universal Portfolio
# Track multiple types of safe havens simultaneously
# ============================================================================

class MultiSafeHavenUP:
    """
    Instead of single tail hedge, use multiple safe havens:
    - Tail hedge (puts/VIX)
    - Gold
    - Cash

    Let Universal Portfolio decide optimal allocation across all
    """

    # Asset order for portfolio matrix columns
    ASSET_ORDER = ['stocks', 'bonds', 'tail_hedge', 'gold']

    def __init__(self, n_portfolios=15):
        # Generate 4D simplex: (stock%, bond%, tail_hedge%, gold%)
        # Cash = 1 - others
        self.portfolios = self._generate_4d_simplex(n_portfolios)
        self.n_portfolios = len(self.portfolios)
        self.log_weights = np.zeros(self.n_portfolios)

        self.wealth_history = []
        self.allocation_history = []

    def _generate_4d_simplex(self, n):
        """Generate portfolios on 4D simplex"""
        portfolios = []
        step = 1.0 / (n - 1)

        for i in range(n):
            for j in range(n - i):
                for k in range(n - i - j):
                    stock = i * step
                    bond = j * step
                    tail = k * step
                    gold = 1.0 - stock - bond - tail

                    if gold >= -1e-10 and gold <= 1.0 + 1e-10:
                        portfolios.append([
                            max(0, stock),
                            max(0, bond),
                            max(0, tail),
                            max(0, gold)
                        ])

        return np.array(portfolios)

    def update(self, price_multipliers):
        """
        Update with price multipliers dict.

        Args:
            price_multipliers: Dict with 'stocks', 'bonds', 'tail_hedge', 'gold' keys
                              Values are price multipliers (e.g., 1.02 = +2%)
        """
        # Convert dict to array in correct order for portfolio matrix multiplication
        mult_array = np.array([price_multipliers[asset] for asset in self.ASSET_ORDER])

        # Calculate return for each portfolio
        portfolio_returns = np.sum(self.portfolios * mult_array, axis=1)
        portfolio_returns = np.clip(portfolio_returns, 0.01, 10.0)

        # Update
        self.log_weights += np.log(portfolio_returns)
        self.log_weights = self.log_weights - np.mean(self.log_weights)

        # Get allocation
        weights = np.exp(self.log_weights - np.max(self.log_weights))
        weights = weights / np.sum(weights)
        allocation = np.sum(weights[:, np.newaxis] * self.portfolios, axis=0)

        # Calculate return
        universal_return = np.sum(allocation * mult_array)

        # Track
        wealth = 1.0 if len(self.wealth_history) == 0 else self.wealth_history[-1]
        self.wealth_history.append(wealth * universal_return)
        self.allocation_history.append(allocation)

        return universal_return


# ============================================================================
# APPROACH 9: Kelly-Criterion Universal Portfolio
# Use Kelly optimal sizing for each portfolio weight
# ============================================================================

class KellyUniversalPortfolio:
    """
    Incorporate Kelly Criterion into weight updates

    Standard UP: w_i ← w_i × r_i
    Kelly UP: w_i ← w_i × r_i^(kelly_fraction)

    Kelly fraction controls leverage/conservatism
    """

    def __init__(self, n_portfolios=101, kelly_fraction=0.5):
        self.portfolios = np.linspace(0, 1, n_portfolios)
        self.kelly_fraction = kelly_fraction
        self.log_weights = np.zeros(n_portfolios)

        self.wealth_history = []
        self.allocation_history = []

    def update(self, price_multipliers):
        """
        Update with price multipliers dict.

        Args:
            price_multipliers: Dict with 'bonds' and 'stocks' keys
                              Values are price multipliers (e.g., 1.02 = +2%)
        """
        bond_mult = price_multipliers['bonds']
        stock_mult = price_multipliers['stocks']

        portfolio_returns = (1 - self.portfolios) * bond_mult + self.portfolios * stock_mult

        # Kelly-adjusted update
        # Instead of log(r), use kelly_fraction * log(r)
        self.log_weights += self.kelly_fraction * np.log(portfolio_returns)

        # Get allocation
        weights = np.exp(self.log_weights - np.max(self.log_weights))
        weights = weights / np.sum(weights)
        allocation = np.sum(weights * self.portfolios)

        # Calculate return
        universal_return = (1 - allocation) * bond_mult + allocation * stock_mult

        # Track
        wealth = 1.0 if len(self.wealth_history) == 0 else self.wealth_history[-1]
        self.wealth_history.append(wealth * universal_return)
        self.allocation_history.append(allocation)

        return universal_return


# ============================================================================
# APPROACH 10: Asymmetric Loss-Averse Universal Portfolio
# Penalize losses more than reward gains
# ============================================================================

class AsymmetricLossAverseUP:
    """
    Loss aversion: Penalize negative returns more heavily

    Standard: w_i ← w_i × r_i
    Loss-averse: w_i ← w_i × r_i^(1.0 if r_i >= 1 else loss_penalty)

    Mimics behavioral finance - losses hurt more than gains feel good
    """

    def __init__(self, n_portfolios=101, loss_penalty=1.5):
        self.portfolios = np.linspace(0, 1, n_portfolios)
        self.loss_penalty = loss_penalty
        self.log_weights = np.zeros(n_portfolios)

        self.wealth_history = []
        self.allocation_history = []

    def update(self, price_multipliers):
        """
        Update with price multipliers dict.

        Args:
            price_multipliers: Dict with 'bonds' and 'stocks' keys
                              Values are price multipliers (e.g., 1.02 = +2%)
        """
        bond_mult = price_multipliers['bonds']
        stock_mult = price_multipliers['stocks']

        portfolio_returns = (1 - self.portfolios) * bond_mult + self.portfolios * stock_mult

        # Asymmetric update
        log_returns = np.log(portfolio_returns)

        # Penalize losses more
        penalties = np.where(portfolio_returns < 1.0, self.loss_penalty, 1.0)
        self.log_weights += penalties * log_returns

        # Get allocation
        weights = np.exp(self.log_weights - np.max(self.log_weights))
        weights = weights / np.sum(weights)
        allocation = np.sum(weights * self.portfolios)

        # Calculate return
        universal_return = (1 - allocation) * bond_mult + allocation * stock_mult

        # Track
        wealth = 1.0 if len(self.wealth_history) == 0 else self.wealth_history[-1]
        self.wealth_history.append(wealth * universal_return)
        self.allocation_history.append(allocation)

        return universal_return


# ============================================================================
# APPROACH 11: Sequential Threshold Universal Portfolio
# Only add hedge after reaching wealth threshold
# ============================================================================

class SequentialThresholdUP:
    """
    Start with growth-only Universal Portfolio
    Once wealth crosses threshold, add tail hedge

    Rationale: Need capital base before paying for insurance
    """

    def __init__(self, n_portfolios=101, wealth_threshold=2.0, hedge_pct=0.15):
        self.portfolios = np.linspace(0, 1, n_portfolios)
        self.log_weights = np.zeros(n_portfolios)
        self.wealth_threshold = wealth_threshold
        self.hedge_pct = hedge_pct

        self.using_hedge = False
        self.wealth_history = []
        self.allocation_history = []
        self.hedge_status_history = []

    def update(self, price_multipliers):
        """
        Update with price multipliers dict.

        Args:
            price_multipliers: Dict with 'bonds', 'stocks', and optionally 'tail_hedge' keys
                              Values are price multipliers (e.g., 1.02 = +2%)
        """
        bond_mult = price_multipliers['bonds']
        stock_mult = price_multipliers['stocks']
        hedge_mult = price_multipliers.get('tail_hedge', 1.0)

        # Check if we should activate hedge
        current_wealth = 1.0 if len(self.wealth_history) == 0 else self.wealth_history[-1]
        if current_wealth >= self.wealth_threshold:
            self.using_hedge = True

        # Update growth portfolio
        portfolio_returns = (1 - self.portfolios) * bond_mult + self.portfolios * stock_mult
        self.log_weights += np.log(portfolio_returns)

        # Get allocation
        weights = np.exp(self.log_weights - np.max(self.log_weights))
        weights = weights / np.sum(weights)
        allocation = np.sum(weights * self.portfolios)

        # Calculate return
        growth_return = (1 - allocation) * bond_mult + allocation * stock_mult

        if self.using_hedge:
            # Use hedge
            final_return = (1 - self.hedge_pct) * growth_return + self.hedge_pct * hedge_mult
        else:
            # No hedge yet
            final_return = growth_return

        # Track
        self.wealth_history.append(current_wealth * final_return)
        self.allocation_history.append(allocation)
        self.hedge_status_history.append(self.using_hedge)

        return final_return


# ============================================================================
# APPROACH 12: Volatility-Scaled Universal Portfolio
# Scale allocations by inverse volatility
# ============================================================================

class VolatilityScaledUP:
    """
    Target constant volatility by scaling allocations

    High volatility → reduce allocation
    Low volatility → increase allocation
    """

    def __init__(self, n_portfolios=101, target_vol=0.12, vol_window=20):
        self.portfolios = np.linspace(0, 1, n_portfolios)
        self.log_weights = np.zeros(n_portfolios)
        self.target_vol = target_vol
        self.vol_window = vol_window

        self.returns_buffer = []
        self.wealth_history = []
        self.allocation_history = []
        self.vol_history = []

    def get_realized_volatility(self):
        """Calculate recent volatility"""
        if len(self.returns_buffer) < 2:
            return self.target_vol

        recent = self.returns_buffer[-self.vol_window:]
        log_returns = [np.log((1-a)*r['bonds'] + a*r['stocks']) for a, r in
                      zip([0.5]*len(recent), recent)]
        return np.std(log_returns)

    def update(self, price_multipliers):
        """
        Update with price multipliers dict.

        Args:
            price_multipliers: Dict with 'bonds' and 'stocks' keys
                              Values are price multipliers (e.g., 1.02 = +2%)
        """
        bond_mult = price_multipliers['bonds']
        stock_mult = price_multipliers['stocks']
        self.returns_buffer.append(price_multipliers)

        # Standard Universal Portfolio update
        portfolio_returns = (1 - self.portfolios) * bond_mult + self.portfolios * stock_mult
        self.log_weights += np.log(portfolio_returns)

        # Get base allocation
        weights = np.exp(self.log_weights - np.max(self.log_weights))
        weights = weights / np.sum(weights)
        base_allocation = np.sum(weights * self.portfolios)

        # Volatility scaling
        realized_vol = self.get_realized_volatility()
        vol_scalar = self.target_vol / max(realized_vol, 0.01)
        vol_scalar = np.clip(vol_scalar, 0.5, 2.0)  # Limit scaling

        # Scaled allocation
        scaled_allocation = base_allocation * vol_scalar
        scaled_allocation = np.clip(scaled_allocation, 0.0, 1.0)

        # Calculate return
        universal_return = (1 - scaled_allocation) * bond_mult + scaled_allocation * stock_mult

        # Track
        wealth = 1.0 if len(self.wealth_history) == 0 else self.wealth_history[-1]
        self.wealth_history.append(wealth * universal_return)
        self.allocation_history.append(scaled_allocation)
        self.vol_history.append(realized_vol)

        return universal_return


# ============================================================================
# APPROACH 13: Momentum-Enhanced Hierarchical
# Combine trend-following with Universal Portfolio
# ============================================================================

class MomentumEnhancedHierarchical:
    """
    Hierarchical UP with momentum overlay

    If stock momentum positive: increase equity allocation
    If stock momentum negative: increase hedge allocation
    """

    def __init__(self, n_portfolios_risky=51, n_portfolios_hedge=21, momentum_window=10):
        self.risky_portfolios = np.linspace(0, 1, n_portfolios_risky)
        self.risky_weights = np.ones(n_portfolios_risky)

        self.hedge_portfolios = np.linspace(0, 0.3, n_portfolios_hedge)
        self.hedge_weights = np.ones(n_portfolios_hedge)

        self.momentum_window = momentum_window
        self.returns_buffer = []

        self.wealth_history = []
        self.allocation_history = []
        self.hedge_allocation_history = []
        self.momentum_history = []

    def get_momentum(self):
        """Calculate stock momentum (simple moving average)"""
        if len(self.returns_buffer) < self.momentum_window:
            return 0.0

        recent = self.returns_buffer[-self.momentum_window:]
        stock_mults = [r['stocks'] for r in recent]
        cumulative = np.prod(stock_mults)

        # Momentum = excess return vs bonds
        bond_cumulative = 1.02 ** len(recent)
        return (cumulative / bond_cumulative) - 1.0

    def update(self, price_multipliers):
        """
        Update with price multipliers dict.

        Args:
            price_multipliers: Dict with 'bonds', 'stocks', 'tail_hedge' keys
                              Values are price multipliers (e.g., 1.02 = +2%)
        """
        bond_mult = price_multipliers['bonds']
        stock_mult = price_multipliers['stocks']
        hedge_mult = price_multipliers['tail_hedge']
        self.returns_buffer.append(price_multipliers)

        # Calculate momentum
        momentum = self.get_momentum()
        self.momentum_history.append(momentum)

        # Momentum boost: positive momentum → increase stock weight
        # Negative momentum → increase hedge weight
        momentum_boost = np.tanh(momentum * 2)  # Scale to [-1, 1]

        # Level 1: Standard UP with momentum boost
        portfolio_returns = (1 - self.risky_portfolios) * bond_mult + self.risky_portfolios * stock_mult

        # Apply momentum: boost winners more, penalize losers more
        momentum_adjusted = portfolio_returns ** (1 + 0.3 * momentum_boost * self.risky_portfolios)
        self.risky_weights *= momentum_adjusted

        weights_norm = self.risky_weights / np.sum(self.risky_weights)
        risky_allocation = np.sum(weights_norm * self.risky_portfolios)
        risky_return = (1 - risky_allocation) * bond_mult + risky_allocation * stock_mult

        # Level 2: Standard UP
        meta_returns = (1 - self.hedge_portfolios) * risky_return + self.hedge_portfolios * hedge_mult
        self.hedge_weights *= meta_returns

        hedge_weights_norm = self.hedge_weights / np.sum(self.hedge_weights)
        base_hedge_pct = np.sum(hedge_weights_norm * self.hedge_portfolios)

        # Momentum adjustment: negative momentum → increase hedge
        hedge_pct = base_hedge_pct * (1 + max(0, -momentum_boost) * 0.5)
        hedge_pct = np.clip(hedge_pct, 0, 0.4)

        # Final return
        final_return = (1 - hedge_pct) * risky_return + hedge_pct * hedge_mult

        # Track
        wealth = 1.0 if len(self.wealth_history) == 0 else self.wealth_history[-1]
        self.wealth_history.append(wealth * final_return)
        self.allocation_history.append(risky_allocation)
        self.hedge_allocation_history.append(hedge_pct)

        return final_return


# ============================================================================
# APPROACH 14: Three-Level Hierarchical (Multi-Asset)
# Stocks, Bonds, Commodities → Risky Bucket → vs Tail Hedge
# ============================================================================

class ThreeLevelHierarchical:
    """
    Level 1: Stocks vs Bonds vs Commodities
    Level 2: Risky bucket (Level 1 output) vs Safe haven assets (gold)
    Level 3: Level 2 output vs Tail hedge

    Most granular separation of concerns
    """

    def __init__(self, n_level1=21, n_level2=11, n_level3=11):
        # Level 1: 2D simplex (stocks, bonds, commodities sum to 1)
        self.level1_portfolios = self._generate_2d_simplex(n_level1)
        self.level1_weights = np.ones(len(self.level1_portfolios))

        # Level 2: Gold allocation (0-50%)
        self.level2_portfolios = np.linspace(0, 0.5, n_level2)
        self.level2_weights = np.ones(n_level2)

        # Level 3: Tail hedge allocation (0-20%)
        self.level3_portfolios = np.linspace(0, 0.2, n_level3)
        self.level3_weights = np.ones(n_level3)

        self.wealth_history = []
        self.level1_allocation_history = []
        self.level2_allocation_history = []
        self.level3_allocation_history = []

    def _generate_2d_simplex(self, n):
        """Generate 2D simplex for 3 assets"""
        portfolios = []
        step = 1.0 / (n - 1)
        for i in range(n):
            for j in range(n - i):
                stocks = i * step
                bonds = j * step
                commodities = 1.0 - stocks - bonds
                if commodities >= -1e-10:
                    portfolios.append([max(0, stocks), max(0, bonds), max(0, commodities)])
        return np.array(portfolios)

    def update(self, price_multipliers):
        """
        Update with price multipliers dict.

        Args:
            price_multipliers: Dict with 'bonds', 'stocks', 'commodities', 'gold', 'tail_hedge' keys
                              Values are price multipliers (e.g., 1.02 = +2%)
        """
        bond_mult = price_multipliers['bonds']
        stock_mult = price_multipliers['stocks']
        commodity_mult = price_multipliers['commodities']
        gold_mult = price_multipliers['gold']
        tail_mult = price_multipliers['tail_hedge']

        # Level 1: Optimize over stocks/bonds/commodities
        level1_returns = (self.level1_portfolios[:, 0] * stock_mult +
                         self.level1_portfolios[:, 1] * bond_mult +
                         self.level1_portfolios[:, 2] * commodity_mult)

        self.level1_weights *= level1_returns
        w1_norm = self.level1_weights / np.sum(self.level1_weights)
        level1_allocation = np.sum(w1_norm[:, np.newaxis] * self.level1_portfolios, axis=0)

        level1_return = np.sum(level1_allocation * [stock_mult, bond_mult, commodity_mult])

        # Level 2: Risky bucket vs Gold
        level2_returns = (1 - self.level2_portfolios) * level1_return + self.level2_portfolios * gold_mult

        self.level2_weights *= level2_returns
        w2_norm = self.level2_weights / np.sum(self.level2_weights)
        gold_allocation = np.sum(w2_norm * self.level2_portfolios)

        level2_return = (1 - gold_allocation) * level1_return + gold_allocation * gold_mult

        # Level 3: Level 2 output vs Tail hedge
        level3_returns = (1 - self.level3_portfolios) * level2_return + self.level3_portfolios * tail_mult

        self.level3_weights *= level3_returns
        w3_norm = self.level3_weights / np.sum(self.level3_weights)
        tail_allocation = np.sum(w3_norm * self.level3_portfolios)

        final_return = (1 - tail_allocation) * level2_return + tail_allocation * tail_mult

        # Track
        wealth = 1.0 if len(self.wealth_history) == 0 else self.wealth_history[-1]
        self.wealth_history.append(wealth * final_return)
        self.level1_allocation_history.append(level1_allocation)
        self.level2_allocation_history.append(gold_allocation)
        self.level3_allocation_history.append(tail_allocation)

        return final_return


# ============================================================================
# APPROACH 15: Dynamic Granularity Universal Portfolio
# Adjust number of portfolios based on market conditions
# ============================================================================

class DynamicGranularityUP:
    """
    Use fine granularity in volatile markets (need precision)
    Use coarse granularity in calm markets (save computation)
    """

    def __init__(self, min_portfolios=21, max_portfolios=101, vol_threshold=0.15):
        self.min_portfolios = min_portfolios
        self.max_portfolios = max_portfolios
        self.vol_threshold = vol_threshold

        self.returns_buffer = []
        self.current_portfolios = self._create_portfolios(min_portfolios)
        self.log_weights = np.zeros(min_portfolios)

        self.wealth_history = []
        self.allocation_history = []
        self.granularity_history = []

    def _create_portfolios(self, n):
        return np.linspace(0, 1, n)

    def get_volatility(self):
        if len(self.returns_buffer) < 2:
            return 0.10
        stock_mults = [r['stocks'] for r in self.returns_buffer[-20:]]
        return np.std(np.log(stock_mults))

    def adjust_granularity(self):
        """Adjust number of portfolios based on volatility"""
        vol = self.get_volatility()

        # High vol → more portfolios, Low vol → fewer portfolios
        if vol > self.vol_threshold * 1.5:
            target_n = self.max_portfolios
        elif vol > self.vol_threshold:
            target_n = (self.min_portfolios + self.max_portfolios) // 2
        else:
            target_n = self.min_portfolios

        current_n = len(self.current_portfolios)

        if target_n != current_n:
            # Resample weights to new granularity
            old_alloc = self.get_allocation()

            self.current_portfolios = self._create_portfolios(target_n)

            # Redistribute weights around current allocation
            new_weights = np.exp(-100 * (self.current_portfolios - old_alloc)**2)
            self.log_weights = np.log(new_weights + 1e-10)

        return len(self.current_portfolios)

    def get_allocation(self):
        weights = np.exp(self.log_weights - np.max(self.log_weights))
        weights = weights / np.sum(weights)
        return np.sum(weights * self.current_portfolios)

    def update(self, price_multipliers):
        """
        Update with price multipliers dict.

        Args:
            price_multipliers: Dict with 'bonds' and 'stocks' keys
                              Values are price multipliers (e.g., 1.02 = +2%)
        """
        bond_mult = price_multipliers['bonds']
        stock_mult = price_multipliers['stocks']
        self.returns_buffer.append(price_multipliers)

        # Possibly adjust granularity
        n_portfolios = self.adjust_granularity()
        self.granularity_history.append(n_portfolios)

        # Standard update
        portfolio_returns = (1 - self.current_portfolios) * bond_mult + self.current_portfolios * stock_mult
        self.log_weights += np.log(portfolio_returns)

        # Get allocation
        allocation = self.get_allocation()

        # Calculate return
        universal_return = (1 - allocation) * bond_mult + allocation * stock_mult

        # Track
        wealth = 1.0 if len(self.wealth_history) == 0 else self.wealth_history[-1]
        self.wealth_history.append(wealth * universal_return)
        self.allocation_history.append(allocation)

        return universal_return
