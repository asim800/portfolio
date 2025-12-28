# Portfolio strategies package
"""
Portfolio allocation and rebalancing strategies.

Modules:
- base: Abstract base classes for strategies
- allocation: Static, EqualWeight, Optimized allocation strategies
- rebalancing: Rebalancing triggers (Never, Periodic, Threshold, etc.)
- universal: Universal Portfolio strategies (17 implementations)
- registry: Strategy registry pattern for dynamic instantiation
"""

# Base classes
from .base import PortfolioStrategy, AllocationStrategyBase, RebalancingTriggerBase

# Allocation strategies
from .allocation import AllocationStrategy, StaticAllocation, EqualWeight, OptimizedAllocation

# Rebalancing triggers
from .rebalancing import RebalancingTrigger, Never, Periodic, Threshold

# Universal Portfolio strategies
from .universal import (
    ThreeAssetUniversalPortfolio,
    ErgodicUniversalPortfolio,
    ConditionalSafeHavenPortfolio,
    UniversalPortfolioBasic,
    BuyAndHold,
    HierarchicalUniversalPortfolio,
    DrawdownAwareUniversalPortfolio,
    TimeVaryingErgodicityUP,
    MultiTimeframeHierarchical,
    MultiSafeHavenUP,
    KellyUniversalPortfolio,
    AsymmetricLossAverseUP,
    SequentialThresholdUP,
    VolatilityScaledUP,
    MomentumEnhancedHierarchical,
    ThreeLevelHierarchical,
    DynamicGranularityUP,
)

# Registry
from .registry import (
    create_portfolio_registry,
    get_strategy_class,
    get_strategy_params,
    get_required_assets,
    instantiate_strategy,
    list_strategies,
)

__all__ = [
    # Base
    'PortfolioStrategy', 'AllocationStrategyBase', 'RebalancingTriggerBase',
    # Allocation
    'AllocationStrategy', 'StaticAllocation', 'EqualWeight', 'OptimizedAllocation',
    # Rebalancing
    'RebalancingTrigger', 'Never', 'Periodic', 'Threshold',
    # Universal Portfolio
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
    # Registry
    'create_portfolio_registry',
    'get_strategy_class',
    'get_strategy_params',
    'get_required_assets',
    'instantiate_strategy',
    'list_strategies',
]
