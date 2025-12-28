# Portfolio Optimization and Monte Carlo Simulation Package
"""
Main package for portfolio optimization, Monte Carlo simulation, and backtesting.

Subpackages:
- data: Market data fetching, covariance estimation, simulated data
- montecarlo: Monte Carlo path generation, bootstrapping, lifecycle simulation
- strategies: Portfolio allocation and rebalancing strategies
- metrics: Performance and risk metrics
- visualization: Plotting and charting utilities
- engine: Core portfolio, optimizer, and backtest classes
- config: Configuration handling

Usage:
    # Import from subpackages
    from src.data import FinData, CovarianceEstimator
    from src.montecarlo import MCPathGenerator
    from src.strategies import create_portfolio_registry
    from src.metrics import sharpe_ratio, max_drawdown
    from src.engine import Portfolio, PortfolioOptimizer
    from src.config import SystemConfig
"""

__version__ = "0.1.0"

# Convenience imports for commonly used classes
from .data import FinData, CovarianceEstimator
from .montecarlo import MCPathGenerator
from .engine import Portfolio, PortfolioOptimizer, PeriodManager
from .config import SystemConfig

__all__ = [
    'FinData',
    'CovarianceEstimator',
    'MCPathGenerator',
    'Portfolio',
    'PortfolioOptimizer',
    'PeriodManager',
    'SystemConfig',
]
