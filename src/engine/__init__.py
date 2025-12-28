# Engine package - core orchestration classes
"""
Core portfolio management and backtesting infrastructure.

Modules:
- portfolio: Portfolio class for managing weights, returns, and metrics
- optimizer: Portfolio optimization (mean-variance, robust)
- backtest: Backtesting engine
- period_manager: Period and frequency management
"""

from .portfolio import Portfolio
from .optimizer import PortfolioOptimizer
from .period_manager import PeriodManager

__all__ = ['Portfolio', 'PortfolioOptimizer', 'PeriodManager']
