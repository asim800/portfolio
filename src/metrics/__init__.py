# Performance and risk metrics package
"""
Performance measurement and risk analysis utilities.

Modules:
- performance: Sharpe, Sortino, Calmar, returns, drawdown calculations
- risk: VaR, CVaR, tail risk metrics
- tracker: Portfolio tracking over time
"""

from .performance import (
    annualized_return,
    annualized_standard_deviation,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    gain_to_pain_ratio,
)

from .risk import (
    calculate_var,
    calculate_cvar,
    calculate_downside_deviation,
    calculate_ulcer_index,
    calculate_tail_ratio,
    calculate_pain_index,
    calculate_omega_ratio,
)

from .tracker import PortfolioTracker

__all__ = [
    # Performance
    'annualized_return',
    'annualized_standard_deviation',
    'sharpe_ratio',
    'sortino_ratio',
    'max_drawdown',
    'calmar_ratio',
    'gain_to_pain_ratio',
    # Risk
    'calculate_var',
    'calculate_cvar',
    'calculate_downside_deviation',
    'calculate_ulcer_index',
    'calculate_tail_ratio',
    'calculate_pain_index',
    'calculate_omega_ratio',
    # Tracker
    'PortfolioTracker',
]
