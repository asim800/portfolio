#!/usr/bin/env python3
"""
Risk Metrics Module.

Provides risk measurement functions including VaR, CVaR, and tail risk metrics.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


def calculate_var(returns: Union[pd.Series, np.ndarray],
                 confidence: float = 0.95,
                 method: str = 'historical') -> float:
    """
    Calculate Value at Risk (VaR).

    Parameters:
    -----------
    returns : pd.Series or np.ndarray
        Return series
    confidence : float
        Confidence level (e.g., 0.95 for 95% VaR)
    method : str
        'historical' (empirical quantile) or 'parametric' (normal assumption)

    Returns:
    --------
    float: VaR (as positive number representing potential loss)
    """
    returns = np.asarray(returns)

    if method == 'historical':
        var = -np.percentile(returns, (1 - confidence) * 100)
    elif method == 'parametric':
        from scipy import stats
        mean = np.mean(returns)
        std = np.std(returns)
        var = -(mean + std * stats.norm.ppf(1 - confidence))
    else:
        raise ValueError(f"Unknown method: {method}")

    return var


def calculate_cvar(returns: Union[pd.Series, np.ndarray],
                  confidence: float = 0.95,
                  method: str = 'historical') -> float:
    """
    Calculate Conditional Value at Risk (CVaR / Expected Shortfall).

    CVaR is the expected loss given that loss exceeds VaR.

    Parameters:
    -----------
    returns : pd.Series or np.ndarray
        Return series
    confidence : float
        Confidence level (e.g., 0.95 for 95% CVaR)
    method : str
        'historical' (empirical) or 'parametric' (normal assumption)

    Returns:
    --------
    float: CVaR (as positive number representing expected loss in tail)
    """
    returns = np.asarray(returns)
    var = calculate_var(returns, confidence, method)

    if method == 'historical':
        # Average of losses worse than VaR
        tail_losses = returns[returns < -var]
        if len(tail_losses) == 0:
            return var  # No observations worse than VaR
        cvar = -np.mean(tail_losses)
    elif method == 'parametric':
        from scipy import stats
        mean = np.mean(returns)
        std = np.std(returns)
        alpha = 1 - confidence
        cvar = -(mean - std * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha)
    else:
        raise ValueError(f"Unknown method: {method}")

    return cvar


def calculate_downside_deviation(returns: Union[pd.Series, np.ndarray],
                                threshold: float = 0.0) -> float:
    """
    Calculate downside deviation (semi-deviation below threshold).

    Parameters:
    -----------
    returns : pd.Series or np.ndarray
        Return series
    threshold : float
        Threshold for downside (default: 0)

    Returns:
    --------
    float: Downside deviation
    """
    returns = np.asarray(returns)
    downside_returns = returns[returns < threshold]

    if len(downside_returns) == 0:
        return 0.0

    return np.sqrt(np.mean((downside_returns - threshold) ** 2))


def calculate_ulcer_index(prices: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate Ulcer Index (measures depth and duration of drawdowns).

    Parameters:
    -----------
    prices : pd.Series or np.ndarray
        Price series

    Returns:
    --------
    float: Ulcer Index (lower is better)
    """
    prices = np.asarray(prices)
    running_max = np.maximum.accumulate(prices)
    drawdown_pct = 100 * (running_max - prices) / running_max
    return np.sqrt(np.mean(drawdown_pct ** 2))


def calculate_tail_ratio(returns: Union[pd.Series, np.ndarray],
                        percentile: float = 5) -> float:
    """
    Calculate tail ratio (right tail / left tail).

    Measures asymmetry of return distribution.
    Values > 1 indicate positive skew (larger gains than losses).

    Parameters:
    -----------
    returns : pd.Series or np.ndarray
        Return series
    percentile : float
        Percentile for tail measurement (default: 5th and 95th)

    Returns:
    --------
    float: Tail ratio
    """
    returns = np.asarray(returns)
    right_tail = np.abs(np.percentile(returns, 100 - percentile))
    left_tail = np.abs(np.percentile(returns, percentile))

    if left_tail == 0:
        return np.inf if right_tail > 0 else 1.0

    return right_tail / left_tail


def calculate_pain_index(prices: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate Pain Index (average drawdown).

    Similar to Ulcer Index but uses mean instead of RMS.

    Parameters:
    -----------
    prices : pd.Series or np.ndarray
        Price series

    Returns:
    --------
    float: Pain Index
    """
    prices = np.asarray(prices)
    running_max = np.maximum.accumulate(prices)
    drawdown_pct = (running_max - prices) / running_max
    return np.mean(drawdown_pct)


def calculate_omega_ratio(returns: Union[pd.Series, np.ndarray],
                         threshold: float = 0.0) -> float:
    """
    Calculate Omega ratio.

    Ratio of probability-weighted gains to probability-weighted losses.

    Parameters:
    -----------
    returns : pd.Series or np.ndarray
        Return series
    threshold : float
        Return threshold (default: 0)

    Returns:
    --------
    float: Omega ratio (higher is better)
    """
    returns = np.asarray(returns)
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]

    if len(losses) == 0 or np.sum(losses) == 0:
        return np.inf if len(gains) > 0 else 1.0

    return np.sum(gains) / np.sum(losses)


def optimize_cvar_portfolio(expected_returns: np.ndarray,
                           scenarios: np.ndarray,
                           confidence: float = 0.95,
                           target_return: Optional[float] = None) -> np.ndarray:
    """
    Optimize portfolio to minimize CVaR using linear programming.

    Uses the Rockafellar-Uryasev formulation.

    Parameters:
    -----------
    expected_returns : np.ndarray
        Expected returns for each asset
    scenarios : np.ndarray
        Historical or simulated return scenarios, shape (n_scenarios, n_assets)
    confidence : float
        Confidence level for CVaR
    target_return : float, optional
        Minimum required portfolio return

    Returns:
    --------
    np.ndarray: Optimal portfolio weights
    """
    if not CVXPY_AVAILABLE:
        raise ImportError("cvxpy is required for CVaR optimization")

    n_scenarios, n_assets = scenarios.shape
    alpha = 1 - confidence

    # Variables
    weights = cp.Variable(n_assets)
    zeta = cp.Variable()  # VaR auxiliary variable
    u = cp.Variable(n_scenarios)  # Shortfall auxiliary variables

    # Portfolio returns for each scenario
    portfolio_returns = scenarios @ weights

    # CVaR formulation
    objective = cp.Minimize(zeta + (1 / (n_scenarios * alpha)) * cp.sum(u))

    constraints = [
        u >= 0,
        u >= -portfolio_returns - zeta,
        cp.sum(weights) == 1,
        weights >= 0,  # Long-only
    ]

    if target_return is not None:
        constraints.append(expected_returns @ weights >= target_return)

    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status != 'optimal':
        raise ValueError(f"Optimization failed: {problem.status}")

    return weights.value
