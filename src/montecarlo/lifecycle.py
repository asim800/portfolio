#!/usr/bin/env python3
"""
Lifecycle Simulation Module.

Provides accumulation and decumulation Monte Carlo simulations
for retirement planning.

Extracted from src/visualize_mc_lifecycle.py.
"""

import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt

from .path_generator import MCPathGenerator


def run_accumulation_mc(
    initial_value: float,
    weights: np.ndarray,
    asset_returns_paths: np.ndarray,
    asset_returns_frequency: int,
    years: int,
    contributions_per_year: int = 1,
    contribution_amount: float = 0.0,
    employer_match_rate: float = 0.0,
    employer_match_cap: Optional[float] = None
) -> np.ndarray:
    """
    Run Monte Carlo for accumulation phase with periodic contributions.

    Uses pre-generated asset-level return paths from MCPathGenerator,
    enabling portfolio comparison on identical market scenarios.

    Parameters:
    -----------
    initial_value : float
        Starting portfolio value
    weights : np.ndarray
        Portfolio weights (must sum to 1, length = num_assets)
    asset_returns_paths : np.ndarray
        Pre-generated asset returns from MCPathGenerator
        Shape: (num_simulations, total_periods, num_assets)
    asset_returns_frequency : int
        Frequency of asset returns (e.g., 26 for biweekly, 12 for monthly)
    years : int
        Number of years to simulate
    contributions_per_year : int
        Number of contributions per year (weekly=52, biweekly=26, monthly=12, annual=1)
    contribution_amount : float
        Contribution amount per period (e.g., $1000 per paycheck)
    employer_match_rate : float
        Employer match as fraction of employee contribution (e.g., 0.5 = 50% match)
    employer_match_cap : float, optional
        Maximum employer match per year (None = unlimited)

    Returns:
    --------
    np.ndarray: (num_simulations, total_periods+1) array of portfolio values at every period
    """
    num_simulations = asset_returns_paths.shape[0]
    expected_periods = years * contributions_per_year

    freq_ratio = asset_returns_frequency / contributions_per_year

    assert freq_ratio == int(freq_ratio), \
        f"asset_returns_frequency {asset_returns_frequency} / contributions_per_year {contributions_per_year} is not an integer"
    freq_ratio = int(freq_ratio)

    if freq_ratio > 1:
        # Reshape to match contribution frequency
        asset_returns_paths = np.expand_dims(asset_returns_paths, axis=2)
        asset_returns_paths = asset_returns_paths.reshape(
            num_simulations, expected_periods, freq_ratio, weights.shape[0]
        )
        # Use geometric compounding instead of additive sum
        # (1+r1)*(1+r2)*...*(1+rn) - 1 = compound return over n periods
        asset_returns_paths = np.prod(1 + asset_returns_paths, axis=2) - 1

    # Initialize results
    values = np.zeros((num_simulations, expected_periods + 1))
    values[:, 0] = initial_value

    # VECTORIZED: Process all simulations at once
    portfolio_values = np.full(num_simulations, initial_value, dtype=float)
    employer_match_ytd = 0.0

    for period in range(1, expected_periods + 1):
        # Get asset returns for this period: (num_simulations, num_assets)
        asset_returns = asset_returns_paths[:, period - 1, :]

        # Calculate portfolio returns (vectorized)
        portfolio_returns = MCPathGenerator.calculate_portfolio_return(weights, asset_returns)

        # Apply returns
        portfolio_values *= (1 + portfolio_returns)

        # Add contributions
        if contribution_amount > 0:
            portfolio_values += contribution_amount

            # Calculate employer match
            period_match = contribution_amount * employer_match_rate

            if employer_match_cap is not None:
                remaining_capacity = employer_match_cap - employer_match_ytd
                period_match = min(period_match, remaining_capacity)
                employer_match_ytd += period_match

            portfolio_values += period_match

        values[:, period] = portfolio_values

        # Reset YTD match at year boundaries
        if period % contributions_per_year == 0:
            employer_match_ytd = 0.0

    return values


def run_decumulation_mc(
    initial_values: np.ndarray,
    weights: np.ndarray,
    asset_returns_paths: np.ndarray,
    asset_returns_frequency: int,
    annual_withdrawal: float,
    inflation_rate: float,
    years: int,
    withdrawals_per_year: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Monte Carlo for decumulation phase with withdrawals.

    Uses pre-generated asset-level return paths from MCPathGenerator,
    enabling portfolio comparison on identical market scenarios.

    Parameters:
    -----------
    initial_values : np.ndarray
        Starting portfolio values from accumulation phase (one per simulation)
    weights : np.ndarray
        Portfolio weights (must sum to 1, length = num_assets)
    asset_returns_paths : np.ndarray
        Pre-generated period-level asset returns from MCPathGenerator
        Shape: (num_simulations, total_periods, num_assets)
    asset_returns_frequency : int
        Frequency of asset returns
    annual_withdrawal : float
        Initial annual withdrawal amount (before inflation)
    inflation_rate : float
        Annual inflation rate (e.g., 0.03 for 3%)
    years : int
        Number of years to simulate
    withdrawals_per_year : int
        Withdrawal frequency (1=annual, 12=monthly, 26=biweekly, 52=weekly)

    Returns:
    --------
    tuple: (portfolio_values, success_flags)
        - portfolio_values: (num_simulations, total_periods+1) array at every period
        - success_flags: (num_simulations,) boolean array (True = survived full period)
    """
    num_simulations = len(initial_values)
    expected_periods = years * withdrawals_per_year
    # expected_periods = asset_returns_paths.shape[1]

    freq_ratio = asset_returns_frequency / withdrawals_per_year

    assert freq_ratio == int(freq_ratio), \
        f"asset_returns_frequency {asset_returns_frequency} / withdrawals_per_year {withdrawals_per_year} is not an integer"
    freq_ratio = int(freq_ratio)

    if freq_ratio > 1:
        asset_returns_paths = np.expand_dims(asset_returns_paths, axis=2)
        asset_returns_paths = asset_returns_paths.reshape(
            num_simulations, expected_periods, freq_ratio, weights.shape[0]
        )
        # Use geometric compounding instead of additive sum
        # (1+r1)*(1+r2)*...*(1+rn) - 1 = compound return over n periods
        asset_returns_paths = np.prod(1 + asset_returns_paths, axis=2) - 1

    total_periods = asset_returns_paths.shape[1]

    if asset_returns_paths.shape[0] != num_simulations:
        raise ValueError(f"asset_returns_paths has {asset_returns_paths.shape[0]} simulations, "
                        f"expected {num_simulations}")
    if total_periods != expected_periods:
        raise ValueError(f"asset_returns_paths has {total_periods} periods, expected {expected_periods}")

    period_withdrawal = annual_withdrawal / withdrawals_per_year

    values = np.zeros((num_simulations, expected_periods + 1))
    values[:, 0] = initial_values
    success = np.ones(num_simulations, dtype=bool)

    for sim in range(num_simulations):
        portfolio_value = initial_values[sim]

        for period in range(1, expected_periods + 1):
            asset_returns = asset_returns_paths[sim, period - 1, :]
            portfolio_return = MCPathGenerator.calculate_portfolio_return(weights, asset_returns)

            portfolio_value *= (1 + portfolio_return)

            # if (period %freq_ratio) == 0:
            current_year = (period - 1) // withdrawals_per_year
            inflation_factor = (1 + inflation_rate) ** current_year
            withdrawal = period_withdrawal * inflation_factor

            portfolio_value -= withdrawal

            if portfolio_value <= 0:
                portfolio_value = 0
                success[sim] = False

            values[sim, period] = portfolio_value

    return values, success


def calculate_success_rate(success_flags: np.ndarray) -> float:
    """
    Calculate the success rate from success flags.

    Parameters:
    -----------
    success_flags : np.ndarray
        Boolean array of success flags from run_decumulation_mc

    Returns:
    --------
    float: Success rate (0.0 to 1.0)
    """
    return success_flags.mean()


def calculate_percentiles(values: np.ndarray,
                         percentiles: list = [5, 25, 50, 75, 95]) -> dict:
    """
    Calculate percentiles across simulations at each time period.

    Parameters:
    -----------
    values : np.ndarray
        Shape (num_simulations, num_periods)
    percentiles : list
        Percentiles to calculate

    Returns:
    --------
    dict: Mapping percentile name to array of values
    """
    return {
        f'{p}th': np.percentile(values, p, axis=0)
        for p in percentiles
    }
