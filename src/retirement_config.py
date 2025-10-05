#!/usr/bin/env python3
"""
Retirement Configuration Module.
Minimal configuration for retirement Monte Carlo simulation.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime


@dataclass
class RetirementConfig:
    """Minimal configuration for retirement simulation"""

    # Portfolio
    initial_portfolio: float  # Starting portfolio value (e.g., 1_000_000)
    current_portfolio: Dict[str, float]  # Ticker: weight mapping

    # Timeline
    start_date: str  # Simulation start (YYYY-MM-DD)
    end_date: str  # Simulation end
    retirement_date: Optional[str] = None  # When withdrawals begin (None = already retired)

    # Withdrawals (simple for now)
    annual_withdrawal: float = 40_000  # Fixed annual withdrawal amount
    inflation_rate: float = 0.03  # Annual inflation rate

    # Strategy
    rebalancing_strategy: str = 'buy_and_hold'  # For now: buy_and_hold, target_weight, equal_weight

    def __post_init__(self):
        """Validate configuration"""
        # Validate dates
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')

        if end <= start:
            raise ValueError("end_date must be after start_date")

        if self.retirement_date:
            retire = datetime.strptime(self.retirement_date, '%Y-%m-%d')
            if retire < start or retire > end:
                raise ValueError("retirement_date must be between start_date and end_date")

        # Validate portfolio
        if self.initial_portfolio <= 0:
            raise ValueError("initial_portfolio must be positive")

        if not self.current_portfolio:
            raise ValueError("current_portfolio cannot be empty")

        weight_sum = sum(self.current_portfolio.values())
        if not (0.99 <= weight_sum <= 1.01):  # Allow small rounding errors
            raise ValueError(f"Portfolio weights must sum to 1.0, got {weight_sum}")

        # Validate withdrawal
        if self.annual_withdrawal < 0:
            raise ValueError("annual_withdrawal cannot be negative")

        if not (0 <= self.inflation_rate <= 0.20):
            raise ValueError("inflation_rate must be between 0 and 0.20")

    @property
    def num_years(self) -> int:
        """Calculate number of years in simulation"""
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')
        return (end - start).days // 365

    @property
    def tickers(self) -> list:
        """Get list of tickers in portfolio"""
        return list(self.current_portfolio.keys())

    @property
    def withdrawal_rate(self) -> float:
        """Calculate initial withdrawal rate as percentage of portfolio"""
        return self.annual_withdrawal / self.initial_portfolio

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'initial_portfolio': self.initial_portfolio,
            'current_portfolio': self.current_portfolio,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'retirement_date': self.retirement_date,
            'annual_withdrawal': self.annual_withdrawal,
            'inflation_rate': self.inflation_rate,
            'rebalancing_strategy': self.rebalancing_strategy,
            'num_years': self.num_years,
            'withdrawal_rate': self.withdrawal_rate
        }
