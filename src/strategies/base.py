#!/usr/bin/env python3
"""
Base classes for portfolio strategies.

Provides abstract base classes for allocation strategies and rebalancing triggers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class PortfolioStrategy(ABC):
    """
    Abstract base class for all portfolio strategies.

    A portfolio strategy combines an allocation method (WHAT weights to use)
    with a rebalancing trigger (WHEN to rebalance).
    """

    @abstractmethod
    def update(self, price_multipliers: Dict[str, float]) -> float:
        """
        Update the strategy with new price data.

        Parameters:
        -----------
        price_multipliers : Dict[str, float]
            Dictionary mapping asset names to price multipliers
            (e.g., {'stocks': 1.02, 'bonds': 1.01} for +2% stocks, +1% bonds)

        Returns:
        --------
        float: Portfolio return for this period
        """
        pass

    @abstractmethod
    def get_allocation(self) -> np.ndarray:
        """
        Get current portfolio allocation.

        Returns:
        --------
        np.ndarray or float: Current allocation weights
        """
        pass

    @property
    def wealth_history(self) -> list:
        """Return wealth history."""
        raise NotImplementedError

    @property
    def allocation_history(self) -> list:
        """Return allocation history."""
        raise NotImplementedError


class AllocationStrategyBase(ABC):
    """
    Abstract base class for allocation strategies.

    Defines WHAT weights to use for portfolio allocation.
    """

    @abstractmethod
    def calculate_weights(self,
                         returns: Optional[np.ndarray] = None,
                         cov_matrix: Optional[np.ndarray] = None,
                         current_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate target portfolio weights.

        Parameters:
        -----------
        returns : np.ndarray, optional
            Expected returns for each asset
        cov_matrix : np.ndarray, optional
            Covariance matrix
        current_weights : np.ndarray, optional
            Current portfolio weights

        Returns:
        --------
        np.ndarray: Target weights (should sum to 1)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return strategy name."""
        pass


class RebalancingTriggerBase(ABC):
    """
    Abstract base class for rebalancing triggers.

    Defines WHEN to rebalance the portfolio.
    """

    @abstractmethod
    def should_rebalance(self,
                        current_weights: np.ndarray,
                        target_weights: np.ndarray,
                        period: int,
                        date: Optional[Any] = None) -> bool:
        """
        Determine if rebalancing should occur.

        Parameters:
        -----------
        current_weights : np.ndarray
            Current portfolio weights
        target_weights : np.ndarray
            Target portfolio weights
        period : int
            Current period number
        date : Any, optional
            Current date

        Returns:
        --------
        bool: True if should rebalance, False otherwise
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return trigger name."""
        pass
