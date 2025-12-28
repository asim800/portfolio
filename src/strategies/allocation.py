#!/usr/bin/env python3
"""
Allocation Strategies Module.

Defines WHAT weights to use for portfolio allocation.
Separate from WHEN to rebalance (see rebalancing_triggers.py).

Each strategy calculates target portfolio weights based on:
- Historical data (for optimization-based strategies)
- Fixed rules (for static strategies)
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, List
from abc import ABC, abstractmethod


class AllocationStrategy(ABC):
    """
    Abstract base class for allocation strategies.

    Allocation strategies determine WHAT weights the portfolio should have.
    They are independent of WHEN rebalancing occurs.
    """

    def __init__(self, name: str = "allocation_strategy"):
        """
        Initialize allocation strategy.

        Parameters:
        -----------
        name : str
            Strategy name for identification
        """
        self.name = name
        logging.info(f"Initialized {self.__class__.__name__}: {name}")

    @abstractmethod
    def calculate_weights(self,
                         current_weights: np.ndarray,
                         lookback_data: Optional[pd.DataFrame] = None,
                         **kwargs) -> np.ndarray:
        """
        Calculate target portfolio weights.

        Parameters:
        -----------
        current_weights : np.ndarray
            Current portfolio weights (may be used for reference)
        lookback_data : pd.DataFrame, optional
            Historical returns data (dates × assets)
            Required for optimization-based strategies
        **kwargs : dict
            Additional parameters

        Returns:
        --------
        np.ndarray
            Target weights (must sum to 1.0)
        """
        pass


class StaticAllocation(AllocationStrategy):
    """
    Static allocation with fixed weights.

    Examples: 60/40 stocks/bonds, equal weight, custom allocation.
    Weights never change regardless of market conditions.
    """

    def __init__(self, weights: np.ndarray, name: str = "static"):
        """
        Initialize static allocation.

        Parameters:
        -----------
        weights : np.ndarray or list
            Fixed portfolio weights
        name : str
            Strategy name

        Example:
        --------
        >>> # 60/40 portfolio
        >>> strategy = StaticAllocation([0.6, 0.4], name="60/40")
        """
        super().__init__(name)
        self.weights = np.array(weights)

        # Normalize to sum to 1
        self.weights = self.weights / self.weights.sum()

        logging.info(f"StaticAllocation: {self.weights}")

    def calculate_weights(self,
                         current_weights: np.ndarray,
                         lookback_data: Optional[pd.DataFrame] = None,
                         **kwargs) -> np.ndarray:
        """Return fixed weights (ignores current weights and data)."""
        return self.weights.copy()


class EqualWeight(AllocationStrategy):
    """
    Equal weight (1/N) allocation.

    Allocates equal weight to all assets regardless of characteristics.
    Simple, robust strategy that often performs surprisingly well.
    """

    def __init__(self, name: str = "equal_weight"):
        """Initialize equal weight allocation."""
        super().__init__(name)

    def calculate_weights(self,
                         current_weights: np.ndarray,
                         lookback_data: Optional[pd.DataFrame] = None,
                         **kwargs) -> np.ndarray:
        """Return equal weights for all assets."""
        n_assets = len(current_weights)
        return np.ones(n_assets) / n_assets


class SingleAsset(AllocationStrategy):
    """
    Single asset allocation (100% to one asset).

    Useful for benchmarking against individual assets (e.g., 100% SPY).
    """

    def __init__(self, asset_index: int, n_assets: int, name: str = "single_asset"):
        """
        Initialize single asset allocation.

        Parameters:
        -----------
        asset_index : int
            Index of asset to allocate 100% to (0-based)
        n_assets : int
            Total number of assets in portfolio
        name : str
            Strategy name

        Example:
        --------
        >>> # 100% SPY (assuming SPY is first asset)
        >>> strategy = SingleAsset(asset_index=0, n_assets=4, name="spy_only")
        """
        super().__init__(name)
        self.asset_index = asset_index
        self.n_assets = n_assets

        # Create weights array
        self.weights = np.zeros(n_assets)
        self.weights[asset_index] = 1.0

        logging.info(f"SingleAsset: 100% to asset {asset_index}")

    def calculate_weights(self,
                         current_weights: np.ndarray,
                         lookback_data: Optional[pd.DataFrame] = None,
                         **kwargs) -> np.ndarray:
        """Return single-asset weights."""
        return self.weights.copy()


class OptimizedAllocation(AllocationStrategy):
    """
    Optimization-based allocation using PortfolioOptimizer.

    Calculates optimal weights using quantitative methods:
    - Mean-variance optimization
    - Robust optimization
    - Risk parity
    - Minimum variance
    - Maximum Sharpe ratio
    etc.
    """

    def __init__(self,
                 optimizer,
                 method: str = 'mean_variance',
                 min_history_periods: int = 60,
                 name: Optional[str] = None):
        """
        Initialize optimization-based allocation.

        Parameters:
        -----------
        optimizer : PortfolioOptimizer
            Optimizer instance to use for calculations
        method : str
            Optimization method name
            Options: 'mean_variance', 'robust_mean_variance', 'risk_parity',
                    'min_variance', 'max_sharpe', etc.
        min_history_periods : int
            Minimum historical periods required for optimization
            If insufficient data, returns current weights
        name : str, optional
            Strategy name (defaults to method name)

        Example:
        --------
        >>> from portfolio_optimizer import PortfolioOptimizer
        >>> optimizer = PortfolioOptimizer(risk_free_rate=0.02)
        >>> strategy = OptimizedAllocation(optimizer, method='mean_variance')
        """
        if name is None:
            name = method
        super().__init__(name)

        self.optimizer = optimizer
        self.method = method
        self.min_history_periods = min_history_periods

        if optimizer is None:
            raise ValueError("optimizer is required for OptimizedAllocation")

        logging.info(f"OptimizedAllocation: method={method}, min_history={min_history_periods}")

    def calculate_weights(self,
                         current_weights: np.ndarray,
                         lookback_data: Optional[pd.DataFrame] = None,
                         **kwargs) -> np.ndarray:
        """
        Calculate optimal weights using portfolio optimizer.

        Parameters:
        -----------
        current_weights : np.ndarray
            Current portfolio weights (fallback if insufficient data)
        lookback_data : pd.DataFrame
            Historical returns data (dates × assets)
            Required for optimization
        **kwargs : dict
            Additional optimizer parameters

        Returns:
        --------
        np.ndarray
            Optimal weights
        """
        # Check if we have sufficient historical data
        if lookback_data is None or len(lookback_data) < self.min_history_periods:
            logging.debug(
                f"OptimizedAllocation: Insufficient history "
                f"({len(lookback_data) if lookback_data is not None else 0} periods), "
                f"using current weights"
            )
            return current_weights

        try:
            # Calculate mean returns and covariance matrix (pandas)
            mean_returns = lookback_data.mean() * 252  # Annualize
            cov_matrix = lookback_data.cov() * 252    # Annualize

            # Optimize using PortfolioOptimizer (convert to numpy at boundary)
            result = self.optimizer.optimize(
                method=self.method,
                mean_returns=mean_returns.values,  # pandas → numpy
                cov_matrix=cov_matrix.values,      # pandas → numpy
                **kwargs
            )

            if result is None or not result.get('success', False):
                logging.warning(
                    f"OptimizedAllocation: Optimization failed for {self.method}, "
                    f"using current weights"
                )
                return current_weights

            optimal_weights = result['weights']

            # Validate weights
            if optimal_weights is None or len(optimal_weights) != len(current_weights):
                logging.warning(
                    f"OptimizedAllocation: Invalid weights from optimizer, "
                    f"using current weights"
                )
                return current_weights

            # Normalize to ensure sum=1
            optimal_weights = optimal_weights / optimal_weights.sum()

            logging.debug(f"OptimizedAllocation: {self.method} weights = {optimal_weights}")

            return optimal_weights

        except Exception as e:
            logging.warning(
                f"OptimizedAllocation: Exception during optimization: {e}, "
                f"using current weights"
            )
            return current_weights


class InverseVolatility(AllocationStrategy):
    """
    Inverse volatility weighting.

    Allocates weights inversely proportional to asset volatility.
    Lower volatility assets get higher weights.
    Simple risk-based allocation without optimization.
    """

    def __init__(self, min_history_periods: int = 30, name: str = "inverse_vol"):
        """
        Initialize inverse volatility allocation.

        Parameters:
        -----------
        min_history_periods : int
            Minimum periods required to estimate volatility
        name : str
            Strategy name
        """
        super().__init__(name)
        self.min_history_periods = min_history_periods

    def calculate_weights(self,
                         current_weights: np.ndarray,
                         lookback_data: Optional[pd.DataFrame] = None,
                         **kwargs) -> np.ndarray:
        """Calculate weights inversely proportional to volatility."""
        # Check if we have sufficient data
        if lookback_data is None or len(lookback_data) < self.min_history_periods:
            logging.debug(
                f"InverseVolatility: Insufficient history, using equal weights"
            )
            return np.ones(len(current_weights)) / len(current_weights)

        try:
            # Calculate volatility for each asset
            volatilities = lookback_data.std()

            # Inverse volatility weights
            inverse_vol = 1.0 / volatilities
            weights = inverse_vol / inverse_vol.sum()

            return weights.values

        except Exception as e:
            logging.warning(
                f"InverseVolatility: Exception calculating weights: {e}, "
                f"using equal weights"
            )
            return np.ones(len(current_weights)) / len(current_weights)


# ============================================================================
# DEMO: Show the allocation strategies
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("ALLOCATION STRATEGIES DEMO")
    print("="*80)

    # Setup
    current_weights = np.array([0.25, 0.25, 0.25, 0.25])
    asset_names = ['SPY', 'AGG', 'NVDA', 'GLD']

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    returns_data = pd.DataFrame(
        np.random.randn(252, 4) * 0.01,
        index=dates,
        columns=asset_names
    )

    print("\n1. Static Allocation (60/40):")
    static = StaticAllocation([0.6, 0.4, 0.0, 0.0], name="60/40")
    weights = static.calculate_weights(current_weights)
    print(f"   Weights: {weights}")

    print("\n2. Equal Weight:")
    equal = EqualWeight()
    weights = equal.calculate_weights(current_weights)
    print(f"   Weights: {weights}")

    print("\n3. Single Asset (100% SPY):")
    spy_only = SingleAsset(asset_index=0, n_assets=4, name="spy_only")
    weights = spy_only.calculate_weights(current_weights)
    print(f"   Weights: {weights}")

    print("\n4. Inverse Volatility:")
    inv_vol = InverseVolatility()
    weights = inv_vol.calculate_weights(current_weights, returns_data)
    print(f"   Weights: {weights}")

    print("\n" + "="*80)
    print("All allocation strategies working correctly!")
    print("="*80)
