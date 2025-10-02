#!/usr/bin/env python3
"""
Rebalancing Strategies Module.
Implements different portfolio rebalancing strategies with configurable periods.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, date
from abc import ABC, abstractmethod


class BaseRebalancingStrategy(ABC):
    """
    Abstract base class for rebalancing strategies.
    """
    
    def __init__(self, rebalancing_period_days: int = 30, **kwargs):
        """
        Initialize base rebalancing strategy.
        
        Parameters:
        -----------
        rebalancing_period_days : int
            How often to rebalance in calendar days (default: 30)
        **kwargs : dict
            Strategy-specific parameters
        """
        self.rebalancing_period_days = rebalancing_period_days
        self.strategy_params = kwargs
        self.last_rebalance_date: Optional[date] = None
        
        logging.info(f"Initialized {self.__class__.__name__} with {rebalancing_period_days}-day periods")
    
    @abstractmethod
    def calculate_target_weights(self, 
                               current_weights: np.ndarray,
                               period_returns: pd.Series,
                               current_date: date,
                               **kwargs) -> np.ndarray:
        """
        Calculate target portfolio weights for rebalancing.
        
        Parameters:
        -----------
        current_weights : np.ndarray
            Current portfolio weights (after drift)
        period_returns : pd.Series
            Returns for the current period
        current_date : date
            Current date
        **kwargs : dict
            Additional parameters
            
        Returns:
        --------
        np.ndarray of target weights
        """
        pass
    
    def needs_rebalancing(self, current_date: date) -> bool:
        """
        Check if portfolio needs rebalancing based on calendar schedule.
        
        Parameters:
        -----------
        current_date : date
            Current date
            
        Returns:
        --------
        bool indicating if rebalancing is needed
        """
        if self.last_rebalance_date is None:
            # First period - always rebalance to establish baseline
            return True
        
        days_since_rebalance = (current_date - self.last_rebalance_date).days
        return days_since_rebalance >= self.rebalancing_period_days
    
    def execute_rebalancing(self, 
                          current_weights: np.ndarray,
                          period_returns: pd.Series,
                          current_date: date,
                          **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Execute rebalancing if needed.
        
        Parameters:
        -----------
        current_weights : np.ndarray
            Current portfolio weights
        period_returns : pd.Series
            Period returns data
        current_date : date
            Current date
        **kwargs : dict
            Additional parameters
            
        Returns:
        --------
        Tuple of (new_weights, rebalancing_info)
        """
        if self.needs_rebalancing(current_date):
            # Calculate target weights
            target_weights = self.calculate_target_weights(
                current_weights, period_returns, current_date, **kwargs
            )
            
            # Calculate rebalancing trades
            trades = self.calculate_trades(current_weights, target_weights)
            
            # Update last rebalance date
            self.last_rebalance_date = current_date
            
            rebalancing_info = {
                'rebalanced': True,
                'rebalance_date': current_date,
                'target_weights': target_weights.copy(),
                'trades': trades,
                'total_turnover': np.sum(np.abs(target_weights - current_weights))
            }
            
            return target_weights, rebalancing_info
        else:
            # No rebalancing needed - return current weights
            rebalancing_info = {
                'rebalanced': False,
                'rebalance_date': None,
                'target_weights': current_weights.copy(),
                'trades': {},
                'total_turnover': 0.0
            }
            
            return current_weights, rebalancing_info
    
    def calculate_trades(self, current_weights: np.ndarray, target_weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate trades needed to move from current to target weights.
        
        Parameters:
        -----------
        current_weights : np.ndarray
            Current portfolio weights
        target_weights : np.ndarray
            Target portfolio weights
            
        Returns:
        --------
        Dict with trade information
        """
        weight_changes = target_weights - current_weights
        
        return {
            'weight_changes': weight_changes,
            'total_turnover': np.sum(np.abs(weight_changes)),
            'max_trade_size': np.max(np.abs(weight_changes)),
            'num_trades': np.sum(np.abs(weight_changes) > 1e-6)  # Count non-trivial trades
        }


class BuyAndHoldStrategy(BaseRebalancingStrategy):
    """
    Buy-and-hold strategy - never rebalances.
    Maintains fixed share counts and lets portfolio weights drift with market performance.
    """

    def __init__(self, **kwargs):
        # Buy and hold never rebalances, so period doesn't matter
        super().__init__(rebalancing_period_days=float('inf'), **kwargs)
        self.initial_weights: Optional[np.ndarray] = None
        self.initial_portfolio_value: Optional[float] = None
        self.share_counts: Optional[np.ndarray] = None

    def calculate_target_weights(self,
                               current_weights: np.ndarray,
                               period_returns: pd.Series,
                               current_date: date,
                               **kwargs) -> np.ndarray:
        """
        For buy-and-hold, return current market weights based on fixed share counts.
        This allows weights to drift naturally with market performance.
        """
        if self.initial_weights is None:
            # First time - set initial weights and calculate share counts
            self.initial_weights = current_weights.copy()
            self.initial_portfolio_value = 1.0  # Assume starting with $1

            # Calculate initial share counts based on weights
            # share_counts[i] = (weight[i] * portfolio_value) / price[i]
            # Since we normalize, we can assume price = 1 initially
            self.share_counts = current_weights.copy()

            logging.info(f"BuyAndHold: Initial weights: {self.initial_weights}")
            logging.info(f"BuyAndHold: Initial share counts: {self.share_counts}")

            return self.initial_weights

        # For subsequent periods, return the current weights (already drifted)
        # The weights have naturally drifted due to different asset performance
        return current_weights

    def needs_rebalancing(self, current_date: date) -> bool:
        """Buy-and-hold never rebalances after the initial setup."""
        return self.initial_weights is None


class TargetWeightStrategy(BaseRebalancingStrategy):
    """
    Target weight rebalancing strategy.
    Rebalances to maintain original target allocation percentages.
    """
    
    def __init__(self, rebalancing_period_days: int = 30, target_weights: Optional[np.ndarray] = None, **kwargs):
        super().__init__(rebalancing_period_days, **kwargs)
        self.target_weights = target_weights
    
    def calculate_target_weights(self, 
                               current_weights: np.ndarray,
                               period_returns: pd.Series,
                               current_date: date,
                               target_weights: Optional[np.ndarray] = None,
                               **kwargs) -> np.ndarray:
        """
        Return the original target weights.
        
        Parameters:
        -----------
        target_weights : np.ndarray, optional
            Target weights to rebalance to (if not set in __init__)
        """
        # Use provided target weights or stored target weights
        if target_weights is not None:
            weights_to_use = target_weights
        elif self.target_weights is not None:
            weights_to_use = self.target_weights
        else:
            # If no target weights specified, use current weights as target
            weights_to_use = current_weights
            # Store for future use
            self.target_weights = current_weights.copy()
        
        # Ensure weights sum to 1
        return weights_to_use / np.sum(weights_to_use)


class EqualWeightStrategy(BaseRebalancingStrategy):
    """
    Equal weight rebalancing strategy.
    Rebalances to equal weights (1/N) across all assets.
    """
    
    def __init__(self, rebalancing_period_days: int = 30, **kwargs):
        super().__init__(rebalancing_period_days, **kwargs)
    
    def calculate_target_weights(self, 
                               current_weights: np.ndarray,
                               period_returns: pd.Series,
                               current_date: date,
                               **kwargs) -> np.ndarray:
        """Return equal weights for all assets."""
        n_assets = len(current_weights)
        return np.ones(n_assets) / n_assets


class ThresholdRebalancingStrategy(BaseRebalancingStrategy):
    """
    Threshold-based rebalancing strategy.
    Only rebalances when weights drift beyond specified tolerance.
    """
    
    def __init__(self, 
                 rebalancing_period_days: int = 30,
                 drift_threshold: float = 0.05,
                 target_weights: Optional[np.ndarray] = None,
                 **kwargs):
        super().__init__(rebalancing_period_days, **kwargs)
        self.drift_threshold = drift_threshold
        self.target_weights = target_weights
    
    def needs_rebalancing(self, current_date: date, current_weights: Optional[np.ndarray] = None) -> bool:
        """
        Check if rebalancing is needed based on both time and threshold criteria.
        
        Parameters:
        -----------
        current_date : date
            Current date
        current_weights : np.ndarray, optional
            Current portfolio weights for threshold check
        """
        # Always check calendar-based rebalancing first
        calendar_rebalance = super().needs_rebalancing(current_date)
        
        if not calendar_rebalance or current_weights is None or self.target_weights is None:
            return calendar_rebalance
        
        # Check if any weight has drifted beyond threshold
        weight_drifts = np.abs(current_weights - self.target_weights)
        max_drift = np.max(weight_drifts)
        
        threshold_rebalance = max_drift > self.drift_threshold
        
        return calendar_rebalance or threshold_rebalance
    
    def calculate_target_weights(self, 
                               current_weights: np.ndarray,
                               period_returns: pd.Series,
                               current_date: date,
                               target_weights: Optional[np.ndarray] = None,
                               **kwargs) -> np.ndarray:
        """Return the target weights, setting them if not already defined."""
        if target_weights is not None:
            weights_to_use = target_weights
        elif self.target_weights is not None:
            weights_to_use = self.target_weights
        else:
            # Set initial target weights
            weights_to_use = current_weights.copy()
            self.target_weights = weights_to_use
        
        return weights_to_use / np.sum(weights_to_use)


class SpyOnlyStrategy(BaseRebalancingStrategy):
    """
    SPY-only strategy for market comparison.
    Allocates 100% to SPY and 0% to all other assets.
    """
    
    def __init__(self, rebalancing_period_days: int = 30, **kwargs):
        super().__init__(rebalancing_period_days, **kwargs)
    
    def calculate_target_weights(self, 
                               current_weights: np.ndarray,
                               period_returns: pd.Series,
                               current_date: date,
                               baseline_weights: Optional[np.ndarray] = None,
                               **kwargs) -> np.ndarray:
        """
        Always return 100% SPY allocation.
        
        Parameters:
        -----------
        current_weights : np.ndarray
            Current portfolio weights (not used for SPY-only)
        period_returns : pd.Series
            Period returns with asset names as index
        current_date : date
            Current date
        baseline_weights : np.ndarray, optional
            Baseline weights to determine asset order
        **kwargs : dict
            Additional parameters
            
        Returns:
        --------
        np.ndarray
            Weights array with 100% SPY, 0% everything else
        """
        # Create weights array with same length as current weights
        target_weights = np.zeros_like(current_weights)
        
        # Find SPY index from period_returns or assume SPY is in the asset list
        if hasattr(period_returns, 'index'):
            asset_names = period_returns.index.tolist()
        else:
            # Fallback: assume standard order BIL, MSFT, NVDA, SPY
            asset_names = ['BIL', 'MSFT', 'NVDA', 'SPY']
        
        # Find SPY index
        spy_index = -1
        for i, asset in enumerate(asset_names):
            if 'SPY' in str(asset):
                spy_index = i
                break
        
        # If SPY found, allocate 100% to it
        if spy_index >= 0 and spy_index < len(target_weights):
            target_weights[spy_index] = 1.0
        else:
            # Fallback: assume SPY is the last asset
            target_weights[-1] = 1.0
        
        return target_weights


class RebalancingStrategies:
    """
    Factory class for creating and managing different rebalancing strategies.
    """
    
    def __init__(self):
        """Initialize the strategies factory."""
        self.available_strategies = {
            'buy_and_hold': BuyAndHoldStrategy,
            'target_weight': TargetWeightStrategy,
            'equal_weight': EqualWeightStrategy,
            'threshold': ThresholdRebalancingStrategy,
            'spy_only': SpyOnlyStrategy
        }
        
        logging.info(f"RebalancingStrategies initialized with {len(self.available_strategies)} strategies")
    
    def create_strategy(self, 
                       strategy_type: str, 
                       rebalancing_period_days: int = 30,
                       **kwargs) -> BaseRebalancingStrategy:
        """
        Create a rebalancing strategy instance.
        
        Parameters:
        -----------
        strategy_type : str
            Type of strategy ('buy_and_hold', 'target_weight', 'equal_weight', 'threshold')
        rebalancing_period_days : int
            Rebalancing period in days (default: 30)
        **kwargs : dict
            Strategy-specific parameters
            
        Returns:
        --------
        BaseRebalancingStrategy instance
        """
        if strategy_type not in self.available_strategies:
            available = list(self.available_strategies.keys())
            raise ValueError(f"Unknown strategy '{strategy_type}'. Available: {available}")
        
        strategy_class = self.available_strategies[strategy_type]
        
        # Create strategy with period and additional parameters
        if strategy_type == 'buy_and_hold':
            # Buy and hold doesn't need a period
            return strategy_class(**kwargs)
        else:
            return strategy_class(rebalancing_period_days=rebalancing_period_days, **kwargs)
    
    def get_available_strategies(self) -> list:
        """Get list of available rebalancing strategies."""
        return list(self.available_strategies.keys())
    
    def get_strategy_parameters(self, strategy_type: str) -> Dict[str, Any]:
        """
        Get parameters for a specific rebalancing strategy.
        
        Parameters:
        -----------
        strategy_type : str
            Strategy type name
            
        Returns:
        --------
        Dict with parameter descriptions
        """
        strategy_params = {
            'buy_and_hold': {
                'description': 'Never rebalances after initial allocation',
                'parameters': {}
            },
            'target_weight': {
                'description': 'Rebalances to maintain original target weights',
                'parameters': {
                    'rebalancing_period_days': 30,
                    'target_weights': 'Original portfolio weights (optional)'
                }
            },
            'equal_weight': {
                'description': 'Rebalances to equal weights (1/N) across all assets',
                'parameters': {
                    'rebalancing_period_days': 30
                }
            },
            'threshold': {
                'description': 'Rebalances when weights drift beyond threshold',
                'parameters': {
                    'rebalancing_period_days': 30,
                    'drift_threshold': 0.05,
                    'target_weights': 'Target portfolio weights (optional)'
                }
            }
        }
        
        return strategy_params.get(strategy_type, {})


# Example usage functions
def create_default_strategies(rebalancing_period_days: int = 30, 
                            baseline_weights: Optional[np.ndarray] = None) -> Dict[str, BaseRebalancingStrategy]:
    """
    Create a standard set of rebalancing strategies for comparison.
    
    Parameters:
    -----------
    rebalancing_period_days : int
        Default rebalancing period in days
    baseline_weights : np.ndarray, optional
        Baseline portfolio weights for target weight strategy
        
    Returns:
    --------
    Dict of strategy name -> strategy instance
    """
    factory = RebalancingStrategies()
    
    strategies = {
        'buy_and_hold': factory.create_strategy('buy_and_hold'),
        'target_weight': factory.create_strategy(
            'target_weight', 
            rebalancing_period_days=rebalancing_period_days,
            target_weights=baseline_weights
        ),
        'equal_weight': factory.create_strategy(
            'equal_weight',
            rebalancing_period_days=rebalancing_period_days
        )
    }
    
    return strategies