#!/usr/bin/env python3
"""
Portfolio-level configuration.

Each portfolio has its own JSON config file specifying:
- Strategy type and rebalancing frequency
- Optimization method and parameters
- Constraints and risk parameters
- Covariance calculation method
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class PortfolioConfig:
    """
    Configuration for a single portfolio.

    Each portfolio is defined by its own JSON file with portfolio-specific settings.
    System-wide settings (dates, output paths, etc.) are in SystemConfig.
    """

    # ============================================================================
    # Identity
    # ============================================================================
    name: str                               # Portfolio name (e.g., "buy_and_hold_baseline")
    description: str = ""                   # Human-readable description

    # ============================================================================
    # Strategy Configuration
    # ============================================================================
    strategy_type: str = 'optimized'        # 'buy_and_hold', 'target_weight', 'equal_weight', 'spy_only', 'optimized'
    rebalancing_frequency: str = 'ME'       # Pandas frequency: 'ME', 'Q', '2W', '21D', etc.

    # Initial/target weights (depends on strategy)
    # For buy_and_hold: these are initial weights
    # For target_weight: these are target weights to rebalance to
    # For optimized: these can be initial weights for first period
    initial_weights: Dict[str, float] = field(default_factory=dict)

    # ============================================================================
    # Optimization Configuration (only for strategy_type='optimized')
    # ============================================================================
    optimization_method: Optional[str] = None    # 'mean_variance', 'robust_mean_variance', 'risk_parity', 'min_variance', 'max_sharpe'
    risk_aversion: float = 1.0                   # Risk aversion parameter for mean-variance
    uncertainty_level: float = 0.1               # Uncertainty level for robust optimization (0-1)

    # ============================================================================
    # Covariance Calculation (portfolio-specific)
    # ============================================================================
    covariance_method: str = 'sample'            # 'sample', 'exponential_weighted', 'shrunk', 'robust', 'factor_model'
    covariance_params: Dict[str, Any] = field(default_factory=dict)  # Method-specific parameters

    # ============================================================================
    # Portfolio Constraints
    # ============================================================================
    long_only: bool = True                       # No short positions
    min_weight: float = 0.001                    # Minimum weight per asset
    max_weight: float = 0.4                      # Maximum weight per asset
    max_concentration: float = 0.6               # Maximum single asset concentration
    sector_constraints: Dict[str, float] = field(default_factory=dict)  # Sector exposure limits

    # Advanced constraints
    max_turnover: Optional[float] = None         # Maximum portfolio turnover per rebalance
    max_volatility: Optional[float] = None       # Maximum portfolio volatility
    max_tracking_error: Optional[float] = None   # Maximum tracking error vs benchmark

    # ============================================================================
    # Costs
    # ============================================================================
    transaction_costs: float = 0.0               # Transaction cost as fraction (e.g., 0.001 = 0.1%)

    # ============================================================================
    # Mixed Portfolio (Cash + Equity)
    # ============================================================================
    cash_percentage: float = 0.0                 # Cash allocation (0.4 = 40% cash, 60% equity)
    cash_interest_rate: float = 0.03             # Annual interest rate on cash (portfolio-specific)

    # ============================================================================
    # Validation
    # ============================================================================

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate strategy type
        valid_strategies = ['buy_and_hold', 'target_weight', 'equal_weight', 'spy_only', 'optimized']
        if self.strategy_type not in valid_strategies:
            raise ValueError(f"strategy_type must be one of {valid_strategies}, got '{self.strategy_type}'")

        # Validate optimization method if optimized strategy
        if self.strategy_type == 'optimized':
            valid_methods = [
                'mean_variance', 'robust_mean_variance', 'risk_parity', 'min_variance',
                'max_sharpe', 'max_diversification', 'hierarchical_clustering', 'black_litterman'
            ]
            if self.optimization_method is None:
                raise ValueError(f"optimization_method required for optimized strategy")
            if self.optimization_method not in valid_methods:
                raise ValueError(f"optimization_method must be one of {valid_methods}, got '{self.optimization_method}'")

        # Validate risk parameters
        if self.risk_aversion <= 0:
            raise ValueError(f"risk_aversion must be positive, got {self.risk_aversion}")

        if not (0 <= self.uncertainty_level <= 1):
            raise ValueError(f"uncertainty_level must be between 0 and 1, got {self.uncertainty_level}")

        # Validate covariance method
        valid_cov_methods = ['sample', 'exponential_weighted', 'shrunk', 'robust', 'factor_model']
        if self.covariance_method not in valid_cov_methods:
            raise ValueError(f"covariance_method must be one of {valid_cov_methods}, got '{self.covariance_method}'")

        # Validate weight constraints
        if not (0 <= self.min_weight <= 1):
            raise ValueError(f"min_weight must be between 0 and 1, got {self.min_weight}")

        if not (0 <= self.max_weight <= 1):
            raise ValueError(f"max_weight must be between 0 and 1, got {self.max_weight}")

        if self.min_weight > self.max_weight:
            raise ValueError(f"min_weight ({self.min_weight}) cannot exceed max_weight ({self.max_weight})")

        if not (0 <= self.max_concentration <= 1):
            raise ValueError(f"max_concentration must be between 0 and 1, got {self.max_concentration}")

        # Validate cash parameters
        if not (0 <= self.cash_percentage <= 1):
            raise ValueError(f"cash_percentage must be between 0 and 1, got {self.cash_percentage}")

        if not (0 <= self.cash_interest_rate <= 1):
            raise ValueError(f"cash_interest_rate must be between 0 and 1, got {self.cash_interest_rate}")

        if not (0 <= self.transaction_costs <= 1):
            raise ValueError(f"transaction_costs must be between 0 and 1, got {self.transaction_costs}")

        # Validate frequency string (basic check - pandas will validate fully)
        if not isinstance(self.rebalancing_frequency, str) or not self.rebalancing_frequency:
            raise ValueError(f"rebalancing_frequency must be a non-empty string, got '{self.rebalancing_frequency}'")

    # ============================================================================
    # Serialization
    # ============================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PortfolioConfig':
        """Create PortfolioConfig from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, filepath: str) -> 'PortfolioConfig':
        """Load PortfolioConfig from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_json(self, filepath: str, indent: int = 2) -> None:
        """Save PortfolioConfig to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)

    # ============================================================================
    # Utility Methods
    # ============================================================================

    def is_optimized(self) -> bool:
        """Check if this portfolio uses optimization."""
        return self.strategy_type == 'optimized'

    def is_mixed_portfolio(self) -> bool:
        """Check if this is a mixed cash/equity portfolio."""
        return self.cash_percentage > 0

    def get_constraint_dict(self) -> Dict[str, Any]:
        """Get all constraints as a dictionary for optimizer."""
        return {
            'long_only': self.long_only,
            'min_weight': self.min_weight,
            'max_weight': self.max_weight,
            'max_concentration': self.max_concentration,
            'sector_constraints': self.sector_constraints,
            'max_turnover': self.max_turnover,
            'max_volatility': self.max_volatility,
            'max_tracking_error': self.max_tracking_error
        }

    def get_optimization_params(self) -> Dict[str, Any]:
        """Get optimization parameters for optimizer."""
        params = self.get_constraint_dict()

        if self.optimization_method in ['mean_variance', 'robust_mean_variance']:
            params['risk_aversion'] = self.risk_aversion

        if self.optimization_method == 'robust_mean_variance':
            params['uncertainty_level'] = self.uncertainty_level

        return params

    def get_covariance_config(self) -> Dict[str, Any]:
        """Get covariance calculation configuration."""
        return {
            'method': self.covariance_method,
            'params': self.covariance_params.copy()
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def load_portfolio_config(filepath: str) -> PortfolioConfig:
    """
    Load portfolio configuration from JSON file.

    Parameters:
    -----------
    filepath : str
        Path to portfolio JSON config file

    Returns:
    --------
    PortfolioConfig instance
    """
    return PortfolioConfig.from_json(filepath)


def load_multiple_portfolios(filepaths: list[str]) -> Dict[str, PortfolioConfig]:
    """
    Load multiple portfolio configs.

    Parameters:
    -----------
    filepaths : list[str]
        List of paths to portfolio JSON files

    Returns:
    --------
    Dict mapping portfolio name to PortfolioConfig
    """
    portfolios = {}
    for filepath in filepaths:
        config = load_portfolio_config(filepath)
        portfolios[config.name] = config
    return portfolios
