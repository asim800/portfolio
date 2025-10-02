#!/usr/bin/env python3
"""
Configuration module for portfolio rebalancing system.
Contains all configurable parameters for dynamic portfolio optimization.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


@dataclass
class RebalancingConfig:
    """Configuration for dynamic portfolio rebalancing system."""
    
    # Rebalancing Parameters
    rebalancing_period_days: int = 30  # Calendar days between rebalancing
    min_history_periods: int = 2  # Minimum periods needed before optimization starts
    use_expanding_window: bool = True  # True for expanding, False for rolling window
    rolling_window_periods: int = 6  # Only used if expanding_window = False
    
    # Portfolio Configuration (simplified single portfolio mode)
    single_portfolio_mode: bool = False  # Run one optimization method by default
    optimization_methods: List[str] = field(default_factory=lambda: ['mean_variance'])  # Single method by default
    comparison_mode: bool = False  # Set to True to compare multiple optimization methods

    # Portfolio Comparison Configuration
    comparison_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [
        ('buy_and_hold', 'mean_variance'),    # Classic active vs passive
        ('spy_only', 'mean_variance'),        # Benchmark vs optimization
        ('equal_weight', 'mean_variance'),    # Naive vs optimization
    ])  # Pairs of portfolios to compare directly
    
    # Rebalancing Strategies Configuration
    rebalancing_strategies: List[str] = field(default_factory=lambda: [
        'buy_and_hold',      # True static baseline - never rebalances
        'target_weight',     # Rebalance to original weights each period
        'equal_weight',      # Rebalance to equal weights each period
        'spy_only'           # 100% SPY allocation for market comparison
    ])
    rebalancing_strategy_periods: Dict[str, int] = field(default_factory=lambda: {
        'target_weight': 30,  # Rebalance every 30 days
        'equal_weight': 30,   # Rebalance every 30 days
        'threshold': 30       # Check threshold every 30 days
    })
    
    # Legacy baseline configuration (for backward compatibility)
    include_baseline: bool = True  # Static baseline (original weights, never rebalanced) 
    include_rebalanced_baseline: bool = False  # Replaced by rebalancing strategies
    
    # Covariance Matrix Calculation
    covariance_method: str = 'sample'  # 'sample', 'exponential_weighted', 'shrunk', 'robust', 'factor_model'
    covariance_params: Dict[str, Any] = field(default_factory=dict)  # Method-specific parameters
    
    # Risk Parameters
    risk_aversion: float = 1.0  # Base risk aversion for mean-variance optimization
    uncertainty_level: float = 0.1  # Uncertainty level for robust optimization (10%)
    risk_free_rate: float = 0.02  # Risk-free rate for Sharpe ratio calculation
    
    # Portfolio Constraints
    long_only: bool = True  # Whether to enforce long-only constraints
    min_weight: float = 0.001  # Minimum weight per asset
    max_weight: float = 0.4  # Maximum weight per asset
    max_concentration: float = 0.6  # Maximum single asset weight (if different from max_weight)
    sector_constraints: Dict[str, float] = field(default_factory=dict)  # Sector exposure limits
    max_turnover: Optional[float] = None  # Maximum portfolio turnover
    max_volatility: Optional[float] = None  # Maximum portfolio volatility
    max_tracking_error: Optional[float] = None  # Maximum tracking error vs benchmark
    
    # Data Parameters
    start_date: str = '2024-01-01'
    end_date: str = '2024-12-31'
    ticker_file: str = '../tickers.txt'
    
    # Performance Tracking
    metrics_to_track: List[str] = field(default_factory=lambda: [
        'returns', 'volatility', 'sharpe_ratio', 'max_drawdown', 
        'calmar_ratio', 'gain_pain_ratio', 'beta', 'total_weight'
    ])
    
    # Output Configuration
    save_plots: bool = True
    show_plots_interactive: bool = True  # Whether to show plots interactively
    close_plots_after_save: bool = False  # Whether to close plots after saving (True = non-blocking, False = keep open)
    plots_directory: str = '../plots/rebalancing'
    save_results: bool = True
    results_directory: str = '../results/rebalancing'
    
    # Mixed Portfolio Options (disabled by default in single portfolio mode)
    include_mixed_portfolios: bool = False  # Whether to track mixed cash/equity portfolios
    mixed_cash_percentage: float = 0.4  # Target cash allocation (40% default)
    cash_interest_rate: float = 0.03  # Annual cash interest rate (3% default)
    
    # Advanced Options (for future use)
    transaction_costs: float = 0.0  # Transaction cost percentage (not implemented yet)
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.rebalancing_period_days <= 0:
            raise ValueError("rebalancing_period_days must be positive")
        
        if self.min_history_periods < 1:
            raise ValueError("min_history_periods must be at least 1")
        
        if not self.optimization_methods:
            raise ValueError("At least one optimization method must be specified")
        
        # Validate single portfolio mode constraints
        if self.single_portfolio_mode and not self.comparison_mode:
            if len(self.optimization_methods) > 1:
                raise ValueError("In single portfolio mode, only one optimization method allowed. "
                               "Set comparison_mode=True to use multiple methods.")
        
        if self.comparison_mode and len(self.optimization_methods) < 2:
            raise ValueError("Comparison mode requires at least 2 optimization methods")
        
        if self.risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")
        
        if not (0 <= self.uncertainty_level <= 1):
            raise ValueError("uncertainty_level must be between 0 and 1")
        
        if not (0 <= self.risk_free_rate <= 1):
            raise ValueError("risk_free_rate must be between 0 and 1")
        
        # Validate mixed portfolio parameters
        if not (0 <= self.mixed_cash_percentage <= 1):
            raise ValueError("mixed_cash_percentage must be between 0 and 1")
        
        if not (0 <= self.cash_interest_rate <= 1):
            raise ValueError("cash_interest_rate must be between 0 and 1")
        
        # Validate covariance method
        valid_cov_methods = ['sample', 'exponential_weighted', 'shrunk', 'robust', 'factor_model']
        if self.covariance_method not in valid_cov_methods:
            raise ValueError(f"covariance_method must be one of {valid_cov_methods}")
        
        # Validate weight constraints
        if not (0 <= self.min_weight <= 1):
            raise ValueError("min_weight must be between 0 and 1")
        
        if not (0 <= self.max_weight <= 1):
            raise ValueError("max_weight must be between 0 and 1")
        
        if self.min_weight > self.max_weight:
            raise ValueError("min_weight cannot be greater than max_weight")
        
        if not (0 <= self.max_concentration <= 1):
            raise ValueError("max_concentration must be between 0 and 1")
        
        # Validate optimization methods (keep legacy support)
        valid_methods = [
            'mean_variance', 'robust_mean_variance', 'risk_parity', 'min_variance',
            'max_sharpe', 'max_diversification', 'hierarchical_clustering', 'black_litterman',
            'vanilla', 'robust'  # Legacy method names for backward compatibility
        ]
        
        for method in self.optimization_methods:
            if method not in valid_methods:
                raise ValueError(f"Unknown optimization method '{method}'. Valid methods: {valid_methods}")
        
        # Validate rebalancing strategies
        valid_strategies = ['buy_and_hold', 'target_weight', 'equal_weight', 'threshold', 'spy_only']
        for strategy in self.rebalancing_strategies:
            if strategy not in valid_strategies:
                raise ValueError(f"Unknown rebalancing strategy '{strategy}'. Valid strategies: {valid_strategies}")
        
        # Validate rebalancing periods
        for strategy, period in self.rebalancing_strategy_periods.items():
            if not isinstance(period, int) or period <= 0:
                raise ValueError(f"Rebalancing period for '{strategy}' must be a positive integer")
        
        # Validate dates
        try:
            datetime.strptime(self.start_date, '%Y-%m-%d')
            datetime.strptime(self.end_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Dates must be in YYYY-MM-DD format")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for easy serialization."""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RebalancingConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def get_period_length_days(self) -> int:
        """Get the rebalancing period length in days."""
        return self.rebalancing_period_days
    
    def get_min_history_days(self) -> int:
        """Get minimum history required in days."""
        return self.min_history_periods * self.rebalancing_period_days
    
    def is_method_enabled(self, method: str) -> bool:
        """Check if a specific optimization method is enabled."""
        return method in self.optimization_methods
    
    def get_optimization_params(self, method: str) -> Dict[str, Any]:
        """Get optimization parameters for a specific method."""
        # Map legacy method names to new ones
        method_mapping = {
            'vanilla': 'mean_variance',
            'robust': 'robust_mean_variance'
        }
        
        normalized_method = method_mapping.get(method, method)
        
        # Base parameters for all methods
        base_params = {
            'long_only': self.long_only,
            'min_weight': self.min_weight,
            'max_weight': self.max_weight,
            'max_concentration': self.max_concentration,
            'sector_constraints': self.sector_constraints,
            'max_turnover': self.max_turnover,
            'max_volatility': self.max_volatility,
            'max_tracking_error': self.max_tracking_error
        }
        
        # Method-specific parameters
        if normalized_method in ['mean_variance', 'robust_mean_variance']:
            base_params['risk_aversion'] = self.risk_aversion
        
        if normalized_method == 'robust_mean_variance':
            base_params['uncertainty_level'] = self.uncertainty_level
        
        return base_params
    
    def get_covariance_params(self) -> Dict[str, Any]:
        """Get covariance calculation parameters."""
        return self.covariance_params.copy()
    
    def normalize_method_name(self, method: str) -> str:
        """Convert legacy method names to new standardized names."""
        method_mapping = {
            'vanilla': 'mean_variance',
            'robust': 'robust_mean_variance'
        }
        return method_mapping.get(method, method)
    
    def get_constraint_dict(self) -> Dict[str, Any]:
        """Get all portfolio constraints as a dictionary."""
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
    
    def get_rebalancing_period(self, strategy: str) -> int:
        """Get rebalancing period for a specific strategy."""
        return self.rebalancing_strategy_periods.get(strategy, 30)  # Default to 30 days
    
    def get_portfolio_names(self) -> List[str]:
        """Get list of all portfolio names that will be tracked."""
        portfolio_names = []
        
        # Add rebalancing strategies
        portfolio_names.extend(self.rebalancing_strategies)
        
        # Add optimization methods
        portfolio_names.extend(self.optimization_methods)
        
        # Add mixed portfolios if enabled
        if self.include_mixed_portfolios:
            portfolio_names += [f'mixed_{method}' for method in self.optimization_methods]
        
        # Add legacy baselines if configured
        if self.include_baseline and 'buy_and_hold' not in self.rebalancing_strategies:
            portfolio_names.append('static_baseline')
        if self.include_rebalanced_baseline:
            portfolio_names.append('rebalanced_baseline')
        
        return portfolio_names


# Default configuration instance
DEFAULT_CONFIG = RebalancingConfig()


def create_custom_config(**kwargs) -> RebalancingConfig:
    """Create a custom configuration with specified parameters."""
    return RebalancingConfig(**kwargs)


def load_config_from_file(file_path: str) -> RebalancingConfig:
    """Load configuration from a JSON or YAML file (future implementation)."""
    # TODO: Implement file loading when needed
    _ = file_path  # Acknowledge parameter for future use
    raise NotImplementedError("Configuration file loading not yet implemented")


def save_config_to_file(config: RebalancingConfig, file_path: str) -> None:
    """Save configuration to a JSON or YAML file (future implementation)."""
    # TODO: Implement file saving when needed
    _ = config, file_path  # Acknowledge parameters for future use
    raise NotImplementedError("Configuration file saving not yet implemented")