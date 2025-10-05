#!/usr/bin/env python3
"""
System-level configuration for portfolio backtesting framework.

This contains global settings that apply to ALL portfolios:
- Data environment (dates, tickers, risk-free rate)
- Backtest engine settings (history windows, metrics)
- Output configuration (where to save results)
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path


@dataclass
class SystemConfig:
    """
    Global configuration for the backtesting system.

    These settings apply to ALL portfolios in a backtest run.
    Portfolio-specific settings are in PortfolioConfig (separate JSON files).
    """

    # ============================================================================
    # Data & Environment Settings
    # ============================================================================
    start_date: str = '2024-01-01'          # Backtest start date (YYYY-MM-DD)
    end_date: str = '2024-12-31'            # Backtest end date (YYYY-MM-DD)
    ticker_file: str = '../tickers.txt'     # Path to ticker file
    risk_free_rate: float = 0.02            # Annual risk-free rate (used for Sharpe ratio across all portfolios)

    # ============================================================================
    # Retirement Monte Carlo Settings (Optional)
    # ============================================================================
    retirement_date: Optional[str] = None   # Retirement date (YYYY-MM-DD). Period between end_date and retirement_date is accumulation phase
    simulation_horizon_date: Optional[str] = None   # Simulation end date (YYYY-MM-DD). Alternative to simulation_horizon_years
    simulation_horizon_years: Optional[int] = None  # Simulation horizon in years from retirement_date. Alternative to simulation_horizon_date
    # Note: Period between retirement_date and simulation horizon is decumulation phase

    # Contribution/Savings Strategy (for accumulation phase - optional)
    contribution_amount: Optional[float] = None         # Fixed contribution amount per period
    contribution_frequency: str = 'biweekly'            # Contribution frequency: 'weekly', 'biweekly', 'monthly', 'annual'
    # Frequency → contributions per year: weekly=52, biweekly=26, monthly=12, annual=1
    employer_match_rate: float = 0.0                    # Employer match as % of contribution (e.g., 0.5 = 50% match)
    employer_match_cap: Optional[float] = None          # Maximum employer match per year (None = unlimited)

    # Withdrawal/Spending Strategy (for decumulation phase - optional, only used if retirement simulation configured)
    withdrawal_strategy: Optional[str] = None  # Withdrawal strategy type (None = no decumulation)
    # Options when configured:
    #   'constant_inflation_adjusted' - Fixed initial amount, adjusted for inflation each year (default 4% rule)
    #   'constant_percentage' - Fixed percentage of current portfolio each year
    #   'guyton_klinger' - Dynamic with guardrails (increase/decrease based on performance)
    #   'vpw' - Variable Percentage Withdrawal (age-based)
    #   'floor_ceiling' - Essential floor + discretionary ceiling
    #   'rmd' - Required Minimum Distribution tables (age 72+)

    annual_withdrawal_amount: Optional[float] = None        # Initial withdrawal amount (for constant strategies)
    withdrawal_percentage: Optional[float] = None           # Withdrawal percentage (for percentage-based strategies)
    inflation_rate: float = 0.03                            # Annual inflation rate for adjustments (default 3%)

    # Strategy-specific parameters
    withdrawal_strategy_params: Dict[str, Any] = field(default_factory=dict)
    # Examples:
    #   Guyton-Klinger: {'upper_guardrail': 0.20, 'lower_guardrail': 0.15, 'withdrawal_adjustment': 0.10}
    #   Floor-Ceiling: {'floor_amount': 30000, 'ceiling_amount': 60000}
    #   VPW: {'use_age_adjustment': True}

    # ============================================================================
    # Backtest Engine Settings
    # ============================================================================
    min_history_periods: int = 2            # Minimum periods of history before optimization starts
    use_expanding_window: bool = True       # True = expanding window, False = rolling window
    rolling_window_periods: int = 6         # Number of periods in rolling window (if use_expanding_window=False)

    # ============================================================================
    # Performance Metrics
    # ============================================================================
    metrics_to_track: List[str] = field(default_factory=lambda: [
        'returns',
        'volatility',
        'sharpe_ratio',
        'max_drawdown',
        'calmar_ratio',
        'gain_pain_ratio',
        'beta',
        'total_weight'
    ])

    # ============================================================================
    # Output Configuration
    # ============================================================================
    save_plots: bool = True                 # Whether to save plots to disk
    show_plots_interactive: bool = True     # Whether to display plots interactively
    close_plots_after_save: bool = False    # Whether to close plots after saving
    plots_directory: str = '../plots/rebalancing'

    save_results: bool = True               # Whether to save results to CSV
    results_directory: str = '../results/rebalancing'

    # ============================================================================
    # Validation
    # ============================================================================

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate dates
        try:
            start = datetime.strptime(self.start_date, '%Y-%m-%d')
            end = datetime.strptime(self.end_date, '%Y-%m-%d')
            if start >= end:
                raise ValueError(f"start_date ({self.start_date}) must be before end_date ({self.end_date})")
        except ValueError as e:
            raise ValueError(f"Dates must be in YYYY-MM-DD format: {e}")

        # Validate retirement Monte Carlo dates (if provided)
        if self.retirement_date is not None:
            try:
                retirement = datetime.strptime(self.retirement_date, '%Y-%m-%d')
                end = datetime.strptime(self.end_date, '%Y-%m-%d')
                if retirement < end:
                    raise ValueError(f"retirement_date ({self.retirement_date}) must be on or after end_date ({self.end_date})")
            except ValueError as e:
                if "retirement_date" in str(e):
                    raise  # Re-raise our custom error
                raise ValueError(f"retirement_date must be in YYYY-MM-DD format: {e}")

        # Validate simulation horizon (must specify one or the other, not both)
        if self.simulation_horizon_date is not None and self.simulation_horizon_years is not None:
            raise ValueError("Cannot specify both simulation_horizon_date and simulation_horizon_years. Choose one.")

        if self.simulation_horizon_date is not None:
            if self.retirement_date is None:
                raise ValueError("simulation_horizon_date requires retirement_date to be set")
            try:
                horizon = datetime.strptime(self.simulation_horizon_date, '%Y-%m-%d')
                retirement = datetime.strptime(self.retirement_date, '%Y-%m-%d')
                if horizon <= retirement:
                    raise ValueError(f"simulation_horizon_date ({self.simulation_horizon_date}) must be after retirement_date ({self.retirement_date})")
            except ValueError as e:
                raise ValueError(f"simulation_horizon_date must be in YYYY-MM-DD format: {e}")

        if self.simulation_horizon_years is not None:
            if self.retirement_date is None:
                raise ValueError("simulation_horizon_years requires retirement_date to be set")
            if self.simulation_horizon_years <= 0:
                raise ValueError(f"simulation_horizon_years must be positive, got {self.simulation_horizon_years}")

        # Validate withdrawal strategy (if configured)
        if self.withdrawal_strategy is not None:
            valid_withdrawal_strategies = [
                'constant_inflation_adjusted',
                'constant_percentage',
                'guyton_klinger',
                'vpw',
                'floor_ceiling',
                'rmd'
            ]
            if self.withdrawal_strategy not in valid_withdrawal_strategies:
                raise ValueError(f"withdrawal_strategy must be one of {valid_withdrawal_strategies}, got '{self.withdrawal_strategy}'")

        # Validate contribution parameters (if configured)
        if self.contribution_amount is not None and self.contribution_amount < 0:
            raise ValueError(f"contribution_amount cannot be negative, got {self.contribution_amount}")

        valid_frequencies = ['weekly', 'biweekly', 'monthly', 'annual']
        if self.contribution_frequency not in valid_frequencies:
            raise ValueError(f"contribution_frequency must be one of {valid_frequencies}, got '{self.contribution_frequency}'")

        if not (0 <= self.employer_match_rate <= 1):
            raise ValueError(f"employer_match_rate must be between 0 and 1, got {self.employer_match_rate}")

        if self.employer_match_cap is not None and self.employer_match_cap < 0:
            raise ValueError(f"employer_match_cap cannot be negative, got {self.employer_match_cap}")

        # Validate withdrawal parameters (if configured)
        if self.annual_withdrawal_amount is not None and self.annual_withdrawal_amount < 0:
            raise ValueError(f"annual_withdrawal_amount cannot be negative, got {self.annual_withdrawal_amount}")

        if self.withdrawal_percentage is not None and not (0 <= self.withdrawal_percentage <= 1):
            raise ValueError(f"withdrawal_percentage must be between 0 and 1, got {self.withdrawal_percentage}")

        if not (0 <= self.inflation_rate <= 0.20):
            raise ValueError(f"inflation_rate must be between 0 and 0.20, got {self.inflation_rate}")

        # Validate risk-free rate
        if not (0 <= self.risk_free_rate <= 1):
            raise ValueError(f"risk_free_rate must be between 0 and 1, got {self.risk_free_rate}")

        # Validate backtest parameters
        if self.min_history_periods < 1:
            raise ValueError(f"min_history_periods must be at least 1, got {self.min_history_periods}")

        if not self.use_expanding_window and self.rolling_window_periods < 1:
            raise ValueError(f"rolling_window_periods must be at least 1, got {self.rolling_window_periods}")

        # Validate ticker file exists
        if not Path(self.ticker_file).exists():
            # Just warn, don't fail - file might be created later
            import warnings
            warnings.warn(f"Ticker file not found: {self.ticker_file}")

    # ============================================================================
    # Serialization
    # ============================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SystemConfig':
        """Create SystemConfig from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, filepath: str) -> 'SystemConfig':
        """Load SystemConfig from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        # Remove comment keys (convention: keys starting with '_')
        config_dict = {k: v for k, v in config_dict.items() if not k.startswith('_')}
        return cls.from_dict(config_dict)

    def to_json(self, filepath: str, indent: int = 2) -> None:
        """Save SystemConfig to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)

    # ============================================================================
    # Utility Methods
    # ============================================================================

    def get_date_range(self) -> tuple[datetime, datetime]:
        """Get start and end dates as datetime objects."""
        return (
            datetime.strptime(self.start_date, '%Y-%m-%d'),
            datetime.strptime(self.end_date, '%Y-%m-%d')
        )

    def get_retirement_date(self) -> Optional[datetime]:
        """Get retirement date as datetime object."""
        if self.retirement_date is None:
            return None
        return datetime.strptime(self.retirement_date, '%Y-%m-%d')

    def get_simulation_horizon_date(self) -> Optional[datetime]:
        """
        Get simulation horizon date as datetime object.

        If simulation_horizon_years is specified, calculates the date from retirement_date.
        Otherwise returns simulation_horizon_date.
        """
        if self.retirement_date is None:
            return None

        if self.simulation_horizon_date is not None:
            return datetime.strptime(self.simulation_horizon_date, '%Y-%m-%d')

        if self.simulation_horizon_years is not None:
            retirement = datetime.strptime(self.retirement_date, '%Y-%m-%d')
            from dateutil.relativedelta import relativedelta
            return retirement + relativedelta(years=self.simulation_horizon_years)

        return None

    def get_accumulation_years(self) -> Optional[float]:
        """
        Get number of years in accumulation phase (end_date to retirement_date).

        Returns None if retirement_date not set.
        """
        if self.retirement_date is None:
            return None

        end = datetime.strptime(self.end_date, '%Y-%m-%d')
        retirement = datetime.strptime(self.retirement_date, '%Y-%m-%d')
        return (retirement - end).days / 365.25

    def get_decumulation_years(self) -> Optional[float]:
        """
        Get number of years in decumulation phase (retirement_date to simulation horizon).

        Returns None if retirement settings not configured.
        """
        retirement = self.get_retirement_date()
        horizon = self.get_simulation_horizon_date()

        if retirement is None or horizon is None:
            return None

        return (horizon - retirement).days / 365.25

    def has_retirement_simulation(self) -> bool:
        """Check if retirement simulation is configured."""
        return (self.retirement_date is not None and
                (self.simulation_horizon_date is not None or
                 self.simulation_horizon_years is not None))

    def has_contributions(self) -> bool:
        """Check if contributions are configured for accumulation phase."""
        return self.contribution_amount is not None and self.contribution_amount > 0

    def get_contribution_config(self) -> Optional[Dict[str, Any]]:
        """
        Get contribution configuration as a dictionary.

        Returns None if no contributions configured.
        """
        if not self.has_contributions():
            return None

        # Convert frequency to contributions per year
        freq_map = {'weekly': 52, 'biweekly': 26, 'monthly': 12, 'annual': 1}
        contributions_per_year = freq_map[self.contribution_frequency]

        return {
            'amount': self.contribution_amount,
            'frequency': self.contribution_frequency,
            'contributions_per_year': contributions_per_year,
            'annual_contribution': self.contribution_amount * contributions_per_year,
            'employer_match_rate': self.employer_match_rate,
            'employer_match_cap': self.employer_match_cap
        }

    def has_withdrawal_strategy(self) -> bool:
        """Check if a withdrawal/decumulation strategy is configured."""
        return self.withdrawal_strategy is not None

    def get_withdrawal_config(self) -> Optional[Dict[str, Any]]:
        """
        Get withdrawal/spending strategy configuration as a dictionary.

        Returns None if no withdrawal strategy configured.
        Returns all withdrawal-related settings for use in retirement simulation.
        """
        if not self.has_withdrawal_strategy():
            return None

        return {
            'strategy': self.withdrawal_strategy,
            'annual_amount': self.annual_withdrawal_amount,
            'percentage': self.withdrawal_percentage,
            'inflation_rate': self.inflation_rate,
            'strategy_params': self.withdrawal_strategy_params.copy()
        }

    def get_output_paths(self) -> Dict[str, Path]:
        """Get all output directory paths as Path objects."""
        return {
            'plots': Path(self.plots_directory),
            'results': Path(self.results_directory)
        }

    def create_output_directories(self) -> None:
        """Create output directories if they don't exist."""
        for path in self.get_output_paths().values():
            path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Default Configuration
# ============================================================================

DEFAULT_SYSTEM_CONFIG = SystemConfig()


# ============================================================================
# Convenience Functions
# ============================================================================

def load_system_config(filepath: str = None) -> SystemConfig:
    """
    Load system configuration from file or return default.

    Parameters:
    -----------
    filepath : str, optional
        Path to JSON config file. If None, returns default config.

    Returns:
    --------
    SystemConfig instance
    """
    if filepath is None:
        return DEFAULT_SYSTEM_CONFIG
    return SystemConfig.from_json(filepath)
