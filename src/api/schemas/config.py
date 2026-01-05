"""Request schemas for Monte Carlo simulation API."""

from typing import List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class SimulationFrequency(str, Enum):
    """Frequency options for simulation periods."""
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class SamplingMethod(str, Enum):
    """Sampling method for MC simulation."""
    PARAMETRIC = "parametric"
    BOOTSTRAP = "bootstrap"
    BOTH = "both"


class WithdrawalStrategy(str, Enum):
    """Withdrawal strategy options."""
    CONSTANT_INFLATION_ADJUSTED = "constant_inflation_adjusted"
    CONSTANT_PERCENTAGE = "constant_percentage"
    GUYTON_KLINGER = "guyton_klinger"
    VPW = "vpw"
    FLOOR_CEILING = "floor_ceiling"
    RMD = "rmd"


class TickerWeight(BaseModel):
    """Single ticker with its portfolio weight."""
    symbol: str = Field(..., description="Ticker symbol (e.g., 'SPY')")
    weight: float = Field(..., ge=0, le=1, description="Portfolio weight (0-1)")


class MCConfigRequest(BaseModel):
    """Complete MC simulation configuration request."""

    # Required parameters
    initial_portfolio_value: float = Field(
        ..., gt=0, description="Initial portfolio value in dollars"
    )
    retirement_date: str = Field(
        ..., description="Retirement date (YYYY-MM-DD)"
    )
    simulation_horizon_years: int = Field(
        ..., gt=0, le=50, description="Years in decumulation phase"
    )
    tickers: List[TickerWeight] = Field(
        ..., min_length=1, description="List of tickers with weights"
    )

    # Historical data dates (for bootstrap sampling)
    start_date: str = Field(
        default="2005-01-01", description="Historical data start date"
    )
    end_date: str = Field(
        default="2025-01-01", description="Historical data end date"
    )
    mc_start_date: Optional[str] = Field(
        default=None, description="MC simulation start date (defaults to end_date)"
    )

    # Simulation settings
    num_simulations: int = Field(
        default=1000, ge=10, le=50000, description="Number of MC simulations"
    )
    simulation_frequency: SimulationFrequency = Field(
        default=SimulationFrequency.WEEKLY, description="Simulation period frequency"
    )
    sampling_method: SamplingMethod = Field(
        default=SamplingMethod.BOTH, description="Sampling method"
    )
    seed: Optional[int] = Field(
        default=42, description="Random seed for reproducibility"
    )

    # Accumulation phase settings
    contribution_amount: float = Field(
        default=0, ge=0, description="Per-period contribution amount"
    )
    contribution_frequency: SimulationFrequency = Field(
        default=SimulationFrequency.BIWEEKLY, description="Contribution frequency"
    )
    employer_match_rate: float = Field(
        default=0, ge=0, le=1, description="Employer match rate (0-1)"
    )
    employer_match_cap: Optional[float] = Field(
        default=None, ge=0, description="Annual employer match cap"
    )

    # Decumulation phase settings
    withdrawal_strategy: WithdrawalStrategy = Field(
        default=WithdrawalStrategy.CONSTANT_INFLATION_ADJUSTED,
        description="Withdrawal strategy"
    )
    annual_withdrawal_amount: float = Field(
        default=40000, ge=0, description="Initial annual withdrawal amount"
    )
    withdrawal_frequency: SimulationFrequency = Field(
        default=SimulationFrequency.BIWEEKLY, description="Withdrawal frequency"
    )
    inflation_rate: float = Field(
        default=0.03, ge=0, le=0.20, description="Annual inflation rate"
    )

    # Time-varying parameters (optional)
    simulated_mean_returns_file: Optional[str] = Field(
        default=None, description="Path to mean returns CSV"
    )
    simulated_cov_matrices_file: Optional[str] = Field(
        default=None, description="Path to covariance matrices file"
    )

    # Async execution
    async_mode: bool = Field(
        default=False, description="Run asynchronously (returns job_id)"
    )

    @field_validator('tickers')
    @classmethod
    def validate_weights_sum(cls, v: List[TickerWeight]) -> List[TickerWeight]:
        """Ensure weights sum to approximately 1.0."""
        total = sum(t.weight for t in v)
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")
        return v

    @field_validator('retirement_date', 'start_date', 'end_date', 'mc_start_date')
    @classmethod
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate date format is YYYY-MM-DD."""
        if v is not None:
            from datetime import datetime
            try:
                datetime.strptime(v, "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"Date must be YYYY-MM-DD format, got {v}")
        return v


class SweepRange(BaseModel):
    """Range specification for parameter sweep."""
    start: Any = Field(..., description="Start value")
    end: Any = Field(..., description="End value")
    step: Any = Field(..., description="Step size (or '1Y', '2Y' for dates)")


class ParameterSweepRequest(BaseModel):
    """Request for single parameter sweep."""
    base_config: MCConfigRequest = Field(..., description="Base simulation config")
    param_name: str = Field(..., description="Parameter to sweep")
    param_values: Optional[List[Any]] = Field(
        default=None, description="Explicit values to sweep"
    )
    param_range: Optional[SweepRange] = Field(
        default=None, description="Range for sweep"
    )
    skip_bootstrap: bool = Field(
        default=False, description="Skip bootstrap sampling"
    )

    @field_validator('param_range')
    @classmethod
    def validate_sweep_spec(cls, v, info):
        """Ensure either param_values or param_range is provided."""
        if v is None and info.data.get('param_values') is None:
            raise ValueError("Must provide either param_values or param_range")
        return v


class GridSweepRequest(BaseModel):
    """Request for 2D grid parameter sweep."""
    base_config: MCConfigRequest = Field(..., description="Base simulation config")

    # First parameter
    param1_name: str = Field(..., description="First parameter to sweep")
    param1_values: Optional[List[Any]] = Field(default=None)
    param1_range: Optional[SweepRange] = Field(default=None)

    # Second parameter
    param2_name: str = Field(..., description="Second parameter to sweep")
    param2_values: Optional[List[Any]] = Field(default=None)
    param2_range: Optional[SweepRange] = Field(default=None)

    skip_bootstrap: bool = Field(default=False)


class ConfigValidationRequest(BaseModel):
    """Request to validate a config before running."""
    config: MCConfigRequest
    check_data_availability: bool = Field(
        default=True, description="Check if historical data is available"
    )
