"""Response schemas for Monte Carlo simulation API."""

from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field


# ==============================================================================
# Table Styling Schemas
# ==============================================================================

class ColorThreshold(BaseModel):
    """
    Threshold-based coloring for table cells.

    Frontend applies colors based on: value >= threshold[i] → colors[i]
    Thresholds should be in descending order.
    """
    thresholds: List[float] = Field(
        ..., description="Threshold values in descending order"
    )
    colors: List[str] = Field(
        ..., description="CSS color values (hex or named), one more than thresholds"
    )
    css_classes: List[str] = Field(
        default_factory=list,
        description="Optional Tailwind/CSS classes instead of colors"
    )


class ColumnStyle(BaseModel):
    """Styling configuration for a table column."""
    column: str = Field(..., description="Column name/key")
    format: str = Field(
        default="auto",
        description="Format type: 'currency', 'percentage', 'number', 'auto'"
    )
    color_scale: Optional[ColorThreshold] = Field(
        default=None, description="Color thresholds for this column"
    )
    align: Literal["left", "center", "right"] = Field(default="right")


class TableStyling(BaseModel):
    """
    Complete table styling configuration.

    Example usage in frontend:
    ```tsx
    const getColor = (value: number, colorScale: ColorThreshold) => {
      for (let i = 0; i < colorScale.thresholds.length; i++) {
        if (value >= colorScale.thresholds[i]) {
          return colorScale.css_classes[i] || colorScale.colors[i];
        }
      }
      return colorScale.css_classes.slice(-1)[0] || colorScale.colors.slice(-1)[0];
    };
    ```
    """
    columns: List[ColumnStyle] = Field(
        default_factory=list, description="Per-column styling"
    )

    @classmethod
    def default_success_rate_style(cls) -> "TableStyling":
        """Default styling for success rate tables."""
        return cls(
            columns=[
                ColumnStyle(
                    column="success_rate",
                    format="percentage",
                    align="center",
                    color_scale=ColorThreshold(
                        thresholds=[0.8, 0.6],  # >= 80% green, >= 60% yellow, < 60% red
                        colors=["#4CAF50", "#FFC107", "#F44336"],
                        css_classes=[
                            "bg-green-100 text-green-800",
                            "bg-yellow-100 text-yellow-800",
                            "bg-red-100 text-red-800"
                        ]
                    )
                ),
                ColumnStyle(
                    column="acc_median",
                    format="currency",
                    align="right"
                ),
                ColumnStyle(
                    column="dec_median",
                    format="currency",
                    align="right"
                ),
            ]
        )


class StyledTableCell(BaseModel):
    """
    A table cell with pre-computed styling.

    Use this when you want the API to compute styles server-side.
    """
    value: Any = Field(..., description="Raw value")
    display: str = Field(..., description="Formatted display string")
    css_class: Optional[str] = Field(
        default=None, description="CSS class to apply"
    )
    color: Optional[str] = Field(
        default=None, description="CSS color value"
    )
    status: Optional[Literal["success", "warning", "danger", "neutral"]] = Field(
        default=None, description="Semantic status for styling"
    )


class StyledTableRow(BaseModel):
    """A complete table row with styled cells."""
    cells: Dict[str, StyledTableCell] = Field(
        ..., description="Cell data keyed by column name"
    )
    row_class: Optional[str] = Field(
        default=None, description="CSS class for the entire row"
    )


class StyledTable(BaseModel):
    """
    Complete styled table response.

    Includes both raw data and styling metadata so frontend can either:
    1. Use pre-computed styles (styled_rows)
    2. Apply its own styling using the styling config
    """
    columns: List[str] = Field(..., description="Column names in order")
    column_labels: Dict[str, str] = Field(
        default_factory=dict, description="Display labels for columns"
    )
    rows: List[Dict[str, Any]] = Field(
        ..., description="Raw data rows"
    )
    styled_rows: Optional[List[StyledTableRow]] = Field(
        default=None, description="Pre-styled rows (optional)"
    )
    styling: TableStyling = Field(
        default_factory=TableStyling, description="Styling configuration"
    )


# ==============================================================================
# Time Series Schemas
# ==============================================================================

class TimeSeriesPoint(BaseModel):
    """Single data point for time series charts (Recharts-compatible)."""
    period: int = Field(..., description="Period index (0-based)")
    date: str = Field(..., description="Date string (YYYY-MM-DD)")
    p5: float = Field(..., description="5th percentile value")
    p25: float = Field(..., description="25th percentile value")
    p50: float = Field(..., description="50th percentile (median)")
    p75: float = Field(..., description="75th percentile value")
    p95: float = Field(..., description="95th percentile value")
    mean: Optional[float] = Field(default=None, description="Mean value")


class FanChartData(BaseModel):
    """Fan chart data for one phase and sampling method."""
    data: List[TimeSeriesPoint] = Field(..., description="Time series data points")
    phase: Literal["accumulation", "decumulation"] = Field(..., description="Simulation phase")
    sampling_method: Literal["parametric", "bootstrap"] = Field(..., description="Sampling method used")


class DistributionBin(BaseModel):
    """Single bin for histogram distribution."""
    bin_start: float
    bin_end: float
    count: int
    percentage: float


class DistributionData(BaseModel):
    """Distribution statistics for final values."""
    bins: List[DistributionBin] = Field(default_factory=list)
    mean: float
    median: float
    std: float
    min_val: float = Field(..., alias="min")
    max_val: float = Field(..., alias="max")
    percentiles: Dict[str, float] = Field(
        ..., description="Percentiles as {'5': value, '25': value, ...}"
    )

    class Config:
        populate_by_name = True


class SimulationMetadata(BaseModel):
    """Metadata about the simulation run."""
    num_simulations: int
    accumulation_years: float
    decumulation_years: float
    accumulation_periods: int
    decumulation_periods: int
    periods_per_year: int
    tickers: List[str]
    weights: List[float]
    sampling_methods_used: List[str]
    execution_time_ms: int
    has_bootstrap: bool


class MCSimulationResponse(BaseModel):
    """Response for single MC simulation."""
    success: bool = Field(..., description="Whether simulation completed successfully")
    metadata: SimulationMetadata

    # Accumulation phase results (keyed by sampling method)
    accumulation: Dict[str, FanChartData] = Field(
        ..., description="Fan chart data by sampling method"
    )
    accumulation_final: Dict[str, DistributionData] = Field(
        ..., description="Final value distributions"
    )

    # Decumulation phase results
    decumulation: Dict[str, FanChartData] = Field(
        ..., description="Fan chart data by sampling method"
    )
    decumulation_final: Dict[str, DistributionData] = Field(
        ..., description="Final value distributions"
    )

    # Summary statistics
    success_rates: Dict[str, float] = Field(
        ..., description="Success rates by sampling method (0-1)"
    )
    percentiles_at_retirement: Dict[str, Dict[str, float]] = Field(
        ..., description="Percentiles at retirement date"
    )
    percentiles_at_horizon: Dict[str, Dict[str, float]] = Field(
        ..., description="Percentiles at end of simulation"
    )

    # Styled summary table for display
    summary_table: Optional[StyledTable] = Field(
        default=None, description="Styled summary table for frontend display"
    )

    # Async job info
    job_id: Optional[str] = Field(default=None, description="Async job ID if applicable")
    status: Literal["completed", "running", "failed", "cancelled"] = Field(
        default="completed", description="Job status"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")


class SweepResultPoint(BaseModel):
    """Single point in parameter sweep results."""
    param_value: Any = Field(..., description="Swept parameter value")
    param_formatted: str = Field(..., description="Formatted display string")

    # Accumulation results
    acc_p5: float
    acc_p25: float
    acc_p50: float
    acc_p75: float
    acc_p95: float

    # Decumulation results
    dec_p5: float
    dec_p25: float
    dec_p50: float
    dec_p75: float
    dec_p95: float

    # Success rate
    success_rate: float


class ParameterSweepResponse(BaseModel):
    """Response for parameter sweep."""
    success: bool

    # Sweep metadata
    param_name: str
    param_type: Literal["currency", "percentage", "numeric", "date"]
    param_description: str
    num_values: int

    # Results
    parametric_results: List[SweepResultPoint]
    bootstrap_results: Optional[List[SweepResultPoint]] = None

    # Pre-formatted chart data for Recharts
    chart_data: Dict[str, Any] = Field(
        ..., description="Pre-formatted data for charts"
    )

    # Summary table data (raw)
    summary_table: List[Dict[str, Any]]

    # Styled summary table with colors
    styled_summary_table: Optional[StyledTable] = Field(
        default=None, description="Styled table with color-coded cells"
    )

    # Execution info
    execution_time_ms: int
    job_id: Optional[str] = None
    status: Literal["completed", "running", "failed", "cancelled"] = "completed"
    error: Optional[str] = None


class GridCell(BaseModel):
    """Single cell in 2D grid sweep results."""
    param1_value: Any
    param2_value: Any
    param1_formatted: str
    param2_formatted: str
    success_rate: float
    acc_median: float
    dec_median: float


class GridSweepResponse(BaseModel):
    """Response for 2D grid sweep."""
    success: bool

    # Grid metadata
    param1_name: str
    param1_type: Literal["currency", "percentage", "numeric", "date"]
    param1_values: List[Any]
    param1_labels: List[str]

    param2_name: str
    param2_type: Literal["currency", "percentage", "numeric", "date"]
    param2_values: List[Any]
    param2_labels: List[str]

    # Results as matrices (for heatmaps)
    parametric_success_matrix: List[List[float]]
    parametric_acc_matrix: List[List[float]]
    parametric_dec_matrix: List[List[float]]

    bootstrap_success_matrix: Optional[List[List[float]]] = None
    bootstrap_acc_matrix: Optional[List[List[float]]] = None
    bootstrap_dec_matrix: Optional[List[List[float]]] = None

    # Flat results for table display
    results: List[GridCell]

    # Execution info
    execution_time_ms: int
    job_id: Optional[str] = None
    status: Literal["completed", "running", "failed", "cancelled"] = "completed"
    error: Optional[str] = None


class SweepParamInfo(BaseModel):
    """Information about a sweepable parameter."""
    name: str
    type: Literal["currency", "percentage", "numeric", "date"]
    description: str
    default_range: Dict[str, Any] = Field(
        ..., description="Default range: {start, end, step}"
    )
    current_value: Optional[Any] = None


class SweepParamsResponse(BaseModel):
    """Response for sweepable parameters query."""
    params: List[SweepParamInfo]


class ConfigValidationResponse(BaseModel):
    """Response for config validation."""
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    data_availability: Optional[Dict[str, Any]] = Field(
        default=None, description="Historical data availability info"
    )
