"""Transform numpy arrays to Recharts-compatible JSON format."""

from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from ..schemas.responses import (
    TimeSeriesPoint,
    FanChartData,
    DistributionData,
    DistributionBin,
    StyledTable,
    StyledTableRow,
    StyledTableCell,
    TableStyling,
    ColumnStyle,
    ColorThreshold,
)


class RechartsFormatter:
    """Transforms simulation output arrays to Recharts-compatible format."""

    @staticmethod
    def values_to_fan_chart(
        values: np.ndarray,
        start_date: datetime,
        periods_per_year: int,
        phase: str,
        sampling_method: str,
        sample_interval: int = 1,
    ) -> FanChartData:
        """
        Convert simulation values array to fan chart data.

        Args:
            values: Array of shape (num_simulations, num_periods)
            start_date: Start date of simulation
            periods_per_year: Number of periods per year
            phase: 'accumulation' or 'decumulation'
            sampling_method: 'parametric' or 'bootstrap'
            sample_interval: Sample every N periods (for large datasets)

        Returns:
            FanChartData with percentiles at each time point
        """
        num_sims, num_periods = values.shape

        # Calculate percentiles across simulations for each period
        percentiles = [5, 25, 50, 75, 95]
        pct_values = np.percentile(values, percentiles, axis=0)  # (5, num_periods)
        mean_values = np.mean(values, axis=0)

        # Generate dates
        days_per_period = 365 / periods_per_year

        # Sample at intervals for large datasets
        period_indices = range(0, num_periods, sample_interval)

        data_points: List[TimeSeriesPoint] = []
        for i, period in enumerate(period_indices):
            date = start_date + timedelta(days=int(period * days_per_period))
            data_points.append(
                TimeSeriesPoint(
                    period=period,
                    date=date.strftime("%Y-%m-%d"),
                    p5=float(pct_values[0, period]),
                    p25=float(pct_values[1, period]),
                    p50=float(pct_values[2, period]),
                    p75=float(pct_values[3, period]),
                    p95=float(pct_values[4, period]),
                    mean=float(mean_values[period]),
                )
            )

        return FanChartData(
            data=data_points,
            phase=phase,
            sampling_method=sampling_method,
        )

    @staticmethod
    def values_to_distribution(
        values: np.ndarray,
        num_bins: int = 20,
    ) -> DistributionData:
        """
        Convert final values to distribution data for histograms.

        Args:
            values: 1D array of final portfolio values
            num_bins: Number of histogram bins

        Returns:
            DistributionData with bins and statistics
        """
        # Ensure 1D
        if values.ndim > 1:
            values = values[:, -1]  # Take final period

        # Calculate histogram
        counts, bin_edges = np.histogram(values, bins=num_bins)
        total = len(values)

        bins = []
        for i in range(len(counts)):
            bins.append(
                DistributionBin(
                    bin_start=float(bin_edges[i]),
                    bin_end=float(bin_edges[i + 1]),
                    count=int(counts[i]),
                    percentage=float(counts[i] / total * 100),
                )
            )

        # Calculate percentiles
        percentile_values = np.percentile(values, [5, 25, 50, 75, 95])

        return DistributionData(
            bins=bins,
            mean=float(np.mean(values)),
            median=float(np.median(values)),
            std=float(np.std(values)),
            min=float(np.min(values)),
            max=float(np.max(values)),
            percentiles={
                "5": float(percentile_values[0]),
                "25": float(percentile_values[1]),
                "50": float(percentile_values[2]),
                "75": float(percentile_values[3]),
                "95": float(percentile_values[4]),
            },
        )

    @staticmethod
    def extract_percentiles(values: np.ndarray) -> Dict[str, float]:
        """
        Extract percentiles from final values.

        Args:
            values: 1D array or last column of 2D array

        Returns:
            Dict with percentile keys and values
        """
        if values.ndim > 1:
            values = values[:, -1]

        pct = np.percentile(values, [5, 25, 50, 75, 95])
        return {
            "5": float(pct[0]),
            "25": float(pct[1]),
            "50": float(pct[2]),
            "75": float(pct[3]),
            "95": float(pct[4]),
        }

    @staticmethod
    def calculate_success_rate(
        dec_values: np.ndarray,
        success_flags: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculate success rate from decumulation values.

        Args:
            dec_values: Decumulation values array (num_sims, num_periods)
            success_flags: Optional pre-computed success flags

        Returns:
            Success rate as float (0-1)
        """
        if success_flags is not None:
            return float(np.mean(success_flags))

        # Success = portfolio never went to zero (final value > 0)
        final_values = dec_values[:, -1] if dec_values.ndim > 1 else dec_values
        success = final_values > 0
        return float(np.mean(success))

    @staticmethod
    def format_sweep_chart_data(
        sweep_results: List[Dict],
        param_name: str,
    ) -> Dict:
        """
        Format sweep results for Recharts LineChart/AreaChart.

        Args:
            sweep_results: List of dicts with param_value and percentiles
            param_name: Name of swept parameter

        Returns:
            Dict with data arrays for Recharts
        """
        return {
            "accumulation": [
                {
                    param_name: r["param_value"],
                    "label": r["param_formatted"],
                    "p5": r["acc_p5"],
                    "p25": r["acc_p25"],
                    "p50": r["acc_p50"],
                    "p75": r["acc_p75"],
                    "p95": r["acc_p95"],
                }
                for r in sweep_results
            ],
            "decumulation": [
                {
                    param_name: r["param_value"],
                    "label": r["param_formatted"],
                    "p5": r["dec_p5"],
                    "p25": r["dec_p25"],
                    "p50": r["dec_p50"],
                    "p75": r["dec_p75"],
                    "p95": r["dec_p95"],
                }
                for r in sweep_results
            ],
            "success_rate": [
                {
                    param_name: r["param_value"],
                    "label": r["param_formatted"],
                    "success_rate": r["success_rate"] * 100,  # As percentage
                }
                for r in sweep_results
            ],
        }

    @staticmethod
    def create_heatmap_matrix(
        grid_results: List[Dict],
        param1_values: List,
        param2_values: List,
        value_key: str,
    ) -> List[List[float]]:
        """
        Create 2D matrix from grid sweep results for heatmap.

        Args:
            grid_results: Flat list of grid cell results
            param1_values: Values for first parameter (rows)
            param2_values: Values for second parameter (columns)
            value_key: Key to extract value from results

        Returns:
            2D list (matrix) of values
        """
        # Create lookup dict
        lookup = {
            (r["param1_value"], r["param2_value"]): r[value_key]
            for r in grid_results
        }

        # Build matrix
        matrix = []
        for p1 in param1_values:
            row = []
            for p2 in param2_values:
                row.append(lookup.get((p1, p2), 0.0))
            matrix.append(row)

        return matrix

    # =========================================================================
    # Table Styling Methods
    # =========================================================================

    @staticmethod
    def format_currency(value: float) -> str:
        """Format value as currency string."""
        if abs(value) >= 1_000_000:
            return f"${value/1_000_000:,.1f}M"
        elif abs(value) >= 1_000:
            return f"${value/1_000:,.0f}K"
        else:
            return f"${value:,.0f}"

    @staticmethod
    def format_percentage(value: float) -> str:
        """Format value as percentage string (expects 0-1 input)."""
        return f"{value * 100:.1f}%"

    @staticmethod
    def get_success_status(rate: float) -> str:
        """Get semantic status for success rate."""
        if rate >= 0.8:
            return "success"
        elif rate >= 0.6:
            return "warning"
        else:
            return "danger"

    @staticmethod
    def get_success_color(rate: float) -> str:
        """Get CSS color for success rate."""
        if rate >= 0.8:
            return "#4CAF50"  # Green
        elif rate >= 0.6:
            return "#FFC107"  # Yellow/Amber
        else:
            return "#F44336"  # Red

    @staticmethod
    def get_success_css_class(rate: float) -> str:
        """Get Tailwind CSS class for success rate."""
        if rate >= 0.8:
            return "bg-green-100 text-green-800"
        elif rate >= 0.6:
            return "bg-yellow-100 text-yellow-800"
        else:
            return "bg-red-100 text-red-800"

    @classmethod
    def create_simulation_summary_table(
        cls,
        results: Dict,
        sampling_methods: List[str],
    ) -> StyledTable:
        """
        Create styled summary table from simulation results.

        Args:
            results: Dict with success_rates, percentiles_at_retirement, etc.
            sampling_methods: List of methods used ['parametric', 'bootstrap']

        Returns:
            StyledTable with formatted and styled data
        """
        columns = [
            "method",
            "success_rate",
            "acc_p50",
            "dec_p50",
            "dec_p5",
            "dec_p95",
        ]
        column_labels = {
            "method": "Method",
            "success_rate": "Success Rate",
            "acc_p50": "Retirement (Median)",
            "dec_p50": "Final (Median)",
            "dec_p5": "Final (5th %)",
            "dec_p95": "Final (95th %)",
        }

        rows = []
        styled_rows = []

        for method in sampling_methods:
            success_rate = results.get("success_rates", {}).get(method, 0.0)
            acc_pct = results.get("percentiles_at_retirement", {}).get(method, {})
            dec_pct = results.get("percentiles_at_horizon", {}).get(method, {})

            # Raw data row
            row = {
                "method": method.capitalize(),
                "success_rate": success_rate,
                "acc_p50": acc_pct.get("50", 0),
                "dec_p50": dec_pct.get("50", 0),
                "dec_p5": dec_pct.get("5", 0),
                "dec_p95": dec_pct.get("95", 0),
            }
            rows.append(row)

            # Styled row
            styled_row = StyledTableRow(
                cells={
                    "method": StyledTableCell(
                        value=method,
                        display=method.capitalize(),
                    ),
                    "success_rate": StyledTableCell(
                        value=success_rate,
                        display=cls.format_percentage(success_rate),
                        color=cls.get_success_color(success_rate),
                        css_class=cls.get_success_css_class(success_rate),
                        status=cls.get_success_status(success_rate),
                    ),
                    "acc_p50": StyledTableCell(
                        value=acc_pct.get("50", 0),
                        display=cls.format_currency(acc_pct.get("50", 0)),
                    ),
                    "dec_p50": StyledTableCell(
                        value=dec_pct.get("50", 0),
                        display=cls.format_currency(dec_pct.get("50", 0)),
                    ),
                    "dec_p5": StyledTableCell(
                        value=dec_pct.get("5", 0),
                        display=cls.format_currency(dec_pct.get("5", 0)),
                    ),
                    "dec_p95": StyledTableCell(
                        value=dec_pct.get("95", 0),
                        display=cls.format_currency(dec_pct.get("95", 0)),
                    ),
                }
            )
            styled_rows.append(styled_row)

        return StyledTable(
            columns=columns,
            column_labels=column_labels,
            rows=rows,
            styled_rows=styled_rows,
            styling=TableStyling.default_success_rate_style(),
        )

    @classmethod
    def create_sweep_summary_table(
        cls,
        sweep_results: List[Dict],
        param_name: str,
    ) -> StyledTable:
        """
        Create styled table from parameter sweep results.

        Args:
            sweep_results: List of dicts with param_value, success_rate, etc.
            param_name: Name of the swept parameter

        Returns:
            StyledTable with formatted and styled data
        """
        columns = [
            "param_value",
            "success_rate",
            "acc_p50",
            "dec_p50",
        ]
        column_labels = {
            "param_value": param_name.replace("_", " ").title(),
            "success_rate": "Success Rate",
            "acc_p50": "Retirement (Median)",
            "dec_p50": "Final (Median)",
        }

        rows = []
        styled_rows = []

        for r in sweep_results:
            param_value = r.get("param_value", 0)
            param_formatted = r.get("param_formatted", str(param_value))
            success_rate = r.get("success_rate", 0.0)
            acc_p50 = r.get("acc_p50", 0)
            dec_p50 = r.get("dec_p50", 0)

            # Raw data row
            row = {
                "param_value": param_value,
                "success_rate": success_rate,
                "acc_p50": acc_p50,
                "dec_p50": dec_p50,
            }
            rows.append(row)

            # Styled row
            styled_row = StyledTableRow(
                cells={
                    "param_value": StyledTableCell(
                        value=param_value,
                        display=param_formatted,
                    ),
                    "success_rate": StyledTableCell(
                        value=success_rate,
                        display=cls.format_percentage(success_rate),
                        color=cls.get_success_color(success_rate),
                        css_class=cls.get_success_css_class(success_rate),
                        status=cls.get_success_status(success_rate),
                    ),
                    "acc_p50": StyledTableCell(
                        value=acc_p50,
                        display=cls.format_currency(acc_p50),
                    ),
                    "dec_p50": StyledTableCell(
                        value=dec_p50,
                        display=cls.format_currency(dec_p50),
                    ),
                }
            )
            styled_rows.append(styled_row)

        return StyledTable(
            columns=columns,
            column_labels=column_labels,
            rows=rows,
            styled_rows=styled_rows,
            styling=TableStyling.default_success_rate_style(),
        )
