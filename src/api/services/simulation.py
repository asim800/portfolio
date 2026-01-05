"""Simulation service wrapping run_mc.py functions."""

import os
import sys
import time
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SystemConfig
from src.run_mc import (
    run_mc_simulation as _run_mc_simulation,
    load_bootstrap_data,
    run_parameter_sweep as _run_parameter_sweep,
    run_parameter_grid_sweep as _run_parameter_grid_sweep,
    SWEEP_PARAMS,
    _format_param_value,
    _get_param_type,
    _generate_sweep_values,
)
from ..schemas.config import MCConfigRequest, TickerWeight
from ..schemas.responses import (
    MCSimulationResponse,
    SimulationMetadata,
    FanChartData,
    DistributionData,
    SweepParamInfo,
    SweepResultPoint,
    ParameterSweepResponse,
    GridCell,
    GridSweepResponse,
)
from ..utils.recharts_formatter import RechartsFormatter


class SimulationService:
    """Service for running Monte Carlo simulations."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or PROJECT_ROOT

    def run_simulation(
        self,
        request: MCConfigRequest,
        skip_bootstrap: bool = False,
    ) -> MCSimulationResponse:
        """
        Run Monte Carlo simulation from API request.

        Args:
            request: MCConfigRequest with simulation parameters
            skip_bootstrap: If True, skip bootstrap sampling

        Returns:
            MCSimulationResponse with Recharts-compatible data
        """
        start_time = time.time()

        # Convert request to SystemConfig
        config = self._request_to_config(request)

        # Load historical data for bootstrap (if not skipped)
        if skip_bootstrap or request.sampling_method.value == "parametric":
            historical_returns = None
        else:
            historical_returns, _ = load_bootstrap_data(config)

        # Run simulation
        result = _run_mc_simulation(
            config=config,
            historical_returns=historical_returns,
            num_simulations=request.num_simulations,
            seed=request.seed or 42,
            verbose=False,
        )

        # Get lifecycle dates and periods info
        mc_start_date = datetime.strptime(
            request.mc_start_date or request.end_date, "%Y-%m-%d"
        )
        periods_per_year = config.frequency_to_periods_per_year(
            request.simulation_frequency.value
        )
        acc_years = config.get_accumulation_years()
        dec_years = config.get_decumulation_years()

        # Extract tickers and weights
        tickers = [t.symbol for t in request.tickers]
        weights = [t.weight for t in request.tickers]

        # Determine which sampling methods were used
        has_bootstrap = result["has_bootstrap"]
        sampling_methods = ["parametric"]
        if has_bootstrap:
            sampling_methods.append("bootstrap")

        # Build metadata
        metadata = SimulationMetadata(
            num_simulations=request.num_simulations,
            accumulation_years=float(acc_years),
            decumulation_years=float(dec_years),
            accumulation_periods=result["acc_values_param"].shape[1],
            decumulation_periods=result["dec_values_param"].shape[1],
            periods_per_year=periods_per_year,
            tickers=tickers,
            weights=weights,
            sampling_methods_used=sampling_methods,
            execution_time_ms=int((time.time() - start_time) * 1000),
            has_bootstrap=has_bootstrap,
        )

        # Determine sample interval for large datasets (keep ~100 points max)
        acc_periods = result["acc_values_param"].shape[1]
        dec_periods = result["dec_values_param"].shape[1]
        acc_interval = max(1, acc_periods // 100)
        dec_interval = max(1, dec_periods // 100)

        # Calculate retirement date for decumulation start
        from datetime import timedelta
        retirement_date = mc_start_date + timedelta(days=int(acc_years * 365))

        # Build accumulation fan chart data
        accumulation = {}
        accumulation["parametric"] = RechartsFormatter.values_to_fan_chart(
            values=result["acc_values_param"],
            start_date=mc_start_date,
            periods_per_year=periods_per_year,
            phase="accumulation",
            sampling_method="parametric",
            sample_interval=acc_interval,
        )
        if has_bootstrap:
            accumulation["bootstrap"] = RechartsFormatter.values_to_fan_chart(
                values=result["acc_values_boot"],
                start_date=mc_start_date,
                periods_per_year=periods_per_year,
                phase="accumulation",
                sampling_method="bootstrap",
                sample_interval=acc_interval,
            )

        # Build decumulation fan chart data
        decumulation = {}
        decumulation["parametric"] = RechartsFormatter.values_to_fan_chart(
            values=result["dec_values_param"],
            start_date=retirement_date,
            periods_per_year=periods_per_year,
            phase="decumulation",
            sampling_method="parametric",
            sample_interval=dec_interval,
        )
        if has_bootstrap:
            decumulation["bootstrap"] = RechartsFormatter.values_to_fan_chart(
                values=result["dec_values_boot"],
                start_date=retirement_date,
                periods_per_year=periods_per_year,
                phase="decumulation",
                sampling_method="bootstrap",
                sample_interval=dec_interval,
            )

        # Build distribution data for final values
        accumulation_final = {}
        accumulation_final["parametric"] = RechartsFormatter.values_to_distribution(
            result["acc_values_param"][:, -1]
        )
        if has_bootstrap:
            accumulation_final["bootstrap"] = RechartsFormatter.values_to_distribution(
                result["acc_values_boot"][:, -1]
            )

        decumulation_final = {}
        decumulation_final["parametric"] = RechartsFormatter.values_to_distribution(
            result["dec_values_param"][:, -1]
        )
        if has_bootstrap:
            decumulation_final["bootstrap"] = RechartsFormatter.values_to_distribution(
                result["dec_values_boot"][:, -1]
            )

        # Success rates
        success_rates = {"parametric": float(result["dec_success_param"])}
        if has_bootstrap:
            success_rates["bootstrap"] = float(result["dec_success_boot"])

        # Percentiles at key dates
        percentiles_at_retirement = {
            "parametric": {
                str(k): float(v) for k, v in result["acc_percentiles_param"].items()
            }
        }
        percentiles_at_horizon = {
            "parametric": {
                str(k): float(v) for k, v in result["dec_percentiles_param"].items()
            }
        }
        if has_bootstrap:
            percentiles_at_retirement["bootstrap"] = {
                str(k): float(v) for k, v in result["acc_percentiles_boot"].items()
            }
            percentiles_at_horizon["bootstrap"] = {
                str(k): float(v) for k, v in result["dec_percentiles_boot"].items()
            }

        # Create styled summary table
        summary_table = RechartsFormatter.create_simulation_summary_table(
            results={
                "success_rates": success_rates,
                "percentiles_at_retirement": percentiles_at_retirement,
                "percentiles_at_horizon": percentiles_at_horizon,
            },
            sampling_methods=sampling_methods,
        )

        return MCSimulationResponse(
            success=True,
            metadata=metadata,
            accumulation=accumulation,
            accumulation_final=accumulation_final,
            decumulation=decumulation,
            decumulation_final=decumulation_final,
            success_rates=success_rates,
            percentiles_at_retirement=percentiles_at_retirement,
            percentiles_at_horizon=percentiles_at_horizon,
            summary_table=summary_table,
            status="completed",
        )

    def get_sweep_params(self) -> List[SweepParamInfo]:
        """Get list of sweepable parameters with their info."""
        params = []
        for name, info in SWEEP_PARAMS.items():
            default_range = info["default_range"]
            params.append(
                SweepParamInfo(
                    name=name,
                    type=info["type"],
                    description=info["description"],
                    default_range={
                        "start": default_range[0],
                        "end": default_range[1],
                        "step": default_range[2],
                    },
                )
            )
        return params

    def validate_config(
        self, request: MCConfigRequest, check_data: bool = True
    ) -> Tuple[bool, List[str], List[str], Optional[Dict]]:
        """
        Validate configuration before running simulation.

        Returns:
            Tuple of (valid, errors, warnings, data_availability)
        """
        errors = []
        warnings = []
        data_availability = None

        # Check date ordering
        try:
            start = datetime.strptime(request.start_date, "%Y-%m-%d")
            end = datetime.strptime(request.end_date, "%Y-%m-%d")
            retirement = datetime.strptime(request.retirement_date, "%Y-%m-%d")

            if start >= end:
                errors.append("start_date must be before end_date")
            if end >= retirement:
                warnings.append("end_date should be before retirement_date for proper bootstrap")

            mc_start = request.mc_start_date
            if mc_start:
                mc_start_dt = datetime.strptime(mc_start, "%Y-%m-%d")
                if mc_start_dt < end:
                    warnings.append("mc_start_date is before end_date, simulation may overlap with historical data")

        except ValueError as e:
            errors.append(f"Invalid date format: {e}")

        # Check weights sum
        weight_sum = sum(t.weight for t in request.tickers)
        if not (0.99 <= weight_sum <= 1.01):
            errors.append(f"Ticker weights must sum to 1.0, got {weight_sum:.4f}")

        # Check simulation count
        if request.num_simulations > 10000:
            warnings.append("Large simulation count (>10000) may take a long time")

        # Check data availability if requested
        if check_data and not errors:
            try:
                config = self._request_to_config(request)
                historical_returns, tickers = load_bootstrap_data(config)
                data_availability = {
                    "bootstrap_available": historical_returns is not None,
                    "periods": len(historical_returns) if historical_returns is not None else 0,
                    "tickers_found": tickers or [],
                }
            except Exception as e:
                data_availability = {
                    "bootstrap_available": False,
                    "error": str(e),
                }

        return len(errors) == 0, errors, warnings, data_availability

    def run_parameter_sweep(
        self,
        request: MCConfigRequest,
        param_name: str,
        param_values: List[Any],
        skip_bootstrap: bool = False,
    ) -> ParameterSweepResponse:
        """
        Run Monte Carlo simulation sweep across parameter values.

        Args:
            request: Base MCConfigRequest
            param_name: Parameter to sweep
            param_values: List of values to test
            skip_bootstrap: If True, skip bootstrap sampling

        Returns:
            ParameterSweepResponse with results for each value
        """
        import tempfile
        start_time = time.time()

        # Create temp config file from request
        config = self._request_to_config(request)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config.to_json(f.name)
            config_path = f.name

        # Run sweep
        df = _run_parameter_sweep(
            config_path=config_path,
            param_name=param_name,
            param_values=param_values,
            num_simulations=request.num_simulations,
            seed=request.seed or 42,
            verbose=False,
            skip_bootstrap=skip_bootstrap,
        )

        # Check if bootstrap was used
        has_bootstrap = "dec_success_boot" in df.columns

        # Build results
        parametric_results = []
        bootstrap_results = [] if has_bootstrap else None

        for _, row in df.iterrows():
            param_value = row[param_name]
            param_formatted = _format_param_value(param_value, param_name)

            parametric_results.append(SweepResultPoint(
                param_value=param_value,
                param_formatted=param_formatted,
                acc_p5=float(row["acc_p5_param"]),
                acc_p25=float(row["acc_p25_param"]),
                acc_p50=float(row["acc_p50_param"]),
                acc_p75=float(row["acc_p75_param"]),
                acc_p95=float(row["acc_p95_param"]),
                dec_p5=float(row["dec_p5_param"]),
                dec_p25=float(row["dec_p25_param"]),
                dec_p50=float(row["dec_p50_param"]),
                dec_p75=float(row["dec_p75_param"]),
                dec_p95=float(row["dec_p95_param"]),
                success_rate=float(row["dec_success_param"]),
            ))

            if has_bootstrap:
                bootstrap_results.append(SweepResultPoint(
                    param_value=param_value,
                    param_formatted=param_formatted,
                    acc_p5=float(row["acc_p5_boot"]),
                    acc_p25=float(row["acc_p25_boot"]),
                    acc_p50=float(row["acc_p50_boot"]),
                    acc_p75=float(row["acc_p75_boot"]),
                    acc_p95=float(row["acc_p95_boot"]),
                    dec_p5=float(row["dec_p5_boot"]),
                    dec_p25=float(row["dec_p25_boot"]),
                    dec_p50=float(row["dec_p50_boot"]),
                    dec_p75=float(row["dec_p75_boot"]),
                    dec_p95=float(row["dec_p95_boot"]),
                    success_rate=float(row["dec_success_boot"]),
                ))

        # Build chart data
        chart_data = RechartsFormatter.format_sweep_chart_data(
            [r.model_dump() for r in parametric_results],
            param_name,
        )

        # Build summary table
        summary_table = [
            {
                "param_value": r.param_value,
                "param_formatted": r.param_formatted,
                "success_rate": r.success_rate,
                "acc_p50": r.acc_p50,
                "dec_p50": r.dec_p50,
            }
            for r in parametric_results
        ]

        # Styled table
        styled_summary_table = RechartsFormatter.create_sweep_summary_table(
            [r.model_dump() for r in parametric_results],
            param_name,
        )

        return ParameterSweepResponse(
            success=True,
            param_name=param_name,
            param_type=_get_param_type(param_name),
            param_description=SWEEP_PARAMS.get(param_name, {}).get("description", ""),
            num_values=len(param_values),
            parametric_results=parametric_results,
            bootstrap_results=bootstrap_results,
            chart_data=chart_data,
            summary_table=summary_table,
            styled_summary_table=styled_summary_table,
            execution_time_ms=int((time.time() - start_time) * 1000),
        )

    def run_grid_sweep(
        self,
        request: MCConfigRequest,
        param1_name: str,
        param1_values: List[Any],
        param2_name: str,
        param2_values: List[Any],
        skip_bootstrap: bool = False,
    ) -> GridSweepResponse:
        """
        Run Monte Carlo simulation sweep across 2D grid of parameter values.

        Args:
            request: Base MCConfigRequest
            param1_name: First parameter to sweep (rows)
            param1_values: Values for first parameter
            param2_name: Second parameter to sweep (columns)
            param2_values: Values for second parameter
            skip_bootstrap: If True, skip bootstrap sampling

        Returns:
            GridSweepResponse with 2D matrices of results
        """
        import tempfile
        start_time = time.time()

        # Create temp config file from request
        config = self._request_to_config(request)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config.to_json(f.name)
            config_path = f.name

        # Run grid sweep
        df = _run_parameter_grid_sweep(
            config_path=config_path,
            param1_name=param1_name,
            param1_values=param1_values,
            param2_name=param2_name,
            param2_values=param2_values,
            num_simulations=request.num_simulations,
            seed=request.seed or 42,
            verbose=False,
            skip_bootstrap=skip_bootstrap,
        )

        # Check if bootstrap was used
        has_bootstrap = "dec_success_boot" in df.columns

        # Build flat results and matrices
        results = []
        param_success_matrix = []
        param_acc_matrix = []
        param_dec_matrix = []
        boot_success_matrix = [] if has_bootstrap else None
        boot_acc_matrix = [] if has_bootstrap else None
        boot_dec_matrix = [] if has_bootstrap else None

        # Build labels
        param1_labels = [_format_param_value(v, param1_name) for v in param1_values]
        param2_labels = [_format_param_value(v, param2_name) for v in param2_values]

        for v1 in param1_values:
            success_row = []
            acc_row = []
            dec_row = []
            boot_success_row = [] if has_bootstrap else None
            boot_acc_row = [] if has_bootstrap else None
            boot_dec_row = [] if has_bootstrap else None

            for v2 in param2_values:
                row = df[(df[param1_name] == v1) & (df[param2_name] == v2)].iloc[0]

                results.append(GridCell(
                    param1_value=v1,
                    param2_value=v2,
                    param1_formatted=_format_param_value(v1, param1_name),
                    param2_formatted=_format_param_value(v2, param2_name),
                    success_rate=float(row["dec_success_param"]),
                    acc_median=float(row["acc_p50_param"]),
                    dec_median=float(row["dec_p50_param"]),
                ))

                success_row.append(float(row["dec_success_param"]))
                acc_row.append(float(row["acc_p50_param"]))
                dec_row.append(float(row["dec_p50_param"]))

                if has_bootstrap:
                    boot_success_row.append(float(row["dec_success_boot"]))
                    boot_acc_row.append(float(row["acc_p50_boot"]))
                    boot_dec_row.append(float(row["dec_p50_boot"]))

            param_success_matrix.append(success_row)
            param_acc_matrix.append(acc_row)
            param_dec_matrix.append(dec_row)

            if has_bootstrap:
                boot_success_matrix.append(boot_success_row)
                boot_acc_matrix.append(boot_acc_row)
                boot_dec_matrix.append(boot_dec_row)

        return GridSweepResponse(
            success=True,
            param1_name=param1_name,
            param1_type=_get_param_type(param1_name),
            param1_values=param1_values,
            param1_labels=param1_labels,
            param2_name=param2_name,
            param2_type=_get_param_type(param2_name),
            param2_values=param2_values,
            param2_labels=param2_labels,
            parametric_success_matrix=param_success_matrix,
            parametric_acc_matrix=param_acc_matrix,
            parametric_dec_matrix=param_dec_matrix,
            bootstrap_success_matrix=boot_success_matrix,
            bootstrap_acc_matrix=boot_acc_matrix,
            bootstrap_dec_matrix=boot_dec_matrix,
            results=results,
            execution_time_ms=int((time.time() - start_time) * 1000),
        )

    def _request_to_config(self, request: MCConfigRequest) -> SystemConfig:
        """Convert API request to SystemConfig."""
        # Create a temporary ticker file content
        ticker_data = {
            "Symbol": [t.symbol for t in request.tickers],
            "Weight": [t.weight for t in request.tickers],
        }

        # Write to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            pd.DataFrame(ticker_data).to_csv(f, index=False)
            ticker_file = f.name

        # Build config dict
        config_dict = {
            "start_date": request.start_date,
            "end_date": request.end_date,
            "ticker_file": ticker_file,
            "initial_portfolio_value": request.initial_portfolio_value,
            "retirement_date": request.retirement_date,
            "simulation_horizon_years": request.simulation_horizon_years,
            "simulation_frequency": request.simulation_frequency.value,
            "num_mc_simulations": request.num_simulations,
            "contribution_amount": request.contribution_amount,
            "contribution_frequency": request.contribution_frequency.value,
            "employer_match_rate": request.employer_match_rate,
            "withdrawal_strategy": request.withdrawal_strategy.value,
            "annual_withdrawal_amount": request.annual_withdrawal_amount,
            "withdrawal_frequency": request.withdrawal_frequency.value,
            "inflation_rate": request.inflation_rate,
            # Use default simulated params files
            "simulated_mean_returns_file": str(
                self.project_root / "configs/data/simulated_mean_returns.csv"
            ),
            "simulated_cov_matrices_file": str(
                self.project_root / "configs/data/simulated_cov_matrices.txt"
            ),
        }

        if request.mc_start_date:
            config_dict["mc_start_date"] = request.mc_start_date
        if request.employer_match_cap:
            config_dict["employer_match_cap"] = request.employer_match_cap
        if request.simulated_mean_returns_file:
            config_dict["simulated_mean_returns_file"] = request.simulated_mean_returns_file
        if request.simulated_cov_matrices_file:
            config_dict["simulated_cov_matrices_file"] = request.simulated_cov_matrices_file

        return SystemConfig.from_dict(config_dict)


# Singleton instance
simulation_service = SimulationService()
