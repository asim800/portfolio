"""
Monte Carlo Simulation API

FastAPI application for portfolio lifecycle Monte Carlo simulations.

Run with:
    cd /path/to/port
    uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8001

API docs at: http://localhost:8001/docs
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

from .schemas import (
    MCConfigRequest,
    MCSimulationResponse,
    ConfigValidationRequest,
    ConfigValidationResponse,
    SweepParamsResponse,
    JobStatus,
    ParameterSweepRequest,
    ParameterSweepResponse,
    GridSweepRequest,
    GridSweepResponse,
)
from .services.simulation import simulation_service
from .services.jobs import job_manager


# Create FastAPI app
app = FastAPI(
    title="Monte Carlo Simulation API",
    description="REST API for portfolio lifecycle Monte Carlo simulations",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Health Check
# ==============================================================================

@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "mc-simulation-api"}


# ==============================================================================
# Simulation Endpoints
# ==============================================================================

@app.post(
    "/api/mc/simulate",
    response_model=MCSimulationResponse,
    tags=["Simulation"],
    summary="Run Monte Carlo simulation",
)
async def run_simulation(
    request: MCConfigRequest,
    background_tasks: BackgroundTasks,
) -> MCSimulationResponse:
    """
    Run Monte Carlo lifecycle simulation.

    Returns fan chart data for accumulation and decumulation phases,
    formatted for direct use with Recharts AreaChart component.

    **Sampling Methods:**
    - `parametric`: Sample from multivariate Gaussian using mean/covariance
    - `bootstrap`: Resample from historical Yahoo Finance data
    - `both`: Run both methods and return comparison (default)

    **Response:**
    - Percentile time series (5th, 25th, 50th, 75th, 95th) for fan charts
    - Final value distributions for histograms
    - Success rates (probability portfolio lasts through decumulation)
    """
    try:
        # Determine if we should skip bootstrap
        skip_bootstrap = request.sampling_method.value == "parametric"

        # For async mode, run in background
        if request.async_mode:
            job_id = job_manager.create_job()
            background_tasks.add_task(
                _run_simulation_async,
                job_id,
                request,
                skip_bootstrap,
            )
            return MCSimulationResponse(
                success=True,
                metadata=_empty_metadata(),
                accumulation={},
                accumulation_final={},
                decumulation={},
                decumulation_final={},
                success_rates={},
                percentiles_at_retirement={},
                percentiles_at_horizon={},
                job_id=job_id,
                status="running",
            )

        # Synchronous execution
        return simulation_service.run_simulation(request, skip_bootstrap)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _run_simulation_async(
    job_id: str,
    request: MCConfigRequest,
    skip_bootstrap: bool,
):
    """Background task to run simulation."""
    try:
        job_manager.start_job(job_id, "Running simulation...")
        result = simulation_service.run_simulation(request, skip_bootstrap)
        result.job_id = job_id
        job_manager.complete_job(job_id, result.model_dump())
    except Exception as e:
        job_manager.fail_job(job_id, str(e))


def _empty_metadata():
    """Create empty metadata for async response."""
    from .schemas.responses import SimulationMetadata
    return SimulationMetadata(
        num_simulations=0,
        accumulation_years=0,
        decumulation_years=0,
        accumulation_periods=0,
        decumulation_periods=0,
        periods_per_year=0,
        tickers=[],
        weights=[],
        sampling_methods_used=[],
        execution_time_ms=0,
        has_bootstrap=False,
    )


# ==============================================================================
# Parameter Sweep Endpoints
# ==============================================================================

@app.post(
    "/api/mc/sweep",
    response_model=ParameterSweepResponse,
    tags=["Sweep"],
    summary="Run single parameter sweep",
)
async def run_sweep(request: ParameterSweepRequest) -> ParameterSweepResponse:
    """
    Run Monte Carlo simulation sweep across a single parameter.

    Sweeps the specified parameter across either explicit values or a range,
    returning success rates and percentiles for each value.

    **Sweepable Parameters:**
    - `initial_portfolio_value`: Starting portfolio value
    - `annual_withdrawal_amount`: Annual withdrawal during decumulation
    - `simulation_horizon_years`: Years in decumulation phase
    - `retirement_date`: Date retirement begins
    - `contribution_amount`: Per-period contribution during accumulation
    - `inflation_rate`: Annual inflation rate

    **Response:**
    - Success rates for each parameter value
    - Percentile values (5th, 25th, 50th, 75th, 95th) at accumulation and decumulation end
    - Chart-ready data for visualization
    """
    try:
        # Determine parameter values
        if request.param_values:
            param_values = request.param_values
        else:
            # Generate values from range
            from src.run_mc import _generate_sweep_values
            param_values = _generate_sweep_values(
                request.param_name,
                request.param_range.start,
                request.param_range.end,
                request.param_range.step,
            )

        return simulation_service.run_parameter_sweep(
            request=request.base_config,
            param_name=request.param_name,
            param_values=param_values,
            skip_bootstrap=request.skip_bootstrap,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/mc/grid-sweep",
    response_model=GridSweepResponse,
    tags=["Sweep"],
    summary="Run 2D grid parameter sweep",
)
async def run_grid_sweep(request: GridSweepRequest) -> GridSweepResponse:
    """
    Run Monte Carlo simulation sweep across a 2D grid of two parameters.

    Sweeps two parameters simultaneously, returning success rates and
    percentiles for each combination. Results are returned as 2D matrices
    suitable for heatmap visualization.

    **Response:**
    - 2D matrices of success rates (rows = param1, columns = param2)
    - 2D matrices of median accumulation values
    - 2D matrices of median decumulation values
    - Flat list of results for table display
    """
    try:
        # Determine parameter values for param1
        if request.param1_values:
            param1_values = request.param1_values
        else:
            from src.run_mc import _generate_sweep_values
            param1_values = _generate_sweep_values(
                request.param1_name,
                request.param1_range.start,
                request.param1_range.end,
                request.param1_range.step,
            )

        # Determine parameter values for param2
        if request.param2_values:
            param2_values = request.param2_values
        else:
            from src.run_mc import _generate_sweep_values
            param2_values = _generate_sweep_values(
                request.param2_name,
                request.param2_range.start,
                request.param2_range.end,
                request.param2_range.step,
            )

        return simulation_service.run_grid_sweep(
            request=request.base_config,
            param1_name=request.param1_name,
            param1_values=param1_values,
            param2_name=request.param2_name,
            param2_values=param2_values,
            skip_bootstrap=request.skip_bootstrap,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# Configuration Endpoints
# ==============================================================================

@app.get(
    "/api/mc/config/sweep-params",
    response_model=SweepParamsResponse,
    tags=["Configuration"],
    summary="Get sweepable parameters",
)
async def get_sweep_params() -> SweepParamsResponse:
    """
    Get list of parameters that can be swept.

    Returns parameter names, types, descriptions, and default ranges.
    Use these for building sweep configuration UI.
    """
    params = simulation_service.get_sweep_params()
    return SweepParamsResponse(params=params)


@app.post(
    "/api/mc/config/validate",
    response_model=ConfigValidationResponse,
    tags=["Configuration"],
    summary="Validate simulation config",
)
async def validate_config(
    request: ConfigValidationRequest,
) -> ConfigValidationResponse:
    """
    Validate configuration before running simulation.

    Checks:
    - Date consistency and ordering
    - Weight sum equals 1.0
    - Historical data availability (if check_data_availability=true)
    - Parameter bounds
    """
    valid, errors, warnings, data_availability = simulation_service.validate_config(
        request.config,
        check_data=request.check_data_availability,
    )
    return ConfigValidationResponse(
        valid=valid,
        errors=errors,
        warnings=warnings,
        data_availability=data_availability,
    )


@app.get(
    "/api/mc/config/schema",
    tags=["Configuration"],
    summary="Get configuration JSON schema",
)
async def get_config_schema() -> Dict[str, Any]:
    """
    Get JSON schema for MC simulation configuration.

    Useful for building dynamic forms in the frontend.
    """
    return MCConfigRequest.model_json_schema()


# ==============================================================================
# Job Management Endpoints
# ==============================================================================

@app.get(
    "/api/mc/jobs/{job_id}",
    response_model=JobStatus,
    tags=["Jobs"],
    summary="Get job status",
)
async def get_job_status(job_id: str) -> JobStatus:
    """
    Get status of an async job.

    When status="completed", the result field contains the full response.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@app.post(
    "/api/mc/jobs/{job_id}/cancel",
    tags=["Jobs"],
    summary="Cancel running job",
)
async def cancel_job(job_id: str) -> Dict[str, str]:
    """Cancel a running or pending job."""
    success = job_manager.cancel_job(job_id)
    if not success:
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status '{job.status}'",
        )
    return {"message": f"Job {job_id} cancelled"}


@app.get(
    "/api/mc/jobs",
    tags=["Jobs"],
    summary="List all jobs",
)
async def list_jobs(status: str = None) -> Dict[str, Any]:
    """List all jobs, optionally filtered by status."""
    jobs = job_manager.list_jobs(status)
    return {
        "jobs": [j.model_dump() for j in jobs],
        "count": len(jobs),
    }


# ==============================================================================
# Startup/Shutdown Events
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on startup."""
    print("=" * 60)
    print("Monte Carlo Simulation API")
    print("=" * 60)
    print("Docs: http://localhost:8001/docs")
    print("Health: http://localhost:8001/health")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on shutdown."""
    # Cleanup old jobs
    removed = job_manager.cleanup_old_jobs(max_age_hours=1)
    if removed:
        print(f"Cleaned up {removed} old jobs")
