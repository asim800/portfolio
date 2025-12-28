# Monte Carlo simulation package
"""
Monte Carlo simulation and bootstrapping utilities.

Modules:
- path_generator: Core MC path generation engine
- bootstrap: Bootstrap sampling methods
- lifecycle: Accumulation/decumulation lifecycle simulation
"""

from .path_generator import MCPathGenerator
from .lifecycle import (
    run_accumulation_mc,
    run_decumulation_mc,
    calculate_success_rate,
    calculate_percentiles
)
from .bootstrap import (
    bootstrap_returns,
    parametric_sample,
    block_bootstrap,
    stationary_bootstrap
)

__all__ = [
    # Path generator
    'MCPathGenerator',
    # Lifecycle
    'run_accumulation_mc',
    'run_decumulation_mc',
    'calculate_success_rate',
    'calculate_percentiles',
    # Bootstrap
    'bootstrap_returns',
    'parametric_sample',
    'block_bootstrap',
    'stationary_bootstrap',
]
