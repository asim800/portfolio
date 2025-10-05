#!/usr/bin/env python3
"""
Comparison/experiment configuration.

Defines which portfolios to compare and how to analyze them.
This is separate from portfolio configs to enable flexible experiment design.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path


@dataclass
class ComparisonConfig:
    """
    Configuration for a portfolio comparison experiment.

    This defines:
    - Which portfolios to include in the comparison
    - Which pairs to compare directly
    - Where to save comparison results
    """

    # ============================================================================
    # Experiment Identity
    # ============================================================================
    name: str                                    # Experiment name (e.g., "active_vs_passive")
    description: str = ""                        # Human-readable description

    # ============================================================================
    # Portfolio Selection
    # ============================================================================
    portfolios: List[str] = field(default_factory=list)  # List of portfolio config file paths
    # Example: ["portfolios/buy_and_hold.json", "portfolios/optimized_mv.json"]

    # ============================================================================
    # Comparison Pairs
    # ============================================================================
    comparison_pairs: List[Tuple[str, str]] = field(default_factory=list)
    # List of (portfolio_name_1, portfolio_name_2) tuples to compare
    # Example: [("buy_and_hold", "optimized_mv"), ("spy_benchmark", "optimized_mv")]

    # ============================================================================
    # Output Configuration
    # ============================================================================
    output_prefix: str = ""                      # Prefix for output files (defaults to experiment name)
    output_subdirectory: Optional[str] = None    # Subdirectory within plots/results dirs

    # ============================================================================
    # Analysis Options
    # ============================================================================
    include_statistical_tests: bool = True       # Run t-tests, etc. on return differences
    include_rolling_metrics: bool = False        # Calculate rolling Sharpe, volatility, etc.
    rolling_window_periods: int = 12             # Periods for rolling metrics

    # ============================================================================
    # Visualization Options
    # ============================================================================
    plot_cumulative_returns: bool = True         # Plot cumulative return comparison
    plot_drawdowns: bool = True                  # Plot drawdown comparison
    plot_weights: bool = False                   # Plot weight evolution (can be cluttered)
    plot_rebalancing_events: bool = True         # Mark rebalancing events on charts

    # ============================================================================
    # Validation
    # ============================================================================

    def __post_init__(self):
        """Validate configuration parameters."""
        if not self.name:
            raise ValueError("Comparison name cannot be empty")

        if not self.portfolios:
            raise ValueError("At least one portfolio must be specified")

        # Set output_prefix to name if not provided
        if not self.output_prefix:
            self.output_prefix = self.name

        # Validate portfolio file paths exist
        for portfolio_path in self.portfolios:
            if not Path(portfolio_path).exists():
                import warnings
                warnings.warn(f"Portfolio config file not found: {portfolio_path}")

        # Validate comparison pairs reference valid portfolios
        if self.comparison_pairs:
            # Extract portfolio names from file paths
            portfolio_names = set()
            for p in self.portfolios:
                # Will be validated when actually loaded
                portfolio_names.add(Path(p).stem)  # Filename without extension

            for pair in self.comparison_pairs:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError(f"Comparison pair must be a 2-element tuple, got {pair}")

        # Validate rolling window
        if self.include_rolling_metrics and self.rolling_window_periods < 1:
            raise ValueError(f"rolling_window_periods must be at least 1, got {self.rolling_window_periods}")

    # ============================================================================
    # Serialization
    # ============================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ComparisonConfig':
        """Create ComparisonConfig from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, filepath: str) -> 'ComparisonConfig':
        """Load ComparisonConfig from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_json(self, filepath: str, indent: int = 2) -> None:
        """Save ComparisonConfig to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)

    # ============================================================================
    # Utility Methods
    # ============================================================================

    def get_portfolio_paths(self, base_dir: str = "configs") -> List[Path]:
        """
        Get absolute paths to portfolio config files.

        Parameters:
        -----------
        base_dir : str
            Base directory containing config files

        Returns:
        --------
        List of Path objects to portfolio configs
        """
        base = Path(base_dir)
        return [base / p if not Path(p).is_absolute() else Path(p) for p in self.portfolios]

    def get_output_paths(self, plots_dir: str, results_dir: str) -> Dict[str, Path]:
        """
        Get output directory paths for this comparison.

        Parameters:
        -----------
        plots_dir : str
            Base plots directory from SystemConfig
        results_dir : str
            Base results directory from SystemConfig

        Returns:
        --------
        Dict with 'plots' and 'results' Path objects
        """
        plots_base = Path(plots_dir)
        results_base = Path(results_dir)

        if self.output_subdirectory:
            plots_base = plots_base / self.output_subdirectory
            results_base = results_base / self.output_subdirectory

        return {
            'plots': plots_base,
            'results': results_base
        }

    def create_output_directories(self, plots_dir: str, results_dir: str) -> None:
        """Create output directories for this comparison."""
        paths = self.get_output_paths(plots_dir, results_dir)
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def get_comparison_pair_names(self) -> List[Tuple[str, str]]:
        """Get comparison pairs as portfolio names (not file paths)."""
        return self.comparison_pairs


# ============================================================================
# Convenience Functions
# ============================================================================

def load_comparison_config(filepath: str) -> ComparisonConfig:
    """
    Load comparison configuration from JSON file.

    Parameters:
    -----------
    filepath : str
        Path to comparison JSON config file

    Returns:
    --------
    ComparisonConfig instance
    """
    return ComparisonConfig.from_json(filepath)


def create_simple_comparison(
    name: str,
    portfolios: List[str],
    pairs: List[Tuple[str, str]] = None
) -> ComparisonConfig:
    """
    Create a simple comparison config programmatically.

    Parameters:
    -----------
    name : str
        Comparison name
    portfolios : List[str]
        List of portfolio config file paths
    pairs : List[Tuple[str, str]], optional
        Comparison pairs. If None, compares all vs first portfolio.

    Returns:
    --------
    ComparisonConfig instance
    """
    if pairs is None:
        # Default: compare all portfolios against the first one
        portfolio_names = [Path(p).stem for p in portfolios]
        if len(portfolio_names) > 1:
            pairs = [(portfolio_names[0], name) for name in portfolio_names[1:]]
        else:
            pairs = []

    return ComparisonConfig(
        name=name,
        portfolios=portfolios,
        comparison_pairs=pairs
    )
