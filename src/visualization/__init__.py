# Visualization package
"""
Plotting and charting utilities for portfolio analysis.

Modules:
- base: Common chart utilities and styling
- backtest: Rebalancing and backtest visualization
- montecarlo: Fan charts, spaghetti plots, lifecycle visualization
- comparison: Strategy comparison grids
- heatmaps: Covariance and correlation heatmaps
"""

from .base import (
    set_style,
    format_currency,
    format_percentage,
    create_figure,
    add_grid,
    add_legend,
    save_figure,
    get_color_palette,
    DEFAULT_COLORS,
)

__all__ = [
    'set_style',
    'format_currency',
    'format_percentage',
    'create_figure',
    'add_grid',
    'add_legend',
    'save_figure',
    'get_color_palette',
    'DEFAULT_COLORS',
]
