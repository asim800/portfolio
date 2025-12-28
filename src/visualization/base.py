#!/usr/bin/env python3
"""
Base visualization utilities.

Common chart utilities and styling for all visualization modules.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Tuple


# Default color palette
DEFAULT_COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf',  # Cyan
]


def set_style(style: str = 'seaborn-v0_8-whitegrid'):
    """
    Set matplotlib style.

    Parameters:
    -----------
    style : str
        Matplotlib style name
    """
    try:
        plt.style.use(style)
    except OSError:
        plt.style.use('seaborn-whitegrid')


def format_currency(value: float, prefix: str = '$') -> str:
    """
    Format value as currency string.

    Parameters:
    -----------
    value : float
        Value to format
    prefix : str
        Currency prefix

    Returns:
    --------
    str: Formatted currency string
    """
    if abs(value) >= 1e9:
        return f'{prefix}{value/1e9:.1f}B'
    elif abs(value) >= 1e6:
        return f'{prefix}{value/1e6:.1f}M'
    elif abs(value) >= 1e3:
        return f'{prefix}{value/1e3:.0f}K'
    else:
        return f'{prefix}{value:.0f}'


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format value as percentage string.

    Parameters:
    -----------
    value : float
        Value to format (0.05 = 5%)
    decimals : int
        Number of decimal places

    Returns:
    --------
    str: Formatted percentage string
    """
    return f'{value*100:.{decimals}f}%'


def create_figure(nrows: int = 1, ncols: int = 1,
                 figsize: Tuple[float, float] = None,
                 **kwargs) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a figure with subplots.

    Parameters:
    -----------
    nrows : int
        Number of rows
    ncols : int
        Number of columns
    figsize : tuple, optional
        Figure size (width, height)
    **kwargs : dict
        Additional arguments to plt.subplots

    Returns:
    --------
    tuple: (fig, axes)
    """
    if figsize is None:
        figsize = (6 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes


def add_grid(ax: plt.Axes, alpha: float = 0.3):
    """Add grid to axes."""
    ax.grid(True, alpha=alpha)


def add_legend(ax: plt.Axes, loc: str = 'best', **kwargs):
    """Add legend to axes."""
    ax.legend(loc=loc, **kwargs)


def save_figure(fig: plt.Figure, path: str, dpi: int = 300, **kwargs):
    """
    Save figure to file.

    Parameters:
    -----------
    fig : plt.Figure
        Figure to save
    path : str
        Output path
    dpi : int
        Resolution
    **kwargs : dict
        Additional arguments to savefig
    """
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', **kwargs)
    print(f"Saved: {path}")


def get_color_palette(n_colors: int) -> List[str]:
    """
    Get a color palette with specified number of colors.

    Parameters:
    -----------
    n_colors : int
        Number of colors needed

    Returns:
    --------
    list: List of color hex codes
    """
    if n_colors <= len(DEFAULT_COLORS):
        return DEFAULT_COLORS[:n_colors]

    # Generate more colors using colormap
    cmap = plt.cm.get_cmap('tab20')
    return [cmap(i / n_colors) for i in range(n_colors)]
