#!/usr/bin/env python3
"""
Visualize time-varying covariance matrices for retirement scenarios.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from src.config import SystemConfig
from src.data import FinData


def safe_det(matrix):
    """Safely compute determinant, returning 0 for singular/problematic matrices."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            det = np.linalg.det(matrix)
            if np.isnan(det) or np.isinf(det):
                return 0.0
            return det
        except np.linalg.LinAlgError:
            return 0.0


def safe_det_ratio(det1, det2):
    """Safely compute percentage change between two determinants."""
    if det1 == 0 or det2 == 0:
        return "N/A"
    ratio = (det2 / det1 - 1) * 100
    if np.isnan(ratio) or np.isinf(ratio):
        return "N/A"
    return f'{ratio:+.1f}%'


def plot_heatmap(ax, matrix, labels, title, cmap='YlOrRd', vmin=None, vmax=None, fmt='.4f', cbar_label='Value'):
    """Plot a heatmap without seaborn."""
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f'{matrix[i, j]:{fmt}}',
                          ha="center", va="center", color="black", fontsize=9)

    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10)

    return im


def plot_covariance_evolution(cov_matrix_acc, cov_matrix_dec, tickers=None,
                              acc_label='Accumulation', dec_label='Decumulation',
                              output_path='../plots/test/covariance_evolution.png'):
    """
    Create comprehensive visualization comparing two covariance matrices.

    Parameters:
    -----------
    cov_matrix_acc : pd.DataFrame or np.ndarray
        Covariance matrix for accumulation phase (num_assets x num_assets)
        If DataFrame, column/index labels are used as ticker names
    cov_matrix_dec : pd.DataFrame or np.ndarray
        Covariance matrix for decumulation phase (num_assets x num_assets)
        If DataFrame, column/index labels are used as ticker names
    tickers : list, optional
        Asset ticker symbols (only needed if passing np.ndarray)
        If None and DataFrames are passed, labels are extracted from DataFrame
    acc_label : str
        Label for accumulation phase (default: 'Accumulation')
    dec_label : str
        Label for decumulation phase (default: 'Decumulation')
    output_path : str
        File path to save visualization

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Extract tickers and convert to numpy if DataFrames
    if isinstance(cov_matrix_acc, pd.DataFrame):
        if tickers is None:
            tickers = list(cov_matrix_acc.columns)
        cov_acc_array = cov_matrix_acc.values
    else:
        if tickers is None:
            raise ValueError("tickers must be provided when using np.ndarray")
        cov_acc_array = cov_matrix_acc

    if isinstance(cov_matrix_dec, pd.DataFrame):
        cov_dec_array = cov_matrix_dec.values
    else:
        cov_dec_array = cov_matrix_dec

    # Calculate correlation matrices from covariance
    def cov_to_corr(cov):
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        return corr

    corr_matrix_acc = cov_to_corr(cov_acc_array)
    corr_matrix_dec = cov_to_corr(cov_dec_array)

    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # ========================================================================
    # Row 1: Covariance Heatmaps
    # ========================================================================

    # Accumulation covariance
    ax1 = fig.add_subplot(gs[0, 0])
    plot_heatmap(ax1, cov_acc_array, tickers,
                f'{acc_label} Phase\nCovariance Matrix (Annual)',
                cmap='YlOrRd', fmt='.4f', cbar_label='Covariance')

    # Decumulation covariance
    ax2 = fig.add_subplot(gs[0, 1])
    plot_heatmap(ax2, cov_dec_array, tickers,
                f'{dec_label} Phase\nCovariance Matrix (Annual)',
                cmap='YlOrRd', fmt='.4f', cbar_label='Covariance')

    # Difference heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    cov_diff = cov_dec_array - cov_acc_array
    vmax_diff = max(abs(cov_diff.min()), abs(cov_diff.max()))
    plot_heatmap(ax3, cov_diff, tickers,
                f'Change at Retirement\n({dec_label} - {acc_label})',
                cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff,
                fmt='.4f', cbar_label='Difference')

    # ========================================================================
    # Row 2: Correlation Heatmaps
    # ========================================================================

    # Accumulation correlation
    ax4 = fig.add_subplot(gs[1, 0])
    plot_heatmap(ax4, corr_matrix_acc, tickers,
                f'{acc_label} Phase\nCorrelation Matrix',
                cmap='coolwarm', vmin=-1, vmax=1, fmt='.3f',
                cbar_label='Correlation')

    # Decumulation correlation
    ax5 = fig.add_subplot(gs[1, 1])
    plot_heatmap(ax5, corr_matrix_dec, tickers,
                f'{dec_label} Phase\nCorrelation Matrix',
                cmap='coolwarm', vmin=-1, vmax=1, fmt='.3f',
                cbar_label='Correlation')

    # Correlation change
    ax6 = fig.add_subplot(gs[1, 2])
    corr_diff = corr_matrix_dec - corr_matrix_acc
    plot_heatmap(ax6, corr_diff, tickers,
                f'Correlation Change\n({dec_label} - {acc_label})',
                cmap='RdBu_r', vmin=-0.5, vmax=0.5, fmt='.3f',
                cbar_label='Difference')

    # ========================================================================
    # Row 3: Volatility and Stats
    # ========================================================================

    # Volatility comparison
    ax7 = fig.add_subplot(gs[2, 0])
    vol_acc = np.sqrt(np.diag(cov_acc_array))
    vol_dec = np.sqrt(np.diag(cov_dec_array))

    x = np.arange(len(tickers))
    width = 0.35
    ax7.bar(x - width/2, vol_acc, width, label=acc_label, alpha=0.8, color='green')
    ax7.bar(x + width/2, vol_dec, width, label=dec_label, alpha=0.8, color='orange')
    ax7.set_xlabel('Asset', fontsize=11)
    ax7.set_ylabel('Volatility (Annual Std Dev)', fontsize=11)
    ax7.set_title('Volatility by Phase', fontsize=12, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(tickers)
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    ax7.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.1%}'))

    # Eigenvalue spectrum
    ax8 = fig.add_subplot(gs[2, 1])
    eigvals_acc = np.sort(np.linalg.eigvals(cov_acc_array))[::-1]
    eigvals_dec = np.sort(np.linalg.eigvals(cov_dec_array))[::-1]

    x_eig = np.arange(1, len(eigvals_acc) + 1)
    ax8.plot(x_eig, eigvals_acc, 'o-', linewidth=2, markersize=8,
             label=acc_label, color='green', alpha=0.8)
    ax8.plot(x_eig, eigvals_dec, 's-', linewidth=2, markersize=8,
             label=dec_label, color='orange', alpha=0.8)
    ax8.set_xlabel('Eigenvalue Index', fontsize=11)
    ax8.set_ylabel('Eigenvalue', fontsize=11)
    ax8.set_title('Covariance Eigenvalue Spectrum', fontsize=12, fontweight='bold')
    ax8.set_yscale('log')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_xticks(x_eig)

    # Summary statistics table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    # Compute determinants safely
    det_acc = safe_det(cov_acc_array)
    det_dec = safe_det(cov_dec_array)

    stats_data = [
        ['Metric', acc_label, dec_label, 'Change'],
        ['', '', '', ''],
        ['Determinant', f'{det_acc:.2e}',
         f'{det_dec:.2e}',
         safe_det_ratio(det_acc, det_dec)],
        ['Trace', f'{np.trace(cov_acc_array):.4f}',
         f'{np.trace(cov_dec_array):.4f}',
         f'{(np.trace(cov_dec_array)/np.trace(cov_acc_array) - 1)*100:+.1f}%'],
        ['Condition #', f'{np.linalg.cond(cov_acc_array):.1f}',
         f'{np.linalg.cond(cov_dec_array):.1f}',
         f'{(np.linalg.cond(cov_dec_array)/np.linalg.cond(cov_acc_array) - 1)*100:+.1f}%'],
        ['Max Eigenval', f'{eigvals_acc[0]:.4f}',
         f'{eigvals_dec[0]:.4f}',
         f'{(eigvals_dec[0]/eigvals_acc[0] - 1)*100:+.1f}%'],
        ['Min Eigenval', f'{eigvals_acc[-1]:.2e}',
         f'{eigvals_dec[-1]:.2e}',
         f'{(eigvals_dec[-1]/eigvals_acc[-1] - 1)*100:+.1f}%'],
        ['', '', '', ''],
        ['Avg Volatility', f'{vol_acc.mean():.2%}',
         f'{vol_dec.mean():.2%}',
         f'{(vol_dec.mean()/vol_acc.mean() - 1)*100:+.1f}%'],
        ['Avg Correlation', f'{(corr_matrix_acc.sum() - len(tickers))/(len(tickers)*(len(tickers)-1)):.3f}',
         f'{(corr_matrix_dec.sum() - len(tickers))/(len(tickers)*(len(tickers)-1)):.3f}',
         f'{((corr_matrix_dec.sum() - len(tickers))/(len(tickers)*(len(tickers)-1))) - ((corr_matrix_acc.sum() - len(tickers))/(len(tickers)*(len(tickers)-1))):+.3f}'],
    ]

    table = ax9.table(cellText=stats_data, cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style data rows
    for i in range(1, len(stats_data)):
        if i == 1 or i == 7:  # Empty rows
            continue
        for j in range(4):
            if j == 0:  # Metric name column
                table[(i, j)].set_facecolor('#E7E6E6')
                table[(i, j)].set_text_props(weight='bold')
            else:
                table[(i, j)].set_facecolor('#F2F2F2')

    ax9.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

    # ========================================================================
    # Save
    # ========================================================================

    plt.suptitle(f'Covariance Matrix Evolution: {acc_label} → {dec_label}',
                 fontsize=16, fontweight='bold', y=0.98)

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved covariance visualization: {output_path}")

    return fig


def visualize_covariance_evolution():
    """Create comprehensive visualization of covariance matrix evolution (legacy wrapper)."""

    print("=" * 80)
    print("COVARIANCE MATRIX VISUALIZATION")
    print("=" * 80)

    # Load data
    config = SystemConfig.from_json('../configs/test_simple_buyhold.json')
    tickers_df = pd.read_csv(config.ticker_file)
    tickers = tickers_df['Symbol'].tolist()

    fin_data = FinData(start_date=config.start_date, end_date=config.end_date)
    fin_data.fetch_ticker_data(tickers)
    returns_data = fin_data.get_returns_data(tickers)

    # Split into accumulation and decumulation regimes
    split_idx = len(returns_data) // 2
    returns_data_acc = returns_data.iloc[:split_idx]
    returns_data_dec = returns_data.iloc[split_idx:]

    # Calculate annual covariance matrices
    cov_matrix_acc = returns_data_acc.cov().values * 252
    cov_matrix_dec = returns_data_dec.cov().values * 252

    print(f"\nData summary:")
    print(f"  Accumulation period: {len(returns_data_acc)} days")
    print(f"  Decumulation period: {len(returns_data_dec)} days")
    print(f"  Assets: {tickers}")

    # Call the reusable function
    fig = plot_covariance_evolution(cov_matrix_acc, cov_matrix_dec, tickers)

    return fig


if __name__ == '__main__':
    fig = visualize_covariance_evolution()

    print("\n" + "=" * 80)
    print("Visualization complete!")
    print("=" * 80)
    print("\nThe 3x3 grid shows:")
    print("  Row 1: Covariance matrices (accumulation, decumulation, difference)")
    print("  Row 2: Correlation matrices (accumulation, decumulation, difference)")
    print("  Row 3: Volatility bars, eigenvalue spectrum, summary stats")
    print("=" * 80)
