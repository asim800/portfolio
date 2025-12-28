"""
Visualization Functions for Portfolio Strategies

This module provides comprehensive visualization functions for comparing
portfolio strategies, including wealth charts, risk-return scatter plots,
allocation charts, and correlation heatmaps.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

from metrics import calculate_max_drawdown, calculate_sharpe, calculate_portfolio_metrics


def create_strategy_visualization(strategies, returns_data, colors=None, title='Strategy Comparison',
                                 baselines=None, detailed_views=None, add_legend_panel=False):
    """
    Generic visualization function for comparing strategies

    Args:
        strategies: Dict of {name: strategy_instance}
        returns_data: Array of returns data
        colors: Dict mapping strategy names to colors (auto-generated if None)
        title: Title for the visualization
        baselines: Optional dict of baseline strategies to include
        detailed_views: Optional list of (strategy_name, chart_type) for detailed panels
        add_legend_panel: DEPRECATED - legend panel removed; full names now shown in individual chart legends

    Returns:
        matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Generate colors if not provided
    if colors is None:
        all_names = list((baselines or {}).keys()) + list(strategies.keys())
        color_cycle = plt.cm.tab20(np.linspace(0, 1, len(all_names)))
        colors = {name: color for name, color in zip(all_names, color_cycle)}

    all_strategies = {**(baselines or {}), **strategies}

    # Panel 1: Cumulative Wealth (Log Scale)
    ax1 = fig.add_subplot(gs[0, :2])

    # Plot baselines with distinct styles
    if baselines:
        for i, (name, strategy) in enumerate(baselines.items()):
            if hasattr(strategy, 'wealth_history') and strategy.wealth_history:
                linestyle = '-' if i == 0 else '--'
                lw = 3 if i == 0 else 2
                alpha = 0.9 if i == 0 else 0.7
                line_color = colors.get(name, 'gray') if i != 0 else 'black'
                ax1.semilogy(strategy.wealth_history, linestyle=linestyle, linewidth=lw,
                           label=name, alpha=alpha, color=line_color)

    # Plot main strategies
    for name, strategy in strategies.items():
        if hasattr(strategy, 'wealth_history') and strategy.wealth_history:
            # Use full portfolio name instead of just number
            display_name = name.split('. ', 1)[1] if '. ' in name else name
            ax1.semilogy(strategy.wealth_history, linewidth=2.5,
                        label=display_name,
                        color=colors[name], alpha=0.8)

    ax1.set_xlabel('Period', fontsize=11)
    ax1.set_ylabel('Cumulative Wealth (log scale)', fontsize=11)
    ax1.set_title(f'Cumulative Wealth: {title}', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3, which='both')

    # Panel 2: Risk-Return Scatter
    ax2 = fig.add_subplot(gs[0, 2])

    for name, strategy in all_strategies.items():
        if hasattr(strategy, 'wealth_history') and strategy.wealth_history:
            wealth = strategy.wealth_history[-1]
            dd = calculate_max_drawdown(strategy.wealth_history)
            marker = '*' if name.startswith('[*]') else 'o'
            size = 300 if name.startswith('[*]') else 200
            ax2.scatter([dd], [wealth], s=size, c=[colors.get(name, 'gray')],
                       alpha=0.7, edgecolors='black', linewidth=2, marker=marker)
            # Use abbreviated portfolio name for scatter plot annotations
            display_name = name.split('. ', 1)[1] if '. ' in name else name
            # Truncate if too long for annotation
            display_name = display_name[:20] + '...' if len(display_name) > 20 else display_name
            ax2.annotate(display_name, (dd, wealth), fontsize=7, ha='right')

    ax2.set_xlabel('Maximum Drawdown (%)', fontsize=11)
    ax2.set_ylabel('Final Wealth ($)', fontsize=11)
    ax2.set_title('Risk-Return Tradeoff', fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Sharpe Ratios
    ax3 = fig.add_subplot(gs[1, :2])

    names = []
    sharpes = []
    bar_colors = []

    for name, strategy in strategies.items():
        if hasattr(strategy, 'wealth_history') and strategy.wealth_history:
            # Use full portfolio name for bar charts
            display_name = name.split('. ', 1)[1] if '. ' in name else name
            names.append(display_name)
            sharpes.append(calculate_sharpe(strategy.wealth_history))
            bar_colors.append(colors[name])

    # Sort by Sharpe
    sorted_idx = np.argsort(sharpes)[::-1]
    names = [names[i] for i in sorted_idx]
    sharpes = [sharpes[i] for i in sorted_idx]
    bar_colors = [bar_colors[i] for i in sorted_idx]

    bars = ax3.barh(names, sharpes, color=bar_colors, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Sharpe Ratio', fontsize=11)
    ax3.set_title('Sharpe Ratio Rankings', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')

    for i, (bar, val) in enumerate(zip(bars, sharpes)):
        ax3.text(val, i, f' {val:.2f}', va='center', fontsize=9, fontweight='bold')

    # Panel 4: Max Drawdowns
    ax4 = fig.add_subplot(gs[1, 2])

    names_dd = []
    dds = []
    dd_colors = []

    for name, strategy in strategies.items():
        if hasattr(strategy, 'wealth_history') and strategy.wealth_history:
            # Use full portfolio name for bar charts
            display_name = name.split('. ', 1)[1] if '. ' in name else name
            names_dd.append(display_name)
            dds.append(calculate_max_drawdown(strategy.wealth_history))
            dd_colors.append(colors[name])

    # Sort by DD (ascending = better)
    sorted_idx = np.argsort(dds)
    names_dd = [names_dd[i] for i in sorted_idx]
    dds = [dds[i] for i in sorted_idx]
    dd_colors = [dd_colors[i] for i in sorted_idx]

    bars = ax4.barh(names_dd, dds, color=dd_colors, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Maximum Drawdown (%)', fontsize=11)
    ax4.set_title('Drawdown Rankings (Lower = Better)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    for i, (bar, val) in enumerate(zip(bars, dds)):
        ax4.text(val, i, f' {val:.1f}%', va='center', fontsize=9, fontweight='bold')

    # Panel 5: Return Distribution
    ax5 = fig.add_subplot(gs[2, 0])

    # Calculate raw returns for all strategies
    all_returns = []
    labels = []
    colors_list = []

    # Add baselines first
    if baselines:
        for name, strategy in baselines.items():
            if hasattr(strategy, 'wealth_history') and strategy.wealth_history and len(strategy.wealth_history) > 1:
                wealth = np.array(strategy.wealth_history)
                # Calculate raw returns (not log returns)
                returns = np.diff(wealth) / wealth[:-1] * 100  # Percentage returns
                all_returns.append(returns)
                labels.append(name)
                colors_list.append(colors.get(name, 'gray'))

    # Add main strategies
    for name, strategy in strategies.items():
        if hasattr(strategy, 'wealth_history') and strategy.wealth_history and len(strategy.wealth_history) > 1:
            wealth = np.array(strategy.wealth_history)
            # Calculate raw returns (not log returns)
            returns = np.diff(wealth) / wealth[:-1] * 100  # Percentage returns
            all_returns.append(returns)
            display_name = name.split('. ', 1)[1] if '. ' in name else name
            labels.append(display_name)
            colors_list.append(colors.get(name, 'gray'))

    # Create violin plots
    if all_returns:
        positions = np.arange(len(all_returns))
        parts = ax5.violinplot(all_returns, positions=positions, widths=0.7,
                               showmeans=True, showmedians=True)

        # Color the violin plots
        for idx, pc in enumerate(parts['bodies']):
            if idx < len(colors_list):
                pc.set_facecolor(colors_list[idx])
                pc.set_alpha(0.6)

        # Style the violin plot elements
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in parts:
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(1)

        ax5.set_xticks(positions)
        ax5.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax5.set_ylabel('Returns (%)', fontsize=11)
        ax5.set_title('Return Distribution', fontsize=12, fontweight='bold')
        ax5.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax5.grid(True, alpha=0.3, axis='y')

    # Panel 6: Allocations over time
    ax6 = fig.add_subplot(gs[2, 1:])

    # Plot baselines
    if baselines:
        for name, strategy in baselines.items():
            if hasattr(strategy, 'allocation_history') and strategy.allocation_history:
                style = 'k-' if name.startswith('[*]') else '--'
                lw = 3 if name.startswith('[*]') else 2
                # Use full baseline name
                ax6.plot(strategy.allocation_history, style, linewidth=lw,
                        label=name, alpha=0.7)

    # Plot main strategies
    for name, strategy in strategies.items():
        if hasattr(strategy, 'allocation_history') and strategy.allocation_history:
            # Handle different allocation formats
            allocs = strategy.allocation_history
            if isinstance(allocs[0], (list, np.ndarray)) and len(allocs[0]) > 1:
                # Multi-asset: plot stock allocation (index 1)
                allocs = [a[1] if len(a) > 1 else a[0] for a in allocs]

            # Use full portfolio name
            display_name = name.split('. ', 1)[1] if '. ' in name else name
            ax6.plot(allocs, linewidth=2,
                    label=display_name,
                    color=colors[name], alpha=0.7)

    ax6.set_xlabel('Period', fontsize=11)
    ax6.set_ylabel('Stock Allocation', fontsize=11)
    ax6.set_title('Dynamic Allocation', fontsize=12, fontweight='bold')
    ax6.legend(loc='best', fontsize=8, ncol=2)
    ax6.grid(True, alpha=0.3)

    # Panel 7: Asset Return QQ Plots (or detailed_views if provided)
    if detailed_views:
        for idx, (panel_info) in enumerate(detailed_views[:3]):
            ax = fig.add_subplot(gs[3, idx])
            _create_detailed_panel(ax, panel_info, strategies, colors)
    else:
        # Show QQ plots for underlying asset returns
        # returns_data is typically shaped as (n_periods, n_assets)
        if returns_data is not None and len(returns_data) > 0:
            # Determine asset names based on number of columns
            n_assets = returns_data.shape[1] if len(returns_data.shape) > 1 else 1

            # Standard asset names based on simulation functions
            asset_names_map = {
                2: ['Bonds', 'Stocks'],
                3: ['Bonds', 'Stocks', 'Tail Hedge'],
                4: ['Bonds', 'Stocks', 'Tail Hedge', 'Commodities'],
                5: ['Bonds', 'Stocks', 'Tail Hedge', 'Commodities', 'Gold']
            }

            asset_names = asset_names_map.get(n_assets, [f'Asset {i+1}' for i in range(n_assets)])

            # Asset colors
            asset_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:n_assets]

            # Collect returns for each asset
            # returns_data contains price multipliers (e.g., 1.02 = +2%, 0.95 = -5%)
            # Convert to percentage returns: (multiplier - 1) * 100
            asset_returns = []
            for i in range(n_assets):
                if len(returns_data.shape) > 1:
                    # Multi-asset case
                    multipliers = returns_data[:, i]
                    asset_ret = (multipliers - 1.0) * 100  # Convert to percentage returns
                else:
                    # Single asset case
                    multipliers = returns_data
                    asset_ret = (multipliers - 1.0) * 100
                asset_returns.append(asset_ret)

            # Create QQ plots - one subplot per asset
            for idx, (asset_ret, asset_name, color) in enumerate(zip(asset_returns, asset_names, asset_colors)):
                ax_qq = fig.add_subplot(gs[3, idx] if n_assets <= 3 else gs[3, idx % 3])
                if idx >= 3 and n_assets > 3:
                    # Skip if we have more than 3 assets (only show first 3)
                    break

                # Calculate theoretical quantiles and sample quantiles
                sorted_data = np.sort(asset_ret)
                n = len(sorted_data)
                theoretical_quantiles = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)

                # Plot QQ points
                ax_qq.scatter(theoretical_quantiles, sorted_data, c=color, alpha=0.6,
                             edgecolors='black', linewidth=0.5, s=30)

                # Add reference line (45-degree line through Q1 and Q3)
                q1_data, q3_data = np.percentile(asset_ret, [25, 75])
                q1_theo, q3_theo = stats.norm.ppf([0.25, 0.75])
                slope = (q3_data - q1_data) / (q3_theo - q1_theo)
                intercept = q1_data - slope * q1_theo
                x_line = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
                ax_qq.plot(x_line, slope * x_line + intercept, 'r--', linewidth=2, label='Normal ref.')

                # Add statistics annotation (with error handling for edge cases)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    try:
                        skew = stats.skew(asset_ret)
                        kurt = stats.kurtosis(asset_ret)
                        if np.isnan(skew) or np.isnan(kurt):
                            skew, kurt = 0.0, 0.0
                    except (RuntimeWarning, FloatingPointError):
                        skew, kurt = 0.0, 0.0
                ax_qq.annotate(f'Skew: {skew:.2f}\nKurt: {kurt:.2f}',
                              xy=(0.05, 0.95), xycoords='axes fraction',
                              fontsize=9, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                ax_qq.set_xlabel('Theoretical Quantiles', fontsize=10)
                ax_qq.set_ylabel('Sample Quantiles (%)', fontsize=10)
                ax_qq.set_title(f'{asset_name} QQ Plot', fontsize=12, fontweight='bold')
                ax_qq.legend(loc='lower right', fontsize=8)
                ax_qq.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

    return fig


def _create_detailed_panel(ax, panel_info, strategies, colors):
    """Helper to create detailed panels for specific strategies"""
    strategy_name, chart_type = panel_info

    strategy = strategies.get(strategy_name)
    if not strategy:
        ax.axis('off')
        return

    if chart_type == 'hedge_allocation':
        if hasattr(strategy, 'hedge_allocation_history'):
            periods = range(len(strategy.hedge_allocation_history))
            hedge_allocs = [h * 100 for h in strategy.hedge_allocation_history]
            ax.plot(periods, hedge_allocs, linewidth=2, color=colors.get(strategy_name, 'red'))
            ax.set_xlabel('Period', fontsize=11)
            ax.set_ylabel('Hedge Allocation (%)', fontsize=11)
            ax.set_title(f'{strategy_name.split(".")[0]}: Hedge Allocation',
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

    elif chart_type == 'allocation_pie':
        if hasattr(strategy, 'allocation_history') and strategy.allocation_history:
            final_alloc = strategy.allocation_history[-1]
            if isinstance(final_alloc, (list, np.ndarray)) and len(final_alloc) > 2:
                labels = ['Stocks', 'Bonds', 'Tail Hedge', 'Gold'][:len(final_alloc)]
                allocs_pct = [a * 100 for a in final_alloc]
                non_zero = [(l, a) for l, a in zip(labels, allocs_pct) if a > 1]

                if non_zero:
                    labels_nz, allocs_nz = zip(*non_zero)
                    ax.pie(allocs_nz, labels=labels_nz, autopct='%1.1f%%', startangle=90)
                    ax.set_title(f'{strategy_name.split(".")[0]}: Final Allocation',
                                fontsize=12, fontweight='bold')


def create_ten_more_visualization(strategies, returns_data):
    """Create comprehensive visualization for 10 new approaches"""

    colors = {
        '6. Time-Varying Ergodicity': 'purple',
        '7. Multi-Timeframe Hierarchical': 'red',
        '8. Multi-Safe-Haven UP': 'green',
        '9. Kelly-Criterion UP': 'blue',
        '10. Asymmetric Loss-Averse': 'orange',
        '11. Sequential Threshold': 'brown',
        '12. Volatility-Scaled UP': 'pink',
        '13. Momentum-Enhanced': 'darkgreen',
        '14. Three-Level Hierarchical': 'gold',
        '15. Dynamic Granularity': 'cyan',
    }

    # Use legend panel instead of detailed views
    return create_strategy_visualization(
        strategies, returns_data,
        colors=colors,
        title='10 More Universal Portfolio + Safe Haven Approaches',
        add_legend_panel=True
    )


def create_comparison_visualization(strategies, baseline_wealth, buy_hold, returns_data):
    """Create comprehensive comparison visualization"""
    from metrics import get_default_colors

    # Use full color palette for all portfolios
    colors = get_default_colors()

    # Create pseudo-strategy for baseline
    class PseudoStrategy:
        def __init__(self, wealth_history):
            self.wealth_history = wealth_history

    standard_up_pseudo = PseudoStrategy(baseline_wealth)

    # Baselines dictionary
    baselines = {
        '[*] Buy & Hold 60/40 (Benchmark)': buy_hold,
        'Standard UP (2-asset)': standard_up_pseudo
    }

    # Use legend panel instead of detailed views
    return create_strategy_visualization(
        strategies, returns_data,
        colors=colors,
        title='Universal Portfolio + Safe Haven vs Buy & Hold 60/40 Benchmark',
        baselines=baselines,
        add_legend_panel=True
    )


# ============================================================================
# Correlation and Covariance Heatmaps
# ============================================================================

def create_asset_correlation_heatmap(returns_data, title="Asset Correlation and Covariance Analysis"):
    """
    Create correlation and covariance heatmaps for asset returns

    Args:
        returns_data: DataFrame or numpy array of returns (price multipliers)
                     Columns should be asset returns
        title: Overall title for the figure

    Returns:
        matplotlib.figure.Figure
    """
    # Convert to DataFrame if needed
    if isinstance(returns_data, np.ndarray):
        # Infer asset names from number of columns
        n_assets = returns_data.shape[1] if len(returns_data.shape) > 1 else 1
        asset_names_map = {
            2: ['Bonds', 'Stocks'],
            3: ['Bonds', 'Stocks', 'Tail Hedge'],
            4: ['Bonds', 'Stocks', 'Tail Hedge', 'Commodities'],
            5: ['Bonds', 'Stocks', 'Tail Hedge', 'Commodities', 'Gold']
        }
        asset_names = asset_names_map.get(n_assets, [f'Asset {i+1}' for i in range(n_assets)])
        returns_df = pd.DataFrame(returns_data, columns=asset_names)
    else:
        returns_df = returns_data.copy()
        # Clean up column names if they exist
        returns_df.columns = [col.replace('_', ' ').title() for col in returns_df.columns]

    # Convert price multipliers to percentage returns
    # multipliers like 1.02, 0.95 -> percentage returns like 2.0, -5.0
    returns_pct = (returns_df - 1.0) * 100

    # Calculate correlation and covariance matrices
    correlation_matrix = returns_pct.corr()
    covariance_matrix = returns_pct.cov()

    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Correlation Heatmap
    im1 = ax1.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Set ticks and labels
    ax1.set_xticks(np.arange(len(correlation_matrix.columns)))
    ax1.set_yticks(np.arange(len(correlation_matrix.columns)))
    ax1.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right', fontsize=10)
    ax1.set_yticklabels(correlation_matrix.columns, fontsize=10)

    # Add correlation values as text
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            value = correlation_matrix.iloc[i, j]
            color = 'white' if abs(value) > 0.5 else 'black'
            ax1.text(j, i, f'{value:.2f}', ha='center', va='center',
                    color=color, fontsize=11, fontweight='bold')

    ax1.set_title('Asset Return Correlations', fontsize=14, fontweight='bold', pad=15)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Correlation Coefficient', fontsize=10)

    # Plot 2: Covariance Heatmap
    im2 = ax2.imshow(covariance_matrix, cmap='viridis', aspect='auto')

    # Set ticks and labels
    ax2.set_xticks(np.arange(len(covariance_matrix.columns)))
    ax2.set_yticks(np.arange(len(covariance_matrix.columns)))
    ax2.set_xticklabels(covariance_matrix.columns, rotation=45, ha='right', fontsize=10)
    ax2.set_yticklabels(covariance_matrix.columns, fontsize=10)

    # Add covariance values as text
    for i in range(len(covariance_matrix.columns)):
        for j in range(len(covariance_matrix.columns)):
            value = covariance_matrix.iloc[i, j]
            # Use white text for darker cells, black for lighter
            max_val = covariance_matrix.max().max()
            min_val = covariance_matrix.min().min()
            # Normalize value to 0-1 range
            norm_value = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            # Use white text for darker cells (low values in viridis are dark)
            color = 'white' if norm_value < 0.6 else 'black'
            ax2.text(j, i, f'{value:.2f}', ha='center', va='center',
                    color=color, fontsize=11, fontweight='bold')

    ax2.set_title('Asset Return Covariances (%² units)', fontsize=14, fontweight='bold', pad=15)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Covariance', fontsize=10)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def create_portfolio_correlation_heatmap(strategies, title="Portfolio Return Correlation and Covariance Analysis"):
    """
    Create correlation and covariance heatmaps for portfolio returns

    Args:
        strategies: Dict of strategy instances with wealth_history attribute
        title: Overall title for the figure

    Returns:
        matplotlib.figure.Figure
    """
    # Collect portfolio returns from wealth histories
    portfolio_returns = {}

    for name, strategy in strategies.items():
        if hasattr(strategy, 'wealth_history') and len(strategy.wealth_history) > 1:
            wealth = np.array(strategy.wealth_history)
            # Calculate returns as price multipliers
            returns = wealth[1:] / wealth[:-1]
            # Convert to percentage returns
            returns_pct = (returns - 1.0) * 100

            # Clean up name for display
            display_name = name.split('. ', 1)[1] if '. ' in name else name
            # Truncate if too long
            display_name = display_name[:25] + '...' if len(display_name) > 25 else display_name

            portfolio_returns[display_name] = returns_pct

    if not portfolio_returns:
        raise ValueError("No portfolio returns found")

    # Create DataFrame
    returns_df = pd.DataFrame(portfolio_returns)

    # Calculate correlation and covariance matrices
    correlation_matrix = returns_df.corr()
    covariance_matrix = returns_df.cov()

    # Determine figure size based on number of portfolios
    n_portfolios = len(portfolio_returns)
    fig_height = max(8, n_portfolios * 0.5)
    fig_width = max(16, n_portfolios * 0.8)

    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    # Plot 1: Correlation Heatmap
    im1 = ax1.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Set ticks and labels
    ax1.set_xticks(np.arange(len(correlation_matrix.columns)))
    ax1.set_yticks(np.arange(len(correlation_matrix.columns)))
    ax1.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right', fontsize=9)
    ax1.set_yticklabels(correlation_matrix.columns, fontsize=9)

    # Add correlation values as text (only if not too many portfolios)
    if n_portfolios <= 10:
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                value = correlation_matrix.iloc[i, j]
                color = 'white' if abs(value) > 0.5 else 'black'
                ax1.text(j, i, f'{value:.2f}', ha='center', va='center',
                        color=color, fontsize=9, fontweight='bold')

    ax1.set_title('Portfolio Return Correlations', fontsize=14, fontweight='bold', pad=15)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Correlation Coefficient', fontsize=10)

    # Plot 2: Covariance Heatmap
    im2 = ax2.imshow(covariance_matrix, cmap='viridis', aspect='auto')

    # Set ticks and labels
    ax2.set_xticks(np.arange(len(covariance_matrix.columns)))
    ax2.set_yticks(np.arange(len(covariance_matrix.columns)))
    ax2.set_xticklabels(covariance_matrix.columns, rotation=45, ha='right', fontsize=9)
    ax2.set_yticklabels(covariance_matrix.columns, fontsize=9)

    # Add covariance values as text (only if not too many portfolios)
    if n_portfolios <= 10:
        for i in range(len(covariance_matrix.columns)):
            for j in range(len(covariance_matrix.columns)):
                value = covariance_matrix.iloc[i, j]
                # Use white text for darker cells, black for lighter
                max_val = covariance_matrix.max().max()
                min_val = covariance_matrix.min().min()
                # Normalize value to 0-1 range
                norm_value = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                # Use white text for darker cells (low values in viridis are dark)
                color = 'white' if norm_value < 0.6 else 'black'
                ax2.text(j, i, f'{value:.2f}', ha='center', va='center',
                        color=color, fontsize=9, fontweight='bold')

    ax2.set_title('Portfolio Return Covariances (%² units)', fontsize=14, fontweight='bold', pad=15)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Covariance', fontsize=10)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def visualize_subset(portfolio_numbers, **kwargs):
    """
    Convenience wrapper accepting portfolio numbers instead of full names.

    Args:
        portfolio_numbers: List of integers (e.g., [1, 4, 7, 8])
        **kwargs: Passed to compare_selected_portfolios()

    Returns:
        fig: Matplotlib figure object
        strategies: Dict of strategy instances with results

    Example:
        # Visualize portfolios 1, 4, 7, and 8
        fig, strategies = visualize_subset([1, 4, 7, 8])
        fig.savefig('results/my_comparison.png')

        # Without baselines
        fig, strategies = visualize_subset([1, 4, 7, 8], include_baselines=False)
    """
    from metrics import compare_selected_portfolios, create_portfolio_registry

    registry = create_portfolio_registry()

    # Build name list
    portfolio_names = []
    for num in portfolio_numbers:
        # Find the portfolio with this number
        for name in registry:
            if name.startswith(f'{num}. '):
                portfolio_names.append(name)
                break
        else:
            raise ValueError(f"No portfolio found with number {num}")

    return compare_selected_portfolios(portfolio_names, **kwargs)
