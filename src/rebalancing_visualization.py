#!/usr/bin/env python3
"""
Visualization module for dynamic portfolio rebalancing results.
Creates comprehensive plots and charts for backtest analysis.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5 backend for interactive plots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Any
import os
import logging
from datetime import datetime

from config import RebalancingConfig
from portfolio_tracker import PortfolioTracker


class RebalancingVisualizer:
    """Creates visualizations for rebalancing backtest results."""
    
    def __init__(self, config: RebalancingConfig):
        """Initialize visualizer with configuration."""
        self.config = config
        self.colors = {
            # Baseline portfolios
            'static_baseline': '#d62728',       # Red
            'rebalanced_baseline': '#ff7f7f',   # Light red
            'baseline': '#d62728',              # Legacy compatibility
            
            # New rebalancing strategy portfolios
            'buy_and_hold': '#d62728',          # Red - natural drift baseline
            'target_weight': '#ff7f0e',         # Orange - rebalancing to initial weights
            'equal_weight': '#2ca02c',          # Green - equal weight rebalancing
            'spy_only': '#8c564b',              # Brown - 100% SPY market benchmark
            
            # Optimization methods
            'mean_variance': '#1f77b4',         # Blue
            'robust_mean_variance': '#17becf',  # Cyan
            'risk_parity': '#bcbd22',           # Olive
            'min_variance': '#9467bd',          # Purple
            'max_sharpe': '#8c564b',            # Brown
            'max_diversification': '#e377c2',   # Pink
            'hierarchical_clustering': '#7f7f7f', # Gray
            
            # Legacy names
            'vanilla': '#1f77b4',               # Blue
            'robust': '#17becf',                # Cyan
            
            # Mixed portfolios
            'mixed_mean_variance': '#1f77b4',   # Blue (same as base)
            'mixed_robust_mean_variance': '#17becf', # Cyan (same as base)
            'mixed_vanilla': '#1f77b4',         # Legacy
            'mixed_robust': '#17becf',          # Legacy
        }
        
        # Create plots directory
        if config.save_plots:
            os.makedirs(config.plots_directory, exist_ok=True)
    
    def plot_cumulative_returns(self, 
                              performance_tracker: PortfolioTracker,
                              save_path: Optional[str] = None) -> None:
        """
        Plot cumulative returns for all portfolios over time with datetime x-axis.
        
        Parameters:
        -----------
        performance_tracker : PortfolioTracker
            Tracker with backtest results
        save_path : str, optional
            Path to save the plot
        """
        # Get performance data
        cumulative_returns_df = performance_tracker.get_all_cumulative_returns()
        portfolio_values_df = performance_tracker.get_all_portfolio_values()

        if cumulative_returns_df.empty:
            logging.warning("No performance data available for plotting")
            return

        # Create figure
        logging.info("Creating cumulative returns plot...")
        fig, ax = plt.subplots(figsize=(14, 8))
        logging.info(f"Figure created with number: {fig.number}")

        # Plot cumulative returns for each portfolio
        portfolios = cumulative_returns_df.columns
        rebalancing_dates = cumulative_returns_df.index  # All dates are rebalancing dates
        
        for portfolio in portfolios:
            # Get cumulative returns for this portfolio
            portfolio_cumulative_returns = cumulative_returns_df[portfolio].dropna()

            if portfolio_cumulative_returns.empty:
                logging.warning(f"Skipping {portfolio}: no cumulative returns data")
                continue

            logging.info(f"Plotting {portfolio}: {len(portfolio_cumulative_returns)} data points")

            # Convert cumulative returns to percentage (subtract 1 to get return, multiply by 100 for percentage)
            cumulative_returns_pct = (portfolio_cumulative_returns - 1) * 100

            # Use the datetime index as x-axis - convert to numpy array for matplotlib compatibility
            dates = portfolio_cumulative_returns.index.to_numpy()

            color = self.colors.get(portfolio, plt.cm.tab10(len(list(portfolios))))

            # Set line styles: dashed for baselines and mixed portfolios, solid for optimization methods
            baseline_names = ['static_baseline', 'rebalanced_baseline', 'baseline']

            if portfolio in baseline_names:
                linestyle = '--'  # Dashed for baseline portfolios
                linewidth = 1.5 if portfolio == 'static_baseline' else 2
            elif portfolio.startswith('mixed_'):
                linestyle = ':'   # Dotted for mixed portfolios
                linewidth = 2
            else:
                linestyle = '-'   # Solid for optimization portfolios
                linewidth = 2.5
            
            # Create descriptive labels for different portfolio types
            if portfolio.startswith('mixed_'):
                method = portfolio.replace('mixed_', '')
                label = f"Mixed {method.title()} ({self.config.mixed_cash_percentage:.0%} Cash)"
            elif portfolio == 'static_baseline':
                label = "Static Baseline (Buy & Hold)"
            elif portfolio == 'rebalanced_baseline':
                label = "Rebalanced Baseline"
            elif portfolio == 'baseline':  # Legacy compatibility
                label = "Baseline"
            else:
                # Optimization method - convert underscores and capitalize properly
                label = portfolio.replace('_', ' ').title()
            
            # Handle overlapping lines by using different alpha and markers for identical values
            alpha = 0.8 if portfolio in ['target_weight', 'mean_variance'] else 1.0
            zorder = 2 if portfolio == 'target_weight' else 1  # Put target_weight on top

            # Use different markers for portfolios with potential identical values
            if portfolio == 'target_weight':
                marker_style = '^' if len(dates) < 50 else None  # Triangle marker
                marker_size = 5
            elif portfolio == 'mean_variance':
                marker_style = 's' if len(dates) < 50 else None  # Square marker
                marker_size = 6  # Larger marker for mean_variance (plotted first)
            else:
                marker_style = 'o' if len(dates) < 20 else None
                marker_size = 4

            ax.plot(dates, cumulative_returns_pct,
                   label=label,
                   color=color,
                   linestyle=linestyle,
                   linewidth=linewidth,
                   marker=marker_style,
                   markersize=marker_size,
                   alpha=alpha,
                   zorder=zorder)
        
        # Add vertical lines for rebalancing dates
        for rebal_date in rebalancing_dates:
            ax.axvline(x=rebal_date, color='gray', linestyle=':', alpha=0.7, linewidth=1)
        
        # Add markers at top of plot for rebalancing dates
        if len(rebalancing_dates) > 0:
            y_max = ax.get_ylim()[1]
            y_marker = y_max * 0.95  # Place markers near top
            
            ax.scatter(rebalancing_dates, [y_marker] * len(rebalancing_dates), 
                      marker='v', s=50, color='red', alpha=0.8, 
                      label='Rebalancing', zorder=5)
        
        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.set_title('Portfolio Performance Comparison\n(Dynamic Rebalancing)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.tick_params(axis='x', rotation=45)
        fig.autofmt_xdate()  # Auto-format date labels
        
        # Add period information to title
        total_periods = len(cumulative_returns_df)
        period_days = self.config.rebalancing_period_days
        ax.text(0.02, 0.98, f'{total_periods} periods × {period_days} days',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', alpha=0.7)
        
        plt.tight_layout()
        
        # Save plot
        if save_path or self.config.save_plots:
            save_file = save_path or os.path.join(self.config.plots_directory, 
                                                'cumulative_returns.png')
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            logging.info(f"Cumulative returns plot saved to {save_file}")
        
        if self.config.show_plots_interactive:
            plt.show(block=False)  # Non-blocking show - allows multiple plots to stay open
        elif self.config.close_plots_after_save:
            plt.close()

    def plot_daily_cumulative_returns(self,
                                     performance_engine,
                                     save_path: Optional[str] = None) -> None:
        """
        Plot daily cumulative returns for all portfolios using underlying daily data.

        This method calculates and plots true daily cumulative returns by reconstructing
        portfolio performance from daily returns data, rather than just plotting
        cumulative returns at rebalancing dates.

        Parameters:
        -----------
        performance_engine : PerformanceEngine
            Engine with access to daily returns data
        save_path : str, optional
            Path to save the plot
        """
        if not hasattr(performance_engine, 'returns_data') or performance_engine.returns_data is None:
            logging.error("No daily returns data available in performance engine")
            return

        if not hasattr(performance_engine, 'period_manager') or performance_engine.period_manager is None:
            logging.error("No period manager available in performance engine")
            return

        logging.info("Creating daily cumulative returns plot...")

        # Get the underlying daily returns data
        daily_returns = performance_engine.returns_data
        period_manager = performance_engine.period_manager
        tracker = performance_engine.tracker

        if tracker is None:
            logging.error("No portfolio tracker available")
            return

        # Get portfolio names
        portfolio_names = tracker.portfolio_names

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        logging.info(f"Figure created with number: {fig.number}")

        # Storage for daily cumulative returns
        daily_cumulative_returns = {}
        rebalancing_dates = []

        # Calculate daily cumulative returns for each portfolio
        for portfolio_name in portfolio_names:
            portfolio_daily_returns = []
            portfolio_dates = []

            # Iterate through all periods to reconstruct daily performance
            for period_num, period_data, period_info in period_manager.iter_periods():
                period_start_date = period_info['period_start']
                period_end_date = period_info['period_end']

                # Get weights for this portfolio at this period
                weights_history = tracker.get_portfolio_weights_history(portfolio_name)

                # Find the appropriate weights for this period
                if period_end_date in weights_history.index:
                    period_weights = weights_history.loc[period_end_date]
                elif len(weights_history) > 0:
                    # Use the most recent weights available
                    available_dates = weights_history.index[weights_history.index <= period_end_date]
                    if len(available_dates) > 0:
                        period_weights = weights_history.loc[available_dates[-1]]
                    else:
                        # Use baseline weights if no weights available
                        period_weights = pd.Series(
                            performance_engine.baseline_weights,
                            index=performance_engine.asset_names
                        )
                else:
                    continue

                # Calculate daily portfolio returns for this period
                period_daily_returns = period_data.dot(period_weights)

                # Store the daily returns and dates
                portfolio_daily_returns.extend(period_daily_returns.tolist())
                portfolio_dates.extend(period_data.index.tolist())

                # Store rebalancing date
                if period_end_date not in rebalancing_dates:
                    rebalancing_dates.append(period_end_date)

            if len(portfolio_daily_returns) == 0:
                logging.warning(f"No daily returns calculated for {portfolio_name}")
                continue

            # Convert to pandas Series
            daily_returns_series = pd.Series(portfolio_daily_returns, index=portfolio_dates)

            # Calculate cumulative returns
            cumulative_returns_series = (1 + daily_returns_series).cumprod()

            # Store for plotting
            daily_cumulative_returns[portfolio_name] = cumulative_returns_series

        # Plot daily cumulative returns for each portfolio
        for portfolio_name, cumulative_returns in daily_cumulative_returns.items():
            if cumulative_returns.empty:
                continue

            logging.info(f"Plotting daily {portfolio_name}: {len(cumulative_returns)} data points")

            # Convert to percentage
            cumulative_returns_pct = (cumulative_returns - 1) * 100

            # Get dates - convert to numpy array for matplotlib compatibility
            dates = cumulative_returns.index.to_numpy()

            # Get color and style
            color = self.colors.get(portfolio_name, plt.cm.tab10(len(daily_cumulative_returns)))

            # Set line styles
            baseline_names = ['static_baseline', 'rebalanced_baseline', 'baseline']

            if portfolio_name in baseline_names:
                linestyle = '--'
                linewidth = 1.5 if portfolio_name == 'static_baseline' else 2
            elif portfolio_name.startswith('mixed_'):
                linestyle = ':'
                linewidth = 2
            else:
                linestyle = '-'
                linewidth = 2.5

            # Create descriptive labels
            if portfolio_name.startswith('mixed_'):
                method = portfolio_name.replace('mixed_', '')
                label = f"Mixed {method.title()} ({self.config.mixed_cash_percentage:.0%} Cash)"
            elif portfolio_name == 'static_baseline':
                label = "Static Baseline (Buy & Hold)"
            elif portfolio_name == 'rebalanced_baseline':
                label = "Rebalanced Baseline"
            elif portfolio_name == 'baseline':
                label = "Baseline"
            else:
                label = portfolio_name.replace('_', ' ').title()

            # Handle overlapping lines
            alpha = 0.8 if portfolio_name in ['target_weight', 'mean_variance'] else 1.0
            zorder = 2 if portfolio_name == 'target_weight' else 1

            # Plot the daily cumulative returns
            ax.plot(dates, cumulative_returns_pct,
                   label=label,
                   color=color,
                   linestyle=linestyle,
                   linewidth=linewidth,
                   alpha=alpha,
                   zorder=zorder)

        # Add vertical lines for rebalancing dates
        for rebal_date in rebalancing_dates:
            if rebal_date in daily_returns.index:  # Only plot if date exists in data
                ax.axvline(x=rebal_date, color='gray', linestyle=':', alpha=0.7, linewidth=1)

        # Add markers at top for rebalancing dates
        if len(rebalancing_dates) > 0:
            y_max = ax.get_ylim()[1]
            y_marker = y_max * 0.95

            # Filter rebalancing dates to only those in the data range
            valid_rebal_dates = [date for date in rebalancing_dates if date in daily_returns.index]

            if valid_rebal_dates:
                ax.scatter(valid_rebal_dates, [y_marker] * len(valid_rebal_dates),
                          marker='v', s=50, color='red', alpha=0.8,
                          label='Rebalancing', zorder=5)

        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.set_title('Portfolio Performance Comparison\n(Daily Cumulative Returns)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        ax.tick_params(axis='x', rotation=45)
        fig.autofmt_xdate()

        # Add data information
        total_days = len(daily_returns)
        total_periods = len(rebalancing_dates)
        period_days = self.config.rebalancing_period_days
        ax.text(0.02, 0.98, f'{total_days} daily observations, {total_periods} rebalancing periods',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', alpha=0.7)

        plt.tight_layout()

        # Save plot
        if save_path or self.config.save_plots:
            save_file = save_path or os.path.join(self.config.plots_directory,
                                                'daily_cumulative_returns.png')
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            logging.info(f"Daily cumulative returns plot saved to {save_file}")

        if self.config.show_plots_interactive:
            plt.show(block=False)
        elif self.config.close_plots_after_save:
            plt.close()

    def plot_period_returns(self,
                          performance_tracker: PortfolioTracker,
                          save_path: Optional[str] = None) -> None:
        """
        Plot period-by-period returns for all portfolios.
        
        Parameters:
        -----------
        performance_tracker : PortfolioTracker
            Tracker with backtest results
        save_path : str, optional
            Path to save the plot
        """
        # Get data from PortfolioTracker
        returns_df = performance_tracker.get_all_portfolio_returns()
        summary_stats = performance_tracker.get_portfolio_summary_statistics()

        if returns_df.empty:
            logging.warning("No portfolio returns data available for plotting")
            return

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        portfolios = returns_df.columns
        periods = returns_df.index  # These are the dates
        
        # Plot 1: Period returns
        width = 0.8 / len(portfolios)
        x_positions = np.arange(len(periods))
        
        for i, portfolio in enumerate(portfolios):
            # Get period returns for this portfolio
            portfolio_returns = returns_df[portfolio].dropna() * 100  # Convert to percentage

            if portfolio_returns.empty:
                continue

            color = self.colors.get(portfolio, plt.cm.tab10(i))
            alpha = 0.6 if portfolio == 'baseline' else 0.8

            ax1.bar(x_positions + i * width - width * (len(portfolios) - 1) / 2,
                   portfolio_returns.values,
                   width=width,
                   label=portfolio.title(),
                   color=color,
                   alpha=alpha)

        ax1.set_xlabel('Rebalancing Period')
        ax1.set_ylabel('Period Return (%)')
        ax1.set_title('Period Returns by Portfolio')
        ax1.set_xticks(x_positions)
        # Use date labels instead of period numbers
        ax1.set_xticklabels([date.strftime('%Y-%m') for date in periods], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Plot 2: Summary Statistics Bar Chart
        if not summary_stats.empty:
            # Create a simplified summary plot
            portfolios_list = list(portfolios)
            total_returns = []
            sharpe_ratios = []

            for portfolio in portfolios_list:
                if portfolio in summary_stats.index:
                    total_returns.append(summary_stats.loc[portfolio, 'Total_Return'] * 100)
                    sharpe_ratios.append(summary_stats.loc[portfolio, 'Sharpe_Ratio'])
                else:
                    total_returns.append(0)
                    sharpe_ratios.append(0)

            x_pos = np.arange(len(portfolios_list))

            # Create twin axis for sharpe ratio
            ax2_twin = ax2.twinx()

            bars1 = ax2.bar(x_pos - 0.2, total_returns, 0.4,
                           label='Total Return (%)', alpha=0.7, color='steelblue')
            bars2 = ax2_twin.bar(x_pos + 0.2, sharpe_ratios, 0.4,
                                label='Sharpe Ratio', alpha=0.7, color='orange')

            ax2.set_xlabel('Portfolio')
            ax2.set_ylabel('Total Return (%)', color='steelblue')
            ax2_twin.set_ylabel('Sharpe Ratio', color='orange')
            ax2.set_title('Portfolio Summary Statistics')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([p.title() for p in portfolios_list], rotation=45)

            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

            # Add legends
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax2.text(0.5, 0.5, 'No summary statistics available',
                    transform=ax2.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        
        # Save plot
        if save_path or self.config.save_plots:
            save_file = save_path or os.path.join(self.config.plots_directory, 
                                                'period_analysis.png')
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            logging.info(f"Period analysis plot saved to {save_file}")
        
        if self.config.show_plots_interactive:
            plt.show(block=False)  # Non-blocking show - allows multiple plots to stay open
        elif self.config.close_plots_after_save:
            plt.close()
    
    def plot_weights_evolution(self, 
                             performance_tracker: PortfolioTracker,
                             asset_names: List[str],
                             portfolio_name: str = 'vanilla',
                             save_path: Optional[str] = None) -> None:
        """
        Plot evolution of portfolio weights over time.
        
        Parameters:
        -----------
        performance_tracker : PortfolioTracker
            Tracker with backtest results
        asset_names : List[str]
            Names of assets
        portfolio_name : str
            Which portfolio to plot
        save_path : str, optional
            Path to save the plot
        """
        # Get weights history for the specific portfolio
        weights_history = performance_tracker.get_portfolio_weights_history(portfolio_name)

        if weights_history.empty:
            logging.warning(f"No weights data found for portfolio '{portfolio_name}'")
            return
        
        # Sort by date
        weights_history = weights_history.sort_index()

        # Create stacked area plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Prepare data for stacking - weights_history has assets as columns and dates as index
        dates = weights_history.index
        asset_labels = weights_history.columns
        weights_matrix = weights_history.T.values  # Transpose to get assets x dates
        
        # Create color map
        colors = plt.cm.tab20(np.linspace(0, 1, len(asset_labels)))
        
        # Create stacked area plot
        ax.stackplot(dates, weights_matrix, 
                    labels=asset_labels, 
                    colors=colors, 
                    alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Rebalancing Period', fontsize=12)
        ax.set_ylabel('Portfolio Weight', fontsize=12)
        ax.set_title(f'{portfolio_name.title()} Portfolio - Weight Evolution', 
                    fontsize=14, fontweight='bold')
        
        # Legend - only show assets with meaningful weights
        handles, labels = ax.get_legend_handles_labels()
        # Show legend for assets with average weight > 1%
        significant_assets = []
        for i, asset in enumerate(asset_labels):
            avg_weight = np.mean(weights_matrix[i])
            if avg_weight > 0.01:  # 1% threshold
                significant_assets.append((handles[i], labels[i]))
        
        if significant_assets:
            legend_handles, legend_labels = zip(*significant_assets)
            ax.legend(legend_handles, legend_labels, 
                     loc='center left', bbox_to_anchor=(1, 0.5))
        
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path or self.config.save_plots:
            save_file = save_path or os.path.join(self.config.plots_directory, 
                                                f'weights_evolution_{portfolio_name}.png')
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            logging.info(f"Weights evolution plot saved to {save_file}")
        
        if self.config.show_plots_interactive:
            plt.show(block=False)  # Non-blocking show - allows multiple plots to stay open
        elif self.config.close_plots_after_save:
            plt.close()
    
    def plot_performance_summary(self, 
                               performance_tracker: PortfolioTracker,
                               save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive performance summary dashboard.
        
        Parameters:
        -----------
        performance_tracker : PortfolioTracker
            Tracker with backtest results
        save_path : str, optional
            Path to save the plot
        """
        # Get summary statistics from PortfolioTracker
        summary_stats = performance_tracker.get_portfolio_summary_statistics()

        if summary_stats.empty:
            logging.warning("No portfolio summary statistics available for plotting")
            return
        
        # Use existing summary statistics
        portfolios = summary_stats.index.tolist()
        stats = {}

        for portfolio in portfolios:
            stats[portfolio] = {
                'total_return': summary_stats.loc[portfolio, 'Total_Return'],
                'volatility': summary_stats.loc[portfolio, 'Volatility'],
                'sharpe_ratio': summary_stats.loc[portfolio, 'Sharpe_Ratio'],
                'max_drawdown': summary_stats.loc[portfolio, 'Max_Drawdown'],
                'annual_return': summary_stats.loc[portfolio, 'Annual_Return']
            }
        
        # Create dashboard with 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        portfolios = list(stats.keys())
        
        # Plot 1: Total returns comparison
        total_returns = [stats[p]['total_return'] * 100 for p in portfolios]
        colors_list = [self.colors.get(p, plt.cm.tab10(i)) for i, p in enumerate(portfolios)]
        
        bars1 = ax1.bar(portfolios, total_returns, color=colors_list, alpha=0.8)
        ax1.set_title('Total Return Comparison', fontweight='bold')
        ax1.set_ylabel('Total Return (%)')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, total_returns):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Plot 2: Sharpe ratios
        sharpe_ratios = [stats[p]['sharpe_ratio'] for p in portfolios]
        
        bars2 = ax2.bar(portfolios, sharpe_ratios, color=colors_list, alpha=0.8)
        ax2.set_title('Average Sharpe Ratio', fontweight='bold')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for bar, value in zip(bars2, sharpe_ratios):
            height = bar.get_height()
            y_pos = height + 0.01 if height >= 0 else height - 0.02
            ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{value:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        # Plot 3: Volatility comparison
        volatilities = [stats[p]['volatility'] * 100 for p in portfolios]
        
        bars3 = ax3.bar(portfolios, volatilities, color=colors_list, alpha=0.8)
        ax3.set_title('Return Volatility', fontweight='bold')
        ax3.set_ylabel('Volatility (%)')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars3, volatilities):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Plot 4: Max drawdown comparison
        max_drawdowns = [abs(stats[p]['max_drawdown']) * 100 for p in portfolios]
        
        bars4 = ax4.bar(portfolios, max_drawdowns, color=colors_list, alpha=0.8)
        ax4.set_title('Maximum Drawdown', fontweight='bold')
        ax4.set_ylabel('Max Drawdown (%)')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars4, max_drawdowns):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Overall title
        total_periods = len(summary_stats) if not summary_stats.empty else 0
        fig.suptitle(f'Portfolio Performance Summary\n'
                    f'{total_periods} Portfolios × {self.config.rebalancing_period_days} Day Periods',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        if save_path or self.config.save_plots:
            save_file = save_path or os.path.join(self.config.plots_directory, 
                                                'performance_summary.png')
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            logging.info(f"Performance summary plot saved to {save_file}")
        
        if self.config.show_plots_interactive:
            plt.show(block=False)  # Non-blocking show - allows multiple plots to stay open
        elif self.config.close_plots_after_save:
            plt.close()
    
    def create_comprehensive_report(self, 
                                  performance_tracker: PortfolioTracker,
                                  asset_names: List[str]) -> None:
        """
        Create all standard plots for rebalancing analysis.
        
        Parameters:
        -----------
        performance_tracker : PortfolioTracker
            Tracker with backtest results
        asset_names : List[str]
            Names of assets
        """
        logging.info("Creating comprehensive rebalancing report...")
        
        # Plot cumulative returns
        self.plot_cumulative_returns(performance_tracker)
        
        # Plot period analysis
        self.plot_period_returns(performance_tracker)
        
        # Plot performance summary
        self.plot_performance_summary(performance_tracker)
        
        # Plot weights evolution for each optimized portfolio
        all_portfolios = performance_tracker.portfolio_names
        portfolios = [p for p in all_portfolios if p not in ['baseline', 'static_baseline', 'rebalanced_baseline']]
        
        for portfolio in portfolios:
            self.plot_weights_evolution(performance_tracker, asset_names, portfolio)
        
        logging.info("Comprehensive rebalancing report completed")

    def plot_portfolio_comparison(self,
                                performance_tracker: PortfolioTracker,
                                portfolio1: str,
                                portfolio2: str,
                                save_path: Optional[str] = None) -> None:
        """
        Create a focused comparison plot between two specific portfolios.

        Parameters:
        -----------
        performance_tracker : PortfolioTracker
            Tracker with backtest results
        portfolio1 : str
            Name of first portfolio to compare
        portfolio2 : str
            Name of second portfolio to compare
        save_path : str, optional
            Path to save the plot
        """
        try:
            # Get cumulative returns for both portfolios
            cumulative_returns_df = performance_tracker.get_all_cumulative_returns()

            if portfolio1 not in cumulative_returns_df.columns or portfolio2 not in cumulative_returns_df.columns:
                logging.error(f"One or both portfolios not found: {portfolio1}, {portfolio2}")
                return

            portfolio1_returns = cumulative_returns_df[portfolio1].dropna()
            portfolio2_returns = cumulative_returns_df[portfolio2].dropna()

            if portfolio1_returns.empty or portfolio2_returns.empty:
                logging.warning(f"No data available for comparison: {portfolio1} vs {portfolio2}")
                return

            # Create comparison plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Portfolio Comparison: {portfolio1} vs {portfolio2}', fontsize=16, fontweight='bold')

            # 1. Cumulative Returns Comparison
            ax1 = axes[0, 0]
            color1 = self.colors.get(portfolio1, '#1f77b4')
            color2 = self.colors.get(portfolio2, '#ff7f0e')

            ax1.plot(portfolio1_returns.index.to_numpy(), (portfolio1_returns - 1) * 100,
                    color=color1, label=portfolio1.replace('_', ' ').title(), linewidth=2)
            ax1.plot(portfolio2_returns.index.to_numpy(), (portfolio2_returns - 1) * 100,
                    color=color2, label=portfolio2.replace('_', ' ').title(), linewidth=2)

            ax1.set_title('Cumulative Returns Comparison')
            ax1.set_ylabel('Cumulative Return (%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Performance Metrics Bar Chart
            ax2 = axes[0, 1]
            metrics = performance_tracker.get_portfolio_summary_statistics()

            if not metrics.empty and portfolio1 in metrics.index and portfolio2 in metrics.index:
                metric_names = ['Total_Return', 'Sharpe_Ratio', 'Volatility', 'Max_Drawdown']
                available_metrics = [m for m in metric_names if m in metrics.columns]

                if available_metrics:
                    x = np.arange(len(available_metrics))
                    width = 0.35

                    values1 = [metrics.loc[portfolio1, m] for m in available_metrics]
                    values2 = [metrics.loc[portfolio2, m] for m in available_metrics]

                    ax2.bar(x - width/2, values1, width, label=portfolio1.replace('_', ' ').title(),
                           color=color1, alpha=0.7)
                    ax2.bar(x + width/2, values2, width, label=portfolio2.replace('_', ' ').title(),
                           color=color2, alpha=0.7)

                    ax2.set_title('Performance Metrics Comparison')
                    ax2.set_xticks(x)
                    ax2.set_xticklabels([m.replace('_', ' ') for m in available_metrics], rotation=45)
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)

            # 3. Rolling Sharpe Ratio (if data available)
            ax3 = axes[1, 0]
            try:
                returns1 = performance_tracker.get_portfolio_returns(portfolio1).dropna()
                returns2 = performance_tracker.get_portfolio_returns(portfolio2).dropna()

                if len(returns1) > 30 and len(returns2) > 30:  # Need sufficient data
                    rolling_sharpe1 = returns1.rolling(30).mean() / returns1.rolling(30).std() * np.sqrt(252)
                    rolling_sharpe2 = returns2.rolling(30).mean() / returns2.rolling(30).std() * np.sqrt(252)

                    ax3.plot(rolling_sharpe1.index.to_numpy(), rolling_sharpe1.values, color=color1,
                            label=portfolio1.replace('_', ' ').title(), alpha=0.7)
                    ax3.plot(rolling_sharpe2.index.to_numpy(), rolling_sharpe2.values, color=color2,
                            label=portfolio2.replace('_', ' ').title(), alpha=0.7)

                    ax3.set_title('30-Day Rolling Sharpe Ratio')
                    ax3.set_ylabel('Sharpe Ratio')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                else:
                    ax3.text(0.5, 0.5, 'Insufficient data\nfor rolling analysis',
                            ha='center', va='center', transform=ax3.transAxes)
                    ax3.set_title('Rolling Analysis (Insufficient Data)')
            except Exception as e:
                logging.warning(f"Could not create rolling analysis: {str(e)}")
                ax3.text(0.5, 0.5, 'Rolling analysis\nnot available',
                        ha='center', va='center', transform=ax3.transAxes)

            # 4. Drawdown Comparison
            ax4 = axes[1, 1]
            try:
                # Calculate drawdowns
                peak1 = portfolio1_returns.cummax()
                drawdown1 = (portfolio1_returns - peak1) / peak1 * 100

                peak2 = portfolio2_returns.cummax()
                drawdown2 = (portfolio2_returns - peak2) / peak2 * 100

                ax4.fill_between(drawdown1.index.to_numpy(), drawdown1.values, 0, color=color1, alpha=0.3,
                               label=f'{portfolio1.replace("_", " ").title()} Drawdown')
                ax4.fill_between(drawdown2.index.to_numpy(), drawdown2.values, 0, color=color2, alpha=0.3,
                               label=f'{portfolio2.replace("_", " ").title()} Drawdown')

                ax4.set_title('Drawdown Comparison')
                ax4.set_ylabel('Drawdown (%)')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            except Exception as e:
                logging.warning(f"Could not create drawdown comparison: {str(e)}")
                ax4.text(0.5, 0.5, 'Drawdown analysis\nnot available',
                        ha='center', va='center', transform=ax4.transAxes)

            plt.tight_layout()

            # Save or show plot
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Portfolio comparison plot saved to: {save_path}")

            # Show plot (non-blocking)
            if self.config.show_plots:
                plt.show(block=False)

        except Exception as e:
            logging.error(f"Error creating portfolio comparison plot: {str(e)}")
            raise

    def plot_max_drawdown_time_series(self,
                                    performance_tracker: PortfolioTracker,
                                    save_path: Optional[str] = None) -> None:
        """
        Plot rolling drawdown comparison for all portfolios.

        Shows the actual rolling drawdown evolution over time using the existing
        drawdown calculation methodology from the codebase.

        Parameters:
        -----------
        performance_tracker : PortfolioTracker
            Tracker with backtest results
        save_path : str, optional
            Path to save the plot
        """
        try:
            # Read portfolio data from CSV files to calculate rolling drawdown
            portfolios_data = {}

            # Get list of portfolios from tracker
            portfolio_names = performance_tracker.portfolio_names

            for portfolio in portfolio_names:
                try:
                    # Read the portfolio CSV file
                    csv_file = f"../results/rebalancing/portfolio_analysis_{portfolio}.csv"
                    if os.path.exists(csv_file):
                        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                        if 'Portfolio_Value' in df.columns:
                            # Use existing rolling drawdown calculation methodology
                            # Based on main.py lines 797-800 and performance_analysis.py lines 53-54
                            portfolio_values = df['Portfolio_Value']

                            # Calculate rolling maximum (expanding window)
                            rolling_max = portfolio_values.expanding().max()

                            # Calculate rolling drawdown: current value / rolling max - 1
                            rolling_drawdown = portfolio_values / rolling_max - 1

                            # Convert to percentage
                            portfolios_data[portfolio] = rolling_drawdown * 100
                        else:
                            logging.warning(f"No Portfolio_Value column found in {csv_file}")
                    else:
                        logging.warning(f"CSV file not found: {csv_file}")
                except Exception as e:
                    logging.warning(f"Could not load data for {portfolio}: {str(e)}")
                    continue

            if not portfolios_data:
                logging.warning("No portfolio data available for rolling drawdown plotting")
                return

            # Create figure with 2x1 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
            logging.info("Creating rolling max and drawdown comparison plots...")

            # First, collect max drawdown data for the top plot
            max_drawdown_data = {}

            # Re-read data to get max drawdown values
            for portfolio in portfolio_names:
                try:
                    csv_file = f"../results/rebalancing/portfolio_analysis_{portfolio}.csv"
                    if os.path.exists(csv_file):
                        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                        if 'Max_Drawdown' in df.columns:
                            # Max drawdown is already in percentage format in the CSV
                            max_drawdown_series = df['Max_Drawdown']
                            max_drawdown_data[portfolio] = max_drawdown_series
                except Exception:
                    continue

            # Plot 1: Maximum Drawdown Values
            for portfolio, max_dd_series in max_drawdown_data.items():
                if max_dd_series.empty:
                    continue

                color = self.colors.get(portfolio, plt.cm.tab10(len(max_drawdown_data)))

                # Set line styles consistent with existing code
                if portfolio in ['static_baseline', 'rebalanced_baseline', 'baseline', 'buy_and_hold']:
                    linestyle = '--'
                    linewidth = 2
                elif portfolio.startswith('mixed_'):
                    linestyle = ':'
                    linewidth = 2
                else:
                    linestyle = '-'
                    linewidth = 2

                # Create descriptive labels
                if portfolio.startswith('mixed_'):
                    method = portfolio.replace('mixed_', '')
                    label = f"Mixed {method.title()} ({self.config.mixed_cash_percentage:.0%} Cash)"
                elif portfolio == 'static_baseline':
                    label = "Static Baseline (Buy & Hold)"
                elif portfolio == 'rebalanced_baseline':
                    label = "Rebalanced Baseline"
                elif portfolio == 'baseline':
                    label = "Baseline"
                elif portfolio == 'buy_and_hold':
                    label = "Buy & Hold"
                elif portfolio == 'target_weight':
                    label = "Target Weight"
                elif portfolio == 'equal_weight':
                    label = "Equal Weight"
                elif portfolio == 'spy_only':
                    label = "SPY Only"
                else:
                    label = portfolio.replace('_', ' ').title()

                ax1.plot(max_dd_series.index.to_numpy(), max_dd_series.values,
                        color=color, linestyle=linestyle, linewidth=linewidth,
                        label=label, marker='o' if len(max_dd_series) < 20 else None,
                        markersize=3, alpha=0.9, zorder=3)

                # Add very subtle fill only for significant max drawdowns (< -1%)
                significant_max_dd = max_dd_series[max_dd_series < -1.0]
                if not significant_max_dd.empty:
                    ax1.fill_between(max_dd_series.index.to_numpy(), max_dd_series.values, 0,
                                    where=(max_dd_series < -1.0),
                                    alpha=0.1, color=color, interpolate=True, zorder=1)

            # Format first subplot (Max Drawdown)
            ax1.set_ylabel('Maximum Drawdown (%)', fontsize=12)
            ax1.set_title('Maximum Drawdown Comparison\n(Worst Peak-to-Trough Loss Over Time)',
                         fontsize=14, fontweight='bold')
            ax1.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)
            ax1.grid(True, alpha=0.3)

            # Add horizontal line at 0 (no drawdown) for max drawdown plot
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)

            # Add reference lines for common max drawdown levels
            for level in [-5, -10, -15, -20]:
                ax1.axhline(y=level, color='red', linestyle=':', alpha=0.3, linewidth=0.5)

            # Plot 2: Rolling Drawdowns
            for portfolio, drawdown_series in portfolios_data.items():
                if drawdown_series.empty:
                    continue

                color = self.colors.get(portfolio, plt.cm.tab10(len(portfolios_data)))

                # Set line styles consistent with existing code
                if portfolio in ['static_baseline', 'rebalanced_baseline', 'baseline', 'buy_and_hold']:
                    linestyle = '--'
                    linewidth = 2
                elif portfolio.startswith('mixed_'):
                    linestyle = ':'
                    linewidth = 2
                else:
                    linestyle = '-'
                    linewidth = 2

                # Create descriptive labels
                if portfolio.startswith('mixed_'):
                    method = portfolio.replace('mixed_', '')
                    label = f"Mixed {method.title()} ({self.config.mixed_cash_percentage:.0%} Cash)"
                elif portfolio == 'static_baseline':
                    label = "Static Baseline (Buy & Hold)"
                elif portfolio == 'rebalanced_baseline':
                    label = "Rebalanced Baseline"
                elif portfolio == 'baseline':
                    label = "Baseline"
                elif portfolio == 'buy_and_hold':
                    label = "Buy & Hold"
                elif portfolio == 'target_weight':
                    label = "Target Weight"
                elif portfolio == 'equal_weight':
                    label = "Equal Weight"
                elif portfolio == 'spy_only':
                    label = "SPY Only"
                else:
                    label = portfolio.replace('_', ' ').title()

                # Plot line first (so it's on top)
                ax2.plot(drawdown_series.index.to_numpy(), drawdown_series.values,
                        color=color, linestyle=linestyle, linewidth=linewidth,
                        label=label, marker='o' if len(drawdown_series) < 20 else None,
                        markersize=3, alpha=0.9, zorder=3)

                # Add very subtle fill only for significant drawdowns (< -1%)
                significant_drawdowns = drawdown_series[drawdown_series < -1.0]
                if not significant_drawdowns.empty:
                    ax2.fill_between(drawdown_series.index.to_numpy(), drawdown_series.values, 0,
                                    where=(drawdown_series < -1.0),
                                    alpha=0.1, color=color, interpolate=True, zorder=1)

            # Format second subplot (Rolling Drawdown)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Rolling Drawdown (%)', fontsize=12)
            ax2.set_title('Rolling Drawdown Comparison\n(Portfolio Risk Evolution Over Time)',
                         fontsize=14, fontweight='bold')

            ax2.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)
            ax2.grid(True, alpha=0.3)

            # Format x-axis dates for both subplots
            ax2.tick_params(axis='x', rotation=45)
            fig.autofmt_xdate()

            # Add horizontal line at 0 (no drawdown) for drawdown plot
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)

            # Add reference lines for common drawdown levels
            for level in [-5, -10, -15, -20]:
                ax2.axhline(y=level, color='red', linestyle=':', alpha=0.3, linewidth=0.5)

            # Adjust layout to prevent overlap
            plt.tight_layout()

            # Add period information to the bottom subplot
            total_periods = len(next(iter(portfolios_data.values())))
            period_days = self.config.rebalancing_period_days
            ax2.text(0.02, 0.98, f'{total_periods} periods × {period_days} days',
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', alpha=0.7)

            plt.tight_layout()

            # Save plot
            if save_path or self.config.save_plots:
                save_file = save_path or os.path.join(self.config.plots_directory,
                                                    'rolling_drawdown_comparison.png')
                plt.savefig(save_file, dpi=300, bbox_inches='tight')
                logging.info(f"Rolling drawdown comparison plot saved to {save_file}")

            if self.config.show_plots_interactive:
                plt.show(block=False)
            elif self.config.close_plots_after_save:
                plt.close()

        except Exception as e:
            logging.error(f"Error creating rolling drawdown comparison plot: {str(e)}")
            raise