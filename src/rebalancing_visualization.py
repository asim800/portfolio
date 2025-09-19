#!/usr/bin/env python3
"""
Visualization module for dynamic portfolio rebalancing results.
Creates comprehensive plots and charts for backtest analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Any
import os
import logging
from datetime import datetime

from config import RebalancingConfig
from performance_tracker import PerformanceTracker


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
                              performance_tracker: PerformanceTracker,
                              save_path: Optional[str] = None) -> None:
        """
        Plot cumulative returns for all portfolios over time with datetime x-axis.
        
        Parameters:
        -----------
        performance_tracker : PerformanceTracker
            Tracker with backtest results
        save_path : str, optional
            Path to save the plot
        """
        # Get performance data
        perf_df = performance_tracker.get_performance_summary()
        
        if perf_df.empty:
            logging.warning("No performance data available for plotting")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot cumulative returns for each portfolio
        portfolios = perf_df['portfolio'].unique()
        rebalancing_dates = []  # Track rebalancing dates for markers
        
        for portfolio in portfolios:
            portfolio_data = perf_df[perf_df['portfolio'] == portfolio].copy()
            portfolio_data = portfolio_data.sort_values('period')
            
            # Convert cumulative returns to percentage
            cumulative_returns = (portfolio_data['cumulative_return'] * 100)
            
            # Use end_date as x-axis (when performance was measured)
            dates = pd.to_datetime(portfolio_data['end_date'])
            
            # Store rebalancing dates (use start_date of each period after first)
            if portfolio == portfolios[0]:  # Only collect dates once
                # Rebalancing happens at the start of each period (except first)
                rebalancing_dates = pd.to_datetime(portfolio_data['start_date'].iloc[1:])  # Skip first period
            
            color = self.colors.get(portfolio, plt.cm.tab10(len(portfolios)))
            
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
            
            ax.plot(dates, cumulative_returns, 
                   label=label, 
                   color=color, 
                   linestyle=linestyle,
                   linewidth=linewidth,
                   marker='o' if len(dates) < 20 else None,
                   markersize=4)
        
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
        total_periods = len(perf_df['period'].unique())
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
    
    def plot_period_returns(self, 
                          performance_tracker: PerformanceTracker,
                          save_path: Optional[str] = None) -> None:
        """
        Plot period-by-period returns for all portfolios.
        
        Parameters:
        -----------
        performance_tracker : PerformanceTracker
            Tracker with backtest results
        save_path : str, optional
            Path to save the plot
        """
        perf_df = performance_tracker.get_performance_summary()
        
        if perf_df.empty:
            return
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        portfolios = perf_df['portfolio'].unique()
        periods = sorted(perf_df['period'].unique())
        
        # Plot 1: Period returns
        width = 0.8 / len(portfolios)
        x_positions = np.arange(len(periods))
        
        for i, portfolio in enumerate(portfolios):
            portfolio_data = perf_df[perf_df['portfolio'] == portfolio].copy()
            portfolio_data = portfolio_data.sort_values('period')
            
            period_returns = portfolio_data['period_return'] * 100
            
            color = self.colors.get(portfolio, plt.cm.tab10(i))
            alpha = 0.6 if portfolio == 'baseline' else 0.8
            
            ax1.bar(x_positions + i * width - width * (len(portfolios) - 1) / 2,
                   period_returns,
                   width=width,
                   label=portfolio.title(),
                   color=color,
                   alpha=alpha)
        
        ax1.set_xlabel('Rebalancing Period')
        ax1.set_ylabel('Period Return (%)')
        ax1.set_title('Period Returns by Portfolio')
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels([f'P{p}' for p in periods])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Plot 2: Rolling Sharpe ratio
        for portfolio in portfolios:
            portfolio_data = perf_df[perf_df['portfolio'] == portfolio].copy()
            portfolio_data = portfolio_data.sort_values('period')
            
            sharpe_ratios = portfolio_data['sharpe_ratio']
            color = self.colors.get(portfolio, plt.cm.tab10(len(portfolios)))
            linestyle = '--' if portfolio == 'baseline' else '-'
            
            ax2.plot(portfolio_data['period'], sharpe_ratios,
                    label=portfolio.title(),
                    color=color,
                    linestyle=linestyle,
                    marker='o',
                    markersize=4)
        
        ax2.set_xlabel('Rebalancing Period')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_title('Portfolio Sharpe Ratios Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
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
                             performance_tracker: PerformanceTracker,
                             asset_names: List[str],
                             portfolio_name: str = 'vanilla',
                             save_path: Optional[str] = None) -> None:
        """
        Plot evolution of portfolio weights over time.
        
        Parameters:
        -----------
        performance_tracker : PerformanceTracker
            Tracker with backtest results
        asset_names : List[str]
            Names of assets
        portfolio_name : str
            Which portfolio to plot
        save_path : str, optional
            Path to save the plot
        """
        weights_df = performance_tracker.get_weights_summary(asset_names)
        
        if weights_df.empty:
            return
        
        # Filter for specific portfolio
        portfolio_weights = weights_df[weights_df['portfolio'] == portfolio_name].copy()
        
        if portfolio_weights.empty:
            logging.warning(f"No weights data found for portfolio '{portfolio_name}'")
            return
        
        portfolio_weights = portfolio_weights.sort_values('period')
        
        # Create stacked area plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data for stacking
        periods = portfolio_weights['period'].values
        weight_cols = [col for col in portfolio_weights.columns if col.startswith('weight_')]
        
        # Extract weights matrix
        weights_matrix = portfolio_weights[weight_cols].values.T
        asset_labels = [col.replace('weight_', '') for col in weight_cols]
        
        # Create color map
        colors = plt.cm.tab20(np.linspace(0, 1, len(asset_labels)))
        
        # Create stacked area plot
        ax.stackplot(periods, weights_matrix, 
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
                               performance_tracker: PerformanceTracker,
                               save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive performance summary dashboard.
        
        Parameters:
        -----------
        performance_tracker : PerformanceTracker
            Tracker with backtest results
        save_path : str, optional
            Path to save the plot
        """
        perf_df = performance_tracker.get_performance_summary()
        
        if perf_df.empty:
            return
        
        # Get summary statistics - calculate directly from performance data to ensure all portfolios are included
        portfolios = perf_df['portfolio'].unique()
        stats = {}
        
        for portfolio in portfolios:
            portfolio_data = perf_df[perf_df['portfolio'] == portfolio]
            
            if not portfolio_data.empty:
                stats[portfolio] = {
                    'total_return': portfolio_data['cumulative_return'].iloc[-1] if len(portfolio_data) > 0 else 0,
                    'avg_period_return': portfolio_data['period_return'].mean(),
                    'volatility': portfolio_data['period_return'].std(),
                    'avg_sharpe': portfolio_data['sharpe_ratio'].mean(),
                    'max_drawdown': portfolio_data['max_drawdown'].min(),
                    'success_rate': portfolio_data['optimization_successful'].mean()
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
        
        # Plot 2: Average Sharpe ratios
        avg_sharpes = [stats[p]['avg_sharpe'] for p in portfolios]
        
        bars2 = ax2.bar(portfolios, avg_sharpes, color=colors_list, alpha=0.8)
        ax2.set_title('Average Sharpe Ratio', fontweight='bold')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for bar, value in zip(bars2, avg_sharpes):
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
        total_periods = len(perf_df['period'].unique())
        fig.suptitle(f'Portfolio Performance Summary\n'
                    f'{total_periods} Rebalancing Periods × {self.config.rebalancing_period_days} Days', 
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
                                  performance_tracker: PerformanceTracker,
                                  asset_names: List[str]) -> None:
        """
        Create all standard plots for rebalancing analysis.
        
        Parameters:
        -----------
        performance_tracker : PerformanceTracker
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
        perf_df = performance_tracker.get_performance_summary()
        portfolios = [p for p in perf_df['portfolio'].unique() if p != 'baseline']
        
        for portfolio in portfolios:
            self.plot_weights_evolution(performance_tracker, asset_names, portfolio)
        
        logging.info("Comprehensive rebalancing report completed")