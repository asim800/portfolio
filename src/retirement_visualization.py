#!/usr/bin/env python3
"""
Retirement Visualization - Simple, clear plotting for Monte Carlo results.

Uses standard matplotlib for all visualizations. Each function creates ONE specific plot.
All functions are independent and easy to understand.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional
import os

from retirement_config import RetirementConfig
from retirement_engine import MonteCarloResults


class RetirementVisualizer:
    """
    Simple visualization class for retirement Monte Carlo results.

    Each method creates ONE specific plot. No complex dependencies.
    Uses standard matplotlib - no custom plotting libraries.
    """

    def __init__(self, config: RetirementConfig):
        """
        Initialize visualizer with retirement configuration.

        Parameters:
        -----------
        config : RetirementConfig
            The configuration used for the simulation
        """
        self.config = config

        # Use a clean, professional matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_fan_chart(self,
                      results: MonteCarloResults,
                      save_path: Optional[str] = None,
                      show: bool = True):
        """
        Create a "fan chart" showing percentile bands over time.

        This is the most important plot - shows the range of possible outcomes.

        What this shows:
        - Blue shaded area: where most (90%) simulations end up
        - Dark blue line: median (50th percentile) outcome
        - X-axis: years
        - Y-axis: portfolio value in millions

        Parameters:
        -----------
        results : MonteCarloResults
            Results from run_monte_carlo()
        save_path : Optional[str]
            If provided, save plot to this path
        show : bool
            If True, display the plot
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # X-axis: years from 0 to num_years
        years = np.arange(self.config.num_years)

        # Calculate percentiles at each year
        # portfolio_values_matrix is shape (num_sims, num_years)
        percentile_5 = np.percentile(results.portfolio_values_matrix, 5, axis=0)
        percentile_25 = np.percentile(results.portfolio_values_matrix, 25, axis=0)
        percentile_50 = np.percentile(results.portfolio_values_matrix, 50, axis=0)  # Median
        percentile_75 = np.percentile(results.portfolio_values_matrix, 75, axis=0)
        percentile_95 = np.percentile(results.portfolio_values_matrix, 95, axis=0)

        # Plot shaded areas (lightest to darkest)
        ax.fill_between(years, percentile_5, percentile_95,
                        alpha=0.2, color='skyblue', label='5th-95th percentile (90% of outcomes)')
        ax.fill_between(years, percentile_25, percentile_75,
                        alpha=0.3, color='steelblue', label='25th-75th percentile (50% of outcomes)')

        # Plot median line (most likely outcome)
        ax.plot(years, percentile_50, 'b-', linewidth=2.5, label='Median (50th percentile)')

        # Add horizontal line at zero (depletion level)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Depletion')

        # Add horizontal line at starting value for reference
        ax.axhline(y=self.config.initial_portfolio, color='gray',
                  linestyle=':', alpha=0.5, label='Initial Portfolio')

        # Labels and title
        ax.set_xlabel('Years', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.set_title(f'Retirement Portfolio Projections - {results.num_simulations:,} Simulations\n'
                    f'Success Rate: {results.success_rate:.1%}',
                    fontsize=14, fontweight='bold')

        # Format y-axis as currency (millions)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K')
        )

        # Add legend
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

        # Add grid for easier reading
        ax.grid(True, alpha=0.3)

        # Tight layout to prevent label cutoff
        plt.tight_layout()

        # Save if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        # Show if requested
        if show:
            plt.show()

        return fig, ax

    def plot_summary_dashboard(self,
                              results: MonteCarloResults,
                              save_path: Optional[str] = None,
                              show: bool = True):
        """
        Create a 2x2 dashboard with key statistics.

        This shows:
        1. Success rate (big number)
        2. Distribution of final values (histogram)
        3. Percentile table
        4. Configuration summary

        Parameters:
        -----------
        results : MonteCarloResults
            Results from run_monte_carlo()
        save_path : Optional[str]
            If provided, save plot to this path
        show : bool
            If True, display the plot
        """
        # Create 2x2 grid of subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # ==========================================
        # Plot 1: Success Rate (top-left)
        # ==========================================
        ax1 = axes[0, 0]
        ax1.axis('off')  # Turn off axis for text display

        # Choose color based on success rate
        if results.success_rate >= 0.90:
            color = 'green'
        elif results.success_rate >= 0.70:
            color = 'orange'
        else:
            color = 'red'

        # Display success rate as big number
        ax1.text(0.5, 0.6, f'{results.success_rate:.1%}',
                ha='center', va='center', fontsize=72, fontweight='bold', color=color)
        ax1.text(0.5, 0.35, 'Success Rate',
                ha='center', va='center', fontsize=18)
        ax1.text(0.5, 0.25, f'({int(results.success_rate * results.num_simulations)}/{results.num_simulations} paths succeeded)',
                ha='center', va='center', fontsize=12, style='italic')

        # ==========================================
        # Plot 2: Final Value Distribution (top-right)
        # ==========================================
        ax2 = axes[0, 1]

        # Get successful final values (ignore zeros from failed paths)
        successful_finals = [p.final_value for p in results.paths if p.success]

        if successful_finals:
            # Create histogram
            ax2.hist(successful_finals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')

            # Add median line
            ax2.axvline(results.median_final_value, color='red',
                       linestyle='--', linewidth=2,
                       label=f'Median: ${results.median_final_value/1e6:.1f}M')

            ax2.set_xlabel('Final Portfolio Value ($)', fontsize=11)
            ax2.set_ylabel('Number of Simulations', fontsize=11)
            ax2.set_title('Distribution of Final Values\n(Successful Paths Only)', fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3, axis='y')

            # Format x-axis as currency
            ax2.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f'${x/1e6:.0f}M' if x >= 1e6 else f'${x/1e3:.0f}K')
            )

        # ==========================================
        # Plot 3: Percentile Table (bottom-left)
        # ==========================================
        ax3 = axes[1, 0]
        ax3.axis('off')

        # Create table data
        table_data = [
            ['Percentile', 'Final Value'],
            ['95th (best)', f'${results.percentiles["95th"]/1e6:.2f}M'],
            ['75th', f'${results.percentiles["75th"]/1e6:.2f}M'],
            ['50th (median)', f'${results.percentiles["50th"]/1e6:.2f}M'],
            ['25th', f'${results.percentiles["25th"]/1e6:.2f}M'],
            ['5th (worst)', f'${results.percentiles["5th"]/1e6:.2f}M'],
        ]

        # Create table
        table = ax3.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.4, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)

        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax3.set_title('Final Portfolio Value Percentiles', fontsize=12, pad=20)

        # ==========================================
        # Plot 4: Configuration Summary (bottom-right)
        # ==========================================
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Format portfolio as string
        portfolio_str = ', '.join([f'{t}:{w:.0%}' for t, w in self.config.current_portfolio.items()])

        # Create summary text
        summary_text = f"""
SIMULATION SETUP
{'='*40}
Initial Portfolio:    ${self.config.initial_portfolio:,.0f}
Annual Withdrawal:    ${self.config.annual_withdrawal:,.0f}
Withdrawal Rate:      {self.config.withdrawal_rate:.2%}
Inflation Rate:       {self.config.inflation_rate:.1%}
Time Horizon:         {self.config.num_years} years
Portfolio:            {portfolio_str}
Simulations:          {results.num_simulations:,}

RESULTS SUMMARY
{'='*40}
Success Rate:         {results.success_rate:.1%}
Median Final Value:   ${results.median_final_value:,.0f}
Mean Final Value:     ${results.mean_final_value:,.0f}
Std Deviation:        ${results.std_final_value:,.0f}
        """

        ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')

        # Overall title
        fig.suptitle('Retirement Monte Carlo Simulation - Summary Dashboard',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        # Show if requested
        if show:
            plt.show()

        return fig, axes

    def plot_sample_paths(self,
                         results: MonteCarloResults,
                         num_paths: int = 50,
                         save_path: Optional[str] = None,
                         show: bool = True):
        """
        Plot individual simulation paths (spaghetti plot).

        This shows the actual trajectories of individual simulations.
        Helps visualize the variability.

        Parameters:
        -----------
        results : MonteCarloResults
            Results from run_monte_carlo()
        num_paths : int
            Number of individual paths to show (default: 50)
        save_path : Optional[str]
            If provided, save plot to this path
        show : bool
            If True, display the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        years = np.arange(self.config.num_years)

        # Plot a sample of individual paths
        num_to_plot = min(num_paths, results.num_simulations)

        for i in range(num_to_plot):
            path = results.paths[i]

            # Color based on success
            color = 'green' if path.success else 'red'
            alpha = 0.3

            # Plot this path's values
            ax.plot(years[:len(path.portfolio_values)],
                   path.portfolio_values,
                   color=color, alpha=alpha, linewidth=0.5)

        # Add median line for reference
        median_values = np.percentile(results.portfolio_values_matrix, 50, axis=0)
        ax.plot(years, median_values, 'b-', linewidth=2.5, label='Median', zorder=10)

        # Add depletion line
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5,
                  alpha=0.5, label='Depletion', zorder=10)

        # Labels
        ax.set_xlabel('Years', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.set_title(f'Sample Portfolio Paths (showing {num_to_plot} of {results.num_simulations})\n'
                    f'Green = Success, Red = Depletion',
                    fontsize=14, fontweight='bold')

        # Format y-axis
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K')
        )

        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        if show:
            plt.show()

        return fig, ax

    def plot_all(self,
                results: MonteCarloResults,
                output_dir: str = '../plots/retirement/',
                show: bool = True):
        """
        Generate all plots and save to directory.

        This is a convenience function that creates all three plots.

        Parameters:
        -----------
        results : MonteCarloResults
            Results from run_monte_carlo()
        output_dir : str
            Directory to save plots
        show : bool
            If True, display plots
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nGenerating visualizations in: {output_dir}")

        # Create each plot
        print("  1/3: Creating fan chart...")
        self.plot_fan_chart(
            results,
            save_path=os.path.join(output_dir, 'fan_chart.png'),
            show=False  # Don't show individual plots
        )

        print("  2/3: Creating summary dashboard...")
        self.plot_summary_dashboard(
            results,
            save_path=os.path.join(output_dir, 'summary_dashboard.png'),
            show=False
        )

        print("  3/3: Creating sample paths plot...")
        self.plot_sample_paths(
            results,
            num_paths=100,
            save_path=os.path.join(output_dir, 'sample_paths.png'),
            show=False
        )

        print(f"\n✓ All plots saved to: {output_dir}")

        # Show all plots at once if requested
        if show:
            plt.show()
