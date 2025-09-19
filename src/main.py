#!/usr/bin/env python3
"""
Main entry point for Options Analysis System.
"""

import sys, os
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
import matplotlib.pyplot as plt
import pickle

import ipdb
from typing import Tuple, Optional, Dict, Any

# IPython autoreload for module development
try:
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic('load_ext', 'autoreload')
        ipython.run_line_magic('autoreload', '2')
except (ImportError, AttributeError):
    pass

import matplotlib
matplotlib.use('Qt5Agg')  # Interactive backend for plotting

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import performance analysis functions
try:
    from performance_analysis import (
        annualized_return, annualized_standard_deviation, max_drawdown, 
        gain_to_pain_ratio, calmar_ratio, sharpe_ratio, sortino_ratio
    )
except ImportError:
    logging.warning("Could not import performance_analysis module - using basic metrics only")


def optimize_portfolio_vanilla(mean_returns: np.ndarray, 
                             cov_matrix: np.ndarray, 
                             risk_aversion: float = 1.0,
                             long_only: bool = True) -> Dict[str, Any]:
    """
    Vanilla mean-variance portfolio optimization using cvxpy.
    
    Parameters:
    -----------
    mean_returns : np.ndarray
        Expected returns for each asset
    cov_matrix : np.ndarray
        Covariance matrix of asset returns
    risk_aversion : float
        Risk aversion parameter (higher = more conservative)
    long_only : bool
        If True, constrains weights to be non-negative
        
    Returns:
    --------
    Dict containing optimization results
    """
    n_assets = len(mean_returns)
    
    # Define optimization variables
    weights = cp.Variable(n_assets)
    
    # Objective: Maximize expected return - risk penalty
    portfolio_return = mean_returns @ weights
    portfolio_variance = cp.quad_form(weights, cov_matrix)
    
    # Objective function (utility maximization)
    objective = cp.Maximize(portfolio_return - 0.5 * risk_aversion * portfolio_variance)
    
    # Constraints
    constraints = [cp.sum(weights) == 1]  # Weights sum to 1
    if long_only:
        constraints.append(weights >= 0)  # Long-only constraint
    
    # Solve optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    # Process results
    logging.debug(f"Vanilla optimization status: {problem.status}")
    if problem.status == cp.OPTIMAL:
        optimal_weights = weights.value
        expected_return = (mean_returns @ optimal_weights) * 252  # Annualized
        volatility = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights) * np.sqrt(252)  # Annualized
        sharpe_ratio = expected_return / volatility if volatility > 0 else 0
        
        return {
            'status': 'optimal',
            'weights': optimal_weights,
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'objective_value': problem.value
        }
    else:
        logging.error(f"Vanilla optimization failed with status: {problem.status}")
        return {
            'status': 'failed',
            'weights': None,
            'expected_return': None,
            'volatility': None,
            'sharpe_ratio': None,
            'objective_value': None
        }


def optimize_portfolio_robust(mean_returns: np.ndarray, 
                            cov_matrix: np.ndarray, 
                            risk_aversion: float = 1.0,
                            uncertainty_level: float = 0.1,
                            long_only: bool = True) -> Dict[str, Any]:
    """
    Robust portfolio optimization using cvxpy with ellipsoidal uncertainty sets.
    
    Parameters:
    -----------
    mean_returns : np.ndarray
        Expected returns for each asset
    cov_matrix : np.ndarray
        Covariance matrix of asset returns
    risk_aversion : float
        Risk aversion parameter (higher = more conservative)
    uncertainty_level : float
        Level of uncertainty in mean returns (e.g., 0.1 = 10% uncertainty)
    long_only : bool
        If True, constrains weights to be non-negative
        
    Returns:
    --------
    Dict containing optimization results
    """
    n_assets = len(mean_returns)
    
    # Define optimization variables
    weights = cp.Variable(n_assets)
    
    # Robust parameters - ellipsoidal uncertainty set for mean returns
    mean_uncertainty = uncertainty_level * np.abs(mean_returns)
    
    # Worst-case expected return under uncertainty
    robust_return = mean_returns @ weights - cp.norm(cp.multiply(mean_uncertainty, weights), 2)
    
    # Portfolio variance (same as vanilla)
    portfolio_variance = cp.quad_form(weights, cov_matrix)
    
    # Robust objective function
    objective = cp.Maximize(robust_return - 0.5 * risk_aversion * portfolio_variance)
    
    # Constraints
    constraints = [cp.sum(weights) == 1]  # Weights sum to 1
    if long_only:
        constraints.append(weights >= 0)  # Long-only constraint
    
    # Solve optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    # Process results
    logging.debug(f"Robust optimization status: {problem.status}")
    if problem.status == cp.OPTIMAL:
        optimal_weights = weights.value
        expected_return_nominal = (mean_returns @ optimal_weights) * 252  # Annualized (nominal)
        volatility = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights) * np.sqrt(252)  # Annualized
        sharpe_ratio = expected_return_nominal / volatility if volatility > 0 else 0
        
        return {
            'status': 'optimal',
            'weights': optimal_weights,
            'expected_return': expected_return_nominal,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'objective_value': problem.value,
            'uncertainty_level': uncertainty_level
        }
    else:
        logging.error(f"Robust optimization failed with status: {problem.status}")
        return {
            'status': 'failed',
            'weights': None,
            'expected_return': None,
            'volatility': None,
            'sharpe_ratio': None,
            'objective_value': None,
            'uncertainty_level': uncertainty_level
        }


def compare_and_visualize_portfolios(portfolios: Dict[str, Dict[str, Any]], 
                                   asset_names: list,
                                   save_path: str = '../plots/portfolio_comparison.png',
                                   show_interactive: bool = False) -> None:
    """
    Compare multiple portfolios and create visualizations.
    
    Parameters:
    -----------
    portfolios : Dict[str, Dict[str, Any]]
        Dictionary of portfolio results from optimization functions
    asset_names : list
        List of asset names/tickers
    save_path : str
        Path to save the comparison plot
    show_interactive : bool, optional
        Whether to show plot interactively (default: False)
    """
    # Filter out failed optimizations
    valid_portfolios = {name: port for name, port in portfolios.items() 
                       if port['status'] == 'optimal'}
    
    if len(valid_portfolios) < 2:
        logging.warning("Need at least 2 successful optimizations for comparison")
        return
    
    # Create comparison DataFrame
    weights_data = {}
    for name, portfolio in valid_portfolios.items():
        weights_data[name] = portfolio['weights']
    
    comparison_df = pd.DataFrame(weights_data, index=asset_names)
    
    # Sort by first portfolio's weights for better visualization
    first_portfolio = list(valid_portfolios.keys())[0]
    comparison_df = comparison_df.sort_values(first_portfolio, ascending=False)
    
    # Log comparison table
    logging.info("=== Portfolio Weight Comparison ===")
    header = f"{'Ticker':<8} " + " ".join([f"{name:<8}" for name in valid_portfolios.keys()])
    logging.info(header)
    logging.info("-" * len(header))
    
    for ticker in comparison_df.index:
        row = f"{ticker:<8} "
        for portfolio_name in valid_portfolios.keys():
            weight = comparison_df.loc[ticker, portfolio_name]
            row += f"{weight:<8.2%} "
        logging.info(row)
    
    # Create visualization
    n_portfolios = len(valid_portfolios)
    fig_height = max(12, n_portfolios * 2)
    fig, axes = plt.subplots(2, 2, figsize=(15, fig_height))
    
    # Plot 1: Portfolio weights comparison
    x_pos = np.arange(len(comparison_df))
    width = 0.8 / n_portfolios
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'lightpink']
    
    for i, (portfolio_name, _) in enumerate(valid_portfolios.items()):
        axes[0,0].bar(x_pos + i * width - width * (n_portfolios-1)/2, 
                     comparison_df[portfolio_name], width, 
                     label=portfolio_name, alpha=0.8, 
                     color=colors[i % len(colors)])
    
    axes[0,0].set_xlabel('Assets')
    axes[0,0].set_ylabel('Portfolio Weights')
    axes[0,0].set_title('Portfolio Weights Comparison')
    axes[0,0].set_xticks(x_pos)
    axes[0,0].set_xticklabels(comparison_df.index, rotation=45)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Performance metrics comparison
    metrics = ['expected_return', 'volatility', 'sharpe_ratio']
    portfolio_names = list(valid_portfolios.keys())
    
    x_pos_metrics = np.arange(len(metrics))
    width_metrics = 0.8 / n_portfolios
    
    for i, portfolio_name in enumerate(portfolio_names):
        portfolio = valid_portfolios[portfolio_name]
        values = [portfolio['expected_return'], portfolio['volatility'], portfolio['sharpe_ratio']]
        axes[0,1].bar(x_pos_metrics + i * width_metrics - width_metrics * (n_portfolios-1)/2,
                     values, width_metrics, label=portfolio_name, 
                     alpha=0.8, color=colors[i % len(colors)])
    
    axes[0,1].set_xlabel('Metrics')
    axes[0,1].set_ylabel('Value')
    axes[0,1].set_title('Portfolio Performance Metrics')
    axes[0,1].set_xticks(x_pos_metrics)
    axes[0,1].set_xticklabels(['Return', 'Volatility', 'Sharpe'])
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3 & 4: Pie charts for each portfolio (up to 2)
    for i, (portfolio_name, portfolio) in enumerate(list(valid_portfolios.items())[:2]):
        ax = axes[1, i]
        weights = comparison_df[portfolio_name]
        nonzero_weights = weights[weights > 0.001]
        
        ax.pie(nonzero_weights, labels=nonzero_weights.index, autopct='%1.1f%%', 
               startangle=90)
        ax.set_title(f'{portfolio_name} Portfolio Allocation')
    
    # If only one portfolio for pie charts, hide the second pie chart
    if len(valid_portfolios) == 1:
        axes[1,1].axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Portfolio comparison plot saved to {save_path}")
    
    # Display the plot
    if show_interactive:
        plt.show(block=False)  # Non-blocking show - allows multiple plots to stay open
    else:
        plt.close()
    
    # Summary statistics
    logging.info("=== Portfolio Summary Statistics ===")
    for portfolio_name, portfolio in valid_portfolios.items():
        weights = portfolio['weights']
        logging.info(f"\n{portfolio_name} Portfolio:")
        logging.info(f"  Expected Return: {portfolio['expected_return']:.2%}")
        logging.info(f"  Volatility: {portfolio['volatility']:.2%}")
        logging.info(f"  Sharpe Ratio: {portfolio['sharpe_ratio']:.3f}")
        
        # Portfolio concentration (Herfindahl Index)
        hhi = np.sum(weights ** 2)
        logging.info(f"  Concentration (HHI): {hhi:.4f}")
        
        # Number of holdings
        n_holdings = np.sum(weights > 0.001)
        logging.info(f"  Number of holdings (>0.1%): {n_holdings}")


def run_dynamic_rebalancing(returns_df: pd.DataFrame, 
                          baseline_weights: np.ndarray,
                          config: Optional['RebalancingConfig'] = None) -> 'PerformanceTracker':
    """
    Run dynamic portfolio rebalancing backtest.
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        Daily returns data with assets as columns
    baseline_weights : np.ndarray
        Baseline portfolio weights
    config : RebalancingConfig, optional
        Configuration for rebalancing. If None, uses default config.
        
    Returns:
    --------
    PerformanceTracker with backtest results
    """
    from config import RebalancingConfig, DEFAULT_CONFIG
    from rebalancing_engine import RebalancingEngine
    from rebalancing_visualization import RebalancingVisualizer
    
    # Use provided config or default
    if config is None:
        config = DEFAULT_CONFIG
    
    logging.info("=== Starting Dynamic Portfolio Rebalancing ===")
    logging.info(f"Rebalancing period: {config.rebalancing_period_days} days")
    logging.info(f"Optimization methods: {config.optimization_methods}")
    logging.info(f"Minimum history: {config.min_history_periods} periods")
    
    # Initialize rebalancing engine
    engine = RebalancingEngine(config)
    
    # Load data
    engine.load_data(returns_df, baseline_weights)
    
    # Run backtest
    performance_tracker = engine.run_backtest()
    
    # Save results
    if config.save_results:
        engine.save_results()
    
    # Create visualizations
    if config.save_plots:
        visualizer = RebalancingVisualizer(config)
        visualizer.create_comprehensive_report(performance_tracker, returns_df.columns.tolist())
    
    # Print summary statistics
    stats = engine.get_summary_statistics()
    logging.info("=== Rebalancing Summary Statistics ===")
    for portfolio_name, portfolio_stats in stats.items():
        logging.info(f"\n{portfolio_name.upper()} Portfolio:")
        logging.info(f"  Total Return: {portfolio_stats['total_return']:.2%}")
        logging.info(f"  Avg Period Return: {portfolio_stats['avg_period_return']:.2%}")
        logging.info(f"  Volatility: {portfolio_stats['volatility']:.2%}")
        logging.info(f"  Avg Sharpe Ratio: {portfolio_stats['avg_sharpe']:.3f}")
        logging.info(f"  Max Drawdown: {portfolio_stats['max_drawdown']:.2%}")
        if 'success_rate' in portfolio_stats:
            logging.info(f"  Optimization Success Rate: {portfolio_stats['success_rate']:.1%}")
    
    return performance_tracker


def create_portfolio_returns(weights: np.ndarray, returns_df: pd.DataFrame, portfolio_name: str = "Portfolio") -> pd.Series:
    """
    Calculate portfolio returns from weights and individual asset returns.
    
    Parameters:
    -----------
    weights : np.ndarray
        Portfolio weights
    returns_df : pd.DataFrame  
        Individual asset returns
    portfolio_name : str
        Name for the portfolio series
        
    Returns:
    --------
    pd.Series with portfolio returns
    """
    portfolio_returns = (returns_df * weights).sum(axis=1)
    portfolio_returns.name = portfolio_name
    return portfolio_returns


def create_equal_weight_portfolio(returns_df: pd.DataFrame) -> pd.Series:
    """
    Create equally weighted portfolio returns.
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        Individual asset returns
        
    Returns:
    --------
    pd.Series with equal-weight portfolio returns
    """
    equal_weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
    return create_portfolio_returns(equal_weights, returns_df, "Equal Weight")


def plot_individual_portfolio_analysis(portfolio_weights: np.ndarray,
                                     returns_df: pd.DataFrame,
                                     portfolio_name: str,
                                     save_path: str = None,
                                     show_interactive: bool = False) -> None:
    """
    Create individual portfolio analysis plot with weights and performance metrics.
    
    Parameters:
    -----------
    portfolio_weights : np.ndarray
        Portfolio weights
    returns_df : pd.DataFrame
        Asset returns data
    portfolio_name : str
        Name of the portfolio
    save_path : str, optional
        Path to save the plot
    show_interactive : bool, optional
        Whether to show plot interactively (default: False)
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Portfolio weights
    nonzero_weights = portfolio_weights[portfolio_weights > 0.001]
    nonzero_tickers = returns_df.columns[portfolio_weights > 0.001]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(nonzero_weights)))
    ax1.pie(nonzero_weights, labels=nonzero_tickers, autopct='%1.1f%%', 
            startangle=90, colors=colors)
    ax1.set_title(f'{portfolio_name} - Weight Allocation')
    
    # Plot 2: Top holdings bar chart
    weight_series = pd.Series(portfolio_weights, index=returns_df.columns).sort_values(ascending=False)
    top_holdings = weight_series.head(10)
    
    ax2.barh(range(len(top_holdings)), top_holdings.values, color='skyblue')
    ax2.set_yticks(range(len(top_holdings)))
    ax2.set_yticklabels(top_holdings.index)
    ax2.set_xlabel('Weight')
    ax2.set_title(f'{portfolio_name} - Top 10 Holdings')
    ax2.grid(True, alpha=0.3)
    
    # Calculate portfolio returns for metrics
    portfolio_returns = create_portfolio_returns(portfolio_weights, returns_df, portfolio_name)
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    
    # Plot 3: Cumulative returns
    ax3.plot(portfolio_cumulative.index, portfolio_cumulative.values, 
             linewidth=2, label=portfolio_name, color='darkblue')
    ax3.set_title(f'{portfolio_name} - Cumulative Returns')
    ax3.set_ylabel('Cumulative Return')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Calculate portfolio weight sum and beta
    weight_sum = np.sum(portfolio_weights)
    
    # Calculate portfolio beta (vs market proxy)
    def calculate_portfolio_beta(portfolio_returns, market_returns):
        """Calculate portfolio beta relative to market"""
        if len(portfolio_returns) != len(market_returns):
            return np.nan
        covariance = np.cov(portfolio_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        return covariance / market_variance if market_variance > 0 else np.nan
    
    # Find market proxy (SPY, SPX, or equal-weighted portfolio)
    if 'SPY' in returns_df.columns:
        market_returns = returns_df['SPY']
        market_name = 'SPY'
    elif 'SPX' in returns_df.columns:
        market_returns = returns_df['SPX']
        market_name = 'SPX'
    else:
        # Use equal-weighted portfolio as market proxy
        equal_weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
        market_returns = (returns_df * equal_weights).sum(axis=1)
        market_name = 'Equal-Weight Index'
    
    portfolio_beta = calculate_portfolio_beta(portfolio_returns.values, market_returns.values)
    
    # Plot 4: Performance metrics using your performance_analysis functions
    try:
        # Create DataFrame for performance analysis (needs price series)
        portfolio_prices = portfolio_cumulative.to_frame()
        
        # Calculate metrics using your functions
        ann_ret = annualized_return(portfolio_prices)
        ann_std = annualized_standard_deviation(portfolio_prices) 
        max_dd = max_drawdown(portfolio_prices)
        calmar = calmar_ratio(portfolio_prices)
        sharpe = sharpe_ratio(portfolio_prices, RF=0.02)  # 2% risk-free rate
        
        metrics = {
            'Weight Sum': f"{weight_sum:.3f}",
            'Annualized Return': f"{ann_ret.iloc[0, 1]:.2%}",
            'Volatility': f"{ann_std.iloc[0, 1]:.2%}",
            'Beta (vs {})'.format(market_name): f"{portfolio_beta:.3f}" if not np.isnan(portfolio_beta) else "N/A",
            'Max Drawdown': f"{max_dd.iloc[0, 1]:.2%}",
            'Calmar Ratio': f"{calmar.iloc[0, 1]:.3f}",
            'Sharpe Ratio': f"{sharpe.iloc[0, 1]:.3f}"
        }
        
    except (NameError, AttributeError, IndexError):
        # Fallback to basic metrics if performance_analysis not available
        ann_return = (portfolio_cumulative.iloc[-1] ** (252/len(portfolio_cumulative)) - 1)
        ann_vol = portfolio_returns.std() * np.sqrt(252)
        rolling_max = portfolio_cumulative.expanding().max()
        drawdown = (portfolio_cumulative / rolling_max - 1).min()
        sharpe_basic = ann_return / ann_vol if ann_vol > 0 else 0
        
        metrics = {
            'Weight Sum': f"{weight_sum:.3f}",
            'Annualized Return': f"{ann_return:.2%}",
            'Volatility': f"{ann_vol:.2%}",
            'Beta (vs {})'.format(market_name): f"{portfolio_beta:.3f}" if not np.isnan(portfolio_beta) else "N/A",
            'Max Drawdown': f"{drawdown:.2%}",
            'Sharpe Ratio': f"{sharpe_basic:.3f}"
        }
    
    # Display metrics as text
    ax4.axis('off')
    metrics_text = '\n'.join([f'{k}: {v}' for k, v in metrics.items()])
    ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax4.set_title(f'{portfolio_name} - Performance Metrics')
    
    plt.suptitle(f'{portfolio_name} Portfolio Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Individual portfolio plot saved to {save_path}")
    
    if show_interactive:
        plt.show(block=False)  # Non-blocking show - allows multiple plots to stay open
    else:
        plt.close()


def plot_portfolio_comparison(desired_weights: np.ndarray,
                            baseline_weights: np.ndarray,
                            returns_df: pd.DataFrame,
                            desired_name: str = "Desired",
                            baseline_name: str = "Baseline",
                            save_path: str = None,
                            show_interactive: bool = False) -> None:
    """
    Compare two portfolios with cumulative returns and risk metrics.
    
    Parameters:
    -----------
    desired_weights : np.ndarray
        Weights for desired portfolio
    baseline_weights : np.ndarray  
        Weights for baseline portfolio
    returns_df : pd.DataFrame
        Asset returns data
    desired_name : str
        Name for desired portfolio
    baseline_name : str
        Name for baseline portfolio
    save_path : str, optional
        Path to save the plot
    show_interactive : bool, optional
        Whether to show plot interactively (default: False)
    """
    # Calculate portfolio returns
    desired_returns = create_portfolio_returns(desired_weights, returns_df, desired_name)
    baseline_returns = create_portfolio_returns(baseline_weights, returns_df, baseline_name)
    
    # Calculate cumulative returns
    desired_cumulative = (1 + desired_returns).cumprod()
    baseline_cumulative = (1 + baseline_returns).cumprod()
    
    # Create subplot layout: main plot + risk metrics + drawdown + metrics table
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 2, height_ratios=[2, 1, 1, 0.8], width_ratios=[2, 1], hspace=0.3, wspace=0.3)
    
    # Main plot: Cumulative returns (spans both columns)
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.plot(desired_cumulative.index, desired_cumulative.values, 
                linewidth=2, label=desired_name, color='darkblue')
    ax_main.plot(baseline_cumulative.index, baseline_cumulative.values, 
                linewidth=2, label=baseline_name, color='darkred', linestyle='--')
    
    ax_main.set_title(f'{desired_name} vs {baseline_name} - Cumulative Returns', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('Cumulative Return')
    ax_main.grid(True, alpha=0.3)
    ax_main.legend()
    
    # Calculate portfolio weights totals
    desired_total_weight = np.sum(desired_weights)
    baseline_total_weight = np.sum(baseline_weights)
    
    # Calculate portfolio beta (vs market proxy - assuming SPY is in the returns if available)
    def calculate_portfolio_beta(portfolio_returns, market_returns):
        """Calculate portfolio beta relative to market"""
        if len(portfolio_returns) != len(market_returns):
            return np.nan
        covariance = np.cov(portfolio_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        return covariance / market_variance if market_variance > 0 else np.nan
    
    # Try to find market proxy (SPY, SPX, or use equal-weighted portfolio as proxy)
    market_proxy_returns = None
    if 'SPY' in returns_df.columns:
        market_proxy_returns = returns_df['SPY']
        market_name = 'SPY'
    elif 'SPX' in returns_df.columns:
        market_proxy_returns = returns_df['SPX']
        market_name = 'SPX'
    else:
        # Use equal-weighted portfolio as market proxy
        equal_weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
        market_proxy_returns = (returns_df * equal_weights).sum(axis=1)
        market_name = 'Equal-Weight Index'
    
    desired_beta = calculate_portfolio_beta(desired_returns.values, market_proxy_returns.values)
    baseline_beta = calculate_portfolio_beta(baseline_returns.values, market_proxy_returns.values)
    
    # Calculate comprehensive metrics for both portfolios
    try:
        # Use your performance analysis functions
        combined_prices = pd.concat([desired_cumulative, baseline_cumulative], axis=1)
        
        ann_ret = annualized_return(combined_prices)
        ann_std = annualized_standard_deviation(combined_prices)
        max_dd = max_drawdown(combined_prices)
        sharpe = sharpe_ratio(combined_prices, RF=0.02)
        calmar = calmar_ratio(combined_prices)
        gain_pain = gain_to_pain_ratio(combined_prices)
        
        metrics_data = {
            'Total Weight': [desired_total_weight, baseline_total_weight],
            'Ann. Return': [ann_ret.iloc[0, 1], ann_ret.iloc[1, 1]],
            'Volatility': [ann_std.iloc[0, 1], ann_std.iloc[1, 1]], 
            'Beta': [desired_beta, baseline_beta],
            'Sharpe Ratio': [sharpe.iloc[0, 1], sharpe.iloc[1, 1]],
            'Max Drawdown': [abs(max_dd.iloc[0, 1]), abs(max_dd.iloc[1, 1])],
            'Calmar Ratio': [calmar.iloc[0, 1], calmar.iloc[1, 1]],
            'Gain/Pain': [gain_pain.iloc[0, 1], gain_pain.iloc[1, 1]]
        }
        
        logging.info(f"Portfolio beta calculated vs {market_name}: Desired={desired_beta:.3f}, Baseline={baseline_beta:.3f}")
        
    except (NameError, AttributeError, IndexError):
        # Fallback metrics
        desired_ann_ret = desired_cumulative.iloc[-1] ** (252/len(desired_cumulative)) - 1
        baseline_ann_ret = baseline_cumulative.iloc[-1] ** (252/len(baseline_cumulative)) - 1
        desired_vol = desired_returns.std() * np.sqrt(252)
        baseline_vol = baseline_returns.std() * np.sqrt(252)
        desired_sharpe = desired_ann_ret / desired_vol if desired_vol > 0 else 0
        baseline_sharpe = baseline_ann_ret / baseline_vol if baseline_vol > 0 else 0
        
        # Calculate basic drawdown
        desired_rolling_max = desired_cumulative.expanding().max()
        baseline_rolling_max = baseline_cumulative.expanding().max()
        desired_dd = (desired_cumulative / desired_rolling_max - 1).min()
        baseline_dd = (baseline_cumulative / baseline_rolling_max - 1).min()
        
        metrics_data = {
            'Total Weight': [desired_total_weight, baseline_total_weight],
            'Ann. Return': [desired_ann_ret, baseline_ann_ret],
            'Volatility': [desired_vol, baseline_vol],
            'Beta': [desired_beta, baseline_beta],
            'Sharpe Ratio': [desired_sharpe, baseline_sharpe],
            'Max Drawdown': [abs(desired_dd), abs(baseline_dd)]
        }
        
        logging.info(f"Portfolio beta calculated vs {market_name}: Desired={desired_beta:.3f}, Baseline={baseline_beta:.3f}")
    
    # Subplot 1: Performance metrics bar chart
    ax_metrics = fig.add_subplot(gs[1, 0])
    
    x_pos = np.arange(len(metrics_data))
    width = 0.35
    
    desired_vals = [metrics_data[metric][0] for metric in metrics_data.keys()]
    baseline_vals = [metrics_data[metric][1] for metric in metrics_data.keys()]
    
    ax_metrics.bar(x_pos - width/2, desired_vals, width, label=desired_name, color='darkblue', alpha=0.7)
    ax_metrics.bar(x_pos + width/2, baseline_vals, width, label=baseline_name, color='darkred', alpha=0.7)
    
    ax_metrics.set_xlabel('Metrics')
    ax_metrics.set_ylabel('Value')
    ax_metrics.set_title('Performance Metrics Comparison')
    ax_metrics.set_xticks(x_pos)
    ax_metrics.set_xticklabels(metrics_data.keys(), rotation=45, ha='right')
    ax_metrics.legend()
    ax_metrics.grid(True, alpha=0.3)
    
    # Subplot 2: Risk Metrics Table
    ax_table = fig.add_subplot(gs[1, 1])
    ax_table.axis('off')
    
    # Create table data
    table_data = []
    for metric, values in metrics_data.items():
        if metric == 'Total Weight':
            desired_str = f"{values[0]:.3f}"
            baseline_str = f"{values[1]:.3f}"
        elif metric == 'Beta':
            desired_str = f"{values[0]:.3f}" if not np.isnan(values[0]) else "N/A"
            baseline_str = f"{values[1]:.3f}" if not np.isnan(values[1]) else "N/A"
        elif 'Ratio' in metric:
            desired_str = f"{values[0]:.3f}"
            baseline_str = f"{values[1]:.3f}"
        elif 'Return' in metric:
            desired_str = f"{values[0]:.2%}"
            baseline_str = f"{values[1]:.2%}"
        else:
            desired_str = f"{values[0]:.2%}"
            baseline_str = f"{values[1]:.2%}"
        
        table_data.append([metric, desired_str, baseline_str])
    
    table = ax_table.table(cellText=table_data,
                          colLabels=['Metric', desired_name, baseline_name],
                          cellLoc='center',
                          loc='center',
                          bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#40466e')
                cell.set_text_props(weight='bold', color='white')
            else:
                if j == 1:  # Desired portfolio column
                    cell.set_facecolor('#e6f2ff')
                elif j == 2:  # Baseline portfolio column
                    cell.set_facecolor('#ffe6e6')
                else:
                    cell.set_facecolor('#f0f0f0')
    
    ax_table.set_title('Risk Metrics Summary', fontweight='bold', pad=20)
    
    # Subplot 3: Rolling drawdown comparison (spans both columns)
    ax_rolling = fig.add_subplot(gs[2, :])
    
    # Calculate rolling drawdown
    desired_rolling_max = desired_cumulative.expanding().max()
    baseline_rolling_max = baseline_cumulative.expanding().max()
    desired_drawdown = desired_cumulative / desired_rolling_max - 1
    baseline_drawdown = baseline_cumulative / baseline_rolling_max - 1
    
    ax_rolling.fill_between(desired_drawdown.index, desired_drawdown.values, 0, 
                           alpha=0.3, color='darkblue', label=f'{desired_name} Drawdown')
    ax_rolling.fill_between(baseline_drawdown.index, baseline_drawdown.values, 0, 
                           alpha=0.3, color='darkred', label=f'{baseline_name} Drawdown')
    
    ax_rolling.set_title('Rolling Drawdown Comparison')
    ax_rolling.set_ylabel('Drawdown')
    ax_rolling.set_xlabel('Date')
    ax_rolling.legend()
    ax_rolling.grid(True, alpha=0.3)
    
    # Subplot 4: Additional risk analysis (spans both columns)
    ax_risk = fig.add_subplot(gs[3, :])
    
    # Calculate rolling volatility (30-day window)
    rolling_window = 30
    desired_rolling_vol = desired_returns.rolling(rolling_window).std() * np.sqrt(252)
    baseline_rolling_vol = baseline_returns.rolling(rolling_window).std() * np.sqrt(252)
    
    ax_risk.plot(desired_rolling_vol.index, desired_rolling_vol.values, 
                linewidth=1.5, label=f'{desired_name} Vol (30d)', color='darkblue', alpha=0.8)
    ax_risk.plot(baseline_rolling_vol.index, baseline_rolling_vol.values, 
                linewidth=1.5, label=f'{baseline_name} Vol (30d)', color='darkred', alpha=0.8)
    
    ax_risk.set_title('Rolling Volatility Comparison (30-day)')
    ax_risk.set_ylabel('Annualized Volatility')
    ax_risk.set_xlabel('Date')
    ax_risk.legend()
    ax_risk.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Portfolio comparison plot saved to {save_path}")
    
    if show_interactive:
        plt.show(block=False)  # Non-blocking show - allows multiple plots to stay open
    else:
        plt.close()


def main():
    """Main entry point for the application."""
    # Set up logging to show important messages on stdout
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # Output to stdout
            logging.FileHandler('../logs/options_analysis.log', 'a')  # Append to log file
        ]
    )
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Options Analysis System')
    parser.add_argument('--file', '--tickers', dest='ticker_file', 
                       help='Ticker file to use (default: tickersv1.txt)')
    parser.add_argument('--ticker', dest='single_ticker',
                       help='Analyze a single ticker instead of a file')
    
    args = parser.parse_args()

    ticker_file = '../tickers.txt'
    start_date = '2024-01-01'
    end_date = '2024-12-31'

    try:
        # Read the ticker file into a DataFrame (with headers)
        tickers_df = pd.read_csv(ticker_file, skipinitialspace=True)
        # Clean column names (remove any extra spaces)
        tickers_df.columns = tickers_df.columns.str.strip()
        
        # Rename columns for consistency
        if 'ticker' in tickers_df.columns:
            tickers_df = tickers_df.rename(columns={'ticker': 'Symbol'})
        if 'weights' in tickers_df.columns:
            tickers_df = tickers_df.rename(columns={'weights': 'Weight'})
        
        logging.info(f"Loaded CSV with columns: {list(tickers_df.columns)}")
        logging.info(f"First few rows:\n{tickers_df.head()}")
        
        # Remove empty rows and ensure weights are numeric
        tickers_df = tickers_df.dropna()
        tickers_df['Weight'] = pd.to_numeric(tickers_df['Weight'], errors='coerce')
        tickers_df = tickers_df.dropna()  # Remove rows where weight conversion failed
        
        # Normalize weights to sum to 1
        tickers_df['Weight'] = tickers_df['Weight'] / tickers_df['Weight'].sum()
        
        logging.info(f"Successfully loaded {len(tickers_df)} tickers from {ticker_file}")
        logging.info(f"Weights sum to: {tickers_df['Weight'].sum():.4f}")

        # Create pickle filename with start and end dates
        pickle_filename = f"../data/price_data_{start_date}_{end_date}.pkl"
        os.makedirs(os.path.dirname(pickle_filename), exist_ok=True)
        
        # Try to load existing data from pickle file
        if os.path.exists(pickle_filename):
            logging.info(f"Loading price data from pickle file: {pickle_filename}")
            try:
                with open(pickle_filename, 'rb') as f:
                    all_prices_df = pickle.load(f)
                logging.info(f"Successfully loaded price data from pickle ({len(all_prices_df.columns)//2} tickers)")
                
                # Verify that we have all the tickers we need
                existing_tickers = set(all_prices_df.columns.get_level_values(0))
                requested_tickers = set(tickers_df['Symbol'])
                missing_tickers = requested_tickers - existing_tickers
                
                if missing_tickers:
                    logging.warning(f"Missing tickers in pickle file: {missing_tickers}")
                    logging.info("Will fetch missing tickers and update pickle file")
                    
                    # Initialize DataFrame if not loaded properly
                    if all_prices_df.empty:
                        columns = pd.MultiIndex.from_tuples([], names=['Ticker', 'Metric'])
                        all_prices_df = pd.DataFrame(columns=columns)
                    
                    # Fetch only missing tickers
                    for ticker in missing_tickers:
                        try:
                            logging.info(f"Fetching missing ticker: {ticker}")
                            ticker_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')[['Close', 'Volume']]
                            # Add the ticker's price data using multi-index columns (ticker, metric)
                            for col in ['Close', 'Volume']:
                                all_prices_df[(ticker, col)] = ticker_data[col]
                            logging.info(f"Successfully downloaded daily price data for {ticker} ({len(ticker_data)} days)")
                        except Exception as e:
                            logging.warning(f"Failed to download data for {ticker}: {e}")
                            continue
                    
                    # Save updated data back to pickle
                    with open(pickle_filename, 'wb') as f:
                        pickle.dump(all_prices_df, f)
                    logging.info(f"Updated pickle file with missing tickers: {pickle_filename}")
                else:
                    logging.info("All requested tickers found in pickle file - no additional fetching needed")
                    
            except Exception as e:
                logging.warning(f"Failed to load pickle file: {e}")
                logging.info("Will fetch data from Yahoo Finance instead")
                # Initialize empty DataFrame and fetch all data
                columns = pd.MultiIndex.from_tuples([], names=['Ticker', 'Metric'])
                all_prices_df = pd.DataFrame(columns=columns)
                
                # Fetch all tickers
                for ticker in tickers_df['Symbol']:
                    try:
                        ticker_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')[['Close', 'Volume']]
                        for col in ['Close', 'Volume']:
                            all_prices_df[(ticker, col)] = ticker_data[col]
                        logging.info(f"Successfully downloaded daily price data for {ticker} ({len(ticker_data)} days)")
                    except Exception as e:
                        logging.warning(f"Failed to download data for {ticker}: {e}")
                        continue
        else:
            logging.info(f"Pickle file not found: {pickle_filename}")
            logging.info("Fetching all data from Yahoo Finance")
            
            # Initialize empty DataFrame to store prices with multi-index columns
            columns = pd.MultiIndex.from_tuples([], names=['Ticker', 'Metric'])
            all_prices_df = pd.DataFrame(columns=columns)

            # Fetch daily prices for each ticker
            for ticker in tickers_df['Symbol']:
                try:
                    # Download historical data from Yahoo Finance with custom date range
                    ticker_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')[['Close', 'Volume']]
                    # Add the ticker's price data using multi-index columns (ticker, metric)
                    for col in ['Close', 'Volume']:
                        all_prices_df[(ticker, col)] = ticker_data[col]
                    logging.info(f"Successfully downloaded daily price data for {ticker} ({len(ticker_data)} days)")
                except Exception as e:
                    logging.warning(f"Failed to download data for {ticker}: {e}")
                    continue
        
        # Save data to pickle file (if we fetched new data or updated existing data)
        if not os.path.exists(pickle_filename) or missing_tickers:
            try:
                with open(pickle_filename, 'wb') as f:
                    pickle.dump(all_prices_df, f)
                logging.info(f"Price data saved to pickle file: {pickle_filename}")
            except Exception as e:
                logging.warning(f"Failed to save pickle file: {e}")


        # Calculate daily returns (preserve ticker information in column names)
        close_prices = all_prices_df.xs('Close', level=1, axis=1)
        close_prices.columns = close_prices.columns.get_level_values(0)  # Keep ticker names
        returns_df = close_prices.pct_change().dropna()
        logging.info(f"Calculated daily returns for {len(returns_df.columns)} tickers")

        # Calculate mean returns and covariance matrix
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        
        logging.info("Successfully computed mean returns and covariance matrix")
        logging.debug(f"Mean returns shape: {mean_returns.shape}, Covariance matrix shape: {cov_matrix.shape}")
        
        # Remove any tickers that failed to download
        if all_prices_df.empty:
            logging.error("Failed to download price data for any tickers")
            sys.exit(1)
        else:
            logging.info(f"Successfully downloaded price data for {len(all_prices_df.columns)//2} tickers")
        
        # Portfolio Optimization using cvxpy functions
        logging.info(f"Starting portfolio optimization for {len(returns_df.columns)} assets")
        
        # Run different portfolio optimization variants
        portfolios = {}
        
        # 1. Vanilla Mean-Variance Optimization
        logging.info("=== Vanilla Mean-Variance Optimization ===")
        vanilla_result = optimize_portfolio_vanilla(
            mean_returns=mean_returns.values,
            cov_matrix=cov_matrix.values,
            risk_aversion=1.0,
            long_only=True
        )
        portfolios['Vanilla'] = vanilla_result
        
        if vanilla_result['status'] == 'optimal':
            logging.info("Vanilla optimization successful!")
            logging.info(f"Expected Annual Return: {vanilla_result['expected_return']:.2%}")
            logging.info(f"Annual Volatility: {vanilla_result['volatility']:.2%}")
            logging.info(f"Sharpe Ratio: {vanilla_result['sharpe_ratio']:.3f}")
            
            # Display top holdings
            vanilla_portfolio = pd.Series(vanilla_result['weights'], index=returns_df.columns).sort_values(ascending=False)
            logging.info("Top 5 Holdings (Vanilla):")
            for ticker, weight in vanilla_portfolio.head().items():
                logging.info(f"  {ticker}: {weight:.2%}")
        else:
            logging.error("Vanilla optimization failed!")
        
        # 2. Robust Portfolio Optimization
        logging.info("=== Robust Portfolio Optimization ===")
        robust_result = optimize_portfolio_robust(
            mean_returns=mean_returns.values,
            cov_matrix=cov_matrix.values,
            risk_aversion=1.0,
            uncertainty_level=0.1,
            long_only=True
        )
        portfolios['Robust'] = robust_result
        
        if robust_result['status'] == 'optimal':
            logging.info("Robust optimization successful!")
            logging.info(f"Expected Annual Return: {robust_result['expected_return']:.2%}")
            logging.info(f"Annual Volatility: {robust_result['volatility']:.2%}")
            logging.info(f"Sharpe Ratio: {robust_result['sharpe_ratio']:.3f}")
            
            # Display top holdings
            robust_portfolio = pd.Series(robust_result['weights'], index=returns_df.columns).sort_values(ascending=False)
            logging.info("Top 5 Holdings (Robust):")
            for ticker, weight in robust_portfolio.head().items():
                logging.info(f"  {ticker}: {weight:.2%}")
        else:
            logging.error("Robust optimization failed!")
        
        # 3. Additional variants (you can easily add more)
        # Conservative vanilla optimization
        logging.info("=== Conservative Vanilla Optimization ===")
        conservative_result = optimize_portfolio_vanilla(
            mean_returns=mean_returns.values,
            cov_matrix=cov_matrix.values,
            risk_aversion=5.0,  # Higher risk aversion
            long_only=True
        )
        portfolios['Conservative'] = conservative_result
        
        if conservative_result['status'] == 'optimal':
            logging.info("Conservative optimization successful!")
            logging.info(f"Expected Annual Return: {conservative_result['expected_return']:.2%}")
            logging.info(f"Annual Volatility: {conservative_result['volatility']:.2%}")
            logging.info(f"Sharpe Ratio: {conservative_result['sharpe_ratio']:.3f}")
        
        # High uncertainty robust optimization
        logging.info("=== High Uncertainty Robust Optimization ===")
        high_uncertainty_result = optimize_portfolio_robust(
            mean_returns=mean_returns.values,
            cov_matrix=cov_matrix.values,
            risk_aversion=1.0,
            uncertainty_level=0.2,  # Higher uncertainty
            long_only=True
        )
        portfolios['High_Uncertainty'] = high_uncertainty_result
        
        if high_uncertainty_result['status'] == 'optimal':
            logging.info("High uncertainty optimization successful!")
            logging.info(f"Expected Annual Return: {high_uncertainty_result['expected_return']:.2%}")
            logging.info(f"Annual Volatility: {high_uncertainty_result['volatility']:.2%}")
            logging.info(f"Sharpe Ratio: {high_uncertainty_result['sharpe_ratio']:.3f}")
        
        # Debug: Check portfolio statuses
        logging.info("=== Portfolio Optimization Status Summary ===")
        for name, portfolio in portfolios.items():
            status = portfolio['status']
            logging.info(f"{name}: {status}")
            if status != 'optimal':
                logging.warning(f"{name} optimization failed!")
        
        # Individual Portfolio Analysis Plots
        if vanilla_result['status'] == 'optimal':
            logging.info("Creating vanilla portfolio analysis plot...")
            plot_individual_portfolio_analysis(
                portfolio_weights=vanilla_result['weights'],
                returns_df=returns_df,
                portfolio_name='Vanilla',
                save_path='../plots/vanilla_portfolio_analysis.png'
            )
        else:
            logging.error("Skipping vanilla plot - optimization failed")
        
        if robust_result['status'] == 'optimal':
            logging.info("Creating robust portfolio analysis plot...")
            plot_individual_portfolio_analysis(
                portfolio_weights=robust_result['weights'], 
                returns_df=returns_df,
                portfolio_name='Robust',
                save_path='../plots/robust_portfolio_analysis.png'
            )
        else:
            logging.error("Skipping robust plot - optimization failed")
        
        # Portfolio Comparisons with Ticker File Baseline Weights
        # Create baseline weights from tickers_df, matching order of returns_df columns
        baseline_weights = np.zeros(len(returns_df.columns))
        for i, ticker in enumerate(returns_df.columns):
            ticker_row = tickers_df[tickers_df['Symbol'] == ticker]
            if not ticker_row.empty:
                baseline_weights[i] = ticker_row['Weight'].iloc[0]
            else:
                logging.warning(f"Ticker {ticker} not found in tickers file, setting weight to 0")
        
        # Normalize in case of any missing tickers
        if baseline_weights.sum() > 0:
            baseline_weights = baseline_weights / baseline_weights.sum()
        else:
            logging.warning("No matching tickers found, using equal weights")
            baseline_weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
        
        logging.info(f"Baseline weights: {dict(zip(returns_df.columns, baseline_weights))}")
        
        if vanilla_result['status'] == 'optimal':
            plot_portfolio_comparison(
                desired_weights=vanilla_result['weights'],
                baseline_weights=baseline_weights,
                returns_df=returns_df,
                desired_name='Vanilla',
                baseline_name='Baseline (Ticker File)',
                save_path='../plots/vanilla_vs_baseline.png'
            )
        
        if robust_result['status'] == 'optimal':
            plot_portfolio_comparison(
                desired_weights=robust_result['weights'],
                baseline_weights=baseline_weights,
                returns_df=returns_df,
                desired_name='Robust', 
                baseline_name='Baseline (Ticker File)',
                save_path='../plots/robust_vs_baseline.png'
            )
        
        # Vanilla vs Robust Comparison
        if vanilla_result['status'] == 'optimal' and robust_result['status'] == 'optimal':
            plot_portfolio_comparison(
                desired_weights=vanilla_result['weights'],
                baseline_weights=robust_result['weights'],
                returns_df=returns_df,
                desired_name='Vanilla',
                baseline_name='Robust',
                save_path='../plots/vanilla_vs_robust.png'
            )
        
        # Optional: Keep the old multi-portfolio comparison for overview
        compare_and_visualize_portfolios(
            portfolios=portfolios,
            asset_names=returns_df.columns.tolist(),
            save_path='../plots/portfolio_optimization_overview.png'
        )
        
        # Dynamic Rebalancing Analysis
        logging.info("\n" + "="*60)
        logging.info("STARTING DYNAMIC REBALANCING ANALYSIS")
        logging.info("="*60)
        
        try:
            # Run dynamic rebalancing with default configuration
            rebalancing_tracker = run_dynamic_rebalancing(
                returns_df=returns_df,
                baseline_weights=baseline_weights
            )
            
            logging.info("Dynamic rebalancing analysis completed successfully!")
            
        except Exception as e:
            logging.error(f"Error in dynamic rebalancing: {e}")
            logging.exception("Full error details:")
        
        # ipdb.set_trace()        

    except FileNotFoundError:
        logging.error(f"Ticker file not found: {ticker_file}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logging.error(f"Ticker file is empty: {ticker_file}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.exception("Full error details:")
        sys.exit(1)


if __name__ == "__main__":
    main()