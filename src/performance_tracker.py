#!/usr/bin/env python3
"""
Performance tracking module for dynamic portfolio rebalancing.
Tracks and stores performance metrics across rebalancing periods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, date
import logging
from dataclasses import dataclass, field

from config import RebalancingConfig


@dataclass
class PeriodPerformance:
    """Performance metrics for a single rebalancing period."""
    
    period_number: int
    start_date: date
    end_date: date
    portfolio_name: str
    
    # Portfolio characteristics
    weights: np.ndarray
    total_weight: float
    
    # Performance metrics
    period_return: float
    cumulative_return: float
    volatility: float
    sharpe_ratio: float
    beta: float
    max_drawdown: float
    calmar_ratio: float
    gain_pain_ratio: float
    
    # Additional info
    optimization_successful: bool = True
    notes: str = ""
    
    # Mixed portfolio specific fields (optional)
    cash_percentage: float = 0.0
    equity_percentage: float = 1.0


class PerformanceTracker:
    """Tracks performance metrics across multiple rebalancing periods."""
    
    def __init__(self, config: RebalancingConfig):
        """Initialize performance tracker with configuration."""
        self.config = config
        self.period_results: List[PeriodPerformance] = []
        # Build portfolio names based on configuration
        self.portfolio_names = []
        
        # Add baseline portfolios
        if config.include_baseline:
            self.portfolio_names.append('static_baseline')
        if config.include_rebalanced_baseline:
            self.portfolio_names.append('rebalanced_baseline')
        
        # Add optimization methods
        self.portfolio_names.extend(config.optimization_methods)
        
        # Add mixed portfolios if enabled
        if config.include_mixed_portfolios:
            self.portfolio_names += [f'mixed_{method}' for method in config.optimization_methods]
        self.current_period = 0
        
        # DataFrames to store results
        self.performance_df: Optional[pd.DataFrame] = None
        self.weights_df: Optional[pd.DataFrame] = None
        self.returns_df: Optional[pd.DataFrame] = None
        
        logging.info(f"PerformanceTracker initialized for portfolios: {self.portfolio_names}")
    
    def add_period_performance(self, 
                             period_number: int,
                             start_date: date,
                             end_date: date,
                             portfolio_performances: Dict[str, Dict[str, Any]]) -> None:
        """
        Add performance results for a complete rebalancing period.
        
        Parameters:
        -----------
        period_number : int
            The period number (starting from 0)
        start_date : date
            Start date of the period
        end_date : date
            End date of the period
        portfolio_performances : Dict[str, Dict[str, Any]]
            Performance metrics for each portfolio
        """
        for portfolio_name, metrics in portfolio_performances.items():
            performance = PeriodPerformance(
                period_number=period_number,
                start_date=start_date,
                end_date=end_date,
                portfolio_name=portfolio_name,
                weights=metrics.get('weights', np.array([])),
                total_weight=metrics.get('total_weight', 0.0),
                period_return=metrics.get('period_return', 0.0),
                cumulative_return=metrics.get('cumulative_return', 0.0),
                volatility=metrics.get('volatility', 0.0),
                sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
                beta=metrics.get('beta', 0.0),
                max_drawdown=metrics.get('max_drawdown', 0.0),
                calmar_ratio=metrics.get('calmar_ratio', 0.0),
                gain_pain_ratio=metrics.get('gain_pain_ratio', 0.0),
                optimization_successful=metrics.get('optimization_successful', True),
                notes=metrics.get('notes', ""),
                cash_percentage=metrics.get('cash_percentage', 0.0),
                equity_percentage=metrics.get('equity_percentage', 1.0)
            )
            
            self.period_results.append(performance)
        
        self.current_period = period_number
        logging.debug(f"Added performance data for period {period_number}")
    
    def calculate_portfolio_returns(self, 
                                  weights: np.ndarray, 
                                  returns_data: pd.DataFrame,
                                  portfolio_name: str) -> pd.Series:
        """Calculate portfolio returns given weights and asset returns."""
        if len(weights) != len(returns_data.columns):
            raise ValueError(f"Weights length ({len(weights)}) doesn't match assets ({len(returns_data.columns)})")
        
        portfolio_returns = (returns_data * weights).sum(axis=1)
        portfolio_returns.name = portfolio_name
        return portfolio_returns
    
    def calculate_period_metrics(self, 
                               portfolio_returns: pd.Series,
                               market_returns: pd.Series,
                               risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics for a period.
        
        Parameters:
        -----------
        portfolio_returns : pd.Series
            Daily returns for the portfolio
        market_returns : pd.Series
            Daily returns for market proxy
        risk_free_rate : float
            Annual risk-free rate
            
        Returns:
        --------
        Dict with calculated metrics
        """
        if len(portfolio_returns) == 0:
            return self._get_empty_metrics()
        
        # Basic return metrics
        period_return = (1 + portfolio_returns).prod() - 1
        cumulative_return = period_return  # Will be calculated properly in backtest
        
        # Risk metrics
        volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
        
        # Sharpe ratio
        excess_return = portfolio_returns.mean() * 252 - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Beta calculation
        if len(market_returns) == len(portfolio_returns):
            covariance = np.cov(portfolio_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance if market_variance > 0 else 0
        else:
            beta = 0
        
        # Drawdown calculation
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / rolling_max - 1)
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        annualized_return = (1 + period_return) ** (252 / len(portfolio_returns)) - 1
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Gain-to-pain ratio
        positive_returns = portfolio_returns[portfolio_returns > 0].sum()
        negative_returns = abs(portfolio_returns[portfolio_returns < 0].sum())
        gain_pain_ratio = positive_returns / negative_returns if negative_returns > 0 else float('inf')
        
        return {
            'period_return': period_return,
            'cumulative_return': cumulative_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'beta': beta,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'gain_pain_ratio': gain_pain_ratio
        }
    
    def _get_empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary for failed periods."""
        return {
            'period_return': 0.0,
            'cumulative_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'beta': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'gain_pain_ratio': 0.0
        }
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Generate summary DataFrame of all period performances."""
        if not self.period_results:
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for result in self.period_results:
            row = {
                'period': result.period_number,
                'start_date': result.start_date,
                'end_date': result.end_date,
                'portfolio': result.portfolio_name,
                'period_return': result.period_return,
                'cumulative_return': result.cumulative_return,
                'volatility': result.volatility,
                'sharpe_ratio': result.sharpe_ratio,
                'beta': result.beta,
                'max_drawdown': result.max_drawdown,
                'calmar_ratio': result.calmar_ratio,
                'gain_pain_ratio': result.gain_pain_ratio,
                'total_weight': result.total_weight,
                'optimization_successful': result.optimization_successful
            }
            data.append(row)
        
        self.performance_df = pd.DataFrame(data)
        return self.performance_df
    
    def get_weights_summary(self, asset_names: List[str]) -> pd.DataFrame:
        """Generate summary DataFrame of portfolio weights over time."""
        if not self.period_results:
            return pd.DataFrame()
        
        weights_data = []
        for result in self.period_results:
            if len(result.weights) == len(asset_names):
                row = {
                    'period': result.period_number,
                    'portfolio': result.portfolio_name,
                    'start_date': result.start_date
                }
                # Add weights for each asset
                for i, asset in enumerate(asset_names):
                    row[f'weight_{asset}'] = result.weights[i]
                
                weights_data.append(row)
        
        self.weights_df = pd.DataFrame(weights_data)
        return self.weights_df
    
    def get_cumulative_returns_series(self) -> pd.DataFrame:
        """Generate time series of cumulative returns for all portfolios."""
        if not self.period_results:
            return pd.DataFrame()
        
        # Group by portfolio
        portfolio_returns = {}
        for portfolio_name in self.portfolio_names:
            portfolio_data = [r for r in self.period_results if r.portfolio_name == portfolio_name]
            if portfolio_data:
                periods = [r.period_number for r in portfolio_data]
                returns = [r.cumulative_return for r in portfolio_data]
                portfolio_returns[portfolio_name] = pd.Series(returns, index=periods)
        
        self.returns_df = pd.DataFrame(portfolio_returns)
        return self.returns_df
    
    def save_results(self, directory: str = None) -> None:
        """Save all tracking results to files."""
        if directory is None:
            directory = self.config.results_directory
        
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save performance summary
        if self.performance_df is not None:
            performance_file = os.path.join(directory, 'performance_summary.csv')
            self.performance_df.to_csv(performance_file, index=False)
            logging.info(f"Performance summary saved to {performance_file}")
        
        # Save weights summary
        if self.weights_df is not None:
            weights_file = os.path.join(directory, 'weights_summary.csv')
            self.weights_df.to_csv(weights_file, index=False)
            logging.info(f"Weights summary saved to {weights_file}")
        
        # Save returns series
        if self.returns_df is not None:
            returns_file = os.path.join(directory, 'cumulative_returns.csv')
            self.returns_df.to_csv(returns_file, index=True)
            logging.info(f"Cumulative returns saved to {returns_file}")
    
    def get_portfolio_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate overall statistics for each portfolio across all periods."""
        if self.performance_df is None:
            self.get_performance_summary()
        
        if self.performance_df.empty:
            return {}
        
        stats = {}
        for portfolio in self.portfolio_names:
            portfolio_data = self.performance_df[self.performance_df['portfolio'] == portfolio]
            
            if not portfolio_data.empty:
                stats[portfolio] = {
                    'total_return': portfolio_data['cumulative_return'].iloc[-1] if len(portfolio_data) > 0 else 0,
                    'avg_period_return': portfolio_data['period_return'].mean(),
                    'volatility': portfolio_data['period_return'].std(),
                    'avg_sharpe': portfolio_data['sharpe_ratio'].mean(),
                    'max_drawdown': portfolio_data['max_drawdown'].min(),
                    'success_rate': portfolio_data['optimization_successful'].mean()
                }
        
        return stats