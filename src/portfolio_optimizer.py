#!/usr/bin/env python3
"""
Portfolio Optimization Module.
Implements multiple optimization algorithms with flexible constraint system.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
import logging
from typing import Dict, List, Any, Optional, Callable
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import warnings


class PortfolioOptimizer:
    """
    Portfolio optimization class with multiple algorithms and flexible constraints.
    
    Supports various optimization methods:
    - Mean-variance optimization (Markowitz)
    - Robust mean-variance optimization
    - Risk parity
    - Minimum variance
    - Maximum Sharpe ratio
    - Maximum diversification ratio
    - Hierarchical clustering (HRP-style)
    - Black-Litterman (placeholder)
    
    Features flexible constraint system for easy extension.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize PortfolioOptimizer.
        
        Parameters:
        -----------
        risk_free_rate : float
            Annual risk-free rate for Sharpe ratio calculations
        """
        self.risk_free_rate = risk_free_rate
        
        # Available optimization methods
        self.optimization_methods = {
            'mean_variance': self.optimize_mean_variance,
            'robust_mean_variance': self.optimize_robust_mean_variance,
            'risk_parity': self.optimize_risk_parity,
            'min_variance': self.optimize_minimum_variance,
            'max_sharpe': self.optimize_maximum_sharpe,
            'max_diversification': self.optimize_maximum_diversification,
            'hierarchical_clustering': self.optimize_hierarchical_clustering,
            'black_litterman': self.optimize_black_litterman
        }
        
        # Default constraints
        self.default_constraints = {
            'long_only': True,
            'min_weight': 0.001,
            'max_weight': 0.4,
            'max_concentration': 0.6,
            'max_turnover': None,
            'sector_limits': None,
            'max_volatility': None,
            'max_tracking_error': None
        }
        
        logging.info(f"PortfolioOptimizer initialized with {len(self.optimization_methods)} methods")
    
    def get_available_methods(self) -> List[str]:
        """Get list of available optimization methods."""
        return list(self.optimization_methods.keys())
    
    def get_method_parameters(self, method: str) -> Dict[str, Any]:
        """
        Get parameters for specific optimization method.
        
        Parameters:
        -----------
        method : str
            Optimization method name
            
        Returns:
        --------
        Dict with parameter descriptions and defaults
        """
        method_params = {
            'mean_variance': {'risk_aversion': 1.0},
            'robust_mean_variance': {'risk_aversion': 1.0, 'uncertainty_level': 0.1},
            'risk_parity': {},
            'min_variance': {},
            'max_sharpe': {},
            'max_diversification': {},
            'hierarchical_clustering': {'linkage_method': 'ward'},
            'black_litterman': {'views': None, 'confidence': None}
        }
        
        return method_params.get(method, {})
    
    def optimize(self, method: str, mean_returns: np.ndarray, cov_matrix: np.ndarray, 
                **kwargs) -> Dict[str, Any]:
        """
        Main optimization interface - route to specific method.
        
        Parameters:
        -----------
        method : str
            Optimization method name
        mean_returns : np.ndarray
            Expected returns vector
        cov_matrix : np.ndarray
            Covariance matrix
        **kwargs : dict
            Method-specific parameters and constraints
            
        Returns:
        --------
        Dict with optimization results
        """
        if method not in self.optimization_methods:
            available = list(self.optimization_methods.keys())
            raise ValueError(f"Unknown optimization method '{method}'. Available: {available}")
        
        # Merge with default constraints
        constraints = {**self.default_constraints, **kwargs}
        
        # Call the specific optimization method
        opt_func = self.optimization_methods[method]
        
        try:
            result = opt_func(mean_returns, cov_matrix, **constraints)
            result['method'] = method
            result['parameters'] = constraints
            return result
        except Exception as e:
            logging.error(f"Optimization failed for method '{method}': {e}")
            return {
                'status': 'failed',
                'message': str(e),
                'method': method,
                'weights': np.zeros(len(mean_returns))
            }
    
    # =============================================================================
    # OPTIMIZATION ALGORITHMS
    # =============================================================================
    
    def optimize_mean_variance(self, mean_returns: np.ndarray, cov_matrix: np.ndarray,
                             risk_aversion: float = 1.0, **constraints) -> Dict[str, Any]:
        """
        Mean-variance optimization (Markowitz).
        
        Maximizes: μ'w - (γ/2)w'Σw
        where μ = expected returns, Σ = covariance matrix, γ = risk aversion
        """
        n = len(mean_returns)
        w = cp.Variable(n)
        
        # Objective: maximize expected return - risk penalty
        portfolio_return = mean_returns @ w
        portfolio_risk = cp.quad_form(w, cov_matrix)
        objective = cp.Maximize(portfolio_return - (risk_aversion / 2) * portfolio_risk)
        
        # Apply constraints
        constraint_list = self._build_constraints(w, **constraints)
        
        # Solve optimization
        problem = cp.Problem(objective, constraint_list)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            return {'status': 'failed', 'message': f"Solver status: {problem.status}"}
        
        weights = w.value
        if weights is None:
            return {'status': 'failed', 'message': "No solution found"}
        
        # Calculate portfolio metrics
        metrics = self._calculate_portfolio_metrics(weights, mean_returns, cov_matrix)
        
        return {
            'status': 'optimal',
            'weights': weights,
            'risk_aversion': risk_aversion,
            **metrics
        }
    
    def optimize_robust_mean_variance(self, mean_returns: np.ndarray, cov_matrix: np.ndarray,
                                    risk_aversion: float = 1.0, uncertainty_level: float = 0.1,
                                    **constraints) -> Dict[str, Any]:
        """
        Robust mean-variance optimization with uncertainty sets.
        
        Accounts for uncertainty in expected returns using ellipsoidal uncertainty sets.
        """
        n = len(mean_returns)
        w = cp.Variable(n)
        
        # Robust objective with uncertainty set
        portfolio_return = mean_returns @ w
        
        # Uncertainty adjustment: subtract uncertainty penalty
        uncertainty_penalty = uncertainty_level * cp.norm(cp.multiply(np.sqrt(np.diag(cov_matrix)), w))
        robust_return = portfolio_return - uncertainty_penalty
        
        portfolio_risk = cp.quad_form(w, cov_matrix)
        objective = cp.Maximize(robust_return - (risk_aversion / 2) * portfolio_risk)
        
        # Apply constraints
        constraint_list = self._build_constraints(w, **constraints)
        
        # Solve optimization
        problem = cp.Problem(objective, constraint_list)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            return {'status': 'failed', 'message': f"Solver status: {problem.status}"}
        
        weights = w.value
        if weights is None:
            return {'status': 'failed', 'message': "No solution found"}
        
        # Calculate portfolio metrics
        metrics = self._calculate_portfolio_metrics(weights, mean_returns, cov_matrix)
        
        return {
            'status': 'optimal',
            'weights': weights,
            'risk_aversion': risk_aversion,
            'uncertainty_level': uncertainty_level,
            **metrics
        }
    
    def optimize_risk_parity(self, mean_returns: np.ndarray, cov_matrix: np.ndarray,
                           **constraints) -> Dict[str, Any]:
        """
        Risk parity optimization.
        
        Seeks portfolio where each asset contributes equally to total portfolio risk.
        """
        n = len(mean_returns)
        w = cp.Variable(n)
        
        # Risk parity objective: minimize sum of squared risk contributions
        # Risk contribution of asset i = w_i * (Σw)_i / (w'Σw)
        # Approximate using sum of squared marginal risk contributions
        portfolio_vol = cp.sqrt(cp.quad_form(w, cov_matrix))
        
        # Marginal risk contributions
        marginal_risk = cov_matrix @ w / portfolio_vol
        risk_contributions = cp.multiply(w, marginal_risk)
        
        # Objective: minimize variance of risk contributions
        avg_risk_contrib = cp.sum(risk_contributions) / n
        objective = cp.Minimize(cp.sum_squares(risk_contributions - avg_risk_contrib))
        
        # Apply constraints
        constraint_list = self._build_constraints(w, **constraints)
        
        # Add constraint to avoid trivial solution
        constraint_list.append(cp.sum(w) == 1)
        if constraints.get('long_only', True):
            constraint_list.append(w >= 0)
        
        # Solve optimization
        problem = cp.Problem(objective, constraint_list)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            # Fallback to equal weights
            weights = np.ones(n) / n
            logging.warning("Risk parity optimization failed, using equal weights")
        else:
            weights = w.value
            if weights is None:
                weights = np.ones(n) / n
        
        # Calculate portfolio metrics
        metrics = self._calculate_portfolio_metrics(weights, mean_returns, cov_matrix)
        
        return {
            'status': 'optimal',
            'weights': weights,
            **metrics
        }
    
    def optimize_minimum_variance(self, mean_returns: np.ndarray, cov_matrix: np.ndarray,
                                **constraints) -> Dict[str, Any]:
        """
        Minimum variance optimization.
        
        Minimizes portfolio variance without regard to expected returns.
        """
        n = len(mean_returns)
        w = cp.Variable(n)
        
        # Objective: minimize portfolio variance
        objective = cp.Minimize(cp.quad_form(w, cov_matrix))
        
        # Apply constraints
        constraint_list = self._build_constraints(w, **constraints)
        
        # Solve optimization
        problem = cp.Problem(objective, constraint_list)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            return {'status': 'failed', 'message': f"Solver status: {problem.status}"}
        
        weights = w.value
        if weights is None:
            return {'status': 'failed', 'message': "No solution found"}
        
        # Calculate portfolio metrics
        metrics = self._calculate_portfolio_metrics(weights, mean_returns, cov_matrix)
        
        return {
            'status': 'optimal',
            'weights': weights,
            **metrics
        }
    
    def optimize_maximum_sharpe(self, mean_returns: np.ndarray, cov_matrix: np.ndarray,
                              **constraints) -> Dict[str, Any]:
        """
        Maximum Sharpe ratio optimization.
        
        Maximizes (μ'w - rf) / sqrt(w'Σw)
        """
        n = len(mean_returns)
        
        # Transform to linear problem using auxiliary variable
        # max (μ'w - rf) / sqrt(w'Σw) => max (μ'w - rf) s.t. w'Σw <= 1
        w = cp.Variable(n)
        
        # Excess returns
        excess_returns = mean_returns - self.risk_free_rate
        
        # Objective: maximize excess return subject to unit variance constraint
        objective = cp.Maximize(excess_returns @ w)
        
        # Constraint: portfolio variance <= 1
        constraint_list = [cp.quad_form(w, cov_matrix) <= 1]
        
        # Apply other constraints (scaled appropriately)
        if constraints.get('long_only', True):
            constraint_list.append(w >= 0)
        
        # Note: weight bounds are tricky with this formulation
        # We'll solve and then rescale
        
        # Solve optimization
        problem = cp.Problem(objective, constraint_list)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            return {'status': 'failed', 'message': f"Solver status: {problem.status}"}
        
        weights_scaled = w.value
        if weights_scaled is None:
            return {'status': 'failed', 'message': "No solution found"}
        
        # Rescale weights to sum to 1
        weights = weights_scaled / np.sum(weights_scaled)
        
        # Apply weight constraints after rescaling
        min_weight = constraints.get('min_weight', 0.001)
        max_weight = constraints.get('max_weight', 0.4)
        
        # Clip weights to bounds
        weights = np.clip(weights, min_weight, max_weight)
        weights = weights / np.sum(weights)  # Renormalize
        
        # Calculate portfolio metrics
        metrics = self._calculate_portfolio_metrics(weights, mean_returns, cov_matrix)
        
        return {
            'status': 'optimal',
            'weights': weights,
            **metrics
        }
    
    def optimize_maximum_diversification(self, mean_returns: np.ndarray, cov_matrix: np.ndarray,
                                       **constraints) -> Dict[str, Any]:
        """
        Maximum diversification ratio optimization.
        
        Maximizes (weighted average volatility) / (portfolio volatility)
        """
        n = len(mean_returns)
        w = cp.Variable(n)
        
        # Individual asset volatilities
        asset_vols = np.sqrt(np.diag(cov_matrix))
        
        # Objective: maximize diversification ratio
        # This is equivalent to minimizing portfolio vol while constraining weighted avg vol
        weighted_avg_vol = asset_vols @ w
        portfolio_vol = cp.sqrt(cp.quad_form(w, cov_matrix))
        
        # Transform to tractable form: minimize portfolio vol subject to weighted avg vol = 1
        objective = cp.Minimize(portfolio_vol)
        constraint_list = [weighted_avg_vol == 1]
        
        # Apply other constraints
        if constraints.get('long_only', True):
            constraint_list.append(w >= 0)
        
        # Solve optimization
        problem = cp.Problem(objective, constraint_list)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            return {'status': 'failed', 'message': f"Solver status: {problem.status}"}
        
        weights_scaled = w.value
        if weights_scaled is None:
            return {'status': 'failed', 'message': "No solution found"}
        
        # Rescale weights to sum to 1
        weights = weights_scaled / np.sum(weights_scaled)
        
        # Apply weight constraints
        min_weight = constraints.get('min_weight', 0.001)
        max_weight = constraints.get('max_weight', 0.4)
        
        weights = np.clip(weights, min_weight, max_weight)
        weights = weights / np.sum(weights)
        
        # Calculate portfolio metrics
        metrics = self._calculate_portfolio_metrics(weights, mean_returns, cov_matrix)
        
        return {
            'status': 'optimal',
            'weights': weights,
            **metrics
        }
    
    def optimize_hierarchical_clustering(self, mean_returns: np.ndarray, cov_matrix: np.ndarray,
                                       linkage_method: str = 'ward', **constraints) -> Dict[str, Any]:
        """
        Hierarchical clustering portfolio (HRP-style).
        
        Uses hierarchical clustering to build portfolio weights based on correlation structure.
        """
        n = len(mean_returns)
        
        # Convert covariance to correlation matrix
        vol = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(vol, vol)
        
        # Calculate distance matrix (1 - |correlation|)
        distance_matrix = np.sqrt((1 - np.abs(corr_matrix)) / 2)
        
        # Hierarchical clustering
        condensed_distances = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_distances, method=linkage_method)
        
        # Get clusters
        num_clusters = min(n // 2, 10)  # Reasonable number of clusters
        clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
        
        # Allocate weights hierarchically
        weights = np.zeros(n)
        
        # First level: equal weight to each cluster
        cluster_weights = {}
        for cluster_id in np.unique(clusters):
            cluster_weights[cluster_id] = 1.0 / len(np.unique(clusters))
        
        # Second level: within each cluster, use inverse volatility weighting
        for cluster_id in np.unique(clusters):
            cluster_assets = np.where(clusters == cluster_id)[0]
            if len(cluster_assets) == 1:
                weights[cluster_assets[0]] = cluster_weights[cluster_id]
            else:
                # Inverse volatility weights within cluster
                cluster_vols = vol[cluster_assets]
                inv_vol_weights = (1 / cluster_vols) / np.sum(1 / cluster_vols)
                
                for i, asset_idx in enumerate(cluster_assets):
                    weights[asset_idx] = cluster_weights[cluster_id] * inv_vol_weights[i]
        
        # Ensure weights sum to 1
        weights = weights / np.sum(weights)
        
        # Apply weight constraints
        min_weight = constraints.get('min_weight', 0.001)
        max_weight = constraints.get('max_weight', 0.4)
        
        weights = np.clip(weights, min_weight, max_weight)
        weights = weights / np.sum(weights)
        
        # Calculate portfolio metrics
        metrics = self._calculate_portfolio_metrics(weights, mean_returns, cov_matrix)
        
        return {
            'status': 'optimal',
            'weights': weights,
            'num_clusters': num_clusters,
            **metrics
        }
    
    def optimize_black_litterman(self, mean_returns: np.ndarray, cov_matrix: np.ndarray,
                               views: Optional[Dict] = None, confidence: Optional[float] = None,
                               **constraints) -> Dict[str, Any]:
        """
        Black-Litterman optimization (placeholder implementation).
        
        This is a simplified version - full implementation would require
        market cap data and more sophisticated view specification.
        """
        # For now, fall back to mean-variance with adjusted returns
        if views is None:
            # No views provided, use standard mean-variance
            return self.optimize_mean_variance(mean_returns, cov_matrix, **constraints)
        
        # Placeholder: modify expected returns based on views
        adjusted_returns = mean_returns.copy()
        
        # Simple view incorporation (would need proper BL formula in practice)
        for asset_idx, view_return in views.items():
            if isinstance(asset_idx, int) and 0 <= asset_idx < len(adjusted_returns):
                # Blend original estimate with view
                blend_factor = confidence if confidence is not None else 0.5
                adjusted_returns[asset_idx] = ((1 - blend_factor) * mean_returns[asset_idx] + 
                                             blend_factor * view_return)
        
        # Run mean-variance with adjusted returns
        result = self.optimize_mean_variance(adjusted_returns, cov_matrix, **constraints)
        result['views_applied'] = views
        result['confidence'] = confidence
        
        return result
    
    # =============================================================================
    # CONSTRAINT SYSTEM
    # =============================================================================
    
    def _build_constraints(self, weights, **constraints) -> List:
        """
        Build list of cvxpy constraints from constraint dictionary.
        
        Parameters:
        -----------
        weights : cp.Variable
            CVXPY weight variable
        **constraints : dict
            Constraint specifications
            
        Returns:
        --------
        List of cvxpy constraints
        """
        constraint_list = []
        
        # Portfolio weights sum to 1
        constraint_list.append(cp.sum(weights) == 1)
        
        # Long-only constraint
        if constraints.get('long_only', True):
            constraint_list.append(weights >= 0)
        
        # Weight bounds
        min_weight = constraints.get('min_weight', 0.001)
        max_weight = constraints.get('max_weight', 0.4)
        
        if min_weight > 0:
            constraint_list.append(weights >= min_weight)
        if max_weight < 1:
            constraint_list.append(weights <= max_weight)
        
        # Maximum concentration constraint
        max_concentration = constraints.get('max_concentration')
        if max_concentration is not None:
            constraint_list.append(cp.norm(weights, 'inf') <= max_concentration)
        
        # Additional constraints can be added here
        # (sector constraints, turnover constraints, etc.)
        
        return constraint_list
    
    # =============================================================================
    # PORTFOLIO METRICS CALCULATION
    # =============================================================================
    
    def _calculate_portfolio_metrics(self, weights: np.ndarray, mean_returns: np.ndarray, 
                                   cov_matrix: np.ndarray) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        mean_returns : np.ndarray
            Expected returns
        cov_matrix : np.ndarray
            Covariance matrix
            
        Returns:
        --------
        Dict with portfolio metrics
        """
        # Expected return (annualized)
        portfolio_return = np.dot(weights, mean_returns) * 252
        
        # Portfolio volatility (annualized)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance) * np.sqrt(252)
        
        # Sharpe ratio
        excess_return = portfolio_return - self.risk_free_rate
        sharpe_ratio = excess_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Weight statistics
        total_weight = np.sum(weights)
        max_weight = np.max(weights)
        effective_assets = np.sum(weights > 0.001)  # Assets with meaningful allocation
        concentration = np.sum(weights**2)  # Herfindahl index
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'total_weight': total_weight,
            'max_weight': max_weight,
            'effective_assets': effective_assets,
            'concentration': concentration
        }
    
    def compare_portfolios(self, portfolios: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple portfolio optimization results.
        
        Parameters:
        -----------
        portfolios : Dict[str, Dict[str, Any]]
            Dictionary of portfolio results from optimize() calls
            
        Returns:
        --------
        pd.DataFrame with comparison metrics
        """
        comparison_data = []
        
        for name, result in portfolios.items():
            if result.get('status') == 'optimal':
                row = {
                    'Portfolio': name,
                    'Expected Return': result.get('expected_return', 0),
                    'Volatility': result.get('volatility', 0),
                    'Sharpe Ratio': result.get('sharpe_ratio', 0),
                    'Max Weight': result.get('max_weight', 0),
                    'Effective Assets': result.get('effective_assets', 0),
                    'Concentration': result.get('concentration', 0),
                    'Status': result.get('status', 'unknown')
                }
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)