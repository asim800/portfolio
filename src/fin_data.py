#!/usr/bin/env python3
"""
Financial Data Management Module.
Handles data loading, caching, and covariance matrix calculations.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import os
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, MinCovDet
from scipy.linalg import sqrtm
import cvxpy as cp


class FinData:
    """
    Financial data management class with multiple covariance calculation methods.
    
    Handles ticker loading, Yahoo Finance data fetching, caching, and provides
    various covariance matrix estimation methods for portfolio optimization.
    """
    
    def __init__(self, start_date: str, end_date: str, cache_dir: str = "../data"):
        """
        Initialize FinData with date range and cache directory.
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        cache_dir : str
            Directory for pickle cache files
        """
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = cache_dir
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Data storage
        self._price_data: Optional[pd.DataFrame] = None
        self._returns_data: Optional[pd.DataFrame] = None
        self._tickers_df: Optional[pd.DataFrame] = None
        
        logging.info(f"FinData initialized for period {start_date} to {end_date}")
    
    def load_tickers(self, ticker_file: str) -> pd.DataFrame:
        """
        Load ticker symbols and weights from CSV file.
        
        Parameters:
        -----------
        ticker_file : str
            Path to ticker file (format: Symbol,Weight with headers)
            
        Returns:
        --------
        pd.DataFrame with columns ['Symbol', 'Weight']
        """
        try:
            # Read CSV with headers
            tickers_df = pd.read_csv(ticker_file, skipinitialspace=True)
            
            # Clean column names
            tickers_df.columns = tickers_df.columns.str.strip()
            
            # Standardize column names
            if 'ticker' in tickers_df.columns:
                tickers_df = tickers_df.rename(columns={'ticker': 'Symbol'})
            if 'weights' in tickers_df.columns:
                tickers_df = tickers_df.rename(columns={'weights': 'Weight'})
            
            # Remove empty rows and ensure weights are numeric
            tickers_df = tickers_df.dropna()
            tickers_df['Weight'] = pd.to_numeric(tickers_df['Weight'], errors='coerce')
            tickers_df = tickers_df.dropna()
            
            # Normalize weights to sum to 1
            tickers_df['Weight'] = tickers_df['Weight'] / tickers_df['Weight'].sum()
            
            self._tickers_df = tickers_df
            
            logging.info(f"Loaded {len(tickers_df)} tickers from {ticker_file}")
            logging.info(f"Weights sum to: {tickers_df['Weight'].sum():.4f}")
            
            return tickers_df
            
        except Exception as e:
            logging.error(f"Failed to load ticker file {ticker_file}: {e}")
            raise
    
    def _generate_ticker_hash(self, tickers: List[str]) -> str:
        """
        Generate a short hash of the ticker list for cache versioning.
        
        Parameters:
        -----------
        tickers : List[str]
            List of ticker symbols
            
        Returns:
        --------
        str: Short hash of the sorted ticker list
        """
        import hashlib
        
        # Sort tickers for consistent hash regardless of order
        sorted_tickers = sorted(tickers)
        ticker_string = ','.join(sorted_tickers)
        
        # Generate short hash (first 8 characters)
        hash_object = hashlib.md5(ticker_string.encode())
        return hash_object.hexdigest()[:8]
    
    def get_cache_filename(self, tickers: List[str] = None) -> str:
        """
        Generate cache filename based on date range and ticker list.
        
        Parameters:
        -----------
        tickers : List[str], optional
            List of ticker symbols for hash generation
            
        Returns:
        --------
        str: Cache filename including ticker hash
        """
        if tickers:
            ticker_hash = self._generate_ticker_hash(tickers)
            return os.path.join(self.cache_dir, f"price_data_{self.start_date}_{self.end_date}_{ticker_hash}.pkl")
        else:
            # Fallback to old naming for backward compatibility
            return os.path.join(self.cache_dir, f"price_data_{self.start_date}_{self.end_date}.pkl")
    
    def load_cache(self, tickers: List[str]) -> Optional[pd.DataFrame]:
        """
        Load price data from pickle cache for specific tickers.
        
        Parameters:
        -----------
        tickers : List[str]
            List of ticker symbols
        
        Returns:
        --------
        pd.DataFrame or None if cache doesn't exist or fails to load
        """
        cache_file = self.get_cache_filename(tickers)
        
        if not os.path.exists(cache_file):
            logging.info(f"Cache file not found: {cache_file}")
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            logging.info(f"Successfully loaded price data from cache ({len(data.columns)//2} tickers)")
            return data
        except Exception as e:
            logging.warning(f"Failed to load cache file: {e}")
            return None
    
    def save_cache(self, data: pd.DataFrame, tickers: List[str]) -> None:
        """
        Save price data to pickle cache.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data to cache
        tickers : List[str]
            List of ticker symbols for filename generation
        """
        cache_file = self.get_cache_filename(tickers)
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logging.info(f"Price data saved to cache: {cache_file}")
        except Exception as e:
            logging.warning(f"Failed to save cache file: {e}")
    
    def fetch_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single ticker from Yahoo Finance.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
            
        Returns:
        --------
        pd.DataFrame with Close and Volume columns or None if failed
        """
        try:
            data = yf.download(ticker, start=self.start_date, end=self.end_date, 
                             interval='1d')[['Close', 'Volume']]
            logging.info(f"Downloaded {len(data)} days of data for {ticker}")
            return data
        except Exception as e:
            logging.warning(f"Failed to download data for {ticker}: {e}")
            return None
    
    def get_price_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Get price data for specified tickers with caching.
        
        Parameters:
        -----------
        tickers : List[str]
            List of ticker symbols
            
        Returns:
        --------
        pd.DataFrame with MultiIndex columns (Ticker, Metric)
        """
        # Try to load from cache first (ticker-specific cache)
        cached_data = self.load_cache(tickers)
        
        if cached_data is not None:
            # Cache hit - all requested tickers should be present
            logging.info("All requested tickers found in cache")
            self._price_data = cached_data
            return cached_data
        else:
            # Cache miss - need to fetch all tickers fresh
            logging.info(f"Cache miss - fetching fresh data for {len(tickers)} tickers")
            columns = pd.MultiIndex.from_tuples([], names=['Ticker', 'Metric'])
            all_prices_df = pd.DataFrame(columns=columns)
            missing_tickers = set(tickers)
        
        # Fetch missing tickers
        for ticker in missing_tickers:
            ticker_data = self.fetch_ticker_data(ticker)
            if ticker_data is not None:
                for col in ['Close', 'Volume']:
                    all_prices_df[(ticker, col)] = ticker_data[col]
        
        # Save updated data to cache
        self.save_cache(all_prices_df, tickers)
        self._price_data = all_prices_df
        
        logging.info(f"Price data ready for {len(all_prices_df.columns)//2} tickers")
        return all_prices_df
    
    def get_returns_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Calculate daily returns from price data.
        
        Parameters:
        -----------
        tickers : List[str]
            List of ticker symbols
            
        Returns:
        --------
        pd.DataFrame with daily returns for each ticker
        """
        if self._price_data is None:
            self.get_price_data(tickers)
        
        # Extract close prices and calculate returns
        close_prices = self._price_data.xs('Close', level=1, axis=1)
        close_prices.columns = close_prices.columns.get_level_values(0)
        returns_df = close_prices.pct_change().dropna()
        
        self._returns_data = returns_df
        logging.info(f"Calculated daily returns for {len(returns_df.columns)} tickers")
        
        return returns_df
    
    def get_baseline_weights(self, tickers: List[str]) -> np.ndarray:
        """
        Get baseline portfolio weights.
        
        Parameters:
        -----------
        tickers : List[str]
            List of ticker symbols
            
        Returns:
        --------
        np.ndarray of weights corresponding to ticker order
        """
        if self._tickers_df is None:
            raise ValueError("Tickers must be loaded first using load_tickers()")
        
        # Create weight array in ticker order
        weights = np.zeros(len(tickers))
        ticker_to_weight = dict(zip(self._tickers_df['Symbol'], self._tickers_df['Weight']))
        
        for i, ticker in enumerate(tickers):
            weights[i] = ticker_to_weight.get(ticker, 0.0)
        
        # If no weights found, use equal weights
        if weights.sum() == 0:
            weights = np.ones(len(tickers)) / len(tickers)
            logging.warning("No weights found in ticker file, using equal weights")
        else:
            # Renormalize to ensure sum = 1
            weights = weights / weights.sum()
        
        return weights
    
    # =============================================================================
    # COVARIANCE MATRIX CALCULATION METHODS
    # =============================================================================
    
    def calculate_covariance_sample(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Calculate sample covariance matrix.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Daily returns data
            
        Returns:
        --------
        np.ndarray covariance matrix
        """
        cov_matrix = returns.cov().values
        logging.info("Calculated sample covariance matrix")
        return cov_matrix
    
    def calculate_covariance_exponential_weighted(self, returns: pd.DataFrame, 
                                                alpha: float = 0.94) -> np.ndarray:
        """
        Calculate exponentially weighted covariance matrix.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Daily returns data
        alpha : float
            Decay factor (0 < alpha < 1)
            
        Returns:
        --------
        np.ndarray covariance matrix
        """
        ewm_cov = returns.ewm(alpha=alpha).cov().iloc[-len(returns.columns):].values
        logging.info(f"Calculated exponentially weighted covariance (alpha={alpha})")
        return ewm_cov
    
    def calculate_covariance_shrunk(self, returns: pd.DataFrame, 
                                  shrinkage: Optional[float] = None) -> np.ndarray:
        """
        Calculate shrunk covariance matrix using Ledoit-Wolf estimator.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Daily returns data
        shrinkage : float, optional
            Shrinkage intensity (auto-estimated if None)
            
        Returns:
        --------
        np.ndarray covariance matrix
        """
        lw = LedoitWolf(shrinkage=shrinkage)
        cov_matrix, shrinkage_used = lw.fit(returns.values).covariance_, lw.shrinkage_
        
        logging.info(f"Calculated Ledoit-Wolf shrunk covariance (shrinkage={shrinkage_used:.3f})")
        return cov_matrix
    
    def calculate_covariance_robust(self, returns: pd.DataFrame, 
                                  method: str = 'mcd') -> np.ndarray:
        """
        Calculate robust covariance matrix.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Daily returns data
        method : str
            Robust method ('mcd' for Minimum Covariance Determinant)
            
        Returns:
        --------
        np.ndarray covariance matrix
        """
        if method == 'mcd':
            robust_cov = MinCovDet().fit(returns.values)
            cov_matrix = robust_cov.covariance_
            logging.info("Calculated robust covariance using Minimum Covariance Determinant")
        else:
            raise ValueError(f"Unknown robust covariance method: {method}")
        
        return cov_matrix
    
    def calculate_covariance_factor_model(self, returns: pd.DataFrame, 
                                        factors: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Calculate factor model covariance matrix.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Daily returns data
        factors : pd.DataFrame, optional
            Factor returns (uses market factor if None)
            
        Returns:
        --------
        np.ndarray covariance matrix
        """
        if factors is None:
            # Use equal-weighted market factor as default
            market_factor = returns.mean(axis=1)
            factors = pd.DataFrame({'Market': market_factor})
        
        # Run factor model regression for each asset
        n_assets = len(returns.columns)
        factor_loadings = np.zeros((n_assets, len(factors.columns)))
        residual_vars = np.zeros(n_assets)
        
        for i, asset in enumerate(returns.columns):
            # Simple linear regression: returns = alpha + beta * factors + residual
            y = returns[asset].values
            X = np.column_stack([np.ones(len(factors)), factors.values])
            
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                factor_loadings[i] = beta[1:]  # Exclude intercept
                
                # Calculate residual variance
                predicted = X @ beta
                residuals = y - predicted
                residual_vars[i] = np.var(residuals)
            except np.linalg.LinAlgError:
                # Fallback to sample covariance for this asset
                factor_loadings[i] = 0
                residual_vars[i] = returns[asset].var()
        
        # Factor model covariance: B * F * B' + D
        # where B = factor loadings, F = factor covariance, D = diagonal residual variances
        factor_cov = factors.cov().values
        systematic_cov = factor_loadings @ factor_cov @ factor_loadings.T
        idiosyncratic_cov = np.diag(residual_vars)
        
        cov_matrix = systematic_cov + idiosyncratic_cov
        
        logging.info(f"Calculated factor model covariance using {len(factors.columns)} factors")
        return cov_matrix
    
    def calculate_covariance_sparse_inverse(self, returns: pd.DataFrame, 
                                          alpha: float = 0.1, 
                                          max_iters: int = 1000,
                                          eps_abs: float = 1e-4,
                                          eps_rel: float = 1e-4) -> np.ndarray:
        """
        Calculate sparse inverse covariance (precision) matrix using graphical lasso.
        
        This method estimates a sparse precision matrix (inverse covariance) using
        L1 regularization, which encourages sparsity in the inverse covariance matrix.
        This is useful for identifying conditional independence relationships between assets.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Daily returns data
        alpha : float, default=0.1
            L1 regularization parameter. Higher values = sparser precision matrix
        max_iters : int, default=1000
            Maximum number of iterations for optimization
        eps_abs : float, default=1e-4
            Absolute tolerance for convergence
        eps_rel : float, default=1e-4
            Relative tolerance for convergence
            
        Returns:
        --------
        np.ndarray
            Covariance matrix (inverse of the estimated sparse precision matrix)
        """
        try:
            # Get empirical covariance as starting point
            S = np.cov(returns.T)
            n_assets = S.shape[0]
            
            # Create CVXPY variables
            # Θ (Theta) is the precision matrix (inverse covariance)
            Theta = cp.Variable((n_assets, n_assets), symmetric=True)
            
            # Objective: Maximize log-likelihood - L1 penalty
            # log det(Θ) - tr(S @ Θ) - α * ||Θ||_1 (off-diagonal)
            
            # Create L1 penalty matrix (zero diagonal, ones off-diagonal)
            L1_mask = np.ones((n_assets, n_assets)) - np.eye(n_assets)
            
            objective = cp.Maximize(
                cp.log_det(Theta) - cp.trace(S @ Theta) - alpha * cp.norm(cp.multiply(L1_mask, Theta), 1)
            )
            
            # Constraints: Precision matrix must be positive definite
            constraints = [Theta >> 1e-8 * np.eye(n_assets)]  # Positive definite constraint
            
            # Create and solve problem
            problem = cp.Problem(objective, constraints)
            
            # Solve with appropriate solver
            try:
                problem.solve(solver=cp.SCS, max_iters=max_iters, eps=eps_abs, normalize=False, verbose=False)
            except cp.SolverError:
                # Fallback to ECOS if SCS fails
                try:
                    problem.solve(solver=cp.ECOS, max_iters=max_iters, abstol=eps_abs, reltol=eps_rel, verbose=False)
                except cp.SolverError:
                    # Final fallback to OSQP
                    problem.solve(solver=cp.OSQP, max_iter=max_iters, eps_abs=eps_abs, eps_rel=eps_rel, verbose=False)
            
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                logging.warning(f"Sparse inverse covariance optimization failed with status: {problem.status}")
                logging.warning("Falling back to sample covariance")
                return self.calculate_covariance_sample(returns)
            
            # Extract the precision matrix
            precision_matrix = Theta.value
            
            # Check if precision matrix is valid
            if precision_matrix is None:
                logging.warning("Failed to extract precision matrix, falling back to sample covariance")
                return self.calculate_covariance_sample(returns)
            
            # Ensure symmetry (numerical precision issues)
            precision_matrix = (precision_matrix + precision_matrix.T) / 2
            
            # Compute covariance matrix as inverse of precision matrix
            try:
                cov_matrix = np.linalg.inv(precision_matrix)
            except np.linalg.LinAlgError:
                logging.warning("Failed to invert precision matrix, falling back to sample covariance")
                return self.calculate_covariance_sample(returns)
            
            # Count number of zero elements in precision matrix (sparsity)
            sparsity = np.sum(np.abs(precision_matrix) < 1e-6) / (n_assets * n_assets)
            
            logging.info(f"Calculated sparse inverse covariance (α={alpha}, sparsity={sparsity:.1%})")
            
            # Store precision matrix for analysis (optional)
            self._last_precision_matrix = precision_matrix
            
            return cov_matrix
            
        except Exception as e:
            logging.error(f"Error in sparse inverse covariance calculation: {e}")
            logging.warning("Falling back to sample covariance")
            return self.calculate_covariance_sample(returns)
    
    def get_precision_matrix(self, returns: pd.DataFrame, **kwargs) -> Optional[np.ndarray]:
        """
        Get the precision matrix (inverse covariance) from sparse inverse method.
        
        This method calculates the sparse precision matrix and returns it directly,
        which is useful for analyzing conditional independence relationships.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Daily returns data
        **kwargs : dict
            Parameters for sparse inverse calculation (alpha, max_iters, etc.)
            
        Returns:
        --------
        np.ndarray or None
            Precision matrix if successful, None if failed
        """
        try:
            # Calculate sparse inverse covariance (this stores precision matrix)
            self.calculate_covariance_sparse_inverse(returns, **kwargs)
            
            # Return the stored precision matrix if available
            if hasattr(self, '_last_precision_matrix'):
                return self._last_precision_matrix
            else:
                logging.warning("Precision matrix not available")
                return None
                
        except Exception as e:
            logging.error(f"Error getting precision matrix: {e}")
            return None
    
    def get_covariance_matrix(self, returns: pd.DataFrame, 
                            method: str = 'sample', **kwargs) -> np.ndarray:
        """
        Calculate covariance matrix using specified method.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Daily returns data
        method : str
            Covariance estimation method:
            - 'sample': Sample covariance
            - 'exponential_weighted': Exponentially weighted
            - 'shrunk': Ledoit-Wolf shrinkage  
            - 'robust': Robust estimation (MCD)
            - 'factor_model': Factor model
            - 'sparse_inverse': Sparse inverse covariance (graphical lasso)
        **kwargs : dict
            Method-specific parameters
            
        Returns:
        --------
        np.ndarray covariance matrix
        """
        method_map = {
            'sample': self.calculate_covariance_sample,
            'exponential_weighted': self.calculate_covariance_exponential_weighted,
            'shrunk': self.calculate_covariance_shrunk,
            'robust': self.calculate_covariance_robust,
            'factor_model': self.calculate_covariance_factor_model,
            'sparse_inverse': self.calculate_covariance_sparse_inverse
        }
        
        if method not in method_map:
            available_methods = list(method_map.keys())
            raise ValueError(f"Unknown covariance method '{method}'. Available: {available_methods}")
        
        cov_func = method_map[method]
        cov_matrix = cov_func(returns, **kwargs)
        
        # Ensure positive semi-definite
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-8)  # Floor negative eigenvalues
        cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        logging.info(f"Covariance matrix calculated using '{method}' method")
        return cov_matrix
    
    def get_available_covariance_methods(self) -> List[str]:
        """Get list of available covariance calculation methods."""
        return ['sample', 'exponential_weighted', 'shrunk', 'robust', 'factor_model', 'sparse_inverse']
    
    def get_method_parameters(self, method: str) -> Dict[str, Any]:
        """
        Get parameters for specific covariance method.
        
        Parameters:
        -----------
        method : str
            Covariance method name
            
        Returns:
        --------
        Dict with parameter descriptions and defaults
        """
        method_params = {
            'sample': {},
            'exponential_weighted': {'alpha': 0.94},
            'shrunk': {'shrinkage': None},
            'robust': {'method': 'mcd'},
            'factor_model': {'factors': None}
        }
        
        return method_params.get(method, {})