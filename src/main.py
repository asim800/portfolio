#!/usr/bin/env python3
"""
Refactored Main Entry Point for Portfolio Optimization System.
Uses new orchestrator pattern with FinData, PortfolioOptimizer, and RebalancingEngine.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5 backend for interactive plots

# IPython autoreload for module development
try:
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic('load_ext', 'autoreload')
        ipython.run_line_magic('autoreload', '2')
except (ImportError, AttributeError):
    pass

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import our new modular components
from fin_data import FinData
from portfolio_optimizer import PortfolioOptimizer
from config import RebalancingConfig, create_custom_config
from performance_engine import PerformanceEngine
from rebalancing_visualization import RebalancingVisualizer

# Import performance analysis functions
try:
    from performance_analysis import (
        annualized_return, annualized_standard_deviation, max_drawdown, 
        gain_to_pain_ratio, calmar_ratio, sharpe_ratio, sortino_ratio
    )
except ImportError:
    logging.warning("Could not import performance_analysis module - using basic metrics only")


class PortfolioOrchestrator:
    """
    Main orchestrator class that coordinates data, optimization, and performance analysis.

    Manages the workflow between FinData, PortfolioOptimizer, and PerformanceEngine
    to provide a complete portfolio analysis and optimization system.
    """
    
    def __init__(self, config: RebalancingConfig):
        """
        Initialize orchestrator with configuration.
        
        Parameters:
        -----------
        config : RebalancingConfig
            Configuration for optimization and rebalancing
        """
        self.config = config
        
        # Initialize components
        self.fin_data = FinData(
            start_date=config.start_date,
            end_date=config.end_date,
            cache_dir="../data"
        )
        
        self.performance_engine = PerformanceEngine(
            data=self.fin_data,
            config=config
        )
        
        # Data storage
        self.tickers_df: pd.DataFrame = None
        self.returns_df: pd.DataFrame = None
        self.baseline_weights: np.ndarray = None
        
        logging.info("PortfolioOrchestrator initialized")
    
    def load_data(self, ticker_file: str) -> None:
        """
        Load ticker data and market data.
        
        Parameters:
        -----------
        ticker_file : str
            Path to ticker file
        """
        logging.info("=== Loading Data ===")
        
        # Load tickers and weights
        self.tickers_df = self.fin_data.load_tickers(ticker_file)
        tickers = self.tickers_df['Symbol'].tolist()

        # Load price data and calculate returns first to get correct column order
        price_data = self.fin_data.get_price_data(tickers)
        self.returns_df = self.fin_data.get_returns_data(tickers)

        # Get baseline weights using the actual returns data column order
        returns_column_order = list(self.returns_df.columns)
        self.baseline_weights = self.fin_data.get_baseline_weights(returns_column_order)
        
        logging.info(f"Data loaded: {len(tickers)} assets, {len(self.returns_df)} trading days")
        logging.info(f"Date range: {self.returns_df.index[0].date()} to {self.returns_df.index[-1].date()}")
    
    def run_static_optimization(self) -> dict:
        """
        Run static portfolio optimization for comparison.
        
        Returns:
        --------
        Dict with optimization results for each method
        """
        logging.info("=== Static Portfolio Optimization ===")
        
        if self.returns_df is None:
            raise ValueError("Data must be loaded first")
        
        # Calculate mean returns and covariance matrix (pandas)
        mean_returns = self.returns_df.mean()
        cov_matrix = self.fin_data.get_covariance_matrix(
            self.returns_df,
            method=self.config.covariance_method,
            **self.config.get_covariance_params()
        )

        # Run optimization for each configured method
        results = {}
        optimizer = self.performance_engine.optimizer

        for method in self.config.optimization_methods:
            logging.info(f"Running {method} optimization...")

            # Get method-specific parameters
            normalized_method = self.config.normalize_method_name(method)
            opt_params = self.config.get_optimization_params(method)

            # Run optimization (convert to numpy at boundary)
            result = optimizer.optimize(
                method=normalized_method,
                mean_returns=mean_returns.values,  # pandas → numpy
                cov_matrix=cov_matrix.values,      # pandas → numpy
                **opt_params
            )
            
            results[method] = result
            
            # Log results
            if result.get('status') == 'optimal':
                logging.info(f"{method} optimization successful!")
                logging.info(f"  Expected Return: {result.get('expected_return', 0):.2%}")
                logging.info(f"  Volatility: {result.get('volatility', 0):.2%}")
                logging.info(f"  Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}")
                
                # Show top holdings
                weights = result['weights']
                portfolio = pd.Series(weights, index=self.returns_df.columns).sort_values(ascending=False)
                logging.info(f"  Top 5 Holdings:")
                for ticker, weight in portfolio.head().items():
                    logging.info(f"    {ticker}: {weight:.2%}")
            else:
                logging.error(f"{method} optimization failed: {result.get('message', 'Unknown error')}")
        
        return results
    
    def run_dynamic_rebalancing(self) -> None:
        """
        Run dynamic rebalancing backtest.
        """
        logging.info("=== Dynamic Rebalancing ===")
        
        if self.returns_df is None:
            raise ValueError("Data must be loaded first")
        
        # Load data into performance engine
        self.performance_engine.load_data(self.returns_df, self.baseline_weights)

        # Run backtest
        logging.info("Starting dynamic portfolio backtest...")
        try:
            self.performance_engine.run_backtest(self.config.start_date, self.config.end_date)
            logging.info("Backtest completed successfully")
        except Exception as e:
            logging.error(f"Backtest failed: {e}")
            raise

        # Get results
        performance_tracker = self.performance_engine.get_results()

        # Generate visualizations
        logging.info("Generating visualizations...")
        self.performance_engine.plot_cumulative_returns()
        self.performance_engine.plot_daily_cumulative_returns()
        self.performance_engine.plot_performance_summary()
        self.performance_engine.plot_max_drawdown_time_series()

        # Generate comparison plots if configured
        if hasattr(self.config, 'comparison_pairs') and self.config.comparison_pairs:
            logging.info("Generating portfolio comparison plots...")
            for portfolio1, portfolio2 in self.config.comparison_pairs:
                try:
                    self.performance_engine.plot_portfolio_comparison(portfolio1, portfolio2)
                    logging.info(f"Created comparison plot: {portfolio1} vs {portfolio2}")
                except Exception as e:
                    logging.warning(f"Could not create comparison plot for {portfolio1} vs {portfolio2}: {e}")

        # Check if plots were created successfully
        import matplotlib.pyplot as plt
        num_figures = len(plt.get_fignums())
        if num_figures > 0:
            logging.info(f"Successfully created {num_figures} plots")
        else:
            logging.warning("No plots were created")

        # Save results
        self.performance_engine.export_results()
        self.performance_engine.create_performance_tables()

        # Display summary statistics
        summary_stats = self.performance_engine.get_performance_metrics()
        self._display_summary_stats(summary_stats)

        # Print portfolio summary tables
        logging.info("=== Portfolio Summary Tables ===")
        self.performance_engine.print_portfolio_summary_tables(max_rows=10)
    
    def _display_summary_stats(self, summary_stats: pd.DataFrame) -> None:
        """Display summary statistics for all portfolios."""
        logging.info("=== Portfolio Performance Summary ===")

        if summary_stats.empty:
            logging.warning("No performance statistics available")
            return

        for portfolio_name in summary_stats.index:
            logging.info(f"\n{portfolio_name.upper()} Portfolio:")

            # Display available metrics
            if 'Total_Return' in summary_stats.columns:
                logging.info(f"  Total Return: {summary_stats.loc[portfolio_name, 'Total_Return']:.2%}")
            if 'Annual_Return' in summary_stats.columns:
                logging.info(f"  Annual Return: {summary_stats.loc[portfolio_name, 'Annual_Return']:.2%}")
            if 'Volatility' in summary_stats.columns:
                logging.info(f"  Volatility: {summary_stats.loc[portfolio_name, 'Volatility']:.2%}")
            if 'Sharpe_Ratio' in summary_stats.columns:
                logging.info(f"  Sharpe Ratio: {summary_stats.loc[portfolio_name, 'Sharpe_Ratio']:.3f}")
            if 'Max_Drawdown' in summary_stats.columns:
                logging.info(f"  Max Drawdown: {summary_stats.loc[portfolio_name, 'Max_Drawdown']:.2%}")
            if 'Num_Periods' in summary_stats.columns:
                logging.info(f"  Number of Periods: {summary_stats.loc[portfolio_name, 'Num_Periods']:.0f}")

        # Display comparison table
        logging.info(f"\n=== Performance Comparison Table ===")
        logging.info(f"\n{summary_stats.to_string()}")
    
    def compare_methods(self, static_results: dict) -> None:
        """
        Compare static optimization methods.

        Parameters:
        -----------
        static_results : dict
            Results from run_static_optimization()
        """
        logging.info("=== Method Comparison ===")

        # Debug: Check static results structure
        logging.debug(f"Static results keys: {list(static_results.keys())}")
        for method, result in static_results.items():
            logging.debug(f"{method} status: {result.get('status', 'unknown')}")

        # Create comparison DataFrame
        comparison_df = self.performance_engine.optimizer.compare_portfolios(static_results)

        if not comparison_df.empty and 'Portfolio' in comparison_df.columns:
            logging.info("\nPortfolio Comparison:")
            for _, row in comparison_df.iterrows():
                logging.info(f"{row['Portfolio']}:")
                logging.info(f"  Return: {row['Expected Return']:.2%}, Vol: {row['Volatility']:.2%}, Sharpe: {row['Sharpe Ratio']:.3f}")
        else:
            if comparison_df.empty:
                logging.info("No successful optimizations to compare")
            else:
                logging.warning(f"Comparison DataFrame missing 'Portfolio' column. Available columns: {comparison_df.columns.tolist()}")
    
    def run_full_analysis(self, ticker_file: str) -> None:
        """
        Run complete portfolio analysis workflow.
        
        Parameters:
        -----------
        ticker_file : str
            Path to ticker file
        """
        try:
            # Load data
            logging.info("Step 1: Loading data...")
            self.load_data(ticker_file)

            # Run static optimization
            logging.info("Step 2: Running static optimization...")
            static_results = self.run_static_optimization()

            # Compare optimization methods
            logging.info("Step 3: Comparing methods...")
            self.compare_methods(static_results)

            # Run dynamic rebalancing
            logging.info("Step 4: Running dynamic rebalancing...")
            self.run_dynamic_rebalancing()

            logging.info("=== Analysis Complete ===")
            
        except Exception as e:
            logging.error(f"Analysis failed: {e}")
            raise


def main():
    """Main entry point for the refactored application."""
    print("MAIN FUNCTION CALLED - STARTING SETUP")

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('../logs/portfolio_analysis.log', 'a')
        ]
    )

    logging.info("=== MAIN FUNCTION STARTED ===")
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Portfolio Optimization System')
    parser.add_argument('--file', '--tickers', dest='ticker_file', 
                       default='../tickers.txt',
                       help='Ticker file to use (default: ../tickers.txt)')
    parser.add_argument('--start-date', dest='start_date', 
                       default='2024-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', dest='end_date', 
                       default='2024-12-31',
                       help='End date (YYYY-MM-DD)')
    # Single portfolio mode (default)
    parser.add_argument('--optimization-method', dest='optimization_method',
                       choices=['mean_variance', 'robust_mean_variance', 'risk_parity', 'min_variance', 'max_sharpe', 'max_diversification'],
                       default='mean_variance',
                       help='Single optimization method to use (default mode)')
    
    # Comparison mode (optional)
    parser.add_argument('--comparison-mode', dest='comparison_mode',
                       action='store_true',
                       help='Enable comparison mode to run multiple optimization methods')
    parser.add_argument('--optimization-methods', dest='optimization_methods',
                       nargs='+',
                       default=['mean_variance', 'robust_mean_variance'],
                       help='Multiple optimization methods for comparison mode')
    parser.add_argument('--covariance-method', dest='covariance_method',
                       choices=['sample', 'exponential_weighted', 'shrunk', 'robust', 'factor_model'],
                       default='sample',
                       help='Covariance calculation method')
    parser.add_argument('--risk-aversion', dest='risk_aversion',
                       type=float, default=1.0,
                       help='Risk aversion parameter')
    parser.add_argument('--max-weight', dest='max_weight',
                       type=float, default=0.4,
                       help='Maximum weight per asset')
    parser.add_argument('--min-weight', dest='min_weight',
                       type=float, default=0.001,
                       help='Minimum weight per asset')
    parser.add_argument('--rebalancing-days', dest='rebalancing_period_days',
                       type=int, default=30,
                       help='Rebalancing period in days')
    
    args = parser.parse_args()
    
    # Determine optimization methods based on mode
    if args.comparison_mode:
        optimization_methods = args.optimization_methods
        single_portfolio_mode = False
        comparison_mode = True
    else:
        optimization_methods = [args.optimization_method]
        single_portfolio_mode = True
        comparison_mode = False
    
    # Create configuration
    config = create_custom_config(
        start_date=args.start_date,
        end_date=args.end_date,
        ticker_file=args.ticker_file,
        optimization_methods=optimization_methods,
        single_portfolio_mode=single_portfolio_mode,
        comparison_mode=comparison_mode,
        covariance_method=args.covariance_method,
        risk_aversion=args.risk_aversion,
        max_weight=args.max_weight,
        min_weight=args.min_weight,
        rebalancing_period_days=args.rebalancing_period_days
    )
    
    logging.info("=== Portfolio Optimization System Started ===")
    logging.info(f"Configuration:")
    logging.info(f"  Mode: {'Comparison' if config.comparison_mode else 'Single Portfolio'}")
    logging.info(f"  Date range: {config.start_date} to {config.end_date}")
    logging.info(f"  Optimization methods: {config.optimization_methods}")
    logging.info(f"  Covariance method: {config.covariance_method}")
    logging.info(f"  Rebalancing period: {config.rebalancing_period_days} days")
    logging.info(f"  Weight constraints: {config.min_weight:.1%} - {config.max_weight:.1%}")
    logging.info(f"  Baselines: Static={config.include_baseline}, Rebalanced={config.include_rebalanced_baseline}")
    
    try:
        # Initialize orchestrator
        orchestrator = PortfolioOrchestrator(config)
        
        # Run full analysis
        orchestrator.run_full_analysis(args.ticker_file)

        return orchestrator
        
    except KeyboardInterrupt:
        logging.info("Analysis interrupted by user")
        sys.exit(0)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    orch = main()