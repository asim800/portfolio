# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python financial analysis system focused on portfolio optimization with dynamic rebalancing capabilities. The project implements multiple optimization strategies (vanilla mean-variance and robust optimization) with configurable mixed cash/equity portfolios. It uses modern Python financial libraries (yfinance, cvxpy, pandas, numpy, matplotlib) for downloading market data, calculating returns, and performing portfolio optimization.

## Development Commands

### Environment Setup
- Uses uv package manager with Python 3.12
- Virtual environment is managed in `.venv/`
- Install dependencies: `uv sync`
- Activate environment: `source .venv/bin/activate` or use `uv run`

### Running the Application
- Main entry point: `uv run main.py`
- Run with custom ticker file: `uv run main.py --file path/to/tickers.txt`
- Run single ticker analysis: `uv run main.py --ticker AAPL`
- Run dynamic rebalancing: Automatically runs when portfolio optimization is executed

### Project Structure
```
/
├── src/                           # Main source code
│   ├── main.py                   # Application entry point with portfolio optimization
│   ├── config.py                 # Configuration system for rebalancing parameters
│   ├── rebalancing_engine.py     # Core dynamic rebalancing logic
│   ├── performance_tracker.py    # Performance metrics collection and analysis
│   ├── rebalancing_visualization.py  # Visualization system for results
│   ├── performance_analysis.py   # Risk metrics and performance calculations
│   ├── pyproject.toml            # Project configuration and dependencies
│   └── .venv/                    # Virtual environment
├── data/                         # Data storage directory (pickle cache files)
├── logs/                         # Application logs
├── plots/rebalancing/            # Generated rebalancing plots and visualizations
├── results/rebalancing/          # CSV results and performance summaries
├── docs/                         # Documentation
└── tickers.txt                   # Default ticker file (format: SYMBOL, WEIGHT)
```

## Architecture Details

### Core Modules

#### Main Application (main.py)
- Command-line application with argparse for ticker file or single ticker analysis
- Implements modular portfolio optimization functions using cvxpy:
  - `optimize_portfolio_vanilla()` - Mean-variance optimization
  - `optimize_portfolio_robust()` - Robust optimization with uncertainty sets
- Integrates dynamic rebalancing system with configurable parameters
- Uses pickle-based data caching to avoid repeated API calls
- Generates comprehensive visualizations and performance analysis

#### Configuration System (config.py)
- `RebalancingConfig` dataclass with comprehensive parameters
- Default 30-day rebalancing periods with expanding window analysis
- Mixed portfolio configuration (40% cash default, 3% interest rate)
- Configurable optimization methods, risk parameters, and output settings
- Validation for all configuration parameters

#### Rebalancing Engine (rebalancing_engine.py)
- `RebalancingEngine` class orchestrating dynamic portfolio rebalancing
- Implements expanding window optimization with fallback strategies
- Supports multiple portfolio types: baseline, vanilla, robust, mixed portfolios
- Mixed portfolios combine optimized equity weights with cash component
- Tracks performance across calendar day periods with comprehensive metrics

#### Portfolio Tracking (portfolio_tracker.py)
- `PortfolioTracker` class for collecting and analyzing portfolio metrics using pandas DataFrames
- Calculates comprehensive metrics: Sharpe ratio, beta, drawdown, Calmar ratio
- Supports mixed portfolio tracking with cash/equity allocation percentages
- Exports results to CSV files for analysis

#### Visualization System (rebalancing_visualization.py)
- `RebalancingVisualizer` for creating comprehensive performance charts
- Dynamic rebalancing comparison plots with datetime x-axis
- Visual markers for rebalancing events and portfolio comparison
- Distinct styling for different portfolio types (solid/dashed lines)
- Mixed portfolio labels showing cash allocation percentages

### Portfolio Types

The system supports 5 different portfolio types:

1. **Baseline Portfolio**: Buy-and-hold with fixed weights from tickers.txt
2. **Vanilla Portfolio**: Mean-variance optimization (rebalanced every 30 days)
3. **Robust Portfolio**: Robust optimization with uncertainty sets (rebalanced every 30 days)
4. **Mixed Vanilla Portfolio**: 40% cash + 60% vanilla optimized equity (rebalanced every 30 days)
5. **Mixed Robust Portfolio**: 40% cash + 60% robust optimized equity (rebalanced every 30 days)

### Data Flow
1. Reads ticker symbols and weights from CSV file (format: Symbol,Weight with headers)
2. Downloads historical daily data from Yahoo Finance (cached with pickle)
3. Constructs multi-index DataFrame with proper ticker structure
4. Splits timeline into 30-day rebalancing periods
5. For each period:
   - Gets expanding window of historical data
   - Optimizes vanilla and robust portfolios using cvxpy
   - Creates mixed portfolios with cash component
   - Calculates performance metrics for all portfolios
   - Tracks cumulative returns and risk metrics
6. Generates visualization and exports results to CSV

### Key Dependencies
- **cvxpy**: Convex optimization for portfolio optimization
- **yfinance**: Yahoo Finance data download
- **pandas/numpy**: Data manipulation and numerical computing
- **matplotlib**: Plotting and visualization
- **ipdb**: Interactive debugging

### Configuration
- Default date range: 2024-01-01 to 2024-12-31
- Default ticker file: `../tickers.txt` (with headers: Symbol,Weight)
- Rebalancing period: 30 calendar days
- Mixed portfolio cash allocation: 40%
- Cash interest rate: 3% annually
- Results saved to: `../results/rebalancing/`
- Plots saved to: `../plots/rebalancing/`

### Key Features
- **Dynamic Rebalancing**: Portfolios rebalance every 30 days using expanding window data
- **Multiple Optimization Methods**: Vanilla mean-variance and robust optimization
- **Mixed Portfolios**: Conservative cash/equity blend with configurable allocation
- **Performance Tracking**: Comprehensive metrics including Sharpe ratio, beta, drawdown
- **Visualization**: Interactive plots with rebalancing markers and datetime axes
- **Data Caching**: Pickle files avoid repeated Yahoo Finance API calls
- **Configurable Plotting**: Non-blocking plots for simultaneous viewing

## Notes for Development
- All optimization uses cvxpy with proper error handling and fallback strategies
- Mixed portfolios inherit equity weights from base optimization methods
- Cash component grows at configured interest rate between rebalancing periods
- Performance analysis includes market beta calculation using SPY/equal-weight proxy
- Visualization uses distinct colors and line styles for easy portfolio identification