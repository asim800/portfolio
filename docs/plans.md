# Retirement Dashboard Implementation Plan

## Overview

**Goal**: Build a Monte Carlo-based retirement planning dashboard that simulates portfolio performance over retirement periods with various withdrawal strategies and portfolio optimization approaches.

**Approach**: MC-First Strategy - Build the core Monte Carlo engine first with simple fixed withdrawals, validate it works correctly, then add sophisticated withdrawal strategies and features.

**Timeline**: 7 days for Phase 1 (working MC retirement engine)

---

## Phase 1: Monte Carlo Retirement Engine

### Stage 1: Core Configuration (Day 1)

#### Task 1.1: Create `retirement_config.py` - minimal version

**Goal**: Just enough config to run a basic simulation

**Implementation**:
```python
from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime

@dataclass
class RetirementConfig:
    """Minimal configuration for retirement simulation"""

    # Portfolio
    initial_portfolio: float  # Starting portfolio value (e.g., 1_000_000)
    current_portfolio: Dict[str, float]  # Ticker: weight mapping

    # Timeline
    start_date: str  # Simulation start (YYYY-MM-DD)
    end_date: str  # Simulation end
    retirement_date: Optional[str] = None  # When withdrawals begin (None = already retired)

    # Withdrawals (simple for now)
    annual_withdrawal: float  # Fixed annual withdrawal amount
    inflation_rate: float = 0.03  # Annual inflation rate

    # Strategy
    rebalancing_strategy: str = 'buy_and_hold'  # For now: buy_and_hold, target_weight, equal_weight

    def __post_init__(self):
        """Validate configuration"""
        # Validate dates
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')

        if end <= start:
            raise ValueError("end_date must be after start_date")

        if self.retirement_date:
            retire = datetime.strptime(self.retirement_date, '%Y-%m-%d')
            if retire < start or retire > end:
                raise ValueError("retirement_date must be between start_date and end_date")

        # Validate portfolio
        if self.initial_portfolio <= 0:
            raise ValueError("initial_portfolio must be positive")

        if not self.current_portfolio:
            raise ValueError("current_portfolio cannot be empty")

        weight_sum = sum(self.current_portfolio.values())
        if not (0.99 <= weight_sum <= 1.01):  # Allow small rounding errors
            raise ValueError(f"Portfolio weights must sum to 1.0, got {weight_sum}")

        # Validate withdrawal
        if self.annual_withdrawal < 0:
            raise ValueError("annual_withdrawal cannot be negative")

        if not (0 <= self.inflation_rate <= 0.20):
            raise ValueError("inflation_rate must be between 0 and 0.20")

    @property
    def num_years(self) -> int:
        """Calculate number of years in simulation"""
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')
        return (end - start).days // 365

    @property
    def tickers(self) -> list:
        """Get list of tickers in portfolio"""
        return list(self.current_portfolio.keys())
```

**Test**:
```python
# test_retirement_config.py
import pytest
from retirement_config import RetirementConfig

def test_basic_config_valid():
    config = RetirementConfig(
        initial_portfolio=1_000_000,
        start_date='2024-01-01',
        end_date='2054-01-01',
        retirement_date='2024-01-01',
        annual_withdrawal=40_000,
        current_portfolio={'SPY': 0.6, 'AGG': 0.4}
    )
    assert config.num_years == 30
    assert config.tickers == ['SPY', 'AGG']

def test_invalid_dates():
    with pytest.raises(ValueError, match="end_date must be after start_date"):
        RetirementConfig(
            initial_portfolio=1_000_000,
            start_date='2054-01-01',
            end_date='2024-01-01',  # Before start!
            annual_withdrawal=40_000,
            current_portfolio={'SPY': 1.0}
        )

def test_invalid_portfolio_weights():
    with pytest.raises(ValueError, match="weights must sum to 1.0"):
        RetirementConfig(
            initial_portfolio=1_000_000,
            start_date='2024-01-01',
            end_date='2054-01-01',
            annual_withdrawal=40_000,
            current_portfolio={'SPY': 0.5, 'AGG': 0.3}  # Sum = 0.8, not 1.0!
        )

def test_negative_withdrawal():
    with pytest.raises(ValueError, match="annual_withdrawal cannot be negative"):
        RetirementConfig(
            initial_portfolio=1_000_000,
            start_date='2024-01-01',
            end_date='2054-01-01',
            annual_withdrawal=-1000,
            current_portfolio={'SPY': 1.0}
        )
```

**Run**:
```bash
uv run pytest test_retirement_config.py -v
```

---

### Stage 2: Return Sampling for Monte Carlo (Day 1-2)

#### Task 2.1: Extend `fin_data.py` with return sampling

**Goal**: Generate random return paths from historical data for Monte Carlo simulation

**Implementation**:
Add these methods to the `FinData` class in `fin_data.py`:

```python
def sample_return_path(self,
                       tickers: List[str],
                       num_days: int,
                       method: str = 'bootstrap',
                       seed: Optional[int] = None) -> pd.DataFrame:
    """
    Sample synthetic return path for Monte Carlo simulation.

    Parameters:
    -----------
    tickers : List[str]
        List of ticker symbols
    num_days : int
        Number of days to sample
    method : str
        Sampling method: 'bootstrap' (resample with replacement) or 'parametric' (fit normal)
    seed : Optional[int]
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        Sampled returns with same structure as historical returns
    """
    if seed is not None:
        np.random.seed(seed)

    # Get historical returns
    historical_returns = self.get_returns_data(tickers)

    if method == 'bootstrap':
        # Bootstrap: randomly sample days with replacement
        sampled_indices = np.random.choice(
            len(historical_returns),
            size=num_days,
            replace=True
        )
        sampled_returns = historical_returns.iloc[sampled_indices].reset_index(drop=True)
        return sampled_returns

    elif method == 'parametric':
        # Parametric: fit multivariate normal and sample
        mean_returns = historical_returns.mean()
        cov_matrix = historical_returns.cov()

        # Sample from multivariate normal
        sampled_array = np.random.multivariate_normal(
            mean_returns.values,
            cov_matrix.values,
            size=num_days
        )

        sampled_returns = pd.DataFrame(
            sampled_array,
            columns=tickers
        )
        return sampled_returns

    else:
        raise ValueError(f"Unknown sampling method: {method}")

def sample_annual_returns(self,
                         tickers: List[str],
                         num_years: int,
                         method: str = 'bootstrap',
                         seed: Optional[int] = None) -> pd.DataFrame:
    """
    Sample annual return paths (convenience method).

    Returns DataFrame with num_years rows, one per year.
    """
    # Assume 252 trading days per year
    daily_returns = self.sample_return_path(
        tickers,
        num_days=num_years * 252,
        method=method,
        seed=seed
    )

    # Aggregate to annual returns
    annual_returns = []
    for year in range(num_years):
        start_idx = year * 252
        end_idx = start_idx + 252
        year_daily = daily_returns.iloc[start_idx:end_idx]

        # Calculate annual return: (1+r1)*(1+r2)*...*(1+r252) - 1
        year_return = (1 + year_daily).prod() - 1
        annual_returns.append(year_return)

    return pd.DataFrame(annual_returns, columns=tickers)
```

**Test**:
```python
# test_fin_data_sampling.py
import pytest
import numpy as np
import pandas as pd
from fin_data import FinData

def test_bootstrap_sampling_basic():
    """Test basic bootstrap sampling"""
    fin_data = FinData(start_date='2020-01-01', end_date='2024-01-01', cache_dir='../data')

    # Load data first
    fin_data.get_returns_data(['SPY', 'AGG'])

    # Sample 252 days (1 year)
    sampled = fin_data.sample_return_path(['SPY', 'AGG'], num_days=252, seed=42)

    assert len(sampled) == 252
    assert list(sampled.columns) == ['SPY', 'AGG']
    assert sampled.isna().sum().sum() == 0  # No NaN values

def test_bootstrap_reproducibility():
    """Test that same seed gives same results"""
    fin_data = FinData(start_date='2020-01-01', end_date='2024-01-01', cache_dir='../data')
    fin_data.get_returns_data(['SPY', 'AGG'])

    # Sample with same seed twice
    sampled1 = fin_data.sample_return_path(['SPY', 'AGG'], num_days=252, seed=42)
    sampled2 = fin_data.sample_return_path(['SPY', 'AGG'], num_days=252, seed=42)

    assert sampled1.equals(sampled2)

def test_bootstrap_statistical_properties():
    """Test that bootstrap preserves statistical properties"""
    fin_data = FinData(start_date='2015-01-01', end_date='2024-01-01', cache_dir='../data')
    historical = fin_data.get_returns_data(['SPY'])

    # Sample many times and check convergence
    sample_means = []
    for i in range(100):
        sampled = fin_data.sample_return_path(['SPY'], num_days=252, seed=i)
        sample_means.append(sampled['SPY'].mean())

    # Mean of sample means should be close to historical mean
    assert abs(np.mean(sample_means) - historical['SPY'].mean()) < 0.0005

def test_annual_returns_aggregation():
    """Test annual returns aggregation"""
    fin_data = FinData(start_date='2020-01-01', end_date='2024-01-01', cache_dir='../data')
    fin_data.get_returns_data(['SPY', 'AGG'])

    annual_returns = fin_data.sample_annual_returns(['SPY', 'AGG'], num_years=10, seed=42)

    assert len(annual_returns) == 10  # 10 years
    assert list(annual_returns.columns) == ['SPY', 'AGG']

    # Annual returns should be larger magnitude than daily
    assert annual_returns['SPY'].std() > 0.05  # Typical annual volatility

def test_parametric_sampling():
    """Test parametric sampling"""
    fin_data = FinData(start_date='2020-01-01', end_date='2024-01-01', cache_dir='../data')
    fin_data.get_returns_data(['SPY', 'AGG'])

    sampled = fin_data.sample_return_path(['SPY', 'AGG'], num_days=252, method='parametric', seed=42)

    assert len(sampled) == 252
    assert list(sampled.columns) == ['SPY', 'AGG']
```

**Run**:
```bash
uv run pytest test_fin_data_sampling.py -v
```

---

### Stage 3: Single-Path Retirement Simulator (Day 2)

#### Task 3.1: Create `retirement_engine.py` - single simulation

**Goal**: Simulate ONE retirement path (deterministic or with given returns)

**Implementation**:

```python
# retirement_engine.py
import numpy as np
import pandas as pd
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from retirement_config import RetirementConfig
from fin_data import FinData

@dataclass
class SimulationPath:
    """Results from a single simulation path"""
    years: List[int]  # Year numbers (0, 1, 2, ...)
    dates: List[datetime]  # Actual dates
    portfolio_values: List[float]  # Portfolio value at start of each year
    withdrawals: List[float]  # Withdrawal amount each year
    annual_returns: List[float]  # Portfolio return each year
    success: bool  # Did portfolio survive?
    depletion_year: Optional[int] = None  # Year when portfolio depleted (if applicable)
    final_value: float = 0.0  # Final portfolio value


class RetirementEngine:
    """
    Monte Carlo retirement simulation engine.

    Simulates portfolio performance during retirement with withdrawals.
    """

    def __init__(self, config: RetirementConfig, fin_data: Optional[FinData] = None):
        """
        Initialize retirement engine.

        Parameters:
        -----------
        config : RetirementConfig
            Retirement configuration
        fin_data : Optional[FinData]
            FinData instance (created if not provided)
        """
        self.config = config

        # Initialize FinData if not provided
        if fin_data is None:
            self.fin_data = FinData(
                start_date=config.start_date,
                end_date=config.end_date,
                cache_dir='../data'
            )
            # Load historical data
            self.fin_data.get_returns_data(config.tickers)
        else:
            self.fin_data = fin_data

        logging.info(f"RetirementEngine initialized: {config.num_years} years, "
                    f"{len(config.tickers)} assets")

    def run_single_path(self, annual_returns: pd.DataFrame) -> SimulationPath:
        """
        Simulate a single retirement path with given returns.

        Parameters:
        -----------
        annual_returns : pd.DataFrame
            Annual returns for each asset (rows=years, cols=tickers)

        Returns:
        --------
        SimulationPath with detailed tracking data
        """
        # Initialize tracking
        portfolio_value = self.config.initial_portfolio
        weights = np.array([self.config.current_portfolio[t] for t in self.config.tickers])

        years = []
        dates = []
        portfolio_values = []
        withdrawals = []
        portfolio_returns = []

        start_date = datetime.strptime(self.config.start_date, '%Y-%m-%d')

        # Simulate each year
        for year_num in range(self.config.num_years):
            current_date = start_date + timedelta(days=year_num * 365)

            # Record start-of-year values
            years.append(year_num)
            dates.append(current_date)
            portfolio_values.append(portfolio_value)

            # Calculate portfolio return for this year
            asset_returns = annual_returns.iloc[year_num]
            portfolio_return = np.dot(weights, asset_returns.values)
            portfolio_returns.append(portfolio_return)

            # Apply return
            portfolio_value *= (1 + portfolio_return)

            # Calculate inflation-adjusted withdrawal
            inflation_adjusted_withdrawal = (
                self.config.annual_withdrawal *
                (1 + self.config.inflation_rate) ** year_num
            )
            withdrawals.append(inflation_adjusted_withdrawal)

            # Subtract withdrawal
            portfolio_value -= inflation_adjusted_withdrawal

            # Check for depletion
            if portfolio_value <= 0:
                return SimulationPath(
                    years=years,
                    dates=dates,
                    portfolio_values=portfolio_values,
                    withdrawals=withdrawals,
                    annual_returns=portfolio_returns,
                    success=False,
                    depletion_year=year_num,
                    final_value=0.0
                )

        # Successful completion
        return SimulationPath(
            years=years,
            dates=dates,
            portfolio_values=portfolio_values,
            withdrawals=withdrawals,
            annual_returns=portfolio_returns,
            success=True,
            depletion_year=None,
            final_value=portfolio_value
        )
```

**Test**:
```python
# test_retirement_engine.py
import pytest
import numpy as np
import pandas as pd
from retirement_config import RetirementConfig
from retirement_engine import RetirementEngine, SimulationPath

@pytest.fixture
def basic_config():
    return RetirementConfig(
        initial_portfolio=1_000_000,
        start_date='2024-01-01',
        end_date='2034-01-01',  # 10 years
        annual_withdrawal=40_000,
        current_portfolio={'SPY': 0.6, 'AGG': 0.4}
    )

def test_single_path_no_depletion(basic_config):
    """Test path with good returns - should not deplete"""
    engine = RetirementEngine(basic_config)

    # Create optimistic return path (8% per year for both assets)
    annual_returns = pd.DataFrame({
        'SPY': [0.08] * 10,
        'AGG': [0.08] * 10
    })

    result = engine.run_single_path(annual_returns)

    assert result.success == True
    assert result.depletion_year is None
    assert len(result.portfolio_values) == 10
    assert result.final_value > 1_000_000  # Should grow despite withdrawals

def test_single_path_with_depletion(basic_config):
    """Test path with poor returns - should deplete"""
    # Modify config for faster depletion
    basic_config.initial_portfolio = 500_000
    basic_config.annual_withdrawal = 100_000  # Withdrawing 20%!

    engine = RetirementEngine(basic_config)

    # Poor returns (-5% per year)
    annual_returns = pd.DataFrame({
        'SPY': [-0.05] * 10,
        'AGG': [-0.05] * 10
    })

    result = engine.run_single_path(annual_returns)

    assert result.success == False
    assert result.depletion_year is not None
    assert result.depletion_year < 10  # Depleted before end
    assert result.final_value == 0.0

def test_withdrawal_inflation_adjustment(basic_config):
    """Test that withdrawals adjust for inflation"""
    engine = RetirementEngine(basic_config)

    # Flat returns
    annual_returns = pd.DataFrame({
        'SPY': [0.0] * 10,
        'AGG': [0.0] * 10
    })

    result = engine.run_single_path(annual_returns)

    # First withdrawal should be base amount
    assert result.withdrawals[0] == 40_000

    # Second withdrawal should be inflated
    expected_year2 = 40_000 * (1 + basic_config.inflation_rate)
    assert result.withdrawals[1] == pytest.approx(expected_year2)

    # 10th withdrawal should compound
    expected_year10 = 40_000 * (1 + basic_config.inflation_rate) ** 9
    assert result.withdrawals[9] == pytest.approx(expected_year10)

def test_portfolio_weighting(basic_config):
    """Test that portfolio returns use correct weights"""
    engine = RetirementEngine(basic_config)

    # Different returns for each asset
    annual_returns = pd.DataFrame({
        'SPY': [0.10] * 10,  # Stocks: 10%
        'AGG': [0.02] * 10   # Bonds: 2%
    })

    result = engine.run_single_path(annual_returns)

    # Expected portfolio return: 0.6*0.10 + 0.4*0.02 = 0.068 (6.8%)
    expected_return = 0.6 * 0.10 + 0.4 * 0.02
    assert result.annual_returns[0] == pytest.approx(expected_return)
```

**Run**:
```bash
uv run pytest test_retirement_engine.py::test_single_path -v
```

---

### Stage 4: Monte Carlo Engine (Day 3)

#### Task 4.1: Implement `run_monte_carlo()` method

**Goal**: Run N simulations with random return paths and aggregate results

**Implementation**:
Add to `retirement_engine.py`:

```python
@dataclass
class MonteCarloResults:
    """Aggregated results from Monte Carlo simulation"""
    num_simulations: int
    success_rate: float  # Fraction of simulations that succeeded
    paths: List[SimulationPath]  # All simulation paths
    final_values: np.ndarray  # Final portfolio values (0 for failed paths)

    # Percentile statistics
    percentiles: dict  # 5th, 25th, 50th, 75th, 95th percentiles
    median_final_value: float
    mean_final_value: float
    std_final_value: float

    # Portfolio values over time (for plotting)
    portfolio_values_matrix: np.ndarray  # Shape: (num_sims, num_years)


class RetirementEngine:
    # ... existing code ...

    def run_monte_carlo(self,
                       num_simulations: int = 5000,
                       method: str = 'bootstrap',
                       seed: Optional[int] = None,
                       show_progress: bool = True) -> MonteCarloResults:
        """
        Run Monte Carlo retirement simulation.

        Parameters:
        -----------
        num_simulations : int
            Number of simulations to run
        method : str
            Return sampling method: 'bootstrap' or 'parametric'
        seed : Optional[int]
            Random seed for reproducibility
        show_progress : bool
            Whether to show progress bar

        Returns:
        --------
        MonteCarloResults with aggregated statistics
        """
        logging.info(f"Starting Monte Carlo simulation: {num_simulations} paths")

        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Storage for results
        all_paths = []
        final_values = []
        success_count = 0
        portfolio_values_matrix = []

        # Progress tracking
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(num_simulations), desc="Simulating")
            except ImportError:
                iterator = range(num_simulations)
        else:
            iterator = range(num_simulations)

        # Run simulations
        for sim_num in iterator:
            # Generate random return path
            # Use different seed for each simulation if base seed provided
            sim_seed = seed + sim_num if seed is not None else None

            annual_returns = self.fin_data.sample_annual_returns(
                self.config.tickers,
                num_years=self.config.num_years,
                method=method,
                seed=sim_seed
            )

            # Run single path simulation
            path = self.run_single_path(annual_returns)

            # Store results
            all_paths.append(path)
            final_values.append(path.final_value)

            if path.success:
                success_count += 1

            # Store portfolio values for this path (pad with 0s if depleted)
            values = path.portfolio_values.copy()
            if len(values) < self.config.num_years:
                # Pad with zeros for depleted paths
                values.extend([0.0] * (self.config.num_years - len(values)))
            portfolio_values_matrix.append(values)

        # Calculate statistics
        final_values_array = np.array(final_values)
        portfolio_values_array = np.array(portfolio_values_matrix)

        success_rate = success_count / num_simulations

        percentiles = {
            '5th': np.percentile(final_values_array, 5),
            '25th': np.percentile(final_values_array, 25),
            '50th': np.percentile(final_values_array, 50),
            '75th': np.percentile(final_values_array, 75),
            '95th': np.percentile(final_values_array, 95)
        }

        results = MonteCarloResults(
            num_simulations=num_simulations,
            success_rate=success_rate,
            paths=all_paths,
            final_values=final_values_array,
            percentiles=percentiles,
            median_final_value=percentiles['50th'],
            mean_final_value=np.mean(final_values_array),
            std_final_value=np.std(final_values_array),
            portfolio_values_matrix=portfolio_values_array
        )

        logging.info(f"Monte Carlo complete: {success_rate:.1%} success rate")

        return results
```

**Test**:
```python
# Add to test_retirement_engine.py

def test_monte_carlo_reproducibility(basic_config):
    """Test that same seed gives identical results"""
    engine = RetirementEngine(basic_config)

    results1 = engine.run_monte_carlo(num_simulations=100, seed=42, show_progress=False)
    results2 = engine.run_monte_carlo(num_simulations=100, seed=42, show_progress=False)

    assert results1.success_rate == results2.success_rate
    assert np.allclose(results1.final_values, results2.final_values)
    assert results1.percentiles == results2.percentiles

def test_monte_carlo_conservative_scenario(basic_config):
    """Test conservative scenario - should have high success rate"""
    # Conservative: large portfolio, small withdrawal
    basic_config.initial_portfolio = 1_500_000
    basic_config.annual_withdrawal = 40_000  # 2.67% withdrawal rate

    engine = RetirementEngine(basic_config)
    results = engine.run_monte_carlo(num_simulations=1000, seed=42, show_progress=False)

    # Should have very high success rate
    assert results.success_rate > 0.90
    assert results.median_final_value > 1_000_000

def test_monte_carlo_aggressive_scenario(basic_config):
    """Test aggressive scenario - should have lower success rate"""
    # Aggressive: high withdrawal rate
    basic_config.initial_portfolio = 1_000_000
    basic_config.annual_withdrawal = 60_000  # 6% withdrawal rate

    engine = RetirementEngine(basic_config)
    results = engine.run_monte_carlo(num_simulations=1000, seed=42, show_progress=False)

    # Should have lower success rate
    assert results.success_rate < 0.80

def test_monte_carlo_percentiles_ordering(basic_config):
    """Test that percentiles are correctly ordered"""
    engine = RetirementEngine(basic_config)
    results = engine.run_monte_carlo(num_simulations=1000, seed=42, show_progress=False)

    # Percentiles should be in ascending order
    assert results.percentiles['5th'] <= results.percentiles['25th']
    assert results.percentiles['25th'] <= results.percentiles['50th']
    assert results.percentiles['50th'] <= results.percentiles['75th']
    assert results.percentiles['75th'] <= results.percentiles['95th']

def test_monte_carlo_matrix_shape(basic_config):
    """Test portfolio values matrix has correct shape"""
    engine = RetirementEngine(basic_config)
    results = engine.run_monte_carlo(num_simulations=100, seed=42, show_progress=False)

    # Matrix should be (num_sims x num_years)
    assert results.portfolio_values_matrix.shape == (100, basic_config.num_years)
```

**Run**:
```bash
uv run pytest test_retirement_engine.py::test_monte_carlo -v
```

---

### Stage 5: Basic Visualization (Day 4)

#### Task 5.1: Create `retirement_visualization.py`

**Goal**: Visualize Monte Carlo results with fan charts and summary plots

**Implementation**:

```python
# retirement_visualization.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from typing import Optional
import os

from retirement_config import RetirementConfig
from retirement_engine import MonteCarloResults


class RetirementVisualizer:
    """Visualization tools for retirement simulations"""

    def __init__(self, config: RetirementConfig):
        self.config = config

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_monte_carlo_paths(self,
                              results: MonteCarloResults,
                              show_individual_paths: bool = False,
                              max_paths_to_show: int = 100,
                              save_path: Optional[str] = None):
        """
        Plot Monte Carlo fan chart with percentile bands.

        Parameters:
        -----------
        results : MonteCarloResults
            Monte Carlo simulation results
        show_individual_paths : bool
            Whether to show individual paths (only for small num_sims)
        max_paths_to_show : int
            Maximum number of individual paths to show
        save_path : Optional[str]
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        years = np.arange(self.config.num_years)
        portfolio_matrix = results.portfolio_values_matrix

        # Calculate percentiles at each time step
        percentiles = {}
        for pct, label in [(5, '5th'), (25, '25th'), (50, '50th'), (75, '75th'), (95, '95th')]:
            percentiles[label] = np.percentile(portfolio_matrix, pct, axis=0)

        # Plot percentile bands
        ax.fill_between(years, percentiles['5th'], percentiles['95th'],
                        alpha=0.15, color='blue', label='5th-95th percentile')
        ax.fill_between(years, percentiles['25th'], percentiles['75th'],
                        alpha=0.25, color='blue', label='25th-75th percentile')

        # Plot median
        ax.plot(years, percentiles['50th'], 'b-', linewidth=2.5, label='Median (50th)')

        # Show individual paths if requested
        if show_individual_paths and results.num_simulations <= max_paths_to_show:
            for i in range(results.num_simulations):
                path = results.paths[i]
                color = 'green' if path.success else 'red'
                ax.plot(years[:len(path.portfolio_values)], path.portfolio_values,
                       color=color, alpha=0.1, linewidth=0.5)

        # Mark depletion line
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Depletion')

        # Mark starting value
        ax.axhline(y=self.config.initial_portfolio, color='gray',
                  linestyle=':', alpha=0.5, label='Initial Portfolio')

        # Formatting
        ax.set_xlabel('Years', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.set_title(f'Monte Carlo Retirement Simulation\n'
                    f'{results.num_simulations:,} simulations | '
                    f'Success Rate: {results.success_rate:.1%}',
                    fontsize=14, fontweight='bold')

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))

        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig, ax

    def plot_summary_statistics(self,
                               results: MonteCarloResults,
                               save_path: Optional[str] = None):
        """
        Plot summary statistics dashboard.

        Shows:
        - Success rate (big number)
        - Final value distribution
        - Percentile table
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Success rate (big number)
        ax1 = axes[0, 0]
        ax1.text(0.5, 0.5, f'{results.success_rate:.1%}',
                ha='center', va='center', fontsize=72, fontweight='bold',
                color='green' if results.success_rate >= 0.85 else 'orange')
        ax1.text(0.5, 0.25, 'Success Rate',
                ha='center', va='center', fontsize=18)
        ax1.text(0.5, 0.15, f'({results.success_rate * results.num_simulations:.0f}/{results.num_simulations} paths)',
                ha='center', va='center', fontsize=12, style='italic')
        ax1.axis('off')

        # 2. Final value distribution
        ax2 = axes[0, 1]
        successful_finals = [p.final_value for p in results.paths if p.success]
        if successful_finals:
            ax2.hist(successful_finals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
            ax2.axvline(results.median_final_value, color='red',
                       linestyle='--', linewidth=2, label=f'Median: ${results.median_final_value/1e6:.2f}M')
            ax2.set_xlabel('Final Portfolio Value ($)', fontsize=11)
            ax2.set_ylabel('Frequency', fontsize=11)
            ax2.set_title('Distribution of Final Values\n(Successful Paths Only)', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

        # 3. Percentile table
        ax3 = axes[1, 0]
        ax3.axis('off')

        table_data = [
            ['Percentile', 'Final Value'],
            ['95th', f'${results.percentiles["95th"]/1e6:.2f}M'],
            ['75th', f'${results.percentiles["75th"]/1e6:.2f}M'],
            ['50th (Median)', f'${results.percentiles["50th"]/1e6:.2f}M'],
            ['25th', f'${results.percentiles["25th"]/1e6:.2f}M'],
            ['5th', f'${results.percentiles["5th"]/1e6:.2f}M'],
        ]

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

        # 4. Key statistics
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Calculate additional stats
        depletion_years = [p.depletion_year for p in results.paths if not p.success]
        avg_depletion = np.mean(depletion_years) if depletion_years else None

        stats_text = f"""
SIMULATION PARAMETERS
{'='*40}
Initial Portfolio:    ${self.config.initial_portfolio:,.0f}
Annual Withdrawal:    ${self.config.annual_withdrawal:,.0f}
Withdrawal Rate:      {self.config.annual_withdrawal/self.config.initial_portfolio:.2%}
Inflation Rate:       {self.config.inflation_rate:.1%}
Time Horizon:         {self.config.num_years} years
Portfolio:            {', '.join([f'{t}:{w:.0%}' for t, w in self.config.current_portfolio.items()])}

RESULTS
{'='*40}
Success Rate:         {results.success_rate:.1%}
Median Final Value:   ${results.median_final_value:,.0f}
Mean Final Value:     ${results.mean_final_value:,.0f}
Std Dev:              ${results.std_final_value:,.0f}
"""

        if avg_depletion:
            stats_text += f"Avg Depletion Year:   {avg_depletion:.1f}\n"

        ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig, axes

    def plot_all(self, results: MonteCarloResults, output_dir: str = '../plots/retirement/'):
        """Generate all plots and save to directory"""
        os.makedirs(output_dir, exist_ok=True)

        # Fan chart
        self.plot_monte_carlo_paths(
            results,
            save_path=os.path.join(output_dir, 'monte_carlo_fan_chart.png')
        )

        # Summary statistics
        self.plot_summary_statistics(
            results,
            save_path=os.path.join(output_dir, 'summary_statistics.png')
        )

        plt.show()
```

**Test**: Visual inspection
```python
# test_retirement_visualization.py
def test_visualization_creates_plots():
    """Test that plots are created without errors"""
    config = RetirementConfig(
        initial_portfolio=1_000_000,
        start_date='2024-01-01',
        end_date='2054-01-01',
        annual_withdrawal=40_000,
        current_portfolio={'SPY': 0.6, 'AGG': 0.4}
    )

    engine = RetirementEngine(config)
    results = engine.run_monte_carlo(num_simulations=1000, seed=42, show_progress=False)

    visualizer = RetirementVisualizer(config)

    # Create test output directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test fan chart
        visualizer.plot_monte_carlo_paths(
            results,
            save_path=f'{tmpdir}/fan_chart.png'
        )
        assert os.path.exists(f'{tmpdir}/fan_chart.png')

        # Test summary statistics
        visualizer.plot_summary_statistics(
            results,
            save_path=f'{tmpdir}/summary.png'
        )
        assert os.path.exists(f'{tmpdir}/summary.png')
```

**Manual inspection**:
```bash
uv run python -c "
from retirement_config import RetirementConfig
from retirement_engine import RetirementEngine
from retirement_visualization import RetirementVisualizer

config = RetirementConfig(
    initial_portfolio=1_000_000,
    start_date='2024-01-01',
    end_date='2054-01-01',
    annual_withdrawal=40_000,
    current_portfolio={'SPY': 0.6, 'AGG': 0.4}
)

engine = RetirementEngine(config)
results = engine.run_monte_carlo(num_simulations=1000)

visualizer = RetirementVisualizer(config)
visualizer.plot_all(results)
"
```

---

### Stage 6: CLI and Output (Day 5)

#### Task 6.1: Create `retirement_main.py`

**Goal**: Command-line interface for running simulations

**Implementation**:

```python
#!/usr/bin/env python3
"""
Retirement Dashboard - Main Entry Point

Run Monte Carlo retirement simulations from command line.
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from retirement_config import RetirementConfig
from retirement_engine import RetirementEngine
from retirement_visualization import RetirementVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def parse_portfolio(portfolio_str: str) -> dict:
    """
    Parse portfolio string like 'SPY:0.6,AGG:0.4' into dict.

    Returns:
    --------
    dict : {ticker: weight}
    """
    portfolio = {}
    for pair in portfolio_str.split(','):
        ticker, weight = pair.split(':')
        portfolio[ticker.strip()] = float(weight)

    # Validate weights sum to 1
    total = sum(portfolio.values())
    if not (0.99 <= total <= 1.01):
        raise ValueError(f"Portfolio weights must sum to 1.0, got {total}")

    return portfolio


def main():
    parser = argparse.ArgumentParser(
        description='Monte Carlo Retirement Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic 30-year retirement simulation
  %(prog)s --initial-portfolio 1000000 --annual-withdrawal 40000 --years 30

  # Custom portfolio allocation
  %(prog)s --initial-portfolio 1500000 --annual-withdrawal 50000 --years 25 \\
           --portfolio "SPY:0.7,AGG:0.3"

  # High-fidelity simulation
  %(prog)s --initial-portfolio 2000000 --annual-withdrawal 80000 --years 30 \\
           --simulations 10000 --output-dir results/high_fidelity/
        """
    )

    # Required arguments
    parser.add_argument('--initial-portfolio', type=float, required=True,
                       help='Initial portfolio value ($)')
    parser.add_argument('--annual-withdrawal', type=float, required=True,
                       help='Annual withdrawal amount ($)')
    parser.add_argument('--years', type=int, required=True,
                       help='Number of years to simulate')

    # Optional arguments
    parser.add_argument('--portfolio', type=str, default='SPY:0.6,AGG:0.4',
                       help='Portfolio allocation (default: SPY:0.6,AGG:0.4)')
    parser.add_argument('--simulations', type=int, default=5000,
                       help='Number of Monte Carlo simulations (default: 5000)')
    parser.add_argument('--inflation-rate', type=float, default=0.03,
                       help='Annual inflation rate (default: 0.03)')
    parser.add_argument('--method', choices=['bootstrap', 'parametric'], default='bootstrap',
                       help='Return sampling method (default: bootstrap)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default='../results/retirement/',
                       help='Output directory for results')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')

    args = parser.parse_args()

    # Parse portfolio
    try:
        portfolio = parse_portfolio(args.portfolio)
    except Exception as e:
        print(f"Error parsing portfolio: {e}")
        sys.exit(1)

    # Create configuration
    from datetime import datetime, timedelta
    start_date = datetime.now().strftime('%Y-%m-%d')
    end_date = (datetime.now() + timedelta(days=args.years * 365)).strftime('%Y-%m-%d')

    config = RetirementConfig(
        initial_portfolio=args.initial_portfolio,
        start_date=start_date,
        end_date=end_date,
        annual_withdrawal=args.annual_withdrawal,
        inflation_rate=args.inflation_rate,
        current_portfolio=portfolio
    )

    # Print configuration
    print("\n" + "="*60)
    print("RETIREMENT SIMULATION CONFIGURATION")
    print("="*60)
    print(f"Initial Portfolio:   ${config.initial_portfolio:,.0f}")
    print(f"Annual Withdrawal:   ${config.annual_withdrawal:,.0f}")
    print(f"Withdrawal Rate:     {config.annual_withdrawal/config.initial_portfolio:.2%}")
    print(f"Time Horizon:        {args.years} years")
    print(f"Inflation Rate:      {config.inflation_rate:.1%}")
    print(f"Portfolio:")
    for ticker, weight in portfolio.items():
        print(f"  {ticker}: {weight:.1%}")
    print(f"Simulations:         {args.simulations:,}")
    print(f"Method:              {args.method}")
    print("="*60 + "\n")

    # Run simulation
    print("Running Monte Carlo simulation...")
    engine = RetirementEngine(config)
    results = engine.run_monte_carlo(
        num_simulations=args.simulations,
        method=args.method,
        seed=args.seed
    )

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Success Rate:        {results.success_rate:.1%}")
    print(f"Median Final Value:  ${results.median_final_value:,.0f}")
    print(f"Mean Final Value:    ${results.mean_final_value:,.0f}")
    print(f"Std Dev:             ${results.std_final_value:,.0f}")
    print("\nPercentiles:")
    for pct, value in results.percentiles.items():
        print(f"  {pct:>4s}: ${value:,.0f}")
    print("="*60 + "\n")

    # Export results
    csv_dir = os.path.join(args.output_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    results.export_to_csv(csv_dir)
    print(f"Results exported to: {csv_dir}")

    # Generate plots
    if not args.no_plots:
        print("\nGenerating visualizations...")
        plots_dir = os.path.join(args.output_dir, 'plots')
        visualizer = RetirementVisualizer(config)
        visualizer.plot_all(results, output_dir=plots_dir)
        print(f"Plots saved to: {plots_dir}")

    print("\nSimulation complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

#### Task 6.2: Add CSV export methods

Add to `retirement_engine.py`:

```python
# Add to MonteCarloResults class

def export_to_csv(self, output_dir: str):
    """Export results to CSV files"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Summary statistics
    summary_df = pd.DataFrame({
        'Metric': ['Num Simulations', 'Success Rate', 'Median Final Value',
                   'Mean Final Value', 'Std Dev Final Value',
                   '5th Percentile', '25th Percentile', '50th Percentile',
                   '75th Percentile', '95th Percentile'],
        'Value': [
            self.num_simulations,
            self.success_rate,
            self.median_final_value,
            self.mean_final_value,
            self.std_final_value,
            self.percentiles['5th'],
            self.percentiles['25th'],
            self.percentiles['50th'],
            self.percentiles['75th'],
            self.percentiles['95th']
        ]
    })
    summary_df.to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)

    # 2. All paths final values
    final_values_df = pd.DataFrame({
        'Simulation': range(self.num_simulations),
        'Final_Value': self.final_values,
        'Success': [p.success for p in self.paths]
    })
    final_values_df.to_csv(os.path.join(output_dir, 'final_values.csv'), index=False)

    # 3. Percentile paths over time
    percentile_data = {}
    for pct, label in [(5, '5th'), (25, '25th'), (50, '50th'), (75, '75th'), (95, '95th')]:
        percentile_data[f'{label}_percentile'] = np.percentile(self.portfolio_values_matrix, pct, axis=0)

    percentile_df = pd.DataFrame(percentile_data)
    percentile_df['Year'] = range(len(percentile_df))
    percentile_df.to_csv(os.path.join(output_dir, 'percentile_paths.csv'), index=False)

    print(f"Exported 3 CSV files to {output_dir}")
```

**Test**: End-to-end CLI test
```bash
# Test basic usage
uv run python retirement_main.py \
  --initial-portfolio 1000000 \
  --annual-withdrawal 40000 \
  --years 30 \
  --simulations 1000 \
  --output-dir ../results/test_run \
  --seed 42

# Check outputs
ls -la ../results/test_run/csv/
ls -la ../results/test_run/plots/

# Verify files exist
test -f ../results/test_run/csv/summary_statistics.csv && echo "✓ Summary CSV exists"
test -f ../results/test_run/csv/final_values.csv && echo "✓ Final values CSV exists"
test -f ../results/test_run/plots/monte_carlo_fan_chart.png && echo "✓ Fan chart exists"
test -f ../results/test_run/plots/summary_statistics.png && echo "✓ Summary plot exists"
```

---

### Stage 7: Integration Testing (Day 6)

#### Task 7.1: End-to-end realistic scenarios

```python
# test_integration.py
import pytest
import numpy as np
from retirement_config import RetirementConfig
from retirement_engine import RetirementEngine

def test_trinity_study_validation():
    """
    Validate against Trinity Study results.

    Trinity Study: 4% withdrawal rule with 75/25 stock/bond allocation
    showed ~95% success rate over 30 years based on historical data.
    """
    config = RetirementConfig(
        initial_portfolio=1_000_000,
        start_date='2024-01-01',
        end_date='2054-01-01',  # 30 years
        annual_withdrawal=40_000,  # 4% of initial
        current_portfolio={'SPY': 0.75, 'AGG': 0.25}
    )

    engine = RetirementEngine(config)
    results = engine.run_monte_carlo(num_simulations=5000, seed=42, show_progress=False)

    # Trinity Study found ~95% success for this scenario
    # Our bootstrap method should show similar results
    assert 0.85 <= results.success_rate <= 0.98, \
        f"Expected ~95% success rate, got {results.success_rate:.1%}"

    print(f"Trinity Study validation: {results.success_rate:.1%} success rate")

def test_conservative_retirement():
    """Test conservative scenario: 3% withdrawal, 60/40 portfolio"""
    config = RetirementConfig(
        initial_portfolio=1_000_000,
        annual_withdrawal=30_000,  # 3% withdrawal
        start_date='2024-01-01',
        end_date='2054-01-01',
        current_portfolio={'SPY': 0.6, 'AGG': 0.4}
    )

    engine = RetirementEngine(config)
    results = engine.run_monte_carlo(num_simulations=5000, seed=42, show_progress=False)

    # Conservative scenario should have very high success
    assert results.success_rate >= 0.95
    assert results.median_final_value > 1_500_000  # Should grow significantly

    print(f"Conservative: {results.success_rate:.1%} success, "
          f"median final: ${results.median_final_value/1e6:.2f}M")

def test_aggressive_retirement():
    """Test aggressive scenario: 5% withdrawal"""
    config = RetirementConfig(
        initial_portfolio=1_000_000,
        annual_withdrawal=50_000,  # 5% withdrawal
        start_date='2024-01-01',
        end_date='2054-01-01',
        current_portfolio={'SPY': 0.8, 'AGG': 0.2}  # Aggressive allocation
    )

    engine = RetirementEngine(config)
    results = engine.run_monte_carlo(num_simulations=5000, seed=42, show_progress=False)

    # Aggressive scenario should have lower success
    assert 0.60 <= results.success_rate <= 0.85

    print(f"Aggressive: {results.success_rate:.1%} success")

def test_different_time_horizons():
    """Test that longer horizons are riskier"""
    base_config_20yr = RetirementConfig(
        initial_portfolio=1_000_000,
        annual_withdrawal=40_000,
        start_date='2024-01-01',
        end_date='2044-01-01',  # 20 years
        current_portfolio={'SPY': 0.6, 'AGG': 0.4}
    )

    base_config_40yr = RetirementConfig(
        initial_portfolio=1_000_000,
        annual_withdrawal=40_000,
        start_date='2024-01-01',
        end_date='2064-01-01',  # 40 years
        current_portfolio={'SPY': 0.6, 'AGG': 0.4}
    )

    engine_20yr = RetirementEngine(base_config_20yr)
    engine_40yr = RetirementEngine(base_config_40yr)

    results_20yr = engine_20yr.run_monte_carlo(num_simulations=2000, seed=42, show_progress=False)
    results_40yr = engine_40yr.run_monte_carlo(num_simulations=2000, seed=42, show_progress=False)

    # 20-year horizon should have higher success rate
    assert results_20yr.success_rate > results_40yr.success_rate

    print(f"20yr: {results_20yr.success_rate:.1%}, 40yr: {results_40yr.success_rate:.1%}")

def test_portfolio_allocation_impact():
    """Test that different allocations affect success rate"""
    # Conservative (30/70)
    config_conservative = RetirementConfig(
        initial_portfolio=1_000_000,
        annual_withdrawal=40_000,
        start_date='2024-01-01',
        end_date='2054-01-01',
        current_portfolio={'SPY': 0.3, 'AGG': 0.7}
    )

    # Aggressive (90/10)
    config_aggressive = RetirementConfig(
        initial_portfolio=1_000_000,
        annual_withdrawal=40_000,
        start_date='2024-01-01',
        end_date='2054-01-01',
        current_portfolio={'SPY': 0.9, 'AGG': 0.1}
    )

    engine_cons = RetirementEngine(config_conservative)
    engine_agg = RetirementEngine(config_aggressive)

    results_cons = engine_cons.run_monte_carlo(num_simulations=2000, seed=42, show_progress=False)
    results_agg = engine_agg.run_monte_carlo(num_simulations=2000, seed=42, show_progress=False)

    print(f"Conservative 30/70: {results_cons.success_rate:.1%}, "
          f"median: ${results_cons.median_final_value/1e6:.2f}M")
    print(f"Aggressive 90/10: {results_agg.success_rate:.1%}, "
          f"median: ${results_agg.median_final_value/1e6:.2f}M")

    # Aggressive should have higher median if successful, but may have lower success rate
    # (This depends on historical data characteristics)
```

**Run**:
```bash
uv run pytest test_integration.py -v -s
```

#### Task 7.2: Performance benchmarking

```python
# test_performance.py
import pytest
import time
from retirement_config import RetirementConfig
from retirement_engine import RetirementEngine

@pytest.mark.benchmark
def test_performance_1000_simulations():
    """Benchmark 1000 simulations - should be fast"""
    config = RetirementConfig(
        initial_portfolio=1_000_000,
        start_date='2024-01-01',
        end_date='2054-01-01',
        annual_withdrawal=40_000,
        current_portfolio={'SPY': 0.6, 'AGG': 0.4}
    )

    engine = RetirementEngine(config)

    start = time.time()
    results = engine.run_monte_carlo(num_simulations=1000, show_progress=False)
    elapsed = time.time() - start

    print(f"\n1000 simulations: {elapsed:.2f} seconds ({elapsed/1000*1000:.1f} ms/sim)")
    assert elapsed < 10, f"Too slow: {elapsed:.1f}s for 1000 sims (target: <10s)"

@pytest.mark.benchmark
def test_performance_5000_simulations():
    """Benchmark 5000 simulations - typical use case"""
    config = RetirementConfig(
        initial_portfolio=1_000_000,
        start_date='2024-01-01',
        end_date='2054-01-01',
        annual_withdrawal=40_000,
        current_portfolio={'SPY': 0.6, 'AGG': 0.4}
    )

    engine = RetirementEngine(config)

    start = time.time()
    results = engine.run_monte_carlo(num_simulations=5000, show_progress=False)
    elapsed = time.time() - start

    print(f"\n5000 simulations: {elapsed:.2f} seconds ({elapsed/5000*1000:.1f} ms/sim)")
    assert elapsed < 30, f"Too slow: {elapsed:.1f}s for 5000 sims (target: <30s)"

@pytest.mark.benchmark
def test_performance_10000_simulations():
    """Benchmark 10000 simulations - high fidelity"""
    config = RetirementConfig(
        initial_portfolio=1_000_000,
        start_date='2024-01-01',
        end_date='2054-01-01',
        annual_withdrawal=40_000,
        current_portfolio={'SPY': 0.6, 'AGG': 0.4}
    )

    engine = RetirementEngine(config)

    start = time.time()
    results = engine.run_monte_carlo(num_simulations=10000, show_progress=False)
    elapsed = time.time() - start

    print(f"\n10000 simulations: {elapsed:.2f} seconds ({elapsed/10000*1000:.1f} ms/sim)")
    assert elapsed < 60, f"Too slow: {elapsed:.1f}s for 10000 sims (target: <60s)"
```

**Run**:
```bash
uv run pytest test_performance.py -v -s -m benchmark
```

---

### Stage 8: Documentation (Day 7)

#### Task 8.1: Update CLAUDE.md

Add retirement dashboard section to existing CLAUDE.md.

#### Task 8.2: Create README

```markdown
# Retirement Dashboard - Monte Carlo Simulation

## Overview

Monte Carlo-based retirement planning tool that simulates thousands of possible portfolio outcomes to estimate the probability of retirement success.

## Features

- **Monte Carlo Simulation**: Run thousands of simulations with randomly sampled historical returns
- **Flexible Configuration**: Customize portfolio allocation, withdrawal rates, time horizons
- **Comprehensive Visualization**: Fan charts, percentile bands, summary statistics
- **CSV Exports**: Detailed data for further analysis
- **Command-line Interface**: Easy to use from terminal

## Quick Start

### Installation

```bash
cd src/
uv sync  # Install dependencies
```

### Basic Usage

```bash
uv run python retirement_main.py \
  --initial-portfolio 1000000 \
  --annual-withdrawal 40000 \
  --years 30 \
  --portfolio "SPY:0.6,AGG:0.4"
```

### Programmatic Usage

```python
from retirement_config import RetirementConfig
from retirement_engine import RetirementEngine
from retirement_visualization import RetirementVisualizer

# Configure simulation
config = RetirementConfig(
    initial_portfolio=1_000_000,
    start_date='2024-01-01',
    end_date='2054-01-01',
    annual_withdrawal=40_000,
    current_portfolio={'SPY': 0.6, 'AGG': 0.4}
)

# Run simulation
engine = RetirementEngine(config)
results = engine.run_monte_carlo(num_simulations=5000)

# Visualize
visualizer = RetirementVisualizer(config)
visualizer.plot_all(results)
```

## Interpreting Results

### Success Rate
- **>90%**: Very safe - portfolio likely to last
- **80-90%**: Relatively safe - good chance of success
- **70-80%**: Moderate risk - consider reducing withdrawals
- **<70%**: High risk - need to adjust plan

### Percentiles
- **5th percentile**: Worst-case scenario (bottom 5% of outcomes)
- **50th percentile (Median)**: Typical outcome
- **95th percentile**: Best-case scenario (top 5% of outcomes)

### Withdrawal Rates (Rule of Thumb)
- **3%**: Very conservative - high success rate
- **4%**: Traditional "safe withdrawal rate" - ~90-95% success
- **5%**: Aggressive - 70-80% success rate
- **6%+**: Very aggressive - <70% success rate

## Examples

### Conservative Retirement
```bash
uv run python retirement_main.py \
  --initial-portfolio 1500000 \
  --annual-withdrawal 45000 \
  --years 30 \
  --portfolio "SPY:0.5,AGG:0.5"
```

### Aggressive Retirement
```bash
uv run python retirement_main.py \
  --initial-portfolio 1000000 \
  --annual-withdrawal 50000 \
  --years 30 \
  --portfolio "SPY:0.8,AGG:0.2"
```

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run integration tests
uv run pytest test_integration.py -v -s

# Run performance benchmarks
uv run pytest test_performance.py -v -s -m benchmark
```

## Roadmap

### Phase 1: Core MC Engine (Complete)
- ✓ Basic Monte Carlo simulation
- ✓ Bootstrap return sampling
- ✓ Visualization and reporting
- ✓ CLI interface

### Phase 2: Advanced Features (Planned)
- [ ] Multiple withdrawal strategies (Guyton-Klinger, VPW, etc.)
- [ ] Accumulation phase with portfolio optimization
- [ ] Financial events (Social Security, pensions, mortgages)
- [ ] Tax-aware withdrawal strategies

### Phase 3: Regime-Switching (Future)
- [ ] Market regime detection
- [ ] Regime-based return sampling
- [ ] Dynamic asset allocation

## References

- [Trinity Study](https://en.wikipedia.org/wiki/Trinity_study) - Original safe withdrawal rate research
- [Guyton-Klinger Rules](https://www.kitces.com/blog/the-problem-with-firings-4-safe-withdrawal-rate-what-it-actually-is-and-how-to-fix-it/)
```

---

## Phase 2: Advanced Features (Deferred)

The following features are deferred until Phase 1 is complete and validated:

### Withdrawal Strategies Module
- Guyton-Klinger dynamic adjustments
- Floor-Ceiling constraints
- VPW (Variable Percentage Withdrawal) based on age
- Adaptive withdrawals responding to market conditions

### Accumulation Phase
- Portfolio optimization during pre-retirement
- Monthly/annual contributions
- Seamless transition to decumulation

### Financial Events
- Mortgage payments and payoff
- Social Security commencement
- Pension income
- One-time expenses (healthcare, gifts, etc.)

### Regime-Switching Monte Carlo
- Integration of existing RSMC.py
- Market regime detection (bull/bear/normal)
- Regime-dependent return parameters
- Transition probability matrices

---

## Testing Strategy Summary

### Unit Tests
- `test_retirement_config.py` - Configuration validation
- `test_fin_data_sampling.py` - Return sampling methods
- `test_retirement_engine.py` - Single-path and MC simulation
- `test_retirement_visualization.py` - Plot generation

### Integration Tests
- `test_integration.py` - End-to-end scenarios
- Trinity Study validation
- Different time horizons
- Portfolio allocation impacts

### Performance Tests
- `test_performance.py` - Benchmark simulation speed
- Target: 5000 simulations < 30 seconds

### Manual Tests
- Visual inspection of plots
- CLI usage with various parameters
- CSV export validation

---

## Success Criteria

### Phase 1 Complete When:
- [ ] All unit tests passing (>90% coverage)
- [ ] Integration tests validate against known benchmarks
- [ ] Performance tests meet targets
- [ ] CLI produces complete output (plots + CSV)
- [ ] Documentation complete and clear
- [ ] Example configs work out of the box

### Key Metrics:
- **Test Coverage**: >90%
- **Performance**: 5000 sims < 30s
- **Success Rate Accuracy**: Within ±5% of Trinity Study
- **Code Quality**: Clean, documented, modular

---

## Implementation Notes

### Design Decisions

1. **Bootstrap vs Parametric Sampling**
   - Default to bootstrap (empirical distribution)
   - Parametric available as option (assumes normality)
   - Bootstrap preserves fat tails and correlations

2. **Annual vs Monthly Granularity**
   - Phase 1: Annual withdrawals for simplicity
   - Phase 2: Add monthly option

3. **Rebalancing**
   - Phase 1: Static allocation (buy-and-hold)
   - Phase 2: Add periodic rebalancing

4. **Data Source**
   - Use existing FinData infrastructure
   - SPY for stocks, AGG for bonds as defaults
   - Extensible to other tickers

### Known Limitations (Phase 1)

- Fixed withdrawal amount (no strategies yet)
- No tax considerations
- No financial events
- Static portfolio allocation
- Single-regime returns (no regime-switching)

These will be addressed in Phase 2+.

---

## File Structure

```
src/
├── retirement_config.py         # Configuration dataclass
├── retirement_engine.py         # MC simulation engine
├── retirement_visualization.py  # Plotting and dashboards
├── retirement_main.py           # CLI entry point
├── fin_data.py                  # Extended with sampling methods
└── tests/
    ├── test_retirement_config.py
    ├── test_retirement_engine.py
    ├── test_fin_data_sampling.py
    ├── test_retirement_visualization.py
    ├── test_integration.py
    └── test_performance.py
```

---

## Timeline

| Day | Stage | Deliverables |
|-----|-------|-------------|
| 1 | Config + Sampling | retirement_config.py, extended fin_data.py |
| 2 | Single-Path Sim | retirement_engine.py (single path) |
| 3 | Monte Carlo | run_monte_carlo() implementation |
| 4 | Visualization | retirement_visualization.py |
| 5 | CLI + Export | retirement_main.py, CSV exports |
| 6 | Testing | Integration and performance tests |
| 7 | Documentation | CLAUDE.md, README, examples |

**Total: 7 days for Phase 1**

Then iterate for Phase 2 (withdrawal strategies, accumulation, financial events) and Phase 3 (regime-switching).
