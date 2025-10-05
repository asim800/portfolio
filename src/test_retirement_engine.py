#!/usr/bin/env python3
"""
Unit tests for RetirementEngine - single path simulation.
"""

import pytest
import numpy as np
import pandas as pd
from retirement_config import RetirementConfig
from retirement_engine import RetirementEngine, SimulationPath


@pytest.fixture
def basic_config():
    """Basic retirement configuration for testing"""
    return RetirementConfig(
        initial_portfolio=1_000_000,
        start_date='2024-01-01',
        end_date='2034-01-01',  # 10 years
        annual_withdrawal=40_000,
        current_portfolio={'SPY': 0.6, 'AGG': 0.4}
    )


@pytest.fixture
def mock_returns_good():
    """Mock good returns (8% per year for both assets)"""
    return pd.DataFrame({
        'SPY': [0.08] * 10,
        'AGG': [0.08] * 10
    })


@pytest.fixture
def mock_returns_poor():
    """Mock poor returns (-5% per year)"""
    return pd.DataFrame({
        'SPY': [-0.05] * 10,
        'AGG': [-0.05] * 10
    })


def test_engine_initialization(basic_config):
    """Test that engine initializes correctly"""
    engine = RetirementEngine(basic_config)

    assert engine.config == basic_config
    assert len(engine.weights) == 2
    assert np.allclose(engine.weights, [0.6, 0.4])
    assert engine.fin_data is not None


def test_single_path_no_depletion(basic_config, mock_returns_good):
    """Test path with good returns - should not deplete"""
    engine = RetirementEngine(basic_config)
    result = engine.run_single_path(mock_returns_good)

    assert result.success == True
    assert result.depletion_year is None
    assert len(result.portfolio_values) == 10
    assert len(result.withdrawals) == 10
    assert len(result.annual_returns) == 10
    assert result.final_value > 1_000_000  # Should grow despite withdrawals


def test_single_path_with_depletion():
    """Test path with poor returns and high withdrawals - should deplete"""
    config = RetirementConfig(
        initial_portfolio=500_000,
        start_date='2024-01-01',
        end_date='2034-01-01',
        annual_withdrawal=100_000,  # Withdrawing 20%!
        current_portfolio={'SPY': 0.6, 'AGG': 0.4}
    )

    # Poor returns
    poor_returns = pd.DataFrame({
        'SPY': [-0.05] * 10,
        'AGG': [-0.05] * 10
    })

    engine = RetirementEngine(config)
    result = engine.run_single_path(poor_returns)

    assert result.success == False
    assert result.depletion_year is not None
    assert result.depletion_year < 10  # Depleted before end
    assert result.final_value == 0.0


def test_withdrawal_inflation_adjustment(basic_config, mock_returns_good):
    """Test that withdrawals adjust for inflation"""
    engine = RetirementEngine(basic_config)
    result = engine.run_single_path(mock_returns_good)

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
    mixed_returns = pd.DataFrame({
        'SPY': [0.10] * 10,  # Stocks: 10%
        'AGG': [0.02] * 10   # Bonds: 2%
    })

    result = engine.run_single_path(mixed_returns)

    # Expected portfolio return: 0.6*0.10 + 0.4*0.02 = 0.068 (6.8%)
    expected_return = 0.6 * 0.10 + 0.4 * 0.02
    assert result.annual_returns[0] == pytest.approx(expected_return)


def test_year_tracking(basic_config, mock_returns_good):
    """Test that years are tracked correctly"""
    engine = RetirementEngine(basic_config)
    result = engine.run_single_path(mock_returns_good)

    # Years should be 0, 1, 2, ..., 9
    assert result.years == list(range(10))

    # Dates should advance by ~365 days each year
    for i in range(1, len(result.dates)):
        days_diff = (result.dates[i] - result.dates[i-1]).days
        assert 365 <= days_diff <= 366  # Account for leap years


def test_portfolio_values_recorded(basic_config, mock_returns_good):
    """Test that portfolio values are recorded at start of each year"""
    engine = RetirementEngine(basic_config)
    result = engine.run_single_path(mock_returns_good)

    # First value should be initial portfolio
    assert result.portfolio_values[0] == 1_000_000

    # Values should change each year
    for i in range(1, len(result.portfolio_values)):
        assert result.portfolio_values[i] != result.portfolio_values[i-1]


def test_zero_withdrawal(mock_returns_good):
    """Test simulation with zero withdrawals (accumulation only)"""
    config = RetirementConfig(
        initial_portfolio=1_000_000,
        start_date='2024-01-01',
        end_date='2034-01-01',
        annual_withdrawal=0,  # No withdrawals
        current_portfolio={'SPY': 1.0}
    )

    engine = RetirementEngine(config)
    result = engine.run_single_path(mock_returns_good)

    assert result.success == True
    # With 8% returns and no withdrawals, portfolio should grow significantly
    assert result.final_value > 1_500_000


def test_100_percent_single_asset():
    """Test with 100% allocation to single asset"""
    config = RetirementConfig(
        initial_portfolio=1_000_000,
        start_date='2024-01-01',
        end_date='2034-01-01',
        annual_withdrawal=40_000,
        current_portfolio={'SPY': 1.0}  # 100% stocks
    )

    returns = pd.DataFrame({
        'SPY': [0.10] * 10
    })

    engine = RetirementEngine(config)
    result = engine.run_single_path(returns)

    # Portfolio return should equal SPY return
    assert all(r == pytest.approx(0.10) for r in result.annual_returns)


def test_flat_returns():
    """Test with zero returns (neither gains nor losses)"""
    config = RetirementConfig(
        initial_portfolio=1_000_000,
        start_date='2024-01-01',
        end_date='2034-01-01',
        annual_withdrawal=40_000,
        current_portfolio={'SPY': 0.6, 'AGG': 0.4}
    )

    flat_returns = pd.DataFrame({
        'SPY': [0.0] * 10,
        'AGG': [0.0] * 10
    })

    engine = RetirementEngine(config)
    result = engine.run_single_path(flat_returns)

    # All portfolio returns should be zero
    assert all(r == 0.0 for r in result.annual_returns)

    # Portfolio should decline by withdrawal amounts (with inflation)
    # This is a good sanity check


def test_depletion_stops_simulation():
    """Test that simulation stops when portfolio depletes"""
    config = RetirementConfig(
        initial_portfolio=100_000,
        start_date='2024-01-01',
        end_date='2034-01-01',  # 10 years
        annual_withdrawal=50_000,  # Withdrawing 50%!
        current_portfolio={'SPY': 1.0}
    )

    poor_returns = pd.DataFrame({
        'SPY': [-0.10] * 10
    })

    engine = RetirementEngine(config)
    result = engine.run_single_path(poor_returns)

    # Should deplete quickly
    assert result.success == False
    assert result.depletion_year < 5
    # Tracked arrays should be shorter than 10 years
    assert len(result.years) < 10
    assert len(result.portfolio_values) == len(result.years)


def test_simulation_path_dataclass():
    """Test SimulationPath dataclass structure"""
    path = SimulationPath(
        years=[0, 1, 2],
        dates=[],
        portfolio_values=[1_000_000, 1_050_000, 1_100_000],
        withdrawals=[40_000, 41_200, 42_436],
        annual_returns=[0.05, 0.05, 0.05],
        success=True,
        depletion_year=None,
        final_value=1_100_000
    )

    assert path.success == True
    assert path.final_value == 1_100_000
    assert len(path.years) == 3


def test_exact_depletion_edge_case():
    """Test edge case where portfolio depletes exactly at end"""
    config = RetirementConfig(
        initial_portfolio=100_000,
        start_date='2024-01-01',
        end_date='2026-01-01',  # 2 years
        annual_withdrawal=50_000,
        inflation_rate=0.0,  # No inflation for simplicity
        current_portfolio={'SPY': 1.0}
    )

    # Returns that exactly offset withdrawals
    returns = pd.DataFrame({
        'SPY': [0.0, 0.0]
    })

    engine = RetirementEngine(config)
    result = engine.run_single_path(returns)

    # After 2 years of $50k withdrawals from $100k: should deplete
    assert result.success == False


def test_different_portfolio_allocations():
    """Test with different portfolio allocations"""
    # 80/20 allocation
    config1 = RetirementConfig(
        initial_portfolio=1_000_000,
        start_date='2024-01-01',
        end_date='2034-01-01',
        annual_withdrawal=40_000,
        current_portfolio={'SPY': 0.8, 'AGG': 0.2}
    )

    # 20/80 allocation
    config2 = RetirementConfig(
        initial_portfolio=1_000_000,
        start_date='2024-01-01',
        end_date='2034-01-01',
        annual_withdrawal=40_000,
        current_portfolio={'SPY': 0.2, 'AGG': 0.8}
    )

    returns = pd.DataFrame({
        'SPY': [0.10] * 10,
        'AGG': [0.03] * 10
    })

    engine1 = RetirementEngine(config1)
    engine2 = RetirementEngine(config2)

    result1 = engine1.run_single_path(returns)
    result2 = engine2.run_single_path(returns)

    # 80/20 should have higher returns than 20/80
    assert result1.final_value > result2.final_value


# =========================================================================
# MONTE CARLO SIMULATION TESTS
# =========================================================================

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
    assert results.success_rate > 0.85
    assert results.median_final_value > 1_000_000


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


def test_monte_carlo_results_structure(basic_config):
    """Test MonteCarloResults dataclass structure"""
    engine = RetirementEngine(basic_config)
    results = engine.run_monte_carlo(num_simulations=50, seed=42, show_progress=False)

    assert results.num_simulations == 50
    assert 0 <= results.success_rate <= 1
    assert len(results.paths) == 50
    assert len(results.final_values) == 50


def test_monte_carlo_csv_export(basic_config):
    """Test CSV export functionality"""
    import tempfile
    import os

    engine = RetirementEngine(basic_config)
    results = engine.run_monte_carlo(num_simulations=50, seed=42, show_progress=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        results.export_to_csv(tmpdir)

        # Check that all files were created
        assert os.path.exists(os.path.join(tmpdir, 'summary_statistics.csv'))
        assert os.path.exists(os.path.join(tmpdir, 'final_values.csv'))
        assert os.path.exists(os.path.join(tmpdir, 'percentile_paths.csv'))

        # Read back and verify structure
        summary_df = pd.read_csv(os.path.join(tmpdir, 'summary_statistics.csv'))
        assert 'Metric' in summary_df.columns
        assert len(summary_df) == 10  # 10 metrics
