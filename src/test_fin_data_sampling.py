#!/usr/bin/env python3
"""
Unit tests for FinData return sampling methods (Monte Carlo).
"""

import pytest
import numpy as np
import pandas as pd
from fin_data import FinData


@pytest.fixture
def fin_data():
    """Create FinData instance with historical data"""
    fin_data = FinData(
        start_date='2020-01-01',
        end_date='2024-01-01',
        cache_dir='../data'
    )
    return fin_data


def test_bootstrap_sampling_basic(fin_data):
    """Test basic bootstrap sampling"""
    # Load data first
    fin_data.get_returns_data(['SPY', 'AGG'])

    # Sample 252 days (1 year)
    sampled = fin_data.sample_return_path(['SPY', 'AGG'], num_days=252, seed=42)

    assert len(sampled) == 252
    assert set(sampled.columns) == {'SPY', 'AGG'}  # Order doesn't matter
    assert sampled.isna().sum().sum() == 0  # No NaN values
    assert isinstance(sampled, pd.DataFrame)


def test_bootstrap_reproducibility(fin_data):
    """Test that same seed gives same results"""
    fin_data.get_returns_data(['SPY', 'AGG'])

    # Sample with same seed twice
    sampled1 = fin_data.sample_return_path(['SPY', 'AGG'], num_days=252, seed=42)
    sampled2 = fin_data.sample_return_path(['SPY', 'AGG'], num_days=252, seed=42)

    assert sampled1.equals(sampled2)


def test_bootstrap_different_seeds(fin_data):
    """Test that different seeds give different results"""
    fin_data.get_returns_data(['SPY', 'AGG'])

    sampled1 = fin_data.sample_return_path(['SPY', 'AGG'], num_days=252, seed=42)
    sampled2 = fin_data.sample_return_path(['SPY', 'AGG'], num_days=252, seed=123)

    # Should be different (very unlikely to be identical)
    assert not sampled1.equals(sampled2)


def test_bootstrap_statistical_properties(fin_data):
    """Test that bootstrap preserves statistical properties"""
    historical = fin_data.get_returns_data(['SPY'])

    # Sample many times and check convergence
    sample_means = []
    for i in range(100):
        sampled = fin_data.sample_return_path(['SPY'], num_days=252, seed=i)
        sample_means.append(sampled['SPY'].mean())

    # Mean of sample means should be close to historical mean
    # (Central Limit Theorem - standard error = std/sqrt(n))
    assert abs(np.mean(sample_means) - historical['SPY'].mean()) < 0.0005


def test_bootstrap_contains_historical_values(fin_data):
    """Test that bootstrap samples contain actual historical values"""
    historical = fin_data.get_returns_data(['SPY'])
    sampled = fin_data.sample_return_path(['SPY'], num_days=100, seed=42)

    # All sampled values should exist in historical data (within floating point tolerance)
    for value in sampled['SPY'].values:
        # Check if this value exists in historical data
        matches = np.isclose(historical['SPY'].values, value, rtol=1e-10)
        assert matches.any(), f"Sampled value {value} not found in historical data"


def test_parametric_sampling_basic(fin_data):
    """Test basic parametric sampling"""
    fin_data.get_returns_data(['SPY', 'AGG'])

    sampled = fin_data.sample_return_path(['SPY', 'AGG'], num_days=252, method='parametric', seed=42)

    assert len(sampled) == 252
    assert list(sampled.columns) == ['SPY', 'AGG']
    assert sampled.isna().sum().sum() == 0


def test_parametric_reproducibility(fin_data):
    """Test parametric sampling reproducibility"""
    fin_data.get_returns_data(['SPY', 'AGG'])

    sampled1 = fin_data.sample_return_path(['SPY', 'AGG'], num_days=252, method='parametric', seed=42)
    sampled2 = fin_data.sample_return_path(['SPY', 'AGG'], num_days=252, method='parametric', seed=42)

    assert sampled1.equals(sampled2)


def test_parametric_statistical_properties(fin_data):
    """Test that parametric sampling has similar mean/std to historical"""
    historical = fin_data.get_returns_data(['SPY'])

    # Sample large dataset for better convergence
    sampled = fin_data.sample_return_path(['SPY'], num_days=10000, method='parametric', seed=42)

    # Mean should be close to historical mean
    assert abs(sampled['SPY'].mean() - historical['SPY'].mean()) < 0.0005

    # Std should be close to historical std
    assert abs(sampled['SPY'].std() - historical['SPY'].std()) < 0.002


def test_invalid_sampling_method(fin_data):
    """Test that invalid method raises error"""
    fin_data.get_returns_data(['SPY'])

    with pytest.raises(ValueError, match="Unknown sampling method"):
        fin_data.sample_return_path(['SPY'], num_days=252, method='invalid_method')


def test_annual_returns_aggregation(fin_data):
    """Test annual returns aggregation"""
    fin_data.get_returns_data(['SPY', 'AGG'])

    annual_returns = fin_data.sample_annual_returns(['SPY', 'AGG'], num_years=10, seed=42)

    assert len(annual_returns) == 10  # 10 years
    assert list(annual_returns.columns) == ['SPY', 'AGG']

    # Annual returns should be larger magnitude than daily
    # Typical annual volatility for SPY is 15-20%
    assert annual_returns['SPY'].std() > 0.05


def test_annual_returns_reproducibility(fin_data):
    """Test annual returns reproducibility"""
    fin_data.get_returns_data(['SPY', 'AGG'])

    annual1 = fin_data.sample_annual_returns(['SPY', 'AGG'], num_years=10, seed=42)
    annual2 = fin_data.sample_annual_returns(['SPY', 'AGG'], num_years=10, seed=42)

    assert annual1.equals(annual2)


def test_annual_returns_calculation_correctness(fin_data):
    """Test that annual returns are calculated correctly"""
    fin_data.get_returns_data(['SPY'])

    # Use parametric for deterministic test
    annual_returns = fin_data.sample_annual_returns(['SPY'], num_years=1, seed=42, method='parametric')

    # Also get daily returns for same seed
    daily_returns = fin_data.sample_return_path(['SPY'], num_days=252, seed=42, method='parametric')

    # Annual return should equal (1+r1)*(1+r2)*...*(1+r252) - 1
    expected_annual = (1 + daily_returns['SPY']).prod() - 1
    actual_annual = annual_returns['SPY'].iloc[0]

    assert abs(expected_annual - actual_annual) < 1e-10


def test_different_trading_days_per_year(fin_data):
    """Test custom trading days per year parameter"""
    fin_data.get_returns_data(['SPY'])

    # Use 200 trading days instead of 252
    annual_returns = fin_data.sample_annual_returns(
        ['SPY'],
        num_years=5,
        trading_days_per_year=200,
        seed=42
    )

    assert len(annual_returns) == 5


def test_single_ticker_sampling(fin_data):
    """Test sampling with single ticker"""
    fin_data.get_returns_data(['SPY'])

    sampled = fin_data.sample_return_path(['SPY'], num_days=100, seed=42)

    assert len(sampled) == 100
    assert list(sampled.columns) == ['SPY']


def test_multiple_tickers_sampling(fin_data):
    """Test sampling with multiple tickers"""
    tickers = ['SPY', 'AGG', 'GLD', 'TLT']
    fin_data.get_returns_data(tickers)

    sampled = fin_data.sample_return_path(tickers, num_days=100, seed=42)

    assert len(sampled) == 100
    assert set(sampled.columns) == set(tickers)  # Order doesn't matter


def test_correlation_preserved_in_bootstrap(fin_data):
    """Test that bootstrap preserves correlations"""
    fin_data.get_returns_data(['SPY', 'AGG'])
    historical = fin_data.get_returns_data(['SPY', 'AGG'])

    # Historical correlation
    hist_corr = historical.corr().loc['SPY', 'AGG']

    # Sample many times and average correlation
    sample_corrs = []
    for i in range(50):
        sampled = fin_data.sample_return_path(['SPY', 'AGG'], num_days=1000, seed=i)
        sample_corrs.append(sampled.corr().loc['SPY', 'AGG'])

    avg_sample_corr = np.mean(sample_corrs)

    # Average sampled correlation should be close to historical
    assert abs(avg_sample_corr - hist_corr) < 0.05


def test_correlation_preserved_in_parametric(fin_data):
    """Test that parametric sampling preserves correlations"""
    historical = fin_data.get_returns_data(['SPY', 'AGG'])
    hist_corr = historical.corr().loc['SPY', 'AGG']

    # Large sample to get good estimate
    sampled = fin_data.sample_return_path(['SPY', 'AGG'], num_days=10000, method='parametric', seed=42)
    sample_corr = sampled.corr().loc['SPY', 'AGG']

    # Should be very close with large sample
    assert abs(sample_corr - hist_corr) < 0.02


def test_no_seed_gives_different_results(fin_data):
    """Test that no seed gives different results each time"""
    fin_data.get_returns_data(['SPY'])

    sampled1 = fin_data.sample_return_path(['SPY'], num_days=100)  # No seed
    sampled2 = fin_data.sample_return_path(['SPY'], num_days=100)  # No seed

    # Very unlikely to be identical
    assert not sampled1.equals(sampled2)


def test_large_sample_size(fin_data):
    """Test that large sample sizes work"""
    fin_data.get_returns_data(['SPY'])

    # 30 years * 252 days = 7560 days
    sampled = fin_data.sample_return_path(['SPY'], num_days=7560, seed=42)

    assert len(sampled) == 7560


def test_annual_returns_with_parametric(fin_data):
    """Test annual returns with parametric method"""
    fin_data.get_returns_data(['SPY', 'AGG'])

    annual_returns = fin_data.sample_annual_returns(
        ['SPY', 'AGG'],
        num_years=10,
        method='parametric',
        seed=42
    )

    assert len(annual_returns) == 10
    assert list(annual_returns.columns) == ['SPY', 'AGG']
