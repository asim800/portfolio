#!/usr/bin/env python3
"""
Unit tests for RetirementConfig.
"""

import pytest
from retirement_config import RetirementConfig


def test_basic_config_valid():
    """Test that valid configuration is accepted"""
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
    assert config.withdrawal_rate == 0.04


def test_config_without_retirement_date():
    """Test configuration when already retired (no retirement_date)"""
    config = RetirementConfig(
        initial_portfolio=1_000_000,
        start_date='2024-01-01',
        end_date='2054-01-01',
        # retirement_date is None (already retired)
        annual_withdrawal=40_000,
        current_portfolio={'SPY': 0.6, 'AGG': 0.4}
    )
    assert config.retirement_date is None
    assert config.num_years == 30


def test_invalid_dates_end_before_start():
    """Test that end_date must be after start_date"""
    with pytest.raises(ValueError, match="end_date must be after start_date"):
        RetirementConfig(
            initial_portfolio=1_000_000,
            start_date='2054-01-01',
            end_date='2024-01-01',  # Before start!
            annual_withdrawal=40_000,
            current_portfolio={'SPY': 1.0}
        )


def test_invalid_retirement_date_before_start():
    """Test that retirement_date cannot be before start_date"""
    with pytest.raises(ValueError, match="retirement_date must be between"):
        RetirementConfig(
            initial_portfolio=1_000_000,
            start_date='2024-01-01',
            end_date='2054-01-01',
            retirement_date='2020-01-01',  # Before start!
            annual_withdrawal=40_000,
            current_portfolio={'SPY': 1.0}
        )


def test_invalid_retirement_date_after_end():
    """Test that retirement_date cannot be after end_date"""
    with pytest.raises(ValueError, match="retirement_date must be between"):
        RetirementConfig(
            initial_portfolio=1_000_000,
            start_date='2024-01-01',
            end_date='2054-01-01',
            retirement_date='2060-01-01',  # After end!
            annual_withdrawal=40_000,
            current_portfolio={'SPY': 1.0}
        )


def test_invalid_portfolio_weights_too_low():
    """Test that portfolio weights must sum to 1.0"""
    with pytest.raises(ValueError, match="weights must sum to 1.0"):
        RetirementConfig(
            initial_portfolio=1_000_000,
            start_date='2024-01-01',
            end_date='2054-01-01',
            annual_withdrawal=40_000,
            current_portfolio={'SPY': 0.5, 'AGG': 0.3}  # Sum = 0.8, not 1.0!
        )


def test_invalid_portfolio_weights_too_high():
    """Test that portfolio weights cannot exceed 1.0"""
    with pytest.raises(ValueError, match="weights must sum to 1.0"):
        RetirementConfig(
            initial_portfolio=1_000_000,
            start_date='2024-01-01',
            end_date='2054-01-01',
            annual_withdrawal=40_000,
            current_portfolio={'SPY': 0.7, 'AGG': 0.5}  # Sum = 1.2, not 1.0!
        )


def test_portfolio_weights_rounding_tolerance():
    """Test that small rounding errors in weights are tolerated"""
    # Should accept weights that sum to 1.001 due to floating point
    config = RetirementConfig(
        initial_portfolio=1_000_000,
        start_date='2024-01-01',
        end_date='2054-01-01',
        annual_withdrawal=40_000,
        current_portfolio={'SPY': 0.6, 'AGG': 0.4, 'BIL': 0.001}  # Sum = 1.001
    )
    assert config is not None


def test_empty_portfolio():
    """Test that empty portfolio is rejected"""
    with pytest.raises(ValueError, match="current_portfolio cannot be empty"):
        RetirementConfig(
            initial_portfolio=1_000_000,
            start_date='2024-01-01',
            end_date='2054-01-01',
            annual_withdrawal=40_000,
            current_portfolio={}  # Empty!
        )


def test_negative_portfolio():
    """Test that negative initial portfolio is rejected"""
    with pytest.raises(ValueError, match="initial_portfolio must be positive"):
        RetirementConfig(
            initial_portfolio=-1_000_000,  # Negative!
            start_date='2024-01-01',
            end_date='2054-01-01',
            annual_withdrawal=40_000,
            current_portfolio={'SPY': 1.0}
        )


def test_zero_portfolio():
    """Test that zero initial portfolio is rejected"""
    with pytest.raises(ValueError, match="initial_portfolio must be positive"):
        RetirementConfig(
            initial_portfolio=0,  # Zero!
            start_date='2024-01-01',
            end_date='2054-01-01',
            annual_withdrawal=40_000,
            current_portfolio={'SPY': 1.0}
        )


def test_negative_withdrawal():
    """Test that negative withdrawal is rejected"""
    with pytest.raises(ValueError, match="annual_withdrawal cannot be negative"):
        RetirementConfig(
            initial_portfolio=1_000_000,
            start_date='2024-01-01',
            end_date='2054-01-01',
            annual_withdrawal=-1000,  # Negative!
            current_portfolio={'SPY': 1.0}
        )


def test_zero_withdrawal():
    """Test that zero withdrawal is allowed"""
    # Zero withdrawal should be allowed (no withdrawals during period)
    config = RetirementConfig(
        initial_portfolio=1_000_000,
        start_date='2024-01-01',
        end_date='2054-01-01',
        annual_withdrawal=0,  # Zero is OK
        current_portfolio={'SPY': 1.0}
    )
    assert config.annual_withdrawal == 0


def test_invalid_inflation_rate_negative():
    """Test that negative inflation rate is rejected"""
    with pytest.raises(ValueError, match="inflation_rate must be between 0 and 0.20"):
        RetirementConfig(
            initial_portfolio=1_000_000,
            start_date='2024-01-01',
            end_date='2054-01-01',
            annual_withdrawal=40_000,
            inflation_rate=-0.01,  # Negative!
            current_portfolio={'SPY': 1.0}
        )


def test_invalid_inflation_rate_too_high():
    """Test that unreasonably high inflation rate is rejected"""
    with pytest.raises(ValueError, match="inflation_rate must be between 0 and 0.20"):
        RetirementConfig(
            initial_portfolio=1_000_000,
            start_date='2024-01-01',
            end_date='2054-01-01',
            annual_withdrawal=40_000,
            inflation_rate=0.50,  # 50% inflation!
            current_portfolio={'SPY': 1.0}
        )


def test_multiple_tickers():
    """Test configuration with multiple tickers"""
    config = RetirementConfig(
        initial_portfolio=1_000_000,
        start_date='2024-01-01',
        end_date='2054-01-01',
        annual_withdrawal=40_000,
        current_portfolio={
            'SPY': 0.5,
            'AGG': 0.3,
            'BIL': 0.1,
            'GLD': 0.1
        }
    )
    assert len(config.tickers) == 4
    assert 'SPY' in config.tickers
    assert 'GLD' in config.tickers


def test_single_ticker():
    """Test configuration with single ticker (100% allocation)"""
    config = RetirementConfig(
        initial_portfolio=1_000_000,
        start_date='2024-01-01',
        end_date='2054-01-01',
        annual_withdrawal=40_000,
        current_portfolio={'SPY': 1.0}
    )
    assert len(config.tickers) == 1
    assert config.tickers[0] == 'SPY'


def test_num_years_calculation():
    """Test that num_years is calculated correctly"""
    # Exactly 10 years
    config = RetirementConfig(
        initial_portfolio=1_000_000,
        start_date='2024-01-01',
        end_date='2034-01-01',
        annual_withdrawal=40_000,
        current_portfolio={'SPY': 1.0}
    )
    assert config.num_years == 10

    # Exactly 30 years
    config = RetirementConfig(
        initial_portfolio=1_000_000,
        start_date='2024-01-01',
        end_date='2054-01-01',
        annual_withdrawal=40_000,
        current_portfolio={'SPY': 1.0}
    )
    assert config.num_years == 30


def test_withdrawal_rate_calculation():
    """Test that withdrawal rate is calculated correctly"""
    # 4% withdrawal rate
    config = RetirementConfig(
        initial_portfolio=1_000_000,
        start_date='2024-01-01',
        end_date='2054-01-01',
        annual_withdrawal=40_000,
        current_portfolio={'SPY': 1.0}
    )
    assert config.withdrawal_rate == pytest.approx(0.04)

    # 3% withdrawal rate
    config = RetirementConfig(
        initial_portfolio=1_000_000,
        start_date='2024-01-01',
        end_date='2054-01-01',
        annual_withdrawal=30_000,
        current_portfolio={'SPY': 1.0}
    )
    assert config.withdrawal_rate == pytest.approx(0.03)


def test_to_dict():
    """Test conversion to dictionary"""
    config = RetirementConfig(
        initial_portfolio=1_000_000,
        start_date='2024-01-01',
        end_date='2054-01-01',
        annual_withdrawal=40_000,
        current_portfolio={'SPY': 0.6, 'AGG': 0.4}
    )

    config_dict = config.to_dict()

    assert config_dict['initial_portfolio'] == 1_000_000
    assert config_dict['annual_withdrawal'] == 40_000
    assert config_dict['num_years'] == 30
    assert config_dict['withdrawal_rate'] == pytest.approx(0.04)
    assert config_dict['current_portfolio'] == {'SPY': 0.6, 'AGG': 0.4}


def test_default_values():
    """Test that default values are applied correctly"""
    config = RetirementConfig(
        initial_portfolio=1_000_000,
        start_date='2024-01-01',
        end_date='2054-01-01',
        # Use defaults for: annual_withdrawal, inflation_rate, rebalancing_strategy
        current_portfolio={'SPY': 1.0}
    )

    assert config.annual_withdrawal == 40_000  # Default
    assert config.inflation_rate == 0.03  # Default
    assert config.rebalancing_strategy == 'buy_and_hold'  # Default
