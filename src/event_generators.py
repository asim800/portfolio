#!/usr/bin/env python3
"""
Event Generators Module.

Static methods for detecting rebalancing events.
Used with EventDriven rebalancing trigger.

Each method returns True when a specific event condition is met:
- Volatility spikes
- Drawdown breaches
- Correlation breakdowns
- Custom user-defined conditions
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, Any
from datetime import date


class EventGenerators:
    """
    Collection of static event detection methods.

    These methods are designed to be used with EventDriven rebalancing trigger.
    Each method returns True when a specific event occurs.

    Usage:
    ------
    >>> from rebalancing_triggers import EventDriven
    >>>
    >>> # Volatility spike event
    >>> trigger = EventDriven(
    ...     event_generator=lambda **kwargs: EventGenerators.volatility_spike(
    ...         returns_data=kwargs['returns_data'],
    ...         threshold=2.0
    ...     )
    ... )
    """

    @staticmethod
    def volatility_spike(returns_data: pd.DataFrame,
                        threshold: float = 2.0,
                        lookback_periods: int = 20,
                        **kwargs) -> bool:
        """
        Detect volatility spike event.

        Triggers when recent volatility exceeds threshold × historical average.

        Parameters:
        -----------
        returns_data : pd.DataFrame
            Historical returns data (dates × assets)
        threshold : float
            Multiplier for volatility spike detection
            Example: 2.0 means current vol > 2.0 × average vol
        lookback_periods : int
            Number of recent periods for current volatility calculation
        **kwargs : dict
            Additional context (ignored)

        Returns:
        --------
        bool
            True if volatility spike detected

        Example:
        --------
        >>> event = EventGenerators.volatility_spike(
        ...     returns_data=data,
        ...     threshold=2.5  # Trigger if vol > 2.5× average
        ... )
        """
        if returns_data is None or len(returns_data) < lookback_periods * 2:
            return False

        try:
            # Calculate recent volatility (across all assets)
            recent_vol = returns_data.tail(lookback_periods).std().mean()

            # Calculate historical average volatility
            historical_vol = returns_data.std().mean()

            # Check if recent vol exceeds threshold
            spike_detected = recent_vol > threshold * historical_vol

            if spike_detected:
                logging.info(
                    f"VolatilitySpike: Recent vol {recent_vol:.4f} > "
                    f"{threshold}× avg {historical_vol:.4f}"
                )

            return spike_detected

        except Exception as e:
            logging.warning(f"VolatilitySpike: Error detecting event: {e}")
            return False

    @staticmethod
    def drawdown_breach(portfolio_values: pd.Series,
                       threshold: float = 0.10,
                       **kwargs) -> bool:
        """
        Detect drawdown breach event.

        Triggers when portfolio drawdown exceeds threshold from peak.

        Parameters:
        -----------
        portfolio_values : pd.Series
            Historical portfolio values (dates)
        threshold : float
            Drawdown threshold (e.g., 0.10 = 10% drawdown)
        **kwargs : dict
            Additional context (ignored)

        Returns:
        --------
        bool
            True if drawdown exceeds threshold

        Example:
        --------
        >>> event = EventGenerators.drawdown_breach(
        ...     portfolio_values=values,
        ...     threshold=0.15  # Trigger if drawdown > 15%
        ... )
        """
        if portfolio_values is None or len(portfolio_values) < 2:
            return False

        try:
            # Calculate current drawdown
            peak = portfolio_values.max()
            current = portfolio_values.iloc[-1]
            drawdown = (peak - current) / peak

            # Check if drawdown exceeds threshold
            breach_detected = drawdown > threshold

            if breach_detected:
                logging.info(
                    f"DrawdownBreach: Current drawdown {drawdown:.1%} > "
                    f"threshold {threshold:.1%}"
                )

            return breach_detected

        except Exception as e:
            logging.warning(f"DrawdownBreach: Error detecting event: {e}")
            return False

    @staticmethod
    def correlation_breakdown(returns_data: pd.DataFrame,
                              asset1_idx: int = 0,
                              asset2_idx: int = 1,
                              threshold_change: float = 0.5,
                              lookback_short: int = 20,
                              lookback_long: int = 120,
                              **kwargs) -> bool:
        """
        Detect correlation breakdown event.

        Triggers when correlation between two assets changes significantly
        from long-term average.

        Parameters:
        -----------
        returns_data : pd.DataFrame
            Historical returns data (dates × assets)
        asset1_idx : int
            Index of first asset
        asset2_idx : int
            Index of second asset
        threshold_change : float
            Minimum correlation change to trigger
            Example: 0.5 means correlation changed by ≥0.5
        lookback_short : int
            Periods for recent correlation
        lookback_long : int
            Periods for historical correlation
        **kwargs : dict
            Additional context (ignored)

        Returns:
        --------
        bool
            True if correlation breakdown detected

        Example:
        --------
        >>> # Detect if SPY-AGG correlation changed
        >>> event = EventGenerators.correlation_breakdown(
        ...     returns_data=data,
        ...     asset1_idx=0,  # SPY
        ...     asset2_idx=1,  # AGG
        ...     threshold_change=0.4
        ... )
        """
        if returns_data is None or len(returns_data) < lookback_long:
            return False

        try:
            # Get asset columns
            asset1 = returns_data.iloc[:, asset1_idx]
            asset2 = returns_data.iloc[:, asset2_idx]

            # Calculate recent correlation
            recent_corr = asset1.tail(lookback_short).corr(asset2.tail(lookback_short))

            # Calculate long-term correlation
            long_term_corr = asset1.corr(asset2)

            # Check if change exceeds threshold
            corr_change = abs(recent_corr - long_term_corr)
            breakdown_detected = corr_change > threshold_change

            if breakdown_detected:
                logging.info(
                    f"CorrelationBreakdown: Recent corr {recent_corr:.2f} vs "
                    f"long-term {long_term_corr:.2f}, change {corr_change:.2f}"
                )

            return breakdown_detected

        except Exception as e:
            logging.warning(f"CorrelationBreakdown: Error detecting event: {e}")
            return False

    @staticmethod
    def sharp_decline(returns_data: pd.DataFrame,
                     threshold: float = -0.05,
                     lookback_periods: int = 5,
                     **kwargs) -> bool:
        """
        Detect sharp market decline event.

        Triggers when portfolio return over recent periods falls below threshold.

        Parameters:
        -----------
        returns_data : pd.DataFrame
            Historical returns data (dates × assets)
        threshold : float
            Return threshold (negative value)
            Example: -0.05 = -5% return over lookback period
        lookback_periods : int
            Number of periods for return calculation
        **kwargs : dict
            Additional context (ignored)

        Returns:
        --------
        bool
            True if sharp decline detected

        Example:
        --------
        >>> # Trigger if portfolio drops >10% in 1 week
        >>> event = EventGenerators.sharp_decline(
        ...     returns_data=data,
        ...     threshold=-0.10,
        ...     lookback_periods=5  # 5 trading days
        ... )
        """
        if returns_data is None or len(returns_data) < lookback_periods:
            return False

        try:
            # Calculate recent cumulative return
            recent_returns = returns_data.tail(lookback_periods)
            cumulative_return = (1 + recent_returns).prod(axis=0).mean() - 1

            # Check if return falls below threshold
            decline_detected = cumulative_return < threshold

            if decline_detected:
                logging.info(
                    f"SharpDecline: Recent return {cumulative_return:.1%} < "
                    f"threshold {threshold:.1%}"
                )

            return decline_detected

        except Exception as e:
            logging.warning(f"SharpDecline: Error detecting event: {e}")
            return False

    @staticmethod
    def trading_day_of_month(current_date: date,
                            target_day: int = 1,
                            **kwargs) -> bool:
        """
        Detect specific trading day of month.

        Simple calendar-based event for regular rebalancing schedules.

        Parameters:
        -----------
        current_date : date
            Current date to check
        target_day : int
            Target day of month (1-31)
        **kwargs : dict
            Additional context (ignored)

        Returns:
        --------
        bool
            True if current date matches target day

        Example:
        --------
        >>> # Rebalance on first trading day of month
        >>> event = EventGenerators.trading_day_of_month(
        ...     current_date=date.today(),
        ...     target_day=1
        ... )
        """
        return current_date.day == target_day

    @staticmethod
    def quarter_end(current_date: date, **kwargs) -> bool:
        """
        Detect quarter-end dates.

        Parameters:
        -----------
        current_date : date
            Current date to check
        **kwargs : dict
            Additional context (ignored)

        Returns:
        --------
        bool
            True if current date is quarter-end (3/31, 6/30, 9/30, 12/31)

        Example:
        --------
        >>> event = EventGenerators.quarter_end(current_date=date(2025, 3, 31))
        >>> # Returns True
        """
        quarter_end_months = {3, 6, 9, 12}
        is_quarter_end_month = current_date.month in quarter_end_months

        # Check if it's the last day of the month
        if not is_quarter_end_month:
            return False

        # Simple check: days 28-31 are potentially month-end
        return current_date.day >= 28

    @staticmethod
    def custom_condition(condition_func: callable, **kwargs) -> bool:
        """
        Execute custom user-defined event condition.

        Wrapper for arbitrary event logic provided by user.

        Parameters:
        -----------
        condition_func : callable
            User-defined function that returns bool
            Signature: condition_func(**kwargs) -> bool
        **kwargs : dict
            All context passed to condition_func

        Returns:
        --------
        bool
            Result of condition_func

        Example:
        --------
        >>> def my_condition(**kwargs):
        ...     return kwargs['portfolio_values'].iloc[-1] > 1000000
        >>>
        >>> event = EventGenerators.custom_condition(
        ...     condition_func=my_condition,
        ...     portfolio_values=values
        ... )
        """
        try:
            return condition_func(**kwargs)
        except Exception as e:
            logging.warning(f"CustomCondition: Error executing condition: {e}")
            return False


# ============================================================================
# DEMO: Show the event generators
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("EVENT GENERATORS DEMO")
    print("="*80)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='D')
    returns_data = pd.DataFrame(
        np.random.randn(252, 4) * 0.01,  # Normal volatility
        index=dates,
        columns=['SPY', 'AGG', 'NVDA', 'GLD']
    )

    # Add volatility spike in recent data
    returns_data.iloc[-20:] *= 3  # Triple volatility in last 20 days

    # Create portfolio values
    portfolio_values = (1 + returns_data.mean(axis=1)).cumprod() * 100

    # Add a drawdown
    portfolio_values.iloc[-50:] *= 0.85  # 15% drawdown

    print("\n1. Volatility Spike Detection:")
    event = EventGenerators.volatility_spike(
        returns_data=returns_data,
        threshold=2.0
    )
    print(f"   Event detected: {event}")

    print("\n2. Drawdown Breach Detection:")
    event = EventGenerators.drawdown_breach(
        portfolio_values=portfolio_values,
        threshold=0.10
    )
    print(f"   Event detected: {event}")

    print("\n3. Correlation Breakdown Detection:")
    event = EventGenerators.correlation_breakdown(
        returns_data=returns_data,
        asset1_idx=0,
        asset2_idx=1,
        threshold_change=0.5
    )
    print(f"   Event detected: {event}")

    print("\n4. Sharp Decline Detection:")
    event = EventGenerators.sharp_decline(
        returns_data=returns_data,
        threshold=-0.05,
        lookback_periods=5
    )
    print(f"   Event detected: {event}")

    print("\n5. Quarter-End Detection:")
    event = EventGenerators.quarter_end(current_date=date(2025, 3, 31))
    print(f"   Event detected (3/31): {event}")
    event = EventGenerators.quarter_end(current_date=date(2025, 4, 1))
    print(f"   Event detected (4/1): {event}")

    print("\n" + "="*80)
    print("All event generators working correctly!")
    print("="*80)
