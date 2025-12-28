#!/usr/bin/env python3
"""
Rebalancing Triggers Module.

Defines WHEN to rebalance the portfolio.
Separate from WHAT weights to use (see allocation_strategies.py).

Each trigger determines if rebalancing should occur based on:
- Time elapsed (periodic)
- Weight drift (threshold)
- External events (event-driven)
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Callable, Dict, Any
from datetime import datetime, date, timedelta
from abc import ABC, abstractmethod


class RebalancingTrigger(ABC):
    """
    Abstract base class for rebalancing triggers.

    Rebalancing triggers determine WHEN the portfolio should rebalance.
    They are independent of WHAT weights are used.
    """

    def __init__(self, name: str = "trigger"):
        """
        Initialize rebalancing trigger.

        Parameters:
        -----------
        name : str
            Trigger name for identification
        """
        self.name = name
        self.last_rebalance_date: Optional[date] = None
        logging.info(f"Initialized {self.__class__.__name__}: {name}")

    @abstractmethod
    def should_rebalance(self,
                        current_date: date,
                        current_weights: Optional[np.ndarray] = None,
                        target_weights: Optional[np.ndarray] = None,
                        **kwargs) -> bool:
        """
        Check if rebalancing should occur.

        Parameters:
        -----------
        current_date : date
            Current date to check
        current_weights : np.ndarray, optional
            Current portfolio weights (for threshold-based triggers)
        target_weights : np.ndarray, optional
            Target portfolio weights (for threshold-based triggers)
        **kwargs : dict
            Additional context (e.g., portfolio_values, returns_data)

        Returns:
        --------
        bool
            True if rebalancing should occur
        """
        pass

    def record_rebalance(self, rebalance_date: date):
        """
        Record that rebalancing occurred.

        Parameters:
        -----------
        rebalance_date : date
            Date when rebalancing was executed
        """
        self.last_rebalance_date = rebalance_date
        logging.debug(f"{self.name}: Recorded rebalance on {rebalance_date}")


class Never(RebalancingTrigger):
    """
    Never rebalance (buy and hold).

    Portfolio weights drift with market performance.
    Share counts remain constant over time.
    """

    def __init__(self, name: str = "never"):
        """Initialize never-rebalance trigger."""
        super().__init__(name)

    def should_rebalance(self,
                        current_date: date,
                        current_weights: Optional[np.ndarray] = None,
                        target_weights: Optional[np.ndarray] = None,
                        **kwargs) -> bool:
        """Always return False - never rebalance."""
        return False


class Periodic(RebalancingTrigger):
    """
    Time-based periodic rebalancing.

    Rebalances at fixed intervals: daily, weekly, monthly, quarterly, annual.
    Uses pandas frequency codes for flexibility.
    """

    def __init__(self, frequency: str = 'QE', name: Optional[str] = None):
        """
        Initialize periodic rebalancing trigger.

        Parameters:
        -----------
        frequency : str
            Pandas frequency code
            Common values:
            - 'D' = Daily
            - 'W' = Weekly
            - '2W' = Biweekly
            - 'ME' = Month-end
            - 'QE' = Quarter-end
            - 'YE' = Year-end
        name : str, optional
            Trigger name (defaults to frequency)

        Example:
        --------
        >>> # Quarterly rebalancing
        >>> trigger = Periodic('QE')
        >>>
        >>> # Monthly rebalancing
        >>> trigger = Periodic('ME')
        """
        if name is None:
            name = f"periodic_{frequency}"
        super().__init__(name)

        self.frequency = frequency

        # Convert frequency to approximate days for validation
        freq_to_days = {
            'D': 1, 'W': 7, '2W': 14, 'ME': 30, 'QE': 90, 'YE': 365
        }
        self.approx_days = freq_to_days.get(frequency, 30)

        logging.info(f"Periodic trigger: {frequency} (~{self.approx_days} days)")

    def should_rebalance(self,
                        current_date: date,
                        current_weights: Optional[np.ndarray] = None,
                        target_weights: Optional[np.ndarray] = None,
                        **kwargs) -> bool:
        """
        Check if enough time has elapsed since last rebalance.

        Uses approximate calendar day calculation.
        """
        if self.last_rebalance_date is None:
            # First period - always rebalance to establish baseline
            return True

        # Calculate days since last rebalance
        days_elapsed = (current_date - self.last_rebalance_date).days

        # Rebalance if enough time has passed
        return days_elapsed >= self.approx_days


class Threshold(RebalancingTrigger):
    """
    Weight drift threshold rebalancing.

    Rebalances when any asset weight drifts beyond specified threshold
    from its target weight.

    Example: If target is 60% stocks but current is 67%, and threshold is 5%,
    then rebalancing is triggered because drift (7%) > threshold (5%).
    """

    def __init__(self, drift_threshold: float = 0.05, name: Optional[str] = None):
        """
        Initialize threshold-based rebalancing trigger.

        Parameters:
        -----------
        drift_threshold : float
            Maximum allowable weight drift before rebalancing
            Example: 0.05 = 5% drift tolerance
        name : str, optional
            Trigger name (defaults to "threshold_{pct}%")

        Example:
        --------
        >>> # Rebalance when any weight drifts > 5%
        >>> trigger = Threshold(drift_threshold=0.05)
        """
        if name is None:
            name = f"threshold_{drift_threshold*100:.0f}pct"
        super().__init__(name)

        self.drift_threshold = drift_threshold
        logging.info(f"Threshold trigger: {drift_threshold*100:.1f}% drift tolerance")

    def should_rebalance(self,
                        current_date: date,
                        current_weights: Optional[np.ndarray] = None,
                        target_weights: Optional[np.ndarray] = None,
                        **kwargs) -> bool:
        """
        Check if weight drift exceeds threshold.

        Parameters:
        -----------
        current_date : date
            Current date (not used for threshold logic)
        current_weights : np.ndarray
            Current portfolio weights (required)
        target_weights : np.ndarray
            Target portfolio weights (required)
        **kwargs : dict
            Additional context

        Returns:
        --------
        bool
            True if any weight has drifted beyond threshold
        """
        # Need both current and target weights
        if current_weights is None or target_weights is None:
            logging.warning("Threshold trigger: Missing weights, cannot check drift")
            return False

        # Calculate absolute drift for each asset
        weight_drifts = np.abs(current_weights - target_weights)
        max_drift = np.max(weight_drifts)

        # Check if max drift exceeds threshold
        should_rebal = max_drift > self.drift_threshold

        if should_rebal:
            logging.debug(
                f"Threshold trigger: Max drift {max_drift:.1%} > "
                f"threshold {self.drift_threshold:.1%}"
            )

        return should_rebal


class Combined(RebalancingTrigger):
    """
    Combined trigger using multiple conditions.

    Rebalances if ANY of the component triggers activates (OR logic).
    Useful for: "Rebalance quarterly OR if drift exceeds 5%"
    """

    def __init__(self, triggers: list, name: str = "combined"):
        """
        Initialize combined trigger.

        Parameters:
        -----------
        triggers : list of RebalancingTrigger
            List of triggers to combine
        name : str
            Trigger name

        Example:
        --------
        >>> # Rebalance quarterly OR if drift > 5%
        >>> trigger = Combined([
        ...     Periodic('QE'),
        ...     Threshold(0.05)
        ... ], name="quarterly_or_threshold")
        """
        super().__init__(name)
        self.triggers = triggers
        logging.info(f"Combined trigger: {len(triggers)} conditions (OR logic)")

    def should_rebalance(self,
                        current_date: date,
                        current_weights: Optional[np.ndarray] = None,
                        target_weights: Optional[np.ndarray] = None,
                        **kwargs) -> bool:
        """Check if ANY component trigger activates."""
        for trigger in self.triggers:
            if trigger.should_rebalance(current_date, current_weights, target_weights, **kwargs):
                logging.debug(f"Combined trigger: {trigger.name} activated")
                return True
        return False

    def record_rebalance(self, rebalance_date: date):
        """Record rebalance in all component triggers."""
        super().record_rebalance(rebalance_date)
        for trigger in self.triggers:
            trigger.record_rebalance(rebalance_date)


class EventDriven(RebalancingTrigger):
    """
    Event-driven rebalancing.

    Rebalances when external event occurs, as determined by
    a user-provided event generator function.

    Events can be:
    - Volatility spikes
    - Drawdown breaches
    - Market regime changes
    - Custom user-defined conditions
    """

    def __init__(self,
                 event_generator: Callable[..., bool],
                 name: str = "event_driven"):
        """
        Initialize event-driven rebalancing trigger.

        Parameters:
        -----------
        event_generator : callable
            Function that returns True when event occurs
            Signature: event_generator(**kwargs) -> bool
        name : str
            Trigger name

        Example:
        --------
        >>> # Rebalance on volatility spike
        >>> def vol_spike(**kwargs):
        ...     returns_data = kwargs.get('returns_data')
        ...     current_vol = returns_data.tail(20).std().mean()
        ...     avg_vol = returns_data.std().mean()
        ...     return current_vol > 2.0 * avg_vol
        >>>
        >>> trigger = EventDriven(vol_spike, name="vol_spike")
        """
        super().__init__(name)
        self.event_generator = event_generator
        logging.info(f"EventDriven trigger: {name}")

    def should_rebalance(self,
                        current_date: date,
                        current_weights: Optional[np.ndarray] = None,
                        target_weights: Optional[np.ndarray] = None,
                        **kwargs) -> bool:
        """
        Check if event has occurred.

        Calls event_generator with all kwargs passed through.
        """
        try:
            # Call event generator with all context
            event_occurred = self.event_generator(
                current_date=current_date,
                current_weights=current_weights,
                target_weights=target_weights,
                **kwargs
            )

            if event_occurred:
                logging.debug(f"EventDriven trigger: Event detected on {current_date}")

            return event_occurred

        except Exception as e:
            logging.warning(f"EventDriven trigger: Exception in event_generator: {e}")
            return False


class CalendarBased(RebalancingTrigger):
    """
    Calendar-based rebalancing on specific dates.

    Rebalances on exact dates (e.g., first day of year, specific month-ends).
    More flexible than periodic for custom schedules.
    """

    def __init__(self,
                 rebalance_dates: list,
                 name: str = "calendar_based"):
        """
        Initialize calendar-based rebalancing trigger.

        Parameters:
        -----------
        rebalance_dates : list of date or str
            Specific dates to rebalance on
        name : str
            Trigger name

        Example:
        --------
        >>> # Rebalance on specific dates
        >>> trigger = CalendarBased([
        ...     date(2025, 3, 31),  # End of Q1
        ...     date(2025, 6, 30),  # End of Q2
        ...     date(2025, 9, 30),  # End of Q3
        ...     date(2025, 12, 31)  # End of Q4
        ... ])
        """
        super().__init__(name)

        # Convert strings to dates if needed
        self.rebalance_dates = set()
        for d in rebalance_dates:
            if isinstance(d, str):
                d = datetime.strptime(d, '%Y-%m-%d').date()
            self.rebalance_dates.add(d)

        logging.info(f"CalendarBased trigger: {len(self.rebalance_dates)} scheduled dates")

    def should_rebalance(self,
                        current_date: date,
                        current_weights: Optional[np.ndarray] = None,
                        target_weights: Optional[np.ndarray] = None,
                        **kwargs) -> bool:
        """Check if current date is in rebalance schedule."""
        return current_date in self.rebalance_dates


# ============================================================================
# DEMO: Show the rebalancing triggers
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("REBALANCING TRIGGERS DEMO")
    print("="*80)

    # Setup
    current_weights = np.array([0.67, 0.33, 0.0, 0.0])
    target_weights = np.array([0.60, 0.40, 0.0, 0.0])
    current_date = date(2025, 4, 1)

    print("\n1. Never (Buy & Hold):")
    never = Never()
    should_rebal = never.should_rebalance(current_date)
    print(f"   Should rebalance: {should_rebal}")

    print("\n2. Periodic (Quarterly):")
    periodic = Periodic('QE')
    periodic.record_rebalance(date(2025, 1, 1))  # Last rebalance 3 months ago
    should_rebal = periodic.should_rebalance(current_date)
    print(f"   Should rebalance: {should_rebal}")

    print("\n3. Threshold (5% drift):")
    threshold = Threshold(drift_threshold=0.05)
    should_rebal = threshold.should_rebalance(
        current_date,
        current_weights=current_weights,
        target_weights=target_weights
    )
    print(f"   Current weights: {current_weights}")
    print(f"   Target weights: {target_weights}")
    print(f"   Max drift: {np.max(np.abs(current_weights - target_weights)):.1%}")
    print(f"   Should rebalance: {should_rebal}")

    print("\n4. Event-Driven (Custom condition):")
    def custom_event(**kwargs):
        # Example: Rebalance if it's first day of month
        return kwargs['current_date'].day == 1

    event = EventDriven(custom_event, name="first_of_month")
    should_rebal = event.should_rebalance(current_date)
    print(f"   Current date: {current_date}")
    print(f"   Should rebalance: {should_rebal}")

    print("\n5. Combined (Quarterly OR 5% drift):")
    combined = Combined([
        Periodic('QE'),
        Threshold(0.05)
    ], name="quarterly_or_threshold")
    should_rebal = combined.should_rebalance(
        current_date,
        current_weights=current_weights,
        target_weights=target_weights
    )
    print(f"   Should rebalance: {should_rebal}")

    print("\n" + "="*80)
    print("All rebalancing triggers working correctly!")
    print("="*80)
