#!/usr/bin/env python3
"""
Period management using pandas resample() - clean and Pythonic!

Uses pandas built-in resampling instead of manual date arithmetic.
Much simpler, more flexible, and easier to maintain.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional
import logging

import ipdb


class PeriodManager:
    """
    Manages rebalancing periods using pandas resample().

    Advantages over manual date arithmetic:
    - Simpler code (uses pandas built-in functionality)
    - Calendar-aligned periods (month-end, quarter-end, etc.)
    - Flexible frequencies (monthly, weekly, quarterly, custom)
    - Handles edge cases automatically (leap years, weekends, etc.)
    - More Pythonic and maintainable
    """

    def __init__(self, returns_data: pd.DataFrame, frequency: str = 'ME'):
        """
        Initialize period manager with returns data.

        Parameters:
        -----------
        returns_data : pd.DataFrame
            Returns data with DatetimeIndex
        frequency : str, optional
            Pandas resample frequency string (default: 'ME' for month-end)

            Supports standard pandas frequencies AND custom multiples (e.g., '5T', '2W', '3ME').
            Any valid pandas offset alias can be used.

            Common frequencies:
            Time-based:
              'D'  : Calendar day
              'W'  : Weekly
              'ME' : Month end (recommended for monthly rebalancing)
              'MS' : Month start
              'Q'  : Quarter end
              'QS' : Quarter start
              'A'  : Year end
              'AS' : Year start

            Business day frequencies:
              'B'   : Business day
              'BME' : Business month end (replaces deprecated 'BM')
              'BMS' : Business month start
              'BQE' : Business quarter end (replaces deprecated 'BQ')
              'BAE' : Business year end (replaces deprecated 'BA')

            Custom periods (number + offset):
              '5T'  : Every 5 minutes (intraday)
              '21D' : Every 21 days
              '2W'  : Bi-weekly (every 2 weeks)
              '3ME' : Every 3 months (quarterly)
              '6ME' : Semi-annual (every 6 months)
              '10B' : Every 10 business days

        Examples:
        ---------
        >>> # Monthly rebalancing (month-end)
        >>> pm = PeriodManager(returns, frequency='ME')

        >>> # Quarterly rebalancing
        >>> pm = PeriodManager(returns, frequency='Q')

        >>> # Every 21 calendar days
        >>> pm = PeriodManager(returns, frequency='21D')

        >>> # Bi-weekly rebalancing (every 2 weeks)
        >>> pm = PeriodManager(returns, frequency='2W')

        >>> # Every 3 months (quarterly using custom multiple)
        >>> pm = PeriodManager(returns, frequency='3ME')

        >>> # Business month-end (excludes weekends/holidays)
        >>> pm = PeriodManager(returns, frequency='BME')
        """
        self.returns_data = returns_data.copy()
        self.frequency = frequency

        # Create period tracking DataFrame using resample
        self._create_period_groups()

        logging.info(f"PeriodManager initialized: {self.num_periods} periods (frequency={self.frequency})")

    def _create_period_groups(self) -> None:
        """
        Create period groupings using pandas resample() - elegant and simple!

        This replaces 46 lines of manual date arithmetic with ~20 lines using
        pandas built-in functionality.
        """
        # Group data by resampling frequency
        grouped = self.returns_data.resample(self.frequency)

        # Create period info for each group
        periods = []

        for period_num, (period_label, group_data) in enumerate(grouped):
            if len(group_data) == 0:
                continue  # Skip empty periods

            period_start = group_data.index[0]
            period_end = group_data.index[-1]

            periods.append({
                'period_num': period_num,
                'period_start': period_start,
                'period_end': period_end,
                'period_length': (period_end - period_start).days + 1,
                'trading_days': len(group_data),
                'label': period_label  # Resample label (for reference)
            })

        # Create DataFrame with period info
        self.periods_df = pd.DataFrame(periods)
        if not self.periods_df.empty:
            self.periods_df.set_index('period_num', inplace=True)

    @property
    def num_periods(self) -> int:
        """Total number of periods."""
        return len(self.periods_df)

    def get_period_info(self, period_num: int) -> pd.Series:
        """Get information for a specific period."""
        if period_num not in self.periods_df.index:
            raise ValueError(f"Period {period_num} not found. Available periods: 0-{self.num_periods-1}")
        return self.periods_df.loc[period_num]

    def get_period_data(self, period_num: int) -> pd.DataFrame:
        """
        Get returns data for a specific period using clean datetime filtering.

        Parameters:
        -----------
        period_num : int
            Period number (0-based)

        Returns:
        --------
        DataFrame with returns data for the period
        """
        period_info = self.get_period_info(period_num)

        # Use clean datetime comparison
        mask = ((self.returns_data.index >= period_info['period_start']) &
                (self.returns_data.index <= period_info['period_end']))

        return self.returns_data[mask]

    def get_expanding_window_data(self, current_period: int) -> pd.DataFrame:
        """
        Get expanding window data for optimization (all data up to previous period end).

        Parameters:
        -----------
        current_period : int
            Current period number

        Returns:
        --------
        DataFrame with historical data through previous period
        """
        if current_period == 0:
            return pd.DataFrame()  # No historical data for first period

        # Get end date of previous period
        prev_period_info = self.get_period_info(current_period - 1)
        end_date = prev_period_info['period_end']

        # Return all data up to and including previous period end
        return self.returns_data[self.returns_data.index <= end_date]

    def get_periods_summary(self) -> pd.DataFrame:
        """Get summary of all periods."""
        summary = self.periods_df.copy()
        summary['start_date'] = summary['period_start'].dt.date
        summary['end_date'] = summary['period_end'].dt.date
        return summary[['start_date', 'end_date', 'period_length', 'trading_days']]

    def iter_periods(self):
        """
        Iterator over periods yielding (period_num, period_data, period_info).

        Yields:
        -------
        tuple: (period_num, period_returns_data, period_info_series)
        """
        for period_num in self.periods_df.index:
            period_data = self.get_period_data(period_num)
            period_info = self.get_period_info(period_num)
            yield period_num, period_data, period_info

    def get_date_to_period_mapping(self) -> pd.Series:
        """
        Create mapping from each trading date to its period number.

        Returns:
        --------
        Series with dates as index and period numbers as values
        """
        date_to_period = pd.Series(index=self.returns_data.index, dtype=int, name='period_num')

        for period_num in self.periods_df.index:
            period_info = self.get_period_info(period_num)
            mask = ((self.returns_data.index >= period_info['period_start']) &
                   (self.returns_data.index <= period_info['period_end']))
            date_to_period[mask] = period_num

        return date_to_period