#!/usr/bin/env python3
"""
Period management using pandas DataFrames and datetime indices.
Provides clean interface for portfolio rebalancing periods.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional
import logging

import ipdb

class PeriodManager:
    """
    Manages rebalancing periods using pandas DatetimeIndex for clean period tracking.
    """

    def __init__(self, returns_data: pd.DataFrame, rebalancing_period_days: int = 30):
        """
        Initialize period manager with returns data.

        Parameters:
        -----------
        returns_data : pd.DataFrame
            Returns data with DatetimeIndex
        rebalancing_period_days : int
            Number of calendar days per rebalancing period
        """
        self.returns_data = returns_data.copy()
        self.rebalancing_period_days = rebalancing_period_days

        # Create period tracking DataFrame
        self.periods_df = self._create_periods_dataframe()

        logging.info(f"PeriodManager initialized: {len(self.periods_df)} periods of ~{rebalancing_period_days} days")

    def _create_periods_dataframe(self) -> pd.DataFrame:
        """
        Create DataFrame tracking all rebalancing periods with datetime indices.

        Returns:
        --------
        DataFrame with columns: period_start, period_end, period_length, trading_days
        """
        start_date = self.returns_data.index[0].date()
        end_date = self.returns_data.index[-1].date()

        periods = []
        current_start = start_date
        period_num = 0

        while current_start <= end_date:
            current_end = current_start + timedelta(days=self.rebalancing_period_days - 1)

            # Don't go beyond available data
            if current_end > end_date:
                current_end = end_date

            # Get actual trading days in this period
            period_mask = ((self.returns_data.index.date >= current_start) &
                          (self.returns_data.index.date <= current_end))
            trading_days_in_period = period_mask.sum()

            periods.append({
                'period_num': period_num,
                'period_start': pd.Timestamp(current_start),
                'period_end': pd.Timestamp(current_end),
                'period_length': (current_end - current_start).days + 1,
                'trading_days': trading_days_in_period
            })

            # Move to next period
            current_start = current_end + timedelta(days=1 )
            period_num += 1

            if current_start > end_date:
                break

        # Create DataFrame with period_num as index
        df = pd.DataFrame(periods)
        df.set_index('period_num', inplace=True)

        return df

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