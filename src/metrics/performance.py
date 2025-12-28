#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 13:55:27 2019

@author: downey
"""
import pandas as pd
import numpy as np

##### Performance and Risk Analysis Functions ####

def annualized_return(x):
    '''Compute Annualized Return'''
    gross_return = x.iloc[-1] / x.iloc[0]
    days = len(x)
    years = days / 252
    ann_return = gross_return ** (1/years) - 1

    if isinstance(x, pd.DataFrame):
        df = pd.DataFrame({
            'Portfolio': ann_return.index,
            'Annualized Return': ann_return.values
        })
    else:
        # Handle Series case
        df = pd.DataFrame({
            'Portfolio': [x.name or 'Portfolio'],
            'Annualized Return': [ann_return]
        })
    return df

def annualized_standard_deviation(x):
    '''Compute Annualized Standard Deviation'''
    returns = x.pct_change().dropna()
    std = returns.std() * np.sqrt(252)

    if isinstance(x, pd.DataFrame):
        df = pd.DataFrame({
            'Portfolio': std.index,
            'Standard Deviation': std.values
        })
    else:
        # Handle Series case
        df = pd.DataFrame({
            'Portfolio': [x.name or 'Portfolio'],
            'Standard Deviation': [std]
        })
    return df

def max_drawdown(x):
    '''Max Peak to Trough Loss'''
    roll_max = x.expanding().max()
    daily_drawdown = x/roll_max - 1.0
    # Calculate the minimum (negative) daily drawdown
    max_daily_drawdown = daily_drawdown.expanding().min()

    # Get the minimum drawdown for each column
    if isinstance(x, pd.DataFrame):
        max_dd_values = max_daily_drawdown.min()
        max_dd = pd.DataFrame({
            'Portfolio': max_dd_values.index,
            'Max Drawdown': max_dd_values.values
        })
    else:
        # Handle Series case
        max_dd_value = max_daily_drawdown.min()
        max_dd = pd.DataFrame({
            'Portfolio': [x.name or 'Portfolio'],
            'Max Drawdown': [max_dd_value]
        })
    return max_dd

def gain_to_pain_ratio(x):
    '''Calculate Schwager's Gain to Pain Ratio'''
    returns = x.pct_change().dropna()
    positive_returns = returns[returns >= 0].sum()
    negative_returns = abs(returns[returns < 0].sum())
    gain_to_pain = positive_returns / negative_returns

    if isinstance(x, pd.DataFrame):
        df = pd.DataFrame({
            'Portfolio': gain_to_pain.index,
            'Gain to Pain Ratio': gain_to_pain.values
        })
    else:
        # Handle Series case
        df = pd.DataFrame({
            'Portfolio': [x.name or 'Portfolio'],
            'Gain to Pain Ratio': [gain_to_pain]
        })
    return df

def calmar_ratio(x):
    '''Annualized Return over Max Drawdown'''
    ann_ret = annualized_return(x)
    max_dd = max_drawdown(x)
    calmar_values = ann_ret['Annualized Return'].values / (-max_dd['Max Drawdown'].values)

    if isinstance(x, pd.DataFrame):
        df = pd.DataFrame({
            'Portfolio': x.columns,
            'Calmar Ratio': calmar_values.values
        })
    else:
        # Handle Series case
        df = pd.DataFrame({
            'Portfolio': [x.name or 'Portfolio'],
            'Calmar Ratio': [calmar_values.iloc[0]]
        })
    return df

def sharpe_ratio(x, RF=0):
    '''Annualized Return - RF rate / Standard Deviation'''
    returns = annualized_return(x)
    std = annualized_standard_deviation(x)
    data = returns.merge(std, on='Portfolio')

    sharpe_col = f'Sharpe Ratio (RF = {RF})'
    data[sharpe_col] = (data['Annualized Return'] - float(RF)) / data['Standard Deviation']

    return data[['Portfolio', sharpe_col]]

def sortino_ratio(x, RF=0):
    '''Similar to Sharpe Ratio but denominator is Std Dev. of downside volatility'''
    returns = annualized_return(x)
    RF_daily = RF / 252
    returns_data = x.pct_change().dropna()

    # Calculate downside deviation (negative excess returns)
    downside_returns = returns_data[returns_data < RF_daily]
    downside_std = downside_returns.std() * np.sqrt(252)

    if isinstance(x, pd.DataFrame):
        df = pd.DataFrame({
            'Portfolio': downside_std.index,
            'Downside Standard Deviation': downside_std.values
        })
    else:
        # Handle Series case
        df = pd.DataFrame({
            'Portfolio': [x.name or 'Portfolio'],
            'Downside Standard Deviation': [downside_std]
        })

    data = returns.merge(df, on='Portfolio')
    sortino_col = f'Sortino Ratio (RF = {RF})'
    data[sortino_col] = (data['Annualized Return'] - float(RF)) / data['Downside Standard Deviation']

    return data[['Portfolio', sortino_col]]
