#!/usr/bin/env python3
"""
Bootstrap Sampling Module.

Provides bootstrap resampling methods for Monte Carlo simulation.
Extracted from src/fin_data.py sampling methods.
"""

import numpy as np
import pandas as pd
from typing import Optional


def bootstrap_returns(historical_returns: pd.DataFrame,
                     num_days: int,
                     seed: Optional[int] = None) -> pd.DataFrame:
    """
    Bootstrap resample returns with replacement.

    Parameters:
    -----------
    historical_returns : pd.DataFrame
        Historical returns data
    num_days : int
        Number of days to sample
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame: Sampled returns
    """
    if seed is not None:
        np.random.seed(seed)

    sampled_indices = np.random.choice(
        len(historical_returns),
        size=num_days,
        replace=True
    )
    return historical_returns.iloc[sampled_indices].reset_index(drop=True)


def parametric_sample(mean_returns: np.ndarray,
                     cov_matrix: np.ndarray,
                     num_samples: int,
                     seed: Optional[int] = None) -> np.ndarray:
    """
    Sample from multivariate normal distribution.

    Parameters:
    -----------
    mean_returns : np.ndarray
        Mean returns for each asset
    cov_matrix : np.ndarray
        Covariance matrix
    num_samples : int
        Number of samples to generate
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    np.ndarray: Sampled returns shape (num_samples, num_assets)
    """
    if seed is not None:
        np.random.seed(seed)

    return np.random.multivariate_normal(
        mean=mean_returns,
        cov=cov_matrix,
        size=num_samples
    )


def block_bootstrap(historical_returns: pd.DataFrame,
                   num_days: int,
                   block_size: int = 20,
                   seed: Optional[int] = None) -> pd.DataFrame:
    """
    Block bootstrap resampling to preserve autocorrelation.

    Parameters:
    -----------
    historical_returns : pd.DataFrame
        Historical returns data
    num_days : int
        Number of days to sample
    block_size : int
        Size of each block to sample
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame: Sampled returns preserving some autocorrelation
    """
    if seed is not None:
        np.random.seed(seed)

    n_obs = len(historical_returns)
    num_blocks = (num_days + block_size - 1) // block_size

    # Sample block starting positions
    max_start = n_obs - block_size
    if max_start <= 0:
        # If block size >= data length, fall back to regular bootstrap
        return bootstrap_returns(historical_returns, num_days, seed)

    block_starts = np.random.randint(0, max_start + 1, size=num_blocks)

    # Build sampled data from blocks
    samples = []
    for start in block_starts:
        block = historical_returns.iloc[start:start + block_size]
        samples.append(block)

    result = pd.concat(samples, ignore_index=True)
    return result.iloc[:num_days]


def stationary_bootstrap(historical_returns: pd.DataFrame,
                        num_days: int,
                        expected_block_length: int = 20,
                        seed: Optional[int] = None) -> pd.DataFrame:
    """
    Stationary bootstrap with random block lengths.

    Block lengths follow a geometric distribution, making the bootstrap
    stationary (probability of starting new block is constant).

    Parameters:
    -----------
    historical_returns : pd.DataFrame
        Historical returns data
    num_days : int
        Number of days to sample
    expected_block_length : int
        Expected length of each block (mean of geometric distribution)
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame: Sampled returns
    """
    if seed is not None:
        np.random.seed(seed)

    n_obs = len(historical_returns)
    p = 1.0 / expected_block_length  # Probability of starting new block

    samples = []
    current_idx = np.random.randint(0, n_obs)

    for _ in range(num_days):
        samples.append(historical_returns.iloc[current_idx])

        # Decide whether to start new block
        if np.random.random() < p:
            current_idx = np.random.randint(0, n_obs)
        else:
            current_idx = (current_idx + 1) % n_obs

    return pd.DataFrame(samples).reset_index(drop=True)
