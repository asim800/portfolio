#!/usr/bin/env python3
"""
Simulated data parameters for Monte Carlo validation testing.

This module contains controlled parameters with known mean/covariance
for validating MC path generation without Yahoo Finance dependency.
"""

import numpy as np


# Simulation settings
NUM_SIMULATIONS = 100  # Number of Monte Carlo simulations for validation
NUM_DAYS = 1825  # Number of days of simulated historical data (5 years)
RANDOM_SEED = 42  # Random seed for reproducibility

# Reindex method for time-varying parameters
# Options: 'ffill' (forward fill) or 'interpolate' (linear interpolation)
REINDEX_METHOD = 'ffill'


# ============================================================================
# Accumulation Phase Parameters (Regime 1)
# ============================================================================
# Annual mean returns for accumulation phase
# Order: [BIL, MSFT, NVDA, SPY]
tickers = ['SPY', 'AGG', 'NVDA', 'GLD']  # sim_params tickers (matches tickers.txt)
MEAN_ANNUAL_ACC = np.array([
    0.05,    # SPY: 5%
    0.025,   # AGG: 2.5% (low return, cash-like)
    0.15,    # MSFT: 15%
    0.175    # NVDA: 17.5% (high return)
])

# Annual covariance matrix for accumulation phase
# Order: [BIL, MSFT, NVDA, SPY]
COV_ANNUAL_ACC = np.array([
    [0.0001, 0.0150, 0.0200, 0.0400],  # SPY
    [0.0001, 0.0002, 0.0003, 0.0001],  # BIL (low volatility)
    [0.0002, 0.0400, 0.0300, 0.0150],  # MSFT
    [0.0003, 0.0300, 0.0900, 0.0200]   # NVDA (high volatility)
])


# ============================================================================
# Decumulation Phase Parameters (Regime 2)
# ============================================================================
# Annual mean returns for decumulation phase (zero returns for testing)
# Order: [BIL, MSFT, NVDA, SPY]
MEAN_ANNUAL_DEC = np.array([
    0.0,  # BIL: 0%
    0.0,  # MSFT: 0%
    0.0,  # NVDA: 0%
    0.0   # SPY: 0%
])

# Annual covariance matrix for decumulation phase (uncorrelated for testing)
# Order: [BIL, MSFT, NVDA, SPY]
COV_ANNUAL_DEC = np.array([
    [0.0001, 0.0000, 0.0000, 0.0000],  # BIL (low volatility, no correlation)
    [0.0000, 0.0400, 0.0000, 0.0000],  # MSFT (no correlation)
    [0.0000, 0.0000, 0.0900, 0.0000],  # NVDA (high volatility, no correlation)
    [0.0000, 0.0000, 0.0000, 0.0400]   # SPY (no correlation)
])


# ============================================================================
# Helper Functions
# ============================================================================

def add_regularization(cov_matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Add small regularization to ensure positive definiteness.

    This is common practice for numerical stability in covariance matrices.

    Parameters:
    -----------
    cov_matrix : np.ndarray
        Covariance matrix to regularize
    epsilon : float
        Small value to add to diagonal (default 1e-8)

    Returns:
    --------
    np.ndarray
        Regularized covariance matrix
    """
    n = cov_matrix.shape[0]
    return cov_matrix + epsilon * np.eye(n)


def get_accumulation_params(regularize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Get accumulation phase parameters.

    Parameters:
    -----------
    regularize : bool
        Whether to add regularization to covariance matrix (default True)

    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        (mean_returns, cov_matrix)
    """
    mean = MEAN_ANNUAL_ACC.copy()
    cov = COV_ANNUAL_ACC.copy()
    if regularize:
        cov = add_regularization(cov)
    return mean, cov


def get_decumulation_params(regularize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Get decumulation phase parameters.

    Parameters:
    -----------
    regularize : bool
        Whether to add regularization to covariance matrix (default True)

    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        (mean_returns, cov_matrix)
    """
    mean = MEAN_ANNUAL_DEC.copy()
    cov = COV_ANNUAL_DEC.copy()
    if regularize:
        cov = add_regularization(cov)
    return mean, cov


def create_simulated_returns_data(tickers: list[str],
                                   num_days: int = NUM_DAYS,
                                   seed: int = RANDOM_SEED) -> 'pd.DataFrame':
    """
    Generate simulated returns data using accumulation phase parameters.

    Parameters:
    -----------
    tickers : list[str]
        List of ticker symbols
    num_days : int
        Number of days of data to generate (default NUM_DAYS)
    seed : int
        Random seed for reproducibility (default RANDOM_SEED)

    Returns:
    --------
    pd.DataFrame
        DataFrame with simulated returns, columns = tickers
    """
    import pandas as pd

    # Get parameters
    mean_annual, cov_annual = get_accumulation_params(regularize=True)

    # Convert to daily
    daily_mean = mean_annual / 252
    daily_cov = cov_annual / 252

    # Generate returns
    np.random.seed(seed)
    returns_data = np.random.multivariate_normal(
        mean=daily_mean,
        cov=daily_cov,
        size=num_days
    )

    # Create DataFrame
    return pd.DataFrame(returns_data, columns=tickers)


# ============================================================================
# Save/Load Functions
# ============================================================================

def save_mean_returns_to_csv(filepath: str, tickers: list[str]) -> None:
    """
    Save mean returns (accumulation and decumulation) to CSV file.

    Creates a DataFrame with 2 rows (accumulation, decumulation) and columns for each ticker.

    Parameters:
    -----------
    filepath : str
        Path to save CSV file
    tickers : list[str]
        List of ticker symbols (column names)

    Example:
    --------
    >>> save_mean_returns_to_csv('../data/simulated_mean_returns.csv', ['BIL', 'MSFT', 'NVDA', 'SPY'])

    CSV format:
        regime,BIL,MSFT,NVDA,SPY
        accumulation,0.025,0.15,0.175,0.05
        decumulation,0.0,0.0,0.0,0.0
    """
    import pandas as pd
    import os

    # Get parameters
    mean_acc, _ = get_accumulation_params(regularize=False)
    mean_dec, _ = get_decumulation_params(regularize=False)

    # Create DataFrame with regime names as index
    df = pd.DataFrame(
        [mean_acc, mean_dec],
        index=['accumulation', 'decumulation'],
        columns=tickers
    )
    df.index.name = 'regime'

    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save to CSV
    df.to_csv(filepath)
    print(f"✓ Saved mean returns to: {filepath}")


def load_mean_returns_from_csv(filepath: str) -> 'pd.DataFrame':
    """
    Load mean returns from CSV file.

    Parameters:
    -----------
    filepath : str
        Path to CSV file

    Returns:
    --------
    pd.DataFrame
        DataFrame with 2 rows (accumulation, decumulation) and columns for each ticker
        Index: ['accumulation', 'decumulation']
        Columns: ticker symbols

    Example:
    --------
    >>> df = load_mean_returns_from_csv('../data/simulated_mean_returns.csv')
    >>> mean_acc = df.loc['accumulation'].values  # Extract as numpy array
    >>> mean_dec = df.loc['decumulation'].values
    """
    import pandas as pd

    df = pd.read_csv(filepath, index_col='regime')
    return df


def save_cov_matrices_to_txt(filepath: str) -> None:
    """
    Save covariance matrices (accumulation and decumulation) to text file.

    Concatenates COV_ANNUAL_ACC and COV_ANNUAL_DEC into a 3D array (2, n, n)
    where dimension 0 = regime (0=accumulation, 1=decumulation).

    Parameters:
    -----------
    filepath : str
        Path to save text file

    Format:
    -------
    Text file contains flattened 3D array that can be reshaped back to (2, 4, 4).
    Header line includes shape information for easy reconstruction.

    Example:
    --------
    >>> save_cov_matrices_to_txt('../data/simulated_cov_matrices.txt')
    """
    import os

    # Get parameters (without regularization for clean values)
    _, cov_acc = get_accumulation_params(regularize=False)
    _, cov_dec = get_decumulation_params(regularize=False)

    # Stack into 3D array: (2, n_assets, n_assets)
    cov_3d = np.stack([cov_acc, cov_dec], axis=0)

    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save with header containing shape
    header = f"Shape: {cov_3d.shape}\nRegimes: [0=accumulation, 1=decumulation]\nFormat: 3D array reshaped to 2D for saving"
    np.savetxt(filepath, cov_3d.reshape(cov_3d.shape[0], -1), header=header, comments='# ')

    print(f"✓ Saved covariance matrices to: {filepath}")
    print(f"  Shape: {cov_3d.shape}")


def load_cov_matrices_from_txt(filepath: str, n_assets: int = 4) -> np.ndarray:
    """
    Load covariance matrices from text file.

    Parameters:
    -----------
    filepath : str
        Path to text file
    n_assets : int
        Number of assets (default 4 for BIL, MSFT, NVDA, SPY)

    Returns:
    --------
    np.ndarray
        3D array with shape (2, n_assets, n_assets)
        [0] = accumulation covariance matrix
        [1] = decumulation covariance matrix

    Example:
    --------
    >>> cov_3d = load_cov_matrices_from_txt('../data/simulated_cov_matrices.txt')
    >>> cov_acc = cov_3d[0]  # Accumulation covariance
    >>> cov_dec = cov_3d[1]  # Decumulation covariance
    """
    # Load 2D array
    cov_2d = np.loadtxt(filepath)

    # Reshape back to 3D: (2, n_assets, n_assets)
    n_regimes = cov_2d.shape[0]
    cov_3d = cov_2d.reshape(n_regimes, n_assets, n_assets)

    return cov_3d


def save_all_parameters(mean_csv_path: str, cov_txt_path: str, tickers: list[str]) -> None:
    """
    Save both mean returns and covariance matrices to files.

    Convenience function to save all parameters at once.

    Parameters:
    -----------
    mean_csv_path : str
        Path to save mean returns CSV
    cov_txt_path : str
        Path to save covariance matrices text file
    tickers : list[str]
        List of ticker symbols

    Example:
    --------
    >>> save_all_parameters(
    ...     '../data/simulated_mean_returns.csv',
    ...     '../data/simulated_cov_matrices.txt',
    ...     ['BIL', 'MSFT', 'NVDA', 'SPY']
    ... )
    """
    print("Saving simulated data parameters...")
    save_mean_returns_to_csv(mean_csv_path, tickers)
    save_cov_matrices_to_txt(cov_txt_path)
    print("✓ All parameters saved successfully")


def load_all_parameters(mean_csv_path: str, cov_txt_path: str,
                       n_assets: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load both mean returns and covariance matrices from files.

    Parameters:
    -----------
    mean_csv_path : str
        Path to mean returns CSV
    cov_txt_path : str
        Path to covariance matrices text file
    n_assets : int
        Number of assets (default 4)

    Returns:
    --------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (mean_acc, mean_dec, cov_acc, cov_dec)

    Example:
    --------
    >>> mean_acc, mean_dec, cov_acc, cov_dec = load_all_parameters(
    ...     '../data/simulated_mean_returns.csv',
    ...     '../data/simulated_cov_matrices.txt'
    ... )
    """
    import pandas as pd

    # Load mean returns
    mean_df = load_mean_returns_from_csv(mean_csv_path)
    mean_acc = mean_df.loc['accumulation'].values
    mean_dec = mean_df.loc['decumulation'].values

    # Load covariance matrices
    cov_3d = load_cov_matrices_from_txt(cov_txt_path, n_assets)
    cov_acc = cov_3d[0]
    cov_dec = cov_3d[1]

    return mean_acc, mean_dec, cov_acc, cov_dec
