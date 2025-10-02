import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt


asset_classes = {
    'us_large': 0.40,
    'us_small': 0.10,
    'international': 0.20,
    'bonds': 0.25,
    'reits': 0.05
}



# Define regime-specific parameters for each
# Include cross-correlations
# High volatility regimes tend to persist
# Add regime duration parameter
# regime_params['bear_market']['avg_duration_months'] = 18
# regime_params['bull_market']['avg_duration_months'] = 48





def generate_regime_returns(regime_params, asset_allocation):
    """
    Generate correlated returns for multiple assets based on regime parameters
    
    Parameters:
    -----------
    regime_params : dict
        Dictionary containing regime-specific parameters:
        {
            'stocks': {'mean': 0.08, 'std': 0.18},
            'bonds': {'mean': 0.04, 'std': 0.06},
            'correlation_matrix': [[1.0, 0.2], [0.2, 1.0]]
        }
    asset_allocation : dict
        Portfolio weights {'stocks': 0.6, 'bonds': 0.4}
    
    Returns:
    --------
    dict : {'stocks_return': float, 'bonds_return': float, 'total_return': float}
    """
    
    # Extract parameters
    stock_mean = regime_params['stocks']['mean']
    stock_std = regime_params['stocks']['std']
    bond_mean = regime_params['bonds']['mean']
    bond_std = regime_params['bonds']['std']
    correlation = regime_params['correlation_matrix']
    
    # Generate correlated random returns using Cholesky decomposition
    # This ensures the returns maintain the specified correlation structure
    mean_vector = np.array([stock_mean, bond_mean])
    
    # Build covariance matrix from std devs and correlation
    cov_matrix = np.array([
        [stock_std**2, correlation[0][1] * stock_std * bond_std],
        [correlation[1][0] * stock_std * bond_std, bond_std**2]
    ])
    
    # Generate correlated normal random variables
    returns = np.random.multivariate_normal(mean_vector, cov_matrix)
    
    # Calculate portfolio return
    stock_return = returns[0]
    bond_return = returns[1]
    total_return = (stock_return * asset_allocation['stocks'] + 
                   bond_return * asset_allocation['bonds'])
    
    return {
        'stocks_return': stock_return,
        'bonds_return': bond_return,
        'total_return': total_return
    }


def generate_regime_returns_monthly(regime_params, asset_allocation):
    """
    Monthly version - converts annual parameters to monthly
    
    Key insight: Monthly mean ≈ annual_mean / 12
                 Monthly std ≈ annual_std / sqrt(12)
    """
    
    # Convert annual to monthly parameters
    stock_mean_monthly = regime_params['stocks']['mean'] / 12
    stock_std_monthly = regime_params['stocks']['std'] / np.sqrt(12)
    bond_mean_monthly = regime_params['bonds']['mean'] / 12
    bond_std_monthly = regime_params['bonds']['std'] / np.sqrt(12)
    
    monthly_params = {
        'stocks': {'mean': stock_mean_monthly, 'std': stock_std_monthly},
        'bonds': {'mean': bond_mean_monthly, 'std': bond_std_monthly},
        'correlation_matrix': regime_params['correlation_matrix']
    }
    
    return generate_regime_returns(monthly_params, asset_allocation)


def transition_regime(current_regime, transition_matrix, min_duration=None, 
                     months_in_regime=0):
    """
    Determine next regime based on transition probabilities
    
    Parameters:
    -----------
    current_regime : str
        Current market regime
    transition_matrix : dict
        Nested dict of transition probabilities
        Example: {'bull': {'bull': 0.9, 'bear': 0.1}, 'bear': {'bull': 0.3, 'bear': 0.7}}
    min_duration : int, optional
        Minimum months to stay in regime (adds persistence)
    months_in_regime : int
        How many months already spent in current regime
    
    Returns:
    --------
    tuple : (next_regime, months_in_new_regime)
    """
    
    # Enforce minimum duration if specified
    if min_duration and months_in_regime < min_duration:
        return current_regime, months_in_regime + 1
    
    # Get transition probabilities for current regime
    if current_regime not in transition_matrix:
        raise ValueError(f"Regime '{current_regime}' not found in transition matrix")
    
    next_regimes = list(transition_matrix[current_regime].keys())
    probabilities = list(transition_matrix[current_regime].values())
    
    # Ensure probabilities sum to 1
    prob_sum = sum(probabilities)
    if not np.isclose(prob_sum, 1.0):
        probabilities = [p / prob_sum for p in probabilities]
    
    # Sample next regime
    next_regime = np.random.choice(next_regimes, p=probabilities)
    
    # Reset counter if regime changed
    new_count = 1 if next_regime != current_regime else months_in_regime + 1
    
    return next_regime, new_count









def cape_to_regime(cape_value):
    """Map current CAPE to regime"""
    if cape_value < 15:
        return 'very_cheap'
    elif cape_value < 20:
        return 'cheap'
    elif cape_value < 25:
        return 'normal'
    elif cape_value < 30:
        return 'expensive'
    else:
        return 'very_expensive'



def adjust_withdrawal(portfolio_value, initial_value, 
                     base_withdrawal, regime):
    """
    Adjust spending based on portfolio performance
    and current market regime
    """
    
    current_ratio = portfolio_value / initial_value
    
    # Reduce spending in bear markets
    if regime in ['bear_market', 'very_expensive']:
        if current_ratio < 0.80:
            return base_withdrawal * 0.90  # 10% cut
    
    # Increase spending in bull markets  
    elif regime in ['bull_market', 'very_cheap']:
        if current_ratio > 1.50:
            return base_withdrawal * 1.10  # 10% raise
    
    return base_withdrawal



def plot_simulation_results(results, num_paths_to_show=100):
    """
    Create visualization of simulation results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Portfolio value paths (sample)
    ax1 = axes[0, 0]
    for i in range(min(num_paths_to_show, len(results['paths']))):
        path = results['paths'][i]
        color = 'green' if path['success'] else 'red'
        alpha = 0.1
        ax1.plot(path['periods'], path['portfolio_values'], 
                color=color, alpha=alpha, linewidth=0.5)
    
    ax1.set_xlabel('Period')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Sample Portfolio Paths')
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution of final values
    ax2 = axes[0, 1]
    successful_finals = [p['portfolio_values'][-1] for p in results['paths'] if p['success']]
    ax2.hist(successful_finals, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(results['median_final_value'], color='red', 
                linestyle='--', label='Median')
    ax2.set_xlabel('Final Portfolio Value ($)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Successful Outcomes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Regime distribution
    ax3 = axes[1, 0]
    regimes = list(results['regime_statistics'].keys())
    percentages = [results['regime_statistics'][r] * 100 for r in regimes]
    ax3.bar(regimes, percentages)
    ax3.set_xlabel('Regime')
    ax3.set_ylabel('Percentage of Time (%)')
    ax3.set_title('Time Spent in Each Regime')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. Success rate summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = f"""
    SIMULATION SUMMARY
    {'='*40}
    
    Success Rate: {results['success_rate']:.1%}
    
    Final Portfolio Values:
      Mean:    ${results['mean_final_value']:,.0f}
      Median:  ${results['median_final_value']:,.0f}
      Std Dev: ${results['std_final_value']:,.0f}
    
    Percentiles:
      5th:  ${results['percentiles']['5th']:,.0f}
      25th: ${results['percentiles']['25th']:,.0f}
      75th: ${results['percentiles']['75th']:,.0f}
      95th: ${results['percentiles']['95th']:,.0f}
    """
    ax4.text(0.1, 0.5, summary_text, fontsize=11, 
            family='monospace', verticalalignment='center')
    
    plt.tight_layout()
    return fig

# Use it:
# fig = plot_simulation_results(results)
# plt.savefig('retirement_simulation.png', dpi=300, bbox_inches='tight')
# plt.show()


def regime_switching_monte_carlo_complete(
    initial_portfolio=1_000_000,
    annual_withdrawal=40_000,
    years=30,
    initial_regime='normal',
    regime_params=None,
    transition_matrix=None,
    asset_allocation=None,
    num_simulations=5000,
    inflation_rate=0.03,
    frequency='annual'  # 'annual' or 'monthly'
):
    """
    Complete regime-switching Monte Carlo simulation for retirement planning
    
    Returns:
    --------
    dict containing:
        - 'paths': list of all simulation paths
        - 'success_rate': probability of success
        - 'median_final_value': median ending portfolio value
        - 'percentiles': portfolio values at key percentiles
        - 'regime_statistics': time spent in each regime
    """
    
    # Default asset allocation if not provided
    if asset_allocation is None:
        asset_allocation = {'stocks': 0.6, 'bonds': 0.4}
    
    # Determine time periods
    periods = years * 12 if frequency == 'monthly' else years
    withdrawal_frequency = 12 if frequency == 'monthly' else 1
    period_withdrawal = annual_withdrawal / withdrawal_frequency
    
    # Storage for results
    all_paths = []
    final_values = []
    success_count = 0
    regime_time = {regime: 0 for regime in transition_matrix.keys()}
    
    for sim in range(num_simulations):
        portfolio_value = initial_portfolio
        current_regime = initial_regime
        months_in_regime = 1
        
        path = {
            'periods': [],
            'portfolio_values': [],
            'regimes': [],
            'returns': [],
            'withdrawals': [],
            'success': True
        }
        
        for period in range(periods):
            # Calculate time-adjusted withdrawal (inflation)
            years_elapsed = period / withdrawal_frequency
            inflation_adjusted_withdrawal = period_withdrawal * (1 + inflation_rate) ** years_elapsed
            
            # Transition regime (with minimum duration enforcement)
            current_regime, months_in_regime = transition_regime(
                current_regime, 
                transition_matrix,
                min_duration=6 if frequency == 'monthly' else 1,
                months_in_regime=months_in_regime
            )
            
            # Generate returns for this period
            if frequency == 'monthly':
                returns = generate_regime_returns_monthly(
                    regime_params[current_regime], 
                    asset_allocation
                )
            else:
                returns = generate_regime_returns(
                    regime_params[current_regime], 
                    asset_allocation
                )
            
            # Apply returns
            portfolio_value *= (1 + returns['total_return'])
            
            # Subtract withdrawal
            portfolio_value -= inflation_adjusted_withdrawal
            
            # Track regime time
            regime_time[current_regime] += 1
            
            # Record this period
            path['periods'].append(period)
            path['portfolio_values'].append(portfolio_value)
            path['regimes'].append(current_regime)
            path['returns'].append(returns['total_return'])
            path['withdrawals'].append(inflation_adjusted_withdrawal)
            
            # Check for depletion
            if portfolio_value <= 0:
                path['success'] = False
                path['depletion_period'] = period
                break
        
        # Record final values
        if path['success']:
            success_count += 1
            final_values.append(portfolio_value)
        else:
            final_values.append(0)
        
        all_paths.append(path)
    
    # Calculate statistics
    success_rate = success_count / num_simulations
    final_values_array = np.array(final_values)
    
    results = {
        'paths': all_paths,
        'success_rate': success_rate,
        'median_final_value': np.median(final_values_array),
        'percentiles': {
            '5th': np.percentile(final_values_array, 5),
            '25th': np.percentile(final_values_array, 25),
            '50th': np.percentile(final_values_array, 50),
            '75th': np.percentile(final_values_array, 75),
            '95th': np.percentile(final_values_array, 95)
        },
        'regime_statistics': {
            regime: time / (num_simulations * periods) 
            for regime, time in regime_time.items()
        },
        'mean_final_value': np.mean(final_values_array),
        'std_final_value': np.std(final_values_array)
    }
    
    return results

# Define regime parameters based on historical CAPE ranges
regime_params = {
    'very_cheap': {  # CAPE < 15
        'stocks': {'mean': 0.125, 'std': 0.18},
        'bonds': {'mean': 0.045, 'std': 0.06},
        'correlation_matrix': [[1.0, 0.15], [0.15, 1.0]]
    },
    'cheap': {  # CAPE 15-20
        'stocks': {'mean': 0.102, 'std': 0.16},
        'bonds': {'mean': 0.042, 'std': 0.06},
        'correlation_matrix': [[1.0, 0.20], [0.20, 1.0]]
    },
    'normal': {  # CAPE 20-25
        'stocks': {'mean': 0.081, 'std': 0.17},
        'bonds': {'mean': 0.040, 'std': 0.06},
        'correlation_matrix': [[1.0, 0.25], [0.25, 1.0]]
    },
    'expensive': {  # CAPE 25-30
        'stocks': {'mean': 0.058, 'std': 0.19},
        'bonds': {'mean': 0.038, 'std': 0.06},
        'correlation_matrix': [[1.0, 0.30], [0.30, 1.0]]
    },
    'very_expensive': {  # CAPE > 30
        'stocks': {'mean': 0.032, 'std': 0.22},
        'bonds': {'mean': 0.036, 'std': 0.06},
        'correlation_matrix': [[1.0, 0.40], [0.40, 1.0]]
    }
}

# Define transition matrix (probabilities of moving between regimes)
# These should be estimated from historical data
transition_matrix = {
    'very_cheap': {'very_cheap': 0.70, 'cheap': 0.25, 'normal': 0.05, 'expensive': 0.0, 'very_expensive': 0.0},
    'cheap': {'very_cheap': 0.10, 'cheap': 0.60, 'normal': 0.25, 'expensive': 0.05, 'very_expensive': 0.0},
    'normal': {'very_cheap': 0.0, 'cheap': 0.15, 'normal': 0.50, 'expensive': 0.30, 'very_expensive': 0.05},
    'expensive': {'very_cheap': 0.0, 'cheap': 0.05, 'normal': 0.30, 'expensive': 0.50, 'very_expensive': 0.15},
    'very_expensive': {'very_cheap': 0.0, 'cheap': 0.0, 'normal': 0.10, 'expensive': 0.35, 'very_expensive': 0.55}
}

# Run simulation
results = regime_switching_monte_carlo_complete(
    initial_portfolio=1_000_000,
    annual_withdrawal=40_000,
    years=30,
    initial_regime='expensive',  # Current CAPE ~27
    regime_params=regime_params,
    transition_matrix=transition_matrix,
    asset_allocation={'stocks': 0.6, 'bonds': 0.4},
    num_simulations=5000,
    frequency='annual'
)

# Print results
print(f"Success Rate: {results['success_rate']:.1%}")
print(f"Median Final Value: ${results['median_final_value']:,.0f}")
print(f"\nPercentiles:")
for pct, value in results['percentiles'].items():
    print(f"  {pct}: ${value:,.0f}")
print(f"\nTime in Each Regime:")
for regime, pct in results['regime_statistics'].items():
    print(f"  {regime}: {pct:.1%}")

fig = plot_simulation_results(results)
plt.savefig('retirement_simulation.png', dpi=300, bbox_inches='tight')
plt.show()