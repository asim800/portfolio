


Quick Start: Run Portfolio with MC Simulated Data
Option 1: Run the Example Script (Easiest)
cd /home/saahmed1/coding/python/fin/port/src3
uv run python example_simple_mc_portfolio.py
This script will:
Create a 4-asset portfolio (SPY, AGG, NVDA, GLD)
Generate 100 Monte Carlo simulations
Run portfolio analysis on one simulation
Show you the results
Option 2: Minimal Python Script (Copy-Paste)
#!/usr/bin/env python3
import sys
import numpy as np
sys.path.insert(0, '/home/saahmed1/coding/python/fin/port/src3')

from mc_path_generator import MCPathGenerator
from config import create_custom_config
from main import PortfolioOrchestrator

# 1. Define your assets
tickers = ['SPY', 'AGG']
mean_returns = np.array([0.10, 0.04])  # 10%, 4% annual returns
cov_matrix = np.array([[0.04, 0.01], [0.01, 0.02]])

# 2. Generate MC paths
generator = MCPathGenerator(tickers, mean_returns, cov_matrix, seed=42)
paths = generator.generate_paths(
    num_simulations=100,
    total_periods=260,  # 10 years × 26 biweekly periods
    periods_per_year=26
)

# 3. Create ticker file
with open('../tickers.txt', 'w') as f:
    f.write("Symbol,Weight\nSPY,0.6\nAGG,0.4\n")

# 4. Configure and run
config = create_custom_config(
    start_date='2025-01-01',
    end_date='2034-12-31',
    ticker_file='../tickers.txt',
    optimization_methods=['mean_variance'],
    rebalancing_strategies=['buy_and_hold']
)

orchestrator = PortfolioOrchestrator(config)
orchestrator.run_full_analysis(
    ticker_file='../tickers.txt',
    mc_generator=generator,    # ← Use MC data!
    simulation_idx=0,
    start_date='2025-01-01',
    frequency='2W'
)

print("Done! Check ../plots/rebalancing/ and ../results/rebalancing/")
Option 3: Interactive Python Session
cd /home/saahmed1/coding/python/fin/port/src3
uv run python
Then paste:
import numpy as np
from mc_path_generator import MCPathGenerator
from config import create_custom_config
from main import PortfolioOrchestrator

# Quick setup
gen = MCPathGenerator(['SPY', 'AGG'], np.array([0.10, 0.04]), 
                      np.array([[0.04, 0.01], [0.01, 0.02]]), seed=42)
gen.generate_paths(100, 260, 26)

# Create ticker file
open('../tickers.txt', 'w').write("Symbol,Weight\nSPY,0.6\nAGG,0.4\n")

# Run
config = create_custom_config(start_date='2025-01-01', end_date='2034-12-31',
                              ticker_file='../tickers.txt',
                              optimization_methods=['mean_variance'],
                              rebalancing_strategies=['buy_and_hold'])
orch = PortfolioOrchestrator(config)
orch.run_full_analysis('../tickers.txt', mc_generator=gen, simulation_idx=0, 
                       start_date='2025-01-01', frequency='2W')
Key Parameters to Customize
MC Generation:
num_simulations - How many scenarios (100-10000)
total_periods - Length of simulation (260 = 10 years biweekly)
periods_per_year - Frequency (26=biweekly, 12=monthly, 252=daily)
seed - Random seed for reproducibility
Portfolio Analysis:
simulation_idx - Which simulation to use (0 to num_simulations-1)
frequency - Date frequency ('D', 'W', '2W', 'ME', 'QE', 'YE')
optimization_methods - Strategies to test (['mean_variance', 'robust_mean_variance'])
rebalancing_strategies - Static strategies (['buy_and_hold', 'target_weight', 'equal_weight'])
Would you like me to run the example script now to show you what the output looks like?


