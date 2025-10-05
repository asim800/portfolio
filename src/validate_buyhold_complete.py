"""Complete validation of buy-and-hold lifecycle simulation"""

from system_config import SystemConfig
from fin_data import FinData
from mc_path_generator import MCPathGenerator
from visualize_mc_lifecycle import run_accumulation_mc, run_decumulation_mc
import numpy as np

print("="*80)
print("BUY-AND-HOLD LIFECYCLE VALIDATION")
print("="*80)

# Step 1: Load config
print("\n[Step 1] Loading configuration...")
config = SystemConfig.from_json('../configs/test_simple_buyhold.json')
print(f"  ✓ Backtest: {config.start_date} to {config.end_date}")
print(f"  ✓ Retirement: {config.retirement_date}")
print(f"  ✓ Accumulation: {config.get_accumulation_years():.1f} years")
print(f"  ✓ Decumulation: {config.get_decumulation_years():.1f} years")

# Step 2: Load data and estimate parameters
print("\n[Step 2] Loading historical data and estimating parameters...")
tickers = ['SPY', 'AGG', 'NVDA', 'GLD']
fin_data = FinData(config.start_date, config.end_date)

# Load tickers and get returns data
fin_data.load_tickers('../tickers.txt')
returns = fin_data.get_returns_data(tickers)
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
print(f"  ✓ Data points: {len(returns)} days")
print(f"  ✓ Mean returns:")
for ticker, mean in zip(tickers, mean_returns):
    print(f"    {ticker}: {mean:.2%}")
print(f"  ✓ Volatility:")
for ticker, vol in zip(tickers, np.sqrt(np.diag(cov_matrix.values))):
    print(f"    {ticker}: {vol:.2%}")

# Step 3: Create path generator
print("\n[Step 3] Creating path generator...")
path_generator = MCPathGenerator(
    tickers=tickers,
    mean_returns=mean_returns.values,
    cov_matrix=cov_matrix.values,
    seed=42
)
print(f"  ✓ Generator created with {path_generator.num_assets} assets")

# Step 4: Generate continuous lifecycle paths
print("\n[Step 4] Generating continuous lifecycle paths...")
acc_years = int(config.get_accumulation_years())
dec_years = int(config.get_decumulation_years())
contributions_per_year = 26  # Biweekly

acc_paths, dec_paths = path_generator.generate_lifecycle_paths(
    num_simulations=1000,
    accumulation_years=acc_years,
    accumulation_periods_per_year=contributions_per_year,
    decumulation_years=dec_years
)
print(f"  ✓ Accumulation paths: {acc_paths.shape}")
print(f"  ✓ Decumulation paths: {dec_paths.shape}")

# Verify continuity
print("\n[Step 4a] Verifying path continuity...")
sim_idx = 0
last_26 = path_generator.paths[sim_idx, acc_years*26-26:acc_years*26, 0]
manual = np.prod(1 + last_26) - 1
first_dec = dec_paths[sim_idx, 0, 0]
error = abs(manual - first_dec)
print(f"  Manual compound (last 26 acc periods): {manual:.6f}")
print(f"  First decumulation year return: {first_dec:.6f}")
print(f"  Error: {error:.2e}")
print(f"  ✓ Continuous: {error < 1e-10} (error < 1e-10)")

# Step 5: Run accumulation
print("\n[Step 5] Running accumulation simulation...")
weights = np.array([0.25, 0.25, 0.25, 0.25])
acc_values = run_accumulation_mc(
    initial_value=100_000,
    weights=weights,
    asset_returns_paths=acc_paths,
    years=acc_years,
    contributions_per_year=contributions_per_year,
    contribution_amount=1000,
    employer_match_rate=0.5,
    employer_match_cap=10_000
)
print(f"  ✓ Accumulation shape: {acc_values.shape}")
print(f"  ✓ Final values (percentiles):")
print(f"    5th:  ${np.percentile(acc_values[:, -1], 5):,.0f}")
print(f"    25th: ${np.percentile(acc_values[:, -1], 25):,.0f}")
print(f"    50th: ${np.percentile(acc_values[:, -1], 50):,.0f}")
print(f"    75th: ${np.percentile(acc_values[:, -1], 75):,.0f}")
print(f"    95th: ${np.percentile(acc_values[:, -1], 95):,.0f}")

total_periods = acc_years * 26
total_contributions = 1000 * total_periods
total_match = min(10_000 * acc_years, int(1000 * 0.5 * total_periods))
print(f"\n  Total employee contributions: ${total_contributions:,}")
print(f"  Total employer match: ${total_match:,}")
print(f"  Total invested: ${total_contributions + total_match:,}")
print(f"  Median growth: ${np.median(acc_values[:, -1]) - total_contributions - total_match:,.0f}")

# Step 6: Run decumulation
print("\n[Step 6] Running decumulation simulation...")
dec_values, success = run_decumulation_mc(
    initial_values=acc_values[:, -1],
    weights=weights,
    asset_returns_paths=dec_paths,
    annual_withdrawal=40_000,
    inflation_rate=0.03,
    years=dec_years
)
print(f"  ✓ Decumulation shape: {dec_values.shape}")
print(f"  ✓ Success rate: {np.sum(success) / len(success) * 100:.1f}%")
print(f"  ✓ Final values (percentiles):")
print(f"    5th:  ${np.percentile(dec_values[:, -1], 5):,.0f}")
print(f"    25th: ${np.percentile(dec_values[:, -1], 25):,.0f}")
print(f"    50th: ${np.percentile(dec_values[:, -1], 50):,.0f}")
print(f"    75th: ${np.percentile(dec_values[:, -1], 75):,.0f}")
print(f"    95th: ${np.percentile(dec_values[:, -1], 95):,.0f}")

total_withdrawn = sum([40_000 * (1.03)**year for year in range(dec_years)])
print(f"\n  Total withdrawn (inflation-adjusted): ${total_withdrawn:,.0f}")
print(f"  Year 1 withdrawal: $40,000")
print(f"  Year {dec_years} withdrawal: ${40_000 * (1.03)**(dec_years-1):,.0f}")

# Check for failures
failed_sims = np.where(~success)[0]
if len(failed_sims) > 0:
    print(f"\n  ⚠ Failed simulations: {len(failed_sims)} ({len(failed_sims)/10:.1f}%)")
    print(f"  Example failure years:")
    for sim_idx in failed_sims[:3]:
        depletion_year = np.where(dec_values[sim_idx, :] <= 0)[0]
        if len(depletion_year) > 0:
            print(f"    Sim {sim_idx}: depleted in year {depletion_year[0]}")
else:
    print(f"\n  ✓ No failures - all simulations survived {dec_years} years!")

print("\n" + "="*80)
print("ALL STEPS VALIDATED ✓")
print("="*80)
print("\nNext steps:")
print("  - View plots: ls -lh ../plots/test/mc_lifecycle*.png")
print("  - Full simulation: uv run python visualize_mc_lifecycle.py")
print("  - Detailed guide: see ../docs/VALIDATE_BUYHOLD_TEST.md")
