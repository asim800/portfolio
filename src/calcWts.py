import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class ReverseBuyHoldCalculator:
    def __init__(self, current_weights=None, symbols=None, weights_file=None):
        """
        Initialize the reverse buy-and-hold calculator.

        Parameters:
        current_weights (dict): Current portfolio weights {symbol: weight}
        symbols (list): List of stock symbols
        weights_file (str): Path to CSV/text file with Symbol,Weight columns
        """
        if weights_file is not None:
            self.current_weights, self.symbols = self.load_weights_from_file(weights_file)
        else:
            self.current_weights = current_weights
            self.symbols = symbols

        self.prices_df = None
        self.weights_df = None
        self.shares_df = None

    def load_weights_from_file(self, filepath):
        """
        Load current portfolio weights from a CSV/text file.

        Parameters:
        filepath (str): Path to file with Symbol,Weight columns

        Returns:
        tuple: (current_weights dict, symbols list)
        """
        try:
            # Try to read the file
            df = pd.read_csv(filepath)

            # Check for required columns (case insensitive)
            columns = [col.lower().strip() for col in df.columns]

            symbol_col = None
            weight_col = None

            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in ['symbol', 'symbols', 'ticker', 'stock']:
                    symbol_col = col
                elif col_lower in ['weight', 'weights', 'allocation']:
                    weight_col = col

            if symbol_col is None or weight_col is None:
                raise ValueError(f"Required columns not found. Available columns: {df.columns.tolist()}. "
                               "Expected columns like 'Symbol' and 'Weight'")

            # Convert to dictionary
            current_weights = {}
            symbols = []

            for _, row in df.iterrows():
                symbol = str(row[symbol_col]).strip().upper()
                weight = float(row[weight_col])

                current_weights[symbol] = weight
                symbols.append(symbol)

            # Validate weights sum to approximately 1.0
            total_weight = sum(current_weights.values())
            if abs(total_weight - 1.0) > 0.01:  # Allow 1% tolerance
                print(f"Warning: Weights sum to {total_weight:.3f}, not 1.0. Consider normalizing.")

                # Ask user if they want to normalize
                response = input("Do you want to normalize weights to sum to 1.0? (y/n): ").lower()
                if response in ['y', 'yes']:
                    normalized_weights = {k: v/total_weight for k, v in current_weights.items()}
                    current_weights = normalized_weights
                    print("Weights normalized to sum to 1.0")

            print(f"Loaded weights from {filepath}:")
            for symbol, weight in current_weights.items():
                print(f"  {symbol}: {weight:.1%}")
            print(f"Total weight: {sum(current_weights.values()):.3f}")

            return current_weights, symbols

        except FileNotFoundError:
            raise FileNotFoundError(f"Weights file not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Error loading weights file {filepath}: {str(e)}")
        
    def fetch_price_data(self, days_back=365):
        """
        Fetch historical price data going back specified days.

        Parameters:
        days_back (int): Number of days to go back (default 365 for 1 year)
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back + 30)  # Extra buffer for weekends/holidays

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        print(f"Fetching price data from {start_str} to {end_str}...")

        # Fetch data for all symbols
        data = yf.download(self.symbols, start=start_str, end=end_str, progress=False)

        print(f"Downloaded data structure: {type(data)}")
        if hasattr(data, 'columns'):
            print(f"Available columns: {data.columns.tolist()}")

        # Handle different data structures from yfinance
        if len(self.symbols) == 1:
            # Single symbol case
            if isinstance(data.columns, pd.MultiIndex):
                # Multi-level columns: use Adj Close
                if 'Adj Close' in [col[0] for col in data.columns]:
                    self.prices_df = pd.DataFrame({self.symbols[0]: data['Adj Close']})
                else:
                    # Fallback to Close if Adj Close not available
                    self.prices_df = pd.DataFrame({self.symbols[0]: data['Close']})
            else:
                # Single level columns
                if 'Adj Close' in data.columns:
                    self.prices_df = pd.DataFrame({self.symbols[0]: data['Adj Close']})
                elif 'Close' in data.columns:
                    self.prices_df = pd.DataFrame({self.symbols[0]: data['Close']})
                else:
                    raise KeyError(f"No price data found. Available columns: {data.columns.tolist()}")
        else:
            # Multiple symbols case
            if isinstance(data.columns, pd.MultiIndex):
                # Multi-level columns
                if 'Adj Close' in data.columns.get_level_values(0):
                    self.prices_df = data['Adj Close']
                elif 'Close' in data.columns.get_level_values(0):
                    self.prices_df = data['Close']
                else:
                    raise KeyError(f"No price data found. Available column levels: {data.columns.get_level_values(0).unique().tolist()}")
            else:
                # This shouldn't happen for multiple symbols, but handle it
                raise ValueError(f"Unexpected data structure for multiple symbols: {data.columns}")

        # Forward fill any missing values
        self.prices_df = self.prices_df.ffill()

        # Get exactly the period we want (most recent 'days_back' trading days)
        self.prices_df = self.prices_df.tail(days_back)

        print(f"Retrieved {len(self.prices_df)} days of data")
        print(f"Date range: {self.prices_df.index[0].date()} to {self.prices_df.index[-1].date()}")
        print(f"Final prices DataFrame shape: {self.prices_df.shape}")
        print(f"Final prices DataFrame columns: {self.prices_df.columns.tolist()}")

        return self.prices_df
    
    def calculate_historical_shares(self, current_portfolio_value=100000):
        """
        Calculate the number of shares that would result in current weights.
        Works backwards from current prices and weights.
        
        Parameters:
        current_portfolio_value (float): Current total portfolio value
        """
        current_prices = self.prices_df.iloc[-1]  # Most recent prices
        
        shares = {}
        for symbol in self.symbols:
            current_weight = self.current_weights[symbol]
            current_allocation = current_portfolio_value * current_weight
            shares[symbol] = current_allocation / current_prices[symbol]
            
        # Create shares dataframe (same shares held throughout the period)
        self.shares_df = pd.DataFrame([shares] * len(self.prices_df), 
                                    index=self.prices_df.index)
        
        print(f"\nCalculated share holdings (based on current weights):")
        for symbol, share_count in shares.items():
            print(f"  {symbol}: {share_count:.2f} shares")
            
        return shares
    
    def calculate_historical_weights(self):
        """Calculate what the portfolio weights were historically."""
        if self.prices_df is None:
            raise ValueError("Must fetch price data first")
            
        if self.shares_df is None:
            raise ValueError("Must calculate shares first")
        
        # Calculate daily portfolio values for each holding
        holdings_value = self.prices_df * self.shares_df
        
        # Calculate total portfolio value each day
        total_portfolio_value = holdings_value.sum(axis=1)
        
        # Calculate weights as percentage of total portfolio
        self.weights_df = holdings_value.div(total_portfolio_value, axis=0)
        
        return self.weights_df
    
    def get_historical_comparison(self):
        """Compare weights from 1 year ago to today."""
        if self.weights_df is None:
            raise ValueError("Must calculate weights first")
            
        one_year_ago_weights = self.weights_df.iloc[0].to_dict()
        current_weights_calculated = self.weights_df.iloc[-1].to_dict()
        
        comparison = {
            'one_year_ago_weights': one_year_ago_weights,
            'current_weights': current_weights_calculated,
            'weight_changes_over_year': {},
            'performance_impact': {}
        }
        
        # Calculate weight changes and performance impact
        for symbol in self.symbols:
            old_weight = one_year_ago_weights[symbol]
            new_weight = current_weights_calculated[symbol]
            weight_change = new_weight - old_weight
            
            comparison['weight_changes_over_year'][symbol] = weight_change
            
            # Calculate individual stock return
            stock_return = (self.prices_df.iloc[-1][symbol] / self.prices_df.iloc[0][symbol]) - 1
            comparison['performance_impact'][symbol] = stock_return
            
        return comparison
    
    def calculate_portfolio_return(self):
        """Calculate the total portfolio return over the period."""
        if self.prices_df is None or self.shares_df is None:
            raise ValueError("Must have price and share data")
            
        initial_value = (self.prices_df.iloc[0] * self.shares_df.iloc[0]).sum()
        final_value = (self.prices_df.iloc[-1] * self.shares_df.iloc[-1]).sum()
        
        return (final_value - initial_value) / initial_value
    
    def plot_weight_evolution(self, figsize=(14, 10)):
        """Plot how portfolio weights evolved over the past year."""
        if self.weights_df is None:
            raise ValueError("Must calculate weights first")
            
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        
        # Plot 1: Weight evolution over time
        for symbol in self.symbols:
            ax1.plot(self.weights_df.index, self.weights_df[symbol], 
                    label=symbol, linewidth=2, marker='o', markersize=1)
        
        ax1.set_title('Portfolio Weight Evolution (Past Year, Buy & Hold)', fontsize=16)
        ax1.set_ylabel('Weight', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, None)
        
        # Plot 2: Weight changes bar chart
        one_year_ago = self.weights_df.iloc[0]
        current = self.weights_df.iloc[-1]
        weight_changes = current - one_year_ago
        
        colors = ['green' if x >= 0 else 'red' for x in weight_changes]
        bars = ax2.bar(range(len(self.symbols)), weight_changes, 
                      color=colors, alpha=0.7)
        ax2.set_title('Weight Changes (Current vs 1 Year Ago)', fontsize=14)
        ax2.set_ylabel('Weight Change', fontsize=12)
        ax2.set_xticks(range(len(self.symbols)))
        ax2.set_xticklabels(self.symbols, rotation=45)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, change in zip(bars, weight_changes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{change:.1%}', ha='center', 
                    va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.show()
    
    def print_detailed_analysis(self):
        """Print a comprehensive analysis of the portfolio evolution."""
        comparison = self.get_historical_comparison()
        portfolio_return = self.calculate_portfolio_return()
        
        print("=" * 80)
        print("BUY & HOLD PORTFOLIO ANALYSIS - REVERSE CALCULATION")
        print("=" * 80)
        
        print(f"\nPORTFOLIO PERFORMANCE:")
        print(f"Total Return (Past Year): {portfolio_return:.2%}")
        
        print(f"\nWEIGHT COMPARISON:")
        print(f"{'Stock':<8} {'Last Period':<12} {'Current':<12} {'Change':<12} {'Stock Return':<12}")
        print("-" * 65)
        
        for symbol in self.symbols:
            old_weight = comparison['one_year_ago_weights'][symbol]
            new_weight = comparison['current_weights'][symbol]
            change = comparison['weight_changes_over_year'][symbol]
            stock_return = comparison['performance_impact'][symbol]
            
            print(f"{symbol:<8} {old_weight:>10.2%} {new_weight:>10.2%} "
                  f"{change:>+10.2%} {stock_return:>+10.2%}")
        
        print(f"\nINSIGHTS:")
        
        # Find best and worst performers
        returns = comparison['performance_impact']
        best_performer = max(returns.keys(), key=lambda x: returns[x])
        worst_performer = min(returns.keys(), key=lambda x: returns[x])
        
        print(f"• Best performing stock: {best_performer} ({returns[best_performer]:+.2%})")
        print(f"• Worst performing stock: {worst_performer} ({returns[worst_performer]:+.2%})")
        
        # Weight drift analysis
        weight_changes = comparison['weight_changes_over_year']
        max_increase = max(weight_changes.keys(), key=lambda x: weight_changes[x])
        max_decrease = min(weight_changes.keys(), key=lambda x: weight_changes[x])
        
        print(f"• Largest weight increase: {max_increase} ({weight_changes[max_increase]:+.2%})")
        print(f"• Largest weight decrease: {max_decrease} ({weight_changes[max_decrease]:+.2%})")
        
        # Calculate concentration change
        current_concentration = max(comparison['current_weights'].values())
        old_concentration = max(comparison['one_year_ago_weights'].values())
        
        if current_concentration > old_concentration:
            print(f"• Portfolio became more concentrated (max weight: {old_concentration:.1%} → {current_concentration:.1%})")
        else:
            print(f"• Portfolio became more diversified (max weight: {old_concentration:.1%} → {current_concentration:.1%})")
    
    def export_results(self, filename='historical_weights_analysis.csv'):
        """Export the historical weights and analysis to CSV."""
        if self.weights_df is None:
            raise ValueError("Must calculate weights first")
        
        # Create summary dataframe
        comparison = self.get_historical_comparison()
        
        summary_data = []
        for symbol in self.symbols:
            summary_data.append({
                'Symbol': symbol,
                'Weight_1_Year_Ago': comparison['one_year_ago_weights'][symbol],
                'Current_Weight': comparison['current_weights'][symbol],
                'Weight_Change': comparison['weight_changes_over_year'][symbol],
                'Stock_Return': comparison['performance_impact'][symbol]
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Export both detailed weights and summary
        with pd.ExcelWriter(filename.replace('.csv', '.xlsx')) as writer:
            self.weights_df.to_excel(writer, sheet_name='Daily_Weights')
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"Analysis exported to {filename.replace('.csv', '.xlsx')}")

# Example usage
def main():
    import sys

    print("REVERSE BUY & HOLD ANALYSIS")
    print("=" * 50)
    print("This analysis shows what your portfolio weights were")
    print("X days ago if you had been following a buy-and-hold strategy")
    print("to arrive at your current weights.\n")

    # Parse command line arguments
    weights_file = None
    days_back = 365  # Default to 1 year

    if len(sys.argv) > 1:
        weights_file = sys.argv[1]
        print(f"Loading weights from file: {weights_file}")

        # Check if days_back is provided as second argument
        if len(sys.argv) > 2:
            try:
                days_back = int(sys.argv[2])
                print(f"Using {days_back} days lookback period")
            except ValueError:
                print(f"Invalid days_back argument: {sys.argv[2]}. Using default 365 days.")
                days_back = 365

        calculator = ReverseBuyHoldCalculator(weights_file=weights_file)
    else:
        # Fallback to hardcoded weights if no file provided
        print("No weights file provided. Using default example weights.")
        print("Usage: python calcWts.py <weights_file.csv> [days_back]")
        print("  weights_file.csv: CSV file with Symbol,Weight columns")
        print("  days_back: Number of days to look back (default: 365)")
        print("\nExpected file format:")
        print("Symbol,Weight")
        print("AAPL,0.35")
        print("GOOGL,0.20")
        print("MSFT,0.30")
        print("TSLA,0.15")
        print("\nExamples:")
        print("  python calcWts.py my_weights.csv")
        print("  python calcWts.py my_weights.csv 730  # 2 years back")
        print("  python calcWts.py my_weights.csv 180  # 6 months back")
        print("\nUsing default weights for demonstration...\n")

        # Define CURRENT portfolio weights (what you have today)
        current_weights = {
            'AAPL': 0.35,   # Apple currently 35% of portfolio
            'GOOGL': 0.20,  # Google currently 20% of portfolio
            'MSFT': 0.30,   # Microsoft currently 30% of portfolio
            'TSLA': 0.15    # Tesla currently 15% of portfolio
        }

        symbols = list(current_weights.keys())
        calculator = ReverseBuyHoldCalculator(current_weights, symbols)

    current_portfolio_value = 100000  # Current portfolio worth $100k

    # Display analysis parameters
    print(f"Analysis Parameters:")
    print(f"  Lookback Period: {days_back} days ({days_back/365:.1f} years)")
    print(f"  Portfolio Value: ${current_portfolio_value:,}")
    print()

    # Fetch price data with configurable days_back
    calculator.fetch_price_data(days_back=days_back)

    # Calculate what share holdings would produce current weights
    calculator.calculate_historical_shares(current_portfolio_value)

    # Calculate historical weights
    calculator.calculate_historical_weights()

    # Print detailed analysis
    calculator.print_detailed_analysis()

    # Plot the evolution
    calculator.plot_weight_evolution()

    # Export results
    calculator.export_results('portfolio_reverse_analysis.xlsx')

if __name__ == "__main__":
    main()
