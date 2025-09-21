"""
Main Market Making Simulation with Delta Hedging
===============================================

Complete simulation that combines:
- Geometric Brownian Motion for underlying price evolution
- Market Maker with sophisticated quote generation
- Delta hedging strategy
- Comprehensive performance analysis

This script demonstrates a realistic market making operation where the agent:
1. Provides bid/ask quotes for options
2. Executes trades with market participants
3. Performs delta hedging to manage risk
4. Tracks P&L including all positions

Author: Main Simulation Module
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our modules
try:
    from market_maker import MarketMaker
    from market_simulator import simulate_stock_price, GeometricBrownianMotion
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure market_maker.py and market_simulator.py are in the same directory.")
    exit(1)


def get_vol_from_surface(underlying_price: float, strike_price: float, 
                        time_to_maturity: float, base_vol: float = 0.25, 
                        smile_curvature: float = 0.1) -> float:
    """
    Calculate volatility from a simplified volatility surface using a parabolic smile.
    
    The volatility smile reflects the market reality that options with different strikes
    and maturities trade at different implied volatilities, creating a "smile" pattern.
    
    Args:
        underlying_price: Current price of the underlying asset
        strike_price: Strike price of the option
        time_to_maturity: Time to maturity in years
        base_vol: Base volatility level (center of the smile)
        smile_curvature: Curvature parameter controlling smile intensity
        
    Returns:
        Implied volatility for the given strike and maturity
    """
    # Calculate moneyness (ratio of strike to spot)
    moneyness = strike_price / underlying_price
    
    # Parabolic smile formula: vol increases as we move away from ATM (moneyness = 1)
    smile_vol = base_vol + smile_curvature * (moneyness - 1) ** 2
    
    # Add time-dependent term decay (shorter maturity = higher vol)
    # This reflects the fact that short-term options often have higher implied vol
    time_adjustment = 0.05 * np.exp(-5 * time_to_maturity)  # Decays with time
    smile_vol += time_adjustment
    
    # Ensure volatility stays within reasonable bounds
    min_vol = 0.05  # 5% minimum
    max_vol = 1.00  # 100% maximum
    smile_vol = np.clip(smile_vol, min_vol, max_vol)
    
    return smile_vol


class MarketMakingSimulation:
    """
    Complete market making simulation with delta hedging.
    
    This class orchestrates the entire simulation including:
    - Price path generation
    - Market making operations
    - Delta hedging execution
    - Performance tracking and analysis
    """
    
    def __init__(self, simulation_params: Dict, market_maker_params: Dict):
        """
        Initialize the market making simulation.
        
        Args:
            simulation_params: Parameters for the simulation (prices, volatility, etc.)
            market_maker_params: Parameters for the market maker agent
        """
        self.sim_params = simulation_params
        self.mm_params = market_maker_params
        
        # Volatility surface parameters
        self.use_vol_surface = simulation_params.get('use_vol_surface', True)
        self.base_vol = simulation_params.get('base_vol', simulation_params['volatility'])
        self.smile_curvature = simulation_params.get('smile_curvature', 0.1)
        
        # Initialize market maker
        self.market_maker = MarketMaker(**market_maker_params)
        
        # Initialize GBM simulator
        self.gbm_simulator = GeometricBrownianMotion(seed=simulation_params.get('seed', 42))
        
        # Simulation results storage
        self.price_path = None
        self.time_grid = None
        self.simulation_log: List[Dict] = []
        self.hedge_log: List[Dict] = []
        self.trade_log: List[Dict] = []
        
        # Performance tracking
        self.portfolio_values: List[float] = []
        self.pnl_components: List[Dict] = []
    
    def generate_price_path(self) -> np.ndarray:
        """
        Generate the underlying asset price path using GBM.
        
        Returns:
            Array of simulated prices
        """
        S0 = self.sim_params['initial_price']
        mu = self.sim_params['drift']
        sigma = self.sim_params['volatility']
        T = self.sim_params['time_horizon']
        dt = self.sim_params['time_step']
        
        self.price_path = self.gbm_simulator.simulate_price_path(S0, mu, sigma, T, dt)
        self.time_grid = self.gbm_simulator.get_time_grid(T, dt)
        
        print(f"Generated price path: {len(self.price_path)} time steps")
        print(f"Initial price: ${S0:.2f}, Final price: ${self.price_path[-1]:.2f}")
        
        return self.price_path
    
    def calculate_time_to_maturity(self, step: int) -> float:
        """
        Calculate remaining time to maturity at given step.
        
        Args:
            step: Current simulation step
            
        Returns:
            Time to maturity in years
        """
        elapsed_time = self.time_grid[step]
        return max(0.001, self.sim_params['option_maturity'] - elapsed_time)  # Minimum 1 day
    
    def get_current_volatility(self, underlying_price: float, strike_price: float, 
                              time_to_maturity: float) -> float:
        """
        Get the current volatility to use for pricing and hedging.
        
        Args:
            underlying_price: Current underlying price
            strike_price: Option strike price
            time_to_maturity: Time to maturity
            
        Returns:
            Current volatility to use
        """
        if self.use_vol_surface:
            return get_vol_from_surface(underlying_price, strike_price, time_to_maturity,
                                      self.base_vol, self.smile_curvature)
        else:
            return self.sim_params['volatility']
    
    def simulate_market_orders(self, step: int, bid: float, ask: float) -> Tuple[bool, str, int, float]:
        """
        Simulate incoming market orders from other participants.
        
        Args:
            step: Current simulation step
            bid: Current bid price
            ask: Current ask price
            
        Returns:
            Tuple of (trade_occurred, side, quantity, execution_price)
        """
        # Order arrival probability
        order_probability = self.sim_params.get('order_probability', 0.08)
        
        if np.random.random() > order_probability:
            return False, '', 0, 0.0
        
        # Determine order side (buy from MM or sell to MM)
        if np.random.random() < 0.5:
            # Client buys from market maker (hits ask)
            side = 'sell'  # Market maker sells
            execution_price = ask
        else:
            # Client sells to market maker (hits bid)
            side = 'buy'   # Market maker buys
            execution_price = bid
        
        # Random order size (1-5 contracts)
        quantity = np.random.randint(1, 6)
        
        # Add some price improvement/slippage
        price_noise = np.random.normal(0, 0.001)  # Small random adjustment
        execution_price *= (1 + price_noise)
        execution_price = max(0.01, execution_price)  # Ensure positive
        
        return True, side, quantity, execution_price
    
    def run_simulation(self) -> Dict:
        """
        Run the complete market making simulation.
        
        Returns:
            Dictionary containing simulation results
        """
        print("Starting Market Making Simulation with Delta Hedging")
        print("=" * 55)
        
        # Generate price path
        if self.price_path is None:
            self.generate_price_path()
        
        # Simulation parameters
        strike_price = self.sim_params['strike_price']
        option_type = self.sim_params['option_type']
        
        # Track initial portfolio value
        initial_vol = self.get_current_volatility(self.price_path[0], strike_price, 
                                                 self.sim_params['option_maturity'])
        initial_portfolio_value = self.market_maker.get_portfolio_value(
            self.price_path[0], strike_price, self.sim_params['option_maturity'], 
            initial_vol, option_type
        )
        self.portfolio_values.append(initial_portfolio_value)
        
        print(f"Initial portfolio value: ${initial_portfolio_value:.2f}")
        print(f"Simulation parameters:")
        print(f"  Initial underlying: ${self.price_path[0]:.2f}")
        print(f"  Strike price: ${strike_price:.2f}")
        print(f"  Option maturity: {self.sim_params['option_maturity']:.4f} years")
        print(f"  Base volatility: {self.base_vol*100:.1f}%")
        print(f"  Using vol surface: {self.use_vol_surface}")
        if self.use_vol_surface:
            print(f"  Smile curvature: {self.smile_curvature:.3f}")
            print(f"  Initial vol (ATM): {initial_vol*100:.1f}%")
        print(f"  Order probability: {self.sim_params.get('order_probability', 0.08)*100:.1f}%")
        print()
        
        # Main simulation loop
        total_steps = len(self.price_path) - 1
        trade_count = 0
        hedge_count = 0
        
        for step in range(1, len(self.price_path)):
            current_price = self.price_path[step]
            time_to_maturity = self.calculate_time_to_maturity(step)
            
            # Skip if option has expired
            if time_to_maturity <= 0.001:
                break
            
            # Get current volatility (dynamic based on vol surface)
            current_volatility = self.get_current_volatility(current_price, strike_price, time_to_maturity)
            
            # Market maker calculates quotes
            bid, ask = self.market_maker.calculate_quotes(
                current_price, strike_price, time_to_maturity, current_volatility, option_type
            )
            
            # Simulate incoming market orders
            trade_occurred, side, quantity, execution_price = self.simulate_market_orders(
                step, bid, ask
            )
            
            step_info = {
                'step': step,
                'time': self.time_grid[step],
                'underlying_price': current_price,
                'time_to_maturity': time_to_maturity,
                'current_volatility': current_volatility,
                'bid': bid,
                'ask': ask,
                'spread': ask - bid,
                'trade_occurred': trade_occurred
            }
            
            # Execute trade if one occurred
            hedge_info = None
            if trade_occurred:
                try:
                    # Execute the option trade
                    trade_info = self.market_maker.execute_trade(
                        quantity, execution_price, side, current_price, strike_price,
                        time_to_maturity, current_volatility, option_type
                    )
                    
                    trade_count += 1
                    step_info.update({
                        'trade_side': side,
                        'trade_quantity': quantity,
                        'trade_price': execution_price,
                        'inventory_after': self.market_maker.inventory,
                        'cash_after': self.market_maker.cash
                    })
                    
                    self.trade_log.append({
                        'step': step,
                        'time': self.time_grid[step],
                        **trade_info
                    })
                    
                    # DELTA HEDGE - This is the key step
                    hedge_info = self.market_maker.delta_hedge(
                        current_price, strike_price, time_to_maturity, current_volatility, option_type
                    )
                    
                    if hedge_info['hedge_type'] != 'no_hedge_needed':
                        hedge_count += 1
                        step_info.update({
                            'hedge_occurred': True,
                            'shares_traded': hedge_info['shares_traded'],
                            'hedge_cost': hedge_info['hedge_cost'],
                            'underlying_shares_after': self.market_maker.underlying_shares,
                            'portfolio_delta': hedge_info.get('portfolio_delta', 0),
                            'net_delta_after': hedge_info.get('net_delta_after', 0)
                        })
                        
                        self.hedge_log.append({
                            'step': step,
                            'time': self.time_grid[step],
                            **hedge_info
                        })
                    else:
                        step_info['hedge_occurred'] = False
                        
                except Exception as e:
                    print(f"Error in step {step}: {e}")
                    step_info['error'] = str(e)
            
            # Calculate current portfolio value
            portfolio_value = self.market_maker.get_portfolio_value(
                current_price, strike_price, time_to_maturity, current_volatility, option_type
            )
            self.portfolio_values.append(portfolio_value)
            
            # Calculate P&L components
            pnl_components = {
                'step': step,
                'time': self.time_grid[step],
                'cash': self.market_maker.cash,
                'inventory': self.market_maker.inventory,
                'underlying_shares': self.market_maker.underlying_shares,
                'portfolio_value': portfolio_value,
                'total_pnl': portfolio_value - initial_portfolio_value
            }
            self.pnl_components.append(pnl_components)
            
            step_info.update({
                'portfolio_value': portfolio_value,
                'total_pnl': portfolio_value - initial_portfolio_value,
                'mm_inventory': self.market_maker.inventory,
                'mm_cash': self.market_maker.cash,
                'mm_shares': self.market_maker.underlying_shares
            })
            
            self.simulation_log.append(step_info)
            
            # Progress reporting
            if step % (total_steps // 10) == 0:
                progress = (step / total_steps) * 100
                print(f"Progress: {progress:.1f}% - Trades: {trade_count}, Hedges: {hedge_count}, "
                      f"P&L: ${portfolio_value - initial_portfolio_value:.2f}")
        
        # Final calculations
        final_price = self.price_path[-1]
        final_portfolio_value = self.portfolio_values[-1]
        
        # For simplicity, assume options expire worthless (as mentioned in requirements)
        # Final P&L calculation including underlying shares
        final_pnl_simple = (self.market_maker.cash - self.market_maker.starting_cash + 
                           self.market_maker.underlying_shares * final_price)
        
        results = {
            'initial_portfolio_value': initial_portfolio_value,
            'final_portfolio_value': final_portfolio_value,
            'total_pnl': final_portfolio_value - initial_portfolio_value,
            'final_pnl_simple': final_pnl_simple,
            'total_trades': trade_count,
            'total_hedges': hedge_count,
            'final_inventory': self.market_maker.inventory,
            'final_cash': self.market_maker.cash,
            'final_underlying_shares': self.market_maker.underlying_shares,
            'final_underlying_price': final_price,
            'simulation_steps': len(self.simulation_log),
            'price_return': (final_price / self.price_path[0]) - 1
        }
        
        print(f"\nSimulation Complete!")
        print(f"Total trades executed: {trade_count}")
        print(f"Total hedges performed: {hedge_count}")
        print(f"Final P&L: ${results['total_pnl']:.2f}")
        print(f"Final P&L (simple): ${results['final_pnl_simple']:.2f}")
        
        return results
    
    def plot_simulation_results(self) -> None:
        """
        Create comprehensive plots of simulation results.
        """
        if not self.simulation_log:
            print("No simulation data to plot. Run simulation first.")
            return
        
        # Convert logs to DataFrames for easier plotting
        sim_df = pd.DataFrame(self.simulation_log)
        pnl_df = pd.DataFrame(self.pnl_components)
        
        # Create comprehensive plotting layout
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Underlying price path with trade markers
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(self.time_grid, self.price_path, 'b-', linewidth=2, label='Underlying Price')
        
        # Mark trade times
        trade_times = sim_df[sim_df['trade_occurred']]['time']
        trade_prices = sim_df[sim_df['trade_occurred']]['underlying_price']
        ax1.scatter(trade_times, trade_prices, c='red', s=30, alpha=0.7, label='Trades')
        
        ax1.set_title('Underlying Price Path with Trade Markers')
        ax1.set_xlabel('Time (Years)')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Portfolio value over time
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(pnl_df['time'], pnl_df['portfolio_value'], 'g-', linewidth=2)
        ax2.set_title('Portfolio Value Over Time')
        ax2.set_xlabel('Time (Years)')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.grid(True, alpha=0.3)
        
        # 3. P&L over time
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(pnl_df['time'], pnl_df['total_pnl'], 'purple', linewidth=2)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax3.set_title('Total P&L Over Time')
        ax3.set_xlabel('Time (Years)')
        ax3.set_ylabel('P&L ($)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Option inventory over time
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(sim_df['time'], sim_df['mm_inventory'], 'orange', linewidth=2)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax4.set_title('Option Inventory Over Time')
        ax4.set_xlabel('Time (Years)')
        ax4.set_ylabel('Contracts')
        ax4.grid(True, alpha=0.3)
        
        # 5. Underlying shares position
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(sim_df['time'], sim_df['mm_shares'], 'brown', linewidth=2)
        ax5.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax5.set_title('Underlying Shares Position (Delta Hedge)')
        ax5.set_xlabel('Time (Years)')
        ax5.set_ylabel('Shares')
        ax5.grid(True, alpha=0.3)
        
        # 6. Cash position over time
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(pnl_df['time'], pnl_df['cash'], 'teal', linewidth=2)
        ax6.axhline(y=self.market_maker.starting_cash, color='red', linestyle='--', alpha=0.5, label='Starting Cash')
        ax6.set_title('Cash Position Over Time')
        ax6.set_xlabel('Time (Years)')
        ax6.set_ylabel('Cash ($)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Bid-Ask spreads
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(sim_df['time'], sim_df['spread'], 'cyan', linewidth=2)
        ax7.set_title('Bid-Ask Spread Over Time')
        ax7.set_xlabel('Time (Years)')
        ax7.set_ylabel('Spread ($)')
        ax7.grid(True, alpha=0.3)
        
        # 8. Trade execution prices
        ax8 = plt.subplot(3, 3, 8)
        trade_data = sim_df[sim_df['trade_occurred']]
        if len(trade_data) > 0:
            buy_trades = trade_data[trade_data['trade_side'] == 'buy']
            sell_trades = trade_data[trade_data['trade_side'] == 'sell']
            
            if len(buy_trades) > 0:
                ax8.scatter(buy_trades['time'], buy_trades['trade_price'], 
                           c='green', alpha=0.7, s=50, label='MM Buys')
            if len(sell_trades) > 0:
                ax8.scatter(sell_trades['time'], sell_trades['trade_price'], 
                           c='red', alpha=0.7, s=50, label='MM Sells')
            
            ax8.set_title('Trade Execution Prices')
            ax8.set_xlabel('Time (Years)')
            ax8.set_ylabel('Price ($)')
            ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. P&L components breakdown
        ax9 = plt.subplot(3, 3, 9)
        cash_pnl = pnl_df['cash'] - self.market_maker.starting_cash
        if len(self.price_path) == len(pnl_df):
            shares_pnl = pnl_df['underlying_shares'] * sim_df['underlying_price'][:len(pnl_df)]
            ax9.plot(pnl_df['time'], cash_pnl, label='Cash P&L', linewidth=2)
            ax9.plot(pnl_df['time'], shares_pnl, label='Shares P&L', linewidth=2)
            ax9.plot(pnl_df['time'], cash_pnl + shares_pnl, label='Total P&L', linewidth=2, linestyle='--')
            ax9.set_title('P&L Components')
            ax9.set_xlabel('Time (Years)')
            ax9.set_ylabel('P&L ($)')
            ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_performance_report(self) -> Dict:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Dictionary containing detailed performance metrics
        """
        if not self.simulation_log:
            return {}
        
        sim_df = pd.DataFrame(self.simulation_log)
        pnl_df = pd.DataFrame(self.pnl_components)
        
        # Basic statistics
        total_trades = len(self.trade_log)
        total_hedges = len(self.hedge_log)
        
        # P&L statistics
        pnl_series = pnl_df['total_pnl']
        max_pnl = pnl_series.max()
        min_pnl = pnl_series.min()
        final_pnl = pnl_series.iloc[-1]
        pnl_volatility = pnl_series.std()
        
        # Trade statistics
        if self.trade_log:
            trade_df = pd.DataFrame(self.trade_log)
            avg_trade_size = trade_df['quantity'].mean()
            total_volume = trade_df['quantity'].sum()
            
            # Calculate trading frequency
            simulation_time = self.time_grid[-1] - self.time_grid[0]
            trade_frequency = total_trades / simulation_time  # trades per year
        else:
            avg_trade_size = 0
            total_volume = 0
            trade_frequency = 0
            simulation_time = 0
        
        # Hedge statistics
        if self.hedge_log:
            hedge_df = pd.DataFrame(self.hedge_log)
            avg_hedge_size = hedge_df['shares_traded'].abs().mean()
            total_hedge_cost = hedge_df['hedge_cost'].abs().sum()
        else:
            avg_hedge_size = 0
            total_hedge_cost = 0
        
        # Market statistics
        underlying_return = (self.price_path[-1] / self.price_path[0]) - 1
        underlying_volatility = np.std(np.diff(np.log(self.price_path))) * np.sqrt(252)  # Annualized
        
        report = {
            'simulation_summary': {
                'total_simulation_time': simulation_time,
                'total_steps': len(self.simulation_log),
                'initial_portfolio_value': self.portfolio_values[0],
                'final_portfolio_value': self.portfolio_values[-1]
            },
            'pnl_metrics': {
                'final_pnl': final_pnl,
                'max_pnl': max_pnl,
                'min_pnl': min_pnl,
                'pnl_volatility': pnl_volatility,
                'max_drawdown': min_pnl if min_pnl < 0 else 0,
                'sharpe_ratio': final_pnl / pnl_volatility if pnl_volatility > 0 else 0
            },
            'trading_metrics': {
                'total_trades': total_trades,
                'total_volume': total_volume,
                'avg_trade_size': avg_trade_size,
                'trade_frequency_per_year': trade_frequency,
                'final_inventory': self.market_maker.inventory,
                'final_cash': self.market_maker.cash
            },
            'hedging_metrics': {
                'total_hedges': total_hedges,
                'avg_hedge_size': avg_hedge_size,
                'total_hedge_cost': total_hedge_cost,
                'final_underlying_shares': self.market_maker.underlying_shares,
                'hedge_frequency_per_year': total_hedges / simulation_time if simulation_time > 0 else 0
            },
            'market_metrics': {
                'initial_underlying_price': self.price_path[0],
                'final_underlying_price': self.price_path[-1],
                'underlying_return': underlying_return,
                'realized_underlying_volatility': underlying_volatility,
                'theoretical_volatility': self.sim_params['volatility']
            }
        }
        
        return report
    
    def print_performance_report(self) -> None:
        """
        Print a formatted performance report.
        """
        report = self.generate_performance_report()
        
        if not report:
            print("No simulation data available for report.")
            return
        
        print("\n" + "="*60)
        print("MARKET MAKING SIMULATION PERFORMANCE REPORT")
        print("="*60)
        
        print(f"\nSIMULATION SUMMARY:")
        print(f"  Simulation Time: {report['simulation_summary']['total_simulation_time']:.4f} years")
        print(f"  Total Steps: {report['simulation_summary']['total_steps']:,}")
        print(f"  Initial Portfolio Value: ${report['simulation_summary']['initial_portfolio_value']:,.2f}")
        print(f"  Final Portfolio Value: ${report['simulation_summary']['final_portfolio_value']:,.2f}")
        
        print(f"\nP&L METRICS:")
        print(f"  Final P&L: ${report['pnl_metrics']['final_pnl']:,.2f}")
        print(f"  Max P&L: ${report['pnl_metrics']['max_pnl']:,.2f}")
        print(f"  Min P&L: ${report['pnl_metrics']['min_pnl']:,.2f}")
        print(f"  P&L Volatility: ${report['pnl_metrics']['pnl_volatility']:.2f}")
        print(f"  Max Drawdown: ${report['pnl_metrics']['max_drawdown']:,.2f}")
        print(f"  Sharpe Ratio: {report['pnl_metrics']['sharpe_ratio']:.4f}")
        
        print(f"\nTRADING METRICS:")
        print(f"  Total Trades: {report['trading_metrics']['total_trades']}")
        print(f"  Total Volume: {report['trading_metrics']['total_volume']} contracts")
        print(f"  Average Trade Size: {report['trading_metrics']['avg_trade_size']:.2f} contracts")
        print(f"  Trade Frequency: {report['trading_metrics']['trade_frequency_per_year']:.2f} trades/year")
        print(f"  Final Inventory: {report['trading_metrics']['final_inventory']} contracts")
        print(f"  Final Cash: ${report['trading_metrics']['final_cash']:,.2f}")
        
        print(f"\nHEDGING METRICS:")
        print(f"  Total Hedges: {report['hedging_metrics']['total_hedges']}")
        print(f"  Average Hedge Size: {report['hedging_metrics']['avg_hedge_size']:.2f} shares")
        print(f"  Total Hedge Cost: ${report['hedging_metrics']['total_hedge_cost']:,.2f}")
        print(f"  Final Underlying Shares: {report['hedging_metrics']['final_underlying_shares']:.2f}")
        print(f"  Hedge Frequency: {report['hedging_metrics']['hedge_frequency_per_year']:.2f} hedges/year")
        
        print(f"\nMARKET METRICS:")
        print(f"  Initial Underlying: ${report['market_metrics']['initial_underlying_price']:.2f}")
        print(f"  Final Underlying: ${report['market_metrics']['final_underlying_price']:.2f}")
        print(f"  Underlying Return: {report['market_metrics']['underlying_return']*100:.2f}%")
        print(f"  Realized Volatility: {report['market_metrics']['realized_underlying_volatility']*100:.2f}%")
        print(f"  Theoretical Volatility: {report['market_metrics']['theoretical_volatility']*100:.2f}%")
        
        print("="*60)


def run_market_making_simulation():
    """
    Main function to run the complete market making simulation.
    """
    print("Initializing Market Making Simulation with Delta Hedging")
    print("="*60)
    
    # Simulation Parameters
    simulation_params = {
        'initial_price': 100.0,      # Initial stock price
        'strike_price': 105.0,       # Option strike price
        'drift': 0.05,              # 5% annual drift
        'volatility': 0.25,         # 25% annual volatility (base vol when using surface)
        'time_horizon': 0.25,       # 3 months simulation
        'time_step': 1/252,         # Daily time steps
        'option_maturity': 0.25,    # Option expires at end of simulation
        'option_type': 'call',      # Call option
        'order_probability': 0.08,  # 8% chance of order each step
        'seed': 42,                 # Random seed for reproducibility
        # Volatility surface parameters
        'use_vol_surface': True,    # Enable volatility surface
        'base_vol': 0.25,          # Base volatility (center of smile)
        'smile_curvature': 0.1     # Smile curvature parameter
    }
    
    # Market Maker Parameters
    market_maker_params = {
        'starting_cash': 50000.0,   # Starting cash
        'gamma': 0.1,               # Risk aversion parameter
        'risk_free_rate': 0.05,     # 5% risk-free rate
        'base_spread': 0.30,        # $0.30 base spread
        'min_spread': 0.05,         # Minimum $0.05 spread
        'max_spread': 1.50          # Maximum $1.50 spread
    }
    
    print("Simulation Parameters:")
    for key, value in simulation_params.items():
        if isinstance(value, float):
            if 'price' in key or 'cash' in key:
                print(f"  {key.replace('_', ' ').title()}: ${value:.2f}")
            elif 'probability' in key or 'volatility' in key or 'drift' in key:
                print(f"  {key.replace('_', ' ').title()}: {value*100:.1f}%")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print(f"\nMarket Maker Parameters:")
    for key, value in market_maker_params.items():
        if isinstance(value, float):
            if 'cash' in key or 'spread' in key:
                print(f"  {key.replace('_', ' ').title()}: ${value:.2f}")
            elif 'rate' in key or 'gamma' in key:
                print(f"  {key.replace('_', ' ').title()}: {value}")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print()
    
    # Initialize and run simulation
    sim = MarketMakingSimulation(simulation_params, market_maker_params)
    
    # Run the simulation
    results = sim.run_simulation()
    
    # Print performance report
    sim.print_performance_report()
    
    # Plot results
    print("\nGenerating visualization plots...")
    sim.plot_simulation_results()
    
    return sim, results


# if __name__ == "__main__":
#     # Run the complete simulation
#     simulation, results = run_market_making_simulation()
    
#     print(f"\nSimulation completed successfully!")
#     print(f"Access simulation object for detailed analysis: simulation")
#     print(f"Access results dictionary: results")
    
#     # Example of accessing detailed data
#     print(f"\nExample analysis:")
#     print(f"Number of simulation steps: {len(simulation.simulation_log)}")
#     print(f"Number of trades executed: {len(simulation.trade_log)}")
#     print(f"Number of hedges performed: {len(simulation.hedge_log)}")
    
#     if simulation.trade_log:
#         trade_df = pd.DataFrame(simulation.trade_log)
#         print(f"Average trade size: {trade_df['quantity'].mean():.2f} contracts")
#         print(f"Total volume traded: {trade_df['quantity'].sum()} contracts")
    
#     if simulation.hedge_log:
#         hedge_df = pd.DataFrame(simulation.hedge_log)
#         print(f"Average hedge size: {hedge_df['shares_traded'].abs().mean():.2f} shares")
#         print(f"Total hedge transactions: {len(hedge_df)}")
