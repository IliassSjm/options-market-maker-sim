"""
Market Maker Agent for Options Trading
=====================================

A comprehensive implementation of a market making agent that provides bid/ask quotes
for options while managing inventory risk using the Black-Scholes model.

The market maker adjusts quotes based on:
- Theoretical fair value (Black-Scholes)
- Inventory risk (using risk aversion parameter)
- Market conditions and volatility

Author: Market Maker Module
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import our Black-Scholes option pricer
try:
    from option_pricer import BlackScholesOptionPricer
except ImportError:
    print("Warning: option_pricer.py not found. Please ensure it's in the same directory.")
    # Fallback simple Black-Scholes implementation
    from scipy.stats import norm
    
    class BlackScholesOptionPricer:
        @staticmethod
        def _d1(S, K, T, r, sigma):
            return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        @staticmethod
        def _d2(S, K, T, r, sigma):
            d1 = BlackScholesOptionPricer._d1(S, K, T, r, sigma)
            return d1 - sigma * np.sqrt(T)
        
        def call_price(self, S, K, T, r, sigma):
            if T <= 0:
                return max(S - K, 0)
            d1 = self._d1(S, K, T, r, sigma)
            d2 = self._d2(S, K, T, r, sigma)
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        
        def put_price(self, S, K, T, r, sigma):
            if T <= 0:
                return max(K - S, 0)
            d1 = self._d1(S, K, T, r, sigma)
            d2 = self._d2(S, K, T, r, sigma)
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        def call_delta(self, S, K, T, r, sigma):
            if T <= 0:
                return 1.0 if S > K else 0.0
            d1 = self._d1(S, K, T, r, sigma)
            return norm.cdf(d1)
        
        def put_delta(self, S, K, T, r, sigma):
            if T <= 0:
                return -1.0 if S < K else 0.0
            d1 = self._d1(S, K, T, r, sigma)
            return norm.cdf(d1) - 1


class MarketMaker:
    """
    Market Maker agent that provides bid/ask quotes for options while managing inventory risk.
    
    The agent uses the Black-Scholes model to calculate fair values and adjusts quotes
    based on current inventory position and risk aversion parameters.
    """
    
    def __init__(self, starting_cash: float, gamma: float, risk_free_rate: float = 0.05,
                 base_spread: float = 0.50, min_spread: float = 0.10, max_spread: float = 2.0):
        """
        Initialize the Market Maker agent.
        
        Args:
            starting_cash: Initial cash position
            gamma: Risk aversion parameter (higher = more risk averse)
            risk_free_rate: Risk-free rate for Black-Scholes calculations
            base_spread: Base spread around reservation price
            min_spread: Minimum allowed spread
            max_spread: Maximum allowed spread
        """
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.inventory = 0  # Number of options held (positive = long, negative = short)
        self.underlying_shares = 0  # Number of underlying shares for delta hedging
        self.gamma = gamma  # Risk aversion parameter
        self.risk_free_rate = risk_free_rate
        self.base_spread = base_spread
        self.min_spread = min_spread
        self.max_spread = max_spread
        
        # Initialize Black-Scholes pricer
        self.bs_pricer = BlackScholesOptionPricer()
        
        # Track trading history
        self.trade_history: List[Dict] = []
        self.quote_history: List[Dict] = []
        
        # Performance metrics
        self.total_trades = 0
        self.total_volume = 0
        self.pnl_history: List[float] = []
    
    def calculate_fair_value(self, underlying_price: float, strike_price: float, 
                           time_to_maturity: float, volatility: float, 
                           option_type: str = 'call') -> float:
        """
        Calculate the theoretical fair value of an option using Black-Scholes.
        
        Args:
            underlying_price: Current price of the underlying asset
            strike_price: Strike price of the option
            time_to_maturity: Time to maturity in years
            volatility: Implied volatility
            option_type: 'call' or 'put'
            
        Returns:
            Theoretical fair value of the option
        """
        if option_type.lower() == 'call':
            return self.bs_pricer.call_price(underlying_price, strike_price, 
                                           time_to_maturity, self.risk_free_rate, volatility)
        elif option_type.lower() == 'put':
            return self.bs_pricer.put_price(underlying_price, strike_price, 
                                          time_to_maturity, self.risk_free_rate, volatility)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
    
    def calculate_delta(self, underlying_price: float, strike_price: float,
                       time_to_maturity: float, volatility: float, 
                       option_type: str = 'call') -> float:
        """
        Calculate the delta of an option.
        
        Args:
            underlying_price: Current price of the underlying asset
            strike_price: Strike price of the option
            time_to_maturity: Time to maturity in years
            volatility: Implied volatility
            option_type: 'call' or 'put'
            
        Returns:
            Option delta
        """
        if option_type.lower() == 'call':
            return self.bs_pricer.call_delta(underlying_price, strike_price,
                                           time_to_maturity, self.risk_free_rate, volatility)
        elif option_type.lower() == 'put':
            return self.bs_pricer.put_delta(underlying_price, strike_price,
                                          time_to_maturity, self.risk_free_rate, volatility)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
    
    def calculate_reservation_price(self, fair_value: float, volatility: float) -> float:
        """
        Calculate the reservation price adjusted for inventory risk.
        
        The reservation price reflects the agent's private value for the option,
        considering current inventory position and risk aversion.
        
        Formula: reservation_price = fair_value - inventory * gamma * volatility²
        
        Args:
            fair_value: Theoretical fair value from Black-Scholes
            volatility: Current volatility
            
        Returns:
            Reservation price adjusted for inventory risk
        """
        inventory_adjustment = self.inventory * self.gamma * (volatility ** 2)
        return fair_value - inventory_adjustment
    
    def calculate_dynamic_spread(self, volatility: float, time_to_maturity: float, 
                               underlying_price: float) -> float:
        """
        Calculate a dynamic spread based on market conditions.
        
        The spread widens with:
        - Higher volatility
        - Shorter time to maturity
        - Larger inventory positions
        
        Args:
            volatility: Current volatility
            time_to_maturity: Time to maturity in years
            underlying_price: Current underlying price
            
        Returns:
            Dynamic spread value
        """
        # Base spread
        spread = self.base_spread
        
        # Volatility adjustment (higher vol = wider spread)
        vol_adjustment = volatility * 0.5
        spread += vol_adjustment
        
        # Time decay adjustment (closer to expiry = wider spread)
        if time_to_maturity > 0:
            time_adjustment = max(0, (0.1 - time_to_maturity) * 2)
            spread += time_adjustment
        
        # Inventory risk adjustment (larger positions = wider spread)
        inventory_adjustment = abs(self.inventory) * 0.1
        spread += inventory_adjustment
        
        # Ensure spread is within bounds
        spread = max(self.min_spread, min(self.max_spread, spread))
        
        return spread
    
    def calculate_quotes(self, underlying_price: float, strike_price: float,
                        time_to_maturity: float, volatility: float, 
                        option_type: str = 'call', use_dynamic_spread: bool = True) -> Tuple[float, float]:
        """
        Calculate bid and ask quotes for an option.
        
        This is the core method that determines the market maker's quotes based on:
        1. Theoretical fair value (Black-Scholes)
        2. Inventory risk adjustment
        3. Dynamic or fixed spread
        
        Args:
            underlying_price: Current price of the underlying asset
            strike_price: Strike price of the option
            time_to_maturity: Time to maturity in years
            volatility: Implied volatility
            option_type: 'call' or 'put'
            use_dynamic_spread: Whether to use dynamic spread calculation
            
        Returns:
            Tuple of (bid_price, ask_price)
        """
        # Calculate fair value
        fair_value = self.calculate_fair_value(underlying_price, strike_price, 
                                             time_to_maturity, volatility, option_type)
        
        # Calculate reservation price
        reservation_price = self.calculate_reservation_price(fair_value, volatility)
        
        # Calculate spread
        if use_dynamic_spread:
            spread = self.calculate_dynamic_spread(volatility, time_to_maturity, underlying_price)
        else:
            spread = self.base_spread
        
        # Calculate bid and ask
        bid_price = reservation_price - spread / 2
        ask_price = reservation_price + spread / 2
        
        # Ensure quotes are non-negative
        bid_price = max(0.01, bid_price)
        ask_price = max(bid_price + 0.01, ask_price)
        
        # Store quote information
        quote_info = {
            'timestamp': datetime.now(),
            'underlying_price': underlying_price,
            'strike_price': strike_price,
            'time_to_maturity': time_to_maturity,
            'volatility': volatility,
            'option_type': option_type,
            'fair_value': fair_value,
            'reservation_price': reservation_price,
            'spread': spread,
            'bid_price': bid_price,
            'ask_price': ask_price,
            'inventory': self.inventory,
            'cash': self.cash
        }
        self.quote_history.append(quote_info)
        
        return bid_price, ask_price
    
    def execute_trade(self, quantity: int, price: float, side: str, 
                     underlying_price: float, strike_price: float,
                     time_to_maturity: float, volatility: float,
                     option_type: str = 'call') -> Dict:
        """
        Execute a trade and update inventory and cash positions.
        
        Args:
            quantity: Number of options traded (positive integer)
            price: Trade execution price
            side: 'buy' or 'sell' from market maker's perspective
            underlying_price: Current underlying price
            strike_price: Strike price of the option
            time_to_maturity: Time to maturity
            volatility: Current volatility
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary containing trade information
        """
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if side.lower() not in ['buy', 'sell']:
            raise ValueError("Side must be 'buy' or 'sell'")
        
        # Update positions
        if side.lower() == 'buy':
            # Market maker buys options (long position)
            self.inventory += quantity
            self.cash -= quantity * price
        else:
            # Market maker sells options (short position)
            self.inventory -= quantity
            self.cash += quantity * price
        
        # Calculate current fair value for P&L tracking
        current_fair_value = self.calculate_fair_value(underlying_price, strike_price,
                                                     time_to_maturity, volatility, option_type)
        
        # Calculate unrealized P&L
        unrealized_pnl = self.inventory * current_fair_value
        total_pnl = self.cash - self.starting_cash + unrealized_pnl
        self.pnl_history.append(total_pnl)
        
        # Record trade
        trade_info = {
            'timestamp': datetime.now(),
            'quantity': quantity,
            'price': price,
            'side': side,
            'underlying_price': underlying_price,
            'strike_price': strike_price,
            'time_to_maturity': time_to_maturity,
            'volatility': volatility,
            'option_type': option_type,
            'inventory_after': self.inventory,
            'cash_after': self.cash,
            'current_fair_value': current_fair_value,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': total_pnl
        }
        
        self.trade_history.append(trade_info)
        self.total_trades += 1
        self.total_volume += quantity
        
        return trade_info
    
    def delta_hedge(self, underlying_price: float, strike_price: float,
                   time_to_maturity: float, volatility: float, 
                   option_type: str = 'call') -> Dict:
        """
        Perform delta hedging by adjusting underlying shares position.
        
        Args:
            underlying_price: Current underlying price
            strike_price: Strike price of the option
            time_to_maturity: Time to maturity in years
            volatility: Current volatility
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary containing hedge information
        """
        if self.inventory == 0:
            # No options to hedge, but may need to close existing hedge
            if self.underlying_shares != 0:
                hedge_cost = self.underlying_shares * underlying_price
                self.cash += hedge_cost
                shares_traded = -self.underlying_shares  # Negative means we're selling
                self.underlying_shares = 0
                
                return {
                    'shares_traded': shares_traded,
                    'hedge_cost': hedge_cost,
                    'new_shares_position': self.underlying_shares,
                    'portfolio_delta': 0,
                    'hedge_type': 'close_hedge'
                }
            else:
                return {
                    'shares_traded': 0,
                    'hedge_cost': 0,
                    'new_shares_position': 0,
                    'portfolio_delta': 0,
                    'hedge_type': 'no_hedge_needed'
                }
        
        # Calculate option delta
        option_delta = self.calculate_delta(underlying_price, strike_price,
                                          time_to_maturity, volatility, option_type)
        
        # Calculate portfolio delta (from options position)
        portfolio_delta = self.inventory * option_delta
        
        # Target hedge: equal and opposite position in underlying
        # Current net delta exposure
        current_net_delta = portfolio_delta + self.underlying_shares
        
        # We want net delta to be zero, so we need to trade:
        shares_to_trade = -current_net_delta
        
        # Execute the hedge
        hedge_cost = shares_to_trade * underlying_price
        self.cash -= hedge_cost
        self.underlying_shares += shares_to_trade
        
        hedge_info = {
            'shares_traded': shares_to_trade,
            'hedge_cost': hedge_cost,
            'new_shares_position': self.underlying_shares,
            'option_delta': option_delta,
            'portfolio_delta': portfolio_delta,
            'net_delta_before': current_net_delta,
            'net_delta_after': portfolio_delta + self.underlying_shares,
            'hedge_type': 'delta_hedge'
        }
        
        return hedge_info
    
    def get_portfolio_value(self, underlying_price: float, strike_price: float,
                           time_to_maturity: float, volatility: float,
                           option_type: str = 'call') -> float:
        """
        Calculate current portfolio value (cash + inventory value + underlying shares value).
        
        Returns:
            Current total portfolio value
        """
        # Cash position
        total_value = self.cash
        
        # Options inventory value
        if self.inventory != 0:
            current_fair_value = self.calculate_fair_value(underlying_price, strike_price,
                                                         time_to_maturity, volatility, option_type)
            total_value += self.inventory * current_fair_value
        
        # Underlying shares value
        total_value += self.underlying_shares * underlying_price
        
        return total_value
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate and return performance metrics.
        
        Returns:
            Dictionary containing performance statistics
        """
        if not self.pnl_history:
            return {'total_pnl': 0, 'max_pnl': 0, 'min_pnl': 0, 'pnl_volatility': 0}
        
        return {
            'total_trades': self.total_trades,
            'total_volume': self.total_volume,
            'current_inventory': self.inventory,
            'current_cash': self.cash,
            'current_underlying_shares': self.underlying_shares,
            'total_pnl': self.pnl_history[-1] if self.pnl_history else 0,
            'max_pnl': max(self.pnl_history) if self.pnl_history else 0,
            'min_pnl': min(self.pnl_history) if self.pnl_history else 0,
            'pnl_volatility': np.std(self.pnl_history) if len(self.pnl_history) > 1 else 0,
            'avg_trade_size': self.total_volume / max(1, self.total_trades)
        }
    
    def plot_performance(self) -> None:
        """
        Plot performance metrics and trading history.
        """
        if not self.pnl_history:
            print("No trading history to plot.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # P&L over time
        ax1.plot(self.pnl_history, linewidth=2, color='blue')
        ax1.set_title('P&L Over Time')
        ax1.set_xlabel('Trade Number')
        ax1.set_ylabel('Total P&L ($)')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Inventory over time
        inventory_history = [trade['inventory_after'] for trade in self.trade_history]
        ax2.plot(inventory_history, linewidth=2, color='green')
        ax2.set_title('Inventory Over Time')
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Inventory (contracts)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Cash over time
        cash_history = [trade['cash_after'] for trade in self.trade_history]
        ax3.plot(cash_history, linewidth=2, color='orange')
        ax3.set_title('Cash Position Over Time')
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('Cash ($)')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=self.starting_cash, color='red', linestyle='--', alpha=0.5, label='Starting Cash')
        ax3.legend()
        
        # Trade prices
        trade_prices = [trade['price'] for trade in self.trade_history]
        buy_prices = [trade['price'] for trade in self.trade_history if trade['side'] == 'buy']
        sell_prices = [trade['price'] for trade in self.trade_history if trade['side'] == 'sell']
        
        ax4.scatter(range(len(buy_prices)), buy_prices, color='green', alpha=0.6, label='Buys', s=30)
        ax4.scatter(range(len(sell_prices)), sell_prices, color='red', alpha=0.6, label='Sells', s=30)
        ax4.set_title('Trade Execution Prices')
        ax4.set_xlabel('Trade Number')
        ax4.set_ylabel('Price ($)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
    
    def reset(self) -> None:
        """
        Reset the market maker to initial state.
        """
        self.cash = self.starting_cash
        self.inventory = 0
        self.underlying_shares = 0
        self.trade_history.clear()
        self.quote_history.clear()
        self.pnl_history.clear()
        self.total_trades = 0
        self.total_volume = 0


# Utility functions for market simulation
def simulate_market_making_session(mm: MarketMaker, underlying_price: float, 
                                  strike_price: float, time_to_maturity: float,
                                  volatility: float, option_type: str = 'call',
                                  n_trades: int = 100, price_noise: float = 0.02) -> None:
    """
    Simulate a market making session with random trades.
    
    Args:
        mm: MarketMaker instance
        underlying_price: Initial underlying price
        strike_price: Option strike price
        time_to_maturity: Initial time to maturity
        volatility: Volatility
        option_type: 'call' or 'put'
        n_trades: Number of trades to simulate
        price_noise: Random noise in trade execution
    """
    print(f"Starting market making simulation with {n_trades} trades...")
    print(f"Initial portfolio value: ${mm.get_portfolio_value(underlying_price, strike_price, time_to_maturity, volatility, option_type):.2f}")
    
    for i in range(n_trades):
        # Generate quotes
        bid, ask = mm.calculate_quotes(underlying_price, strike_price, time_to_maturity, 
                                     volatility, option_type)
        
        # Simulate random trade (50% chance of buy vs sell)
        side = np.random.choice(['buy', 'sell'])
        quantity = np.random.randint(1, 6)  # 1-5 contracts
        
        # Execute at bid (if selling to MM) or ask (if buying from MM) with some noise
        if side == 'sell':  # Client sells to market maker (MM buys)
            execution_price = bid * (1 + np.random.normal(0, price_noise))
        else:  # Client buys from market maker (MM sells)
            execution_price = ask * (1 + np.random.normal(0, price_noise))
        
        execution_price = max(0.01, execution_price)  # Ensure positive price
        
        # Execute trade
        try:
            mm.execute_trade(quantity, execution_price, side, underlying_price, 
                           strike_price, time_to_maturity, volatility, option_type)
        except Exception as e:
            print(f"Trade {i+1} failed: {e}")
            continue
        
        # Randomly update market conditions
        if i % 10 == 0:
            underlying_price *= (1 + np.random.normal(0, 0.01))  # 1% price moves
            time_to_maturity *= 0.99  # Time decay
            volatility *= (1 + np.random.normal(0, 0.05))  # Volatility changes
            volatility = max(0.05, min(1.0, volatility))  # Keep vol reasonable
    
    final_value = mm.get_portfolio_value(underlying_price, strike_price, time_to_maturity, 
                                       volatility, option_type)
    print(f"Final portfolio value: ${final_value:.2f}")
    print(f"Total P&L: ${final_value - mm.starting_cash:.2f}")


# Example usage and demonstration
if __name__ == "__main__":
    print("Market Maker Agent Demonstration")
    print("=" * 40)
    
    # Initialize market maker
    starting_cash = 10000.0
    gamma = 0.1  # Risk aversion parameter
    mm = MarketMaker(starting_cash=starting_cash, gamma=gamma)
    
    # Market parameters
    underlying_price = 100.0
    strike_price = 105.0
    time_to_maturity = 0.25  # 3 months
    volatility = 0.25  # 25% volatility
    option_type = 'call'
    
    print(f"Market Maker initialized with:")
    print(f"  Starting cash: ${starting_cash}")
    print(f"  Risk aversion (γ): {gamma}")
    print(f"  Initial inventory: {mm.inventory}")
    print()
    
    # Calculate initial quotes
    print("Initial market conditions:")
    print(f"  Underlying price: ${underlying_price}")
    print(f"  Strike price: ${strike_price}")
    print(f"  Time to maturity: {time_to_maturity} years")
    print(f"  Volatility: {volatility*100}%")
    print()
    
    # Get initial quotes
    bid, ask = mm.calculate_quotes(underlying_price, strike_price, time_to_maturity, 
                                 volatility, option_type)
    fair_value = mm.calculate_fair_value(underlying_price, strike_price, time_to_maturity, 
                                       volatility, option_type)
    
    print("Initial quotes:")
    print(f"  Fair value: ${fair_value:.4f}")
    print(f"  Bid: ${bid:.4f}")
    print(f"  Ask: ${ask:.4f}")
    print(f"  Spread: ${ask - bid:.4f}")
    print()
    
    # Execute some sample trades
    print("Executing sample trades...")
    
    # Trade 1: Sell 2 contracts at ask
    trade1 = mm.execute_trade(2, ask, 'sell', underlying_price, strike_price, 
                            time_to_maturity, volatility, option_type)
    print(f"Trade 1: Sold 2 contracts at ${ask:.4f}")
    print(f"  New inventory: {mm.inventory}")
    print(f"  New cash: ${mm.cash:.2f}")
    
    # Get new quotes (should reflect inventory change)
    bid2, ask2 = mm.calculate_quotes(underlying_price, strike_price, time_to_maturity, 
                                   volatility, option_type)
    print(f"  New quotes - Bid: ${bid2:.4f}, Ask: ${ask2:.4f}")
    print()
    
    # Trade 2: Buy 3 contracts at bid
    trade2 = mm.execute_trade(3, bid2, 'buy', underlying_price, strike_price, 
                            time_to_maturity, volatility, option_type)
    print(f"Trade 2: Bought 3 contracts at ${bid2:.4f}")
    print(f"  New inventory: {mm.inventory}")
    print(f"  New cash: ${mm.cash:.2f}")
    
    # Get performance metrics
    metrics = mm.get_performance_metrics()
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print("\nRunning full market making simulation...")
    simulate_market_making_session(mm, underlying_price, strike_price, time_to_maturity, 
                                  volatility, option_type, n_trades=50)
    
    # Plot results
    mm.plot_performance()
    
    # Final performance
    final_metrics = mm.get_performance_metrics()
    print("\nFinal Performance Summary:")
    print("=" * 30)
    for key, value in final_metrics.items():
        if isinstance(value, float):
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")
