"""
Market Simulator using Geometric Brownian Motion
================================================

A comprehensive implementation of stock price simulation using Geometric Brownian Motion (GBM).
Includes visualization tools and statistical analysis functions.

The GBM model follows the stochastic differential equation:
dS = μ * S * dt + σ * S * dW

Where:
- S: Stock price
- μ: Drift rate (expected return)
- σ: Volatility
- dW: Wiener process (random walk)

Author: Market Simulation Module
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional
import pandas as pd


class GeometricBrownianMotion:
    """
    Geometric Brownian Motion simulator for stock price paths.
    
    This class provides methods to simulate stock price paths using GBM
    and includes various analysis and visualization tools.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the GBM simulator.
        
        Args:
            seed: Random seed for reproducible results
        """
        if seed is not None:
            np.random.seed(seed)
    
    def simulate_price_path(self, S0: float, mu: float, sigma: float, 
                           T: float, dt: float) -> np.ndarray:
        """
        Simulate a single stock price path using Geometric Brownian Motion.
        
        The GBM formula used is:
        S(t+dt) = S(t) * exp((μ - 0.5*σ²)*dt + σ*sqrt(dt)*Z)
        
        Where Z ~ N(0,1) is a standard normal random variable.
        
        Args:
            S0: Initial stock price
            mu: Drift rate (expected annual return)
            sigma: Volatility (annual standard deviation)
            T: Total time horizon in years
            dt: Time step size in years
            
        Returns:
            NumPy array containing simulated prices at each time step
        """
        # Calculate number of time steps
        n_steps = int(T / dt)
        
        # Initialize price array
        prices = np.zeros(n_steps + 1)
        prices[0] = S0
        
        # Generate random shocks
        random_shocks = np.random.standard_normal(n_steps)
        
        # Calculate price evolution
        for i in range(n_steps):
            drift = (mu - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * random_shocks[i]
            prices[i + 1] = prices[i] * np.exp(drift + diffusion)
        
        return prices
    
    def simulate_multiple_paths(self, S0: float, mu: float, sigma: float, 
                               T: float, dt: float, n_paths: int) -> np.ndarray:
        """
        Simulate multiple stock price paths.
        
        Args:
            S0: Initial stock price
            mu: Drift rate
            sigma: Volatility
            T: Total time horizon in years
            dt: Time step size in years
            n_paths: Number of paths to simulate
            
        Returns:
            2D NumPy array where each column is a price path
        """
        n_steps = int(T / dt) + 1
        paths = np.zeros((n_steps, n_paths))
        
        for i in range(n_paths):
            paths[:, i] = self.simulate_price_path(S0, mu, sigma, T, dt)
        
        return paths
    
    def get_time_grid(self, T: float, dt: float) -> np.ndarray:
        """
        Generate time grid for the simulation.
        
        Args:
            T: Total time horizon in years
            dt: Time step size in years
            
        Returns:
            NumPy array of time points
        """
        n_steps = int(T / dt)
        return np.linspace(0, T, n_steps + 1)
    
    def calculate_returns(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate log returns from price series.
        
        Args:
            prices: Array of prices
            
        Returns:
            Array of log returns
        """
        return np.diff(np.log(prices))
    
    def calculate_statistics(self, prices: np.ndarray, dt: float) -> dict:
        """
        Calculate statistical properties of the simulated price path.
        
        Args:
            prices: Array of simulated prices
            dt: Time step size
            
        Returns:
            Dictionary containing statistical measures
        """
        returns = self.calculate_returns(prices)
        
        # Annualize statistics
        annual_factor = 1 / dt
        
        stats = {
            'final_price': prices[-1],
            'total_return': (prices[-1] / prices[0]) - 1,
            'mean_return': np.mean(returns) * annual_factor,
            'volatility': np.std(returns) * np.sqrt(annual_factor),
            'max_price': np.max(prices),
            'min_price': np.min(prices),
            'max_drawdown': self._calculate_max_drawdown(prices)
        }
        
        return stats
    
    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """
        Calculate maximum drawdown from peak.
        
        Args:
            prices: Array of prices
            
        Returns:
            Maximum drawdown as a percentage
        """
        cumulative_max = np.maximum.accumulate(prices)
        drawdown = (prices - cumulative_max) / cumulative_max
        return np.min(drawdown)


def simulate_stock_price(S0: float, mu: float, sigma: float, T: float, dt: float) -> np.ndarray:
    """
    Simple wrapper function to simulate a stock price path using GBM.
    
    Args:
        S0: Initial stock price
        mu: Drift rate (expected annual return)
        sigma: Volatility (annual standard deviation)
        T: Total time horizon in years
        dt: Time step size in years
        
    Returns:
        NumPy array containing simulated prices at each time step
    """
    simulator = GeometricBrownianMotion()
    return simulator.simulate_price_path(S0, mu, sigma, T, dt)


def plot_price_path(prices: np.ndarray, T: float, dt: float, 
                   title: str = "Simulated Stock Price Path") -> None:
    """
    Plot a single stock price path.
    
    Args:
        prices: Array of simulated prices
        T: Total time horizon in years
        dt: Time step size in years
        title: Plot title
    """
    simulator = GeometricBrownianMotion()
    time_grid = simulator.get_time_grid(T, dt)
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_grid, prices, linewidth=2, color='blue')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Time (Years)', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_multiple_paths(paths: np.ndarray, T: float, dt: float, 
                       title: str = "Multiple Stock Price Paths") -> None:
    """
    Plot multiple stock price paths.
    
    Args:
        paths: 2D array where each column is a price path
        T: Total time horizon in years
        dt: Time step size in years
        title: Plot title
    """
    simulator = GeometricBrownianMotion()
    time_grid = simulator.get_time_grid(T, dt)
    
    plt.figure(figsize=(12, 8))
    
    # Plot individual paths with transparency
    for i in range(paths.shape[1]):
        plt.plot(time_grid, paths[:, i], alpha=0.3, color='blue', linewidth=0.8)
    
    # Plot mean path
    mean_path = np.mean(paths, axis=1)
    plt.plot(time_grid, mean_path, color='red', linewidth=3, label='Mean Path')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Time (Years)', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def analyze_simulation_results(paths: np.ndarray, dt: float, 
                              S0: float, mu: float, sigma: float) -> None:
    """
    Analyze and display simulation results with statistics.
    
    Args:
        paths: 2D array of simulated paths
        dt: Time step size
        S0: Initial price
        mu: Theoretical drift
        sigma: Theoretical volatility
    """
    simulator = GeometricBrownianMotion()
    
    # Calculate statistics for each path
    final_prices = paths[-1, :]
    all_returns = []
    
    for i in range(paths.shape[1]):
        returns = simulator.calculate_returns(paths[:, i])
        all_returns.extend(returns)
    
    all_returns = np.array(all_returns)
    
    # Calculate empirical statistics
    annual_factor = 1 / dt
    empirical_mu = np.mean(all_returns) * annual_factor
    empirical_sigma = np.std(all_returns) * np.sqrt(annual_factor)
    
    print("Simulation Analysis Results")
    print("=" * 40)
    print(f"Number of paths: {paths.shape[1]}")
    print(f"Initial price: ${S0:.2f}")
    print(f"Theoretical drift (μ): {mu:.4f} ({mu*100:.2f}%)")
    print(f"Theoretical volatility (σ): {sigma:.4f} ({sigma*100:.2f}%)")
    print()
    print("Empirical Results:")
    print(f"Empirical drift: {empirical_mu:.4f} ({empirical_mu*100:.2f}%)")
    print(f"Empirical volatility: {empirical_sigma:.4f} ({empirical_sigma*100:.2f}%)")
    print()
    print("Final Price Statistics:")
    print(f"Mean final price: ${np.mean(final_prices):.2f}")
    print(f"Median final price: ${np.median(final_prices):.2f}")
    print(f"Std final price: ${np.std(final_prices):.2f}")
    print(f"Min final price: ${np.min(final_prices):.2f}")
    print(f"Max final price: ${np.max(final_prices):.2f}")
    
    # Plot distribution of final prices
    plt.figure(figsize=(10, 6))
    plt.hist(final_prices, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(final_prices), color='red', linestyle='--', linewidth=2, label='Mean')
    plt.axvline(np.median(final_prices), color='green', linestyle='--', linewidth=2, label='Median')
    plt.xlabel('Final Stock Price ($)')
    plt.ylabel('Density')
    plt.title('Distribution of Final Stock Prices')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Example usage and demonstration
if __name__ == "__main__":
    # Set parameters
    S0 = 100.0      # Initial price
    mu = 0.08       # 8% annual drift
    sigma = 0.25    # 25% annual volatility
    T = 1.0         # 1 year
    dt = 1/252      # Daily time steps (252 trading days per year)
    
    print("Geometric Brownian Motion Stock Price Simulator")
    print("=" * 50)
    print(f"Initial Price: ${S0}")
    print(f"Annual Drift: {mu*100}%")
    print(f"Annual Volatility: {sigma*100}%")
    print(f"Time Horizon: {T} year(s)")
    print(f"Time Step: {dt:.4f} years ({1/dt:.0f} steps per year)")
    print()
    
    # Create simulator instance
    simulator = GeometricBrownianMotion(seed=42)
    
    # Simulate single path
    print("1. Simulating single price path...")
    single_path = simulator.simulate_price_path(S0, mu, sigma, T, dt)
    plot_price_path(single_path, T, dt, "Single GBM Price Path")
    
    # Calculate and display statistics for single path
    stats = simulator.calculate_statistics(single_path, dt)
    print("Single Path Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            if 'return' in key or 'drawdown' in key:
                print(f"  {key.replace('_', ' ').title()}: {value:.4f} ({value*100:.2f}%)")
            else:
                print(f"  {key.replace('_', ' ').title()}: ${value:.2f}")
    print()
    
    # Simulate multiple paths
    print("2. Simulating multiple price paths...")
    n_paths = 1000
    multiple_paths = simulator.simulate_multiple_paths(S0, mu, sigma, T, dt, n_paths)
    plot_multiple_paths(multiple_paths, T, dt, f"{n_paths} GBM Price Paths")
    
    # Analyze results
    print("3. Analyzing simulation results...")
    analyze_simulation_results(multiple_paths, dt, S0, mu, sigma)
    
    # Demonstrate the simple wrapper function
    print("4. Using simple wrapper function...")
    simple_path = simulate_stock_price(S0, mu, sigma, T, dt)
    plot_price_path(simple_path, T, dt, "Price Path using Wrapper Function")
    
    print("\nSimulation complete!")
