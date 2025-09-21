"""
Black-Scholes Option Pricer
===========================

A comprehensive implementation of the Black-Scholes model for European options.
Calculates option prices and the primary Greeks (Delta, Gamma, Vega, Theta).

Author: Option Pricing Module
Date: 2024
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Union


class BlackScholesOptionPricer:
    """
    Black-Scholes option pricing calculator for European options.
    
    This class provides methods to calculate option prices and Greeks
    for both call and put options using the Black-Scholes formula.
    """
    
    def __init__(self):
        """Initialize the option pricer."""
        pass
    
    @staticmethod
    def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate d1 parameter for Black-Scholes formula.
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            d1 value
        """
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate d2 parameter for Black-Scholes formula.
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            d2 value
        """
        d1 = BlackScholesOptionPricer._d1(S, K, T, r, sigma)
        return d1 - sigma * np.sqrt(T)
    
    def call_price(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate European call option price using Black-Scholes formula.
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate (annualized)
            sigma: Volatility (annualized)
            
        Returns:
            Call option price
        """
        if T <= 0:
            return max(S - K, 0)
        
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    def put_price(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate European put option price using Black-Scholes formula.
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate (annualized)
            sigma: Volatility (annualized)
            
        Returns:
            Put option price
        """
        if T <= 0:
            return max(K - S, 0)
        
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price
    
    def call_delta(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Delta for call option (rate of change of option price with respect to underlying).
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Call delta
        """
        if T <= 0:
            return 1.0 if S > K else 0.0
        
        d1 = self._d1(S, K, T, r, sigma)
        return norm.cdf(d1)
    
    def put_delta(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Delta for put option.
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Put delta
        """
        if T <= 0:
            return -1.0 if S < K else 0.0
        
        d1 = self._d1(S, K, T, r, sigma)
        return norm.cdf(d1) - 1
    
    def gamma(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Gamma (rate of change of delta with respect to underlying).
        Gamma is the same for both call and put options.
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Gamma
        """
        if T <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, sigma)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma
    
    def vega(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Vega (rate of change of option price with respect to volatility).
        Vega is the same for both call and put options.
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Vega (expressed per 1% change in volatility)
        """
        if T <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, sigma)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Divided by 100 for 1% change
        return vega
    
    def call_theta(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Theta for call option (rate of change of option price with respect to time).
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Call theta (per day)
        """
        if T <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        return theta
    
    def put_theta(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Theta for put option.
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Put theta (per day)
        """
        if T <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        return theta
    
    def rho_call(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Rho for call option (rate of change of option price with respect to interest rate).
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Call rho (per 1% change in interest rate)
        """
        if T <= 0:
            return 0.0
        
        d2 = self._d2(S, K, T, r, sigma)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        return rho
    
    def rho_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Rho for put option.
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Put rho (per 1% change in interest rate)
        """
        if T <= 0:
            return 0.0
        
        d2 = self._d2(S, K, T, r, sigma)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        return rho
    
    def option_summary(self, S: float, K: float, T: float, r: float, sigma: float, 
                      option_type: str = 'both') -> dict:
        """
        Calculate all option metrics for call and/or put options.
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call', 'put', or 'both'
            
        Returns:
            Dictionary containing all calculated metrics
        """
        results = {}
        
        if option_type.lower() in ['call', 'both']:
            results['call'] = {
                'price': self.call_price(S, K, T, r, sigma),
                'delta': self.call_delta(S, K, T, r, sigma),
                'gamma': self.gamma(S, K, T, r, sigma),
                'vega': self.vega(S, K, T, r, sigma),
                'theta': self.call_theta(S, K, T, r, sigma),
                'rho': self.rho_call(S, K, T, r, sigma)
            }
        
        if option_type.lower() in ['put', 'both']:
            results['put'] = {
                'price': self.put_price(S, K, T, r, sigma),
                'delta': self.put_delta(S, K, T, r, sigma),
                'gamma': self.gamma(S, K, T, r, sigma),
                'vega': self.vega(S, K, T, r, sigma),
                'theta': self.put_theta(S, K, T, r, sigma),
                'rho': self.rho_put(S, K, T, r, sigma)
            }
        
        return results


# Convenience functions for direct usage
def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate European call option price using Black-Scholes formula.
    
    Args:
        S: Current underlying price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        
    Returns:
        Call option price
    """
    pricer = BlackScholesOptionPricer()
    return pricer.call_price(S, K, T, r, sigma)


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate European put option price using Black-Scholes formula.
    
    Args:
        S: Current underlying price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        
    Returns:
        Put option price
    """
    pricer = BlackScholesOptionPricer()
    return pricer.put_price(S, K, T, r, sigma)


def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, 
                    option_type: str = 'both') -> dict:
    """
    Calculate all Greeks for European options.
    
    Args:
        S: Current underlying price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call', 'put', or 'both'
        
    Returns:
        Dictionary containing all Greeks
    """
    pricer = BlackScholesOptionPricer()
    return pricer.option_summary(S, K, T, r, sigma, option_type)


# Example usage and testing
if __name__ == "__main__":
    # Example parameters
    S = 100    # Current stock price
    K = 105    # Strike price
    T = 0.25   # Time to maturity (3 months)
    r = 0.05   # Risk-free rate (5%)
    sigma = 0.2  # Volatility (20%)
    
    # Create option pricer instance
    pricer = BlackScholesOptionPricer()
    
    print("Black-Scholes Option Pricing Calculator")
    print("=" * 40)
    print(f"Underlying Price: ${S}")
    print(f"Strike Price: ${K}")
    print(f"Time to Maturity: {T} years")
    print(f"Risk-free Rate: {r*100}%")
    print(f"Volatility: {sigma*100}%")
    print()
    
    # Calculate option prices
    call_price = pricer.call_price(S, K, T, r, sigma)
    put_price = pricer.put_price(S, K, T, r, sigma)
    
    print("Option Prices:")
    print(f"Call Option: ${call_price:.4f}")
    print(f"Put Option: ${put_price:.4f}")
    print()
    
    # Calculate Greeks
    print("Greeks:")
    print(f"Call Delta: {pricer.call_delta(S, K, T, r, sigma):.4f}")
    print(f"Put Delta: {pricer.put_delta(S, K, T, r, sigma):.4f}")
    print(f"Gamma: {pricer.gamma(S, K, T, r, sigma):.4f}")
    print(f"Vega: {pricer.vega(S, K, T, r, sigma):.4f}")
    print(f"Call Theta: {pricer.call_theta(S, K, T, r, sigma):.4f}")
    print(f"Put Theta: {pricer.put_theta(S, K, T, r, sigma):.4f}")
    print(f"Call Rho: {pricer.rho_call(S, K, T, r, sigma):.4f}")
    print(f"Put Rho: {pricer.rho_put(S, K, T, r, sigma):.4f}")
    print()
    
    # Full summary
    summary = pricer.option_summary(S, K, T, r, sigma)
    print("Complete Option Summary:")
    print("-" * 25)
    for option_type, metrics in summary.items():
        print(f"{option_type.capitalize()} Option:")
        for metric, value in metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        print()
    
    # Verify put-call parity
    parity_check = call_price - put_price - S + K * np.exp(-r * T)
    print(f"Put-Call Parity Check (should be ~0): {parity_check:.6f}")
