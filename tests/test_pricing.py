import pytest
from market_maker import MarketMaker

def test_black_scholes_call_known_value():
    # Create a dummy market maker instance to access the pricer
    mm = MarketMaker(starting_cash=0, gamma=0.1, risk_free_rate=0.05)
    
    # Test with a known case (values can be verified online)
    price = mm.bs_pricer.call_price(
        S=100, K=100, T=1, r=0.05, sigma=0.2
    )
    
    # The expected value is approximately 10.45
    assert abs(price - 10.45) < 0.01