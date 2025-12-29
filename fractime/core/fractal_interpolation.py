import numpy as np
import pandas as pd
from fractime.core.returns import log_returns
from fractime.backtest.engine import rs_backtest
from fractime.core.hurst import hurst_exponent

def fractal_interpolate(prices, expansion_factor=2, alpha=0.5):
    if hasattr(prices, 'values'):
        prices = prices.values
    prices = np.asarray(prices, dtype=np.float64)
    
    result = []
    
    for i in range(len(prices) - 1):
        p1 = prices[i]
        p2 = prices[i + 1]
        result.append(p1)

        for j in range(1, expansion_factor):
            t = j / expansion_factor
            linear_point = p1 + t * (p2 - p1)
            new_point = linear_point + alpha * np.random.randn() * (p2 - p1)
            
            result.append(new_point)

    result.append(prices[-1])
    
    return np.array(result)

def fractal_interpolate_adaptive(prices, expansion_factor=2):
    log_returns = np.diff(np.log(prices))
    H = hurst_exponent(log_returns)
    alpha = 1 - H
    return fractal_interpolate(prices, expansion_factor, alpha), alpha

def test_interpolation_benefit(prices, test_size=200):
    small_prices = prices.iloc[:test_size]
    returns_original = log_returns(small_prices)
    prices_interpolated, alpha = fractal_interpolate_adaptive(small_prices.values)
    returns_interpolated = pd.Series(np.diff(np.log(prices_interpolated)))
    accuracy_original = rs_backtest(returns_original, window_hurst=50, window_trend=5)
    accuracy_interpolated = rs_backtest(returns_interpolated, window_hurst=50, window_trend=5)
    improvement = (accuracy_interpolated - accuracy_original) / accuracy_original * 100
    
    return accuracy_original, accuracy_interpolated, improvement