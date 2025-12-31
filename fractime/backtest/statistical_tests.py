import numpy as np
from scipy import stats
import pandas as pd

def diebold_mariano_test(errors1, errors2, loss='squared'):
    if loss == 'squared':
        loss1 = errors1 ** 2
        loss2 = errors2 ** 2
    elif loss == 'absolute':
        loss1 = np.abs(errors1)
        loss2 = np.abs(errors2)
    else:
        raise ValueError("Loss must be 'squared' or 'absolute'")
    d = loss1 - loss2
    n = len(d)
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    dm_statistic = mean_d / np.sqrt(var_d / n)
    p_value = 2 * stats.norm.sf(np.abs(dm_statistic))
    return dm_statistic, p_value

def collect_forecast_errors(returns, forecaster_func, min_history=100, **kwargs):
    errors = []
    actuals = []
    predictions = []
    
    # Keep as pandas Series if possible
    if hasattr(returns, 'iloc'):
        returns_series = returns
    else:
        returns_series = pd.Series(returns)
    
    for i in range(min_history, len(returns_series) - 1):
        past = returns_series.iloc[:i]
        
        try:
            pred = forecaster_func(past, **kwargs)
        except Exception:
            continue
        
        actual = returns_series.iloc[i]
        
        errors.append(actual - pred)
        actuals.append(actual)
        predictions.append(pred)
    
    return np.array(errors), np.array(actuals), np.array(predictions)

def collect_directional_errors(prices, forecaster_func, min_history=100, **kwargs):
    errors = []
    
    if hasattr(prices, 'values'):
        prices_arr = prices.values
    else:
        prices_arr = np.array(prices)
    
    for i in range(min_history, len(prices_arr) - 1):
        historical = prices_arr[:i]
        
        try:
            predicted_price = forecaster_func(historical, **kwargs)
        except Exception:
            continue
        
        current = prices_arr[i-1]
        actual_next = prices_arr[i]

        predicted_up = predicted_price > current
        actual_up = actual_next > current

        error = 0 if predicted_up == actual_up else 1
        errors.append(error)
    
    return np.array(errors)

def calculate_qlike(actual_volatility, predicted_volatility):
    actual_volatility = np.clip(actual_volatility, 1e-10, None)
    predicted_volatility = np.clip(predicted_volatility, 1e-10, None)
    qlike_values = np.log(predicted_volatility) + actual_volatility / predicted_volatility  
    return np.mean(qlike_values)