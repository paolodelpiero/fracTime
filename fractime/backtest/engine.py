from fractime.forecasters.rs_forecaster import rs_forecast

def rs_backtest(returns, window_hurst, window_trend):
    correct = 0
    total = 0
    
    for i in range(window_hurst, len(returns) - 1): 

        past_returns = returns[:i]

        forecast = rs_forecast(past_returns, window_hurst, window_trend)

        actual = returns.iloc[i]

        if (forecast > 0 and actual > 0) or (forecast < 0 and actual < 0):
            correct += 1
        total += 1
    
    accuracy = correct / total
    return accuracy


