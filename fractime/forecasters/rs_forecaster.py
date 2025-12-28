from fractime.core.hurst import hurst_exponent

def rs_forecast(returns, window_hurst, window_trend):
    H = hurst_exponent(returns[-window_hurst:].values)
    Trend = returns[-window_trend:].mean()
    if H > 0.5:
        forecast = Trend * (H - 0.5) * 2
    else:
        forecast = -Trend * (0.5 - H) * 2
    return forecast
