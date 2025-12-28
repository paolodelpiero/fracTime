import numpy as np

# Trading Time Warping
def calculate_activity(returns, volume=None):
    vr_5 = returns.rolling(window=5).std()
    vr_21 = returns.rolling(window=21).std()
    vr_63 = returns.rolling(window=63).std()
    volatility = (vr_5 + vr_21 + vr_63) / 3
    vol_rel = volatility / volatility.rolling(window=21).mean()

    if volume is not None:
        vol_norm = volume / volume.rolling(window=21).mean()
        activity = np.sqrt(vol_rel * vol_norm)
    else:
        activity = vol_rel

    return activity

# Time Dilation Factor
def calculate_dilation(activity, alpha=0.5, min_scale=0.1, max_scale=10.0):
    dilation = 1.0 / (activity ** alpha) 
    dilation = dilation.clip(lower=min_scale, upper=max_scale)
    
    return dilation

def calculate_trading_time(dilation):
    dilation = dilation.dropna()
    trading_time = dilation.cumsum()
    trading_time_normalized = (trading_time / trading_time.max()) * len(dilation)

    return trading_time_normalized

def apply_ttw(prices, returns, volume=None, alpha=0.5):
    activity = calculate_activity(returns, volume)
    dilation = calculate_dilation(activity, alpha=alpha)
    trading_time = calculate_trading_time(dilation)
    valid_start = len(prices) - len(trading_time)
    prices_aligned = prices.iloc[valid_start:].values
    uniform_grid = np.linspace(trading_time.min(), trading_time.max(), len(trading_time))
    warped_prices = np.interp(uniform_grid, trading_time.values, prices_aligned)
    
    return warped_prices