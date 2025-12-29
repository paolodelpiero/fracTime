from fractime.core.hurst import hurst_exponent
import numpy as np

def rolling_hurst(returns, window=100, step=10):
    hurst_values = []
    if hasattr(returns, 'values'):
        returns_arr = returns.values
    else:
        returns_arr = np.asarray(returns)
    
    for start in range(0, len(returns_arr) - window + 1, step):
        windowed_returns = returns_arr[start:start + window]
        H = hurst_exponent(windowed_returns)
        hurst_values.append((start + window - 1, H))

    return hurst_values

def fractal_coherence(price_returns, volume_returns, window=100, step=10):
    hurst_price = rolling_hurst(price_returns, window, step)
    hurst_volume = rolling_hurst(volume_returns, window, step)
    H_price = np.array([h[1] for h in hurst_price if not np.isnan(h[1])])
    H_volume = np.array([h[1] for h in hurst_volume if not np.isnan(h[1])])
    min_length = min(len(H_price), len(H_volume))
    if min_length == 0:
        return np.nan, H_price, H_volume
    correlation = np.corrcoef(H_price[:min_length], H_volume[:min_length])[0, 1]

    return correlation, H_price, H_volume

def rolling_coherence(price_returns, volume_returns, coherence_window=50, hurst_window=100, step=10):
    hurst_price = rolling_hurst(price_returns, hurst_window, step)
    hurst_volume = rolling_hurst(volume_returns, hurst_window, step)
    H_price = np.array([h[1] for h in hurst_price if not np.isnan(h[1])])
    H_volume = np.array([h[1] for h in hurst_volume if not np.isnan(h[1])]) 
    min_length = min(len(H_price), len(H_volume))
    coherence_series = []
    for start in range(0, min_length - coherence_window + 1, step):
        window_price = H_price[start:start + coherence_window]
        window_volume = H_volume[start:start + coherence_window]
        corr = np.corrcoef(window_price, window_volume)[0, 1]
        coherence_series.append((start + coherence_window - 1, corr))
    return coherence_series

