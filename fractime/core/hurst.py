from numba import jit
import numpy as np

@jit(nopython=True)
def _calculate_rs_numba(returns, n):
    num_blocks = len(returns) // n
    if num_blocks == 0:
        return np.nan
    
    rs_values = np.empty(num_blocks)
    
    for i in range(num_blocks):
        block = returns[i*n : (i+1)*n]
        mean = np.mean(block)
        deviations = block - mean
        cumsum = np.cumsum(deviations)
        R = np.max(cumsum) - np.min(cumsum)
        S = np.std(block)
        if S == 0:
            rs_values[i] = np.nan
        else:
            rs_values[i] = R / S
    
    return np.nanmean(rs_values)

@jit(nopython=True)
def _hurst_numba(returns):
    n_values = np.array([10, 20, 50, 100, 200])
    valid_n = n_values[n_values < len(returns) // 2]
    
    if len(valid_n) < 2:
        return np.nan
    
    log_n = np.log(valid_n.astype(np.float64))
    log_rs = np.empty(len(valid_n))
    
    for i, n in enumerate(valid_n):
        log_rs[i] = np.log(_calculate_rs_numba(returns, n))

    # Manual linear regression
    n_points = len(log_n)
    sum_x = np.sum(log_n)
    sum_y = np.sum(log_rs)
    sum_xy = np.sum(log_n * log_rs)
    sum_xx = np.sum(log_n * log_n)
    
    slope = (n_points * sum_xy - sum_x * sum_y) / (n_points * sum_xx - sum_x * sum_x)
    
    return slope

# Wrapper functions with original names
def calculate_rs(returns, n):
    return _calculate_rs_numba(np.asarray(returns, dtype=np.float64), n)

def hurst_exponent(returns):
    return _hurst_numba(np.asarray(returns, dtype=np.float64))