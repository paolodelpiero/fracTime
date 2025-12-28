import numpy as np

def log_returns(prices):
    try:
        log_returns = np.log(prices).diff()
        log_returns.dropna(inplace=True)
        return log_returns
    except Exception as e:
        raise ValueError(f"Error calculating log returns: {e}")