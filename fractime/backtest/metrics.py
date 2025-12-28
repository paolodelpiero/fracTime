import numpy as np

def calculate_rmse(actual, predicted):
    # Root Mean Squared Error
    return np.sqrt(np.mean((actual - predicted) ** 2))

def calculate_mae(actual, predicted):
    # Mean Absolute Error
    return np.mean(np.abs(actual - predicted))

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    # Sharpe = (mean(returns) - risk_free_rate) / std(returns)
    excess_returns = returns - risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    sharpe_ratio *= np.sqrt(252)
    return sharpe_ratio

def calculate_directional_accuracy(actual, predicted):
    correct_directions = np.sign(actual) == np.sign(predicted)
    return np.mean(correct_directions)
