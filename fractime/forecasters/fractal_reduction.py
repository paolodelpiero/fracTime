import numpy as np

def decompose_to_binary(prices, n_levels=5):
    min_price = np.min(prices)
    max_price = np.max(prices)
    thresholds = np.linspace(min_price, max_price, n_levels + 2)[1:-1]
    binary_series = np.zeros((n_levels, len(prices)))
    for i, threshold in enumerate(thresholds):
        binary_series[i] = (prices > threshold).astype(int)
    return binary_series, thresholds

def decompose_to_binary_adaptive(prices, n_levels=5, lookback=50):
    recent = prices[-lookback:] if len(prices) > lookback else prices
    min_price = np.min(recent)
    max_price = np.max(recent)
    thresholds = np.linspace(min_price, max_price, n_levels + 2)[1:-1]
    
    binary_series = np.zeros((n_levels, len(prices)))
    for i, threshold in enumerate(thresholds):
        binary_series[i] = (prices > threshold).astype(int)
    return binary_series, thresholds

def forecast_binary_series(binary_series, window=10):
    n_levels = binary_series.shape[0]
    predictions = np.zeros(n_levels)
    probabilities = np.zeros(n_levels)
    
    for i in range(n_levels):
        recent = binary_series[i, -window:]
        prob = np.mean(recent)
        probabilities[i] = prob
        predictions[i] = 1 if prob > 0.5 else 0
    
    return predictions, probabilities

def reconstruct_price(predictions, thresholds, gate='AND'):
    if gate == 'AND':
        for i in range(len(predictions)-1, -1, -1):
            if predictions[i] == 1:
                return thresholds[i]
        return thresholds[0] 
    
    elif gate == 'OR':
        for i in range(len(predictions)):
            if predictions[i] == 1:
                return thresholds[i]
        return thresholds[-1]
    
    elif gate == 'MAJORITY':
        count_ones = np.sum(predictions)
        if count_ones > len(predictions) / 2:
            active_thresholds = [thresholds[i] for i in range(len(predictions)) if predictions[i] == 1]
            return np.mean(active_thresholds)
        else:
            inactive_thresholds = [thresholds[i] for i in range(len(predictions)) if predictions[i] == 0]
            return np.mean(inactive_thresholds) if len(inactive_thresholds) > 0 else thresholds[0]
        
def fractal_reduction_forecast(prices, n_levels=5, window=10, gate='MAJORITY', lookback=50):
    prices = np.asarray(prices)
    binary_series, thresholds = decompose_to_binary_adaptive(prices, n_levels, lookback)
    predictions, probabilities = forecast_binary_series(binary_series, window)
    predicted_price = reconstruct_price(predictions, thresholds, gate)
    return predicted_price

def fractal_reduction_backtest(prices, n_levels=5, window=10, gate='MAJORITY', min_history=50, lookback=50):
    correct = 0
    total = 0
    
    if hasattr(prices, 'values'):
        prices = prices.values
    
    for i in range(min_history, len(prices) - 1):
        historical = prices[:i]
        predicted_price = fractal_reduction_forecast(historical, n_levels, window, gate, lookback)
        
        current = prices[i-1]
        actual_next = prices[i]
        
        predicted_up = predicted_price > current
        actual_up = actual_next > current
        
        if predicted_up == actual_up:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0