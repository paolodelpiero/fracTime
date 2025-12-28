import numpy as np

def pattern_similarity(pattern1, pattern2):
    # Normalizza entrambi i pattern (media 0, std 1)
    p1 = (pattern1 - np.mean(pattern1)) / np.std(pattern1)
    p2 = (pattern2 - np.mean(pattern2)) / np.std(pattern2)
    
    # Correlazione (prodotto scalare normalizzato)
    similarity = np.dot(p1, p2) / len(p1)
    
    return similarity

def find_similar_patterns(returns, pattern_length=20, min_similarity=0.7):
    recent_pattern = returns[-pattern_length:].values
    similar_patterns = []
    for start in range(len(returns) - 2 *pattern_length): 
        historical_pattern = returns[start:start + pattern_length]
        similarity = pattern_similarity(recent_pattern, historical_pattern)
        if similarity >= min_similarity:
            similar_patterns.append((start, similarity))
    return similar_patterns

def fractal_projection_forecast(returns, pattern_length=20, min_similarity=0.7, lookahead=1):
    similar_patterns = find_similar_patterns(returns, pattern_length, min_similarity)
    if len(similar_patterns) == 0:
        return 0.0
    else:
        weighted_returns = []
        total_weight = 0.0
        for start, similarity in similar_patterns:
            future_index = start + pattern_length + lookahead - 1
            if future_index < len(returns):
                future_return = returns.iloc[future_index]
                weighted_returns.append(future_return * similarity)
                total_weight += similarity
        if total_weight == 0:
            return 0.0
        forecast = sum(weighted_returns) / total_weight
        return forecast

