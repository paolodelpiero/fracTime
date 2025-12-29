from numba import jit
import numpy as np

@jit(nopython=True)
def _pattern_similarity_numba(p1, p2):
    # Normalize
    p1_mean = np.mean(p1)
    p1_std = np.std(p1)
    p2_mean = np.mean(p2)
    p2_std = np.std(p2)
    
    if p1_std == 0 or p2_std == 0:
        return 0.0
    
    p1_norm = (p1 - p1_mean) / p1_std
    p2_norm = (p2 - p2_mean) / p2_std
    
    similarity = np.dot(p1_norm, p2_norm) / len(p1)
    return similarity

@jit(nopython=True)
def _find_similar_patterns_numba(returns, pattern_length, min_similarity):
    n = len(returns)
    recent_pattern = returns[-pattern_length:]

    # Pre-allocate array for results (maximum possible)
    max_matches = n - 2 * pattern_length
    indices = np.empty(max_matches, dtype=np.int64)
    similarities = np.empty(max_matches, dtype=np.float64)
    count = 0
    
    for start in range(n - 2 * pattern_length):
        historical_pattern = returns[start:start + pattern_length]
        similarity = _pattern_similarity_numba(recent_pattern, historical_pattern)
        
        if similarity >= min_similarity:
            indices[count] = start
            similarities[count] = similarity
            count += 1
    
    return indices[:count], similarities[:count]

# Public wrappers
def pattern_similarity(pattern1, pattern2):
    return _pattern_similarity_numba(
        np.asarray(pattern1, dtype=np.float64),
        np.asarray(pattern2, dtype=np.float64)
    )

def find_similar_patterns(returns, pattern_length=20, min_similarity=0.7):
    if hasattr(returns, 'values'):
        returns = returns.values
    returns = np.asarray(returns, dtype=np.float64)
    
    indices, similarities = _find_similar_patterns_numba(returns, pattern_length, min_similarity)
    
    return list(zip(indices, similarities))

def fractal_projection_forecast(returns, pattern_length=20, min_similarity=0.7, lookahead=1):
    similar_patterns = find_similar_patterns(returns, pattern_length, min_similarity)
    
    if len(similar_patterns) == 0:
        return 0.0
    
    if hasattr(returns, 'values'):
        returns_arr = returns.values
    else:
        returns_arr = returns
    
    weighted_returns = []
    total_weight = 0.0
    
    for start, similarity in similar_patterns:
        future_index = start + pattern_length + lookahead - 1
        if future_index < len(returns_arr):
            future_return = returns_arr[future_index]
            weighted_returns.append(future_return * similarity)
            total_weight += similarity
    
    if total_weight == 0:
        return 0.0
    
    return sum(weighted_returns) / total_weight