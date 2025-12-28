import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew

def calculate_classification_features(returns, window=20):
    local_mean = returns.rolling(window=window).mean()
    local_std = returns.rolling(window=window).std()
    local_skew = returns.rolling(window=window).apply(skew, raw=True)
    
    return local_mean, local_std, local_skew

def classify_states(returns, n_states=3, window_long=20):
    mean, std, skewness = calculate_classification_features(returns, window= window_long)
    features = pd.DataFrame({
        'mean': mean,
        'std': std,
        'skewness': skewness
    })
    features_clean = features.dropna()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_clean)
    kmeans = KMeans(n_clusters=n_states, random_state=42)       
    states = kmeans.fit_predict(features_scaled)
    state_series = pd.Series(index=features.index, dtype='float64')
    state_series.loc[features_clean.index] = states
    return state_series, kmeans, scaler

def calculate_transition_matrix(states, n_states=3):
    states_clean = states.dropna().astype(int)
    transition_matrix = np.zeros((n_states, n_states))
    for i in range(len(states_clean) - 1):
        current_state = states_clean.iloc[i]
        next_state = states_clean.iloc[i + 1]
        transition_matrix[current_state, next_state] += 1
  
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums!=0, out=None)

    return transition_matrix

def fractal_classification_forecast(returns, n_classes=4, window=20):
    states, kmeans, scaler = classify_states(returns, n_states=n_classes, window_long=window)
    current_state = int(states.dropna().iloc[-1])
    transition_matrix = calculate_transition_matrix(states, n_classes)
    next_state_probs = transition_matrix[current_state]
    predicted_state = np.argmax(next_state_probs)
    mean_returns_per_state = returns.groupby(states).mean()
    return mean_returns_per_state[predicted_state]

def fractal_classification_backtest(returns, n_classes=4, window=20, min_history=100):
    correct = 0
    total = 0  
    for i in range(min_history, len(returns) - 1):
        past_returns = returns.iloc[:i]     
        try:
            forecast = fractal_classification_forecast(past_returns, n_classes, window)
        except Exception:
            continue       
        actual = returns.iloc[i]
        if (forecast > 0 and actual > 0) or (forecast < 0 and actual < 0):
            correct += 1
        total += 1
    accuracy = correct / total if total > 0 else 0
    return accuracy