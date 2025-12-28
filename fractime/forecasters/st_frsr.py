from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def calculate_local_features(returns, window_short=5, window_long=20):
    local_volatility = returns.rolling(window=window_long).std()
    local_trend = returns.rolling(window=window_long).mean()
    vol_short = returns.rolling(window=window_short).std()
    vol_long = returns.rolling(window=window_long).std()
    scale_ratio = vol_short / vol_long

    return local_volatility, local_trend, scale_ratio


def identify_states(returns, n_states=3, window_short=5, window_long=20):
    volatility, trend, scale_ratio = calculate_local_features(returns, window_short, window_long)
    features = pd.DataFrame({
        'volatility': volatility,
        'trend': trend,
        'scale_ratio': scale_ratio
    })
    features_clean = features.dropna()
    # Normalize features
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

def st_frsr_forecast(returns, window_short=5, window_long=20, n_states=3):
    states, kmeans, scaler = identify_states(returns, n_states, window_short, window_long)
    transition_matrix = calculate_transition_matrix(states, n_states)
    current_state = int(states.dropna().iloc[-1])
    next_state_probs = transition_matrix[current_state]
    predicted_state = np.argmax(next_state_probs)
    mean_returns_per_state = returns.groupby(states).mean()
    return mean_returns_per_state[predicted_state]