from fractime.forecasters.rs_forecaster import rs_forecast
from fractime.forecasters.st_frsr import st_frsr_forecast
from fractime.forecasters.fractal_projections import fractal_projection_forecast
from fractime.forecasters.fractal_classification import fractal_classification_forecast

def fractal_ensemble_forecast(returns, method='vote'):
    rs_forecast_prevision = rs_forecast(returns, window_hurst=500, window_trend=10)
    st_frsr_forecast_prevision = st_frsr_forecast(returns, window_short=5, window_long=20, n_states=3)
    fractal_projection_prevision = fractal_projection_forecast(returns, pattern_length=10, min_similarity=0.7, lookahead=1)
    fractal_classification_prevision = fractal_classification_forecast(returns, n_classes=4, window=20)
    forecasts = [rs_forecast_prevision, st_frsr_forecast_prevision, fractal_projection_prevision, fractal_classification_prevision]
    if method == 'vote':
        signs = [1 if f > 0 else -1 if f < 0 else 0 for f in forecasts]
        vote = sum(signs)
        combined_forecast = 1.0 if vote > 0 else -1.0 if vote < 0 else 0.0
    elif method == 'mean':
        combined_forecast = sum(forecasts) / len(forecasts)
    elif method == 'weighted':
        weights = [0.10, 0.10, 0.40, 0.40]  # RS, ST-FRSR, Projection, Classification
        combined_forecast = sum(f * w for f, w in zip(forecasts, weights))
    else:
        raise ValueError("Metodo sconosciuto. Usa 'vote', 'mean' o 'weighted'.")    
    return combined_forecast

def fractal_ensemble_backtest(returns, method='vote', min_history=500):
    correct = 0
    total = 0
    
    for i in range(min_history, len(returns) - 1):
        past_returns = returns.iloc[:i]
        
        try:
            forecast = fractal_ensemble_forecast(past_returns, method)
        except Exception:
            continue
            
        actual = returns.iloc[i]
        
        if (forecast > 0 and actual > 0) or (forecast < 0 and actual < 0):
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy
