import numpy as np
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import warnings

def garch_forecast(returns, p=1, q=1):

    if hasattr(returns, 'values'):
        returns = returns.values

    scale = 100
    returns_scaled = returns * scale
 
    model = arch_model(returns_scaled, vol='Garch', p=p, q=q, mean='Zero')
    
    try:
        fitted_model = model.fit(disp='off')
        forecast = fitted_model.forecast(horizon=1)
        predicted_variance_scaled = forecast.variance.values[-1, 0]
    except Exception as e:
        print(f"Error in GARCH model fitting or forecast: {e}")
        predicted_variance_scaled = np.nan

    if not np.isnan(predicted_variance_scaled):
        predicted_variance = predicted_variance_scaled / (scale ** 2)
    else:
        predicted_variance = np.nan
    
    return predicted_variance

def garch_mean_forecast(returns, p=1, q=1):
    if hasattr(returns, 'values'):
        returns = returns.values
    
    scale = 100
    returns_scaled = returns * scale
    
    # Use mean='AR' to also model the mean
    model = arch_model(returns_scaled, vol='Garch', p=p, q=q, mean='AR', lags=1)
    
    try:
        fitted = model.fit(disp='off')
        forecast = fitted.forecast(horizon=1)
        
        predicted_mean = forecast.mean.values[-1, 0] / scale
        predicted_var = forecast.variance.values[-1, 0] / (scale ** 2)
        
        return predicted_mean, predicted_var
    except Exception as e:
        print(f"Error in GARCH model with mean fitting or forecast: {e}")
        return np.nan, np.nan
    
    
def arima_forecast(returns, p=1, d=0, q=1):
    if hasattr(returns, 'values'):
        returns = returns.values
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(returns, order=(p, d, q))
            fitted = model.fit()
            forecast = fitted.forecast(steps=1)
            return forecast[0]
    except Exception:
        return np.nan