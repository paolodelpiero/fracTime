from fractime.core.returns import log_returns
from fractime.data.loader import load_data
from fractime.core.hurst import hurst_exponent
from fractime.forecasters.rs_forecaster import rs_forecast
from fractime.backtest.engine import rs_backtest
from fractime.core.ttw import calculate_activity, calculate_dilation, calculate_trading_time, apply_ttw
from fractime.forecasters.st_frsr import identify_states
import numpy as np
import pandas as pd

data = load_data("BTC-USD", "2022-01-01", "2025-12-01", "1d")

returns = log_returns(data['Close'])
hurst = hurst_exponent(returns.values)
forecast = rs_forecast(returns, window_hurst=500, window_trend=10)
print(f"Hurst Exponent: {hurst}")   
print(f"RS Forecast: {forecast}")

accuracy = rs_backtest(returns, window_hurst=500, window_trend=10)
print(f"Backtest Accuracy: {accuracy}")

activity = calculate_activity(returns, data["Volume"])
print(f"Activity Measure: {activity}")

dilation = calculate_dilation(activity)
print(f"Dilation Measure: {dilation}")

trading_time = calculate_trading_time(dilation)
print(f"Trading Time: {trading_time}")

warped_prices = apply_ttw(data['Close'], returns, data["Volume"])
print(f"Warped Prices: {warped_prices}")

print("\n\nReal Prices Hurst vs Warped Prices Hurst Comparison:")
hurst_warped = hurst_exponent(np.diff(np.log(warped_prices)))
print(f"Real Prices Hurst: {hurst}")
print(f"Warped Prices Hurst: {hurst_warped}")

print("\n\nHurst Exponent for different alpha values in TTW:")
for alpha in [0.25, 0.5, 0.75, 1.0, 1.5]:
    warped_prices = apply_ttw(data['Close'], returns, data["Volume"], alpha=alpha)
    hurst_warped = hurst_exponent(np.diff(np.log(warped_prices)))
    print(f"Alpha {alpha}: Hurst warped = {hurst_warped:.4f}")


print("\n\nBacktest Accuracy Comparison between Original and Warped Prices:")
accuracy_original = rs_backtest(returns, window_hurst=500, window_trend=10)

warped_prices = apply_ttw(data['Close'], returns, data["Volume"], alpha=0.5)
returns_warped = pd.Series(np.diff(np.log(warped_prices)))
accuracy_warped = rs_backtest(returns_warped, window_hurst=500, window_trend=10)

print(f"Accuracy originale: {accuracy_original:.4f}")
print(f"Accuracy warped: {accuracy_warped:.4f}")

states, kmeans_model, scaler_model = identify_states(returns)
print(f"\n\nIdentified States:\n{states.value_counts()}")