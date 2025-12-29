from fractime.core.returns import log_returns
from fractime.data.loader import load_data
from fractime.core.hurst import hurst_exponent
from fractime.forecasters.rs_forecaster import rs_forecast
from fractime.backtest.engine import rs_backtest, st_frsr_backtest, fractal_projections_backtest
from fractime.core.ttw import calculate_activity, calculate_dilation, calculate_trading_time, apply_ttw
from fractime.forecasters.st_frsr import identify_states, calculate_transition_matrix, st_frsr_forecast
from fractime.forecasters.fractal_projections import fractal_projection_forecast
from fractime.forecasters.fractal_classification import fractal_classification_forecast, fractal_classification_backtest
from fractime.forecasters.ensemble import fractal_ensemble_forecast, fractal_ensemble_backtest
from fractime.backtest.engine  import full_backtest
from fractime.analysis.scanner import scan_asset, scan_multiple_assets
from fractime.core.fractal_interpolation import fractal_interpolate
from fractime.core.fractal_interpolation import fractal_interpolate_adaptive, test_interpolation_benefit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


'''print("\n\nBacktest Accuracy Comparison between Original and Warped Prices:")
accuracy_original = rs_backtest(returns, window_hurst=500, window_trend=10)

warped_prices = apply_ttw(data['Close'], returns, data["Volume"], alpha=0.5)
returns_warped = pd.Series(np.diff(np.log(warped_prices)))
accuracy_warped = rs_backtest(returns_warped, window_hurst=500, window_trend=10)

print(f"Original Accuracy: {accuracy_original:.4f}")
print(f"Warped Accuracy: {accuracy_warped:.4f}")

states, kmeans_model, scaler_model = identify_states(returns)
print(f"\n\nIdentified States:\n{states.value_counts()}")

transition_matrix = calculate_transition_matrix(states)
print(f"\nTransition Matrix:\n{transition_matrix}")
predicted_return = st_frsr_forecast(returns)

print("\n\nFR-Forecaster vs ST-FRSR Forecaster Comparison: ")
print(f"ST-FRSR Predicted Return: {predicted_return}")
print(f"RS-Forecaster Predicted Return: {forecast}")

accuracy_st_frsr = st_frsr_backtest(returns, window_short=5, window_long=20, n_states=3)
print(f"ST-FRSR Backtest Accuracy: {accuracy_st_frsr}")

fractal_forecast = fractal_projection_forecast(returns, pattern_length=10, min_similarity=0.7, lookahead=1)
print(f"Fractal Projection Forecasted Return: {fractal_forecast}")

accuracy_fractal = fractal_projections_backtest(returns, window_hurst=500, window_trend=10)
print(f"\nFractal Projections Backtest Accuracy: {accuracy_fractal}")

# Single forecast test
forecast_fc = fractal_classification_forecast(returns, n_classes=4, window=20)
print(f"Fractal Classification Forecast: {forecast_fc}")

# Backtest
print("Fractal Classification Backtest in progress...")
accuracy_fc = fractal_classification_backtest(returns, n_classes=4, window=20, min_history=100)
print(f"Fractal Classification Accuracy: {accuracy_fc:.4f}")

# Single test
ensemble_forecast = fractal_ensemble_forecast(returns, method='vote')
print(f"Ensemble Forecast (vote): {ensemble_forecast}")

# Backtest all methods
for method in ['vote', 'mean', 'weighted']:
    accuracy = fractal_ensemble_backtest(returns, method=method, min_history=500)
    print(f"Ensemble Accuracy ({method}): {accuracy:.4f}")

metrics = full_backtest(returns, fractal_projection_forecast, min_history=100, 
                        pattern_length=20, min_similarity=0.5)
print("Fractal Projection Metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

print("=== COMPLETE FORECASTER COMPARISON ===\n")

# RS Forecaster
metrics_rs = full_backtest(returns, rs_forecast, min_history=500, 
                           window_hurst=500, window_trend=10)
print("RS Forecaster:")
for k, v in metrics_rs.items():
    print(f"  {k}: {v:.4f}")

# ST-FRSR
metrics_st = full_backtest(returns, st_frsr_forecast, min_history=200,
                           window_short=5, window_long=20, n_states=3)
print("\nST-FRSR:")
for k, v in metrics_st.items():
    print(f"  {k}: {v:.4f}")

# Fractal Projection
metrics_fp = full_backtest(returns, fractal_projection_forecast, min_history=100,
                           pattern_length=20, min_similarity=0.5)
print("\nFractal Projection:")
for k, v in metrics_fp.items():
    print(f"  {k}: {v:.4f}")

# Fractal Classification  
metrics_fc = full_backtest(returns, fractal_classification_forecast, min_history=100,
                           n_classes=4, window=20)
print("\nFractal Classification:")
for k, v in metrics_fc.items():
    print(f"  {k}: {v:.4f}")

# Ensemble
metrics_ens = full_backtest(returns, fractal_ensemble_forecast, min_history=500,
                            method='weighted')
print("\nEnsemble (weighted):")
for k, v in metrics_ens.items():
    print(f"  {k}: {v:.4f}")

results = scan_multiple_assets(['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD',
                                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
                                'SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'SLV',
                                'CL=F', 'GC=F', 'SI=F',
                                'EURUSD=X', 'GBPUSD=X', 'USDJPY=X'], "2024-01-01", "2025-12-01", interval='1h')
                    
print("\n\nAsset Scan Results (sorted by Sharpe Ratio):")
print(results[['ticker', 'sharpe_ratio', 'hurst', 'data_points']].sort_values(by='sharpe_ratio', ascending=False))'''

# Take a piece of prices
original_prices = data['Close'].values[:50]
print(f"Original points: {len(original_prices)}")

# Interpolate
interpolated = fractal_interpolate(original_prices, expansion_factor=2, alpha=0.3)
print(f"Interpolated points: {len(interpolated)}")

plt.figure(figsize=(12, 6))
plt.plot(range(0, len(interpolated), 2), original_prices, 'bo-', label='Original', markersize=4)
plt.plot(range(len(interpolated)), interpolated, 'r-', alpha=0.5, label='Interpolated')
plt.legend()
plt.title('Fractal Interpolation')
plt.savefig('fractal_interpolation.png')
plt.show()

interpolated_adaptive, interpolated_alpha = fractal_interpolate_adaptive(original_prices, expansion_factor=2)
print(f"Adaptive interpolated points: {len(interpolated_adaptive)}")
print(f"Adaptive calculated alpha: {interpolated_alpha}")

benefit = test_interpolation_benefit(data['Close'], test_size=200)
print("\n\nInterpolation Benefit Test:")
print(f"Original Accuracy: {benefit[0]:.4f}")   
print(f"Interpolated Accuracy: {benefit[1]:.4f}")
print(f"Improvement (%): {benefit[2]:.2f}%")