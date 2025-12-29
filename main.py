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
from fractime.analysis.mdfa import fractal_coherence, rolling_coherence
from fractime.simulation.fbm import generate_fbm_path, generate_price_scenarios, calculate_risk_metrics
from fractime.forecasters.fractal_reduction import decompose_to_binary, forecast_binary_series
from fractime.forecasters.fractal_reduction import fractal_reduction_forecast, fractal_reduction_backtest, decompose_to_binary_adaptive


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = load_data("BTC-USD", "2024-01-01", "2025-12-01", "1h")

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
print(results[['ticker', 'sharpe_ratio', 'hurst', 'data_points']].sort_values(by='sharpe_ratio', ascending=False))

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

volume_returns = log_returns(data['Volume'])

coherence, H_price, H_volume = fractal_coherence(returns, volume_returns, window=100, step=10)

print(f"Fractal Coherence (Price-Volume): {coherence:.4f}")
print(f"Hurst points calculated: {len(H_price)}")
print(f"Average price Hurst: {np.mean(H_price):.4f}")
print(f"Average volume Hurst: {np.mean(H_volume):.4f}")

coherence_series = rolling_coherence(returns, volume_returns, coherence_window=50, hurst_window=100, step=10)
print(f"Coherence points calculated: {len(coherence_series)}")
coherence_values = [c[1] for c in coherence_series]
print(f"Average coherence: {np.mean(coherence_values):.4f}")
print(f"Min coherence: {np.min(coherence_values):.4f}")
print(f"Max coherence: {np.max(coherence_values):.4f}")

indices = [c[0] for c in coherence_series]
values = [c[1] for c in coherence_series]

plt.figure(figsize=(12, 6))
plt.plot(indices, values, 'b-', label='Fractal Coherence')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='High threshold')
plt.xlabel('Time')
plt.ylabel('Coherence')
plt.title('Rolling Fractal Coherence (Price-Volume)')
plt.legend()
plt.savefig('rolling_coherence.png')
plt.show()

np.random.seed(42)

plt.figure(figsize=(12, 6))

for H in [0.3, 0.5, 0.7, 0.9]:
    path = generate_fbm_path(200, H, seed=42)
    plt.plot(path, label=f'H={H}')

plt.legend()
plt.title('Fractional Brownian Motion - Different Hurst')
plt.xlabel('Time')
plt.ylabel('Value')
plt.savefig('fbm_paths.png')
plt.show()

price_scenarios = generate_price_scenarios(current_price=100, H=0.7, volatility=0.01, n_steps=200, n_scenarios=5)
plt.figure(figsize=(12, 6))
for i in range(price_scenarios.shape[0]):
    plt.plot(price_scenarios[i], label=f'Scenario {i+1}')
plt.title('Price Scenarios via Fractional Brownian Motion')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.savefig('fbm_price_scenarios.png')
plt.show()

risk_metrics = calculate_risk_metrics(price_scenarios, confidence_level=0.95)
print("\n\nRisk Metrics from FBM Price Scenarios:")
for k, v in risk_metrics.items():
    print(f"  {k}: {v:.4f}")

# Parameters from historical data
current_price = data['Close'].iloc[-1]
H = hurst_exponent(returns.values)
volatility = returns.std()

print(f"Current price: {current_price:.2f}")
print(f"Hurst: {H:.4f}")
print(f"Daily volatility: {volatility:.4f}")

# Generate scenarios
scenarios = generate_price_scenarios(
    current_price=current_price,
    H=H,
    volatility=volatility,
    n_steps=30,  # 30 days
    n_scenarios=1000
)

# Calculate risk
risk = calculate_risk_metrics(scenarios, confidence_level=0.95)
print("\nRisk Metrics (30 days, 1000 scenarios):")
for k, v in risk.items():
    print(f"  {k}: {v:.4f}")'''

# Test Fractal Reduction with adaptive thresholds
print("=" * 60)
print("FRACTAL REDUCTION TEST")
print("=" * 60)

# Manual test
historical_prices = data['Close'].values[-100:-1]
print(f"\nHistorical price range: {historical_prices.min():.2f} - {historical_prices.max():.2f}")
print(f"Last 10 prices: {historical_prices[-10:]}")

binary, thresholds = decompose_to_binary_adaptive(historical_prices, n_levels=5, lookback=50)
predictions, probabilities = forecast_binary_series(binary, window=10)

print(f"\nAdaptive thresholds (last 50 prices): {thresholds}")
print(f"Predictions: {predictions}")
print(f"Probabilities: {probabilities}")

# Forecast
predicted = fractal_reduction_forecast(historical_prices, n_levels=5, window=10, gate='MAJORITY', lookback=50)
current_price = historical_prices[-1]
print(f"\nCurrent price: {current_price:.2f}")
print(f"Prediction: {predicted:.2f}")
print(f"Direction: {'UP' if predicted > current_price else 'DOWN'}")

# Backtest
print("\nBacktest in progress...")
accuracy_fr = fractal_reduction_backtest(data['Close'], n_levels=5, window=10, gate='MAJORITY', min_history=100, lookback=50)
print(f"Fractal Reduction Accuracy (adaptive): {accuracy_fr:.4f}")

# Comparison between different gates
print("\nComparison between different gates:")
for gate in ['AND', 'OR', 'MAJORITY']:
    acc = fractal_reduction_backtest(data['Close'], n_levels=5, window=10, gate=gate, min_history=100, lookback=50)
    print(f"  {gate}: {acc:.4f}")