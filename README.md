# fracTime

A Python library for fractal-based financial time series analysis and forecasting.

## Overview

fracTime provides advanced tools for analyzing and forecasting financial time series using fractal analysis techniques. The library implements multiple forecasting methods based on the Hurst exponent, state-based models, pattern recognition, and trading time warping.

## Features

### Core Functionality
- **Hurst Exponent Calculation**: R/S analysis for measuring long-term memory in time series
- **Log Returns**: Financial returns calculation
- **Trading Time Warping (TTW)**: Transform prices into trading time based on activity and volume

### Forecasting Methods
- **R/S Forecaster**: Trend-following forecaster based on Hurst exponent
- **State-based FRSR (ST-FRSR)**: State transition-based forecasting using clustering
- **Fractal Projections**: Pattern-based forecasting using historical similarity
- **Fractal Classification**: Classification-based approach with machine learning
- **Fractal Reduction**: Binary decomposition with adaptive thresholds for multi-level forecasting
- **Ensemble Methods**: Combine multiple forecasters (voting, mean, weighted)
- **Benchmark Models**: GARCH and ARIMA for comparison

### Backtesting & Analysis
- Comprehensive backtesting engine with multiple metrics
- Performance evaluation (accuracy, precision, recall, F1-score, Sharpe ratio)
- Statistical tests (Diebold-Mariano test for forecast comparison)
- Support for all forecasting methods
- Multi-asset scanner for batch analysis

### Advanced Features
- **Fractal Interpolation**: Adaptive and fixed alpha interpolation methods
- **Fractional Brownian Motion (FBM)**: Price scenario generation with risk metrics
- **Fractal Coherence (MDFA)**: Multi-dimensional fractal analysis for correlation
- **Rolling Coherence**: Time-varying coherence analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/paolodelpiero/fracTime.git
cd fracTime

# Install dependencies
pip install -r requirements.txt
# or with uv
uv pip install -e .
```

### Requirements
- Python >= 3.13
- scikit-learn >= 1.8.0
- yfinance >= 1.0

## Quick Start

```python
from fractime.core.returns import log_returns
from fractime.data.loader import load_data
from fractime.core.hurst import hurst_exponent
from fractime.forecasters.rs_forecaster import rs_forecast
from fractime.backtest.engine import rs_backtest

# Load data
data = load_data("BTC-USD", "2022-01-01", "2025-12-01", "1d")

# Calculate returns
returns = log_returns(data['Close'])

# Calculate Hurst exponent
hurst = hurst_exponent(returns.values)
print(f"Hurst Exponent: {hurst}")

# Generate forecast
forecast = rs_forecast(returns, window_hurst=500, window_trend=10)
print(f"Forecast: {forecast}")

# Backtest the strategy
accuracy = rs_backtest(returns, window_hurst=500, window_trend=10)
print(f"Backtest Accuracy: {accuracy}")
```

## Usage Examples

### Trading Time Warping

```python
from fractime.core.ttw import calculate_activity, calculate_dilation, apply_ttw

# Calculate activity based on returns and volume
activity = calculate_activity(returns, data["Volume"])

# Calculate dilation factor
dilation = calculate_dilation(activity)

# Apply time warping to prices
warped_prices = apply_ttw(data['Close'], returns, data["Volume"], alpha=0.5)
```

### State-based Forecasting

```python
from fractime.forecasters.st_frsr import (
    identify_states,
    calculate_transition_matrix,
    st_frsr_forecast
)

# Identify market states
states, kmeans_model, scaler_model = identify_states(returns)

# Calculate state transition probabilities
transition_matrix = calculate_transition_matrix(states)

# Generate forecast
forecast = st_frsr_forecast(returns)
```

### Fractal Projections

```python
from fractime.forecasters.fractal_projections import fractal_projection_forecast

# Forecast based on similar historical patterns
forecast = fractal_projection_forecast(
    returns,
    pattern_length=10,
    min_similarity=0.7,
    lookahead=1
)
```

### Ensemble Forecasting

```python
from fractime.forecasters.ensemble import (
    fractal_ensemble_forecast,
    fractal_ensemble_backtest
)

# Single forecast using ensemble
forecast = fractal_ensemble_forecast(returns, method='vote')

# Backtest with different ensemble methods
for method in ['vote', 'mean', 'weighted']:
    accuracy = fractal_ensemble_backtest(returns, method=method)
    print(f"Ensemble Accuracy ({method}): {accuracy:.4f}")
```

### Full Backtesting

```python
from fractime.backtest.engine import full_backtest
from fractime.forecasters.fractal_projections import fractal_projection_forecast

# Run comprehensive backtest with multiple metrics
metrics = full_backtest(
    returns,
    fractal_projection_forecast,
    min_history=100,
    pattern_length=20,
    min_similarity=0.5
)

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

### Fractal Interpolation

```python
from fractime.core.fractal_interpolation import (
    fractal_interpolate,
    fractal_interpolate_adaptive,
    test_interpolation_benefit
)

# Fixed alpha interpolation
prices = data['Close'].values[:50]
interpolated = fractal_interpolate(prices, expansion_factor=2, alpha=0.3)

# Adaptive alpha interpolation
interpolated_adaptive, alpha = fractal_interpolate_adaptive(
    prices,
    expansion_factor=2
)
print(f"Calculated alpha: {alpha}")

# Test if interpolation improves forecasts
benefit = test_interpolation_benefit(data['Close'], test_size=200)
print(f"Improvement: {benefit[2]:.2f}%")
```

### Fractional Brownian Motion Simulation

```python
from fractime.simulation.fbm import (
    generate_fbm_path,
    generate_price_scenarios,
    calculate_risk_metrics
)

# Generate FBM path
path = generate_fbm_path(n_steps=200, H=0.7, seed=42)

# Generate price scenarios
scenarios = generate_price_scenarios(
    current_price=100,
    H=0.7,
    volatility=0.01,
    n_steps=30,
    n_scenarios=1000
)

# Calculate risk metrics
risk = calculate_risk_metrics(scenarios, confidence_level=0.95)
print(f"VaR: {risk['var']:.4f}")
print(f"CVaR: {risk['cvar']:.4f}")
```

### Fractal Coherence Analysis

```python
from fractime.analysis.mdfa import fractal_coherence, rolling_coherence

# Calculate coherence between price and volume
volume_returns = log_returns(data['Volume'])
coherence, H_price, H_volume = fractal_coherence(
    returns,
    volume_returns,
    window=100,
    step=10
)
print(f"Fractal Coherence: {coherence:.4f}")

# Rolling coherence over time
coherence_series = rolling_coherence(
    returns,
    volume_returns,
    coherence_window=50,
    hurst_window=100,
    step=10
)
```

### Fractal Reduction Forecaster

```python
from fractime.forecasters.fractal_reduction import (
    fractal_reduction_forecast,
    fractal_reduction_backtest,
    decompose_to_binary_adaptive
)

# Single forecast
prices = data['Close'].values[-100:]
prediction = fractal_reduction_forecast(
    prices,
    n_levels=5,
    window=10,
    gate='MAJORITY',
    lookback=50
)

# Backtest with different logic gates
for gate in ['AND', 'OR', 'MAJORITY']:
    accuracy = fractal_reduction_backtest(
        data['Close'],
        n_levels=5,
        window=10,
        gate=gate,
        min_history=100
    )
    print(f"{gate}: {accuracy:.4f}")
```

### Statistical Testing

```python
from fractime.backtest.statistical_tests import (
    diebold_mariano_test,
    collect_forecast_errors
)

# Collect forecast errors from two models
errors1, actuals1, preds1 = collect_forecast_errors(
    returns,
    fractal_projection_forecast,
    min_history=100,
    pattern_length=20
)

errors2, actuals2, preds2 = collect_forecast_errors(
    returns,
    rs_forecast,
    min_history=500,
    window_hurst=500
)

# Compare forecasts with Diebold-Mariano test
dm_stat, p_value = diebold_mariano_test(errors1, errors2, loss='squared')
print(f"DM Statistic: {dm_stat:.4f}, p-value: {p_value:.4f}")
print(f"Significant difference: {p_value < 0.05}")
```

### Multi-Asset Scanning

```python
from fractime.analysis.scanner import scan_multiple_assets

# Scan multiple assets
results = scan_multiple_assets(
    ['BTC-USD', 'ETH-USD', 'AAPL', 'MSFT', 'SPY'],
    start_date="2024-01-01",
    end_date="2025-12-01",
    interval='1h'
)

# View results sorted by Sharpe ratio
print(results[['ticker', 'sharpe_ratio', 'hurst', 'data_points']]
      .sort_values(by='sharpe_ratio', ascending=False))
```

## Project Structure

```
fractime/
├── core/              # Core functionality
│   ├── hurst.py       # Hurst exponent calculation
│   ├── returns.py     # Returns calculation
│   ├── ttw.py         # Trading Time Warping
│   └── fractal_interpolation.py  # Fractal interpolation methods
├── forecasters/       # Forecasting models
│   ├── rs_forecaster.py           # R/S based forecaster
│   ├── st_frsr.py                 # State-based forecaster
│   ├── fractal_projections.py     # Pattern-based forecaster
│   ├── fractal_classification.py  # ML classification forecaster
│   ├── fractal_reduction.py       # Binary reduction forecaster
│   ├── benchmark_models.py        # GARCH and ARIMA models
│   └── ensemble.py                # Ensemble methods
├── backtest/          # Backtesting framework
│   ├── engine.py      # Backtesting engine
│   ├── metrics.py     # Performance metrics
│   └── statistical_tests.py  # Statistical comparison tests
├── simulation/        # Simulation tools
│   └── fbm.py         # Fractional Brownian Motion
├── analysis/          # Analysis tools
│   ├── scanner.py     # Multi-asset scanner
│   └── mdfa.py        # Fractal coherence analysis
├── data/              # Data utilities
│   └── loader.py      # Data loading from yfinance
└── tests/             # Unit tests
```

## Understanding the Hurst Exponent

The Hurst exponent (H) characterizes the long-term memory of time series:
- **H = 0.5**: Random walk (no memory)
- **H > 0.5**: Persistent (trending) behavior
- **H < 0.5**: Mean-reverting (anti-persistent) behavior

This library uses R/S analysis to calculate the Hurst exponent and incorporates it into various forecasting strategies.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

This project is a Python conversion, optimization, and extension of **The FracTime Framework: A Unified Toolkit for Fractal Geometry and Probability-Weighted Time Series Forecasting** by Rick Galbo (Quantitative Finance Research).

If you use this library in your research, please cite the original work:

```
Galbo, R. "The FracTime Framework: A Unified Toolkit for Fractal Geometry and
Probability-Weighted Time Series Forecasting." Quantitative Finance Research.
```

This implementation includes additional discoveries, optimizations, and Python-specific enhancements beyond the original framework.
