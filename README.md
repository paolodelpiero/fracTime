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
- **Ensemble Methods**: Combine multiple forecasters (voting, mean, weighted)

### Backtesting & Analysis
- Comprehensive backtesting engine with multiple metrics
- Performance evaluation (accuracy, precision, recall, F1-score, Sharpe ratio)
- Support for all forecasting methods

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

## Project Structure

```
fractime/
├── core/              # Core functionality
│   ├── hurst.py       # Hurst exponent calculation
│   ├── returns.py     # Returns calculation
│   └── ttw.py         # Trading Time Warping
├── forecasters/       # Forecasting models
│   ├── rs_forecaster.py           # R/S based forecaster
│   ├── st_frsr.py                 # State-based forecaster
│   ├── fractal_projections.py     # Pattern-based forecaster
│   ├── fractal_classification.py  # ML classification forecaster
│   └── ensemble.py                # Ensemble methods
├── backtest/          # Backtesting framework
│   ├── engine.py      # Backtesting engine
│   └── metrics.py     # Performance metrics
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
