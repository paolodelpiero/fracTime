from fractime.forecasters.rs_forecaster import rs_forecast
from fractime.forecasters.st_frsr import identify_states, calculate_transition_matrix
from fractime.forecasters.fractal_projections import fractal_projection_forecast
from fractime.backtest.metrics import (
    calculate_rmse,
    calculate_mae,
    calculate_sharpe_ratio,
    calculate_directional_accuracy,
)
import numpy as np


def rs_backtest(returns, window_hurst, window_trend):
    correct = 0
    total = 0

    for i in range(window_hurst, len(returns) - 1):

        past_returns = returns[:i]

        forecast = rs_forecast(past_returns, window_hurst, window_trend)

        actual = returns.iloc[i]

        if (forecast > 0 and actual > 0) or (forecast < 0 and actual < 0):
            correct += 1
        total += 1

    accuracy = correct / total
    return accuracy


def st_frsr_backtest(
    returns, window_short=5, window_long=20, n_states=3, min_history=200
):
    correct = 0
    total = 0

    for i in range(min_history, len(returns) - 1):
        past_returns = returns[:i]
        states, kmeans, scaler = identify_states(
            past_returns, n_states, window_short, window_long
        )
        transition_matrix = calculate_transition_matrix(states, n_states)
        current_state = int(states.dropna().iloc[-1])
        next_state_probs = transition_matrix[current_state]
        predicted_state = np.argmax(next_state_probs)
        mean_returns_per_state = past_returns.groupby(states).mean()
        forecast = mean_returns_per_state[predicted_state]

        actual = returns.iloc[i]

        if (forecast > 0 and actual > 0) or (forecast < 0 and actual < 0):
            correct += 1
        total += 1

    accuracy = correct / total
    return accuracy


def fractal_projections_backtest(returns, window_hurst, window_trend):
    correct = 0
    total = 0

    for i in range(window_hurst, len(returns) - 1):

        past_returns = returns[:i]

        forecast = fractal_projection_forecast(
            past_returns, pattern_length=window_trend, min_similarity=0.7, lookahead=1
        )

        actual = returns.iloc[i]

        if (forecast > 0 and actual > 0) or (forecast < 0 and actual < 0):
            correct += 1
        total += 1

    accuracy = correct / total
    return accuracy


def full_backtest(returns, forecaster_func, min_history=500, **forecaster_params):

    predictions = []
    actuals = []
    strategy_returns = []  # rendimenti della strategia

    for i in range(min_history, len(returns) - 1):
        past_returns = returns.iloc[:i]

        try:
            forecast = forecaster_func(past_returns, **forecaster_params)
        except Exception:
            continue

        actual = returns.iloc[i]

        # Salva previsione e reale
        predictions.append(forecast)
        actuals.append(actual)

        # Calcola rendimento strategia: se prevedi positivo, vai long; se negativo, vai short
        if forecast > 0:
            strategy_returns.append(actual)  # long
        elif forecast < 0:
            strategy_returns.append(-actual)  # short
        else:
            strategy_returns.append(0)  # no trade

    # Converti in array numpy
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    strategy_returns = np.array(strategy_returns)

    # Calcola metriche
    metrics = {
        "directional_accuracy": calculate_directional_accuracy(actuals, predictions),
        "rmse": calculate_rmse(actuals, predictions),
        "mae": calculate_mae(actuals, predictions),
        "sharpe_ratio": calculate_sharpe_ratio(strategy_returns),
        "total_return": np.sum(strategy_returns),
        "n_trades": len(predictions),
    }

    return metrics
