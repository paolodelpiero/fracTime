import pandas as pd
from fractime.data.loader import load_data
from fractime.core.returns import log_returns
from fractime.core.hurst import hurst_exponent
from fractime.backtest.engine import full_backtest
from fractime.forecasters.fractal_projections import fractal_projection_forecast

def scan_asset(ticker, start_date, end_date, interval='1d'):
    """
    Analyzes a single asset and returns metrics.
    """
    try:
        data = load_data(ticker, start_date, end_date, interval)
        returns = log_returns(data['Close'])
        if len(returns) < 300:
            return None
        
        H = hurst_exponent(returns.values)
 
        metrics = full_backtest(
            returns, 
            fractal_projection_forecast, 
            min_history=100,
            pattern_length=20, 
            min_similarity=0.5
        )
        metrics['ticker'] = ticker
        metrics['hurst'] = H
        metrics['data_points'] = len(returns)
        
        return metrics
        
    except Exception as e:
        print(f"Error for {ticker}: {e}")
        return None

def scan_multiple_assets(tickers, start_date, end_date, interval='1d'):
    """
    Scans multiple assets and returns DataFrame sorted by Sharpe ratio.
    """
    results = []
    
    for i, ticker in enumerate(tickers):
        print(f"Scanning {ticker} ({i+1}/{len(tickers)})...")
        metrics = scan_asset(ticker, start_date, end_date, interval)
        if metrics:
            results.append(metrics)
    
    df = pd.DataFrame(results)
    df = df.sort_values('sharpe_ratio', ascending=False)
    
    return df