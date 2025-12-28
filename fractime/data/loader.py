import yfinance as yf 

def load_data(ticker_symbol, start_date, end_date, interval):
    try:
        data = yf.download(ticker_symbol, start=start_date, end=end_date, interval=interval, group_by='ticker')
        data.columns = data.columns.droplevel(0) 

    except Exception as e:
        raise ValueError(f"Failed to download {ticker_symbol}: {e}")    
    
    if data.empty:
        raise ValueError(f"No data found for {ticker_symbol} in the given date range.")  
    return data