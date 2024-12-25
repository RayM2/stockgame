import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            raise ValueError("No data found for the given date range.")
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()
