import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def fetch_stock_data(ticker, look_back=120):
    try:
        stock = yf.Ticker(ticker)

        # Determine the most recent available date
        today = datetime.today()
        adjusted_start_date = today - timedelta(days=look_back)

        # Fetch the last 60 days of data
        data = stock.history(
            start=adjusted_start_date.strftime('%Y-%m-%d'),
            end=today.strftime('%Y-%m-%d')
        )

        if data.empty:
            raise ValueError(f"No data found for the training period: {adjusted_start_date} to {today}")

        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if there's an error