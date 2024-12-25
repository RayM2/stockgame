import yfinance as yf
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from model import StockPerformanceModel

def fetch_stock_data(ticker, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)  # Reset index to include 'Date' as a column
    return data[['Date', 'Close']]  # Select relevant columns

def fetch_index_data(index_symbol, start_date, end_date):
    """Fetch historical index data."""
    data = yf.download(index_symbol, start=start_date, end=end_date)
    data.reset_index(inplace=True)  # Reset index to include 'Date' as a column
    return data[['Date', 'Close']]  # Select relevant columns

def calculate_index_performance(index_data):
    """Calculate performance metrics for the S&P 500 index."""
    # Assuming 'Close' column exists
    initial_price = index_data['Close'].iloc[0]
    final_price = index_data['Close'].iloc[-1]
    percent_return = (final_price - initial_price) / initial_price * 100
    return percent_return

def evaluate_model_performance(tickers, start_date, end_date):
    """Evaluate model performance on a list of tickers."""
    predicted_prices = []
    actual_prices = []
    
    for ticker in tickers:
        try:
            data = fetch_stock_data(ticker, start_date, end_date)
            if data.empty:
                continue
            
            look_back = 60
            model = StockPerformanceModel(look_back=look_back)
            model.train(data)
            future_price = model.predict(data)
            actual_future_price = data['Close'].iloc[-1]
            
            predicted_prices.append(future_price)
            actual_prices.append(actual_future_price)
        except ValueError:
            continue
    
    if len(predicted_prices) == 0:
        raise ValueError("No valid predictions were made.")
    
    mape = mean_absolute_percentage_error(actual_prices, predicted_prices) * 100
    return mape

if __name__ == "__main__":
    tickers = fetch_sp500_tickers()  # Define or fetch the list of S&P 500 tickers
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    # Evaluate model performance
    model_mape = evaluate_model_performance(tickers, start_date, end_date)
    model_accuracy = 100 - model_mape
    print(f"Model MAPE: {model_mape:.2f}%")
    print(f"Model Accuracy: {model_accuracy:.2f}%")
    
    # Fetch and evaluate S&P 500 index performance
    sp500_data = fetch_index_data('^GSPC', start_date, end_date)
    sp500_percent_return = calculate_index_performance(sp500_data)
    print(f"S&P 500 Percent Return: {sp500_percent_return:.2f}%")
