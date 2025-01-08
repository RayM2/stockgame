import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, look_back=250):  # Fetch 250 trading days (~1 year)
    try:
        stock = yf.Ticker(ticker)

        # Adjust the start date to include more historical data
        today = datetime.today()
        adjusted_start_date = today - timedelta(days=look_back)

        # Fetch the stock data
        data = stock.history(
            start=adjusted_start_date.strftime('%Y-%m-%d'),
            end=today.strftime('%Y-%m-%d')
        )

        if data.empty:
            raise ValueError(f"No data found for the training period: {adjusted_start_date} to {today}")

        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

class StockPerformanceModel:
    def __init__(self, look_back=60, n_estimators=100):
        self.look_back = look_back
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        self.scaler = MinMaxScaler()

    def preprocess_data(self, data):
        # Scale the 'Close' prices between 0 and 1
        data_scaled = self.scaler.fit_transform(data[['Close']])
        X, y = [], []

        # Create the look-back dataset
        for i in range(len(data_scaled) - self.look_back):
            X.append(data_scaled[i:i + self.look_back].flatten())  # Flatten for RandomForest
            y.append(data_scaled[i + self.look_back][0])  # Next price as target

        return np.array(X), np.array(y)

    def train(self, data):
        # Preprocess data and fit the model
        X, y = self.preprocess_data(data)
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Insufficient data for training.")
        self.model.fit(X, y)

    def predict(self, X):
        # Predict and inverse transform the prediction
        predicted_scaled = self.model.predict(X.reshape(1, -1))[0]
        return self.scaler.inverse_transform([[predicted_scaled]])[0, 0]

# Workflow
if __name__ == "__main__":
    # Parameters
    ticker = "AAPL"  # Replace with desired stock ticker
    start_date = "2020-01-01"
    end_date = "2023-01-01"

    # Fetch data
    data = fetch_stock_data(ticker, start_date, end_date)

    if not data.empty:
        # Split data into training and testing sets
        train_data = data[:int(0.8 * len(data))]
        test_data = data[int(0.8 * len(data)):]

        # Initialize and train the model
        model = StockPerformanceModel()
        model.train(train_data)

        # Evaluate the model
        mape, directional_accuracy = model.evaluate_model(model, test_data)

        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"Directional Accuracy: {directional_accuracy:.2f}%")