import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

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

class StockPerformanceModel:
    def __init__(self, look_back=60):
        self.look_back = look_back
        self.model = None
        self.scaler = MinMaxScaler()

    def preprocess_data(self, data):
        data = self.scaler.fit_transform(data[['Close']])
        X, y = [], []
        for i in range(len(data) - self.look_back):
            X.append(data[i:i + self.look_back])
            y.append(data[i + self.look_back])
        X, y = np.array(X), np.array(y)
        return X, y

    def train(self, data):
        X, y = self.preprocess_data(data)
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.look_back, 1)),
            LSTM(50),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    def predict(self, data):
        data_scaled = self.scaler.transform(data[['Close']])
        X_test = [data_scaled[-self.look_back:]]
        X_test = np.array(X_test)
        predicted_scaled = self.model.predict(X_test)
        predicted = self.scaler.inverse_transform(predicted_scaled)
        return predicted[0, 0]

    def evaluate_model(model, test_data):
        # Extract the actual values
        actual_prices = test_data['Close'].values
        
        # Scale and preprocess the test data
        data_scaled = model.scaler.transform(test_data[['Close']])
        X_test, y_test = [], []
        for i in range(len(data_scaled) - model.look_back):
            X_test.append(data_scaled[i:i + model.look_back])
            y_test.append(data_scaled[i + model.look_back])
        X_test, y_test = np.array(X_test), np.array(y_test)

        # Predict the scaled values
        predicted_scaled = model.model.predict(X_test)
        
        # Inverse transform the predicted and actual values
        predicted_prices = model.scaler.inverse_transform(predicted_scaled)
        actual_prices = model.scaler.inverse_transform(y_test.reshape(-1, 1))

        # Calculate MAPE
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

        # Calculate Directional Accuracy
        predicted_directions = np.sign(np.diff(predicted_prices.flatten()))
        actual_directions = np.sign(np.diff(actual_prices.flatten()))
        directional_accuracy = np.mean(predicted_directions == actual_directions) * 100

        return mape, directional_accuracy
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