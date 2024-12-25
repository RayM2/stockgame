import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import json


def preprocess_data(data):
    # Fill missing values
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    # Normalize 'Close' prices and 'Volume'
    scaler_close = MinMaxScaler()
    scaler_volume = MinMaxScaler()
    data['Close_normalized'] = scaler_close.fit_transform(data[['Close']])
    data['Volume_normalized'] = scaler_volume.fit_transform(data[['Volume']])
    
    # Feature engineering: adding moving averages as features
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    
    # Adding lag features
    for lag in range(1, 6):
        data[f'Lag_{lag}'] = data['Close'].shift(lag)
    
    # Adding time-based feature: number of days since start
    data['Days'] = (data.index - data.index[0]).days
    
    # Fill any NaN values that may have resulted from moving averages and lags
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    return data, scaler_close



def save_portfolio(portfolio, filename="portfolio.json"):
    with open(filename, 'w') as f:
        json.dump(portfolio, f)

def load_portfolio(filename="portfolio.json"):
    try:
        with open(filename, 'r') as f:
            portfolio = json.load(f)
            return portfolio
    except FileNotFoundError:
        return {}
