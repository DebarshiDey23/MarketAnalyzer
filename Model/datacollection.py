import requests
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()
API_KEY = os.getenv("API_KEY")

def fetch_stock_data(symbol):
    """Fetch historical data for a given stock symbol from Alpha Vantage."""
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}&outputsize=full'
    response = requests.get(url)
    data = response.json()

    if "Time Series (Daily)" not in data:
        print(f"Error fetching data for {symbol}.")
        return None

    df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
    df = df.astype(float)
    
    # Add stock symbol as a column
    df['symbol'] = symbol
    return df

def data_collection(symbols):
    """Fetch and combine historical stock data for multiple symbols."""
    all_data = []

    for symbol in symbols:
        df = fetch_stock_data(symbol)
        if df is not None:
            all_data.append(df)

    # Combine all stock data
    combined_df = pd.concat(all_data, axis=0).reset_index()
    combined_df.rename(columns={'index': 'date'}, inplace=True)
    
    # Sort by date and symbol
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df.sort_values(by=['date', 'symbol'], inplace=True)
    
    return combined_df

def prepare_data(df):
    """Prepare feature engineering for stock price movement prediction."""
    # Convert date to numerical format
    df['date'] = df['date'].map(pd.Timestamp.toordinal)

    # Compute Moving Averages and RSI
    df['SMA_50'] = df.groupby('symbol')['4. close'].transform(lambda x: x.rolling(window=50).mean())
    
    delta = df['4. close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Drop NaN values after feature engineering
    df.dropna(inplace=True)

    # Define features (X) and target (y)
    X = df[['date', '4. close', 'SMA_50', 'RSI']]
    y = (df['4. close'].shift(-1) > df['4. close']).astype(int)  # 1 if price goes up, 0 if down

    # Drop last row (since it has no future value)
    X = X[:-1]
    y = y[:-1]

    return X, y
