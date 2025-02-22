import yfinance as yf
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import requests
from sentiment_analyzer import analyze_sentiment  # Import sentiment function


load_dotenv()
API_KEY = os.getenv("API_KEY")  # Not needed for yfinance but can be useful for other APIs

def fetch_stock_data(symbol):
    """Fetches stock data and handles API issues."""
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}'

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if 'Time Series (Daily)' not in data:
            print(f"API Error for {symbol}: {data}")
            return None

        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
        df = df.astype(float)  # Convert to float
        df['symbol'] = symbol
        return df

    except requests.exceptions.RequestException as e:
        print(f"Request Error for {symbol}: {e}")
        return None

def fetch_sentiment_data(symbol):
    """Dummy function: Replace with real scraping of news & tweets for sentiment."""
    sample_texts = [
        "Stock is performing well!",
        "Market is crashing, huge losses!",
        "Analysts predict steady growth."
    ]
    
    sentiments = [analyze_sentiment(text, source="news") for text in sample_texts]  # Change to 'social' for tweets
    avg_sentiment = np.mean(sentiments)  # Get average sentiment score

    return avg_sentiment

def prepare_data(symbols):
    """Fetch stock data and integrate sentiment scores."""
    all_data = []

    for symbol in symbols:
        df = fetch_stock_data(symbol)
        if df is not None:
            sentiment_score = fetch_sentiment_data(symbol)  # Get sentiment score
            df["sentiment"] = sentiment_score  # Add to DataFrame
            print(f"Fetched {symbol}: {df.shape}")  # Debugging line
            all_data.append(df)

    if not all_data:
        print("Error: No valid stock data fetched.")
        return None, None

    full_df = pd.concat(all_data, ignore_index=True)

    # Ensure the required columns exist
    if "4. close" not in full_df.columns:
        print("Error: '4. close' column missing in fetched data.")
        return None, None

    # Convert column names to lowercase (if needed)
    full_df.columns = [col.lower() for col in full_df.columns]

    # Features (X) and target (y)
    full_df["SMA_10"] = full_df["4. close"].rolling(window=10).mean()
    full_df["SMA_50"] = full_df["4. close"].rolling(window=50).mean()
    full_df["Momentum"] = full_df["4. close"].diff(3)
    full_df["Volatility"] = full_df["4. close"].rolling(window=10).std()
    full_df["sentiment_change"] = full_df["sentiment"].diff()
    full_df["sentiment_rolling"] = full_df["sentiment"].rolling(window=5).mean()

    # Fill NaN values
    full_df = full_df.fillna(0)

    # Ensure sentiment column exists
    if "sentiment" not in full_df.columns:
        full_df["sentiment"] = 0  # Default if missing

    X = full_df[["4. close", "SMA_10", "SMA_50", "Momentum", "Volatility", "sentiment", "sentiment_change", "sentiment_rolling"]]
    y = full_df["4. close"].shift(-1)

    # Drop the last row to ensure X and y have the same length
    X, y = X.iloc[:-1], y.iloc[:-1]

    print(f"Final Data Shape - X: {X.shape}, y: {y.shape}")  # Debugging line
    return X, y