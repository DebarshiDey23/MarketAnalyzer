import requests
from dotenv import load_dotenv
import os
import pandas as pd

# Load API Key from .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Function to fetch and process stock data
def data_collection(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}'
    response = requests.get(url)
    data = response.json()

    # Check if data is valid
    if 'Time Series (Daily)' not in data:
        print("Error fetching data or invalid symbol.")
        return None

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
    df = df.astype(float)  # Convert strings to float for numerical operations

    # Calculate moving average (MA)
    df['SMA_50'] = df['4. close'].rolling(window=50).mean()

    # Calculate relative strength index (RSI)
    delta = df['4. close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Drop missing values
    df = df.dropna()

    return df
