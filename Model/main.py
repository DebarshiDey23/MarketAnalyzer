from datacollection import data_collection, prepare_data
from training import train_and_evaluate

def main(symbols):
    """Runs the stock prediction model on multiple tech stocks."""
    df = data_collection(symbols)

    if df is None or df.empty:
        print("No data collected.")
        return

    X, y = prepare_data(df)

    # Train and evaluate the model
    train_and_evaluate(X, y)

# Define the tech stocks to train on
tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMZN']

# Run main function with multiple stocks
main(tech_stocks)
