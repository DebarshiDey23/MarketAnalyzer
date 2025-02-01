from datacollection import data_collection
from training import prepare_data, train_and_evaluate

def main(symbol):
    # Get the data for the given stock symbol
    df = data_collection(symbol)

    if df is None:
        return

    # Prepare the features and target
    X, y = prepare_data(df)

    # Train and evaluate the model
    train_and_evaluate(X, y)

# Call the main function with a stock symbol (e.g., 'AAPL' for Apple)
main('AAPL')
