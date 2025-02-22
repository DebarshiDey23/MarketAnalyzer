from datacollection import prepare_data
from training import train_and_evaluate

def main():
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    X, y = prepare_data(symbols)
    print("Columns in X:", X.columns)

    if X is None or y is None or X.empty or y.empty:
        print("Error: No valid data available for training.")
        return

    print(f"Data Shape - X: {X.shape}, y: {y.shape}")  # Debugging line
    model = train_and_evaluate(X, y)
    
    print("Training completed. Model is ready for predictions.")

if __name__ == "__main__":
    main()
