from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Function to prepare the data for training
def prepare_data(df):
    # Define features (X) and target (y)
    X = df[['4. close', 'SMA_50', 'RSI']]  # Example: using closing price, 50-day SMA, and RSI
    y = df['4. close'].shift(-1)  # Predict the next day's closing price

    # Drop the last row because the shifted target will have NaN for the last row
    X = X[:-1]
    y = y[:-1]

    return X, y

# Function to train and evaluate the model
def train_and_evaluate(X, y):
    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Plot actual vs predicted values
    plt.plot(y_test.index, y_test, label='Actual')
    plt.plot(y_test.index, y_pred, label='Predicted')
    plt.legend()
    plt.show()
