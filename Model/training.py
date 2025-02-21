import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

def train_and_evaluate(X, y):
    """Trains a model to predict stock movement (up/down)."""
    
    # Convert target into binary classification (1 = up, 0 = down)
    y_binary = np.where(y > X["4. close"], 1, 0)

    # Ensure X and y_binary are aligned
    X, y_binary = X.iloc[:-1], y_binary[:-1]

    print(f"Training Data Shape - X: {X.shape}, y: {y_binary.shape}")  # Debugging

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, shuffle=False, random_state=42)

    # Train model using XGBClassifier (not Booster)
    model = XGBClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # Cross-validation
    cv_scores = cross_val_score(model, X, y_binary, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.4f}")

    return model