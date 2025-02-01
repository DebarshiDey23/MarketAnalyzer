from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def train_and_evaluate(X, y):
    """Trains and evaluates a stock movement prediction model with multiple stocks."""

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    # Initialize Random Forest Classifier
    rf = RandomForestClassifier(random_state=42)

    # Hyperparameter tuning using RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=rf, param_distributions=param_grid,
        n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)

    # Get best model
    best_model = search.best_estimator_

    # Predict on test set
    y_pred = best_model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Best Model Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    # Cross-validation
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.4f}")

    return best_model
