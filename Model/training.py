import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

def train_and_evaluate(X, y):
    """Trains an XGBoost model with sentiment analysis and hyperparameter tuning."""
    
    # Convert target into binary classification (1 = up, 0 = down)
    y_binary = np.where(y > X["4. close"], 1, 0)

    # Ensure X and y_binary are aligned
    X, y_binary = X.iloc[:-1], y_binary[:-1]

    print(f"Training Data Shape - X: {X.shape}, y: {y_binary.shape}")  # Debugging

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, shuffle=False, random_state=42)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Compute class weights
    classes = np.unique(y_binary)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_binary)
    weights_dict = {c: w for c, w in zip(classes, class_weights)}

    # **Hyperparameter Tuning with GridSearchCV**
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    grid_search = GridSearchCV(XGBClassifier(scale_pos_weight=weights_dict[1]), 
                               param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print(f"Best Parameters: {grid_search.best_params_}")

    # Evaluate the best model
    y_pred = best_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # Cross-validation
    cv_scores = cross_val_score(best_model, X, y_binary, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.4f}")

    return best_model
