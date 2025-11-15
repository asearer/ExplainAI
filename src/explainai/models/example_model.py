"""
Example model for ExplainAI.

Uses scikit-learn's built-in Diabetes dataset (no remote downloads)
so tests are stable and fast. Trains a small RandomForestRegressor
and returns both the model and feature metadata.
"""

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


def train_example_model():
    """
    Train a simple RandomForest model using the built-in diabetes dataset.

    Returns:
        model: Trained RandomForestRegressor
        X_df:  Pandas DataFrame of features
        feature_names: List of feature names
    """
    # Load dataset (local, no network calls)
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Small model to keep tests fast
    model = RandomForestRegressor(
        n_estimators=20,
        random_state=42,
    )

    model.fit(X, y)

    return model, X, data.feature_names


# Optional: allow quick standalone run
if __name__ == "__main__":
    model, X, features = train_example_model()
    print("Model trained successfully.")
    print(f"Feature count: {len(features)}")
