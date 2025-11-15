import numpy as np
import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    return pd.read_csv(path)


def to_numpy(data):
    """Convert pandas DataFrame or list to numpy array."""
    if isinstance(data, pd.DataFrame):
        return data.values
    return np.array(data)
