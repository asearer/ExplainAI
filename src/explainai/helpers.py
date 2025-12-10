import os
from typing import Union, List
import numpy as np
import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.

    Args:
        path (str): The absolute or relative path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the file at `path` does not exist.
        ValueError: If the file is not a valid CSV or is empty.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file at {path} was not found.")
    
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
         raise ValueError(f"The file at {path} is empty.")
    except Exception as e:
        raise ValueError(f"Failed to load dataset from {path}: {str(e)}")
        
    return df


def to_numpy(data: Union[pd.DataFrame, pd.Series, List, np.ndarray]) -> np.ndarray:
    """
    Convert various data structures to a NumPy array.

    Args:
        data (Union[pd.DataFrame, pd.Series, List, np.ndarray]): 
            Input data to convert. Can be a DataFrame, Series, python list, or existing array.

    Returns:
        np.ndarray: The data converted to a NumPy array.

    Raises:
        TypeError: If the input data type is not supported.
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.values
    elif isinstance(data, list):
        return np.array(data)
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Expected DataFrame, Series, list, or ndarray.")

