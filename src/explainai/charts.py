from typing import Union, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_feature_importance(
    shap_values: Union[np.ndarray, pd.DataFrame, Any], 
    feature_names: List[str]
) -> Figure:
    """
    Create a simple bar chart of mean absolute SHAP values to visualize feature importance.

    Args:
        shap_values (Union[np.ndarray, pd.DataFrame, Any]): 
            SHAP values calculated for the features. Can be a numpy array, pandas DataFrame,
            or an object with a `.values` attribute.
        feature_names (List[str]): 
            List of names corresponding to the features in `shap_values`.

    Returns:
        Figure: The matplotlib Figure object containing the plot.

    Raises:
        ValueError: If the number of feature names does not match the number of columns in shap_values.
        TypeError: If inputs are of incorrect types.
    """

    # If DataFrame → convert to NumPy
    if hasattr(shap_values, "values"):
        shap_arr = shap_values.values
    elif isinstance(shap_values, np.ndarray):
        shap_arr = shap_values
    elif isinstance(shap_values, list):
         shap_arr = np.array(shap_values)
    else:
         try:
             shap_arr = np.array(shap_values)
         except Exception as e:
             raise TypeError(f"Could not convert shap_values to numpy array: {e}")

    # Validation: Check dimensions
    # shap_arr expected shape: (n_samples, n_features)
    if shap_arr.ndim != 2:
         raise ValueError(f"shap_values must be 2-dimensional (samples, features). Got shape {shap_arr.shape}.")
         
    num_features = shap_arr.shape[1]
    if len(feature_names) != num_features:
        raise ValueError(
            f"Length of feature_names ({len(feature_names)}) does not match "
            f"number of features in shap_values ({num_features})."
        )

    # Compute mean absolute SHAP values
    mean_abs = np.mean(np.abs(shap_arr), axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(feature_names, mean_abs)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance (SHAP)")

    fig.tight_layout()
    return fig
