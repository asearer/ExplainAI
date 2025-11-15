import numpy as np
import matplotlib.pyplot as plt


def plot_feature_importance(shap_values, feature_names):
    """
    Create a simple bar chart of mean absolute SHAP values.
    Accepts either NumPy arrays or pandas DataFrames.
    """

    # If DataFrame → convert to NumPy
    if hasattr(shap_values, "values"):
        shap_arr = shap_values.values
    else:
        shap_arr = np.array(shap_values)

    # Compute mean absolute SHAP values
    mean_abs = np.mean(np.abs(shap_arr), axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(feature_names, mean_abs)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance (SHAP)")

    fig.tight_layout()
    return fig
