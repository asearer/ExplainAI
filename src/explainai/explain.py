import shap
import numpy as np


def compute_shap_values(model, X):
    """
    Computes SHAP values for a model and dataset X.
    Works with tree-based or model-agnostic explainers.
    """
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.Explainer(model.predict, X)

    shap_vals = explainer(X)

    return np.array(shap_vals.values)
