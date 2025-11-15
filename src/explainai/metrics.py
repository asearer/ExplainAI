import numpy as np


def metric_fidelity(model, X, shap_values):
    """
    A simple interpretability fidelity metric:
    Reconstruct model predictions from SHAP values.
    """
    original_preds = model.predict(X)
    reconstructed = shap_values.sum(axis=1)

    return np.corrcoef(original_preds, reconstructed)[0, 1]


def metric_consistency(values_a, values_b):
    """
    Consistency metric: correlation between two SHAP runs.
    """
    return np.corrcoef(values_a.flatten(), values_b.flatten())[0, 1]
