from typing import Any, Union
import numpy as np
import pandas as pd


def metric_fidelity(
    model: Any, 
    X: Union[np.ndarray, pd.DataFrame], 
    shap_values: Union[np.ndarray, pd.DataFrame]
) -> float:
    """
    Computes a fidelity metric by correlating model predictions with the sum of SHAP values.
    
    Assumption: For many additive feature attribution methods (like SHAP), 
    the sum of attribution values + expected_value should approximate the prediction.
    Here we check correlation between (sum of shap values) and (model predictions).

    Args:
        model (Any): The model which has a `predict` method.
        X (Union[np.ndarray, pd.DataFrame]): Input features.
        shap_values (Union[np.ndarray, pd.DataFrame]): Computed SHAP values.

    Returns:
        float: Pearson correlation coefficient between predictions and reconstructed values.

    Raises:
        ValueError: If input dimensions mismatch.
    """
    # Standardize inputs
    if hasattr(X, "values"): X = X.values
    if hasattr(shap_values, "values"): shap_values = shap_values.values
    
    if len(X) != len(shap_values):
        raise ValueError(f"X has {len(X)} samples but shap_values has {len(shap_values)} samples.")

    original_preds = model.predict(X)
    
    # If model.predict returns probabilities (multi-class), needed to flatten or select class.
    # For simplicity, if multi-dim, we flatten to check overall correlation or take max.
    # Here we assume regression or binary clf outcome for basic fidelity.
    if isinstance(original_preds, np.ndarray) and original_preds.ndim > 1:
        # Flatten for rudimentary correlation check
        original_preds = original_preds.flatten()
        # If predictions flattened, shap sum implies we sum over features but still have (samples, classes)
        # This is complex, so for this simple metric we might just sum SHAP over features.
    
    # Sum SHAP values across features
    reconstructed = shap_values.sum(axis=1)
    
    # Robustness: ensure shapes align
    if original_preds.shape != reconstructed.shape:
         # Attempt to match simple cases
         if original_preds.size == reconstructed.size:
             original_preds = original_preds.flatten()
             reconstructed = reconstructed.flatten()
         else:
             # Just return 0.0 or raise error? Raising error is safer for "robustness".
             raise ValueError(f"Shape mismatch: preds {original_preds.shape} vs reconstructed {reconstructed.shape}")

    # Avoid NaN correlation if constant
    if np.std(original_preds) == 0 or np.std(reconstructed) == 0:
        return 0.0

    return float(np.corrcoef(original_preds, reconstructed)[0, 1])


def metric_consistency(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """
    Computes the consistency metric: correlation between two sets of SHAP values.
    Useful to check if the explanation is stable across runs or slight data perturbations.

    Args:
        values_a (np.ndarray): First set of SHAP values.
        values_b (np.ndarray): Second set of SHAP values.

    Returns:
        float: Pearson correlation coefficient between flattened arrays.
    """
    a_flat = np.array(values_a).flatten()
    b_flat = np.array(values_b).flatten()

    if len(a_flat) != len(b_flat):
         raise ValueError(f"Input arrays have different sizes: {len(a_flat)} vs {len(b_flat)}")

    if np.std(a_flat) == 0 or np.std(b_flat) == 0:
        return 0.0

    return float(np.corrcoef(a_flat, b_flat)[0, 1])

