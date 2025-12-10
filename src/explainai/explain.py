import logging
from typing import Any, Union, Callable
import shap
import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def compute_shap_values(model: Any, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Computes SHAP values for a given model and dataset.

    Attempts to use TreeExplainer first (optimized for tree-based models),
    and falls back to the generic Explainer (or KernelExplainer/PermutationExplainer)
    if the model is not supported by TreeExplainer.

    Args:
        model (Any): The trained machine learning model to explain. 
                     Should have a `predict` method.
        X (Union[pd.DataFrame, np.ndarray]): The input features dataset.

    Returns:
        np.ndarray: The array of SHAP values corresponding to X.

    Raises:
        ValueError: If SHAP value computation fails.
    """
    logger.info("Starting SHAP value computation...")
    
    try:
        logger.debug("Attempting to use TreeExplainer.")
        explainer = shap.TreeExplainer(model)
        # TreeExplainer usually requires checking expected_value or similar for verification,
        # but here we just instantiate.
        shap_vals = explainer(X)
        logger.info("Successfully calculated SHAP values using TreeExplainer.")
        
    except Exception as tree_err:
        logger.warning(f"TreeExplainer failed: {tree_err}. Falling back to generic Explainer.")
        try:
            # Fallback to generic Explainer which selects the appropriate one (often Permutation or Partition)
            # We pass model.predict for black-box models if model itself isn't supported directly
            prediction_fn = getattr(model, "predict", model)
            explainer = shap.Explainer(prediction_fn, X)
            shap_vals = explainer(X)
            logger.info("Successfully calculated SHAP values using generic Explainer.")
            
        except Exception as gen_err:
            error_msg = f"Failed to compute SHAP values with both TreeExplainer and generic Explainer. Error: {gen_err}"
            logger.error(error_msg)
            raise ValueError(error_msg) from gen_err

    # shap_vals can be an Explanation object or array depending on the explainer and version.
    # We standardize to numpy array of values.
    if hasattr(shap_vals, "values"):
        return np.array(shap_vals.values)
    
    return np.array(shap_vals)

