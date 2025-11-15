import numpy as np
from explainai.explain import compute_shap_values
from explainai.models.example_model import train_example_model


def test_compute_shap_values():
    model, X_test, _ = train_example_model()
    shap_vals = compute_shap_values(model, X_test[:10])

    assert isinstance(shap_vals, np.ndarray)
    assert shap_vals.shape[0] == 10
