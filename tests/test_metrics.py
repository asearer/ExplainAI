import numpy as np
from explainai.metrics import metric_fidelity, metric_consistency
from explainai.models.example_model import train_example_model
from explainai.explain import compute_shap_values


def test_metrics():
    model, X_test, _ = train_example_model()
    shap_vals = compute_shap_values(model, X_test[:20])

    fidelity = metric_fidelity(model, X_test[:20], shap_vals)
    consistency = metric_consistency(shap_vals, shap_vals)

    assert -1 <= fidelity <= 1
    assert consistency == 1.0
