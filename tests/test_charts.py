from explainai.charts import plot_feature_importance
from explainai.models.example_model import train_example_model
from explainai.explain import compute_shap_values


def test_plot_importance():
    model, X_test, feat_names = train_example_model()
    shap_vals = compute_shap_values(model, X_test[:10])

    fig = plot_feature_importance(shap_vals, feat_names)

    assert fig is not None
