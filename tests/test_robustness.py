import pytest
import numpy as np
import pandas as pd
from explainai.helpers import load_dataset, to_numpy
from explainai.charts import plot_feature_importance
from explainai.metrics import metric_fidelity, metric_consistency

def test_load_dataset_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_dataset("non_existent_file.csv")

def test_to_numpy_valid_inputs():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    ser = pd.Series([1, 2, 3])
    lst = [1, 2, 3]
    arr = np.array([1, 2, 3])
    
    assert isinstance(to_numpy(df), np.ndarray)
    assert isinstance(to_numpy(ser), np.ndarray)
    assert isinstance(to_numpy(lst), np.ndarray)
    assert isinstance(to_numpy(arr), np.ndarray)

def test_to_numpy_invalid_input():
    with pytest.raises(TypeError):
        to_numpy("invalid")

def test_plot_feature_importance_mismatch_features():
    shap_values = np.random.rand(10, 5) # 5 features
    feature_names = ["f1", "f2", "f3"] # 3 names
    
    with pytest.raises(ValueError, match="Length of feature_names"):
        plot_feature_importance(shap_values, feature_names)

def test_metric_fidelity_shape_mismatch():
    # Mock model
    class MockModel:
        def predict(self, X):
            return np.zeros(len(X))
            
    model = MockModel()
    X = np.random.rand(10, 5)
    shap_values = np.random.rand(5, 5) # Mismatch sample count
    
    with pytest.raises(ValueError, match="X has 10 samples but shap_values has 5"):
        metric_fidelity(model, X, shap_values)

def test_metric_consistency_size_mismatch():
    a = np.random.rand(10)
    b = np.random.rand(5)
    
    with pytest.raises(ValueError, match="Input arrays have different sizes"):
        metric_consistency(a, b)
