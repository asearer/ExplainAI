from explainai.models.example_model import train_example_model


def test_example_model():
    model, X_test, features = train_example_model()

    assert len(features) == X_test.shape[1]
    assert hasattr(model, "predict")
