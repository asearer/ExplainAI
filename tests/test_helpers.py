from explainai.helpers import to_numpy
import pandas as pd


def test_to_numpy():
    df = pd.DataFrame({"a": [1, 2, 3]})
    arr = to_numpy(df)
    assert arr.shape == (3, 1)
