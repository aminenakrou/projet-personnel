import pytest
import pandas as pd
from src.data_cleaning import preprocess_data

def test_preprocess_data():
    df = pd.DataFrame({"feature1": [0.4, 0.6], "feature2": [1, 2], "feature3": [3, 5], "churn": [0,1]})
    X, y = preprocess_data(df)
    assert X.shape[0] == df.shape[0]
    assert len(y) == df.shape[0]
