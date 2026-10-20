import pandas as pd
import pytest
from churn.data import validate_dataset

def test_target_normalized():
    frame = validate_dataset(pd.DataFrame({"tenure": [1, 2], "Churn": ["Yes", "No"]}))
    assert frame["Churn"].tolist() == [1, 0]

def test_invalid_target_rejected():
    with pytest.raises(ValueError):
        validate_dataset(pd.DataFrame({"Churn": ["maybe"]}))
