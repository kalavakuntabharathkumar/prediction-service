import pandas as pd
from churn.features import clean_features

def test_clean_features_converts_total_charges_and_drops_target():
    frame = pd.DataFrame({"TotalCharges": ["100.5", ""], "Contract": [" Month-to-month ", None],
                          "Churn": [1, 0]})
    cleaned = clean_features(frame)
    assert "Churn" not in cleaned
    assert cleaned["TotalCharges"].notna().all()
    assert cleaned.loc[0, "Contract"] == "Month-to-month"
