import pandas as pd

def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize common telecom CSV quirks without leaking the target."""
    data = frame.copy()
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
        data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())
    for column in data.select_dtypes(include="object").columns:
        data[column] = data[column].fillna("Unknown").astype(str).str.strip()
    for column in data.select_dtypes(include="number").columns:
        data[column] = data[column].fillna(data[column].median())
    return data.drop(columns=["Churn"], errors="ignore")
