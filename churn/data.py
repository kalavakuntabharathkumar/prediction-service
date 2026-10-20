from pathlib import Path
import pandas as pd

TARGET = "Churn"

def load_dataset(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if TARGET not in frame.columns:
        raise ValueError(f"Dataset must contain target column {TARGET!r}")
    return frame

def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Dataset is empty")
    if TARGET not in frame:
        raise ValueError(f"Missing target column: {TARGET}")
    frame = frame.copy()
    frame[TARGET] = frame[TARGET].astype(str).str.strip().str.lower().map(
        {"yes": 1, "no": 0, "1": 1, "0": 0}
    )
    if frame[TARGET].isna().any():
        raise ValueError("Target must contain only Yes/No or 1/0 values")
    return frame
