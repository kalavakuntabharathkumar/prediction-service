from pathlib import Path
import joblib
import pandas as pd
from .config import settings
from .store import save_prediction

_model_bundle = None

def load_model():
    global _model_bundle
    if _model_bundle is None:
        path = Path(settings.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}. Train it first.")
        _model_bundle = joblib.load(path)
    return _model_bundle

def predict_one(features: dict, customer_id: str | None = None):
    bundle = load_model()
    pipeline = bundle["pipeline"]
    threshold = float(bundle.get("threshold", settings.threshold))
    frame = pd.DataFrame([features])
    probability = float(pipeline.predict_proba(frame)[0, 1])
    prediction = int(probability >= threshold)
    log_id = save_prediction(customer_id, probability, prediction)
    return {"customer_id": customer_id, "churn_probability": round(probability, 6),
            "will_churn": bool(prediction), "threshold": threshold, "prediction_id": log_id}
