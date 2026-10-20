from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .service import predict_one
from .store import get_summary

app = FastAPI(title="Customer Churn Prediction Service", version="1.0.0")

class PredictionRequest(BaseModel):
    customer_id: str | None = None
    features: dict = Field(min_length=1)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: PredictionRequest):
    try:
        return predict_one(payload.features, payload.customer_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

@app.get("/metrics/summary")
def summary():
    return get_summary()
