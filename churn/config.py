import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./churn.db")
    model_path: str = os.getenv("MODEL_PATH", "artifacts/churn_pipeline.joblib")
    threshold: float = float(os.getenv("CHURN_THRESHOLD", "0.5"))

settings = Settings()
