import json
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from .config import settings

engine = create_engine(settings.database_url, future=True)

def initialize():
    with engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT,
                churn_probability REAL NOT NULL,
                prediction INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """))

def save_prediction(customer_id: str | None, probability: float, prediction: int):
    initialize()
    with engine.begin() as connection:
        result = connection.execute(text("""
            INSERT INTO prediction_logs(customer_id, churn_probability, prediction, created_at)
            VALUES (:customer_id, :probability, :prediction, :created_at)
        """), {"customer_id": customer_id, "probability": probability, "prediction": prediction,
               "created_at": datetime.now(timezone.utc).isoformat()})
        return int(result.lastrowid)

def get_summary():
    initialize()
    with engine.connect() as connection:
        row = connection.execute(text("""
            SELECT COUNT(*) AS total, AVG(churn_probability) AS avg_probability,
                   SUM(prediction) AS flagged
            FROM prediction_logs
        """)).mappings().one()
    return dict(row)
