# Customer Churn Prediction Service

A reproducible classical-ML workflow that compares Logistic Regression, Random Forest, and XGBoost with stratified cross-validation, selects a model by churn recall, logs runs to MLflow, and serves predictions through Flask. A FastAPI alternative is also included.

## Setup
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

## Generate smoke-test data and train
```bash
python scripts/make_sample_data.py
python -m churn.train --data data/sample_churn.csv
```
This generator creates **synthetic** data for checking the pipeline. For meaningful model performance, replace it with a properly licensed telecom churn dataset with the same `Churn` target column. Do not report metrics from synthetic data as benchmark results.

## Run the Flask API
```bash
python app.py
```
- `GET /health`
- `POST /predict` with JSON: `{"customer_id":"C123","features":{"tenure":12,"MonthlyCharges":75.5,"TotalCharges":906,"Contract":"Month-to-month","InternetService":"Fiber optic","PaperlessBilling":"Yes","PaymentMethod":"Electronic check"}}`
- `GET /metrics/summary`

Train the model before calling `/predict`; the model artifact is intentionally excluded from Git.

## FastAPI alternative
```bash
uvicorn churn.api:app --reload
```

## MLflow
```bash
mlflow ui --backend-store-uri ./mlruns
```
Training logs cross-validation metrics for each candidate and holdout metrics for the selected pipeline.

## Tests and CI
```bash
pytest -q
```
GitHub Actions runs the test suite on pushes and pull requests to `main`.

## Evaluation and portfolio claims
The portfolio brief mentions a 50K-record telecom dataset, three model comparisons, 52%→74% churn recall, 20+ MLflow experiments, and sub-200ms response time. This repository provides the reproducible pipeline and logging infrastructure, but those figures are **not asserted as measured by the included synthetic sample**. Validate them using the actual 50K dataset, a fixed holdout split, recorded MLflow runs, and a load test on the deployment target before publishing the claims.

## Security / production checklist
Authenticate prediction clients, validate feature schemas per model version, avoid logging direct identifiers unless necessary, configure persistent storage for model artifacts, and add request limits and monitoring.
