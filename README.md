# Customer Churn Prediction Service

A reproducible classical-ML project that compares Logistic Regression, Random Forest, and XGBoost with stratified 5-fold cross-validation, selects the candidate with the strongest churn recall, tracks metrics in MLflow, persists a preprocessing-plus-model pipeline, and serves predictions through Flask.

## Setup (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

## Train
```bash
python scripts/make_sample_data.py
python -m churn.train --data data/sample_churn.csv
```
The included generator creates synthetic smoke-test data only. Replace it with a licensed telecom churn CSV containing a `Churn` column (`Yes`/`No`) to reproduce real model evaluation. The training pipeline logs CV recall/F1/ROC-AUC for all three algorithms, then evaluates the selected model on a stratified holdout set.

## Serve
```bash
python app.py
```
Routes: `GET /health`, `POST /predict`, and `GET /metrics/summary`. A FastAPI alternative is available with `uvicorn churn.api:app --reload`.

## MLflow and latency check
```bash
mlflow ui --backend-store-uri ./mlruns
python evaluation/load_test.py
```
Train a model and start the service before running the load-test script. It reports mean and p95 latency for 100 sequential requests; it is a local smoke test, not a production load benchmark.

## Tests / CI
```bash
pytest -q
```
GitHub Actions installs requirements and runs tests for pushes and pull requests to `main`.

## Reproducing the resume figures
The supplied project implements the experimental workflow but does not claim that the sample synthetic dataset achieves 52%→74% churn recall or that the service meets a sub-200ms production SLO. Validate those claims on the stated 50K-record telecom dataset, keep the dataset split fixed, retain MLflow run IDs, and run latency tests on the target deployment environment. The 20+ experiment count should be supported by the actual MLflow tracking history.
