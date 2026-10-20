from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from .data import load_dataset, validate_dataset, TARGET

def build_preprocessor(X: pd.DataFrame):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)])

def train(dataset_path: str, output_path: str = "artifacts/churn_pipeline.joblib"):
    frame = validate_dataset(load_dataset(dataset_path))
    y = frame[TARGET]
    X = frame.drop(columns=[TARGET, "customerID"], errors="ignore")
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=2,
                                                class_weight="balanced", random_state=42, n_jobs=-1),
        "xgboost": XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.04,
                                 subsample=0.85, colsample_bytree=0.85, eval_metric="logloss",
                                 random_state=42, n_jobs=-1),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
    scores = {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocess", build_preprocessor(X)), ("model", estimator)])
        result = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1)
        scores[name] = {metric: float(result[f"test_{metric}"].mean()) for metric in scoring}
    best_name = max(scores, key=lambda name: scores[name]["recall"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    best = Pipeline([("preprocess", build_preprocessor(X_train)),
                     ("model", candidates[best_name])])
    best.fit(X_train, y_train)
    probabilities = best.predict_proba(X_test)[:, 1]
    threshold = 0.5
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": best, "threshold": threshold, "model_name": best_name}, output_path)
    mlflow.set_experiment("telecom-churn")
    with mlflow.start_run(run_name=best_name):
        mlflow.log_param("selected_model", best_name)
        mlflow.log_param("threshold", threshold)
        for name, values in scores.items():
            for metric, value in values.items():
                mlflow.log_metric(f"cv_{name}_{metric}", value)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best, "model")
    return {"selected_model": best_name, "cross_validation": scores, "holdout_metrics": metrics,
            "model_path": output_path}

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/telco_churn.csv")
    parser.add_argument("--output", default="artifacts/churn_pipeline.joblib")
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.output), indent=2))
