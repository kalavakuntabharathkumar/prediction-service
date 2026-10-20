from flask import Flask, jsonify, request
from churn.service import predict_one
from churn.store import get_summary

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify(status="ok")

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    features = payload.get("features")
    if not isinstance(features, dict) or not features:
        return jsonify(error="JSON body must include a non-empty 'features' object"), 400
    try:
        return jsonify(predict_one(features, payload.get("customer_id")))
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 503
    except Exception as exc:
        return jsonify(error=f"Prediction failed: {exc}"), 400

@app.get("/metrics/summary")
def summary():
    return jsonify(get_summary())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
