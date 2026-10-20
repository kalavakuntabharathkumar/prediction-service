"""Small repeatable latency smoke test; start app.py and train a model first."""
import json, statistics, time, urllib.request

URL = "http://127.0.0.1:5000/predict"
PAYLOAD = {
    "customer_id": "load-test",
    "features": {
        "tenure": 12, "MonthlyCharges": 75.5, "TotalCharges": 906,
        "Contract": "Month-to-month", "InternetService": "Fiber optic",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check"
    }
}

def main(requests=100):
    samples = []
    body = json.dumps(PAYLOAD).encode()
    for _ in range(requests):
        req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=10) as response:
            response.read()
        samples.append((time.perf_counter() - start) * 1000)
    print({"requests": requests, "mean_ms": round(statistics.mean(samples), 2),
           "p95_ms": round(sorted(samples)[int(.95 * (len(samples)-1))], 2)})

if __name__ == "__main__":
    main()
