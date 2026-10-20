"""Generate a deterministic synthetic telecom-like dataset for smoke tests only."""
from pathlib import Path
import numpy as np
import pandas as pd

def main(rows=1200, output="data/sample_churn.csv"):
    rng = np.random.default_rng(19)
    tenure = rng.integers(0, 73, rows)
    monthly = rng.normal(72, 22, rows).clip(18, 125)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], rows, p=[.55, .25, .20])
    internet = rng.choice(["DSL", "Fiber optic", "No"], rows, p=[.35, .45, .20])
    logit = -1.4 + (contract == "Month-to-month") * 1.1 - tenure * .025 + (monthly > 90) * .35
    probability = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.random(rows) < probability, "Yes", "No")
    frame = pd.DataFrame({
        "customerID": [f"C{i:05d}" for i in range(rows)],
        "tenure": tenure, "MonthlyCharges": monthly.round(2),
        "TotalCharges": (tenure * monthly).round(2),
        "Contract": contract, "InternetService": internet,
        "PaperlessBilling": rng.choice(["Yes", "No"], rows),
        "PaymentMethod": rng.choice(["Electronic check", "Bank transfer", "Credit card"], rows),
        "Churn": churn,
    })
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote {rows} synthetic rows to {output}")

if __name__ == "__main__":
    main()
