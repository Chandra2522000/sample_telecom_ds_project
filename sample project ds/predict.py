"""
Score customers with the trained churn model.

    python predict.py                      # scores 3 demo customers
    python predict.py my_customers.csv     # scores every row of your CSV

Your CSV needs these raw columns (same as data/telco_churn_clean.csv, minus 'churn'):
gender, senior_citizen, partner, dependents, tenure, contract, internet_service,
online_security, tech_support, streaming_tv, paperless_billing, payment_method,
support_calls, monthly_charges, total_charges
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from churn_project import CATEGORICAL, NUMERIC, add_features  # reuse the same feature code

BASE = Path(__file__).resolve().parent
model = joblib.load(BASE / "models" / "churn_model.joblib")
threshold = json.loads((BASE / "models" / "metrics.json").read_text())["threshold"]


def score(customers: pd.DataFrame) -> pd.DataFrame:
    X = add_features(customers)[NUMERIC + CATEGORICAL]
    out = customers.copy()
    out["churn_probability"] = model.predict_proba(X)[:, 1].round(3)
    out["risk"] = pd.cut(out["churn_probability"], [-1, threshold / 2, threshold, 2],
                         labels=["Low", "Medium", "High"])
    out["action"] = out["risk"].map({
        "High": "Call + retention offer", "Medium": "Light-touch email", "Low": "No action"})
    return out


DEMO = pd.DataFrame([
    # new, month-to-month, fiber, pays by e-check, no support add-ons, lots of calls -> risky
    dict(gender="Female", senior_citizen=0, partner="No", dependents="No", tenure=3,
         contract="Month-to-month", internet_service="Fiber optic", online_security="No",
         tech_support="No", streaming_tv="Yes", paperless_billing="Yes",
         payment_method="Electronic check", support_calls=4, monthly_charges=95.0, total_charges=285.0),
    # long-term, two-year contract, DSL with support -> safe
    dict(gender="Male", senior_citizen=0, partner="Yes", dependents="Yes", tenure=60,
         contract="Two year", internet_service="DSL", online_security="Yes",
         tech_support="Yes", streaming_tv="No", paperless_billing="No",
         payment_method="Credit card", support_calls=0, monthly_charges=57.0, total_charges=3420.0),
    # in between
    dict(gender="Female", senior_citizen=1, partner="No", dependents="No", tenure=18,
         contract="One year", internet_service="DSL", online_security="No",
         tech_support="No", streaming_tv="No", paperless_billing="Yes",
         payment_method="Bank transfer", support_calls=2, monthly_charges=52.0, total_charges=936.0),
])

if __name__ == "__main__":
    data = pd.read_csv(sys.argv[1]) if len(sys.argv) > 1 else DEMO
    result = score(data)
    cols = ["tenure", "contract", "internet_service", "monthly_charges", "churn_probability", "risk", "action"]
    print(result[cols].to_string(index=False))
