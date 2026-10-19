"""
Score employees with the saved attrition model.

Usage:
    python predict.py                      # scores the first 10 rows of the Kaggle CSV
    python predict.py my_employees.csv     # scores your own file (same columns as the Kaggle CSV)

Writes predictions.csv with a leave probability and an AT RISK flag per row.
"""

import json
import sys

import joblib
import pandas as pd

# The pipeline inside model.pkl calls attrition_features.add_features, so that
# module must be importable (it sits next to this script).
import attrition_features  # noqa: F401

# --- Load the trained pipeline and the tuned decision threshold ---
model = joblib.load("model.pkl")
threshold = json.load(open("model_metadata.json"))["decision_threshold"]

# --- Read the input rows (raw HR columns; Attrition column is optional) ---
path = sys.argv[1] if len(sys.argv) > 1 else "WA_Fn-UseC_-HR-Employee-Attrition.csv"
df = pd.read_csv(path)
if len(sys.argv) == 1:
    df = df.head(10)

# --- Predict: probability of leaving, then compare with the threshold ---
X = df.drop(columns=["Attrition"], errors="ignore")
df["leave_probability"] = model.predict_proba(X)[:, 1].round(3)
df["at_risk"] = df["leave_probability"] >= threshold

# --- Save and show a short summary ---
df.to_csv("predictions.csv", index=False)
cols = ["EmployeeNumber", "JobRole", "OverTime", "MonthlyIncome", "leave_probability", "at_risk"]
print(f"Threshold = {threshold}")
print(df[[c for c in cols if c in df.columns]].to_string(index=False))
print(f"\n{int(df['at_risk'].sum())} of {len(df)} employees flagged. Full output: predictions.csv")
