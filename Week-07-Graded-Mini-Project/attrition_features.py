"""
Feature engineering helpers for the attrition model.

These live in their own file (not inside the notebook) on purpose:
the saved model.pkl contains a pipeline that calls `add_features`, and
joblib can only re-load it if the function is importable from a module.
"""

import pandas as pd

# Columns that carry no information for prediction:
# - EmployeeCount, Over18, StandardHours have the same value for every row
# - EmployeeNumber is just an ID
DROP_COLS = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]

# Text columns with no natural order -> one-hot encoded later
NOMINAL_COLS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop useless columns and add a few derived features.

    Works on raw HR rows (same columns as the Kaggle CSV, with or
    without the Attrition column).
    """
    # Work on a copy so the caller's DataFrame is never modified
    out = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()

    # Average of the four 1-4 satisfaction / involvement ratings.
    # One low score can be noise; a low average is a stronger signal.
    out["AvgSatisfaction"] = out[
        ["EnvironmentSatisfaction", "JobSatisfaction",
         "RelationshipSatisfaction", "WorkLifeBalance"]
    ].mean(axis=1)

    # Share of the whole career spent at this company (0..1).
    # +1 avoids divide-by-zero for people with 0 total years.
    out["TenureRatio"] = out["YearsAtCompany"] / (out["TotalWorkingYears"] + 1)

    # Average years per employer: low values = frequent job hopper
    out["YearsPerCompany"] = out["TotalWorkingYears"] / (out["NumCompaniesWorked"] + 1)

    # Share of company tenure spent without a promotion
    out["PromotionStagnation"] = out["YearsSinceLastPromotion"] / (out["YearsAtCompany"] + 1)

    # Is this person paid less than the typical person at the same job level?
    # Uses a fixed reference table (medians from the training split) so the
    # value does not depend on which rows are in the current batch.
    level_median = out["JobLevel"].map(LEVEL_MEDIAN_INCOME)
    out["IncomeVsLevel"] = out["MonthlyIncome"] / level_median

    return out


# Median MonthlyIncome per JobLevel in the TRAINING split (computed once in the
# notebook and hard-coded here so predictions stay consistent).
LEVEL_MEDIAN_INCOME = {1: 2658.0, 2: 5346.5, 3: 9985.0, 4: 16207.5, 5: 19197.0}
