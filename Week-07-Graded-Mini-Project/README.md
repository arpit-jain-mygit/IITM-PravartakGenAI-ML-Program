# Week 7 Graded Mini Project: Predicting Employee Attrition

IITM Pravartak - Professional Certificate Programme in Generative AI and Machine Learning

This project uses the IBM HR Analytics dataset (1,470 employees, 16.1% attrition) to predict whether an employee is likely to leave, and to find what drives attrition.

## Files

| File | What it is |
|---|---|
| `notebook.ipynb` | Full analysis: EDA, preprocessing, imbalance handling, 5 tuned models, evaluation, threshold tuning, drivers, model saving |
| `report.pdf` | Executive summary: dataset overview, challenges, model comparison, insights and recommendations |
| `model.pkl` | Final trained pipeline (feature engineering + encoding + scaling + Logistic Regression), saved with joblib |
| `model_metadata.json` | Chosen model, best hyperparameters, tuned decision threshold (0.68), test metrics |
| `attrition_features.py` | Feature engineering used inside the pipeline (must sit next to `model.pkl` to load it) |
| `predict.py` | Small script that loads `model.pkl` and scores a CSV of employees |
| `requirements.txt` | Python packages used |
| `WA_Fn-UseC_-HR-Employee-Attrition.csv` | Dataset from [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| `outputs/` | Charts and result tables written by the notebook |

## How to run

You need Python 3.11 or newer. The project was tested on 3.14.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Run the notebook.** Open `notebook.ipynb` in Jupyter or VS Code and choose **Run All**. On a laptop it takes about 1-2 minutes, most of it grid search. To run it without opening Jupyter:

```bash
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

The notebook regenerates `model.pkl`, `model_metadata.json` and everything in `outputs/`. A fixed random seed (42) is used everywhere, so you get the same numbers on every run.

**Score employees with the saved model:**

```bash
python predict.py                    # demo on the first 10 rows of the dataset
python predict.py new_employees.csv  # your own file, same columns as the Kaggle CSV
```

You can also load the model from your own code:

```python
import joblib, json, pandas as pd
import attrition_features             # needed so the pipeline can be unpickled

model = joblib.load("model.pkl")
threshold = json.load(open("model_metadata.json"))["decision_threshold"]

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv").drop(columns="Attrition")
prob = model.predict_proba(df)[:, 1]   # probability of leaving
at_risk = prob >= threshold
```

The model takes **raw rows** with the original columns. The dropping of useless columns, the feature engineering, the one-hot encoding and the scaling all happen inside the pipeline.

## How to interpret the results

**Output per employee**
- `leave_probability` is the model's estimate (0 to 1) that the employee will leave.
- `at_risk` is True when the probability is at or above the tuned threshold of **0.68**.

**Why 0.68 and not 0.5?**
- The model is trained with class weights, so leavers count about 5x more during training, which pushes probabilities up.
- 0.68 is the threshold that gave the best F1 on cross-validated training predictions.
- If you want to catch more leavers and can accept more false alarms, lower the `decision_threshold` in `model_metadata.json`. Raise it if you want fewer, more certain flags.

**Final model performance on the held-out test set** (294 employees, 47 leavers)

| Metric | Value | Meaning |
|---|---|---|
| ROC-AUC | 0.80 | Picks a random leaver over a random stayer 80% of the time |
| PR-AUC | 0.57 | Average precision across thresholds (random guessing = 0.16) |
| Precision | 0.58 | 23 of the 40 people flagged actually left |
| Recall | 0.49 | Caught 23 of the 47 people who left |
| F1 | 0.53 | Balance of precision and recall |

Accuracy (86%) is not the headline metric. A model that always predicts "stays" already gets 84%.

**Main attrition drivers**
1. Overtime (odds of leaving × 3.4)
2. Low monthly income
3. Early career and frequent job changes
4. Years since last promotion
5. Frequent business travel and long commutes
6. Low job involvement and low satisfaction
7. Lab Technician and Sales Representative roles
8. Being single

Time with the current manager is protective. These are associations in historical data, not proven causes. Use the model as a ranked watch-list for HR conversations, not as an automatic decision.
