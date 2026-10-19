# Week 9 Graded Mini Project: House Sales Data Analysis

IITM Pravartak - Professional Certificate Programme in Generative AI and Machine Learning

This project analyses 1,460 house sales (Ames, Iowa, 2006-2010) to find what drives sale prices, and predicts SalePrice with XGBoost, LightGBM and CatBoost.

## Files

| File | What it is |
|---|---|
| `Week 9_Graded Mini Project_Jain.pdf` | **Submission report.** The course template filled in for every task, with code snippets, charts, tables and answers |
| `Week 9_Graded Mini Project_Jain.ipynb` | Full notebook with all code and outputs |
| `Week 9_Graded Mini Project_Jain.docx` | Editable Word version of the report |
| `Week 9_Graded Mini Project_Jain.zip` | Upload bundle: PDF + notebook + data + outputs + this README |
| `houseprice.csv` | Dataset provided by the course |
| `outputs/` | Every chart and result table written by the notebook |
| `requirements.txt` | Python packages used |

## How to run

You need Python 3.11 or newer. The project was tested on 3.14.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, XGBoost and LightGBM also need the OpenMP runtime:

```bash
brew install libomp
```

Then open the notebook and choose **Run All**, or run it without opening Jupyter:

```bash
jupyter nbconvert --to notebook --execute --inplace "Week 9_Graded Mini Project_Jain.ipynb"
```

A full run takes about **10-11 minutes** on a laptop. Almost all of that time is the hyperparameter search: 25 combinations × 5 folds for each of the three boosters. A fixed seed (42) is used everywhere, so the numbers are the same on every run.

## Results at a glance

**Test-set results** (292 houses never seen during training or tuning):

| Model | Test RMSE | Test MAE | Test R² |
|---|---|---|---|
| **CatBoost** | **$18,205** | **$13,087** | **0.940** |
| Average of 3 boosters | $18,528 | $13,319 | 0.938 |
| LightGBM | $19,297 | $13,634 | 0.933 |
| XGBoost | $19,427 | $13,969 | 0.932 |
| Ridge (baseline) | $19,088 | $13,775 | 0.934 |

**Interaction features help.** Removing the 17 engineered interaction features makes every booster worse: cross-validated RMSE goes up by 2.5-4.0%, in 5 of 5 folds.

**Main price drivers**
1. Overall quality × size (the top feature in all three models)
2. Total square footage
3. Kitchen, exterior and basement quality
4. House age and remodelling
5. Neighbourhood (a 3x spread in median price)
6. Garage
7. Bathrooms
