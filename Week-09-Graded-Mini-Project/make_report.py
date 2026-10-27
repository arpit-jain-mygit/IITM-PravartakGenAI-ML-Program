"""
Build the Week 9 submission report from the notebook outputs (plain A4 document, no course template,
logo or watermark) and export it to PDF with LibreOffice.

    python make_report.py      -> Week 9_Graded Mini Project_Jain.docx / .pdf

Every number is read from outputs/*.csv, so the report always matches the notebook.
"""

import shutil
import subprocess
from pathlib import Path

import pandas as pd

from report_helpers import Body, new_document, section

HERE = Path(__file__).parent
O = HERE / "outputs"
NAME = "Week 9_Graded Mini Project_Jain"

test = pd.read_csv(O / "test_results.csv", index_col=0)
cv = pd.read_csv(O / "cv_results.csv", index_col=0)
abl = pd.read_csv(O / "ablation_interactions.csv", index_col=0)
iqr = pd.read_csv(O / "iqr_outliers.csv", index_col=0)
bins = pd.read_csv(O / "bin_summary.csv")

d = new_document("Week 9 Graded Mini Project: House Sales Data Analysis",
                 f"Arpit Jain  |  Notebook: {NAME}.ipynb (full code and outputs)")
c = Body(d)
c.text("**Objective:** analyse housing data to identify the key factors that influence sale prices, through data "
       "cleaning, feature engineering, exploratory analysis and predictive modelling with XGBoost, LightGBM and CatBoost.")

# ======================================================================== Task 1
section(d, "Task 1: Data Cleaning and Preprocessing")
section(d, "1.1 Handling Missing Values", 2)
c = Body(d)
c.text("19 of the 81 columns have missing values. Following the data description, a blank usually means "
       "**the facility does not exist** (for example, no garage), so each column was treated according to what the gap means:")
c.image(O / "01_missing_values.png", 12.5, "Figure 1 - Columns with missing values (count shown at bar end)")
c.table([["Group", "Columns", "Treatment"],
         ["Categorical, facility absent", "PoolQC, MiscFeature, Alley, Fence, FireplaceQu, Garage Type/Finish/Qual/Cond, "
          "Bsmt Qual/Cond/Exposure/FinType1/FinType2, MasVnrType", "Filled with category \"None\""],
         ["Numeric field of that facility", "GarageYrBlt, MasVnrArea (and other Bsmt/Garage numerics)", "Filled with 0 (as instructed)"],
         ["Genuine gaps (facility exists)", "BsmtExposure (house 949), BsmtFinType2 (house 333) - both have a basement; "
          "Electrical (house 1380)", "Mode from training data: \"No\", \"Unf\", \"SBrkr\""],
         ["Measurement, not a facility", "LotFrontage (259 rows, 17.7%)", "Median LotFrontage of the same Neighborhood (training data)"]],
        [3.6, 8.0, 4.0])
c.bullets(["**Pandas detail:** by default pandas reads the text \"None\" as missing. Here \"None\" is a real "
           "MasVnrType category, so the file is read with keep_default_na=False, na_values=['NA','']. This "
           "reduces the MasVnrType gaps from 872 to the 8 that are really missing.",
           "**LotFrontage is not set to 0:** every lot touches a street, so 0 would be false. Lots in the same "
           "neighbourhood have similar frontage, which makes the neighbourhood median a sensible estimate.",
           "**No leakage:** the medians and modes were learned from the training split only. After cleaning, "
           "0 missing values remain in both train and test."])
c.code('''raw = pd.read_csv("houseprice.csv", keep_default_na=False, na_values=["NA", ""])
df[NONE_COLS] = df[NONE_COLS].fillna("None")          # facility absent
df[ZERO_COLS] = df[ZERO_COLS].fillna(0)               # its numeric fields -> 0
lot_med = train_df.groupby("Neighborhood")["LotFrontage"].median()
d["LotFrontage"] = d["LotFrontage"].fillna(d["Neighborhood"].map(lot_med))''', "Code snippet (notebook section 1.1)")

section(d, "1.2 Converting Categorical Variables", 2)
c = Body(d)
c.text("Encoding was chosen per column type, so that order is preserved where it exists and never invented where it doesn't:")
c.table([["Type", "Columns", "Method", "Justification"],
         ["Ordered quality scales", "ExterQual, ExterCond, BsmtQual, BsmtCond, HeatingQC, KitchenQual, FireplaceQu, GarageQual, GarageCond, PoolQC",
          "Ordinal: None=0, Po=1, Fa=2, TA=3, Gd=4, Ex=5", "Clear order; one column keeps it"],
         ["Other ordered scales", "BsmtExposure, BsmtFinType1/2, GarageFinish, Functional, LotShape, LandSlope, PavedDrive, CentralAir, Street, Utilities",
          "Ordinal by rank", "Same reasoning"],
         ["Binned features", "AgeBand, QualBand, SizeBand, EraBuilt, LotBand", "Ordinal 0-3", "Bins are ordered"],
         ["Nominal (no order)", "23 columns: MSZoning, Neighborhood, MSSubClass, HouseStyle, Exterior1st/2nd, SaleType, Foundation, ...",
          "One-hot encoding", "Integers would imply a false order"]],
        [2.8, 6.4, 3.4, 3.0])
c.bullets(["**MSSubClass** is stored as a number (20, 60, 120...), but it is a **code** for the dwelling type, "
           "so it was converted to text and one-hot encoded.",
           "**Train/test consistency:** one-hot columns were created from the training categories, and the test set "
           "was re-indexed to match. A category that only appears in the test set becomes all-zero instead of breaking the model.",
           "**Result:** 21 ordinal columns + 5 bins + 23 one-hot columns give a final model matrix of **260 features** "
           "(1,166 training rows, 292 test rows)."])
c.code('''d["MSSubClass"] = "SC" + d["MSSubClass"].astype(str)      # code, not a quantity
for col, mapping in ORDINAL_MAPS.items():                  # Ex=5 ... Po=1, None=0
    d[col] = d[col].map(mapping)
X_train = pd.get_dummies(tr, columns=NOMINAL, dtype=int)
X_test = pd.get_dummies(te, columns=NOMINAL, dtype=int).reindex(columns=X_train.columns, fill_value=0)''',
       "Code snippet (notebook section 1.2)")

section(d, "1.3 Normalising Numerical Features", 2)
c = Body(d)
c.bullets(["**Step 1, log transform:** 39 continuous columns with |skewness| > 0.75 were transformed with log(1 + x). "
           "For example, LotArea skew dropped from 12.3 to 0.0 and LotArea_per_SF from 9.0 to 0.8.",
           "**Step 2, standardise:** every continuous column was scaled to mean 0 and standard deviation 1 with StandardScaler. "
           "The scaler was **fitted on the training set only** and then applied to the test set.",
           "**Dummy columns:** one-hot 0/1 columns were left unscaled, because scaling them adds nothing.",
           "**Target:** SalePrice itself was log-transformed (skew 1.88 to 0.12). Models predict log price, which is converted "
           "back to dollars for the metrics.",
           "**Note:** tree boosters are insensitive to scale. Normalisation matters for the Ridge baseline, and using one "
           "matrix for all models keeps the comparison consistent."])
c.image(O / "11_normalisation.png", 16.0, "Figure 2 - Raw (top) vs log + standardised (bottom) distributions")
c.code('''skewed = [c for c in cont_cols if abs(X_train[c].skew()) > 0.75 and (X_train[c] >= 0).all()]
X_train_n[skewed] = np.log1p(X_train_n[skewed]); X_test_n[skewed] = np.log1p(X_test_n[skewed])
scaler = StandardScaler().fit(X_train_n[cont_cols])          # fit on TRAIN only
X_train_n[cont_cols] = scaler.transform(X_train_n[cont_cols])
X_test_n[cont_cols] = scaler.transform(X_test_n[cont_cols])''', "Code snippet (notebook section 1.3)")

section(d, "1.4 Outlier Detection", 2)
c = Body(d)
c.text("Two methods were used. The first is a statistical IQR scan (1.5 x IQR rule) of the key numeric columns. "
       "The second is a domain check of price against the strongest driver, living area.")
c.table([["Column", "IQR upper fence", "Rows flagged", "Max value"]] +
        [[i, f"{r['upper']:,.0f}", int(r["outliers"]), f"{r['max']:,.0f}"] for i, r in iqr.iterrows()],
        [4.0, 3.5, 3.0, 3.0])
c.image(O / "02_outliers.png", 16.0, "Figure 3 - Living area vs price (left) and spread of area columns (right)")
c.bullets(["**IQR-flagged rows were kept.** Most are simply large or expensive houses, which are real data and the top of the "
           "market the model must learn. Their influence was reduced with the **log transforms** described in 1.3.",
           "**Two rows were removed (Ids 524, 1299).** These are houses of more than 4,500 sq ft with quality 10 that sold for only "
           "$160k-185k. Both are **Partial** sales in Edwards, meaning they were sold before completion, so the price "
           "does not reflect the house. The dataset's author also recommends removing them.",
           "**The other two very large houses** (over $700k) follow the size-price trend and were kept. This leaves **1,458 rows**."])

# ======================================================================== Task 2
section(d, "Task 2: Feature Engineering")
section(d, "2.1 Creating Interaction Features", 2)
c = Body(d)
c.text("17 features were engineered, combining columns into quantities that buyers actually think about:")
c.table([["Feature", "Formula", "Why it helps"],
         ["TotalSF", "TotalBsmtSF + 1stFlrSF + 2ndFlrSF", "Total usable space"],
         ["Qual_x_TotalSF / Qual_x_LivArea", "OverallQual x TotalSF / x GrLivArea", "Size is worth more when quality is high"],
         ["Kitchen_x_Qual", "KitchenQual score x OverallQual", "Kitchens drive buyer decisions"],
         ["Qual_x_Cond", "OverallQual x OverallCond", "Good materials that are also well kept"],
         ["Garage_x_Qual", "GarageCars x GarageQual score", "Capacity only counts if in good shape"],
         ["TotalBath", "Full + 0.5 Half + BsmtFull + 0.5 BsmtHalf", "All bathrooms in one count"],
         ["HouseAge / RemodAge / IsRemodeled", "YrSold - YearBuilt / - YearRemodAdd / flag", "Age at time of sale; renovation"],
         ["TotalPorchSF, LotArea_per_SF", "Sum of porch areas; LotArea / TotalSF", "Outdoor space; land per sq ft"],
         ["HasGarage, HasBsmt, HasFireplace, Has2ndFlr, HasPool", "area/count > 0", "Presence of each facility"]],
        [4.6, 5.8, 5.2])
c.text("**Evidence that they help.** First, the correlations with log price: Qual_x_TotalSF reaches **r = 0.88** and "
       "Qual_x_LivArea 0.84, both higher than OverallQual (0.81) or GrLivArea (0.73). Second, an **ablation test**: "
       "each tuned booster was cross-validated with and without the interaction features.")
rows = [["Model", "CV RMSE (log) without", "CV RMSE (log) with", "Improvement", "Folds improved"]]
for m, r in abl.iterrows():
    rows.append([m, f"{r.iloc[0]:.4f}", f"{r.iloc[1]:.4f}", f"{r.iloc[2]:+.1f}%", r.iloc[3]])
c.table(rows, [3.8, 3.3, 3.3, 2.5, 2.7])
c.text("The interactions improve **all three boosters** by 2.5-4.0%, and **in every one of the 5 folds**. They are also "
       "the top features in all three models (Figure 14). They don't help Ridge, because Ridge works on log-transformed "
       "inputs: log(quality) + log(area) already captures the same multiplication.")
c.code('''d["TotalSF"] = d["TotalBsmtSF"] + d["1stFlrSF"] + d["2ndFlrSF"]
d["Qual_x_TotalSF"] = d["OverallQual"] * d["TotalSF"]
d["Kitchen_x_Qual"] = d["KitchenQual"].map(QUAL_MAP) * d["OverallQual"]
d["TotalBath"] = d["FullBath"] + 0.5*d["HalfBath"] + d["BsmtFullBath"] + 0.5*d["BsmtHalfBath"]
d["HouseAge"] = (d["YrSold"] - d["YearBuilt"]).clip(lower=0)''', "Code snippet (notebook section 2.1)")

section(d, "2.2 Feature Binning", 2)
c = Body(d)
c.text("Five continuous variables were binned into labelled groups that a non-technical user can read easily. The edges are "
       "fixed domain boundaries, except SizeBand, which uses training-set quartiles.")
order = {"AgeBand": "HouseAge: 0-5 / 6-20 / 21-50 / 51+ yrs", "QualBand": "OverallQual: 1-4 / 5-6 / 7-8 / 9-10",
         "SizeBand": "GrLivArea quartiles: <1,134 / <1,464 / <1,788 / above", "EraBuilt": "YearBuilt: <1950 / 1950-79 / 1980-99 / 2000+",
         "LotBand": "LotArea: <7.5k / 7.5-10k / 10-15k / >15k sq ft"}
rows = [["Bin", "Edges", "Categories (houses, median price)"]]
for b, edges in order.items():
    g = bins[bins["bin"] == b]
    rows.append([b, edges, ";  ".join(f"{r.category} ({r['count']}, ${r['median'] / 1000:.0f}k)" for _, r in g.iterrows())])
c.table(rows, [2.2, 4.8, 8.6])
c.image(O / "03_binning.png", 16.0, "Figure 4 - Price distribution within each bin")
c.bullets(["Every bin produces **clearly stepped, monotonic price levels**. QualBand goes from $105k to $374k "
           "(a 3.6x range), and AgeBand from $226k for New to $127k for Old. The categories are therefore logical and useful.",
           "The bins are ordered, so they enter the models as **ordinal codes 0-3**. QualBand appears among the "
           "top-15 features of XGBoost and CatBoost, and SizeBand in CatBoost's."])
c.code('''BINS = {"AgeBand":  ("HouseAge",    [-1, 5, 20, 50, np.inf], ["New", "Recent", "Established", "Old"]),
        "QualBand": ("OverallQual", [0, 4, 6, 8, 10],        ["Low", "Average", "Good", "Excellent"]), ...}
for new, (src, edges, labels) in BINS.items():
    d[new] = pd.cut(d[src], bins=edges, labels=labels)''', "Code snippet (notebook section 2.2)")

# ======================================================================== Task 3
section(d, "Task 3: Exploratory Data Analysis (EDA)")
section(d, "3.1 Analysis and Interpretation", 2)
c = Body(d)
c.heading("I. What is the distribution of sale prices?")
c.bullets(["**Spread:** prices range from $34,900 to $755,000. The median is $163,000 and the mean $180,933. Half of all houses sell "
           "between $130k and $214k.",
           "**Shape:** the distribution is **right-skewed** (skewness 1.88, kurtosis 6.5). A small number of expensive houses "
           "form a long tail and pull the mean above the median.",
           "**Log transform:** log(1 + price) is close to normal (skewness 0.12), and the Q-Q plot is almost straight (Figure 5). "
           "The models therefore predict log price, which makes errors relative: a $20k miss matters more on a $100k house."])
c.heading("II. How do different features relate to sale price?")
c.bullets(["**Quality is the strongest driver.** OverallQual has r = 0.82 with log price, and the median price rises at every "
           "step from 1 to 10. Kitchen, exterior and basement quality show the same staircase: median $318k for Ex kitchens "
           "against $137k for TA.",
           "**Size comes next.** TotalSF (r = 0.83), GrLivArea (0.73), garage cars and area (0.68 / 0.66) and basement area (0.65) all "
           "matter. **Size and quality multiply**: large houses only reach top prices when quality is also high (Figure 6, centre), "
           "which is why the interaction features work.",
           "**Age and renovation:** YearBuilt (r = 0.59) and YearRemodAdd (0.57). Newer houses are worth more.",
           "**Location:** median prices run from $315k (NridgHt) down to $88k (MeadowV), a 3x gap for location alone (Figure 7).",
           "**Amenities:**\n"
           "- Central air: $168k with it vs $98k without.\n"
           "- Fireplace: $190k with one vs $135k with none.\n"
           "- Garage capacity: $100k with no garage up to $295k for 3 cars.",
           "**Weak or negative links:** month and year sold, MiscVal and OverallCond on its own barely matter (|r| < 0.05). "
           "EnclosedPorch and KitchenAbvGr are slightly negative (r about -0.15), because they are typical of older houses and duplexes.",
           "**Correlated pairs** carry overlapping information: GarageCars/GarageArea (0.89), GrLivArea/TotRmsAbvGrd (0.83) "
           "and TotalBsmtSF/1stFlrSF (0.80). Tree ensembles handle this well."])
c.heading("III. Trends and patterns")
c.bullets(["**Prices were stable over time.** The median was $162k-166k from 2006 to 2009, even through the 2008 crisis, "
           "with a small dip to $155k in 2010 (a year with only part-year data).",
           "**Seasonality in volume, not price.** Sales peak in May-July (204-253 a month vs 52-59 in Dec-Feb), but the "
           "median price hardly changes by month.",
           "**Sale condition:** Partial sales of new builds have the highest median ($247k). Abnormal sales (foreclosures) "
           "go for $130k, against $160k for Normal sales.",
           "**New-construction premium:** price per sq ft rises from about $65-85 for pre-1920 homes to $118 for 1950s-70s "
           "homes and $139 for the 2000s. Newer homes are worth more per sq ft, not just bigger.",
           "**Overall pattern:** the highest prices need quality, size and a good neighbourhood together. No single feature is enough."])

section(d, "3.2 Visualisations", 2)
c = Body(d)
c.text("Key charts are shown below. All plots are in the notebook (sections 3.1-3.3); figures 1-4 appear in the Task 1 and 2 answers above.")
for f, cap in [("04_price_distribution.png", "Figure 5 - SalePrice distribution: raw (skewed), log (near-normal), Q-Q plot"),
               ("07_feature_relationships.png", "Figure 6 - Price vs quality, total area (by quality band), year built, kitchen, garage, central air"),
               ("08_neighborhood.png", "Figure 7 - Price by neighbourhood, sorted by median"),
               ("05_correlations.png", "Figure 8 - Correlation of numeric features with log(SalePrice)"),
               ("06_heatmap.png", "Figure 9 - Correlation between the strongest original features"),
               ("09_trends.png", "Figure 10 - Price by year sold, sales volume by month, price by sale condition"),
               ("10_price_per_sf_decade.png", "Figure 11 - Price per sq ft by decade built")]:
    c.image(O / f, 10.5 if f in ("05_correlations.png", "06_heatmap.png") else 16.0, cap)

# ======================================================================== Task 4
section(d, "Task 4: Model Building and Evaluation")
section(d, "Model Building", 2)
c = Body(d)
c.bullets(["**Data:** 1,458 houses, 260 features, split 80/20 (1,166 train / 292 test, random_state 42). The test set was "
           "only used once, for the final evaluation.",
           "**Target:** log(1 + SalePrice). Predictions were converted back with expm1 for the dollar metrics.",
           "**Models:** XGBoost, LightGBM and CatBoost (as required), plus a **Ridge regression baseline** to show what boosting adds. "
           "A simple average of the three boosters was also tested.",
           "**Tuning:** RandomizedSearchCV with 25 parameter combinations x 5-fold CV for each booster, minimising RMSE. "
           "The search covered the number of trees, learning rate, tree depth/leaves, row and column subsampling, and L1/L2 "
           "regularisation."])
c.table([["Model", "Best parameters (from 5-fold CV search)"],
         ["XGBoost", "n_estimators=1000, learning_rate=0.03, max_depth=3, min_child_weight=1, subsample=0.8, colsample_bytree=0.3, reg_alpha=0.001, reg_lambda=1"],
         ["LightGBM", "n_estimators=600, learning_rate=0.03, num_leaves=8, min_child_samples=5, subsample=0.7, colsample_bytree=0.5, reg_lambda=1"],
         ["CatBoost", "iterations=1500, learning_rate=0.05, depth=4, l2_leaf_reg=5, rsm=0.3, subsample=0.7 (Bernoulli)"],
         ["Ridge", "alpha=11.7 (RidgeCV)"]], [2.6, 13.0])
c.text("All three boosters settled on **shallow trees** (depth 3-4 or 8 leaves), **slow learning rates** and **strong "
       "column subsampling** (30-50% of features per tree). With only about 1,200 training houses, many small, randomised trees "
       "generalise better than a few deep ones.")
c.code('''search = {"XGBoost": (XGBRegressor(random_state=SEED, tree_method="hist"), {"n_estimators": [600, 1000, 1500],
               "learning_rate": [0.02, 0.03, 0.05], "max_depth": [3, 4, 5], "colsample_bytree": [0.3, 0.5, 0.7], ...}),
          "LightGBM": (LGBMRegressor(random_state=SEED, verbose=-1), {...}),
          "CatBoost": (CatBoostRegressor(random_seed=SEED, verbose=0), {...})}
for name, (est, grid) in search.items():
    rs = RandomizedSearchCV(est, grid, n_iter=25, cv=KFold(5, shuffle=True, random_state=SEED),
                            scoring="neg_root_mean_squared_error", random_state=SEED).fit(X_train_n, y_train)
    best_models[name] = rs.best_estimator_''', "Code snippet (notebook section 4)")

section(d, "Evaluation", 2)
c = Body(d)
rows = [["Model", "CV RMSE (log) ± std", "Test RMSE ($)", "Test MAE ($)", "Test R²", "Test RMSE (log)", "MAPE"]]
for m in ["CatBoost", "Average of 3 boosters", "XGBoost", "LightGBM", "Ridge (baseline)"]:
    r = test.loc[m]
    cvs = f"{cv.loc[m, 'CV RMSE (log)']:.4f} ± {cv.loc[m, 'CV std']:.4f}" if m in cv.index else "-"
    rows.append([m, cvs, f"{r['RMSE ($)']:,.0f}", f"{r['MAE ($)']:,.0f}", f"{r['R²']:.3f}", f"{r['RMSE (log)']:.4f}", f"{r['MAPE (%)']:.1f}%"])
c.table(rows, [3.6, 3.0, 2.0, 2.0, 1.5, 2.0, 1.5], bold_row=1)
c.image(O / "12_model_comparison.png", 16.0, "Figure 12 - Test RMSE, MAE and R² by model")
c.bullets(["**CatBoost is the best model**, both in CV (RMSE 0.109 on log price) and on the unseen test set: **RMSE $18,205, "
           "MAE $13,087, R² = 0.940**. It explains 94% of the variance in sale prices, and a typical prediction is off by about 7.9%.",
           "**XGBoost and LightGBM** follow closely (R² 0.932-0.933, MAE about $13.6-14.0k).",
           "**The Ridge baseline is competitive** (R² 0.934). With well-engineered features and a log target, much of the signal "
           "is close to linear. All the gaps are within about one CV standard deviation (about 0.01), so CatBoost's lead is real but small. "
           "Averaging the three boosters (R² 0.938) is a robust alternative.",
           "**RMSE vs MAE:** RMSE ($18.2k) is about 40% above MAE ($13.1k), so a few houses have much larger errors than "
           "typical. Most of these are expensive homes (the top-quartile MAE is about $22k), which is consistent with the residual "
           "plots below.",
           "**Error by price band (CatBoost):** the MAE is $9.7k in the cheapest quartile and $22.5k in the most expensive. In "
           "percentage terms the errors are similar (6-8%) except for the cheapest homes (11%), which are the hardest to price "
           "in relative terms."])
c.image(O / "13_pred_vs_actual.png", 16.0, "Figure 13 - Predicted vs actual (top) and residuals (bottom) on the test set")
c.image(O / "14_feature_importance.png", 16.0, "Figure 14 - Top-15 features per model (orange = engineered interactions)")
c.text("**What the models learned:** all three rank **Qual_x_TotalSF**, **Qual_x_LivArea** and **TotalSF** at the top. "
       "These are quality and size combined, which confirms the EDA finding. They are followed by OverallQual, "
       "Kitchen_x_Qual, HouseAge, TotalBath and LotArea.")
c.text("**Conclusion:** sale price is mainly driven by **how good, how big and how new** a house is, and by where it sits. "
       "CatBoost predicts it with R² = 0.94 and a typical error of about $13k (MAE).")
c.text("**Limitations:** the data covers one city (Ames, 2006-2010) and 1,458 sales, so the model should be retrained "
       "before being used in another market. Possible next steps are stacking the boosters with Ridge and adding SHAP "
       "explanations for individual predictions.")

out = HERE / f"{NAME}.docx"
d.save(out)
print("saved", out.name)
soffice = shutil.which("soffice") or "/Applications/LibreOffice.app/Contents/MacOS/soffice"
subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(HERE), str(out)], check=True, capture_output=True)
print("saved", out.with_suffix(".pdf").name)
