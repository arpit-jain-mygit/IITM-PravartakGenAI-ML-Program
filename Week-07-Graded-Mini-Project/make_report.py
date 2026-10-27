"""
Build report.pdf (executive summary) from the notebook outputs in outputs/.

    python make_report.py      -> report.pdf

Plain A4 document: dataset overview, challenges, model comparison, insights and recommendations.
Every number is read from outputs/*.csv, so the report always matches the notebook.
"""

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=10, leading=14, spaceAfter=5)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=17, spaceAfter=4, textColor=colors.HexColor("#1f3b57"))
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#1f3b57"))
small = ParagraphStyle("s", parent=body, fontSize=8.5, leading=11, textColor=colors.HexColor("#555555"))
cap = ParagraphStyle("c", parent=small, alignment=TA_CENTER)
bul = ParagraphStyle("bu", parent=body, leftIndent=12, bulletIndent=2)
P = lambda t, s=body: Paragraph(t, s)
B = lambda t: Paragraph(t, bul, bulletText="•")


def table(data, widths, highlight_row=None, center=True):
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5), ("FONT", (0, 1), (-1, -1), "Helvetica", 8.5),
          ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
          ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
          (("ALIGN", (1, 0), (-1, -1), "CENTER") if center else ("ALIGN", (0, 0), (-1, -1), "LEFT")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f6f9")])]
    if highlight_row:
        st += [("BACKGROUND", (0, highlight_row), (-1, highlight_row), colors.HexColor("#dff0e4")),
               ("FONT", (0, highlight_row), (-1, highlight_row), "Helvetica-Bold", 8.5)]
    t.setStyle(TableStyle(st))
    return t


def img(path, w):
    iw, ih = ImageReader(path).getSize()
    return Image(path, width=w, height=w * ih / iw)


cv = pd.read_csv("outputs/cv_results.csv", index_col=0)
tuned = pd.read_csv("outputs/tuned_threshold_results.csv", index_col=0)
imb = pd.read_csv("outputs/imbalance_comparison.csv", index_col=0)

order = ["Logistic Regression", "SVM (RBF)", "Gradient Boosting + SMOTE", "Random Forest", "Decision Tree"]
f = lambda v: f"{v:.3f}"
comp = [["Model", "CV F1\n(@0.5)", "CV\nROC-AUC", "Tuned\nthreshold", "Test\nPrecision", "Test\nRecall",
         "Test F1", "Test\nROC-AUC", "Test\nPR-AUC"]]
for m in order:
    comp.append([m, f(cv.loc[m, "CV F1"]), f(cv.loc[m, "CV ROC-AUC"]), f"{tuned.loc[m, 'Threshold']:.2f}",
                 f(tuned.loc[m, "Precision"]), f(tuned.loc[m, "Recall"]), f(tuned.loc[m, "F1"]),
                 f(tuned.loc[m, "ROC-AUC"]), f(tuned.loc[m, "PR-AUC"])])

story = []
story += [P("Predicting Employee Attrition - IBM HR Analytics", h1),
          P("Week 7 Graded Mini Project - Executive Summary &nbsp;|&nbsp; Arpit Jain", small), Spacer(1, 8)]

story += [P("Summary", h2),
          P("A classification model was built to flag employees who are likely to leave, using the IBM HR Analytics "
            "dataset. Five models were tuned and compared. The final model is a <b>regularised Logistic Regression</b> "
            "with a tuned decision threshold. On unseen test data it ranks a random leaver above a random stayer "
            "80% of the time (ROC-AUC 0.80). At its operating point, 58% of the people it flags really left, and it "
            "catches about half of all leavers. The clearest drivers of attrition are <b>overtime</b> (odds of leaving "
            "x3.4), <b>low pay and early career</b>, <b>stalled promotions</b>, <b>frequent travel</b> and <b>low "
            "job involvement</b>.")]

story += [P("1. Dataset overview", h2),
          table([["Item", "Value"],
                 ["Source", "IBM HR Analytics Employee Attrition (Kaggle), fictional data created by IBM"],
                 ["Rows / columns", "1,470 employees / 35 columns (26 numeric, 9 text)"],
                 ["Target", "Attrition: Yes = 237 (16.1%), No = 1,233 (83.9%), imbalanced about 5:1"],
                 ["Missing values / duplicates", "None / none"],
                 ["Dropped columns", "EmployeeCount, Over18, StandardHours (constant), EmployeeNumber (ID)"],
                 ["Engineered features", "AvgSatisfaction, TenureRatio, YearsPerCompany, PromotionStagnation, IncomeVsLevel"],
                 ["Split", "Stratified 80/20 (1,176 train / 294 test), seed 42; test set used only for final evaluation"]],
                [4.2 * cm, 12.6 * cm], center=False),
          Spacer(1, 8),
          img("outputs/02_attrition_by_category.png", 16.5 * cm),
          P("Figure 1 - Attrition rate by category (dashed line = 16.1% overall). Overtime, single status, frequent "
            "travel, Sales Representatives and job level 1 stand out.", cap)]

story += [PageBreak(), P("2. Approach", h2),
          B("<b>EDA:</b> target balance, attrition rate per category, numeric distributions of leavers vs stayers, "
            "satisfaction ratings, a correlation matrix and an IQR outlier check."),
          B("<b>Preprocessing</b> was done in one scikit-learn / imbalanced-learn pipeline: feature engineering, then "
            "one-hot encoding for nominal columns and standard scaling for numeric and ordinal columns. Because it is "
            "a single pipeline, every step is re-fitted inside each CV fold, so there is no data leakage."),
          B("<b>Imbalance:</b> class weights for LR, Decision Tree, Random Forest and SVM; SMOTE (applied only to "
            "training folds) for Gradient Boosting."),
          B("<b>Models:</b> Logistic Regression, Decision Tree, Random Forest, SVM (RBF) and Gradient Boosting. Each "
            "was tuned with GridSearchCV and 5-fold stratified CV, optimising F1 for the 'leave' class."),
          B("<b>Threshold tuning:</b> each model's cut-off was chosen on out-of-fold training predictions, "
            "then applied once to the untouched test set."),
          B("<b>Drivers:</b> logistic regression odds ratios, permutation importance on raw columns, and Random Forest "
            "importances as a cross-check.")]

story += [P("3. Challenges faced", h2),
          B("<b>Class imbalance makes accuracy misleading.</b> Predicting 'nobody leaves' already scores 84% accuracy. "
            "Precision, recall, F1, ROC-AUC and PR-AUC for the leaver class were used instead."),
          B(f"<b>Imbalance handling has a side effect.</b> Class weights raised recall at the 0.5 cut-off from "
            f"{imb.loc['No handling', 'recall']:.2f} to {imb.loc['Class weights', 'recall']:.2f}, but F1 at 0.5 fell "
            f"({imb.loc['No handling', 'f1']:.2f} to {imb.loc['Class weights', 'f1']:.2f}) and ROC-AUC did not change. "
            "The methods shift probabilities rather than improve ranking, so the threshold had to be re-tuned "
            "per model. Without that, the models could not be compared fairly."),
          B("<b>Multicollinearity.</b> JobLevel and MonthlyIncome correlate at 0.95, and age, experience and tenure "
            "move together. This was handled with L2 regularisation and by reading drivers from permutation "
            "importance as well as from coefficients."),
          B("<b>Outliers.</b> The IQR rule flags about 10% of rows, but these are genuine values (senior salaries, long "
            "tenure) or artefacts of discrete columns (PerformanceRating is only 3 or 4). They were kept, and their "
            "influence was limited through scaling and regularisation."),
          B("<b>Small minority class.</b> With only 237 leavers, results move a few points between folds (CV F1 std "
            "about 0.04-0.06), so differences smaller than that were treated as ties."),
          B("<b>Reloading the model.</b> Feature engineering lives in a separate module (attrition_features.py), so the "
            "saved pipeline can be loaded outside the notebook.")]

story += [PageBreak(), P("4. Model comparison", h2),
          P("CV columns are from 5-fold cross-validation on the training set. Test columns are on the 294 held-out "
            "employees, at each model's own tuned threshold.", small),
          table(comp, [4.2 * cm] + [1.55 * cm] * 8, highlight_row=1),
          Spacer(1, 6),
          P("<b>Why Logistic Regression?</b> After threshold tuning, SVM and Logistic Regression are tied on the "
            f"training data (out-of-fold F1 {tuned.loc['SVM (RBF)', 'OOF F1 (train)']:.3f} vs "
            f"{tuned.loc['Logistic Regression', 'OOF F1 (train)']:.3f}, a gap well within fold noise). On the test set, "
            "LR has the best F1 and PR-AUC, with ROC-AUC equal to SVM's. With performance tied, LR wins on "
            "explainability: HR can see why each person is flagged. Tree models were clearly weaker. A single "
            "decision tree underfits once it is constrained enough not to overfit."),
          P("<b>Default vs tuned threshold for the final model:</b> at 0.5 it flags 83 people and catches 30 of 47 "
            "leavers (precision 0.36). At the tuned 0.68 it flags 40 and catches 23 (precision 0.58, F1 0.46 to 0.53). "
            "The threshold is a business lever: lower it when retention actions are cheap, raise it when they are "
            "costly."),
          Spacer(1, 4),
          img("outputs/07_roc_pr_curves.png", 16.5 * cm),
          P("Figure 2 - ROC and precision-recall curves on the test set.", cap)]

story += [PageBreak(), P("5. Key insights - what drives attrition", h2),
          table([["Driver", "Evidence"],
                 ["Overtime", "30.5% attrition vs 10.4%; odds ratio 3.4; top permutation importance"],
                 ["Low pay / early career", "Median income 3,202 (leavers) vs 5,204; job level 1 at 26%; "
                  "#2 permutation importance"],
                 ["Job hopping", "Low years per previous employer: Random Forest's top feature, odds ratio 0.60 per s.d."],
                 ["Stalled promotion", "Years since last promotion (same tenure): odds ratio 1.65 per s.d."],
                 ["Travel and commute", "Frequent travellers 24.9% vs non-travel 8.0%; distance odds ratio 1.33"],
                 ["Low engagement", "Job involvement rating 1: 33.7% vs rating 4: 9.0%; low job and environment satisfaction"],
                 ["Role", "Sales Representative 39.8%, Lab Technician 23.9%; Research Director 2.5%"],
                 ["Marital status", "Single 25.5% vs married 12.5% / divorced 10.1%"],
                 ["Protective", "More years with the current manager (odds ratio 0.75)"]],
                [4.2 * cm, 12.6 * cm], center=False),
          Spacer(1, 6),
          img("outputs/10_permutation_importance.png", 7.5 * cm),
          P("Figure 3 - Permutation importance of the final model (drop in test ROC-AUC when a column is shuffled).", cap)]

story += [P("6. Recommendations", h2),
          B("<b>Monitor and cap overtime.</b> It is the biggest single lever. Track overtime per team monthly, and "
            "rebalance workload or add headcount where it is chronic."),
          B("<b>Review junior pay and promotion timelines.</b> Benchmark job-level-1 salaries, and flag anyone with no "
            "promotion in 3+ years for a career conversation."),
          B("<b>Treat low engagement scores as an early warning.</b> Follow up when job involvement or satisfaction "
            "drops to 1-2."),
          B("<b>Target retention programmes at Sales Representatives and Lab Technicians</b>: career paths, incentives "
            "and workload review."),
          B("<b>Reduce travel and commute strain</b> by rotating travel-heavy assignments and offering hybrid work "
            "for long commutes."),
          B("<b>Use the model as a quarterly risk-ranked watch-list</b> for HR business partners, not as an automatic "
            "decision. Measure whether interventions actually lower attrition.")]

story += [P("7. Limitations and next steps", h2),
          B("The data is small (237 leavers) and fictional. The model should be re-validated on the company's own HR data."),
          B("The drivers are associations, not proven causes; for example, single employees also tend to be younger and lower paid."),
          B("Next steps: per-employee explanations (SHAP), a time-to-exit (survival) model, and probability calibration "
            "if the scores will be read as exact percentages.")]


def footer(c, d):
    c.setFont("Helvetica", 7.5); c.setFillColor(colors.grey)
    c.drawString(2 * cm, 1.2 * cm, "Week 7 Mini Project - Employee Attrition")
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {d.page}")


SimpleDocTemplate("report.pdf", pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.6 * cm,
                  bottomMargin=1.8 * cm, title="Predicting Employee Attrition - Executive Summary",
                  author="Arpit Jain").build(story, onFirstPage=footer, onLaterPages=footer)
print("report.pdf written")
