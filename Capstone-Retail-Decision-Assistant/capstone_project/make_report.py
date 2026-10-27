"""
Build the short project report (PDF) from the saved outputs, so every number matches the notebooks.

    python make_report.py                      # live outputs   -> outputs/Capstone_Report.pdf
    RETAIL_OFFLINE=1 python make_report.py     # offline outputs -> outputs_offline/Capstone_Report.pdf
"""

import json
import sys
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

sys.path.insert(0, str(Path(__file__).parent))
import src  # noqa: F401
from src import config
from src.llm import mode_label

O = config.OUTPUTS
ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13, spaceAfter=4)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, textColor=colors.HexColor("#1f3b57"), spaceAfter=2)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, textColor=colors.HexColor("#1f3b57"), spaceBefore=8, spaceAfter=4)
small = ParagraphStyle("s", parent=body, fontSize=8, leading=10, textColor=colors.HexColor("#555"))
cap = ParagraphStyle("c", parent=small, alignment=TA_CENTER)
mono = ParagraphStyle("m", parent=body, fontName="Courier", fontSize=7.3, leading=9, backColor=colors.HexColor("#f3f5f8"))
P = lambda t, s=body: Paragraph(t.replace("₹", "Rs "), s)          # base PDF fonts have no rupee glyph
B = lambda t: Paragraph(t, ParagraphStyle("bu", parent=body, leftIndent=10), bulletText="•")


def table(rows, widths, fs=8, hl=None):
    t = Table([[Paragraph(str(c), ParagraphStyle("tc", parent=body, fontSize=fs, leading=fs + 2)) for c in r] for r in rows],
              colWidths=[w * cm for w in widths], repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
          ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#bbb")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f6f9")])]
    if hl:
        st.append(("BACKGROUND", (0, hl), (-1, hl), colors.HexColor("#dff0e4")))
    t.setStyle(TableStyle(st))
    for r in range(len(rows)):     # white header text
        for c in range(len(rows[0])):
            if r == 0:
                t._cellvalues[r][c].style = ParagraphStyle("th", parent=body, fontSize=fs, leading=fs + 2, textColor=colors.white,
                                                           fontName="Helvetica-Bold")
    return t


def img(name, w):
    from reportlab.lib.utils import ImageReader
    p = O / name
    iw, ih = ImageReader(str(p)).getSize()
    return Image(str(p), width=w * cm, height=w * cm * ih / iw)


sup = json.load(open(O / "supervised_summary.json"))
test = pd.read_csv(O / "02_test_results.csv", index_col=0)
cvr = pd.read_csv(O / "02_cv_results.csv", index_col=0)
seg = pd.read_csv(O / "02_segment_profile.csv", index_col=0)
ret = pd.read_csv(O / "03_retrieval_inspection.csv").iloc[0]
pol = pd.read_csv(O / "04_policy_retrieval.csv")
ev = pd.read_csv(O / "eval_results.csv")
evs = json.load(open(O / "evaluation_summary.json"))
log = [json.loads(l) for l in open(O / "eval_log.jsonl")]
dd = pd.read_csv(config.PROCESSED / "data_dictionary.csv")
imp = pd.read_csv(O / "02_permutation_importance.csv", index_col=0)

s = []
s += [P("Intelligent Retail Decision Assistant - Capstone Report", h1),
      P(f"Capstone Project · Arpit Jain · "
        f"Run mode: {mode_label()}", small), Spacer(1, 6)]

s += [P("1. Problem and approach", h2),
      P("A mid-sized online fashion retailer needs one assistant that (a) explains and predicts <b>conversion</b> for "
        "merchandising, (b) segments customers for CRM, (c) recommends products grounded in the catalogue, (d) answers "
        "return / exchange / shipping / refund / warranty questions from policy documents only, and (e) routes every "
        "request to the right tool with a visible trace. Architecture: <i>tabular data → classical ML (scikit-learn) → "
        "retrieval-ready product & policy documents → embeddings + FAISS → grounded LLM generation with validators → "
        "rule-first hybrid controller</i>. Code: <font face='Courier'>src/</font>; walkthrough: notebooks 01-06."),
      P("2. Data and assumptions", h2),
      B(f"Course dataset: 1,500 interaction rows × 46 columns + 5 policies (retail_policies.json). "
        f"{(dd.origin.str.startswith('engineered')).sum()} engineered columns added; full data dictionary in "
        f"data/processed/data_dictionary.csv (original vs engineered, with how each was created)."),
      B("<b>Audit findings:</b> product_id and customer_id are not stable keys (each id spans 5 / 6 rows describing different "
        "products / people) → every row is treated as its own catalogue item (new item_id) and analysis is interaction-level. "
        "size is 18% missing at random → 'Unknown' + indicator. Post-purchase and checkout columns are excluded from the "
        "purchase model (leakage)."),
      B("Text columns (titles, descriptions, reviews, queries, tags) are template-generated in the source data; no policy text "
        "was invented. Image files are not supplied, so the optional multimodal extension was not attempted.")]

lr = test.loc["Logistic Regression"]
s += [P("3. Supervised ML - predicting purchase_flag", h2),
      table([["Model", "CV ROC-AUC (±std)", "Test ROC-AUC", "Test PR-AUC", "Accuracy", "Precision", "Recall", "F1"]] +
            [[m, f"{cvr.loc[m, 'CV ROC-AUC']:.3f} ± {cvr.loc[m, 'CV std']:.3f}", f"{r['ROC-AUC']:.3f}", f"{r['PR-AUC']:.3f}",
              f"{r.accuracy:.3f}", f"{r.precision:.3f}", f"{r.recall:.3f}", f"{r.F1:.3f}"] for m, r in test.iterrows()],
            [4.2, 2.7, 1.8, 1.8, 1.6, 1.6, 1.5, 1.3], hl=2),
      Spacer(1, 4),
      P(f"Balanced target (51/49) → no resampling. Pipelines impute, scale and one-hot encode inside each CV fold; "
        f"80/20 stratified hold-out + 5-fold grid search. All models reach CV ROC-AUC ≈ 0.64 (gap < 1 std), so the "
        f"interpretable <b>{sup['model']}</b> is chosen (test ROC-AUC {lr['ROC-AUC']:.3f}). Permutation importance: "
        f"<b>{', '.join(imp.index[:4])}</b>. Membership tier dominates (Gold/Platinum ≈ 61-63% conversion vs ≈ 40% "
        f"Bronze/Silver); price and discount barely matter on average. The signal is modest - useful for prioritising "
        f"nudges, not for certain individual decisions.")]

s += [P("4. Unsupervised ML - RFM customer segments", h2),
      P("K-Means on standardised Recency (days since last purchase), Frequency (orders) and Monetary value (avg order value); "
        "k = 4 chosen by silhouette (≈ 0.26; a 9-feature mix only reached ≈ 0.09) and the elbow. Price sensitivity = conversion "
        "on ≥25% discounts minus conversion on ≤10% discounts."),
      table([["Segment", "Share", "Days since last", "Orders", "Avg order value", "Conversion", "Discount lift"]] +
            [[r["name"], f"{r.share:.0%}", f"{r.days_since_last_purchase:.0f}", f"{r.total_orders:.1f}",
              f"Rs {r.avg_order_value:,.0f}", f"{r.purchase_rate:.0%}", f"{r.discount_lift:+.1%}"] for _, r in seg.iterrows()],
            [5.0, 1.4, 2.3, 1.5, 2.6, 2.0, 2.0], hl=int(seg.discount_lift.values.argmax()) + 1),
      Spacer(1, 3),
      P(f"<b>{seg.loc[seg.discount_lift.idxmax(), 'name']}</b> are the most price-sensitive segment - target them with coupons "
        f"and bundles; win back <b>Lapsing Customers</b> with reminders; protect High-Value Regulars with service and loyalty perks.")]

s += [PageBreak(), P("5. GenAI: product RAG, policy QA and the controller", h2),
      B(f"<b>Product RAG:</b> one document per item (title, description, brand, category, attributes, price context, tags, review) → "
        f"embeddings → FAISS (exact cosine search). Budget, sub-category, colour and brand are parsed by explicit rules and applied as "
        f"hard filters (relaxed step by step, logged in the trace). Retrieval inspection on 300 dataset queries: sub-category "
        f"precision@5 = {ret['sub_category_precision@5']:.2f}, colour precision@5 = {ret['colour_precision@5']:.2f}, exact item in "
        f"top-5 = {ret['exact_item_hit@5']:.2f}. The LLM must return JSON (recommended_products, explanation, supporting_ids); a "
        f"validator drops any id not retrieved or over budget; empty / weak retrieval returns a clear no-match message."),
      B(f"<b>Policy QA:</b> sentence-level policy chunks; hit@1 = {pol['hit@1'].mean():.0%} on 10 labelled questions. Answers must cite "
        f"the policy_doc_id and quote it verbatim (validated); below the similarity threshold the assistant refuses."),
      B("<b>Controller:</b> guardrails → rule-based intent scores with specificity rules → multi-intent split (≤3 tools, each gets its own "
        "clause) → LLM router only as fallback, validated against a fixed tool list. Tools: product_search, policy_qa, segment_insights, "
        "review_summary, summarise_products (session memory), calculator. Every step is printed in a trace."),
      Spacer(1, 4), img("06_test_suite.png", 17), P("Curated test queries with regression checks (notebook 06).", cap)]

s += [P("6. Evaluation", h2),
      P(f"<b>{evs['suite_pass']}/{evs['suite_total']} curated queries pass</b> (routing accuracy {evs['routing_accuracy']:.0%}); "
        f"checks include budget respected, correct sub-category/colour, ids grounded in retrieval, correct policy cited, quotes "
        f"verbatim, out-of-scope refused. pytest regression suite: {evs['pytest']}."),
      P("Example queries and outputs", h2)]
for r in log[:5]:
    ans = r["final_answer"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
    s += [P(f"<b>[{r['id']}] {r['query']}</b> → tools: {r['routed_tools']}"), P(ans[:900], mono), Spacer(1, 3)]

s += [P("7. Findings, limitations and next steps", h2),
      B("Conversion is driven by who the customer is (membership, income) and product quality (rating) more than by price; discounts "
        "pay off mainly for one behavioural segment - so discounting should be targeted, not blanket."),
      B("Limitations: modest tabular signal (ROC-AUC ≈ 0.65-0.68); moderately separated segments; templated review text; "
        "unstable source ids; no product images."),
      B("Next steps: richer behavioural data (sessions, prior views), a cross-encoder reranker, LLM-as-judge groundedness scoring, "
        "A/B tests of segment-specific offers."),
      P("Reproducibility: pip install -r requirements.txt; key in .env (OPENAI_API_KEY, optional OPENAI_BASE_URL for Vocareum); "
        "./run_all.sh. Seeds fixed (42), temperature 0, LLM and embedding calls cached on disk; RETAIL_OFFLINE=1 runs everything "
        "without a key.", small)]

out = O / "Capstone_Report.pdf"
SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                  title="Intelligent Retail Decision Assistant - Capstone Report", author="Arpit Jain").build(s)
print("saved", out)
