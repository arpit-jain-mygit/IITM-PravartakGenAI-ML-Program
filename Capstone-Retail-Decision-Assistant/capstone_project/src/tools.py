"""
Extra tools the controller can route to (beyond product search and policy QA):

- segment_insights : answers business questions from the clustering results of notebook 02 (no LLM;
                     every number comes from outputs/segment_summary.json)
- review_summary   : collects real review texts/labels for matching products and summarises them
                     (LLM summary grounded in those reviews; counts are computed, not generated)
- summarise_products : explains the top-3 items from the previous product search (session memory)
- calculator       : safe arithmetic, e.g. "what is 15% off 3999"
"""

import ast
import json
import operator
import re
from collections import Counter

from src.config import OUTPUTS
from src.llm import chat_json
from src.recommender import parse_constraints, retrieve_products

# ---------------------------------------------------------------- segment insights
QUESTION_MAP = [  # (pattern, column to rank by, ascending?, label)
    (r"price[- ]sensitiv|discount|bargain|deal", "discount_lift", False, "most price-sensitive (largest conversion lift from deep discounts)"),
    (r"valuable|high[- ]value|spend|premium|loyal", "avg_order_value", False, "highest-value (largest average order value)"),
    (r"conver|likely to buy|purchase rate", "purchase_rate", False, "highest-converting"),
    (r"return", "return_rate", False, "most return-prone"),
    (r"inactive|lapsing|churn|recency", "days_since_last_purchase", False, "least recently active"),
]


def segment_insights(question: str) -> dict:
    path = OUTPUTS / "segment_summary.json"
    if not path.exists():
        return {"tool": "segment_insights", "status": "unavailable",
                "answer": "Segment results are not available yet - run notebook 02 first.", "trace": []}
    segs = json.loads(path.read_text())["segments"]
    q = question.lower()
    pat, col, asc, label = next(((p, c, a, l) for p, c, a, l in QUESTION_MAP if re.search(p, q)),
                                (None, "share", False, "largest"))
    ranked = sorted(segs, key=lambda s: s[col], reverse=not asc)
    top = ranked[0]
    lines = [f"{s['name']} (segment {s['segment']}, {s['share']:.0%} of interactions): {col}={s[col]:.3f}" for s in ranked]
    answer = (f"The {label} segment is **{top['name']}** ({top['share']:.0%} of interactions). "
              f"Profile: avg age {top['age']:.0f}, avg order value Rs {top['avg_order_value']:,.0f}, "
              f"avg item price Rs {top['final_price']:,.0f}, avg discount seen {top['discount_percent']:.0f}%, "
              f"conversion {top['purchase_rate']:.0%} overall, {top['conv_deep_discount']:.0%} on >=25% discounts vs "
              f"{top['conv_low_discount']:.0%} on <=10% discounts (lift {top['discount_lift']:+.0%}).")
    return {"tool": "segment_insights", "status": "ok", "question": question, "answer": answer,
            "ranking_metric": col, "ranking": lines, "source": "outputs/segment_summary.json (notebook 02, K-Means)",
            "trace": [f"matched pattern '{pat}' -> rank segments by {col}" if pat else "no specific metric -> rank by size"]}


# ---------------------------------------------------------------- review summary
REVIEW_SYSTEM = """Summarise what customers say about the products below, using ONLY the REVIEWS given.
Do not add facts that are not in the reviews. Mention recurring positives and negatives.
Return JSON only: {"summary": "3-4 sentences", "positives": ["..."], "negatives": ["..."]}"""


def review_summary(question: str, k: int = 25) -> dict:
    hits, cons, trace = retrieve_products(question, k=k)
    if not hits:
        return {"tool": "review_summary", "status": "no_match", "answer": "No matching products with reviews.", "trace": trace}
    labels = Counter(h["meta"]["review_sentiment_label"] for h in hits)
    ratings = [h["meta"]["rating_given"] for h in hits]
    reviews = Counter(h["meta"]["customer_review_text"] for h in hits)
    block = "\n".join(f"- ({n}x) {t}" for t, n in reviews.most_common(10))
    resp = chat_json(REVIEW_SYSTEM, f"Products matching: {question}\nREVIEWS:\n{block}")
    if resp:
        data = resp["data"]
    else:
        data = {"summary": "Most frequent review themes: " + " | ".join(t for t, _ in reviews.most_common(3)),
                "positives": [], "negatives": []}
    stats = (f"Based on {len(hits)} matching items: sentiment {dict(labels)}, average rating given "
             f"{sum(ratings) / len(ratings):.1f}/5.")
    return {"tool": "review_summary", "status": "ok", "question": question, "answer": stats + " " + data.get("summary", ""),
            "positives": data.get("positives", []), "negatives": data.get("negatives", []),
            "supporting_ids": [h["doc_id"] for h in hits], "trace": trace + [f"aggregated {len(hits)} reviews"]}


# ---------------------------------------------------------------- summarise previous results
SUMMARY_SYSTEM = """Summarise the candidate products for the shopper using ONLY the candidate text.
Return JSON only: {"summary": [{"item_id": "...", "fit": "one sentence on why it fits the request"}], "overall": "one sentence"}"""


def summarise_products(question: str, last_result: dict | None) -> dict:
    if not last_result or not last_result.get("retrieved"):
        return {"tool": "summarise_products", "status": "no_context",
                "answer": "There are no earlier product results to summarise - ask for a recommendation first.", "trace": []}
    top = last_result["retrieved"][:3]
    ctx = "\n".join(f"- item_id={t['item_id']} | {t['title']} | Rs {t['price']:,.0f} | rating {t['rating']}" for t in top)
    resp = chat_json(SUMMARY_SYSTEM, f"Original request: {last_result['query']}\nCANDIDATES:\n{ctx}")
    allowed = {t["item_id"] for t in top}
    if resp:
        items = [s for s in resp["data"].get("summary", []) if s.get("item_id") in allowed]
        overall = resp["data"].get("overall", "")
    else:
        items = [{"item_id": t["item_id"], "fit": f"{t['title']} at Rs {t['price']:,.0f}, rated {t['rating']}/5."} for t in top]
        overall = "These are the three closest matches retrieved for the earlier request."
    return {"tool": "summarise_products", "status": "ok", "question": question, "summary": items, "answer": overall,
            "supporting_ids": [i["item_id"] for i in items],
            "trace": [f"used the top-3 items from the previous product search for '{last_result['query']}'"]}


# ---------------------------------------------------------------- calculator
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg}


def _safe(node):
    if isinstance(node, ast.Expression):
        return _safe(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        r = _safe(node.right)
        if isinstance(node.op, ast.Div) and r == 0:
            raise ZeroDivisionError("division by zero")
        return _OPS[type(node.op)](_safe(node.left), r)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe(node.operand))
    raise ValueError("only arithmetic is allowed")


def calculator(question: str) -> dict:
    q = question.lower().replace(",", "").replace("₹", "").replace("rs.", "").replace("rs", "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:off|discount)\s*(?:on|of)?\s*(\d+(?:\.\d+)?)", q)
    try:
        if m:
            pct, price = float(m.group(1)), float(m.group(2))
            val = round(price * (1 - pct / 100), 2)
            expr = f"{price} * (1 - {pct}/100)"
            ans = f"{pct:g}% off Rs {price:,.0f} = Rs {val:,.2f} (you save Rs {price - val:,.2f})."
        else:
            expr = re.search(r"[\d.\s+\-*/()]{3,}", q).group(0).strip()
            val = _safe(ast.parse(expr, mode="eval"))
            ans = f"{expr} = {round(val, 2):,}"
        return {"tool": "calculator", "status": "ok", "question": question, "expression": expr, "answer": ans, "trace": []}
    except Exception as e:
        return {"tool": "calculator", "status": "error", "question": question, "answer": f"Could not compute: {e}", "trace": []}
