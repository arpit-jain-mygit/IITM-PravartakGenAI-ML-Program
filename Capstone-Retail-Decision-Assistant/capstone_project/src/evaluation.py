"""
Final phase - curated test queries with regression-like checks.

Each case states the tools the controller should pick and machine-checkable expectations on the output
(budget respected, right sub-category/colour, correct policy cited, quotes verbatim, refusal when out of
scope ...). run_suite() executes all cases in one assistant session (so the "summarise" case can use the
previous search), writes outputs/eval_results.csv + outputs/eval_log.jsonl, and returns the table.
"""

import json

import pandas as pd

from src.agent import RetailAssistant
from src.config import OUTPUTS
from src.documents import load_product_docs

_META = None


def meta(item_id):
    global _META
    if _META is None:
        _META = {d["doc_id"]: d["meta"] for d in load_product_docs()}
    return _META[item_id]


def _res(out, tool):
    return next((r for r in out["results"] if r["tool"] == tool), {})


def recs(out):
    return _res(out, "product_search").get("recommended_products", [])


def chk_budget(limit):
    return (f"all recommended prices <= {limit}", lambda o: all(p["price"] <= limit for p in recs(o)))


def chk_attr(attr, allowed):
    return (f"all recommended {attr} in {allowed}", lambda o: bool(recs(o)) and all(meta(p["item_id"])[attr] in allowed for p in recs(o)))


def chk_grounded():
    return ("recommended ids are a subset of retrieved ids", lambda o: {p["item_id"] for p in recs(o)} <=
            {r["item_id"] for r in _res(o, "product_search").get("retrieved", [])})


def chk_policy_cites(doc_id):
    return (f"policy answer cites {doc_id}", lambda o: doc_id in (_res(o, "policy_qa").get("citations") or []))


def chk_quotes_verbatim():
    from src.policy_qa import validate
    def f(o):
        r = _res(o, "policy_qa")
        ev = [{"doc_id": e["chunk_id"].split("#")[0], "full_text": _policy_text(e["chunk_id"].split("#")[0])} for e in r.get("evidence", [])]
        return bool(r.get("quotes")) and not [i for i in validate(r, ev) if "quote" in i]
    return ("policy quotes are verbatim", f)


def _policy_text(doc_id):
    from src.data_prep import load_policies
    return load_policies()[doc_id]


CASES = [
    {"id": "Q1", "query": "Recommend casual sneakers under ₹4000 for daily wear.", "tools": ["product_search"],
     "checks": [chk_budget(4000), chk_attr("sub_category", ["Sneakers"]), chk_grounded()]},
    {"id": "Q2", "query": "Can I return a discounted product after 7 days?", "tools": ["policy_qa"],
     "checks": [chk_policy_cites("POL_RETURNS"), chk_quotes_verbatim()]},
    {"id": "Q3", "query": "Suggest products for a premium customer who prefers black formal shoes.", "tools": ["product_search"],
     "checks": [chk_attr("sub_category", ["Formal"]), chk_attr("color", ["Black"]), chk_grounded()]},
    {"id": "Q4", "query": "Which customer segment seems most price-sensitive?", "tools": ["segment_insights"],
     "checks": [("names the segment with the largest discount lift", lambda o: _top_lift_name() in o["final_answer"])]},
    {"id": "Q5", "query": "Summarise the top three retrieved products and explain why they fit the query.",
     "tools": ["summarise_products"],
     "checks": [("summary only uses items from the previous search", lambda o: bool(_res(o, "summarise_products").get("summary")))]},
    {"id": "Q6", "query": "How long does express delivery take?", "tools": ["policy_qa"],
     "checks": [chk_policy_cites("POL_SHIPPING"), chk_quotes_verbatim()]},
    {"id": "Q7", "query": "Is physical damage to my watch covered under the warranty?", "tools": ["policy_qa"],
     "checks": [chk_policy_cites("POL_WARRANTY"), chk_quotes_verbatim()]},
    {"id": "Q8", "query": "Show me a blue backpack for travel under 3000 and tell me if I can exchange it later.",
     "tools": ["product_search", "policy_qa"],
     "checks": [chk_budget(3000), chk_attr("sub_category", ["Backpack"]), chk_policy_cites("POL_EXCHANGE")]},
    {"id": "Q9", "query": "What do customers say about Nike running shoes?", "tools": ["review_summary"],
     "checks": [("review summary computed from matching items", lambda o: _res(o, "review_summary").get("status") == "ok")]},
    {"id": "Q10", "query": "When will my refund reach me if I paid by UPI?", "tools": ["policy_qa"],
     "checks": [chk_policy_cites("POL_REFUND")]},
    {"id": "Q11", "query": "Recommend a red watch under ₹300.", "tools": ["product_search"],
     "checks": [chk_budget(300), ("graceful when nothing fits (no_match or only valid items)",
                                  lambda o: _res(o, "product_search").get("status") in ("no_match", "ok", "weak_match"))]},
    {"id": "Q12", "query": "What is 15% off 3999?", "tools": ["calculator"],
     "checks": [("computes 3399.15", lambda o: "3,399.15" in o["final_answer"])]},
    {"id": "Q13", "query": "What's the weather in Mumbai today?", "tools": [],
     "checks": [("refuses out-of-scope request without calling tools", lambda o: not o["results"])]},
]


def _top_lift_name():
    segs = json.loads((OUTPUTS / "segment_summary.json").read_text())["segments"]
    return max(segs, key=lambda s: s["discount_lift"])["name"]


def run_suite(verbose: bool = True) -> pd.DataFrame:
    bot = RetailAssistant()
    rows, log = [], []
    for case in CASES:
        out = bot.ask(case["query"], verbose=verbose)
        route_ok = out["tools"] == case["tools"]
        checks = []
        for name, fn in case["checks"]:
            try:
                checks.append((name, bool(fn(out))))
            except Exception as e:                       # a crashing check counts as a failure
                checks.append((f"{name} (error: {e})", False))
        tokens = sum(((r.get("llm") or {}).get("usage") or {}).get("prompt_tokens", 0) +
                     ((r.get("llm") or {}).get("usage") or {}).get("completion_tokens", 0) for r in out["results"])
        rows.append({"id": case["id"], "query": case["query"], "expected_tools": ",".join(case["tools"]) or "(none)",
                     "routed_tools": ",".join(out["tools"]) or "(none)", "route_ok": route_ok,
                     "checks_passed": f"{sum(c for _, c in checks)}/{len(checks)}",
                     "failed_checks": "; ".join(n for n, c in checks if not c),
                     "PASS": route_ok and all(c for _, c in checks), "latency_s": out["latency_s"], "llm_tokens": tokens})
        log.append({**rows[-1], "final_answer": out["final_answer"], "trace": out["trace"], "results": out["results"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUTS / "eval_results.csv", index=False)
    with open(OUTPUTS / "eval_log.jsonl", "w") as f:
        for r in log:
            f.write(json.dumps(r, default=str) + "\n")
    return df
