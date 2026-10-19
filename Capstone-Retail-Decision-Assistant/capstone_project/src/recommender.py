"""
Phase 4 - grounded product recommendation.

Pipeline for a shopper query:
 1. parse explicit constraints with simple, inspectable rules (budget, sub-category, colour, premium intent)
 2. semantic retrieval over product documents with those constraints as hard metadata filters
    (if nothing survives, relax colour, then sub-category, and record that in the trace)
 3. the LLM writes a structured answer ONLY from the retrieved items:
    {"recommended_products": [{"item_id", "title", "price", "why"}], "explanation", "supporting_ids"}
 4. validation: every recommended item_id must be in the retrieved set and within budget;
    anything else is dropped. Weak or empty retrieval returns a clear "no good match" answer instead.
"""

import json
import re
from functools import lru_cache

from src.config import MIN_PRODUCT_SIM, TOP_K_PRODUCTS
from src.documents import load_product_docs
from src.llm import chat_json
from src.retrieval import product_index

SUBCATS = {
    "Sneakers": [r"sneaker"], "Running": [r"running", r"jogging", r"\brun\b"], "Formal": [r"formal shoes?", r"formal footwear", r"\bformals?\b(?!.*(shirt|wear|jacket))"],
    "Casual": [r"casual shoe", r"loafer"], "Watch": [r"watch"], "Backpack": [r"backpack", r"\bbag\b"],
    "Wallet": [r"wallet"], "Cap": [r"\bcap\b", r"\bhat\b"], "Jacket": [r"jacket"], "Jeans": [r"jeans", r"denim"],
    "Shirt": [r"\bshirts?\b"], "Tshirt": [r"t-?shirt", r"\btee\b"],
}
CATEGORY_WORDS = {"Shoes": [r"\bshoes?\b", r"footwear"], "Apparel": [r"clothes", r"clothing", r"apparel", r"outfit"],
                  "Accessories": [r"accessor"]}
COLOURS = ["black", "white", "blue", "grey", "gray", "red", "green", "brown", "olive", "navy", "beige", "pink", "yellow"]

SYSTEM = """You are a retail shopping assistant. Recommend products ONLY from the CANDIDATES list.
Rules:
- Use only facts present in the candidate text (price, brand, colour, rating, use case, review). Never invent products or facts.
- Recommend at most 3 items that best fit the shopper's request; respect any budget.
- Every recommended item must be referenced by its exact item_id.
- If no candidate fits, return an empty list and say so in the explanation.
Return JSON only:
{"recommended_products": [{"item_id": "...", "title": "...", "price": 0, "why": "one sentence grounded in the candidate text"}],
 "explanation": "2-3 sentences summarising why these fit", "supporting_ids": ["..."]}"""


@lru_cache(maxsize=1)
def known_brands() -> tuple:
    return tuple(sorted({d["meta"]["brand"] for d in load_product_docs()}))


def parse_constraints(query: str) -> dict:
    q = query.lower().replace(",", "")
    c = {}
    brands = [b for b in known_brands() if re.search(rf"\b{re.escape(b.lower())}\b", q)]
    if brands:
        c["brand"] = brands
    m = re.search(r"(?:under|below|less than|within|upto|up to|max(?:imum)?|budget(?: of)?)\s*(?:₹|rs\.?|inr)?\s*(\d{3,6})", q)
    if m:
        c["max_price"] = float(m.group(1))
    m = re.search(r"(?:above|over|more than)\s*(?:₹|rs\.?|inr)?\s*(\d{3,6})", q)
    if m:
        c["min_price"] = float(m.group(1))
    for sub, pats in SUBCATS.items():
        if any(re.search(p, q) for p in pats):
            c.setdefault("sub_category", []).append(sub)
    if "sub_category" not in c:
        for cat, pats in CATEGORY_WORDS.items():
            if any(re.search(p, q) for p in pats):
                c["category"] = cat
    cols = [("Grey" if w in ("grey", "gray") else w.title()) for w in COLOURS if re.search(rf"\b{w}\b", q)]
    if cols:
        c["color"] = cols
    if re.search(r"premium|luxury|high[- ]end|platinum|gold member", q):
        c["premium"] = True
    return c


def _filter(c: dict):
    def ok(d):
        m = d["meta"]
        if m["stock_level"] <= 0:
            return False
        if "max_price" in c and m["final_price"] > c["max_price"]:
            return False
        if "min_price" in c and m["final_price"] < c["min_price"]:
            return False
        if "sub_category" in c and m["sub_category"] not in c["sub_category"]:
            return False
        if "category" in c and m["category"] != c["category"]:
            return False
        if "color" in c and m["color"] not in c["color"]:
            return False
        if "brand" in c and m["brand"] not in c["brand"]:
            return False
        return True
    return ok


def retrieve_products(query: str, k: int = TOP_K_PRODUCTS) -> tuple[list[dict], dict, list[str]]:
    c = parse_constraints(query)
    trace = [f"constraints parsed: {json.dumps(c)}"]
    tried = dict(c)
    hits = product_index().search(query, k=k * 3, filter_fn=_filter(tried), min_sim=MIN_PRODUCT_SIM)
    for relax in ["color", "brand", "sub_category"]:                   # graceful relaxation, recorded in the trace
        if hits or relax not in tried:
            continue
        tried.pop(relax)
        trace.append(f"no items matched all constraints -> relaxed '{relax}'")
        hits = product_index().search(query, k=k * 3, filter_fn=_filter(tried), min_sim=MIN_PRODUCT_SIM)
    if c.get("premium") and hits:                              # premium intent: favour higher price band + rating
        hits.sort(key=lambda h: (-h["score"] - 0.05 * (h["meta"]["price_band"] == "Premium")
                                 - 0.02 * (h["meta"]["avg_product_rating"] - 4)))
        trace.append("premium intent -> boosted Premium price band and higher-rated items")
    hits = hits[:k]
    trace.append(f"retrieved {len(hits)} items (min similarity {MIN_PRODUCT_SIM}): " +
                 ", ".join(f"{h['doc_id']}({h['score']:.2f})" for h in hits))
    return hits, {"requested": c, "applied": tried}, trace


def _candidates_block(hits):
    return "\n".join(f"- item_id={h['doc_id']} | {h['text']}" for h in hits)


def _offline_answer(query, hits):
    top = hits[:3]
    return {"recommended_products": [
        {"item_id": h["doc_id"], "title": h["meta"]["product_title"], "price": h["meta"]["final_price"],
         "why": f"{h['meta']['sub_category']} by {h['meta']['brand']} in {h['meta']['color'].lower()}, "
                f"Rs {h['meta']['final_price']:,.0f}, rated {h['meta']['avg_product_rating']}/5, "
                f"suited to {h['meta']['recommended_use_case'].lower()}."} for h in top],
        "explanation": "Top matches by semantic similarity that satisfy the parsed constraints "
                       "(offline mode: template text instead of an LLM).",
        "supporting_ids": [h["doc_id"] for h in top]}


def validate(answer: dict, hits: list[dict], constraints: dict) -> tuple[dict, list[str]]:
    """Drop anything not grounded in the retrieved items or outside the budget."""
    allowed = {h["doc_id"]: h for h in hits}
    issues, kept = [], []
    for p in answer.get("recommended_products", []) or []:
        iid = p.get("item_id")
        if iid not in allowed:
            issues.append(f"dropped {iid}: not in retrieved context")
            continue
        price = allowed[iid]["meta"]["final_price"]
        if "max_price" in constraints and price > constraints["max_price"]:
            issues.append(f"dropped {iid}: price {price} over budget")
            continue
        p["price"] = price                                   # always report the catalogue price, not the LLM's
        p["title"] = allowed[iid]["meta"]["product_title"]
        kept.append(p)
    answer["recommended_products"] = kept
    answer["supporting_ids"] = [p["item_id"] for p in kept]
    return answer, issues


def recommend(query: str, k: int = TOP_K_PRODUCTS) -> dict:
    hits, cons, trace = retrieve_products(query, k)
    if not hits:
        return {"tool": "product_search", "status": "no_match", "query": query, "constraints": cons,
                "recommended_products": [], "supporting_ids": [], "retrieved": [], "trace": trace,
                "explanation": "I could not find products in the catalogue that match this request "
                               "(including after relaxing colour / sub-category). Try a higher budget or a broader description."}
    resp = chat_json(SYSTEM, f"Shopper request: {query}\nParsed constraints: {json.dumps(cons['applied'])}\n\n"
                             f"CANDIDATES:\n{_candidates_block(hits)}")
    answer = resp["data"] if resp else _offline_answer(query, hits)
    answer, issues = validate(answer, hits, cons["applied"])
    if not answer["recommended_products"]:
        issues.append("LLM returned no valid grounded item -> fell back to top retrieved items")
        answer, _ = validate(_offline_answer(query, hits), hits, cons["applied"])
    trace += [f"validation: {i}" for i in issues] or ["validation: all recommended items grounded and within budget"]
    weak = hits[0]["score"] < MIN_PRODUCT_SIM + 0.1
    return {"tool": "product_search", "status": "weak_match" if weak else "ok", "query": query, "constraints": cons,
            "retrieved": [{"item_id": h["doc_id"], "score": h["score"], "title": h["meta"]["product_title"],
                           "price": h["meta"]["final_price"], "rating": h["meta"]["avg_product_rating"]} for h in hits],
            **answer, "llm": {k2: resp[k2] for k2 in ("usage", "latency_s", "cached")} if resp else None,
            "trace": trace}
