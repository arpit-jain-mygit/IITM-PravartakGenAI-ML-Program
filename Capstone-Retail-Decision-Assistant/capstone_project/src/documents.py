"""
Phase 3 - turn structured records into retrieval-ready text documents.

Product documents: one per catalogue item (item_id). The text combines title, description, brand, category,
attributes, price context, tags, use case and the review, because shoppers search with all of these
("black formal shoes for office under Rs 4000"). Structured fields are also kept as metadata so that hard
constraints (budget, category, colour) are applied as filters rather than left to embedding similarity.

Policy documents: the 5 course policies, kept whole (each is only 2 sentences) and also split into
sentence-level chunks, so an answer can quote exactly the sentence that supports it.
"""

import json
import re

import pandas as pd

from src.config import PROCESSED
from src.data_prep import load_policies

POLICY_TITLES = {"POL_RETURNS": "Returns policy", "POL_EXCHANGE": "Exchange policy",
                 "POL_SHIPPING": "Shipping policy", "POL_REFUND": "Refund policy", "POL_WARRANTY": "Warranty policy"}

META_COLS = ["item_id", "product_id", "product_title", "category", "sub_category", "brand", "color", "size_filled",
             "material", "season", "final_price", "base_price", "discount_percent", "avg_product_rating",
             "stock_level", "is_returnable", "recommended_use_case", "occasion_tags", "style_tags", "price_band",
             "review_sentiment_label", "customer_review_text", "rating_given"]


def product_text(r) -> str:
    disc = f", {int(r.discount_percent)}% off list price Rs {r.base_price:,.0f}" if r.discount_percent else ""
    return (f"{r.product_title}. {r.product_description} "
            f"Brand: {r.brand}. Category: {r.category} > {r.sub_category}. Colour: {r.color}. "
            f"Material: {r.material}. Season: {r.season}. Size: {r.size_filled}. "
            f"Price: Rs {r.final_price:,.0f}{disc} ({r.price_band}). Average rating: {r.avg_product_rating}/5. "
            f"Use case: {r.recommended_use_case}. Style tags: {r.style_tags}. Occasion: {r.occasion_tags}. "
            f"{'Returnable' if r.is_returnable else 'Not returnable'}. "
            f"Customer review: \"{r.customer_review_text}\"")


def build_product_docs(df: pd.DataFrame) -> list[dict]:
    docs = [{"doc_id": r.item_id, "text": product_text(r), "meta": {c: getattr(r, c) for c in META_COLS}}
            for r in df.itertuples(index=False)]
    for d in docs:   # make metadata JSON-safe
        d["meta"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in d["meta"].items()}
    return docs


def build_policy_docs() -> tuple[list[dict], list[dict]]:
    pol = load_policies()
    full = [{"doc_id": k, "title": POLICY_TITLES.get(k, k), "text": v} for k, v in pol.items()]
    chunks = []
    for k, v in pol.items():
        for i, sent in enumerate(s.strip() for s in re.split(r"(?<=\.)\s+", v) if s.strip()):
            chunks.append({"doc_id": k, "chunk_id": f"{k}#{i + 1}", "title": POLICY_TITLES.get(k, k),
                           "text": sent, "full_text": v})
    return full, chunks


def write_all(df: pd.DataFrame):
    products = build_product_docs(df)
    full, chunks = build_policy_docs()
    with open(PROCESSED / "product_docs.jsonl", "w") as f:
        for d in products:
            f.write(json.dumps(d) + "\n")
    with open(PROCESSED / "policy_docs.jsonl", "w") as f:
        for d in chunks:
            f.write(json.dumps(d) + "\n")
    return products, full, chunks


def load_product_docs() -> list[dict]:
    return [json.loads(l) for l in open(PROCESSED / "product_docs.jsonl")]


def load_policy_chunks() -> list[dict]:
    return [json.loads(l) for l in open(PROCESSED / "policy_docs.jsonl")]
