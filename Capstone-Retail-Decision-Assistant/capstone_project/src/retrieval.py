"""
Phase 4 - FAISS vector indexes for products and policies.

Vectors are L2-normalised, so an inner-product index (IndexFlatIP) returns cosine similarity.
The catalogue is small (1,500 items), so search is exact: we score every item, apply metadata filters
(budget, category, colour, in stock) and keep the top-k. Exact search keeps results deterministic.
"""

from functools import lru_cache

import faiss
import numpy as np

from src.documents import load_policy_chunks, load_product_docs
from src.llm import embed


class VectorIndex:
    def __init__(self, docs: list[dict], text_key: str = "text"):
        self.docs = docs
        vecs = embed([d[text_key] for d in docs])
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        self.dim = vecs.shape[1]

    def search(self, query: str, k: int = 5, filter_fn=None, min_sim: float = -1.0) -> list[dict]:
        q = embed([query])
        scores, ids = self.index.search(q, len(self.docs))       # exact, all items scored
        hits = []
        for s, i in zip(scores[0], ids[0]):
            d = self.docs[i]
            if s < min_sim:
                break                                              # sorted descending, so we can stop
            if filter_fn is None or filter_fn(d):
                hits.append({**d, "score": round(float(s), 4)})
                if len(hits) == k:
                    break
        return hits


@lru_cache(maxsize=1)
def product_index() -> VectorIndex:
    return VectorIndex(load_product_docs())


@lru_cache(maxsize=1)
def policy_index() -> VectorIndex:
    return VectorIndex(load_policy_chunks())


def retrieval_hit_rate(index: VectorIndex, queries: list[str], target_ids: list[str], targets_meta: list[dict],
                       k: int = 5) -> dict:
    """Retrieval inspection on the dataset's own customer_query column: is the item the query was written for
    in the top-k, and do the top-k share its sub-category / colour?"""
    hit, subcat, colour = [], [], []
    for q, tid, tm in zip(queries, target_ids, targets_meta):
        res = index.search(q, k=k)
        hit.append(tid in [r["doc_id"] for r in res])
        subcat.append(np.mean([r["meta"]["sub_category"] == tm["sub_category"] for r in res]))
        colour.append(np.mean([r["meta"]["color"] == tm["color"] for r in res]))
    return {"queries": len(queries), f"exact_item_hit@{k}": round(float(np.mean(hit)), 3),
            f"sub_category_precision@{k}": round(float(np.mean(subcat)), 3),
            f"colour_precision@{k}": round(float(np.mean(colour)), 3)}
