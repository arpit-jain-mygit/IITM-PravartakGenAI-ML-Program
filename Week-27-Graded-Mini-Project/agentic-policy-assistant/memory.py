"""
FAISS-backed semantic memory of VERIFIED facts.

Write rule: a fact is stored only if (1) it cites at least one doc_id, (2) every cited doc_id exists in the
corpus, and (3) every quote appears verbatim in one of the cited policies. The graph also only calls write
after the critic has approved the answer.

Read rule ("cheapest sufficient first"): find stored facts whose question is semantically close enough
(cosine similarity >= threshold). Among those, pick the one that is cheapest to verify (fewest doc_ids to
re-read), breaking ties by similarity. That fact is passed to the agent as a hint, so it can confirm with a
single read_policy call instead of searching.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from models import DOC_ID_RE, QUOTE_RE, MemoryFact
from tools import get_embeddings, load_policies

READ_THRESHOLD = 0.80        # cosine similarity needed to treat a stored question as "the same question"


def _norm(s: str) -> str:
    return " ".join(s.lower().replace("’", "'").split())


def verify_fact(answer: str) -> tuple[bool, list[str], list[str], str]:
    """Return (ok, doc_ids, quotes, reason). Used by both the memory writer and the critic."""
    policies = load_policies()
    doc_ids = sorted(set(DOC_ID_RE.findall(answer)))
    quotes = [q.strip() for q in QUOTE_RE.findall(answer)]
    if not doc_ids:
        return False, doc_ids, quotes, "no doc_id token"
    unknown = [d for d in doc_ids if d not in policies]
    if unknown:
        return False, doc_ids, quotes, f"unknown doc_id(s): {unknown}"
    cited_text = _norm(" ".join(policies[d].text for d in doc_ids))
    bad = [q for q in quotes if _norm(q).rstrip(".") not in cited_text]
    if bad:
        return False, doc_ids, quotes, f"quote(s) not found verbatim in cited policies: {bad}"
    return True, doc_ids, quotes, "verified"


class FactMemory:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.store: FAISS | None = None
        if (self.dir / "index.faiss").exists():
            self.store = FAISS.load_local(str(self.dir), get_embeddings(), allow_dangerous_deserialization=True)

    # -- read ---------------------------------------------------------------
    def read(self, question: str, k: int = 4) -> dict | None:
        if self.store is None:
            return None
        hits = self.store.similarity_search_with_score(question, k=k)
        candidates = []
        for doc, dist in hits:
            sim = 1 - float(dist) / 2        # normalised vectors: L2 distance squared -> cosine similarity
            if sim >= READ_THRESHOLD:
                fact = json.loads(doc.metadata["fact"])
                candidates.append({**fact, "similarity": round(sim, 3), "cost": len(fact["doc_ids"])})
        if not candidates:
            return None
        return sorted(candidates, key=lambda c: (c["cost"], -c["similarity"]))[0]

    # -- write --------------------------------------------------------------
    def write(self, question: str, answer: str) -> tuple[bool, str]:
        ok, doc_ids, quotes, reason = verify_fact(answer)
        if not ok:
            return False, f"not written: {reason}"
        if self.read(question) and self.read(question)["similarity"] >= 0.98:
            return False, "not written: already in memory"
        fact = MemoryFact(question=question, answer=answer, doc_ids=doc_ids, quotes=quotes,
                          created_at=datetime.now().isoformat(timespec="seconds"))
        doc = Document(page_content=question, metadata={"fact": fact.model_dump_json()})
        if self.store is None:
            self.store = FAISS.from_documents([doc], get_embeddings())
        else:
            self.store.add_documents([doc])
        self.dir.mkdir(parents=True, exist_ok=True)
        self.store.save_local(str(self.dir))
        return True, f"written ({', '.join(doc_ids)})"

    def size(self) -> int:
        return 0 if self.store is None else self.store.index.ntotal

    def clear(self):
        self.store = None
        for f in ["index.faiss", "index.pkl"]:
            (self.dir / f).unlink(missing_ok=True)
