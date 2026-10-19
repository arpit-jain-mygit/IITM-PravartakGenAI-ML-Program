"""
The agent's three tools and the FAISS index behind policy_search.

Tool contracts
- policy_search(query, k)  -> list of {doc_id, date, snippet, score}; never raises
- read_policy(doc_id)      -> full policy text; the underlying function raises PolicyNotFoundError,
                              the tool wrapper turns it into a soft error message the agent can act on
- calculator(expression)   -> number; safe arithmetic only, division by zero -> CalculatorError (soft)
"""

from __future__ import annotations

import ast
import difflib
import json
import operator
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from models import CalculatorError, PolicyDoc, PolicyNotFoundError, PolicyToolError, SearchHit

DATA_FILE = Path(__file__).parent / "data" / "policies.jsonl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"     # local, open-source, CPU


# ---------------------------------------------------------------------------
# Corpus + index (built once per process)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_policies(path: Path = DATA_FILE) -> dict[str, PolicyDoc]:
    docs = [PolicyDoc(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    return {d.doc_id: d for d in docs}


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})


def _sentences(text: str, size: int = 2) -> list[str]:
    """Split a policy into small overlapping windows of 2 sentences, so hits are precise."""
    sents = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    sents = [s if s.endswith(".") else s + "." for s in sents]
    return [" ".join(sents[i:i + size]) for i in range(0, len(sents), size - 1 or 1)] or [text]


@lru_cache(maxsize=1)
def get_index() -> FAISS:
    chunks = [Document(page_content=w, metadata={"doc_id": d.doc_id, "date": d.date})
              for d in load_policies().values() for w in _sentences(d.text)]
    return FAISS.from_documents(chunks, get_embeddings())


# ---------------------------------------------------------------------------
# Plain functions (typed, unit-testable)
# ---------------------------------------------------------------------------
def search_policies(query: str, k: int = 4) -> list[SearchHit]:
    k = max(1, min(int(k), 8))                      # clamp: protects the context budget
    hits = get_index().similarity_search_with_score(query, k=k * 2)
    out, seen = [], set()
    for doc, score in hits:                         # keep the best snippet per (doc, snippet)
        key = (doc.metadata["doc_id"], doc.page_content)
        if key not in seen:
            seen.add(key)
            out.append(SearchHit(doc_id=doc.metadata["doc_id"], date=doc.metadata["date"],
                                 snippet=doc.page_content, score=round(float(score), 4)))
    return out[:k]


def read_policy_text(doc_id: str) -> PolicyDoc:
    policies = load_policies()
    doc_id = doc_id.strip().strip("[]")
    if doc_id not in policies:
        suggestions = difflib.get_close_matches(doc_id, list(policies), n=3, cutoff=0.3) or sorted(policies)[:3]
        raise PolicyNotFoundError(doc_id, suggestions)
    return policies[doc_id]


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expression: str) -> float:
    """Evaluate +, -, *, /, %, ** on numbers only. No names, calls or attributes are allowed."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:
                raise CalculatorError("CalculatorError: division by zero")
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise CalculatorError("CalculatorError: exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise CalculatorError(f"CalculatorError: unsupported expression '{expression}'")
    try:
        tree = ast.parse(expression.replace(",", ""), mode="eval")
    except SyntaxError as e:
        raise CalculatorError(f"CalculatorError: could not parse '{expression}'") from e
    result = ev(tree)
    return int(result) if float(result).is_integer() else round(result, 6)


# ---------------------------------------------------------------------------
# LangChain tools (what the agent sees). Errors become short "ERROR ..." strings: fail softly.
# ---------------------------------------------------------------------------
@tool
def policy_search(query: str, k: int = 4) -> str:
    """Search the internal policy corpus. Returns JSON hits with doc_id, date and a snippet.
    Use this first for any policy question, or after read_policy reports an unknown doc_id."""
    return json.dumps([h.model_dump() for h in search_policies(query, k)])


@tool
def read_policy(doc_id: str) -> str:
    """Return the full text and date of one policy by its exact doc_id (for example leave_v3)."""
    try:
        d = read_policy_text(doc_id)
        return json.dumps({"doc_id": d.doc_id, "date": d.date, "text": d.text})
    except PolicyToolError as e:
        return f"ERROR {e}"


@tool
def calculator(expression: str) -> str:
    """Safely evaluate an arithmetic expression such as '12 * 4' or '2500 / 3'."""
    try:
        return str(safe_eval(expression))
    except PolicyToolError as e:
        return f"ERROR {e}"


TOOLS = [policy_search, read_policy, calculator]
