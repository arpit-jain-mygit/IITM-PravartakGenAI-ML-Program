"""
Thin wrapper around an OpenAI-compatible API (OpenAI or the Vocareum-provided endpoint).

- embed(texts)       -> L2-normalised numpy matrix (so inner product = cosine similarity)
- chat_json(...)     -> parsed JSON dict from the LLM (temperature 0, fixed seed, JSON mode)

Both are cached on disk (keyed by model + exact input), so re-running a notebook gives identical results and
costs nothing. In OFFLINE mode embeddings come from a local sentence-transformers model and chat_json
returns None, which tells callers to use their deterministic fallback.
"""

import hashlib
import json
import os
import sqlite3
import time
from functools import lru_cache

import numpy as np

from src.config import CACHE, EMBED_MODEL, LLM_MODEL, LOCAL_EMBED_MODEL, OFFLINE, SEED, TEMPERATURE

USAGE = {"llm_calls": 0, "cached_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "embedding_tokens": 0}


def _key(*parts) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()


@lru_cache(maxsize=1)
def _db():
    con = sqlite3.connect(str(CACHE / "llm_cache.sqlite"), check_same_thread=False)
    con.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
    return con


def _get(k):
    row = _db().execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
    return json.loads(row[0]) if row else None


def _put(k, v):
    _db().execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (k, json.dumps(v)))
    _db().commit()


@lru_cache(maxsize=1)
def client():
    from openai import OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to capstone_project/.env (or run with RETAIL_OFFLINE=1).")
    return OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None, timeout=60, max_retries=3)


@lru_cache(maxsize=1)
def _local_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(LOCAL_EMBED_MODEL)


def embedding_model_name() -> str:
    return LOCAL_EMBED_MODEL if OFFLINE else EMBED_MODEL


def embed(texts: list[str], batch: int = 128) -> np.ndarray:
    model = embedding_model_name()
    vecs, todo = [None] * len(texts), []
    for i, t in enumerate(texts):
        hit = _get(_key("emb", model, t))
        if hit is None:
            todo.append(i)
        else:
            vecs[i] = hit
    for s in range(0, len(todo), batch):
        idx = todo[s:s + batch]
        chunk = [texts[i] for i in idx]
        if OFFLINE:
            out = _local_model().encode(chunk, normalize_embeddings=True).tolist()
        else:
            resp = client().embeddings.create(model=model, input=chunk)
            out = [d.embedding for d in resp.data]
            USAGE["embedding_tokens"] += resp.usage.total_tokens
        for i, v in zip(idx, out):
            vecs[i] = v
            _put(_key("emb", model, texts[i]), v)
    m = np.asarray(vecs, dtype="float32")
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def chat_json(system: str, user: str, max_tokens: int = 600) -> dict | None:
    """Returns {"data": parsed_json, "usage": {...}, "latency_s": float, "cached": bool} or None offline."""
    if OFFLINE:
        return None
    k = _key("chat", LLM_MODEL, TEMPERATURE, SEED, system, user)
    hit = _get(k)
    if hit is not None:
        USAGE["cached_calls"] += 1
        return {**hit, "cached": True}
    t0 = time.perf_counter()
    resp = client().chat.completions.create(
        model=LLM_MODEL, temperature=TEMPERATURE, seed=SEED, max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    text = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {"_parse_error": text}
    out = {"data": data, "latency_s": round(time.perf_counter() - t0, 3),
           "usage": {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}}
    USAGE["llm_calls"] += 1
    USAGE["prompt_tokens"] += resp.usage.prompt_tokens
    USAGE["completion_tokens"] += resp.usage.completion_tokens
    _put(k, out)
    return {**out, "cached": False}


def mode_label() -> str:
    return (f"OFFLINE (local {LOCAL_EMBED_MODEL} + deterministic extractive generator)" if OFFLINE
            else f"LIVE ({LLM_MODEL} + {EMBED_MODEL})")
