"""
Policy-aware question answering: answer ONLY from retrieved policy evidence, with doc_id citations and a
verbatim quote. Conservative by design:
- if no policy chunk is similar enough, the assistant says it cannot find a policy (no guess, no citation);
- the LLM must return {"answer", "citations", "quotes", "covered"}; citations must be retrieved doc_ids and
  quotes must appear word for word in the cited policy, otherwise the answer is replaced by a safe,
  purely extractive one.
"""

import re

from src.config import MIN_POLICY_SIM, TOP_K_POLICIES
from src.llm import chat_json
from src.retrieval import policy_index

SYSTEM = """You answer customer questions about store policies using ONLY the POLICY EVIDENCE provided.
Rules:
- Do not use outside knowledge or assume rules that are not written in the evidence.
- Quote the exact sentence(s) you rely on, word for word.
- Cite each policy you use by its doc_id (e.g. POL_RETURNS).
- If the evidence does not fully answer the question, say what the policy does say and that the rest is
  not specified; set "covered" to false. Recommend contacting customer support in that case.
- Keep the answer to 2-4 sentences, starting with a direct yes / no / it depends when possible.
Return JSON only: {"answer": "...", "citations": ["POL_..."], "quotes": ["exact sentence"], "covered": true}"""

REFUSAL = ("I could not find a store policy that covers this question, so I can't give a reliable answer. "
           "Please contact customer support.")


def _norm(s):
    return " ".join(s.lower().split()).rstrip(".")


def validate(ans: dict, hits: list[dict]) -> list[str]:
    retrieved = {h["doc_id"]: h["full_text"] for h in hits}
    issues = []
    cites = ans.get("citations") or []
    if not cites:
        issues.append("no citation")
    for c in cites:
        if c not in retrieved:
            issues.append(f"citation {c} was not retrieved")
    cited_text = " ".join(_norm(retrieved.get(c, "")) for c in cites)
    for q in ans.get("quotes") or []:
        if _norm(q) not in cited_text:
            issues.append(f"quote not verbatim in cited policy: '{q[:60]}'")
    if not ans.get("quotes"):
        issues.append("no quote")
    return issues


def _extractive(question, hits):
    best = hits[0]
    return {"answer": f"According to {best['doc_id']}: \"{best['text']}\"" +
                      (f" Also: \"{hits[1]['text']}\"" if len(hits) > 1 and hits[1]["doc_id"] == best["doc_id"] else ""),
            "citations": [best["doc_id"]],
            "quotes": [best["text"]] + ([hits[1]["text"]] if len(hits) > 1 and hits[1]["doc_id"] == best["doc_id"] else []),
            "covered": None}


def answer_policy(question: str, k: int = TOP_K_POLICIES) -> dict:
    hits = policy_index().search(question, k=k)
    trace = ["retrieved policy chunks: " + ", ".join(f"{h['chunk_id']}({h['score']:.2f})" for h in hits)]
    if not hits or hits[0]["score"] < MIN_POLICY_SIM:
        trace.append(f"best similarity below {MIN_POLICY_SIM} -> refuse instead of guessing")
        return {"tool": "policy_qa", "status": "not_covered", "question": question, "answer": REFUSAL,
                "citations": [], "quotes": [], "evidence": [], "trace": trace}
    hits = [h for h in hits if h["score"] >= MIN_POLICY_SIM]
    # give the LLM the full text of every policy that had a relevant chunk
    seen, evidence = set(), []
    for h in hits:
        if h["doc_id"] not in seen:
            seen.add(h["doc_id"])
            evidence.append(f"[{h['doc_id']}] ({h['title']}) {h['full_text']}")
    resp = chat_json(SYSTEM, f"Question: {question}\n\nPOLICY EVIDENCE:\n" + "\n".join(evidence))
    ans = resp["data"] if resp else _extractive(question, hits)
    issues = validate(ans, hits)
    if issues and resp:
        trace.append("validation failed (" + "; ".join(issues) + ") -> replaced with extractive answer")
        ans, issues = _extractive(question, hits), []
    trace.append("validation: " + ("; ".join(issues) if issues else "citations retrieved, quotes verbatim"))
    return {"tool": "policy_qa", "status": "ok" if ans.get("covered", True) is not False else "partially_covered",
            "question": question, **{k2: ans.get(k2) for k2 in ("answer", "citations", "quotes", "covered")},
            "evidence": [{"chunk_id": h["chunk_id"], "score": h["score"], "text": h["text"]} for h in hits],
            "llm": {k2: resp[k2] for k2 in ("usage", "latency_s", "cached")} if resp else None, "trace": trace}
