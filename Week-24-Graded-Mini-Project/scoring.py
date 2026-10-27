"""
Draft rubric scores for every logged answer, written to logs/manual_scores.csv for HUMAN REVIEW.

The draft applies the Task 7 rubric mechanically, using the gold answer key in questions.json:
  relevance      0-2  key facts present (all = 2, some = 1, none = 0)
  groundedness   0-2  answer stays within the retrieved chunks (citations only to retrieved pages,
                      gold page retrieved when facts are stated; a refusal is always grounded)
  citation       0-2  citations exist, all point to retrieved pages, at least one is the gold page
  conciseness    0-1  answer within 120 output tokens
  memory         0-2  turns 2-3 only: the turn was answered correctly in the context of the conversation
  refusal_correct 0-1 boundary questions only: said "don't know" and cited nothing

Read each answer and adjust any score you disagree with, then set reviewed = True.
Existing rows are never overwritten, so your edits are safe when the notebook is re-run.
"""

import re
from pathlib import Path

import pandas as pd

COLS = ["run_id", "qid", "relevance", "groundedness", "citation", "conciseness", "memory",
        "refusal_correct", "reviewed", "note"]
REFUSAL = "don't know"


def _has(answer, alternatives):
    a = answer.lower()
    for alt in alternatives:
        pat = rf"(?<![\d.]){re.escape(alt)}(?![\d])" if alt[0].isdigit() else re.escape(alt.lower())
        if re.search(pat, a):
            return True
    return False


def draft_row(r, q):
    ans, cited, used, gold = r["answer"], set(r["citations"]), set(r["used_pages"]), set(q["gold_pages"])
    refused = REFUSAL in ans.lower()
    notes = []
    s = {"run_id": r["run_id"], "qid": r["qid"], "conciseness": int(r["answer_tokens"] <= 120),
         "memory": None, "refusal_correct": None, "reviewed": False}

    if q["type"] == "boundary":
        ok = refused and not cited
        s.update(relevance=2 * ok, groundedness=2 if ok or refused else 0, citation=2 if not cited else 0,
                 refusal_correct=int(ok))
        notes.append("correct refusal" if ok else "answered an out-of-corpus question (possible hallucination)")
    elif refused:
        s.update(relevance=0, groundedness=2, citation=0)
        notes.append("refused although the answer is in the pack" +
                     ("" if used & gold else " - gold page was not retrieved"))
    else:
        hits = sum(_has(ans, alts) for alts in q["must_contain"])
        s["relevance"] = 2 if hits == len(q["must_contain"]) else (1 if hits else 0)
        valid = bool(cited) and cited <= used
        s["citation"] = 2 if valid and cited & gold else (1 if cited & gold or valid else 0)
        if s["citation"] == 2 and re.search(r"\[[^\[\]#]+?#\d+#[\w-]+\]", ans):
            s["citation"] = 1          # right page, but format [Title#page#chunk] instead of [Title#page]
            notes.append("citation format includes the chunk number ([Title#page#chunk]) - page is correct")
        if cited - used:
            s["groundedness"] = 0
            notes.append(f"cites pages that were not retrieved: {sorted(cited - used)}")
        elif not used & gold:
            s["groundedness"] = 1
            notes.append("gold page not retrieved - check the answer is supported")
        else:
            s["groundedness"] = 2 if s["relevance"] == 2 else 1
        if s["relevance"] < 2:
            notes.append(f"{hits}/{len(q['must_contain'])} key facts found")
        if not cited:
            notes.append("no citations")

    if q["turn"] >= 2:
        s["memory"] = 2 if s["relevance"] == 2 else (1 if s["relevance"] == 1 else 0)
        if q["type"] == "follow-up":
            notes.append(f"standalone query: {r['standalone_question']}")
    s["note"] = "; ".join(notes) or "ok"
    return s


def ensure_scores(log, questions, path):
    """Create or extend the scores file with drafted rows; never touch rows that already exist."""
    path = Path(path)
    qmap = {q["qid"]: q for q in questions}
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLS)
    have = set(zip(existing["run_id"], existing["qid"]))
    new = [draft_row(r, qmap[r["qid"]]) for _, r in log.iterrows() if (r["run_id"], r["qid"]) not in have]
    if new:
        frames = [f for f in (existing, pd.DataFrame(new, columns=COLS)) if len(f)]
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return len(new)
