"""
Acceptance tests and quick metrics over 8 canned questions.

For every question we record latency, agent tool steps, graph steps, has_docid, has_quote, whether the
expected policy was cited, whether the key facts are present, the critic verdict, memory hit/write and
pass/fail. Results are printed as a table and saved under runs/:
  runs/logs/<label>_<timestamp>.jsonl   one full record per question (answer, tool calls, node trace)
  runs/eval_<label>.csv                 the table
  runs/report_<label>.json              summary + all records (the "JSON run report")
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd

from agent import RUN_DIR, rel
from graph import NOT_FOUND, run_graph
from models import DOC_ID_RE, QUOTE_RE

CANNED = [
    {"id": "E1", "q": "What is our remote work limit? Cite and quote.", "expect_doc": "remote_work_v2",
     "facts": [["3 days"]], "kind": "factual"},
    {"id": "E2", "q": "How many annual leave days and can they be carried over? Cite and quote.", "expect_doc": "leave_v3",
     "facts": [["24"], ["10"]], "kind": "factual"},
    {"id": "E3", "q": "What's the travel meals cap (INR) and any receipt rule? Cite and quote.", "expect_doc": "travel_v4",
     "facts": [["2,500", "2500"], ["500"]], "kind": "factual"},
    {"id": "E4", "q": "Is MFA required for VPN? Cite and quote.", "expect_doc": "security_v1",
     "facts": [["mandatory", "required"]], "kind": "factual"},
    {"id": "E5", "q": "Read policy remote_work_v9.9 and answer the limit.", "expect_doc": "remote_work_v2",
     "facts": [["3 days"]], "kind": "recovery"},
    {"id": "E6", "q": "Compute 12 * 4 and then tell me meals cap with citation.", "expect_doc": "travel_v4",
     "facts": [["48"], ["2,500", "2500"]], "kind": "multi-tool"},
    {"id": "E7", "q": "What is the minimum password length? Cite and quote.", "expect_doc": "security_v1",
     "facts": [["14"]], "kind": "factual"},
    {"id": "E8", "q": "What is the CEO's annual salary?", "expect_doc": None, "facts": [], "kind": "boundary"},
]


def score(case: dict, out: dict) -> dict:
    ans = out.get("answer", "")
    cited = sorted(set(DOC_ID_RE.findall(ans)))
    has_docid, has_quote = bool(cited), bool(QUOTE_RE.search(ans))
    refused = NOT_FOUND in ans.lower()
    facts_ok = all(any(f.lower() in ans.lower() for f in alts) for alts in case["facts"])
    if case["kind"] == "boundary":
        passed = refused and not has_docid and out.get("status") == "ok"
    else:
        passed = (out.get("status") == "ok" and has_docid and has_quote and case["expect_doc"] in cited and facts_ok)
    tools = [c["tool"] for c in out.get("tool_calls", [])]
    return {
        "id": case["id"], "kind": case["kind"], "question": case["q"], "status": out.get("status"),
        "latency_s": out.get("latency_s"), "agent_steps": out.get("agent_steps", 0),
        "graph_steps": out.get("graph_steps", 0), "has_docid": has_docid, "has_quote": has_quote,
        "cited": ",".join(cited), "expected_doc_cited": (case["expect_doc"] in cited) if case["expect_doc"] else not has_docid,
        "facts_ok": facts_ok if case["facts"] else refused, "critic_valid": out.get("critic", {}).get("valid"),
        "retries": out.get("retries", 0), "memory_hit": bool(out.get("memory_hit")),
        "memory_written": out.get("memory_written", False), "recovered": "read_policy" in tools and any(
            c["output_preview"].startswith("ERROR") for c in out.get("tool_calls", [])),
        "tokens": out.get("tokens", 0), "PASS": passed,
    }


def run_eval(app, label: str, max_steps: int = 8) -> pd.DataFrame:
    (RUN_DIR / "logs").mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = RUN_DIR / "logs" / f"{label}_{ts}.jsonl"
    rows, records = [], []
    for case in CANNED:
        out = run_graph(case["q"], app=app, max_steps=max_steps)
        row = score(case, out)
        rows.append(row)
        rec = {**row, "answer": out.get("answer"), "plan": out.get("plan"), "tool_calls": out.get("tool_calls"),
               "trace": out.get("trace"), "critic": out.get("critic"), "memory_hit_detail": out.get("memory_hit")}
        records.append(rec)
        with log_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    df = pd.DataFrame(rows)
    df.to_csv(RUN_DIR / f"eval_{label}.csv", index=False)
    summary = {"label": label, "timestamp": ts, "questions": len(df), "pass_rate": round(df["PASS"].mean(), 3),
               "meets_80pct": bool(df["PASS"].mean() >= 0.8), "avg_latency_s": round(df["latency_s"].mean(), 3),
               "avg_agent_steps": round(df["agent_steps"].mean(), 2), "memory_hits": int(df["memory_hit"].sum()),
               "memory_writes": int(df["memory_written"].sum()), "log_file": rel(log_path)}
    json.dump({"summary": summary, "records": records}, open(RUN_DIR / f"report_{label}.json", "w"), indent=2, default=str)
    return df


def print_table(df: pd.DataFrame, label: str):
    cols = ["id", "kind", "status", "latency_s", "agent_steps", "graph_steps", "has_docid", "has_quote",
            "cited", "critic_valid", "retries", "memory_hit", "memory_written", "PASS"]
    print(f"\n=== Evaluation: {label} ===")
    print(df[cols].to_string(index=False))
    print(f"\nPass rate: {df['PASS'].mean():.0%} ({int(df['PASS'].sum())}/{len(df)})   "
          f"avg latency {df['latency_s'].mean():.2f}s   avg tool steps {df['agent_steps'].mean():.2f}   "
          f"memory hits {int(df['memory_hit'].sum())}   memory writes {int(df['memory_written'].sum())}")
    print("Acceptance (>= 80% pass):", "MET" if df["PASS"].mean() >= 0.8 else "NOT MET")


def run_agent_eval(executor, label: str = "v1_agent_only") -> pd.DataFrame:
    """Evaluation Version 1 (Part 1): the bare ReAct agent, no graph / memory / critic."""
    from agent import run_agent
    (RUN_DIR / "logs").mkdir(parents=True, exist_ok=True)
    log_path = RUN_DIR / "logs" / f"{label}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    rows = []
    for case in CANNED:
        res = run_agent(case["q"], executor=executor)
        out = {"answer": res.answer, "status": "ok", "latency_s": res.latency_s, "agent_steps": res.steps,
               "tool_calls": res.tool_calls, "tokens": res.input_tokens + res.output_tokens}
        row = score(case, out)
        rows.append({k: row[k] for k in ["id", "kind", "latency_s", "agent_steps", "has_docid", "has_quote",
                                          "cited", "expected_doc_cited", "facts_ok", "recovered", "tokens", "PASS"]})
        with log_path.open("a") as f:
            f.write(json.dumps({**rows[-1], "question": case["q"], "answer": res.answer,
                                "tool_calls": res.tool_calls}, default=str) + "\n")
    df = pd.DataFrame(rows)
    df.to_csv(RUN_DIR / f"eval_{label}.csv", index=False)
    return df
