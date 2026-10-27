"""
2-page analysis summary (the "1-2 page PDF summarising your analysis" deliverable), built from the run outputs.

    python make_summary.py                     # live results   -> Week 27_Graded Mini Project_Jain_Summary.pdf
    POLICY_OFFLINE=1 python make_summary.py    # offline dry run -> agentic-policy-assistant/runs_offline/summary_offline.pdf

Every number is read from runs/ and results/, so the summary always matches the notebook.
"""

import json
import os
import subprocess
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).parent
REPO = HERE / "agentic-policy-assistant"
OFFLINE = os.getenv("POLICY_OFFLINE") == "1"
RUN = REPO / ("runs_offline" if OFFLINE else "runs")
RES = REPO / ("results_offline" if OFFLINE else "results")
sfx = "_offline" if OFFLINE else ""
OUT = RUN / "summary_offline.pdf" if OFFLINE else HERE / "Week 27_Graded Mini Project_Jain_Summary.pdf"

# ---------------------------------------------------------------- results
v1 = pd.read_csv(RUN / "eval_v1_agent_only.csv")
cold = pd.read_csv(RUN / f"eval_cold{sfx}.csv")
warm = pd.read_csv(RUN / f"eval_warm{sfx}.csv")
summ = json.load(open(RES / "summary.json"))
model = summ["model"]

# ---------------------------------------------------------------- styles
ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.8, leading=11.6, spaceAfter=3)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=14, textColor=colors.HexColor("#1f3b57"), spaceAfter=1)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=10.5, textColor=colors.HexColor("#1f3b57"), spaceBefore=5, spaceAfter=2)
small = ParagraphStyle("s", parent=body, fontSize=7.6, leading=9.4, textColor=colors.HexColor("#555"))
cap = ParagraphStyle("c", parent=small, alignment=TA_CENTER)
P = lambda t, s=body: Paragraph(t, s)
B = lambda t: Paragraph(t, ParagraphStyle("bu", parent=body, leftIndent=9), bulletText="•")


def table(rows, widths, fs=7.6, hl_col=None):
    cell = ParagraphStyle("tc", parent=body, fontSize=fs, leading=fs + 2, spaceAfter=0)
    head = ParagraphStyle("th", parent=cell, textColor=colors.white, fontName="Helvetica-Bold")
    t = Table([[Paragraph(str(c), head if i == 0 else cell) for c in r] for i, r in enumerate(rows)],
              colWidths=[w * cm for w in widths])
    st = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")), ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#bbb")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
          ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f6f9")])]
    if hl_col is not None:
        for r in range(1, len(rows)):
            ok = str(rows[r][hl_col]) in ("PASS", "True")
            st.append(("BACKGROUND", (hl_col, r), (hl_col, r), colors.HexColor("#d3f9d8" if ok else "#ffe3e3")))
    t.setStyle(TableStyle(st))
    return t


def img(name, w):
    p = RES / name
    iw, ih = ImageReader(str(p)).getSize()
    return Image(str(p), width=w * cm, height=w * cm * ih / iw)


pct = lambda s: f"{s.mean():.0%} ({int(s.sum())}/{len(s)})"
ans = lambda df, c: pct(df.loc[df.kind != "boundary", c])        # doc_id / quote only expected on answerable questions
avg = lambda df, c: df[c].mean()
steps_saved = 1 - avg(warm, "agent_steps") / avg(cold, "agent_steps")
tok_saved = 1 - avg(warm, "tokens") / avg(cold, "tokens") if avg(cold, "tokens") else 0
lat_c, lat_w = avg(cold, "latency_s"), avg(warm, "latency_s")
rec = cold[cold.kind == "recovery"].iloc[0]
retries = int(cold.retries.sum() + warm.retries.sum())

s = []
s += [P("Policy Assistant Agent - Analysis Summary", h1),
      P(f"Week 27 Graded Mini Project · Arpit Jain · Model: {model} (temperature 0) · Full details: notebook, repository and full report", small),
      Spacer(1, 3)]

# ---- 1. what was built
s += [P("1. What was built", h2),
      P("A single tool-calling agent that answers employee questions <b>only</b> from an 8-document local policy corpus "
        "(data/policies.jsonl, one deliberately superseded version), orchestrated by a LangGraph state machine with a step "
        "budget, a deterministic critic and a FAISS memory of verified facts. Modules: models (pydantic schemas, typed errors), "
        "tools (policy_search over FAISS, read_policy with a typed PolicyNotFoundError returned as a soft error, safe calculator), "
        "prompts (acceptance rules), agent (ReAct AgentExecutor, max 6 iterations, verbose trace), memory, graph, eval, app (CLI)."),
      img("01_graph.png", 15.5), P("Graph: plan → memory_read → agent_answer → critic → memory_write → stop; one retry on "
                                   "critic failure; budget router after every node.", cap)]

# ---- 2. results
s += [P("2. Acceptance results", h2),
      table([["Run", "Pass rate", "Has doc_id*", "Has quote*", "Avg tool steps", "Avg latency (s)", "Avg tokens", "Memory hits / writes"],
             ["V1: agent only (no graph/memory/critic)", pct(v1.PASS), ans(v1, "has_docid"), ans(v1, "has_quote"),
              f"{avg(v1, 'agent_steps'):.2f}", f"{avg(v1, 'latency_s'):.2f}", f"{avg(v1, 'tokens'):.0f}", "- / -"],
             ["Graph, cold run (memory cleared)", pct(cold.PASS), ans(cold, "has_docid"), ans(cold, "has_quote"),
              f"{avg(cold, 'agent_steps'):.2f}", f"{lat_c:.2f}", f"{avg(cold, 'tokens'):.0f}",
              f"{int(cold.memory_hit.sum())} / {int(cold.memory_written.sum())}"],
             ["Graph, warm run (memory kept)", pct(warm.PASS), ans(warm, "has_docid"), ans(warm, "has_quote"),
              f"{avg(warm, 'agent_steps'):.2f}", f"{lat_w:.2f}", f"{avg(warm, 'tokens'):.0f}",
              f"{int(warm.memory_hit.sum())} / {int(warm.memory_written.sum())}"]],
            [5.2, 1.7, 1.6, 1.6, 1.6, 1.7, 1.5, 2.3]),
      Spacer(1, 3),
      table([["id", "kind", "question", "cited", "tool steps (cold→warm)", "PASS"]] +
            [[r.id, r.kind, r.question, r.cited if isinstance(r.cited, str) else "-",
              f"{r.agent_steps} → {warm.loc[i, 'agent_steps']}", "PASS" if r.PASS and warm.loc[i, "PASS"] else "FAIL"]
             for i, r in cold.iterrows()], [0.8, 1.6, 7.6, 2.6, 2.4, 1.2], fs=7.2, hl_col=5),
      P(f"Acceptance target (≥ 80% of queries pass): <b>{'met' if min(cold.PASS.mean(), warm.PASS.mean()) >= 0.8 else 'NOT met'}</b>. "
        f"Pass = critic-approved answer that cites the expected doc_id with a verbatim quote and contains the key facts "
        f"(boundary question: refuses and cites nothing). *Over the 7 answerable questions; the boundary question correctly has "
        f"neither. Tests: {summ['tests']}.", small)]

# ---- 3. analysis
s += [P("3. Analysis", h2),
      B(f"<b>Grounding.</b> Every factual answer carries a [doc_id] token and a short verbatim quote; the critic re-checks both "
        f"against the corpus, rejects numbers that do not appear in the cited policy, and rejects the superseded remote_work_v1. "
        f"The out-of-corpus question (CEO salary) is refused without citations."),
      B(f"<b>Tool use and recovery.</b> The bad-id case (E5) calls read_policy('remote_work_v9.9'), receives a soft "
        f"PolicyNotFoundError with suggestions, switches to policy_search and answers from remote_work_v2 in {rec.agent_steps} tool "
        f"steps. The multi-tool case (E6) uses the calculator for 12 × 4 = 48 and then cites the meals cap. All runs stay within "
        f"max_iterations (6) and the graph budget (8 nodes); {retries} critic retr{'y was' if retries == 1 else 'ies were'} needed "
        f"in the cold and warm evaluations."),
      B(f"<b>Memory.</b> Only critic-approved answers with existing doc_ids and verbatim quotes are written "
        f"({int(cold.memory_written.sum())} writes on the cold run; the refusal is not stored). On the warm run "
        f"{int(warm.memory_hit.sum())}/8 questions hit memory and the agent confirms the remembered doc_id with one read_policy "
        f"instead of searching: average tool steps fall by {steps_saved:.0%} and tokens by {tok_saved:.0%}"
        + (f"; latency fell from {lat_c:.2f}s to {lat_w:.2f}s." if lat_w < lat_c else
           f"; latency did not improve here ({lat_c:.2f}s → {lat_w:.2f}s) because the run is dominated by fixed overheads.")),
      B("<b>Orchestration.</b> Explicit nodes and routers make every run traceable (node trace in each log); controlled tests "
        "show the budget stop (max_steps = 3 stops before the critic) and exactly one retry before failed_critic, with no memory "
        "write. Optional SQLite checkpointing persists each node's state."),
      Spacer(1, 2),
      KeepTogether([img("04_cold_vs_warm.png", 15.5), P("Per-question tool steps, latency and tokens: cold vs warm run.", cap)]),
      P("4. Limitations and next steps", h2),
      B("Tiny synthetic corpus (8 policies) and similarity thresholds tuned on it; the deterministic critic checks quotes and numbers "
        "but not paraphrased claims (the optional LLM critic covers this at extra cost)."),
      B("Next: per-node timeouts, a planner/solver/critic multi-agent variant, SQLite-table memory with expiry when a policy version "
        "changes, and a larger regression set of real employee questions."),
      P("Reproduce: put OPENAI_API_KEY in agentic-policy-assistant/.env and run ./run_all.sh (tests → notebook → PNGs → full "
        "report → this summary → zip). POLICY_OFFLINE=1 runs everything without a key.", small)]

SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.3 * cm, bottomMargin=1.3 * cm,
                  title="Policy Assistant Agent - Analysis Summary", author="Arpit Jain").build(s)
pages = int(next(l.split()[-1] for l in subprocess.run(["pdfinfo", str(OUT)], capture_output=True, text=True).stdout.splitlines()
                 if l.startswith("Pages")))
print(f"saved {OUT.relative_to(HERE)} ({pages} pages)")
if pages > 2:
    print("WARNING: summary is longer than 2 pages")
