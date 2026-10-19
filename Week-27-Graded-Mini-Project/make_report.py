"""
Fill the course template with the results and export the PDF.

    python make_report.py              # live results   -> Week 27_Graded Mini Project_Jain.docx / .pdf
    POLICY_OFFLINE=1 python make_report.py   # offline dry run -> agentic-policy-assistant/runs_offline/report_offline.*

The template path defaults to the course folder on Google Drive; override with TEMPLATE_PATH=... .
All numbers are read from the run outputs, so the report always matches the actual run.
"""

import copy
import inspect
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import docx
import pandas as pd
from docx.text.paragraph import Paragraph

from report_helpers import Cell

HERE = Path(__file__).parent
REPO = HERE / "agentic-policy-assistant"
OFFLINE = os.getenv("POLICY_OFFLINE") == "1"
RUN = REPO / ("runs_offline" if OFFLINE else "runs")
RES = REPO / ("results_offline" if OFFLINE else "results")
os.environ["POLICY_AGENT_RUN_DIR"] = str(RUN)
sys.path.insert(0, str(REPO))
import graph, prompts, tools  # noqa: E402

NAME = "Week 27_Graded Mini Project_Jain"
OUT_DOCX = RUN / "report_offline.docx" if OFFLINE else HERE / f"{NAME}.docx"
TEMPLATE = Path(os.getenv("TEMPLATE_PATH", Path.home() / "Library/CloudStorage/GoogleDrive-sachin.arpit@gmail.com/My Drive/"
                "0-Arpit-June-2026-Onwards/IITM Pravartak - Professional Certificate Programme in Generative AI and "
                "Machine Learning/Week 27-Graded Mini Project/Week 27_Graded Mini Project_Template.docx"))

sfx = "_offline" if OFFLINE else ""
v1 = pd.read_csv(RUN / "eval_v1_agent_only.csv")
cold = pd.read_csv(RUN / f"eval_cold{sfx}.csv")
warm = pd.read_csv(RUN / f"eval_warm{sfx}.csv")
cmp = pd.read_csv(RES / "cold_vs_warm.csv", index_col=0)
summary = json.load(open(RES / "summary.json"))
pol = pd.read_json(REPO / "data" / "policies.jsonl", lines=True)
model_name = summary["model"]

d = docx.Document(TEMPLATE)
T = d.tables
assert len(T) == 9, "unexpected template layout"
obj = d.paragraphs[2]
new_p = copy.deepcopy(obj._p); obj._p.addnext(new_p)
p = Paragraph(new_p, obj._parent)
for r in p.runs[1:]:
    r._r.getparent().remove(r._r)
p.runs[0].text = "Submitted by: "
r = p.add_run(f"Arpit Jain   |   Repo: agentic-policy-assistant/ · Notebook: {NAME}.ipynb · Results: results/*.png")
r.font.size = p.runs[0].font.size


def src(fn, n=30):
    s = textwrap.dedent(inspect.getsource(fn)).rstrip().split("\n")
    return "\n".join(s[:n] + (["    ..."] if len(s) > n else []))


pct = lambda s: f"{s.mean():.0%} ({int(s.sum())}/{len(s)})"

# ---------------- 1.1 Data prep
c = Cell(T[0].cell(0, 0))
c.text("data/policies.jsonl holds **8 short policies of a fictional company**, one JSON object per line with doc_id, "
       "text and date. The doc_ids carry a version (leave_v3), and remote_work_v1 is kept on purpose as a "
       "**superseded** version, to test that the agent cites the latest policy.")
c.table([["doc_id", "date", "words", "covers"]] +
        [[r.doc_id, r.date if isinstance(r.date, str) else r.date.strftime("%Y-%m-%d"), len(r.text.split()),
          r.text.split(". ")[0].split("(")[0].strip()] for r in pol.itertuples()], [3.4, 2.4, 1.6, 7.2])
c.code('{"doc_id": "leave_v3", "text": "Leave Policy (v3). Full-time employees receive 24 days of annual leave ...", "date": "2026-01-01"}',
       "One JSONL record")

# ---------------- 1.2 Tools
c = Cell(T[1].cell(0, 0))
c.table([["Tool", "Contract", "Errors"],
         ["policy_search(query, k)", "FAISS (CPU) over 2-sentence windows of each policy, local MiniLM embeddings; "
          "returns JSON hits {doc_id, date, snippet, score}; k is clamped to 1-8", "never raises"],
         ["read_policy(doc_id)", "full text + date of one policy", "read_policy_text raises the typed PolicyNotFoundError "
          "(with close-match suggestions); the tool returns it as a soft 'ERROR ...' string"],
         ["calculator(expression)", "AST-based arithmetic only (+ - * / % **); no names, calls or attributes", "CalculatorError for "
          "division by zero or non-arithmetic input (soft)"]], [3.4, 6.6, 4.6])
c.code(src(tools.read_policy_text) + "\n\n" + src(tools.read_policy.func), "tools.py - typed error, then soft failure in the tool wrapper")
c.code(f'read_policy("remote_work_v9.9") -> {tools.read_policy.invoke({"doc_id": "remote_work_v9.9"})[:150]}...\n'
       f'calculator("12 * 4") -> {tools.calculator.invoke({"expression": "12 * 4"})}     '
       f'calculator("5 / 0") -> {tools.calculator.invoke({"expression": "5 / 0"})}', "Tool outputs")

# ---------------- 1.3 Prompts
c = Cell(T[2].cell(0, 0))
c.text("prompts.py holds the system prompt below, a retry suffix (the critic's issues are fed back to the agent), "
       "the memory-hint template and the optional LLM critic prompt. The acceptance rules map one-to-one to the "
       "checks the critic and eval.py enforce:")
c.code(prompts.SYSTEM_PROMPT.strip())
c.bullets([f"**doc_id tokens:** rule 1. The critic rejects answers without a valid [doc_id]; eval.py records has_docid.",
           "**Quoting:** rule 2. The critic checks that every quote appears verbatim in the cited policy.",
           f"**Step limits:** rule 5, enforced in code too: AgentExecutor max_iterations={prompts.MAX_ITERATIONS} and "
           f"graph max_steps={graph.DEFAULT_MAX_STEPS}."])

# ---------------- 1.4 Agent
c = Cell(T[3].cell(0, 0))
c.bullets([f"**Agent:** create_tool_calling_agent + AgentExecutor (a ReAct loop: the model picks a tool, observes the "
           f"result, decides again). Model: **{model_name}**, temperature=0, seed=42, max_iterations={prompts.MAX_ITERATIONS}, "
           "verbose=True. Every response is cached in SQLite, so reruns are identical.",
           "**Observability:** run_agent returns the answer, every tool call (tool, input, output preview), the step "
           "count, latency and tokens.",
           "**Bad-id recovery:** read_policy('remote_work_v9.9') returns a soft PolicyNotFoundError message with "
           "suggestions. Prompt rule tells the agent not to retry the same id but to policy_search the topic, then "
           "read the latest matching policy. The verbose trace below shows exactly that sequence."])
c.image(RES / "05_error_recovery.png", 14.6, "Error-recovery case (terminal output of app.py with the verbose agent trace)")

# ---------------- 1.5 Evaluation V1
c = Cell(T[4].cell(0, 0))
c.text(f"8 canned questions: the 6 starter queries + a password-length question + an out-of-corpus boundary "
       f"question. Pass = has_docid AND has_quote AND expected doc cited AND key facts present (boundary: refuses and cites "
       f"nothing). **Version 1 = the bare agent (no graph, memory or critic): pass rate {pct(v1.PASS)}.**")
c.table([["id", "kind", "latency (s)", "steps", "has_docid", "has_quote", "cited", "PASS"]] +
        [[r.id, r.kind, f"{r.latency_s:.2f}", r.agent_steps, r.has_docid, r.has_quote, r.cited if isinstance(r.cited, str) else "-",
          "PASS" if r.PASS else "FAIL"] for r in v1.itertuples()], [1.0, 2.0, 1.8, 1.2, 1.8, 1.8, 3.2, 1.6])
c.text("Logs: every question is saved as one JSON line (answer, tool calls, latency, steps) in runs/logs/; the "
       "table is runs/eval_v1_agent_only.csv.", size=8.5)

# ---------------- 2.1 Memory
c = Cell(T[5].cell(0, 0))
c.bullets(["**Write only verified facts:** memory.write stores a fact only if the answer has a [doc_id], every doc_id exists, "
           "and every quote is verbatim in the cited policy. The graph calls it only after the critic approves. "
           "Duplicates are skipped.",
           "**Read cheapest sufficient first:** FAISS similarity over stored questions, threshold 0.80 cosine. Among the "
           "qualifying facts, the one with the fewest doc_ids (cheapest to re-verify) wins. It is injected into the question "
           "as a 'VERIFIED MEMORY' hint, so the agent can confirm with one read_policy instead of searching."])
c.table([["metric"] + list(cmp.columns)] + [[i] + [f"{v:g}" for v in row] for i, row in cmp.iterrows()], [5, 4.8, 4.8])
c.image(RES / "04_cold_vs_warm.png", 14.6, "Cold run (memory cleared) vs warm run (memory kept), per question")
saved = 1 - cmp.loc["avg tool steps", "warm"] / cmp.loc["avg tool steps", "cold"]
c.text(f"On the warm run, memory hit {int(cmp.loc['memory hits', 'warm'])} of 8 questions and cut tool steps by {saved:.0%}. "
       f"No new facts were written, because they were already stored. The pass rate stayed at {cmp.loc['pass rate', 'warm']:.0%}, so the "
       "hints made the agent cheaper without reducing grounding.")

# ---------------- 2.2 Graph
c = Cell(T[6].cell(0, 0))
c.image(RES / "01_graph.png", 14.6, "graph.py - LangGraph StateGraph")
c.bullets(["**Nodes:** plan (rule-based tool plan) → memory_read → agent_answer → critic → memory_write → stop. "
           "Every node appends a line to state.trace and increments graph_steps.",
           f"**Budget:** a router after every node sends the run to stop with status budget_exhausted once graph_steps reaches "
           f"max_steps ({graph.DEFAULT_MAX_STEPS}). LangGraph's recursion_limit is a hard safety net on top.",
           "**Single retry:** if the critic rejects the answer, a retry node re-runs agent_answer once with the critic's "
           "issues, but only if the budget still fits. A second rejection gives failed_critic, with no memory write."])
c.image(RES / "06_budget_and_retry.png", 14.6, "Budget stop (max_steps=3) and the critic → retry → stop path (scripted bad model)")

# ---------------- 2.3 Critic
c = Cell(T[7].cell(0, 0))
c.text("graph.critic_check is deterministic and returns the minimal JSON {\"valid\": bool, \"issues\": [...]}. It checks:")
c.bullets(["a [doc_id] is present and the doc_id exists; at least one quote; each quote is verbatim in the cited policy",
           "simple claims: every number in the answer must appear in the cited policies (or come from the question / calculator)",
           "no superseded policy is cited; a 'not found' answer cites nothing",
           "optional: an LLM critic (prompts.CRITIC_PROMPT, same JSON) to check claim semantics"])
ex = ["Remote work is allowed 3 days a week.",
      '"Employees may work remotely every day" [remote_work_v2]',
      'You get 30 days. "Full-time employees receive 24 days of annual leave per calendar year" [leave_v3]',
      '"Employees could work remotely for a maximum of 2 days per week" [remote_work_v1]',
      '"Employees may work remotely for a maximum of 3 days per week" [remote_work_v2]']
c.code("\n".join(f"{json.dumps(graph.critic_check('q', a).model_dump())}\n    <- {a}" for a in ex), "Critic verdicts")
c.text(f"In the evaluation, the critic approved {int(cold.critic_valid.sum())}/8 answers on the cold run, with "
       f"{int(cold.retries.sum())} retries needed.")

# ---------------- 2.4 Tests + docs
c = Cell(T[8].cell(0, 0))
c.text(f"**PyTest: {summary['tests']}** (offline: a deterministic stand-in model emits real tool calls, so no key is needed):")
c.bullets(["tests/test_tools.py: corpus fields, typed search hits, k clamp, full text, typed PolicyNotFoundError, soft "
           "tool error, calculator safe division and code rejection",
           "tests/test_agent_acceptance.py: doc_id + verbatim quote, bad-id recovery, ≥80% acceptance on the canned "
           "set, critic rejections, memory stores only verified facts, warm run uses fewer steps, no write after a failed critic",
           "tests/test_graph_budget.py: node order, budget stop, exactly one retry, retry skipped when the budget is too "
           "small, agent max_iterations enforced"])
c.image(RES / "08_eval_table.png", 14.6, f"Final evaluation table, printed by eval.py ({model_name})")
c.bullets([f"**Acceptance:** cold {pct(cold.PASS)}, warm {pct(warm.PASS)}. Target ≥ 80%: "
           f"{'met' if min(cold.PASS.mean(), warm.PASS.mean()) >= 0.8 else 'NOT met'}.",
           "**Docs and run scripts:** README.md (setup, run, design notes, limitations); app.py CLI (ask, eval --cold, "
           "eval, reset-memory, --report for a JSON run report, --sqlite); run_all.sh builds everything.",
           "**Screenshots in results/:** 02 cold run, 03 warm run (memory hits), 05 error recovery, 06 budget/retry, "
           "07 pytest, 08 eval table.",
           "**Optional extensions done:** SQLite checkpointing (graph.sqlite_checkpointer, every node state persisted), a "
           "JSON run report via the CLI flag, and an optional LLM critic.",
           "**Limitations:** a tiny synthetic corpus; the deterministic critic checks numbers and quotes but not paraphrased "
           "claims; similarity thresholds were tuned on this corpus."])

d.save(OUT_DOCX)
print("saved", OUT_DOCX)
soffice = shutil.which("soffice") or "/Applications/LibreOffice.app/Contents/MacOS/soffice"
if Path(soffice).exists():
    subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(OUT_DOCX.parent), str(OUT_DOCX)],
                   check=True, capture_output=True)
    print("saved", OUT_DOCX.with_suffix(".pdf"))
