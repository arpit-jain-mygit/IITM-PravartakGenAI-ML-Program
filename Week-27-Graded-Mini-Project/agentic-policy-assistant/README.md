# Policy Assistant Agent

This is a single agent with memory and graph orchestration. It answers employee questions **only** from a small local policy corpus. Every answer cites `[doc_id]` tokens with short verbatim quotes, recovers from bad tool calls, stays within explicit step budgets, and remembers only facts a critic has verified.

**Stack**
- Python 3.12, LangChain 1.x (`langchain-classic` AgentExecutor) and LangGraph
- FAISS on CPU, with local MiniLM embeddings
- OpenAI `gpt-4o-mini` at temperature 0 (or a Vocareum-compatible endpoint)
- SQLite for the LLM cache and the optional graph checkpoints

## Layout

```
agentic-policy-assistant/
├─ data/policies.jsonl        8 policies of a fictional company: doc_id, text, date (remote_work_v1 is superseded)
├─ models.py                  pydantic schemas, typed errors, GraphState
├─ tools.py                   policy_search (FAISS), read_policy (typed error -> soft failure), calculator (safe AST)
├─ prompts.py                 system prompt with acceptance rules, retry suffix, memory hint, critic prompt
├─ agent.py                   ReAct tool-calling agent/executor factory (max_iterations=6, verbose tracing)
├─ memory.py                  FAISS semantic memory of verified facts (write rules, cheapest-sufficient read)
├─ graph.py                   LangGraph: plan -> memory_read -> agent_answer -> critic -> memory_write -> stop
├─ eval.py                    8 canned questions, acceptance checks, metrics, logs, JSON run report
├─ app.py                     CLI: ask / eval / reset-memory
├─ fake_llm.py                offline stand-in model (tool-calling) for tests and no-key demos
├─ tests/                     test_tools.py, test_agent_acceptance.py, test_graph_budget.py (22 tests)
├─ Week 27_Graded Mini Project_Jain.ipynb   notebook that runs everything and saves results/*.png
├─ README.md, requirements.txt, .env.example
```

## Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # then put your OPENAI_API_KEY in .env
```

For Vocareum, also set `OPENAI_BASE_URL` in `.env`. The model can be changed with `OPENAI_MODEL`.

## Run

```bash
python app.py ask "Is MFA required for VPN? Cite and quote."     # one question with the verbose agent trace + node trace
python app.py eval --cold                                        # clear memory, run the 8 canned questions
python app.py eval                                               # warm run: memory hits
python app.py ask "Read policy remote_work_v9.9 and answer the limit."   # error-recovery case
python app.py eval --report run_report.json                      # export the JSON run report
python app.py ask "..." --sqlite                                 # persist graph checkpoints to runs/checkpoints.sqlite
python -m pytest -q tests                                        # 22 tests, offline, no key needed
```

Add `--offline` to any command to use the stand-in model, with no key and no cost. Outputs then go to `runs_offline/`.

**Outputs**
- `runs/logs/*.jsonl`: one record per question, with the answer, tool calls, node trace, critic verdict and memory hit
- `runs/eval_*.csv`: the evaluation tables
- `runs/report_*.json`: the JSON run reports
- `results/*.png`: graph diagram, cold/warm/recovery screenshots, budget and retry, pytest, evaluation table

## Design notes

**Acceptance rules** (`prompts.SYSTEM_PROMPT`). These are enforced again in code by the critic:
- `[doc_id]` tokens for every policy used
- a verbatim quote of 25 words or fewer for each exact rule
- no external facts
- the latest version of a policy wins
- at most 6 tool calls

**Agent**
- `create_tool_calling_agent` + `AgentExecutor`, with temperature 0, seed 42 and `max_iterations=6`.
- Tools never crash the loop. `read_policy_text` raises `PolicyNotFoundError` (a typed error with suggestions), and the tool returns it as `ERROR ...`. The prompt then tells the agent to `policy_search` instead of retrying the bad id.

**Graph**
- Every node increments `graph_steps`. A router after each node stops the run when `graph_steps >= max_steps` (default 8).
- After the critic, the router sends a valid answer to `memory_write`. An invalid answer gets **one** retry, with the critic's issues fed back. If that also fails, the run stops with status `failed_critic`.
- `recursion_limit` is a second safety net.

**Critic** (`graph.critic_check`). It returns `{"valid": bool, "issues": [...]}` and checks:
- a `[doc_id]` is present and exists
- every quote appears verbatim in the cited policy
- every number in the answer appears in the cited policies (or the question, or a calculator result)
- no superseded policy is cited
- "not found" answers cite nothing

An LLM critic (`prompts.CRITIC_PROMPT`) can be added for semantic checks.

**Memory**
- **Write** only after the critic passes, and only if the doc_ids exist and the quotes are verified.
- **Read:** the stored questions with cosine similarity ≥ 0.80 qualify, and the cheapest one wins (fewest doc_ids to re-verify). It is passed into the question as a `VERIFIED MEMORY` hint, so the agent confirms it with one `read_policy` call instead of searching.

**Observability:** each run records latency, agent steps, graph steps, tokens, the tool calls with output previews, and the node trace.

## Limitations

- The corpus is tiny and synthetic (8 policies). Similarity thresholds were tuned on it.
- The deterministic critic checks numbers and quotes, but not paraphrased claims. The optional LLM critic covers this, at extra cost.
- The offline stand-in model is rule-based. It proves the plumbing, but it is not a measure of answer quality. The graded numbers come from the OpenAI run.
