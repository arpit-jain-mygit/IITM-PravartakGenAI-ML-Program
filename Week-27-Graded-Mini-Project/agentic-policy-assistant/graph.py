"""
LangGraph orchestration:

    plan -> memory_read -> agent_answer -> critic --valid--> memory_write -> stop
                                 ^             |
                                 +--invalid, retries < 1, budget left
                                               |
                                               +--otherwise--> stop

- Budget: every node increments graph_steps. Once graph_steps reaches max_steps, the router sends the run
  to stop with status "budget_exhausted". LangGraph's recursion_limit is a second, hard safety net.
- Single retry: if the critic rejects the answer, agent_answer runs once more with the critic's issues
  appended. A second rejection ends the run with status "failed_critic". Nothing is written to memory then.
- The critic is deterministic (critic_check) and returns the minimal JSON {"valid": bool, "issues": [...]}.
  An optional LLM critic (prompts.CRITIC_PROMPT) can be switched on to check the semantics of claims.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from langgraph.graph import END, StateGraph

from agent import RUN_DIR, build_agent, run_agent
from memory import FactMemory, verify_fact
from models import DOC_ID_RE, CriticVerdict, GraphState
from prompts import CRITIC_PROMPT, MEMORY_HINT, RETRY_SUFFIX
from tools import load_policies

NOT_FOUND = "could not find this in the policy corpus"
DEFAULT_MAX_STEPS = 8          # 6 nodes on the happy path + 2 for one retry (agent_answer + critic)


def critic_check(question: str, answer: str, llm=None) -> CriticVerdict:
    """Deterministic checks first; optionally an LLM checks the claims against the cited texts."""
    issues = []
    if NOT_FOUND in answer.lower():
        if DOC_ID_RE.search(answer):
            issues.append("a 'not found' answer must not cite documents")
        return CriticVerdict(valid=not issues, issues=issues)
    if "Agent stopped due to max iterations" in answer:
        return CriticVerdict(valid=False, issues=["agent hit its iteration limit without a final answer"])

    ok, doc_ids, quotes, reason = verify_fact(answer)
    if not doc_ids:
        issues.append("no [doc_id] token in the answer")
    elif not ok:
        issues.append(reason)
    if doc_ids and not quotes:
        issues.append("no verbatim quote for the rule")
    pol = load_policies()
    superseded = [d for d in doc_ids if d in pol and "SUPERSEDED" in pol[d].text]
    if superseded:
        issues.append(f"cites superseded policy {superseded}")

    # simple claim check: every number stated in the answer must appear in a cited policy
    # (numbers that the user supplied or that came from the calculator are allowed)
    if doc_ids and all(d in pol for d in doc_ids):
        num = lambda t: {n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", t)}
        allowed = num(" ".join(pol[d].text for d in doc_ids) + " " + question)
        allowed |= {str(eval_simple(question))}                 # result of a requested calculation
        unsupported = sorted(num(DOC_ID_RE.sub("", answer)) - allowed)
        if unsupported:
            issues.append(f"numbers not found in cited policies: {unsupported}")

    if llm is not None and not issues:
        sources = "\n".join(f"[{d}] {load_policies()[d].text}" for d in doc_ids)
        raw = llm.invoke(CRITIC_PROMPT.format(question=question, answer=answer, sources=sources)).text
        try:
            verdict = CriticVerdict(**json.loads(raw[raw.find("{"): raw.rfind("}") + 1]))
            issues += verdict.issues if not verdict.valid else []
        except Exception:
            issues.append("LLM critic returned invalid JSON")
    return CriticVerdict(valid=not issues, issues=issues)


def eval_simple(question: str):
    """Result of a 'compute A op B' request in the question (used to allow calculator numbers)."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*([*x/+-])\s*(\d+(?:\.\d+)?)", question)
    if not m:
        return None
    a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
    r = {"*": a * b, "x": a * b, "+": a + b, "-": a - b, "/": a / b if b else None}[op]
    return int(r) if r is not None and float(r).is_integer() else r


def build_graph(executor=None, memory: FactMemory | None = None, llm_critic=None, checkpointer=None):
    """Compile the graph. executor/memory are injectable so tests can use a fake LLM and a temp memory."""
    executor = executor or build_agent(verbose=True)
    memory = memory or FactMemory(RUN_DIR / "memory")

    def tick(state: GraphState, node: str, note: str = "") -> dict:
        steps = state.get("graph_steps", 0) + 1
        return {"graph_steps": steps, "trace": state.get("trace", []) + [f"{steps}. {node}{': ' + note if note else ''}"]}

    # ---------------- nodes ----------------
    def plan(state: GraphState):
        q = state["question"].lower()
        p = ["check memory for a verified fact"]
        if re.search(r"\b[a-z][a-z_]*_v\d+(\.\d+)?\b", q):
            p.append("read the named policy; if the id is unknown, fall back to policy_search")
        else:
            p.append("policy_search for the topic, then read_policy if a snippet is not enough")
        if re.search(r"\d+\s*[*x/+-]\s*\d+", q) or "compute" in q:
            p.append("use calculator for the arithmetic")
        p += ["answer with [doc_id] tokens and short quotes", "critic review", "write memory if verified"]
        return {"plan": p, "status": "running", "retries": 0, "agent_steps": 0, "tool_calls": [],
                "memory_written": False, "max_steps": state.get("max_steps", DEFAULT_MAX_STEPS),
                **tick(state, "plan", f"{len(p)} steps")}

    def memory_read(state: GraphState):
        hit = memory.read(state["question"])
        if hit:
            hint = MEMORY_HINT.format(answer=hit["answer"], doc_ids=", ".join(hit["doc_ids"]))
            return {"memory_hit": {k: hit[k] for k in ["question", "doc_ids", "similarity"]}, "memory_hint": hint,
                    **tick(state, "memory_read", f"HIT sim={hit['similarity']} -> {hit['doc_ids']}")}
        return {"memory_hit": None, "memory_hint": "", **tick(state, "memory_read", "miss")}

    def agent_answer(state: GraphState):
        prompt = state["question"]
        if state.get("memory_hint"):
            prompt = f"{prompt}\n\n{state['memory_hint']}"
        if state.get("critic_feedback"):
            prompt += RETRY_SUFFIX.format(issues=state["critic_feedback"])
        res = run_agent(prompt, executor=executor)
        return {"answer": res.answer, "agent_steps": state.get("agent_steps", 0) + res.steps,
                "tool_calls": state.get("tool_calls", []) + res.tool_calls,
                "tokens": state.get("tokens", 0) + res.input_tokens + res.output_tokens,
                **tick(state, "agent_answer", f"{res.steps} tool calls " + str([c['tool'] for c in res.tool_calls]))}

    def critic(state: GraphState):
        v = critic_check(state["question"], state["answer"], llm=llm_critic)
        upd = {"critic": v.model_dump(), **tick(state, "critic", json.dumps(v.model_dump()))}
        if not v.valid:
            upd["critic_feedback"] = "\n".join(f"- {i}" for i in v.issues)
        return upd

    def retry_prep(state: GraphState):
        return {"retries": state.get("retries", 0) + 1}

    def memory_write(state: GraphState):
        if NOT_FOUND in state["answer"].lower():
            return {"memory_written": False, **tick(state, "memory_write", "skipped (no fact to store)")}
        ok, msg = memory.write(state["question"], state["answer"])
        return {"memory_written": ok, **tick(state, "memory_write", msg)}

    def stop(state: GraphState):
        status = state.get("status", "running")
        if status == "running":
            status = "ok" if state.get("critic", {}).get("valid") else (
                "budget_exhausted" if state.get("graph_steps", 0) >= state.get("max_steps", DEFAULT_MAX_STEPS)
                else "failed_critic")
        return {"status": status, **tick(state, "stop", status)}

    # ---------------- routers ----------------
    def over_budget(state: GraphState) -> bool:
        return state.get("graph_steps", 0) >= state.get("max_steps", DEFAULT_MAX_STEPS)

    def after_node(nxt):
        def route(state: GraphState):
            return "stop" if over_budget(state) else nxt
        return route

    def after_critic(state: GraphState):
        if state["critic"]["valid"]:
            return "memory_write" if not over_budget(state) else "stop"
        if state.get("retries", 0) < 1 and state.get("graph_steps", 0) + 2 <= state.get("max_steps", DEFAULT_MAX_STEPS):
            return "retry"
        return "stop"

    g = StateGraph(GraphState)
    for name, fn in [("plan", plan), ("memory_read", memory_read), ("agent_answer", agent_answer),
                     ("critic", critic), ("retry", retry_prep), ("memory_write", memory_write), ("stop", stop)]:
        g.add_node(name, fn)
    g.set_entry_point("plan")
    g.add_conditional_edges("plan", after_node("memory_read"), {"memory_read": "memory_read", "stop": "stop"})
    g.add_conditional_edges("memory_read", after_node("agent_answer"), {"agent_answer": "agent_answer", "stop": "stop"})
    g.add_conditional_edges("agent_answer", after_node("critic"), {"critic": "critic", "stop": "stop"})
    g.add_conditional_edges("critic", after_critic, {"memory_write": "memory_write", "retry": "retry", "stop": "stop"})
    g.add_edge("retry", "agent_answer")
    g.add_edge("memory_write", "stop")
    g.add_edge("stop", END)
    return g.compile(checkpointer=checkpointer)


def run_graph(question: str, app=None, max_steps: int = DEFAULT_MAX_STEPS, thread_id: str | None = None) -> dict:
    app = app or build_graph()
    t0 = time.perf_counter()
    cfg = {"recursion_limit": max_steps + 4}
    if thread_id:
        cfg["configurable"] = {"thread_id": thread_id}
    out = app.invoke({"question": question, "max_steps": max_steps}, config=cfg)
    out["latency_s"] = round(time.perf_counter() - t0, 3)
    return out


def sqlite_checkpointer(path: Path = RUN_DIR / "checkpoints.sqlite"):
    """Optional: persist every graph step to SQLite (resumable, inspectable runs)."""
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))
