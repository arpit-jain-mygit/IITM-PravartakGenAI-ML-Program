"""Graph orchestration: node order, budget stop, single retry, recursion safety."""
from langchain_core.messages import AIMessage

from agent import build_agent
from fake_llm import OfflinePolicyLLM
from graph import build_graph, run_graph


def test_happy_path_visits_nodes_in_order(offline_app):
    out = run_graph("Is MFA required for VPN? Cite and quote.", app=offline_app)
    nodes = [t.split(". ", 1)[1].split(":")[0] for t in out["trace"]]
    assert nodes == ["plan", "memory_read", "agent_answer", "critic", "memory_write", "stop"]
    assert out["status"] == "ok" and out["graph_steps"] <= 8


def test_budget_stop_before_agent_finishes(offline_app):
    out = run_graph("Is MFA required for VPN? Cite and quote.", app=offline_app, max_steps=3)
    nodes = [t.split(". ", 1)[1].split(":")[0] for t in out["trace"]]
    assert out["status"] == "budget_exhausted"
    assert "critic" not in nodes and "memory_write" not in nodes and nodes[-1] == "stop"


def test_single_retry_then_stop(memory):
    llm = OfflinePolicyLLM(script=[AIMessage(content="No citation here."), AIMessage(content="Still none.")])
    app = build_graph(executor=build_agent(llm, verbose=False), memory=memory)
    out = run_graph("What is our remote work limit?", app=app)
    nodes = [t.split(". ", 1)[1].split(":")[0] for t in out["trace"]]
    assert nodes.count("agent_answer") == 2 and nodes.count("critic") == 2      # exactly one retry
    assert out["status"] == "failed_critic" and nodes[-1] == "stop"


def test_retry_skipped_when_budget_too_small(memory):
    llm = OfflinePolicyLLM(script=[AIMessage(content="No citation here.")])
    app = build_graph(executor=build_agent(llm, verbose=False), memory=memory)
    out = run_graph("What is our remote work limit?", app=app, max_steps=5)
    nodes = [t.split(". ", 1)[1].split(":")[0] for t in out["trace"]]
    assert nodes.count("agent_answer") == 1 and out["status"] in {"failed_critic", "budget_exhausted"}


def test_agent_iteration_limit_is_enforced(memory):
    # a model that only ever searches: the executor must stop at max_iterations
    loop = [AIMessage(content="", tool_calls=[{"name": "policy_search", "args": {"query": "x"}, "id": f"c{i}"}])
            for i in range(20)]
    ex = build_agent(OfflinePolicyLLM(script=loop), verbose=False, max_iterations=3)
    app = build_graph(executor=ex, memory=memory)
    out = run_graph("anything", app=app)
    assert out["agent_steps"] <= 3 * 2                      # 3 per attempt, at most one retry
    assert out["status"] in {"failed_critic", "budget_exhausted"}
