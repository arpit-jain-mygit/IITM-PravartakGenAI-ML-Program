"""Acceptance: grounded answers with doc_id + quotes, bad-id recovery, critic, memory write rules."""
from langchain_core.messages import AIMessage

from agent import build_agent, run_agent
from eval import CANNED, score
from fake_llm import OfflinePolicyLLM
from graph import critic_check, run_graph
from memory import verify_fact


def test_agent_answer_has_docid_and_verbatim_quote():
    res = run_agent("Is MFA required for VPN? Cite and quote.", build_agent(OfflinePolicyLLM(), verbose=False))
    ok, doc_ids, quotes, _ = verify_fact(res.answer)
    assert ok and doc_ids == ["security_v1"] and quotes
    assert 1 <= res.steps <= 6


def test_bad_id_recovery_fails_softly_then_searches():
    res = run_agent("Read policy remote_work_v9.9 and answer the limit.", build_agent(OfflinePolicyLLM(), verbose=False))
    tools = [c["tool"] for c in res.tool_calls]
    assert tools[0] == "read_policy" and res.tool_calls[0]["output_preview"].startswith("ERROR")
    assert "policy_search" in tools[1:]
    assert "[remote_work_v2]" in res.answer


def test_canned_set_meets_80_percent_acceptance(offline_app):
    results = [score(c, run_graph(c["q"], app=offline_app)) for c in CANNED]
    assert sum(r["PASS"] for r in results) / len(results) >= 0.8


def test_critic_rejects_missing_docid_and_fake_quote():
    v = critic_check("q", "Remote work is allowed 3 days a week.")
    assert not v.valid and any("doc_id" in i for i in v.issues)
    v = critic_check("q", '"Employees may work remotely every day" [remote_work_v2]')
    assert not v.valid and any("verbatim" in i for i in v.issues)
    v = critic_check("q", '"Employees may work remotely for a maximum of 3 days per week" [remote_work_v2]')
    assert v.valid and v.model_dump() == {"valid": True, "issues": []}


def test_critic_flags_unsupported_numbers():
    v = critic_check("q", 'You get 30 days. "Full-time employees receive 24 days of annual leave per calendar year" [leave_v3]')
    assert not v.valid and any("30" in i for i in v.issues)


def test_memory_only_stores_verified_facts(memory):
    assert memory.write("q1", "Remote work is 3 days.")[0] is False                        # no doc_id
    assert memory.write("q2", '"made up rule" [remote_work_v2]')[0] is False               # quote not verbatim
    assert memory.write("q3", '"Employees may work remotely for a maximum of 3 days per week" [remote_work_v2]')[0]
    assert memory.size() == 1


def test_warm_run_hits_memory_and_uses_fewer_steps(offline_app):
    q = "How many annual leave days and can they be carried over? Cite and quote."
    cold, warm = run_graph(q, app=offline_app), run_graph(q, app=offline_app)
    assert cold["memory_written"] and not cold.get("memory_hit")
    assert warm["memory_hit"] and warm["agent_steps"] < cold["agent_steps"]


def test_no_memory_write_when_critic_fails(memory):
    from graph import build_graph
    bad = AIMessage(content="Remote work is unlimited.")                      # no doc_id -> critic fails twice
    llm = OfflinePolicyLLM(script=[bad, AIMessage(content="Still unlimited.")])
    app = build_graph(executor=build_agent(llm, verbose=False), memory=memory)
    out = run_graph("What is our remote work limit?", app=app)
    assert out["status"] == "failed_critic" and out["retries"] == 1
    assert memory.size() == 0
