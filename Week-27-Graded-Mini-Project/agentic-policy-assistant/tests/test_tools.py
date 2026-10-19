"""Tool contracts: search returns typed hits, read_policy raises a typed error, calculator is safe."""
import json

import pytest

from models import CalculatorError, PolicyNotFoundError, SearchHit
from tools import calculator, load_policies, policy_search, read_policy, read_policy_text, safe_eval, search_policies


def test_corpus_has_required_fields():
    docs = load_policies()
    assert len(docs) >= 6
    for d in docs.values():
        assert d.doc_id and d.text and d.date


def test_policy_search_returns_relevant_typed_hits():
    hits = search_policies("Is MFA required for the VPN?", k=3)
    assert 1 <= len(hits) <= 3 and all(isinstance(h, SearchHit) for h in hits)
    assert hits[0].doc_id == "security_v1"


def test_policy_search_tool_returns_json_with_doc_ids():
    hits = json.loads(policy_search.invoke({"query": "annual leave carry over", "k": 2}))
    assert hits and {"doc_id", "date", "snippet", "score"} <= set(hits[0])
    assert "leave_v3" in [h["doc_id"] for h in hits]


def test_search_k_is_clamped():
    assert len(search_policies("policy", k=100)) <= 8


def test_read_policy_returns_full_text():
    doc = json.loads(read_policy.invoke({"doc_id": "leave_v3"}))
    assert doc["doc_id"] == "leave_v3" and "24 days of annual leave" in doc["text"]


def test_read_policy_raises_typed_error_for_missing_id():
    with pytest.raises(PolicyNotFoundError) as e:
        read_policy_text("remote_work_v9.9")
    assert "remote_work_v2" in e.value.suggestions


def test_read_policy_tool_fails_softly():
    out = read_policy.invoke({"doc_id": "remote_work_v9.9"})
    assert out.startswith("ERROR PolicyNotFoundError") and "policy_search" in out


def test_calculator_basic_and_safe_division():
    assert safe_eval("12 * 4") == 48
    assert safe_eval("2,500 / 2") == 1250
    with pytest.raises(CalculatorError):
        safe_eval("5 / 0")
    assert calculator.invoke({"expression": "5/0"}).startswith("ERROR CalculatorError")


def test_calculator_rejects_code():
    with pytest.raises(CalculatorError):
        safe_eval("__import__('os').system('echo hi')")
