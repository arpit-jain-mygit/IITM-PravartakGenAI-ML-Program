"""Shared fixtures: every test runs offline in a temporary run directory (no API key, no shared memory)."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["POLICY_AGENT_RUN_DIR"] = tempfile.mkdtemp(prefix="policy_agent_tests_")

import pytest  # noqa: E402


@pytest.fixture
def memory(tmp_path):
    from memory import FactMemory
    return FactMemory(tmp_path / "memory")


@pytest.fixture
def offline_app(memory):
    from agent import build_agent
    from fake_llm import OfflinePolicyLLM
    from graph import build_graph
    return build_graph(executor=build_agent(OfflinePolicyLLM(), verbose=False), memory=memory)
