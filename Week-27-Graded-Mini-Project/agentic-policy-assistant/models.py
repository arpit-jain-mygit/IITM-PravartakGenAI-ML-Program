"""
Pydantic schemas, typed errors and the LangGraph state shared by every module.

Keeping all data shapes in one file means each module (tools, memory, graph, eval) agrees on
exactly what a policy, a search hit, an answer, a critic verdict and a memory fact look like.
"""

from __future__ import annotations

import re
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

# A doc_id token in an answer looks like [leave_v3] - lowercase words, underscores, a version suffix
DOC_ID_RE = re.compile(r"\[([a-z][a-z0-9_]*_v\d+(?:\.\d+)?)\]")
# A quote is any text in straight or curly double quotes (at least 3 characters)
QUOTE_RE = re.compile(r"[\"“]([^\"”]{3,})[\"”]")


# ---------------------------------------------------------------------------
# Corpus and tool outputs
# ---------------------------------------------------------------------------
class PolicyDoc(BaseModel):
    """One line of data/policies.jsonl."""
    doc_id: str
    text: str
    date: str                                   # ISO date the policy took effect


class SearchHit(BaseModel):
    """One result returned by policy_search."""
    doc_id: str
    date: str
    snippet: str
    score: float = Field(description="FAISS L2 distance; lower is more similar")


# ---------------------------------------------------------------------------
# Typed errors: tools raise these, the agent receives a short, readable message instead
# ---------------------------------------------------------------------------
class PolicyToolError(Exception):
    """Base class for all tool errors (lets callers catch every tool failure in one place)."""


class PolicyNotFoundError(PolicyToolError):
    def __init__(self, doc_id: str, suggestions: list[str]):
        self.doc_id, self.suggestions = doc_id, suggestions
        super().__init__(f"PolicyNotFoundError: no policy with doc_id '{doc_id}'. "
                         f"Closest known ids: {', '.join(suggestions)}. Use policy_search to find the right document.")


class CalculatorError(PolicyToolError):
    """Raised for division by zero or any expression that is not plain arithmetic."""


# ---------------------------------------------------------------------------
# Agent, critic and memory results
# ---------------------------------------------------------------------------
class AgentResult(BaseModel):
    answer: str
    tool_calls: list[dict[str, Any]] = []       # [{"tool": ..., "input": ..., "output_preview": ...}]
    steps: int = 0                              # number of tool calls the agent made
    hit_iteration_limit: bool = False
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class CriticVerdict(BaseModel):
    """The critic's minimal JSON contract: {"valid": bool, "issues": [...]}."""
    valid: bool
    issues: list[str] = []


class MemoryFact(BaseModel):
    """A verified fact stored in semantic memory (only written after the critic passes)."""
    question: str
    answer: str
    doc_ids: list[str]
    quotes: list[str]
    created_at: str


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------
class GraphState(TypedDict, total=False):
    question: str                    # the user's question, unchanged
    plan: list[str]                  # short plan from the plan node
    memory_hint: str                 # text added to the question when memory has a relevant verified fact
    memory_hit: dict | None          # the memory entry used (question, doc_ids, similarity)
    answer: str
    agent_steps: int                 # tool calls made by the agent (all attempts)
    tool_calls: list[dict]
    critic: dict                     # last CriticVerdict as a dict
    critic_feedback: str             # issues passed back to the agent on the retry
    retries: int                     # how many times agent_answer was re-run after a critic failure
    graph_steps: int                 # nodes executed so far (the graph budget counter)
    max_steps: int                   # graph budget
    memory_written: bool
    status: Literal["running", "ok", "failed_critic", "budget_exhausted"]
    trace: list[str]                 # one line per node, for observability
    latency_s: float
    tokens: int
