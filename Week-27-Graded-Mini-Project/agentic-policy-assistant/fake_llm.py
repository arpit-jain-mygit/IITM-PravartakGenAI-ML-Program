"""
Offline stand-in for the OpenAI model, used by the pytest suite and by `app.py --offline`.

It is a deterministic, rule-based "ReAct policy" that emits REAL tool calls through LangChain's
tool-calling interface, so the AgentExecutor, tools, critic, memory and graph are all exercised exactly
as in a live run, just without network access or cost. It is NOT used for the graded live results.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

STOP = {"what", "is", "our", "the", "and", "any", "can", "they", "be", "how", "many", "for", "cite", "quote",
        "tell", "me", "with", "citation", "then", "read", "policy", "answer", "a", "an", "of", "to", "it", "do",
        "i", "my", "are", "rule", "compute"}


RELEVANCE_CUTOFF = 1.2      # FAISS L2 distance; above this the best hit is treated as "not in the corpus"


def _words(t: str) -> set[str]:
    """Content words with a crude plural stem ('passwords' -> 'password')."""
    return {w[:-1] if w.endswith("s") and len(w) > 4 else w
            for w in re.findall(r"[a-z]+", t.lower()) if w not in STOP and len(w) > 2}


class OfflinePolicyLLM(BaseChatModel):
    """Deterministic tool-calling model. `script` can force exact responses (used by tests)."""
    script: list[AIMessage] | None = None
    cache: bool = False                  # never cache scripted/offline responses
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "offline-policy-llm"

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls += 1
        msg = self.script.pop(0) if self.script else self._decide(messages)
        msg.usage_metadata = {"input_tokens": sum(len(str(m.content)) for m in messages) // 4,
                              "output_tokens": max(1, len(str(msg.content)) // 4), "total_tokens": 0}
        return ChatResult(generations=[ChatGeneration(message=msg)])

    # ---------------------------------------------------------------- policy
    def _decide(self, messages) -> AIMessage:
        human = next(m for m in messages if m.type == "human").content
        question = human.split("\n\n")[0]
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        made = [tc for m in messages if isinstance(m, AIMessage) for tc in (m.tool_calls or [])]
        n = len(made)

        def call(name, args):
            return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{n}"}])

        last = tool_msgs[-1].content if tool_msgs else ""
        done = {tc["name"] for tc in made}
        calc = re.search(r"(\d+(?:\.\d+)?\s*[*x/+-]\s*\d+(?:\.\d+)?)", question)
        named = re.search(r"\b([a-z][a-z_]*_v\d+(?:\.\d+)?)\b", question)
        hint = re.search(r"Sources: ([a-z0-9_]+)", human)

        if not made:
            if calc:
                return call("calculator", {"expression": calc.group(1).replace("x", "*")})
            if named:
                return call("read_policy", {"doc_id": named.group(1)})
            if hint:
                return call("read_policy", {"doc_id": hint.group(1)})
            return call("policy_search", {"query": question, "k": 3})
        if last.startswith("ERROR PolicyNotFound"):                     # bad id -> fail softly, then search
            # search for the topic words inside the bad id ("remote_work_v9.9" -> "remote work") + the question
            topic = re.sub(r"\b([a-z][a-z_]*)_v\d+(?:\.\d+)?\b", lambda m: m.group(1).replace("_", " "), question)
            return call("policy_search", {"query": topic, "k": 3})
        if "policy_search" not in done and "read_policy" not in done:
            return call("policy_search", {"query": re.sub(r".*?(and then|then)", "", question), "k": 3})
        if last.startswith("["):                                          # search results -> read best doc
            hits = [h for h in json.loads(last) if h["score"] < RELEVANCE_CUTOFF]
            if not hits:
                return AIMessage(content="I could not find this in the policy corpus.")
            # rule 4 of the system prompt: within a policy family (remote_work_v1, _v2...) use the latest date
            family = hits[0]["doc_id"].rsplit("_v", 1)[0]
            same = [h for h in json.loads(last) if h["doc_id"].rsplit("_v", 1)[0] == family]
            return call("read_policy", {"doc_id": max(same, key=lambda h: h["date"])["doc_id"]})
        if last.startswith("{"):                                          # full policy text -> final answer
            doc = json.loads(last)
            return AIMessage(content=self._answer(question, doc, tool_msgs))
        return AIMessage(content="I could not find this in the policy corpus.")

    def _answer(self, question: str, doc: dict, tool_msgs) -> str:
        sents = [s.strip().rstrip(".") for s in re.split(r"(?<=\.)\s+", doc["text"]) if s.strip()]
        q = _words(question)
        ranked = sorted(sents[1:], key=lambda s: -len(q & _words(s)))
        best = [s for s in ranked[:3] if q & _words(s)]
        if not best:
            return "I could not find this in the policy corpus."
        parts = [f"\"{s}\" [{doc['doc_id']}]." for s in best]
        calc = next((m.content for m in tool_msgs if re.fullmatch(r"-?\d+(\.\d+)?", m.content.strip())), None)
        prefix = f"The calculation gives {calc}. " if calc else ""
        return prefix + "According to the policy, " + " ".join(parts)
