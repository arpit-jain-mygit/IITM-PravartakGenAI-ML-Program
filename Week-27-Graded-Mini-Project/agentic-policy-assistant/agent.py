"""
ReAct-style tool-calling agent: the model decides which tool to call, observes the result, and repeats
until it can give a final answer or hits MAX_ITERATIONS.

Deterministic defaults: temperature 0 and a fixed seed. Verbose tracing prints every tool call and
observation. Every LLM response is cached on disk, so an identical rerun gives identical output.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.cache import SQLiteCache
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.globals import set_llm_cache
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from models import AgentResult
from prompts import MAX_ITERATIONS, SYSTEM_PROMPT
from tools import TOOLS

load_dotenv(Path(__file__).parent / ".env")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SEED = 42
RUN_DIR = Path(os.getenv("POLICY_AGENT_RUN_DIR", Path(__file__).parent / "runs"))
RUN_DIR.mkdir(parents=True, exist_ok=True)
set_llm_cache(SQLiteCache(database_path=str(RUN_DIR / "llm_cache.sqlite")))


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """OpenAI chat model (or a Vocareum-compatible endpoint via OPENAI_BASE_URL), temperature 0."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0, seed=SEED, max_retries=3, timeout=60)


class TokenCounter(BaseCallbackHandler):
    """Adds up prompt/completion tokens across all LLM calls of one agent run."""
    def __init__(self):
        self.input_tokens = self.output_tokens = 0

    def on_llm_end(self, response, **kwargs):
        for gen in response.generations:
            for g in gen:
                um = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                self.input_tokens += um.get("input_tokens", 0)
                self.output_tokens += um.get("output_tokens", 0)


def build_agent(llm: BaseChatModel | None = None, verbose: bool = True,
                max_iterations: int = MAX_ITERATIONS) -> AgentExecutor:
    """Agent/executor factory. Pass a fake llm in tests; by default the OpenAI model is used."""
    llm = llm or get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),        # tool calls + observations go here (ReAct loop)
    ])
    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=TOOLS, verbose=verbose, max_iterations=max_iterations,
                         early_stopping_method="force", return_intermediate_steps=True,
                         handle_parsing_errors=True)


def run_agent(question: str, executor: AgentExecutor | None = None, verbose: bool = True) -> AgentResult:
    executor = executor or build_agent(verbose=verbose)
    counter = TokenCounter()
    t0 = time.perf_counter()
    out = executor.invoke({"input": question}, config={"callbacks": [counter]})
    latency = time.perf_counter() - t0
    steps = out.get("intermediate_steps", [])
    calls = [{"tool": a.tool, "input": a.tool_input, "output_preview": str(obs)[:160]} for a, obs in steps]
    answer = out["output"] if isinstance(out["output"], str) else str(out["output"])
    return AgentResult(answer=answer.strip(), tool_calls=calls, steps=len(steps),
                       hit_iteration_limit="Agent stopped due to max iterations" in answer,
                       latency_s=round(latency, 3), input_tokens=counter.input_tokens,
                       output_tokens=counter.output_tokens)
