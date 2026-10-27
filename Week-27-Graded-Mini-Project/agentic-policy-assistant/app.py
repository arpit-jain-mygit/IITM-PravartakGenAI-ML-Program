"""
CLI entry point.

  python app.py ask "Is MFA required for VPN? Cite and quote."     # one question, verbose trace
  python app.py eval --cold                                        # clear memory, run the 8 canned questions
  python app.py eval                                               # warm run (memory kept from before)
  python app.py reset-memory

Options:
  --offline        use the offline stand-in model (no API key; outputs go to runs_offline/)
  --max-steps N    graph budget (default 8)
  --sqlite         persist graph checkpoints to runs/checkpoints.sqlite
  --report FILE    also export the JSON run report to FILE
  --quiet          hide the AgentExecutor's verbose trace
"""

import argparse
import json
import os
import shutil
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).parent


def main(argv=None):
    ap = argparse.ArgumentParser(description="Policy Assistant Agent")
    ap.add_argument("command", choices=["ask", "eval", "reset-memory"])
    ap.add_argument("question", nargs="?")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--cold", action="store_true", help="eval: clear memory first")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--sqlite", action="store_true")
    ap.add_argument("--report")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")          # keep library deprecation notices (with install paths) out of the output

    # the run directory must be chosen before agent.py is imported (it sets up the LLM cache there)
    os.environ.setdefault("POLICY_AGENT_RUN_DIR", str(HERE / ("runs_offline" if a.offline else "runs")))

    from agent import RUN_DIR, build_agent, rel
    from graph import build_graph, run_graph, sqlite_checkpointer
    from memory import FactMemory

    memory = FactMemory(RUN_DIR / "memory")
    if a.command == "reset-memory" or (a.command == "eval" and a.cold):
        memory.clear()
        print(f"memory cleared ({rel(RUN_DIR / 'memory')})")
        if a.command == "reset-memory":
            return

    if not a.offline and not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set: add it to .env, or use --offline for a no-key demo run.")
    llm = None
    if a.offline:
        from fake_llm import OfflinePolicyLLM
        llm = OfflinePolicyLLM()
    executor = build_agent(llm, verbose=not a.quiet)
    app = build_graph(executor=executor, memory=memory, checkpointer=sqlite_checkpointer() if a.sqlite else None)

    if a.command == "ask":
        if not a.question:
            sys.exit("ask needs a question")
        out = run_graph(a.question, app=app, max_steps=a.max_steps, thread_id="cli" if a.sqlite else None)
        print("\n--- node trace ---")
        print("\n".join(out["trace"]))
        print("\n--- result ---")
        print(f"status={out['status']}  latency={out['latency_s']}s  tool_steps={out['agent_steps']}  "
              f"graph_steps={out['graph_steps']}  memory_hit={bool(out.get('memory_hit'))}  "
              f"memory_written={out.get('memory_written')}")
        print("ANSWER:", out["answer"])
        if a.report:
            Path(a.report).write_text(json.dumps(out, indent=2, default=str))
    else:
        from eval import print_table, run_eval
        label = ("cold" if a.cold else "warm") + ("_offline" if a.offline else "")
        df = run_eval(app, label, max_steps=a.max_steps)
        print_table(df, label)
        if a.report:
            shutil.copy(RUN_DIR / f"report_{label}.json", a.report)
        print(f"\nlogs: {rel(RUN_DIR / 'logs')}   table: {rel(RUN_DIR / f'eval_{label}.csv')}")


if __name__ == "__main__":
    main()
