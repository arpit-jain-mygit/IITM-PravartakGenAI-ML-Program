"""
Phase 5 - the Intelligent Retail Decision Assistant: a hybrid, inspectable controller.

Routing (in this order, every decision written to the trace):
 1. Guardrail: empty / very long input, or clearly out-of-scope topics -> polite refusal, no tool call.
 2. Rule-based intent scoring: each tool has a list of regex cues; a tool is selected when its score >= 1.
    A query can trigger at most MAX_TOOL_CALLS tools (e.g. "recommend sneakers and can I return them?").
 3. LLM fallback (live mode only): if no rule fires, the LLM chooses ONE tool from the fixed list or "none",
    as JSON {"tool": ..., "reason": ...}. Its choice is validated against the list.
 4. Each tool returns a structured result with its own trace; the controller validates grounding
    (citations for policy answers, supporting ids for products) and assembles the final response.
Session memory: the last product search is kept so "summarise the top three" can refer to it.
"""

import re
import time

from src.config import MAX_TOOL_CALLS
from src.llm import chat_json
from src.policy_qa import answer_policy
from src.recommender import recommend
from src.tools import calculator, review_summary, segment_insights, summarise_products

TOOLS = {
    "product_search": "recommend / find products from the catalogue (grounded in retrieved items)",
    "policy_qa": "answer return / refund / exchange / shipping / warranty questions from policy documents",
    "segment_insights": "business analytics on customer segments from the clustering model",
    "review_summary": "summarise customer reviews for a type of product",
    "summarise_products": "summarise / compare the products returned by the previous search",
    "calculator": "arithmetic such as discounts and totals",
}

CUES = {
    "policy_qa": [r"\breturn(s|ed|ing)?\b(?!.*segment)", r"refund", r"exchange", r"warranty", r"guarantee",
                  r"shipping", r"deliver(y|ed)", r"how long .*(take|arrive)", r"\bpolicy\b", r"cancel"],
    "product_search": [r"recommend", r"suggest", r"\bfind\b", r"\bshow\b", r"looking for", r"\bneed\b", r"\bwant\b",
                       r"\bbuy\b", r"(under|below|budget).*\d{3,}", r"sneaker|shoe|watch|backpack|wallet|jacket|jeans|shirt|cap\b"],
    "segment_insights": [r"segment", r"cluster", r"persona", r"which customers", r"customer group",
                         r"price[- ]sensitiv", r"most valuable customers"],
    "review_summary": [r"review", r"what do (customers|people|buyers) say", r"feedback", r"complain"],
    "summarise_products": [r"summari[sz]e .*(retrieved|above|these|those|top|previous|results)",
                           r"compare (the|these|those) (products|items|top)", r"why (they|these) fit"],
    "calculator": [r"\d+\s*%\s*(off|discount)", r"\d+(\.\d+)?\s*[-+*/x]\s*\d+", r"\bcalculate\b|\bcompute\b"],
}
# tools that should not fire together with a more specific one
SUPPRESS = {"segment_insights": ["product_search", "policy_qa"], "review_summary": ["product_search"],
            "summarise_products": ["product_search"], "calculator": ["product_search"]}
SHOPPING_INTENT = r"recommend|suggest|\bfind\b|\bshow\b|looking for|\bneed\b|\bwant\b|\bbuy\b|(under|below|budget).*\d{3,}"
OUT_OF_SCOPE = r"weather|stock market|share price|politic|cricket|recipe|poem|joke|bitcoin|medical|diagnos"

ROUTER_SYSTEM = ("You route retail assistant requests. Choose exactly one tool from this list or 'none' if the "
                 "request is unrelated to the store's products, policies, reviews or customer analytics.\n" +
                 "\n".join(f"- {k}: {v}" for k, v in TOOLS.items()) +
                 '\nReturn JSON only: {"tool": "<name or none>", "reason": "short reason"}')


def score_intents(query: str) -> dict:
    q = query.lower()
    return {tool: sum(bool(re.search(p, q)) for p in pats) for tool, pats in CUES.items()}


def route(query: str) -> tuple[list[str], list[str]]:
    trace = []
    q = query.strip()
    if not q or len(q) > 500:
        return [], ["guardrail: empty or overly long input -> no tool"]
    if re.search(OUT_OF_SCOPE, q.lower()):
        return [], ["guardrail: out-of-scope topic -> polite refusal, no tool"]
    scores = score_intents(q)
    trace.append("intent scores: " + ", ".join(f"{t}={s}" for t, s in scores.items() if s) if any(scores.values())
                 else "intent scores: none matched")
    chosen = [t for t, s in sorted(scores.items(), key=lambda kv: -kv[1]) if s >= 1]
    # a product noun alone ("my watch") inside a policy question is not a shopping request
    if "policy_qa" in chosen and "product_search" in chosen and not re.search(SHOPPING_INTENT, q.lower()):
        chosen.remove("product_search")
        trace.append("product noun without shopping intent inside a policy question -> product_search dropped")
    for strong, weaker in SUPPRESS.items():
        if strong in chosen:
            dropped = [w for w in weaker if w in chosen and scores[w] <= scores[strong] + 1]
            # keep policy_qa when the query really asks about a policy too (multi-intent)
            dropped = [w for w in dropped if not (w == "policy_qa" and scores[w] >= 2)]
            chosen = [t for t in chosen if t not in dropped]
            if dropped:
                trace.append(f"'{strong}' is more specific -> suppressed {dropped}")
    if not chosen:
        resp = chat_json(ROUTER_SYSTEM, query, max_tokens=80)
        if resp and resp["data"].get("tool") in TOOLS:
            chosen = [resp["data"]["tool"]]
            trace.append(f"no rule fired -> LLM router chose {chosen[0]}: {resp['data'].get('reason', '')}")
        else:
            trace.append("no rule fired and no valid LLM route -> ask the user to rephrase")
    if len(chosen) > MAX_TOOL_CALLS:
        trace.append(f"bounded: keeping first {MAX_TOOL_CALLS} of {chosen}")
        chosen = chosen[:MAX_TOOL_CALLS]
    # run policy questions after product search so a combined answer reads naturally
    order = ["product_search", "summarise_products", "review_summary", "policy_qa", "segment_insights", "calculator"]
    chosen.sort(key=order.index)
    trace.append(f"selected tools: {chosen}")
    return chosen, trace


def sub_query(tool: str, query: str, tools: list[str]) -> str:
    """For multi-intent queries, give each tool only the clause that is about it."""
    if len(tools) < 2:
        return query
    clauses = [c.strip() for c in re.split(r"\band\b(?= (?:tell|let|check|can|is|what|how|does|do|will|if|also))|[?;]|\balso\b", query) if c and c.strip()]
    mine = [c for c in clauses if any(re.search(p, c.lower()) for p in CUES[tool])]
    return " ".join(mine) if mine else query


class RetailAssistant:
    def __init__(self):
        self.last_products = None          # session memory for "summarise the top three"
        self.history = []

    def ask(self, query: str, verbose: bool = True) -> dict:
        t0 = time.perf_counter()
        tools, trace = route(query)
        results = []
        for i, tool in enumerate(tools, 1):
            t1 = time.perf_counter()
            tq = sub_query(tool, query, tools)
            if tq != query:
                trace.append(f"multi-intent: {tool} receives clause '{tq}'")
            if tool == "product_search":
                r = recommend(tq)
                self.last_products = r
            elif tool == "policy_qa":
                r = answer_policy(tq)
            elif tool == "segment_insights":
                r = segment_insights(query)
            elif tool == "review_summary":
                r = review_summary(query)
            elif tool == "summarise_products":
                r = summarise_products(query, self.last_products)
            else:
                r = calculator(query)
            r["latency_s"] = round(time.perf_counter() - t1, 3)
            trace.append(f"step {i}: {tool} -> status={r['status']} ({r['latency_s']}s)")
            trace += [f"    {line}" for line in r.get("trace", [])]
            results.append(r)
        final = self._compose(query, results)
        out = {"query": query, "tools": tools, "results": results, "final_answer": final, "trace": trace,
               "latency_s": round(time.perf_counter() - t0, 3)}
        self.history.append(out)
        if verbose:
            print_response(out)
        return out

    @staticmethod
    def _compose(query, results) -> str:
        if not results:
            return ("I can help with product recommendations, store policies (returns, refunds, exchange, shipping, "
                    "warranty), customer reviews and customer-segment insights. Could you rephrase your request?")
        parts = []
        for r in results:
            if r["tool"] == "product_search":
                if r["recommended_products"]:
                    parts.append("Recommended products:\n" + "\n".join(
                        f"  - [{p['item_id']}] {p['title']} - Rs {p['price']:,.0f}: {p.get('why', '')}"
                        for p in r["recommended_products"]) + f"\n  Why: {r.get('explanation', '')}")
                else:
                    parts.append(r.get("explanation", "No matching products."))
            elif r["tool"] == "policy_qa":
                cites = f" [sources: {', '.join(r['citations'])}]" if r.get("citations") else ""
                parts.append(f"Policy: {r['answer']}{cites}")
            elif r["tool"] == "summarise_products" and r.get("summary"):
                parts.append("Summary of the top results:\n" + "\n".join(
                    f"  - [{s['item_id']}] {s['fit']}" for s in r["summary"]) + f"\n  {r['answer']}")
            else:
                parts.append(r["answer"])
        return "\n\n".join(parts)


def print_response(out: dict):
    print(f"USER: {out['query']}")
    print("TRACE:")
    for line in out["trace"]:
        print(f"  {line}")
    print(f"ASSISTANT ({out['latency_s']}s):\n{out['final_answer']}\n" + "-" * 100)
