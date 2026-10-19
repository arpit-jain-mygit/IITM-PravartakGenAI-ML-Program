"""
Turns the logs + reviewed scores into the summary tables, example Q/A pairs and the recommendation.
Used by both the notebook (Task 8) and make_report.py, so the two always agree.
"""

import json
from pathlib import Path

import pandas as pd

SC = ["relevance", "groundedness", "citation", "conciseness", "memory", "refusal_correct"]
MAX = {"relevance": 2, "groundedness": 2, "citation": 2, "conciseness": 1, "memory": 2, "refusal_correct": 1}
CRIT = ["relevance", "groundedness", "citation", "conciseness", "memory", "refusal"]


def load_eval(work_dir):
    """Logged answers joined with the (reviewed) rubric scores."""
    work_dir = Path(work_dir)
    log = pd.read_csv(work_dir / "logs" / "run_log.csv",
                      converters={c: eval for c in ["citations", "used_chunks", "used_pages"]})
    scores = pd.read_csv(work_dir / "logs" / "manual_scores.csv")
    return log.merge(scores, on=["run_id", "qid"], how="left", validate="one_to_one")


def runs_summary(ev):
    norm = ev.copy()
    for c in SC:
        norm[c] = norm[c] / MAX[c]                         # 0-1 so criteria are comparable
    agg = norm.groupby("run_id").agg(
        change=("change", "first"), chunk=("chunk_size", "first"), overlap=("overlap", "first"),
        embedding=("embedding", "first"), prompt=("prompt_id", "first"),
        relevance=("relevance", "mean"), groundedness=("groundedness", "mean"), citation=("citation", "mean"),
        conciseness=("conciseness", "mean"), memory=("memory", "mean"), refusal=("refusal_correct", "mean"),
        avg_tokens_out=("answer_tokens", "mean"), avg_latency_s=("latency_s", "mean"),
        input_tokens=("input_tokens", "sum"), output_tokens=("output_tokens", "sum"))
    agg["total_tokens"] = agg.pop("input_tokens") + agg.pop("output_tokens")
    agg["overall"] = agg[CRIT].mean(axis=1)
    pts = ev.groupby("run_id")[SC].sum().sum(axis=1)
    mx = ev[SC].notna().mul(pd.Series(MAX)).sum(axis=1).groupby(ev["run_id"]).sum()
    agg["points"] = pts.astype(int).astype(str) + " / " + mx.astype(int).astype(str)
    return agg.sort_index(key=lambda s: s.str[1:].astype(int)).round(3)


def _total(r):
    return sum(r[c] for c in SC if pd.notna(r[c]))


def pick_examples(ev, agg):
    """A follow-up win, a boundary-question result, and the weakest answer overall (a failure)."""
    ev = ev.assign(total=ev.apply(_total, axis=1),
                   maxpts=ev[SC].notna().mul(pd.Series(MAX)).sum(axis=1))
    ev["frac"] = ev["total"] / ev["maxpts"]
    best_run = agg["overall"].idxmax()
    picks = []
    fu = ev[(ev.run_id == best_run) & (ev.q_type == "follow-up")].sort_values(["frac", "qid"], ascending=[False, False])
    picks.append(("Win: follow-up resolved through memory", fu.iloc[0]))
    bd = ev[ev.q_type == "boundary"].sort_values("frac")
    worst_bd = bd.iloc[0]
    if worst_bd["frac"] < 1:
        picks.append(("Failure: boundary question", worst_bd))
    else:
        picks.append(("Win: correct refusal on an out-of-corpus question", bd[bd.run_id == best_run].iloc[0]))
    rest = ev[~ev.index.isin([p[1].name for p in picks])].sort_values(["frac", "run_id"])
    picks.append(("Failure: weakest answer across all runs" if rest.iloc[0]["frac"] < 1
                  else "No failures: lowest-scoring answer", rest.iloc[0]))
    return picks


def recommendation(agg, bench, cold):
    """Pick the best chunking, embedding and prompt from the one-factor-at-a-time runs."""
    a = agg
    # --- chunking: R1-R5 differ only in chunk size / overlap ---
    chunk_runs = a.loc[["R1", "R2", "R3", "R4", "R5"]]
    ob = bench[bench.embedding == "openai"].copy()
    ob["key"] = ob["chunk_size"].astype(str) + "/" + ob["overlap"]
    mrr = ob.set_index("key")["MRR"]
    chunk_runs = chunk_runs.assign(MRR=[mrr.get(f"{r.chunk}/{r.overlap}", 0) for r in chunk_runs.itertuples()])
    best_chunk = chunk_runs.sort_values(["overall", "MRR", "total_tokens"], ascending=[False, False, True]).iloc[0]

    # --- embeddings: R1 (OpenAI) vs R6 (BGE) + retrieval benchmark ---
    o, b = a.loc["R1"], a.loc["R6"]
    bm = bench.groupby("embedding")[["recall@4", "precision@4", "MRR"]].mean()
    quality_gap = o["overall"] - b["overall"]
    retrieval_gap = bm.loc["openai", "MRR"] - bm.loc["bge", "MRR"]
    use_bge = quality_gap <= 0.02 and retrieval_gap <= 0.05
    emb = "Open-source BGE-small" if use_bge else "OpenAI text-embedding-3-small"

    # --- prompt: R1 (P1) vs R7 (P2) ---
    p1, p2 = a.loc["R1"], a.loc["R7"]
    use_p2 = p2["overall"] > p1["overall"] + 0.02
    prompt = "P2 (Evidence -> Answer)" if use_p2 else "P1 (concise)"

    oc = cold.loc[[i for i in cold.index if "OpenAI" in i][0]]
    bc = cold.loc[[i for i in cold.index if "BGE" in i][0]]
    pq = "per query (ms)"
    lines = [
        f"**Chunking: {best_chunk.chunk} characters, {best_chunk.overlap} overlap** (run {best_chunk.name}). Among the "
        f"chunking runs it scored the highest overall ({best_chunk.overall:.2f}), with retrieval MRR "
        f"{best_chunk.MRR:.2f}. Smaller chunks give more precise passages, but a policy rule and its condition can be "
        f"split across chunks. Larger chunks keep rules together, but they put more unrelated text into the "
        f"prompt (total tokens per run: " +
        ", ".join(f"{r}={int(chunk_runs.loc[r, 'total_tokens'])}" for r in chunk_runs.index) + ").",
        f"**Embeddings: {emb}.** In the end-to-end runs, OpenAI scored {o.overall:.2f} and BGE {b.overall:.2f}. "
        f"Across all {bench.groupby(['chunk_size', 'overlap']).ngroups} chunking configurations, average retrieval was: OpenAI recall@4 {bm.loc['openai', 'recall@4']:.2f} "
        f"/ MRR {bm.loc['openai', 'MRR']:.2f}, against BGE {bm.loc['bge', 'recall@4']:.2f} / {bm.loc['bge', 'MRR']:.2f}. "
        f"OpenAI costs ${oc['cost per 1M tokens (USD)']} per 1M tokens (${oc['cost this corpus (USD)']:.6f} for this "
        f"corpus) and takes about {oc[pq]:.0f} ms per query over the network. BGE is free, runs locally in about "
        f"{bc[pq]:.0f} ms per query, and keeps policy text inside the company. " +
        ("Since quality is essentially equal, the free and private model wins." if use_bge else
         "The quality gain is large enough to justify the very small cost."),
        f"**Prompt: {prompt}.** P1 averaged {p1.avg_tokens_out:.0f} output tokens with overall {p1.overall:.2f}, "
        f"and P2 averaged {p2.avg_tokens_out:.0f} tokens with overall {p2.overall:.2f} (citation {p1.citation:.2f} vs "
        f"{p2.citation:.2f}). " +
        ("P2's evidence bullets made the citations more reliable and are worth the extra tokens." if use_p2 else
         "P2's evidence bullets make answers easier to audit, but they did not improve accuracy enough to justify "
         "the longer output. P1 stays the default, and P2 is useful when an auditor needs to see the evidence."),
    ]
    return {"chunk": f"{best_chunk.chunk} / {best_chunk.overlap}", "embedding": emb, "prompt": prompt,
            "lines": lines, "markdown": "\n\n".join(f"- {l}" for l in lines)}
