from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from app.services.eval.metrics import recall_at_k, mrr
from app.services.retrieval.retriever import Retriever


@dataclass
class EvalConfig:
    baseline_top_children: int = 3
    method_top_children: int = 20
    method_top_parents: int = 5


def _parents_from_children(child_hits: List[Dict[str, Any]], k_parents: int) -> List[str]:
    seen = set()
    out: list[str] = []
    for h in child_hits:
        pid = str(h.get("parent_id", ""))
        if pid and pid not in seen:
            seen.add(pid)
            out.append(pid)
        if len(out) >= k_parents:
            break
    return out


def run_eval(
    questions: List[Dict[str, Any]],
    retriever: Retriever,
    cfg: EvalConfig = EvalConfig(),
) -> Dict[str, Any]:
    rows: list[dict] = []

    agg = {
        "baseline_recall@5": 0.0,
        "baseline_mrr@5": 0.0,
        "method_recall@5": 0.0,
        "method_mrr@5": 0.0,
        "n": 0,
    }

    for q in questions:
        query = str(q.get("query", "")).strip()
        gold_parent = str(q.get("gold_parent_id", "")).strip()
        if not query:
            continue

        # Baseline A: top3 children -> unique parents
        base_children = retriever.search_children(query, top_k=cfg.baseline_top_children)
        base_parent_rank = _parents_from_children(base_children, k_parents=5)

        # Method B: top20 children -> aggregator -> top parents
        method = retriever.search(query, top_k_children=cfg.method_top_children, top_k_parents=cfg.method_top_parents)
        method_parent_rank = [p["parent_id"] for p in method["parents"]]

        row = {
            "query": query,
            "gold_parent_id": gold_parent,
            "baseline_parents": base_parent_rank,
            "method_parents": method_parent_rank,
            "baseline_recall@5": recall_at_k(base_parent_rank, gold_parent, 5),
            "baseline_mrr@5": mrr(base_parent_rank, gold_parent, 5),
            "method_recall@5": recall_at_k(method_parent_rank, gold_parent, 5),
            "method_mrr@5": mrr(method_parent_rank, gold_parent, 5),
        }
        rows.append(row)

        agg["baseline_recall@5"] += row["baseline_recall@5"]
        agg["baseline_mrr@5"] += row["baseline_mrr@5"]
        agg["method_recall@5"] += row["method_recall@5"]
        agg["method_mrr@5"] += row["method_mrr@5"]
        agg["n"] += 1

    n = max(agg["n"], 1)
    summary = {
        "baseline_recall@5": agg["baseline_recall@5"] / n,
        "baseline_mrr@5": agg["baseline_mrr@5"] / n,
        "method_recall@5": agg["method_recall@5"] / n,
        "method_mrr@5": agg["method_mrr@5"] / n,
        "n": agg["n"],
    }

    return {"summary": summary, "rows": rows}
