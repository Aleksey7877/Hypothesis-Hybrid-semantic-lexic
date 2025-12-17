from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class AggregatorConfig:
    w_max: float = 1.0
    w_sum: float = 0.25
    w_cnt: float = 0.03


def aggregate_parents(
    child_hits: List[Dict[str, Any]],
    top_k: int = 5,
    cfg: AggregatorConfig = AggregatorConfig(),
) -> List[Dict[str, Any]]:
    """
    Простейшая агрегация:
      parent_score = w_max*max_child + w_sum*sum_child + w_cnt*count
    """
    agg: dict[str, dict] = {}

    for h in child_hits:
        pid = str(h.get("parent_id", ""))
        if not pid:
            continue
        rec = agg.setdefault(
            pid,
            {
                "parent_id": pid,
                "doc_id": str(h.get("doc_id", "")),
                "max": 0.0,
                "sum": 0.0,
                "cnt": 0,
                "child_ids": [],
            },
        )
        sc = float(h.get("score", 0.0))
        rec["max"] = max(rec["max"], sc)
        rec["sum"] += sc
        rec["cnt"] += 1
        rec["child_ids"].append(str(h.get("child_id", "")))

    out: list[dict] = []
    for pid, r in agg.items():
        score = cfg.w_max * r["max"] + cfg.w_sum * r["sum"] + cfg.w_cnt * float(r["cnt"])
        out.append(
            {
                "parent_id": pid,
                "doc_id": r["doc_id"],
                "score": float(score),
                "child_ids": r["child_ids"],
                "cnt": r["cnt"],
            }
        )

    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top_k]
