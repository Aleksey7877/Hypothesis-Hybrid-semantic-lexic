from __future__ import annotations

from typing import List, Optional


def recall_at_k(ranked_ids: List[str], gold_id: str, k: int) -> float:
    if not gold_id:
        return 0.0
    return 1.0 if gold_id in ranked_ids[:k] else 0.0


def mrr(ranked_ids: List[str], gold_id: str, k: Optional[int] = None) -> float:
    if not gold_id:
        return 0.0
    ids = ranked_ids if k is None else ranked_ids[:k]
    for i, rid in enumerate(ids, start=1):
        if rid == gold_id:
            return 1.0 / float(i)
    return 0.0
