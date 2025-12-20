from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class FusionConfig:
    k0: int = 60
    w_qdrant: float = 1.0
    w_elastic: float = 1.0


def weighted_rrf_fuse(
    qdrant_ids: List[str],
    elastic_ids: List[str],
    *,
    top_k: int,
    cfg: FusionConfig | None = None,
) -> List[Dict[str, Any]]:
    cfg = cfg or FusionConfig()

    q_rank = {pid: i + 1 for i, pid in enumerate(qdrant_ids) if pid}
    e_rank = {pid: i + 1 for i, pid in enumerate(elastic_ids) if pid}

    all_ids = set(q_rank) | set(e_rank)
    rows: List[Dict[str, Any]] = []

    for pid in all_ids:
        qr = q_rank.get(pid)
        er = e_rank.get(pid)

        score = 0.0
        if qr is not None:
            score += float(cfg.w_qdrant) / float(cfg.k0 + qr)
        if er is not None:
            score += float(cfg.w_elastic) / float(cfg.k0 + er)

        if (qr is not None) and (er is not None):
            src = "both"
        elif qr is not None:
            src = "qdrant"
        else:
            src = "elastic"

        rows.append(
            {"parent_id": pid, "rrf_score": score, "q_rank": qr, "e_rank": er, "source": src}
        )

    def _min_rank(x: Dict[str, Any]) -> int:
        big = 10**9
        qr = x.get("q_rank")
        er = x.get("e_rank")
        return min(qr if qr is not None else big, er if er is not None else big)

    rows.sort(key=lambda x: (-float(x["rrf_score"]), _min_rank(x), x["parent_id"]))

    out = rows[: int(top_k)]
    for i, r in enumerate(out, start=1):
        r["rank"] = i
    return out
