from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class FusionConfig:
    rrf_k: int = 60  # k0 в формуле RRF


def rrf_fuse_parents(
    retriever_parent_ids: List[str],
    elastic_parent_ids: List[str],
    top_k: int,
    cfg: FusionConfig = FusionConfig(),
) -> List[Dict[str, Any]]:
    """
    RRF по parent_id, без весов.

    score(pid) = 1/(k0 + rank_retriever) + 1/(k0 + rank_elastic)

    Tie-break (если score одинаковый):
      1) предпочтение retriever (если pid есть в retriever списке)
      2) меньший rank_retriever
      3) меньший rank_elastic
      4) pid (стабильность)
    """
    k0 = int(cfg.rrf_k)

    r_rank: Dict[str, int] = {pid: i for i, pid in enumerate(retriever_parent_ids, start=1)}
    e_rank: Dict[str, int] = {pid: i for i, pid in enumerate(elastic_parent_ids, start=1)}

    all_ids = set(r_rank) | set(e_rank)

    fused: List[Dict[str, Any]] = []
    for pid in all_ids:
        rr: Optional[int] = r_rank.get(pid)
        er: Optional[int] = e_rank.get(pid)

        score = 0.0
        if rr is not None:
            score += 1.0 / (k0 + rr)
        if er is not None:
            score += 1.0 / (k0 + er)

        fused.append(
            {
                "parent_id": pid,
                "score": float(score),
                "rank_retriever": rr,
                "rank_elastic": er,
            }
        )

    # tie-break как ты просил: при равном score выигрывает retriever
    def in_retriever(pid: str) -> int:
        return 1 if pid in r_rank else 0

    fused.sort(
        key=lambda x: (
            -x["score"],
            -in_retriever(x["parent_id"]),               # tie -> retriever wins
            x["rank_retriever"] if x["rank_retriever"] is not None else 10**9,
            x["rank_elastic"] if x["rank_elastic"] is not None else 10**9,
            x["parent_id"],
        )
    )

    return fused[:top_k]
