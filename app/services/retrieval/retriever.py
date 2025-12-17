from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

from app.db.repository import Repository
from app.services.embeddings.e5 import E5Embedder
from app.services.vectorstore.qdrant import QdrantStore
from app.services.lexical.elastic import ElasticStore
from app.services.retrieval.fusion import rrf_fuse_parents, FusionConfig


def _extract_parent_id(hit: Dict[str, Any]) -> str | None:
    """
    Унифицированно вытаскиваем parent_id и из Qdrant child-hit, и из Elastic parent-hit.

    Qdrant child-hit (обычно):
      {"child_id": "...", "score": ..., "payload": {"parent_id": "..."}}

    Elastic parent-hit (как мы делали в elastic.py):
      {"id": "...", "score": ..., "payload": {"parent_id": "...", ...}}
    """
    if hit.get("parent_id"):
        return str(hit["parent_id"])

    if hit.get("id"):
        # для Elastic: id == parent_id
        return str(hit["id"])

    payload = hit.get("payload") or {}
    if isinstance(payload, dict) and payload.get("parent_id"):
        return str(payload["parent_id"])

    return None


def _unique_parent_ids_from_children(child_hits: List[Dict[str, Any]], top_k: int) -> List[str]:
    """
    Children -> parent_id (top-k уникальных parent_id по порядку появления детей).
    Это и есть твой "top10 чайлдов->парентов от ретривала" без лишних весов.
    """
    seen: set[str] = set()
    out: List[str] = []

    for h in child_hits:
        pid = _extract_parent_id(h)
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
        if len(out) >= top_k:
            break

    return out


def _unique_parent_ids_from_elastic_hits(elastic_hits: List[Dict[str, Any]], top_k: int) -> List[str]:
    """Elastic hits -> parent_id list (в исходном порядке, уникально)."""
    seen: set[str] = set()
    out: List[str] = []

    for h in elastic_hits:
        pid = _extract_parent_id(h)
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
        if len(out) >= top_k:
            break

    return out


@dataclass
class RetrieverConfig:
    # Qdrant children
    top_k_children_raw: int = 50      # сколько children тянем из Qdrant (для агрегации)
    top_k_children_debug: int = 20    # сколько children возвращаем наружу (для анализа)

    # Parents
    top_k_parents_vec: int = 10       # сколько parents берём из Qdrant(children->parents)
    top_k_parents_lex: int = 10       # сколько parents берём из Elastic(parents)
    top_k_parents_out: int = 5        # сколько parents отдаём наружу


class Retriever:
    def __init__(
        self,
        repo: Repository | None = None,
        embedder: E5Embedder | None = None,
        qdrant: QdrantStore | None = None,
        elastic: ElasticStore | None = None,
        cfg: RetrieverConfig = RetrieverConfig(),
    ):
        self.repo = repo or Repository()
        self.embedder = embedder or E5Embedder()
        self.qdrant = qdrant or QdrantStore(embedder=self.embedder)
        self.elastic = elastic or ElasticStore()
        self.cfg = cfg

        self.rrf_k0 = int(os.getenv("RRF_K0", "60"))

        # Не обязательно, но полезно: если Elastic доступен — индекс будет создан/проверен
        try:
            self.elastic.ensure_index()
        except Exception:
            pass

    def search_children(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        """Только Qdrant по children (для анализа/дебага)."""
        k = top_k or self.cfg.top_k_children_debug
        raw_k = max(self.cfg.top_k_children_raw, k)
        hits = self.qdrant.search(query, top_k=raw_k) or []
        return hits[:k]

    def search(self, query: str) -> Dict[str, Any]:
        """
        Новая логика:
          1) top10 parents от Elastic (parents-index)
          2) top10 parents из Qdrant: children -> unique parent_id
          3) RRF без весов (parent-level)
          4) tie -> prefer retriever(Qdrant-side)
          5) выдаём top5 parents
        """
        # 1) Qdrant children (raw)
        q_children_raw = self.qdrant.search(query, top_k=self.cfg.top_k_children_raw) or []
        vec_parent_ids = _unique_parent_ids_from_children(q_children_raw, top_k=self.cfg.top_k_parents_vec)

        # 2) Elastic parents (raw)
        lex_hits = self.elastic.search(query, top_k=self.cfg.top_k_parents_lex) or []
        lex_parent_ids = _unique_parent_ids_from_elastic_hits(lex_hits, top_k=self.cfg.top_k_parents_lex)

        # 3) Parent-level RRF
        fused = rrf_fuse_parents(
            retriever_parent_ids=vec_parent_ids,
            elastic_parent_ids=lex_parent_ids,
            top_k=self.cfg.top_k_parents_out,
            cfg=FusionConfig(rrf_k=self.rrf_k0),
        )

        parent_ids = [x["parent_id"] for x in fused]
        meta = self.repo.get_parents_meta(parent_ids) if parent_ids else {}
        texts = self.repo.get_parent_texts(parent_ids) if parent_ids else {}

        for p in fused:
            pid = p["parent_id"]
            p.update(meta.get(pid, {}))
            p["content"] = texts.get(pid, "")

        return {
            "query": query,
            "children": (q_children_raw[: self.cfg.top_k_children_debug] if q_children_raw else []),
            "parents": fused,
            "mode": "hybrid_parent_rrf",
            "debug": {
                "vec_parent_ids_top10": vec_parent_ids,
                "lex_parent_ids_top10": lex_parent_ids,
                "rrf_k0": self.rrf_k0,
            },
        }
