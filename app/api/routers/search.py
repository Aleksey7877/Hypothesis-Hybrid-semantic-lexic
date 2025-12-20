from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db.models import ParentChunk
from app.db.postgres import SessionLocal
from app.services.lexical.elastic import ElasticStore
from app.services.retrieval.fusion import FusionConfig, weighted_rrf_fuse
from app.services.reranking.bge import rerank_passages

log = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)

    top_k: int = Field(default=5, ge=1, le=50)

    k_sem: int = Field(default=10, ge=1, le=50)
    k_lex: int = Field(default=10, ge=1, le=50)
    k_rrf: int = Field(default=10, ge=1, le=50)

    # сколько кандидатов прогонять через реранкер (по умолчанию = max(k_sem,k_lex,k_rrf))
    k_rerank: Optional[int] = Field(default=None, ge=1, le=50)

    # Оставляем поле для совместимости со старым req.json,
    # но в ответе debug больше не возвращаем и highlight не включаем.
    debug: bool = Field(default=False)

    # если true — выдача (hits) будет уже в порядке reranker-а
    rerank_output: bool = Field(default=False)


class SearchHit(BaseModel):
    parent_id: str

    # итоговый скор выдачи:
    # - если rerank_output=true -> rerank_score
    # - иначе -> rrf_score
    score: float = 0.0

    # поля для эксперимента (before/after)
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    before_rank: Optional[int] = None
    after_rank: Optional[int] = None

    source: str = ""
    title: str = ""
    anchor: str = ""
    section_path: str = ""
    part_no: int = 1
    doc_id: str = ""
    content: str = ""


class SearchResponse(BaseModel):
    query: str
    hits: List[SearchHit]


def _truthy(v: str | None) -> bool:
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _fallback_postgres_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    q = f"%{query}%"
    with SessionLocal() as s:
        stmt = (
            select(
                ParentChunk.parent_id,
                ParentChunk.doc_id,
                ParentChunk.title,
                ParentChunk.anchor,
                ParentChunk.section_path,
                ParentChunk.part_no,
                ParentChunk.content,
            )
            .where(ParentChunk.content.ilike(q))
            .limit(top_k)
        )
        rows = s.execute(stmt).all()

    return [
        {
            "parent_id": str(r.parent_id),
            "score": 0.0,
            "source": "postgres_fallback",
            "title": r.title or "",
            "anchor": r.anchor or "",
            "section_path": r.section_path or "",
            "part_no": int(r.part_no or 1),
            "doc_id": r.doc_id or "",
            "content": r.content or "",
        }
        for r in rows
    ]


def _load_parents(parent_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not parent_ids:
        return {}
    with SessionLocal() as s:
        stmt = (
            select(
                ParentChunk.parent_id,
                ParentChunk.doc_id,
                ParentChunk.title,
                ParentChunk.anchor,
                ParentChunk.section_path,
                ParentChunk.part_no,
                ParentChunk.content,
            )
            .where(ParentChunk.parent_id.in_(parent_ids))
        )
        rows = s.execute(stmt).all()

    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        out[str(r.parent_id)] = {
            "parent_id": str(r.parent_id),
            "doc_id": str(r.doc_id or ""),
            "title": str(r.title or ""),
            "anchor": str(r.anchor or ""),
            "section_path": str(r.section_path or ""),
            "part_no": int(r.part_no or 1),
            "content": str(r.content or ""),
        }
    return out


def _passage_text(meta_row: Dict[str, Any]) -> str:
    title = (meta_row.get("title") or "").strip()
    section = (meta_row.get("section_path") or "").strip()
    anchor = (meta_row.get("anchor") or "").strip()
    content = (meta_row.get("content") or "").strip()

    header_parts = [p for p in [title, section, anchor] if p]
    header = " — ".join(header_parts)

    if header and content:
        return f"{header}\n\n{content}"
    if content:
        return content
    return header


def _truncate_for_rerank(text: str, max_chars: int = 6000) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + "…"


def _unique_ids(seq: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


@lru_cache(maxsize=1)
def _qdrant_store():
    from app.services.embeddings.e5 import E5Embedder
    from app.services.vectorstore.qdrant import QdrantStore

    return QdrantStore(E5Embedder())


@lru_cache(maxsize=1)
def _elastic_store() -> ElasticStore:
    es = ElasticStore()
    es.ensure_index(recreate=_truthy(os.getenv("ELASTIC_RECREATE", "0")))
    return es


def _qdrant_parent_search(query: str, parents_k: int) -> List[Dict[str, Any]]:
    """
    semantic: Qdrant возвращает children → агрегируем в parents (берём max score по parent_id).
    """
    store = _qdrant_store()
    children_k = min(400, max(parents_k * 12, 60))
    raw = store.search(query, top_k=children_k)

    best_by_parent: Dict[str, float] = {}
    for h in raw:
        pid = (h.get("parent_id") or "").strip()
        if not pid:
            continue
        score = float(h.get("score") or 0.0)
        prev = best_by_parent.get(pid)
        if prev is None or score > prev:
            best_by_parent[pid] = score

    hits = [{"parent_id": pid, "score": sc} for pid, sc in best_by_parent.items()]
    hits.sort(key=lambda x: float(x["score"]), reverse=True)
    return hits[:parents_k]


def _elastic_parent_search(query: str, parents_k: int) -> List[Dict[str, Any]]:
    """
    lexical: ES /_search size=K → parent_id + bm25 _score
    (highlight выключен, потому что debug больше не отдаём)
    """
    es = _elastic_store()
    hits = es.search(query, top_k=parents_k, highlight=False)

    out: List[Dict[str, Any]] = []
    for h in hits:
        pid = str(h.get("parent_id") or h.get("_id") or "").strip()
        if not pid:
            continue
        out.append(
            {
                "parent_id": pid,
                "bm25_score": float(h.get("score") or 0.0),
            }
        )
    return out


@router.post("", response_model=SearchResponse)
@router.post("/", response_model=SearchResponse, include_in_schema=False)
def search(req: SearchRequest) -> SearchResponse:
    query = req.query.strip()
    top_k = int(req.top_k)

    k_sem = int(req.k_sem)
    k_lex = int(req.k_lex)
    k_rrf = int(req.k_rrf)
    k_rerank = int(req.k_rerank) if req.k_rerank is not None else max(k_sem, k_lex, k_rrf)

    max_k = max(top_k, k_sem, k_lex, k_rrf)
    parents_k = min(200, max(max_k * 10, 50))

    cfg = FusionConfig(
        k0=int(os.getenv("RRF_K0", "60")),
        w_qdrant=float(os.getenv("W_QDRANT", "1.0")),
        w_elastic=float(os.getenv("W_ELASTIC", "1.0")),
    )

    q_hits: List[Dict[str, Any]] = []
    e_hits: List[Dict[str, Any]] = []

    try:
        q_hits = _qdrant_parent_search(query, parents_k)
    except Exception as e:
        log.warning("[search] qdrant failed err=%r", e)

    try:
        e_hits = _elastic_parent_search(query, parents_k)
    except Exception as e:
        log.warning("[search] elastic failed err=%r", e)

    if not q_hits and not e_hits:
        raw = _fallback_postgres_search(query, top_k)
        return SearchResponse(query=query, hits=[SearchHit(**h) for h in raw])

    # RRF по рангам (берём списки parent_id в порядке убывания)
    q_ids_all = [h["parent_id"] for h in q_hits]
    e_ids_all = [h["parent_id"] for h in e_hits]

    fuse_top = max(top_k, k_rrf)
    fused_rows_all = weighted_rrf_fuse(q_ids_all, e_ids_all, top_k=fuse_top, cfg=cfg)
    fused_top_rows = fused_rows_all[:k_rrf]

    # базовая выдача: top_k из RRF
    out_rows: List[Dict[str, Any]] = fused_rows_all[:top_k]

    # метаданные под то, что реально понадобится:
    need_ids = [r["parent_id"] for r in out_rows]
    if req.rerank_output:
        rerank_pool = min(len(fused_top_rows), k_rerank)
        need_ids += [r["parent_id"] for r in fused_top_rows[:rerank_pool]]

    meta = _load_parents(_unique_ids(need_ids))

    # rerank только RRF-top (кандидаты берём из fused_top_rows)
    if req.rerank_output:
        rerank_pool = min(len(fused_top_rows), k_rerank)

        rrf_passages: List[Dict[str, Any]] = []
        base_by_id: Dict[str, Dict[str, Any]] = {}

        for i, r in enumerate(fused_top_rows[:rerank_pool]):
            pid = r["parent_id"]
            base_by_id[pid] = r
            m = meta.get(pid, {"parent_id": pid})
            rrf_passages.append(
                {
                    "parent_id": pid,
                    "text": _truncate_for_rerank(_passage_text(m)),
                    "base_rank": i + 1,  # before_rank (RRF)
                    "base_score": float(r.get("rrf_score", 0.0)),  # rrf_score
                    "source": str(r.get("source", "")),
                }
            )

        reranked = rerank_passages(query, rrf_passages)

        new_rows: List[Dict[str, Any]] = []
        for after_rank, p in enumerate(reranked[:top_k], start=1):
            pid = str(p.get("parent_id") or "")
            base = base_by_id.get(pid) or {"parent_id": pid, "source": ""}

            rrf_score = float(p.get("base_score", base.get("rrf_score", 0.0)) or 0.0)
            ce_score = float(p.get("rerank_score", 0.0) or 0.0)

            new_rows.append(
                {
                    "parent_id": pid,
                    "source": str(p.get("source", base.get("source", ""))),
                    "rrf_score": rrf_score,
                    "rerank_score": ce_score,
                    "before_rank": int(p.get("base_rank", 0) or 0),
                    "after_rank": int(after_rank),
                    "score": ce_score,  # итоговый
                }
            )
        out_rows = new_rows

    hits: List[SearchHit] = []
    for row in out_rows:
        pid = row["parent_id"]
        m = meta.get(pid, {"parent_id": pid})

        rrf_score = row.get("rrf_score")
        rerank_score = row.get("rerank_score")

        final_score = (
            float(row.get("score"))
            if row.get("score") is not None
            else float(row.get("rrf_score", 0.0))
        )

        hits.append(
            SearchHit(
                parent_id=pid,
                score=final_score,
                rrf_score=float(rrf_score) if rrf_score is not None else float(row.get("rrf_score", 0.0)),
                rerank_score=float(rerank_score) if rerank_score is not None else None,
                before_rank=int(row.get("before_rank")) if row.get("before_rank") is not None else None,
                after_rank=int(row.get("after_rank")) if row.get("after_rank") is not None else None,
                source=str(row.get("source", "")),
                title=str(m.get("title", "")),
                anchor=str(m.get("anchor", "")),
                section_path=str(m.get("section_path", "")),
                part_no=int(m.get("part_no", 1) or 1),
                doc_id=str(m.get("doc_id", "")),
                content=str(m.get("content", "")),
            )
        )

    return SearchResponse(query=query, hits=hits)
