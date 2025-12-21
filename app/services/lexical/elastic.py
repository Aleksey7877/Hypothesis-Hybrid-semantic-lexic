from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx

log = logging.getLogger("uvicorn.error")


def _truthy(v: Optional[str]) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "y", "on"}


_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+", re.UNICODE)


def _normalize_ru(s: str) -> str:
    # ё→е, lower, trim
    return (s or "").replace("ё", "е").replace("Ё", "Е").strip().lower()


def _make_prefix_query(q: str, *, min_prefix: int = 4) -> str:
    """
    Делает "потол* провод*" из "потолок проводка".
    Это НЕ морфология, но очень практичный fallback для RU:
    потолок ~ потолочное, проводка ~ провода/проводов.
    """
    qn = _normalize_ru(q)
    tokens = _WORD_RE.findall(qn)
    out: List[str] = []
    for t in tokens:
        if len(t) >= min_prefix:
            out.append(t[:min_prefix] + "*")
        else:
            out.append(t)
    return " ".join(out)


class ElasticStore:
    """Мини-клиент Elasticsearch для parent_chunks."""

    def __init__(self, url: str | None = None, index: str | None = None) -> None:
        self.url = (url or os.getenv("ELASTIC_URL", "http://localhost:9200")).rstrip("/")
        self.index = index or os.getenv("ELASTIC_INDEX", "parent_chunks")

        self._wait_seconds = int(os.getenv("ELASTIC_WAIT_SECONDS", "30"))
        self._timeout = float(os.getenv("ELASTIC_TIMEOUT", "30"))

        self._client = httpx.Client(
            base_url=self.url,
            timeout=self._timeout,
            headers={"Accept": "application/json"},
        )

    # ----------------------------
    # infra
    # ----------------------------
    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def _wait_ready(self) -> None:
        """Подождать, пока ES начнёт отвечать."""
        deadline = time.time() + max(0, self._wait_seconds)
        last_err: Exception | None = None

        while time.time() <= deadline:
            try:
                r = self._client.get("/")
                # 200 (ok) / 401 (security) / 403 / 404 — это уже "жив"
                if 200 <= r.status_code < 500:
                    return
            except Exception as e:
                last_err = e
            time.sleep(0.5)

        if last_err:
            log.warning("[elastic] not_ready url=%s err=%r", self.url, last_err)
        else:
            log.warning("[elastic] not_ready url=%s", self.url)

    def _index_exists(self) -> bool:
        r = self._client.head(f"/{self.index}")
        return r.status_code == 200

    # ----------------------------
    # index init
    # ----------------------------
    def ensure_index(self, recreate: bool = False) -> None:
        """Создать индекс с ru-анализатором, если его ещё нет."""
        self._wait_ready()

        legacy = _truthy(os.getenv("ELASTIC_LEGACY", "0"))
        log.info("[elastic] ensure_index index=%s recreate=%s legacy=%s", self.index, recreate, legacy)

        try:
            if self._index_exists():
                if not recreate:
                    return
                self._client.delete(f"/{self.index}")

            if legacy:
                body: Dict[str, Any] = {
                    "mappings": {
                        "properties": {
                            "parent_id": {"type": "keyword"},
                            "doc_id": {"type": "keyword"},
                            "part_no": {"type": "integer"},
                            "title": {"type": "text"},
                            "section_path": {"type": "keyword"},
                            "anchor": {"type": "keyword"},
                            "content": {"type": "text"},
                        }
                    }
                }
            else:
                body = {
                    "settings": {
                        "analysis": {
                            "char_filter": {
                                "ru_yo": {"type": "mapping", "mappings": ["ё=>е", "Ё=>Е"]}
                            },
                            "filter": {
                                "ru_stop": {"type": "stop", "stopwords": "_russian_"},
                                "ru_stemmer": {"type": "stemmer", "language": "russian"},
                            },
                            "analyzer": {
                                "ru_text": {
                                    "type": "custom",
                                    "tokenizer": "standard",
                                    "char_filter": ["ru_yo"],
                                    "filter": ["lowercase", "ru_stop", "ru_stemmer"],
                                }
                            },
                            "normalizer": {
                                "kw_lower": {
                                    "type": "custom",
                                    "filter": ["lowercase"]
                                }
                            },
                        }
                    },
                    "mappings": {
                        "dynamic": "strict",
                        "properties": {
                            "parent_id": {"type": "keyword"},
                            "doc_id": {"type": "keyword"},
                            "part_no": {"type": "integer"},
                            "title": {
                                "type": "text",
                                "analyzer": "ru_text",
                                "fields": {
                                    "raw": {
                                        "type": "keyword",
                                        "ignore_above": 256,
                                        "normalizer": "kw_lower",
                                    }
                                },
                            },
                            "section_path": {"type": "keyword"},
                            "anchor": {"type": "keyword"},
                            "content": {"type": "text", "analyzer": "ru_text"},
                        }
                    },
                }

            r = self._client.put(
                f"/{self.index}",
                content=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            r.raise_for_status()
            log.info("[elastic] index_created index=%s url=%s", self.index, self.url)

        except Exception as e:
            log.exception("[elastic] ensure_index_failed index=%s url=%s err=%r", self.index, self.url, e)

    # ----------------------------
    # data ops
    # ----------------------------
    def delete_by_doc_id(self, doc_id: str) -> int:
        body = {"query": {"term": {"doc_id": doc_id}}}
        r = self._client.post(
            f"/{self.index}/_delete_by_query?conflicts=proceed&refresh=false",
            content=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        r.raise_for_status()
        return int((r.json() or {}).get("deleted", 0) or 0)

    def bulk_index_parents(self, parents: List[Dict[str, Any]], refresh: bool = False) -> int:
        """Bulk-индексация parent chunks. Возвращает число успешно проиндексированных документов."""
        if not parents:
            return 0

        lines: List[str] = []
        for p in parents:
            pid = str(p.get("parent_id", "")).strip()
            if not pid:
                continue
            lines.append(json.dumps({"index": {"_index": self.index, "_id": pid}}, ensure_ascii=False))
            lines.append(json.dumps(p, ensure_ascii=False))

        payload = ("\n".join(lines) + "\n").encode("utf-8")
        params = "?refresh=wait_for" if refresh else ""

        r = self._client.post(
            f"/_bulk{params}",
            content=payload,
            headers={"Content-Type": "application/x-ndjson; charset=utf-8"},
        )
        r.raise_for_status()
        data = r.json()

        items = data.get("items") or []
        ok = 0
        err_samples: List[Dict[str, Any]] = []
        for it in items:
            op = it.get("index") or it.get("create") or it.get("update") or it.get("delete") or {}
            status = int(op.get("status") or 0)
            if 200 <= status < 300:
                ok += 1
            else:
                if len(err_samples) < 5:
                    err_samples.append(
                        {
                            "id": op.get("_id"),
                            "status": status,
                            "error": (op.get("error") or {}).get("reason") or op.get("error"),
                        }
                    )

        if data.get("errors"):
            log.warning("[elastic] bulk_errors index=%s samples=%s", self.index, err_samples)

        return ok

    # ----------------------------
    # search
    # ----------------------------
    def search(self, query: str, top_k: int = 50, highlight: bool = False) -> List[Dict[str, Any]]:
        q = _normalize_ru(query)
        if not q:
            return []

        min_should_match = os.getenv("ELASTIC_MIN_SHOULD_MATCH", "60%")
        min_prefix = int(os.getenv("ELASTIC_PREFIX_LEN", "4"))

        # 1) строгий multi_match
        # 2) мягкий multi_match
        # 3) prefix fallback (потол* провод*)
        attempts: List[Dict[str, Any]] = []

        attempts.append(
            {
                "size": int(top_k),
                "query": {
                    "multi_match": {
                        "query": q,
                        "fields": ["title^2", "content"],
                        "type": "best_fields",
                        "operator": "and",
                    }
                },
            }
        )

        attempts.append(
            {
                "size": int(top_k),
                "query": {
                    "multi_match": {
                        "query": q,
                        "fields": ["title^2", "content"],
                        "type": "best_fields",
                        "operator": "or",
                        "minimum_should_match": min_should_match,
                    }
                },
            }
        )

        prefix_q = _make_prefix_query(q, min_prefix=min_prefix)
        attempts.append(
            {
                "size": int(top_k),
                "query": {
                    "simple_query_string": {
                        "query": prefix_q,
                        "fields": ["title^3", "content"],
                        "default_operator": "and",
                        "analyze_wildcard": True,
                        # PREFIX даёт работать '*' нормально, остальное не нужно
                        "flags": "AND|OR|PHRASE|PREFIX",
                    }
                },
            }
        )

        for body in attempts:
            if highlight:
                body["highlight"] = {
                    "pre_tags": ["<<<"],
                    "post_tags": [">>>"],
                    "fields": {"content": {"number_of_fragments": 2, "fragment_size": 180}},
                }

            r = self._client.post(
                f"/{self.index}/_search",
                content=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            r.raise_for_status()

            hits = (r.json() or {}).get("hits", {}).get("hits", []) or []
            if not hits:
                continue

            out: List[Dict[str, Any]] = []
            for h in hits:
                src = h.get("_source") or {}
                out.append(
                    {
                        "parent_id": (src.get("parent_id") or h.get("_id") or ""),
                        "score": float(h.get("_score") or 0.0),
                        "doc_id": src.get("doc_id", ""),
                        "part_no": int(src.get("part_no") or 1),
                        "title": src.get("title", ""),
                        "anchor": src.get("anchor", ""),
                        "section_path": src.get("section_path", ""),
                        "highlight": (h.get("highlight") or {}).get("content", []),
                    }
                )

            if out:
                return out

        return []
