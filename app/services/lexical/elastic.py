from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Sequence

import httpx

log = logging.getLogger(__name__)


class ElasticStore:
    """Минимальный Elasticsearch-клиент (HTTP) под parents-index.

    Документ в индексе:
    {
      "parent_id": "...",
      "doc_id": "...",
      "part_no": 1,
      "title": "...",
      "section_path": "",
      "anchor": "",
      "content": "..."
    }
    """

    def __init__(
        self,
        url: str | None = None,
        index: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.url = (url or os.getenv("ELASTIC_URL", "http://localhost:9200")).rstrip("/")
        self.index = index or os.getenv("ELASTIC_INDEX", "parent_chunks")
        self.timeout = timeout or float(os.getenv("ELASTIC_TIMEOUT", "10"))

        user = os.getenv("ELASTIC_USER")
        pwd = os.getenv("ELASTIC_PASS")
        auth = (user, pwd) if (user and pwd) else None

        self._client = httpx.Client(base_url=self.url, timeout=self.timeout, auth=auth)

    # --------- index management ---------

    def ensure_index(self, recreate: bool = False) -> None:
        """Создаёт индекс при отсутствии (опционально — пересоздаёт)."""
        try:
            if self._index_exists():
                if recreate:
                    self._client.delete(f"/{self.index}")
                else:
                    return

            body = {
                "settings": {
                    "number_of_shards": int(os.getenv("ELASTIC_SHARDS", "1")),
                    "number_of_replicas": int(os.getenv("ELASTIC_REPLICAS", "0")),
                },
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
                },
            }
            r = self._client.put(f"/{self.index}", json=body)
            r.raise_for_status()
            log.warning("[elastic] index_created index=%s url=%s", self.index, self.url)
        except Exception as e:
            # не валим сервис: в retrieval при отсутствии эластика просто будет пусто
            log.warning("[elastic] ensure_index_failed index=%s err=%r", self.index, e)

    def _index_exists(self) -> bool:
        r = self._client.head(f"/{self.index}")
        if r.status_code == 200:
            return True
        if r.status_code == 404:
            return False
        r.raise_for_status()
        return False

    # --------- write ---------

    def bulk_index_parents(self, parents: Sequence[Any], refresh: bool = False) -> int:
        """Bulk-index parents. Принимает список ParentChunk (или dict) и возвращает число успешно записанных."""
        docs: List[Dict[str, Any]] = []
        for p in parents:
            if isinstance(p, dict):
                d = p
            else:
                d = {
                    "parent_id": getattr(p, "parent_id", None),
                    "doc_id": getattr(p, "doc_id", None),
                    "part_no": getattr(p, "part_no", None),
                    "title": getattr(p, "title", None),
                    "section_path": getattr(p, "section_path", "") or "",
                    "anchor": getattr(p, "anchor", "") or "",
                    "content": getattr(p, "content", "") or "",
                }
            if not d.get("parent_id"):
                continue
            docs.append(d)

        if not docs:
            return 0

        lines: List[str] = []
        for d in docs:
            pid = str(d["parent_id"])
            action = {"index": {"_index": self.index, "_id": pid}}
            lines.append(_json(action))
            lines.append(_json(d))

        payload = "\n".join(lines) + "\n"
        params: Dict[str, Any] = {}
        if refresh:
            params["refresh"] = "true"

        r = self._client.post(
            "/_bulk",
            content=payload,
            params=params,
            headers={"Content-Type": "application/x-ndjson"},
        )
        r.raise_for_status()
        data = r.json()

        ok = 0
        for item in data.get("items", []):
            res = item.get("index") or item.get("create") or item.get("update") or {}
            status = int(res.get("status", 0) or 0)
            if 200 <= status < 300:
                ok += 1

        if data.get("errors"):
            log.warning("[elastic] bulk_partial_errors index=%s ok=%d total=%d", self.index, ok, len(docs))

        return ok

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Удаляет все parents по doc_id (чтобы не копились старые версии)."""
        if not doc_id:
            return 0
        try:
            if not self._index_exists():
                return 0
            body = {"query": {"term": {"doc_id": doc_id}}}
            params = {"conflicts": "proceed"}
            r = self._client.post(f"/{self.index}/_delete_by_query", json=body, params=params)
            if r.status_code == 404:
                return 0
            r.raise_for_status()
            data = r.json()
            return int(data.get("deleted", 0) or 0)
        except Exception as e:
            log.warning("[elastic] delete_by_doc_id_failed doc_id=%s err=%r", doc_id, e)
            return 0

    # --------- read ---------

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """BM25 по parent chunks. Возвращает hits: {id, score, payload}."""
        if not query:
            return []
        try:
            if not self._index_exists():
                return []

            body = {
                "size": int(top_k),
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["content^3", "title"],
                        "type": "best_fields",
                        "operator": "and",
                    }
                },
                "_source": ["parent_id", "doc_id", "part_no", "title", "section_path", "anchor"],
            }
            r = self._client.post(f"/{self.index}/_search", json=body)
            r.raise_for_status()
            data = r.json()

            hits: List[Dict[str, Any]] = []
            for h in ((data.get("hits") or {}).get("hits", []) or []):
                src = h.get("_source") or {}
                pid = src.get("parent_id") or h.get("_id")
                if not pid:
                    continue
                hits.append(
                    {
                        "id": str(pid),
                        "score": float(h.get("_score") or 0.0),
                        "payload": {
                            "parent_id": str(pid),
                            "doc_id": src.get("doc_id", ""),
                            "part_no": src.get("part_no", 0),
                            "title": src.get("title", ""),
                            "section_path": src.get("section_path", ""),
                            "anchor": src.get("anchor", ""),
                        },
                    }
                )
            return hits
        except Exception as e:
            log.warning("[elastic] search_failed err=%r", e)
            return []

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


def _json(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
