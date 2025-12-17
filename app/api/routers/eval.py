from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.routers.search import SearchRequest
from app.api.routers.search import search as search_call

router = APIRouter(prefix="/eval", tags=["eval"])


class EvalRequest(BaseModel):
    # путь внутри контейнера, например: /app/data/questions/questions.json
    questions_path: str = Field(default="/app/data/questions/questions.json")
    top_k: int = Field(default=5, ge=1, le=50)
    save_report: bool = True


@router.post("")
def run_eval(req: EvalRequest) -> Dict[str, Any]:
    """
    Где смотреть отчёт:
      - в контейнере: /app/reports/run_YYYYMMDD_HHMMSS.json
      - локально (если папка смонтирована): repo/reports/...

    Как проверить руками:
      docker exec -it aedev-api ls -la /app/reports
    """
    questions_path = req.questions_path
    if not os.path.isabs(questions_path):
        questions_path = os.path.join("/app", questions_path)

    if not os.path.exists(questions_path):
        raise HTTPException(status_code=400, detail=f"questions_path not found: {questions_path}")

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Пытаемся использовать твой runner, если он есть
    try:
        from app.services.eval.runner import EvalRunner  # если у тебя он есть

        runner = EvalRunner()
        report = runner.run(questions=questions, top_k=req.top_k)

    except Exception:
        # Fallback: тупо гоняем /search и сохраняем ответы
        report_items: List[Dict[str, Any]] = []
        for item in questions:
            q = item.get("query") or item.get("question") or ""
            if not q:
                continue
            resp = search_call(SearchRequest(query=q, top_k=req.top_k))
            report_items.append(
                {
                    "query": q,
                    "hits": [h.model_dump() for h in resp.hits],
                    "gold_doc_id": item.get("gold_doc_id"),
                    "gold_parent_id": item.get("gold_parent_id"),
                }
            )

        report = {
            "ts": datetime.utcnow().isoformat(),
            "top_k": req.top_k,
            "count": len(report_items),
            "items": report_items,
        }

    if req.save_report:
        os.makedirs("/app/reports", exist_ok=True)
        fname = datetime.utcnow().strftime("run_%Y%m%d_%H%M%S.json")
        out_path = f"/app/reports/{fname}"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        report["report_path"] = out_path

    return report
