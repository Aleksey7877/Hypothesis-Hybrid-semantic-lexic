# AE Dev RAG Retrieval (Experiment Skeleton)

This repository implements:
- DOCX parsing + structural chunking (parent/child)
- Hybrid retrieval fusion (RRF)
- Child→Parent aggregation
- FastAPI `/search` endpoint returning JSON (children + parents)

## Run
/repo
docker compose up --build -d

# После запуска:

curl -sS -X POST "http://localhost:8000/documents/ingest" \
  -H "Content-Type: application/json" \
  --data-binary '{
    "folder_path": "data",
    "extensions": [".docx"],
    "recursive": true
  }'
# После выполнения (долго, зависит от числа файлов):
python -X utf8 - <<'PY' | curl -sS "http://localhost:8000/search" \
  -H "Content-Type: application/json" --data-binary @-
import json
body={
  "query":"Как ухаживать за бетоном в жару",
  "top_k":10,
  "k_sem":10,
  "k_lex":10,
  "k_rrf":20,
  "k_rerank":20,
  "rerank_output": True,
  "debug": True
}
print(json.dumps(body, ensure_ascii=False))
PY



## Configure
By default the API runs in "in-memory" mode (no external Qdrant/Elastic).
Later you can plug real Qdrant/Elastic by implementing the wrappers in:
- `app/services/vectorstore/qdrant.py`
- `app/services/lexical/elastic.py`

## Endpoints
- `POST /search` — returns top children + top parents (RRF + aggregation)

