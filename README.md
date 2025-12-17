# AE Dev RAG Retrieval (Experiment Skeleton)

This repository implements:
- DOCX parsing + structural chunking (parent/child)
- Hybrid retrieval fusion (RRF)
- Child→Parent aggregation
- FastAPI `/search` endpoint returning JSON (children + parents)

## Run
/repo
docker compose up --build -d

## Configure
By default the API runs in "in-memory" mode (no external Qdrant/Elastic).
Later you can plug real Qdrant/Elastic by implementing the wrappers in:
- `app/services/vectorstore/qdrant.py`
- `app/services/lexical/elastic.py`

## Endpoints
- `POST /search` — returns top children + top parents (RRF + aggregation)

