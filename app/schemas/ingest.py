from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    folder_path: str = Field(..., description="Путь к папке внутри контейнера (например /app/data/kb)")
    recursive: bool = Field(True, description="Искать файлы рекурсивно")
    extensions: List[str] = Field(default_factory=lambda: [".pdf", ".docx", ".txt"], description="Какие расширения индексировать")


class IngestResponse(BaseModel):
    indexed_files: int
    parents: int
    children: int
    skipped_files: int = 0
    errors: List[str] = Field(default_factory=list)
