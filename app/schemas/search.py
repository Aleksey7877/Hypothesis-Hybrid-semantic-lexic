from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1)


class FusedChild(BaseModel):
    child_id: str
    parent_id: str
    score: float
    rank_vec: Optional[int] = None
    rank_lex: Optional[int] = None
    doc_id: Optional[str] = None
    section_path: Optional[str] = None
    text: Optional[str] = None


class ParentResult(BaseModel):
    parent_id: str
    score: float
    doc_id: Optional[str] = None
    section_path: Optional[str] = None
    supporting_children: List[str] = Field(default_factory=list)
    parent_text: Optional[str] = None


class SearchResponse(BaseModel):
    question: str
    top_children: List[FusedChild]
    top_parents: List[ParentResult]
    debug: Dict[str, str] = Field(default_factory=dict)
