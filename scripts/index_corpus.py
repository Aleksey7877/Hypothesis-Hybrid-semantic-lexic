import os
from glob import glob
from pathlib import Path

from app.core.config import settings
from app.services.ingestion.docx_parser import read_docx
from app.services.ingestion.structure_parser import parse_blocks
from app.services.ingestion.chunker import chunk_blocks
from app.services.retrieval.retriever import InMemoryIndex, ChildDoc, ParentDoc
from app.utils.ids import stable_id

def build_index(kb_path: str) -> InMemoryIndex:
    kb = Path(kb_path)
    files = sorted(glob(str(kb / "*.docx")))
    parents_map = {}
    children_docs = []

    for fp in files:
        doc_name = Path(fp).stem
        doc_id = stable_id(doc_name)
        paragraphs = read_docx(fp)
        blocks = parse_blocks(paragraphs)
        parents, children = chunk_blocks(blocks, doc_id=doc_id, parent_max_chars=settings.parent_max_chars, child_max_chars=settings.child_max_chars)

        for p in parents:
            parents_map[p.parent_id] = ParentDoc(parent_id=p.parent_id, doc_id=p.doc_id, anchor=p.anchor, title=p.title, text=p.text)

        for c in children:
            children_docs.append(ChildDoc(child_id=c.child_id, parent_id=c.parent_id, doc_id=c.doc_id, anchor=c.anchor, text=c.text))

    return InMemoryIndex(children=children_docs, parents=parents_map)

if __name__ == "__main__":
    index = build_index(settings.kb_path)
    print(f"Indexed children={len(index.children)} parents={len(index.parents)} from {settings.kb_path}")
