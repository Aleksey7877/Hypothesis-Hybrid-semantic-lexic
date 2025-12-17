from __future__ import annotations

from typing import Iterable, Dict, List, Sequence

from sqlalchemy import select, delete
from sqlalchemy.dialects.postgresql import insert

from app.db.postgres import SessionLocal, engine
from app.db.models import Base, Document, ParentChunk, ChildChunk


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


class Repository:
    def upsert_document(self, doc_id: str, title: str, doc_number: str = "", doc_date: str = "") -> None:
        with SessionLocal() as s:
            stmt = (
                insert(Document)
                .values(doc_id=doc_id, title=title, doc_number=doc_number, doc_date=doc_date)
                .on_conflict_do_update(
                    index_elements=[Document.doc_id],
                    set_={"title": title, "doc_number": doc_number, "doc_date": doc_date},
                )
            )
            s.execute(stmt)
            s.commit()

    def upsert_parents(self, parents: Iterable[ParentChunk]) -> int:
        rows = [
            {
                "parent_id": p.parent_id,
                "doc_id": p.doc_id,
                "section_path": p.section_path,
                "anchor": p.anchor,
                "title": p.title,
                "part_no": p.part_no,
                "content": p.content,
            }
            for p in parents
        ]
        if not rows:
            return 0

        with SessionLocal() as s:
            stmt = (
                insert(ParentChunk)
                .values(rows)
                .on_conflict_do_update(
                    index_elements=[ParentChunk.parent_id],
                    set_={
                        "doc_id": insert(ParentChunk).excluded.doc_id,
                        "section_path": insert(ParentChunk).excluded.section_path,
                        "anchor": insert(ParentChunk).excluded.anchor,
                        "title": insert(ParentChunk).excluded.title,
                        "part_no": insert(ParentChunk).excluded.part_no,
                        "content": insert(ParentChunk).excluded.content,
                    },
                )
            )
            s.execute(stmt)
            s.commit()
        return len(rows)

    def upsert_children(self, children: Iterable[ChildChunk]) -> int:
        rows = [
            {
                "child_id": c.child_id,
                "parent_id": c.parent_id,
                "doc_id": c.doc_id,
                "child_no": c.child_no,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "section_path": c.section_path,
                "anchor": c.anchor,
                "title": c.title,
                "parent_part_no": c.parent_part_no,
                "content": c.content,
            }
            for c in children
        ]
        if not rows:
            return 0

        with SessionLocal() as s:
            stmt = (
                insert(ChildChunk)
                .values(rows)
                .on_conflict_do_update(
                    index_elements=[ChildChunk.child_id],
                    set_={
                        "parent_id": insert(ChildChunk).excluded.parent_id,
                        "doc_id": insert(ChildChunk).excluded.doc_id,
                        "child_no": insert(ChildChunk).excluded.child_no,
                        "start_char": insert(ChildChunk).excluded.start_char,
                        "end_char": insert(ChildChunk).excluded.end_char,
                        "section_path": insert(ChildChunk).excluded.section_path,
                        "anchor": insert(ChildChunk).excluded.anchor,
                        "title": insert(ChildChunk).excluded.title,
                        "parent_part_no": insert(ChildChunk).excluded.parent_part_no,
                        "content": insert(ChildChunk).excluded.content,
                    },
                )
            )
            s.execute(stmt)
            s.commit()
        return len(rows)

    def get_parent_texts(self, parent_ids: List[str]) -> Dict[str, str]:
        if not parent_ids:
            return {}
        with SessionLocal() as s:
            stmt = select(ParentChunk.parent_id, ParentChunk.content).where(ParentChunk.parent_id.in_(parent_ids))
            rows = s.execute(stmt).all()
        return {pid: txt for pid, txt in rows}

    def get_parents_meta(self, parent_ids: Sequence[str]) -> Dict[str, dict]:
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
                )
                .where(ParentChunk.parent_id.in_(list(parent_ids)))
            )
            rows = s.execute(stmt).all()
        return {
            r.parent_id: {
                "parent_id": r.parent_id,
                "doc_id": r.doc_id,
                "title": r.title,
                "anchor": r.anchor,
                "section_path": r.section_path,
                "part_no": r.part_no,
            }
            for r in rows
        }

    def delete_document(self, doc_id: str) -> None:
        with SessionLocal() as s:
            s.execute(delete(ChildChunk).where(ChildChunk.doc_id == doc_id))
            s.execute(delete(ParentChunk).where(ParentChunk.doc_id == doc_id))
            s.execute(delete(Document).where(Document.doc_id == doc_id))
            s.commit()
