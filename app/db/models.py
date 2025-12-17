from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Text, Integer, ForeignKey


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    doc_number: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    doc_date: Mapped[str] = mapped_column(String(64), default="", nullable=False)

    parents = relationship("ParentChunk", back_populates="document")


class ParentChunk(Base):
    __tablename__ = "parent_chunks"

    parent_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.doc_id"), index=True, nullable=False)

    section_path: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    anchor: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    title: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    part_no: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    content: Mapped[str] = mapped_column(Text, nullable=False)

    document = relationship("Document", back_populates="parents")
    children = relationship("ChildChunk", back_populates="parent", cascade="all, delete-orphan")


class ChildChunk(Base):
    __tablename__ = "child_chunks"

    child_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    parent_id: Mapped[str] = mapped_column(ForeignKey("parent_chunks.parent_id"), index=True, nullable=False)
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.doc_id"), index=True, nullable=False)

    child_no: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    start_char: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    end_char: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # для удобства поиска/дебага можно продублировать “шапку”
    section_path: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    anchor: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    title: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    parent_part_no: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    content: Mapped[str] = mapped_column(Text, nullable=False)

    parent = relationship("ParentChunk", back_populates="children")
