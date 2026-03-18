"""
api/database/models.py — SQLAlchemy ORM models for DocIntel.
Tables: documents, chunks, query_logs, synthetic_datasets, feedback
"""
from __future__ import annotations
from datetime import datetime, UTC
from sqlalchemy import (
    Column, String, Integer, Float, Boolean,
    Text, DateTime, JSON, ForeignKey, Index, event,
)
from sqlalchemy.orm import DeclarativeBase, relationship
from api.database.connection import engine


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id           = Column(String(36), primary_key=True)
    filename     = Column(String(512), nullable=False)
    source_type  = Column(String(64), nullable=False, default="text")
    content_hash = Column(String(64), unique=True, nullable=False)
    total_chars  = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    is_embedded  = Column(Boolean, default=False)
    created_at   = Column(DateTime, default=lambda: datetime.now(UTC))
    updated_at   = Column(DateTime, default=lambda: datetime.now(UTC),
                          onupdate=lambda: datetime.now(UTC))
    meta_info    = Column(JSON, default=dict)

    chunks         = relationship("Chunk",            back_populates="document",
                                  cascade="all, delete-orphan")
    query_logs     = relationship("QueryLog",         back_populates="document")
    synthetic_data = relationship("SyntheticDataset", back_populates="document",
                                  cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_documents_created_at",  "created_at"),
        Index("ix_documents_source_type", "source_type"),
        Index("ix_documents_is_embedded", "is_embedded"),
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id               = Column(String(36), primary_key=True)
    document_id      = Column(String(36), ForeignKey("documents.id"), nullable=False)
    chunk_index      = Column(Integer, nullable=False)
    content          = Column(Text, nullable=False)
    char_count       = Column(Integer, default=0)
    token_estimate   = Column(Integer, default=0)
    chroma_vector_id = Column(String(36), unique=True, nullable=True)
    created_at       = Column(DateTime, default=lambda: datetime.now(UTC))

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_chunk_index", "chunk_index"),
    )


class QueryLog(Base):
    __tablename__ = "query_logs"

    id                  = Column(String(36), primary_key=True)
    document_id         = Column(String(36), ForeignKey("documents.id"), nullable=True)
    query_text          = Column(Text, nullable=False)
    query_type          = Column(String(32), default="search")
    top_k               = Column(Integer, default=5)
    retrieved_chunk_ids = Column(JSON, default=list)
    llm_response        = Column(Text, nullable=True)
    latency_ms          = Column(Float, nullable=True)
    tokens_used         = Column(Integer, nullable=True)
    user_session_id     = Column(String(64), nullable=True)
    created_at          = Column(DateTime, default=lambda: datetime.now(UTC))

    document = relationship("Document", back_populates="query_logs")
    feedback = relationship("Feedback", back_populates="query_log",
                            cascade="all, delete-orphan", uselist=False)

    __table_args__ = (
        Index("ix_query_logs_created_at",   "created_at"),
        Index("ix_query_logs_query_type",   "query_type"),
        Index("ix_query_logs_user_session", "user_session_id"),
    )


class SyntheticDataset(Base):
    __tablename__ = "synthetic_datasets"

    id              = Column(String(36), primary_key=True)
    document_id     = Column(String(36), ForeignKey("documents.id"), nullable=False)
    question        = Column(Text, nullable=False)
    answer          = Column(Text, nullable=False)
    source_chunk_id = Column(String(36), nullable=True)
    created_at      = Column(DateTime, default=lambda: datetime.now(UTC))

    document = relationship("Document", back_populates="synthetic_data")

    __table_args__ = (Index("ix_synthetic_document_id", "document_id"),)


class Feedback(Base):
    __tablename__ = "feedback"

    id           = Column(String(36), primary_key=True)
    query_log_id = Column(String(36), ForeignKey("query_logs.id"), nullable=False)
    rating       = Column(Integer, nullable=False)   # 1 = thumbs up, -1 = thumbs down
    comment      = Column(Text, nullable=True)
    created_at   = Column(DateTime, default=lambda: datetime.now(UTC))

    query_log = relationship("QueryLog", back_populates="feedback")

    __table_args__ = (
        Index("ix_feedback_query_log_id", "query_log_id"),
        Index("ix_feedback_rating",       "rating"),
    )


def init_db() -> None:
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
