"""
api/models/schemas.py — Pydantic v2 schemas for request validation
and response serialization across all DocIntel API endpoints.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator


# ── Documents ─────────────────────────────────────────────────────────────────

class IngestTextRequest(BaseModel):
    content: str = Field(..., min_length=10, description="Raw text content to ingest")
    filename: str = Field(..., description="Logical name for this document")
    meta_info: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    total_chars: int
    total_chunks: int
    is_embedded: bool
    message: str


class DocumentOut(BaseModel):
    id: str
    filename: str
    source_type: str
    total_chars: int
    total_chunks: int
    is_embedded: bool
    created_at: datetime
    meta_info: dict[str, Any]

    class Config:
        from_attributes = True


# ── Search ────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="User query for semantic search")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to retrieve")
    document_id: Optional[str] = Field(None, description="Restrict search to one document")


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    content: str
    similarity_score: float
    chunk_index: int


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_found: int


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=3, description="User question")
    top_k: int = Field(5, ge=1, le=15)
    document_id: Optional[str] = None
    system_prompt: Optional[str] = Field(
        None, description="Optional custom system prompt override"
    )

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be blank")
        return v.strip()


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[SearchResult]
    tokens_used: int
    latency_ms: float
    query_log_id: str


# ── Synthetic Data ────────────────────────────────────────────────────────────

class SyntheticGenerateRequest(BaseModel):
    document_id: str
    num_pairs: int = Field(5, ge=1, le=50, description="Number of QA pairs to generate")


class SyntheticQAPair(BaseModel):
    question: str
    answer: str
    source_chunk_id: Optional[str] = None


class SyntheticGenerateResponse(BaseModel):
    document_id: str
    generated: int
    pairs: list[SyntheticQAPair]


# ── Feedback ──────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    query_log_id: str
    rating: int = Field(..., description="1 = thumbs up, -1 = thumbs down")
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: str
    rating: int
    message: str


# ── Stats ─────────────────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    total_queries: int
    vector_store: dict[str, Any]


# ── Generic ───────────────────────────────────────────────────────────────────

class MessageResponse(BaseModel):
    message: str
    detail: Optional[str] = None
