"""
api/database/connection.py — Database connection layer.
Provides SQLAlchemy engine/session for SQLite and ChromaDB vector store client.
"""
from __future__ import annotations
from typing import Optional
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
import chromadb
from chromadb.config import Settings as ChromaSettings
from api.core.config import get_settings

settings = get_settings()

# ── SQLite ─────────────────────────────────────────────────────────────────────

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
    echo=settings.debug,
)


@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, _):
    """Enable WAL mode and foreign keys for performance and integrity."""
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI dependency: yields a SQLAlchemy session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── ChromaDB ───────────────────────────────────────────────────────────────────

_chroma_client: Optional[chromadb.PersistentClient] = None
_chroma_collection = None


def get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_chroma_collection():
    """Get or create the main DocIntel collection (cosine similarity)."""
    global _chroma_collection
    if _chroma_collection is None:
        client = get_chroma_client()
        _chroma_collection = client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collection


def upsert_vectors(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    """Insert or update chunk vectors in ChromaDB."""
    collection = get_chroma_collection()
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def query_vectors(
    query_embedding: list[float],
    top_k: int = 5,
    where: Optional[dict] = None,
) -> dict:
    """Top-K cosine similarity search."""
    collection = get_chroma_collection()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )


def delete_vectors_by_doc(document_id: str) -> int:
    """Remove all vectors for a document from ChromaDB."""
    collection = get_chroma_collection()
    results = collection.get(where={"document_id": document_id}, include=["metadatas"])
    ids_to_delete = results.get("ids", [])
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)


def vector_store_stats() -> dict:
    """Return basic stats about the ChromaDB collection."""
    collection = get_chroma_collection()
    return {
        "collection_name": settings.chroma_collection_name,
        "total_vectors": collection.count(),
        "persist_dir": settings.chroma_persist_dir,
    }
