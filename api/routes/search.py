"""
api/routes/search.py — Semantic search endpoints.

POST /search/         — Top-K vector similarity search
GET  /search/history  — recent search query logs
"""
import uuid
import time
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.database.connection import get_db, query_vectors
from api.database.models import QueryLog
from api.models.schemas import SearchRequest, SearchResponse, SearchResult
from api.core.embeddings import embed_query

router = APIRouter(prefix="/search", tags=["Search"])


async def _semantic_search(
    query: str,
    db: Session,
    top_k: int = 5,
    document_id: str | None = None,
) -> list[SearchResult]:
    """Embed query → ChromaDB Top-K → return ranked SearchResult list."""
    query_vector = await embed_query(query)
    where_filter = {"document_id": document_id} if document_id else None
    raw = query_vectors(query_embedding=query_vector, top_k=top_k, where=where_filter)

    ids       = raw.get("ids", [[]])[0]
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    results: list[SearchResult] = []
    for idx, (chunk_id, content, meta, distance) in enumerate(
        zip(ids, documents, metadatas, distances)
    ):
        results.append(SearchResult(
            chunk_id=chunk_id,
            document_id=meta.get("document_id", ""),
            filename=meta.get("filename", "unknown"),
            content=content,
            similarity_score=round(1.0 - distance, 4),
            chunk_index=meta.get("chunk_index", idx),
        ))
    return results


@router.post("/", response_model=SearchResponse, summary="Top-K semantic search")
async def search(request: SearchRequest, db: Session = Depends(get_db)):
    """
    Embed the query, search ChromaDB for the most similar chunks,
    and return ranked results with similarity scores.
    """
    t0 = time.perf_counter()
    results = await _semantic_search(
        query=request.query,
        db=db,
        top_k=request.top_k,
        document_id=request.document_id,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    log = QueryLog(
        id=str(uuid.uuid4()),
        query_text=request.query,
        query_type="search",
        top_k=request.top_k,
        document_id=request.document_id,
        retrieved_chunk_ids=[r.chunk_id for r in results],
        latency_ms=latency_ms,
    )
    db.add(log)
    db.commit()

    return SearchResponse(query=request.query, results=results, total_found=len(results))


@router.get("/history", summary="Recent search history")
def search_history(limit: int = 20, db: Session = Depends(get_db)):
    logs = (
        db.query(QueryLog)
        .filter(QueryLog.query_type == "search")
        .order_by(QueryLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": log.id,
            "query": log.query_text,
            "top_k": log.top_k,
            "results_count": len(log.retrieved_chunk_ids or []),
            "latency_ms": log.latency_ms,
            "created_at": log.created_at,
        }
        for log in logs
    ]
