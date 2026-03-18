"""
api/routes/chat.py — RAG chat and synthetic data generation endpoints.

POST /chat/                   — full RAG pipeline (retrieve + GPT-4o)
GET  /chat/history            — chat history with LLM responses
POST /chat/synthetic/generate — auto-generate QA pairs from a document
GET  /chat/synthetic/{doc_id} — list QA pairs for a document
POST /chat/feedback           — submit thumbs up/down on an answer
GET  /chat/feedback/summary   — aggregate feedback counts
GET  /stats/health            — health check
GET  /stats/                  — system statistics
"""
import uuid
import json
import time
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from api.database.connection import get_db, vector_store_stats
from api.database.models import QueryLog, Document, Chunk, SyntheticDataset, Feedback
from api.models.schemas import (
    ChatRequest, ChatResponse,
    SyntheticGenerateRequest, SyntheticGenerateResponse, SyntheticQAPair,
    FeedbackRequest, FeedbackResponse,
    StatsResponse,
)
from api.core.embeddings import embed_query, get_openai_client
from api.core.workflow import run_rag_workflow
from api.utils.prompts import SYNTHETIC_QA_PROMPT
from api.routes.search import _semantic_search

router = APIRouter(tags=["Chat"])


# ── RAG Chat ──────────────────────────────────────────────────────────────────

@router.post("/chat/", response_model=ChatResponse, summary="RAG chat — grounded answer")
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """Full RAG pipeline: embed → retrieve → augment → GPT-4o → log."""
    t0 = time.perf_counter()

    sources = await _semantic_search(
        query=request.query,
        db=db,
        top_k=request.top_k,
        document_id=request.document_id,
    )
    answer, tokens_used = await run_rag_workflow(
        query=request.query,
        sources=sources,
        system_prompt=request.system_prompt,
    )

    latency_ms = (time.perf_counter() - t0) * 1000
    log_id = str(uuid.uuid4())

    log = QueryLog(
        id=log_id,
        query_text=request.query,
        query_type="chat",
        top_k=request.top_k,
        document_id=request.document_id,
        retrieved_chunk_ids=[s.chunk_id for s in sources],
        llm_response=answer,
        latency_ms=latency_ms,
        tokens_used=tokens_used,
    )
    db.add(log)
    db.commit()

    return ChatResponse(
        query=request.query,
        answer=answer,
        sources=sources,
        tokens_used=tokens_used,
        latency_ms=round(latency_ms, 2),
        query_log_id=log_id,
    )


@router.get("/chat/history", summary="Recent chat history with LLM responses")
def chat_history(limit: int = 20, db: Session = Depends(get_db)):
    logs = (
        db.query(QueryLog)
        .filter(QueryLog.llm_response.isnot(None))
        .order_by(QueryLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": log.id,
            "query": log.query_text,
            "answer": log.llm_response,
            "tokens_used": log.tokens_used,
            "latency_ms": log.latency_ms,
            "created_at": log.created_at,
        }
        for log in logs
    ]


# ── Synthetic Data ────────────────────────────────────────────────────────────

@router.post("/chat/synthetic/generate",
             response_model=SyntheticGenerateResponse,
             summary="Generate synthetic QA pairs from a document")
async def generate_synthetic(request: SyntheticGenerateRequest, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == request.document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    client = get_openai_client()
    chunks = (
        db.query(Chunk)
        .filter(Chunk.document_id == request.document_id)
        .order_by(Chunk.chunk_index)
        .all()
    )
    if not chunks:
        return SyntheticGenerateResponse(document_id=request.document_id, generated=0, pairs=[])

    pairs_per_chunk = max(1, request.num_pairs // len(chunks))
    remainder = request.num_pairs - (pairs_per_chunk * len(chunks))
    generated: list[SyntheticQAPair] = []

    for i, chunk in enumerate(chunks):
        n = pairs_per_chunk + (1 if i < remainder else 0)
        if n == 0:
            continue
        prompt = SYNTHETIC_QA_PROMPT.format(n=n, chunk=chunk.content[:1500])
        try:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            pairs_data: list[dict] = json.loads(raw.strip())
            for pair in pairs_data:
                qa = SyntheticQAPair(
                    question=pair.get("question", ""),
                    answer=pair.get("answer", ""),
                    source_chunk_id=chunk.id,
                )
                generated.append(qa)
                db.add(SyntheticDataset(
                    id=str(uuid.uuid4()),
                    document_id=request.document_id,
                    question=qa.question,
                    answer=qa.answer,
                    source_chunk_id=chunk.id,
                ))
        except Exception:
            continue

    db.commit()
    return SyntheticGenerateResponse(
        document_id=request.document_id,
        generated=len(generated),
        pairs=generated,
    )


@router.get("/chat/synthetic/{document_id}", summary="List all QA pairs for a document")
def list_synthetic(document_id: str, db: Session = Depends(get_db)):
    records = (
        db.query(SyntheticDataset)
        .filter(SyntheticDataset.document_id == document_id)
        .all()
    )
    return [
        {
            "id": r.id,
            "question": r.question,
            "answer": r.answer,
            "source_chunk_id": r.source_chunk_id,
            "created_at": r.created_at,
        }
        for r in records
    ]


# ── Feedback ──────────────────────────────────────────────────────────────────

@router.post("/chat/feedback", response_model=FeedbackResponse, summary="Submit answer feedback")
def submit_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    if req.rating not in (1, -1):
        raise HTTPException(status_code=422, detail="Rating must be 1 or -1.")
    log = db.query(QueryLog).filter(QueryLog.id == req.query_log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Query log not found.")
    if log.feedback:
        raise HTTPException(status_code=409, detail="Feedback already submitted.")
    fb = Feedback(
        id=str(uuid.uuid4()),
        query_log_id=req.query_log_id,
        rating=req.rating,
        comment=req.comment,
    )
    db.add(fb)
    db.commit()
    return FeedbackResponse(id=fb.id, rating=fb.rating, message="Feedback recorded.")


@router.get("/chat/feedback/summary", summary="Aggregate feedback counts")
def feedback_summary(db: Session = Depends(get_db)):
    up   = db.query(func.count(Feedback.id)).filter(Feedback.rating == 1).scalar() or 0
    down = db.query(func.count(Feedback.id)).filter(Feedback.rating == -1).scalar() or 0
    return {"thumbs_up": up, "thumbs_down": down}


# ── Stats & Health ────────────────────────────────────────────────────────────

@router.get("/stats/health", summary="Health check")
def health_check():
    return {"status": "ok", "service": "DocIntel API"}


@router.get("/stats/", response_model=StatsResponse, summary="System-wide statistics")
def get_stats(db: Session = Depends(get_db)):
    return StatsResponse(
        total_documents=db.query(Document).count(),
        total_chunks=db.query(Chunk).count(),
        total_queries=db.query(QueryLog).count(),
        vector_store=vector_store_stats(),
    )
