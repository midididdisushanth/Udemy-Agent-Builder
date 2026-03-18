"""
api/routes/documents.py — Document ingestion endpoints.

POST /documents/ingest/text   — ingest raw text
POST /documents/ingest/file   — upload a .txt file
GET  /documents/              — list all documents
GET  /documents/{id}          — get document by ID
DELETE /documents/{id}        — delete document + vectors
"""
import hashlib
import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from api.database.connection import get_db, upsert_vectors, delete_vectors_by_doc
from api.database.models import Document, Chunk
from api.models.schemas import IngestTextRequest, IngestResponse, DocumentOut
from api.core.processor import split_text, estimate_tokens
from api.core.embeddings import embed_texts

router = APIRouter(prefix="/documents", tags=["Documents"])


def _generate_uuid() -> str:
    return str(uuid.uuid4())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def _ingest_content(
    content: str,
    filename: str,
    source_type: str,
    meta_info: dict,
    db: Session,
) -> IngestResponse:
    """Core ingestion pipeline: hash → chunk → embed → store."""

    content_hash = _sha256(content)

    # Deduplication check
    existing = db.query(Document).filter(Document.content_hash == content_hash).first()
    if existing:
        return IngestResponse(
            document_id=existing.id,
            filename=existing.filename,
            total_chars=existing.total_chars,
            total_chunks=existing.total_chunks,
            is_embedded=existing.is_embedded,
            message="Document already ingested (duplicate detected).",
        )

    # 1 — Split text into chunks
    chunks_text = split_text(content)
    if not chunks_text:
        raise HTTPException(status_code=422, detail="Content produced zero chunks after splitting.")

    # 2 — Create SQLite document record
    doc_id = _generate_uuid()
    doc = Document(
        id=doc_id,
        filename=filename,
        source_type=source_type,
        content_hash=content_hash,
        total_chars=len(content),
        total_chunks=len(chunks_text),
        is_embedded=False,
        meta_info=meta_info,
    )
    db.add(doc)

    # 3 — Create chunk records
    chunk_ids: list[str] = []
    chunk_objects: list[Chunk] = []
    for i, chunk_text in enumerate(chunks_text):
        cid = _generate_uuid()
        chunk_ids.append(cid)
        chunk_objects.append(
            Chunk(
                id=cid,
                document_id=doc_id,
                chunk_index=i,
                content=chunk_text,
                char_count=len(chunk_text),
                token_estimate=estimate_tokens(chunk_text),
                chroma_vector_id=cid,
            )
        )
    db.add_all(chunk_objects)

    # 4 — Generate embeddings
    embeddings = await embed_texts(chunks_text)

    # 5 — Upsert into ChromaDB
    metadatas = [
        {"document_id": doc_id, "filename": filename, "chunk_index": i}
        for i in range(len(chunks_text))
    ]
    upsert_vectors(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=chunks_text,
        metadatas=metadatas,
    )

    # 6 — Mark as embedded and commit
    doc.is_embedded = True
    db.commit()

    return IngestResponse(
        document_id=doc_id,
        filename=filename,
        total_chars=len(content),
        total_chunks=len(chunks_text),
        is_embedded=True,
        message=f"Successfully ingested {len(chunks_text)} chunks.",
    )


@router.post("/ingest/text", response_model=IngestResponse, summary="Ingest raw text")
async def ingest_text(request: IngestTextRequest, db: Session = Depends(get_db)):
    return await _ingest_content(
        content=request.content,
        filename=request.filename,
        source_type="text",
        meta_info=request.meta_info,
        db=db,
    )


@router.post("/ingest/file", response_model=IngestResponse, summary="Ingest a .txt file")
async def ingest_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")
    content = (await file.read()).decode("utf-8", errors="replace")
    return await _ingest_content(
        content=content,
        filename=file.filename,
        source_type="file",
        meta_info={},
        db=db,
    )


@router.get("/", response_model=list[DocumentOut], summary="List all documents")
def list_documents(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    return db.query(Document).order_by(Document.created_at.desc()).offset(skip).limit(limit).all()


@router.get("/{document_id}", response_model=DocumentOut, summary="Get document by ID")
def get_document(document_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


@router.delete("/{document_id}", summary="Delete document and its vectors")
def delete_document(document_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    deleted_vectors = delete_vectors_by_doc(document_id)
    db.delete(doc)
    db.commit()
    return {"message": f"Document '{doc.filename}' deleted.", "vectors_removed": deleted_vectors}
