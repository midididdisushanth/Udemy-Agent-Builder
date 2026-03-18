"""
api/main.py — DocIntel FastAPI application entry point.

Architecture:
  ┌──────────────────────────────────────────────────────┐
  │  FastAPI + Swagger UI (/docs)                        │
  │  Pydantic v2 request/response validation             │
  ├──────────────┬───────────────────────────────────────┤
  │  Documents   │  /documents/ingest/text|file          │
  │              │  GET/DELETE /documents/{id}           │
  │  Search      │  POST /search/                        │
  │  Chat        │  POST /chat/                          │
  │  Synthetic   │  POST /chat/synthetic/generate        │
  │  Feedback    │  POST /chat/feedback                  │
  │  Stats       │  GET /stats/health  GET /stats/       │
  ├──────────────┴───────────────────────────────────────┤
  │  SQLite (SQLAlchemy ORM + Alembic migrations)        │
  │  ChromaDB (vector store, cosine similarity, HNSW)    │
  │  OpenAI GPT-4o + text-embedding-3-small              │
  │  LangGraph RAG workflow orchestration                │
  └──────────────────────────────────────────────────────┘
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.core.config import get_settings
from api.database.models import init_db
from api.routes import documents_router, search_router, chat_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise database tables on startup."""
    init_db()
    print(f"[startup] DocIntel ready — DB: {settings.database_url}")
    print(f"[startup] ChromaDB store — {settings.chroma_persist_dir}")
    yield
    print("[shutdown] DocIntel stopped.")


app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=(
        "DocIntel — Document Intelligence Platform. "
        "RAG-powered search and Q&A using FastAPI, ChromaDB, SQLite, and GPT-4o."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "path": str(request.url)},
    )


app.include_router(documents_router)
app.include_router(search_router)
app.include_router(chat_router)


@app.get("/", tags=["Root"])
def root():
    return {
        "service": settings.app_title,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/stats/health",
    }
