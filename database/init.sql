-- DocIntel Database Schema
-- SQLite with WAL mode, foreign keys, and performance indexes.
-- Run via: sqlite3 docintel.db < database/init.sql
-- OR let SQLAlchemy auto-create via: python -c "from api.database.models import init_db; init_db()"

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;

-- ── documents ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    source_type  TEXT NOT NULL DEFAULT 'text',
    content_hash TEXT UNIQUE NOT NULL,
    total_chars  INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 0,
    is_embedded  INTEGER DEFAULT 0,
    created_at   DATETIME,
    updated_at   DATETIME,
    meta_info    TEXT  -- JSON stored as text
);

CREATE INDEX IF NOT EXISTS ix_documents_created_at  ON documents(created_at);
CREATE INDEX IF NOT EXISTS ix_documents_source_type ON documents(source_type);
CREATE INDEX IF NOT EXISTS ix_documents_is_embedded ON documents(is_embedded);

-- ── chunks ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chunks (
    id               TEXT PRIMARY KEY,
    document_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index      INTEGER NOT NULL,
    content          TEXT NOT NULL,
    char_count       INTEGER DEFAULT 0,
    token_estimate   INTEGER DEFAULT 0,
    chroma_vector_id TEXT UNIQUE,
    created_at       DATETIME
);

CREATE INDEX IF NOT EXISTS ix_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS ix_chunks_chunk_index ON chunks(chunk_index);

-- ── query_logs ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS query_logs (
    id                  TEXT PRIMARY KEY,
    document_id         TEXT REFERENCES documents(id) ON DELETE SET NULL,
    query_text          TEXT NOT NULL,
    query_type          TEXT DEFAULT 'search',
    top_k               INTEGER DEFAULT 5,
    retrieved_chunk_ids TEXT,  -- JSON array
    llm_response        TEXT,
    latency_ms          REAL,
    tokens_used         INTEGER,
    user_session_id     TEXT,
    created_at          DATETIME
);

CREATE INDEX IF NOT EXISTS ix_query_logs_created_at   ON query_logs(created_at);
CREATE INDEX IF NOT EXISTS ix_query_logs_query_type   ON query_logs(query_type);
CREATE INDEX IF NOT EXISTS ix_query_logs_user_session ON query_logs(user_session_id);

-- ── synthetic_datasets ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS synthetic_datasets (
    id              TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    question        TEXT NOT NULL,
    answer          TEXT NOT NULL,
    source_chunk_id TEXT,
    created_at      DATETIME
);

CREATE INDEX IF NOT EXISTS ix_synthetic_document_id ON synthetic_datasets(document_id);

-- ── feedback ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS feedback (
    id           TEXT PRIMARY KEY,
    query_log_id TEXT NOT NULL REFERENCES query_logs(id) ON DELETE CASCADE,
    rating       INTEGER NOT NULL,   -- 1 = thumbs up, -1 = thumbs down
    comment      TEXT,
    created_at   DATETIME
);

CREATE INDEX IF NOT EXISTS ix_feedback_query_log_id ON feedback(query_log_id);
CREATE INDEX IF NOT EXISTS ix_feedback_rating       ON feedback(rating);
