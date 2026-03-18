# DocIntel — Document Intelligence Platform

A production-ready RAG (Retrieval-Augmented Generation) system for intelligent document search, Q&A, and synthetic dataset generation.

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Pydantic v2 |
| Vector Store | ChromaDB (cosine similarity) |
| Relational DB | SQLite + SQLAlchemy ORM + Alembic |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o` |
| Frontend | Gradio |
| Workflow | LangGraph |

## Project Structure

```
DocIntel/
├── .env                      # API keys, database URL, LLM config
├── .gitignore
├── requirements.txt
├── README.md
├── venv/                     # Python virtual environment
│
├── api/                      # FastAPI Backend
│   ├── main.py               # REST API entry point
│   ├── config.py             # Configuration settings
│   ├── models/
│   │   └── schemas.py        # Pydantic models
│   ├── routes/
│   │   ├── documents.py      # Document ingestion endpoints
│   │   ├── search.py         # Search/RAG endpoints
│   │   └── chat.py           # Chat endpoints
│   ├── core/
│   │   ├── config.py         # API keys, LLM config
│   │   ├── processor.py      # Document processing
│   │   ├── embeddings.py     # Vector embeddings
│   │   └── workflow.py       # LangGraph workflow
│   ├── utils/
│   │   └── prompts.py        # LLM prompts
│   └── database/
│       ├── connection.py     # DB connection
│       └── models.py         # SQLAlchemy models
│
├── frontend/                 # Gradio Web Interface
│   ├── app.py                # Gradio UI entry point
│   ├── components/
│   │   ├── ingest.py         # Document upload component
│   │   ├── search.py         # Search component
│   │   └── chat.py           # Chat component
│   └── assets/
│       └── style.css         # Custom styling
│
├── database/                 # Database Layer
│   ├── init.sql              # Database schema
│   └── migrations/
│       └── init_tables.sql   # Table creation SQL
│
└── tests/
    ├── test_documents.py
    ├── test_search.py
    ├── test_chat.py
    └── test_document.txt     # Sample test document
```

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env .env.backup    # Already filled in
# Edit .env → set OPENAI_API_KEY

# 4. Run database migrations
alembic upgrade head

# 5. Start API server
uvicorn api.main:app --reload --port 8000

# 6. Start Gradio frontend (new terminal)
python frontend/app.py
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/documents/ingest/text` | Ingest raw text |
| `POST` | `/documents/ingest/file` | Upload .txt file |
| `GET` | `/documents/` | List all documents |
| `DELETE` | `/documents/{id}` | Delete document |
| `POST` | `/search/` | Semantic search |
| `POST` | `/chat/` | RAG chat with GPT-4o |
| `GET` | `/chat/history` | Chat history |
| `POST` | `/chat/synthetic/generate` | Generate QA pairs |
| `GET` | `/stats/health` | Health check |
