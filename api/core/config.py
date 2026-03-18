"""
api/core/config.py — Pydantic Settings: reads from .env file.
All API keys, database URLs, and LLM configuration live here.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o", env="OPENAI_MODEL")
    openai_embedding_model: str = Field(
        "text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL"
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field("sqlite:///./docintel.db", env="DATABASE_URL")

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_persist_dir: str = Field("./chroma_store", env="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field("docintel_documents", env="CHROMA_COLLECTION_NAME")

    # ── RAG Pipeline ──────────────────────────────────────────────────────────
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    top_k_results: int = Field(5, env="TOP_K_RESULTS")

    # ── App ───────────────────────────────────────────────────────────────────
    app_title: str = Field("DocIntel API", env="APP_TITLE")
    app_version: str = Field("1.0.0", env="APP_VERSION")
    debug: bool = Field(True, env="DEBUG")

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
