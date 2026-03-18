"""
api/core/processor.py — Document processing pipeline.
Handles text splitting using LangChain's RecursiveCharacterTextSplitter.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from api.core.config import get_settings

settings = get_settings()


def get_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """
    Build a RecursiveCharacterTextSplitter that tries to preserve:
    1. Paragraphs  (\\n\\n)
    2. Lines       (\\n)
    3. Sentences   (. ! ?)
    4. Words       (space)
    5. Characters  (fallback)
    """
    return RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )


def split_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """
    Split raw text into semantically coherent chunks.
    Returns a list of non-empty string chunks.
    """
    splitter = get_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c.strip()]


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate: ~4 characters per token (OpenAI average).
    """
    return max(1, len(text) // 4)
