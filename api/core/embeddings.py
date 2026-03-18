"""
api/core/embeddings.py — Vector embedding generation via OpenAI's
text-embedding-3-small model. Supports single and batch calls.
"""
from openai import AsyncOpenAI
from api.core.config import get_settings

settings = get_settings()

_client: AsyncOpenAI | None = None


def get_openai_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts in a single API call (up to 2048 inputs).
    Returns a list of float vectors, one per input text.
    """
    if not texts:
        return []

    client = get_openai_client()
    response = await client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
        encoding_format="float",
    )
    sorted_data = sorted(response.data, key=lambda d: d.index)
    return [d.embedding for d in sorted_data]


async def embed_query(query: str) -> list[float]:
    """Embed a single query string and return its vector."""
    vectors = await embed_texts([query])
    return vectors[0]
