"""
api/core/workflow.py — LangGraph-based RAG workflow.
Orchestrates: embed → retrieve → augment → generate steps.
"""
from __future__ import annotations
from typing import TypedDict, Optional
from api.core.config import get_settings
from api.core.embeddings import get_openai_client
from api.models.schemas import SearchResult

settings = get_settings()

DEFAULT_SYSTEM_PROMPT = """You are DocIntel, a precise document intelligence assistant.
Answer the user's question using ONLY the provided context chunks.
If the answer is not in the context, say "I don't have enough information to answer that."
Always cite which part of the context you used (e.g. "According to chunk 2...").
Be concise and accurate."""


# ── LangGraph State ───────────────────────────────────────────────────────────

class RAGState(TypedDict):
    query: str
    sources: list[SearchResult]
    context: str
    answer: str
    tokens_used: int
    system_prompt: Optional[str]


# ── Workflow Nodes ────────────────────────────────────────────────────────────

def build_context(state: RAGState) -> RAGState:
    """Format retrieved chunks into a numbered context block."""
    lines = []
    for i, src in enumerate(state["sources"], 1):
        lines.append(
            f"[Chunk {i} | File: {src.filename} | Score: {src.similarity_score:.2f}]\n"
            f"{src.content}"
        )
    state["context"] = "\n\n---\n\n".join(lines)
    return state


async def generate_answer(state: RAGState) -> RAGState:
    """Call GPT-4o with the context and user query."""
    client = get_openai_client()
    system = state.get("system_prompt") or DEFAULT_SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"Context:\n{state['context']}\n\n"
                f"Question: {state['query']}\n\n"
                "Answer:"
            ),
        },
    ]

    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )

    state["answer"] = response.choices[0].message.content.strip()
    state["tokens_used"] = response.usage.total_tokens if response.usage else 0
    return state


async def run_rag_workflow(
    query: str,
    sources: list[SearchResult],
    system_prompt: str | None = None,
) -> tuple[str, int]:
    """
    Execute the full RAG workflow: build context → generate answer.
    Returns (answer_text, tokens_used).
    """
    state: RAGState = {
        "query": query,
        "sources": sources,
        "context": "",
        "answer": "",
        "tokens_used": 0,
        "system_prompt": system_prompt,
    }
    state = build_context(state)
    state = await generate_answer(state)
    return state["answer"], state["tokens_used"]
