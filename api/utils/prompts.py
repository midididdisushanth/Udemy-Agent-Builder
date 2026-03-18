"""
api/utils/prompts.py — Centralised LLM prompt templates.
All prompts used across the DocIntel system are defined here.
"""

# ── RAG / Chat ────────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are DocIntel, a precise document intelligence assistant.
Answer the user's question using ONLY the provided context chunks.
If the answer is not in the context, say "I don't have enough information to answer that."
Always cite which part of the context you used (e.g. "According to chunk 2...").
Be concise and accurate."""

RAG_USER_TEMPLATE = """Context:
{context}

Question: {query}

Answer:"""

# ── Synthetic QA Generation ───────────────────────────────────────────────────

SYNTHETIC_QA_PROMPT = """You are a dataset generator.
Given the following text chunk, generate {n} diverse question-answer pairs.
The questions should test understanding of the chunk content.

Text chunk:
{chunk}

Respond with ONLY a JSON array:
[
  {{"question": "...", "answer": "..."}},
  ...
]
Do not include any other text."""

# ── Document Summary ──────────────────────────────────────────────────────────

SUMMARIZE_PROMPT = """Summarize the following document in 3-5 sentences.
Focus on the main topics, key findings, and most important information.

Document:
{content}

Summary:"""

# ── Search Intent Classification ──────────────────────────────────────────────

SEARCH_INTENT_PROMPT = """Classify the following user query into one of these categories:
- factual: asking for a specific fact
- conceptual: asking to explain a concept
- procedural: asking how to do something
- comparative: comparing two or more things

Query: {query}

Respond with ONLY the category name."""
