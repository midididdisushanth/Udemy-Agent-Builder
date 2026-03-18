"""
frontend/components/search.py — Semantic search Gradio component.
Calls POST /search/ and renders ranked results with similarity scores.
"""
import httpx
import gradio as gr

API_BASE = "http://localhost:8000"


def run_search(query: str, top_k: int, doc_filter: str) -> str:
    if not query.strip():
        return "⚠️ Please enter a search query."
    try:
        payload = {
            "query": query.strip(),
            "top_k": int(top_k),
            "document_id": doc_filter.strip() if doc_filter.strip() else None,
        }
        r = httpx.post(f"{API_BASE}/search/", json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        if not results:
            return "🔍 No results found. Try a different query or ingest more documents."

        lines = [f"### 🔍 Results for: *\"{query}\"*\n"]
        lines.append(f"Found **{len(results)}** chunks\n")
        for i, r_ in enumerate(results, 1):
            score = r_.get("similarity_score", 0)
            bar = "🟩" * round(score * 5) + "⬜" * (5 - round(score * 5))
            lines.append(
                f"---\n**[{i}] `{r_['filename']}`** — chunk #{r_['chunk_index']} "
                f"— {bar} `{score:.3f}`\n\n"
                f"> {r_['content'][:400]}{'…' if len(r_['content']) > 400 else ''}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Error: {e}"


def build_search_tab() -> gr.Tab:
    with gr.Tab("🔍 Search") as tab:
        gr.Markdown("## Semantic document search\nFind relevant chunks by meaning, not just keywords.")
        with gr.Row():
            query_input = gr.Textbox(
                label="Search query",
                placeholder="e.g. What is retrieval-augmented generation?",
                scale=4,
            )
            top_k_slider = gr.Slider(minimum=1, maximum=15, value=5, step=1, label="Top-K", scale=1)
        doc_filter = gr.Textbox(
            label="Filter by Document ID (optional)",
            placeholder="Leave blank to search all documents",
        )
        search_btn = gr.Button("Search", variant="primary")
        results_output = gr.Markdown()
        search_btn.click(
            run_search,
            inputs=[query_input, top_k_slider, doc_filter],
            outputs=results_output,
        )
        query_input.submit(
            run_search,
            inputs=[query_input, top_k_slider, doc_filter],
            outputs=results_output,
        )

    return tab
