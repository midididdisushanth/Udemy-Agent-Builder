"""
frontend/app.py — DocIntel Gradio web interface entry point.

Tabs:
  💬 Chat      — RAG chat powered by GPT-4o
  🔍 Search    — Raw Top-K semantic search
  📥 Ingest    — Upload or paste documents
  📊 Dashboard — System stats and health

Usage:
    python frontend/app.py
"""
import os
import httpx
import gradio as gr

from frontend.components.ingest import build_ingest_tab
from frontend.components.search import build_search_tab
from frontend.components.chat import build_chat_tab

API_BASE = "http://localhost:8000"

# Load custom CSS
CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "style.css")
with open(CSS_PATH) as f:
    CUSTOM_CSS = f.read()


def get_stats() -> str:
    """Fetch system stats from the API and format as Markdown."""
    try:
        r = httpx.get(f"{API_BASE}/stats/", timeout=10)
        r.raise_for_status()
        s = r.json()
        vs = s.get("vector_store", {})
        return (
            "### 📊 System Stats\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Documents | **{s.get('total_documents', 0)}** |\n"
            f"| Chunks | **{s.get('total_chunks', 0)}** |\n"
            f"| Queries | **{s.get('total_queries', 0)}** |\n"
            f"| Vectors stored | **{vs.get('total_vectors', 0)}** |\n"
            f"| Collection | `{vs.get('collection_name', '—')}` |\n"
        )
    except Exception as e:
        return f"❌ Could not reach API: {e}\n\nMake sure the backend is running at `{API_BASE}`."


def build_dashboard_tab() -> gr.Tab:
    with gr.Tab("📊 Dashboard") as tab:
        gr.Markdown("## System overview")
        stats_display = gr.Markdown(value=get_stats())
        refresh_btn = gr.Button("🔄 Refresh Stats")
        refresh_btn.click(get_stats, outputs=stats_display)

        gr.Markdown("---\n### Quick links\n- API docs: [localhost:8000/docs](http://localhost:8000/docs)\n- ReDoc: [localhost:8000/redoc](http://localhost:8000/redoc)")
    return tab


# ── Build the App ──────────────────────────────────────────────────────────────

with gr.Blocks(
    title="DocIntel",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="teal",
        font=gr.themes.GoogleFont("DM Sans"),
    ),
) as demo:
    gr.Markdown(
        """
        # 🔮 DocIntel
        ### Document Intelligence Platform — RAG-powered search, Q&A, and dataset generation
        """
    )

    build_chat_tab()
    build_search_tab()
    build_ingest_tab()
    build_dashboard_tab()


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
    )
