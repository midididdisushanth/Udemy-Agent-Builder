"""
frontend/components/ingest.py — Document ingestion Gradio component.
Provides file upload and text paste tabs that call the DocIntel API.
"""
import httpx
import gradio as gr

API_BASE = "http://localhost:8000"


def ingest_text(filename: str, content: str) -> str:
    if not filename.strip() or not content.strip():
        return "⚠️ Please provide both a filename and content."
    try:
        r = httpx.post(f"{API_BASE}/documents/ingest/text", json={
            "content": content,
            "filename": filename if filename.endswith(".txt") else filename + ".txt",
        }, timeout=60)
        r.raise_for_status()
        data = r.json()
        return (
            f"✅ **Ingested successfully!**\n\n"
            f"- Document ID: `{data['document_id']}`\n"
            f"- Filename: `{data['filename']}`\n"
            f"- Chunks created: **{data['total_chunks']}**\n"
            f"- Characters: **{data['total_chars']:,}**\n"
            f"- Embedded: {'✅' if data['is_embedded'] else '⏳'}\n\n"
            f"_{data['message']}_"
        )
    except Exception as e:
        return f"❌ Error: {e}"


def ingest_file(file) -> str:
    if file is None:
        return "⚠️ Please upload a .txt file."
    try:
        with open(file.name, "rb") as f:
            file_bytes = f.read()
        filename = file.name.split("/")[-1]
        r = httpx.post(
            f"{API_BASE}/documents/ingest/file",
            files={"file": (filename, file_bytes, "text/plain")},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        return (
            f"✅ **File ingested!**\n\n"
            f"- Document ID: `{data['document_id']}`\n"
            f"- Filename: `{data['filename']}`\n"
            f"- Chunks: **{data['total_chunks']}**\n"
            f"- Characters: **{data['total_chars']:,}**\n\n"
            f"_{data['message']}_"
        )
    except Exception as e:
        return f"❌ Error: {e}"


def list_documents() -> str:
    try:
        r = httpx.get(f"{API_BASE}/documents/", timeout=30)
        r.raise_for_status()
        docs = r.json()
        if not docs:
            return "📭 No documents ingested yet."
        lines = ["### 📂 Ingested Documents\n"]
        lines.append("| # | Filename | Chunks | Embedded | Created |")
        lines.append("|---|----------|--------|----------|---------|")
        for i, doc in enumerate(docs, 1):
            embedded = "✅" if doc.get("is_embedded") else "⏳"
            created = doc.get("created_at", "")[:10] if doc.get("created_at") else "—"
            lines.append(
                f"| {i} | `{doc['filename']}` | {doc['total_chunks']} | {embedded} | {created} |"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Error: {e}"


def build_ingest_tab() -> gr.Tab:
    with gr.Tab("📥 Ingest") as tab:
        gr.Markdown("## Upload or paste documents into DocIntel")
        with gr.Tab("File Upload"):
            file_input = gr.File(label="Upload .txt file", file_types=[".txt"])
            upload_btn = gr.Button("Ingest File", variant="primary")
            file_output = gr.Markdown()
            upload_btn.click(ingest_file, inputs=file_input, outputs=file_output)

        with gr.Tab("Paste Text"):
            name_input = gr.Textbox(label="Document name", placeholder="e.g. company_policy")
            text_input = gr.Textbox(label="Content", lines=10, placeholder="Paste document text here…")
            text_btn = gr.Button("Ingest Text", variant="primary")
            text_output = gr.Markdown()
            text_btn.click(ingest_text, inputs=[name_input, text_input], outputs=text_output)

        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh Document List")
            doc_list_output = gr.Markdown(value=list_documents())
        refresh_btn.click(list_documents, outputs=doc_list_output)

    return tab
