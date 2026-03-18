"""
frontend/components/chat.py — RAG chat Gradio component.
Full pipeline: embed → retrieve → GPT-4o → display answer with sources.
"""
import httpx
import gradio as gr

API_BASE = "http://localhost:8000"


def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return ""
    lines = ["\n\n---\n### 📚 Sources\n"]
    for i, src in enumerate(sources, 1):
        score = src.get("similarity_score", 0)
        bar = "🟩" * round(score * 5) + "⬜" * (5 - round(score * 5))
        lines.append(
            f"**[{i}] `{src.get('filename', '?')}`** — chunk #{src.get('chunk_index', '?')} "
            f"— {bar} `{score:.2f}`\n\n"
            f"> {src.get('content', '')[:300]}{'…' if len(src.get('content', '')) > 300 else ''}\n"
        )
    return "\n".join(lines)


def chat(
    user_message: str,
    history: list,
    top_k: int,
    doc_filter: str,
    system_prompt: str,
) -> tuple[list, str]:
    """Send a RAG chat query and return updated history."""
    if not user_message.strip():
        return history, ""
    try:
        payload = {
            "query": user_message.strip(),
            "top_k": int(top_k),
            "document_id": doc_filter.strip() if doc_filter.strip() else None,
            "system_prompt": system_prompt.strip() if system_prompt.strip() else None,
        }
        r = httpx.post(f"{API_BASE}/chat/", json=payload, timeout=90)
        r.raise_for_status()
        data = r.json()

        answer = data.get("answer", "No answer generated.")
        sources = data.get("sources", [])
        tokens = data.get("tokens_used", 0)
        latency = data.get("latency_ms", 0)
        log_id = data.get("query_log_id", "")

        response_md = (
            f"{answer}"
            f"{_format_sources(sources)}\n\n"
            f"---\n_⏱ {latency:.0f}ms · 🔢 {tokens} tokens · log `{log_id[:8]}…`_"
        )
        history.append((user_message, response_md))
    except Exception as e:
        history.append((user_message, f"❌ Error: {e}"))

    return history, ""


def build_chat_tab() -> gr.Tab:
    with gr.Tab("💬 Chat") as tab:
        gr.Markdown("## RAG Chat\nAsk questions about your documents — GPT-4o answers grounded in your data.")

        with gr.Row():
            top_k_slider = gr.Slider(minimum=1, maximum=15, value=5, step=1, label="Top-K chunks", scale=1)
            doc_filter = gr.Textbox(label="Filter by Document ID (optional)", scale=2)

        system_prompt = gr.Textbox(
            label="Custom system prompt (optional)",
            placeholder="Leave blank to use the default DocIntel prompt",
            lines=2,
        )

        chatbot = gr.Chatbot(height=460, label="DocIntel Chat")

        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Ask anything about your documents…",
                label="Your question",
                scale=5,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        clear_btn = gr.Button("🗑️ Clear chat")

        send_btn.click(
            chat,
            inputs=[msg_input, chatbot, top_k_slider, doc_filter, system_prompt],
            outputs=[chatbot, msg_input],
        )
        msg_input.submit(
            chat,
            inputs=[msg_input, chatbot, top_k_slider, doc_filter, system_prompt],
            outputs=[chatbot, msg_input],
        )
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_input])

    return tab
