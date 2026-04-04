import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from rag_pipeline import get_merged_vector_store, build_qa_chain
import os
# -------------------------
# Setup: process one or more URLs
# -------------------------
def setup_videos(urls_text):
    """
    urls_text: newline or comma-separated YouTube URLs
    Returns: (status_message, qa_chain_state, chat_history_state)
    """
    # Parse URLs — support both newline and comma separation
    raw = urls_text.replace(",", "\n")
    urls = [u.strip() for u in raw.splitlines() if u.strip()]

    if not urls:
        return "⚠️ Please enter at least one YouTube URL.", None, []

    try:
        vector_store, errors = get_merged_vector_store(urls)
        qa_chain = build_qa_chain(vector_store)

        status_parts = [f"✅ Successfully processed {len(urls) - len(errors)} video(s)."]
        if errors:
            status_parts.append("\n".join(errors))

        return "\n".join(status_parts), qa_chain, []

    except Exception as e:
        return f"❌ Error: {e}", None, []

# -------------------------
# Chat: ask a question with history
# -------------------------
def chat(user_message, chat_history, qa_chain):
    if qa_chain is None:
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": "⚠️ Please process at least one video URL first."})
        return "", chat_history, chat_history

    if not user_message.strip():
        return "", chat_history, chat_history

    # Convert Gradio chat history (dicts) → LangChain messages
    lc_history = []
    for msg in chat_history:
        if msg["role"] == "user":
            lc_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_history.append(AIMessage(content=msg["content"]))

    try:
        answer = qa_chain.invoke({
            "question": user_message,
            "chat_history": lc_history
        })
        if not answer.strip():
            answer = "Sorry, I couldn't find a relevant answer in the video(s)."
    except Exception as e:
        answer = f"❌ Error generating answer: {e}"

    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": answer})
    return "", chat_history, chat_history

# -------------------------
# Clear chat history
# -------------------------
def clear_history():
    return [], []

# -------------------------
# Gradio UI
# -------------------------
with gr.Blocks(title="YouTube RAG Chatbot") as demo:
    gr.Markdown("# 🎥 YouTube RAG Chatbot")
    gr.Markdown("Enter one or more YouTube URLs, then ask questions about the video(s).")

    # State
    qa_chain_state = gr.State(None)
    history_state = gr.State([])

    # --- Video Setup ---
    with gr.Group():
        gr.Markdown("### 📎 Step 1: Load Video(s)")
        urls_input = gr.Textbox(
            label="YouTube URL(s)",
            placeholder="Paste one or more URLs (comma or newline separated):\nhttps://youtube.com/watch?v=abc\nhttps://youtube.com/watch?v=xyz",
            lines=3
        )
        setup_btn = gr.Button("🚀 Process Video(s)", variant="primary")
        setup_status = gr.Textbox(label="Status", interactive=False)

    # --- Chat ---
    with gr.Group():
        gr.Markdown("### 💬 Step 2: Ask Questions")
        chatbot = gr.Chatbot(label="Chat", height=400)
        with gr.Row():
            user_input = gr.Textbox(
                label="Your question",
                placeholder="Ask anything about the video(s)...",
                scale=4
            )
            ask_btn = gr.Button("Ask", variant="primary", scale=1)
        clear_btn = gr.Button("🗑️ Clear Chat History", variant="secondary")

    # --- Wiring ---
    setup_btn.click(
        fn=setup_videos,
        inputs=[urls_input],
        outputs=[setup_status, qa_chain_state, history_state]
    ).then(
        fn=lambda: ([], []),  # clear chatbot display and history on new setup
        outputs=[chatbot, history_state]
    )

    ask_btn.click(
        fn=chat,
        inputs=[user_input, history_state, qa_chain_state],
        outputs=[user_input, chatbot, history_state]
    )

    user_input.submit(
        fn=chat,
        inputs=[user_input, history_state, qa_chain_state],
        outputs=[user_input, chatbot, history_state]
    )

    clear_btn.click(
        fn=clear_history,
        outputs=[chatbot, history_state]
    )

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 10000)))