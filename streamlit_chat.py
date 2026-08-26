""" Streamlit Dynamic RAG Chat """
import os
import sys
import time
import re
import base64
import io
import tempfile
from typing import Generator
from dataclasses import dataclass
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
from rich.console import Console
from chat import (Chat,
                  SessionContext,
                  ChatOptions,
                  parse_args,
                  seed_from_string)
from src.chat_utils import RAGTag
from src import RenderWindow

# ── Reasoning/Thinking RE ─────────────────────────────────────────
THINK_END_RE = re.compile(
    r'</\s*(?:mm:)?(?:think|thinking|reasoning)\s*>',
    re.IGNORECASE
)
THINK_START_RE = re.compile(
    r'<\s*(?:mm:)?(?:think|thinking|reasoning)\s*>',
    re.IGNORECASE
)
IS_THINKING = False
NEVER_THINK = False
SHADOW_THINK = False

@dataclass
class StreamMetrics:
    """ Model performance object data """
    model: str = ""
    token_count: int = 0
    generation_time: float = 0.0
    turn_count: int = 0
    prompt_tokens: int = 0
    token_savings: int = 0
    pre_process_time: float = 0.0
    ttft: float = 0.0  # time to first token

console = Console(highlight=True)
current_dir = os.path.dirname(os.path.abspath(__file__))

# ── Session Init ───────────────────────────────────────────────────
@st.cache_resource
def get_chat_session() -> Chat:
    """ Instantiate Chat """
    opts = ChatOptions.from_yaml(current_dir)
    args = parse_args(sys.argv[1:], opts)
    _opts = ChatOptions.from_args(current_dir, args, opts)
    _opts.seed = seed_from_string(_opts.seed)
    session = SessionContext.from_args(console, _opts)
    return Chat(session, _opts)

def call_llm_stream(o_renderer: RenderWindow,
                    o_documents: dict,
                    o_meta: RAGTag,
                    o_metrics: StreamMetrics,
                    o_pre_worker: object) -> Generator:
    """Stream LLM response and collect performance metrics."""
    o_pre_worker.markdown('RAG/AGENT Working...')
    o_renderer.set_llm(o_meta, o_documents)
    messages = o_renderer.get_messages(o_meta, o_documents)
    o_pre_worker.markdown(f':small[Loading Model/Processing Prompt [{o_renderer.llm.model_name}]]')
    o_metrics.model = o_renderer.llm.model_name
    o_metrics.prompt_tokens = o_documents["prompt_tokens"]
    o_metrics.token_savings = o_documents["token_savings"]

    start_time = time.time()
    first_token_flag = True

    for token in o_renderer.stream_response(messages):
        o_pre_worker.empty()
        if first_token_flag:
            o_metrics.ttft = time.time() - start_time
            first_token_flag = False

        o_metrics.token_count += o_renderer.response_count(token.content)
        yield token

    o_metrics.generation_time = time.time() - start_time

def save_messages(o_renderer: RenderWindow, o_documents: dict, response: str) -> None:
    """ Save Turn """
    # pylint: disable-next=global-statement
    global NEVER_THINK, SHADOW_THINK, IS_THINKING
    IS_THINKING = False
    NEVER_THINK = False
    SHADOW_THINK = False
    o_documents['llm_response'] = response
    o_renderer.save_history(o_documents, response)

def get_history(o_chat: Chat) -> list:
    """ Return chat history """
    o_history = o_chat.session.common.load_chat()
    branch = 'assistant' if o_chat.opts.assistant_mode else o_history.get('current', 'default')
    return o_history[branch]

def is_thinking(o_chunk) -> bool:
    """ Return whether or not model is reasoning/thinking """
    # pylint: disable-next=global-statement
    global NEVER_THINK, SHADOW_THINK, IS_THINKING
    if NEVER_THINK:
        return False

    is_start_tag = bool(THINK_START_RE.search(o_chunk.content))
    is_end_tag   = bool(THINK_END_RE.search(o_chunk.content))

    if not IS_THINKING and (not o_chunk.content or is_start_tag):
        if not o_chunk.content:
            SHADOW_THINK = True
        IS_THINKING = True

    elif SHADOW_THINK and o_chunk.content:
        IS_THINKING = False
        NEVER_THINK = True
        return False

    elif not IS_THINKING:
        NEVER_THINK = True
        return False

    if IS_THINKING and is_end_tag:
        NEVER_THINK = True
        return False

    return True

# ── Attachment Handler (Streamlit-only) ───────────────────────────
# pylint: disable=redefined-outer-name   # intended behavior
def handle_attachments(documents: dict) -> dict:
    """Process attachments from session state and add to documents."""
    if "attachments" not in st.session_state or not st.session_state.attachments:
        return documents

    if "dynamic_images" not in documents:
        documents["dynamic_images"] = []
    if "dynamic_files" not in documents:
        documents["dynamic_files"] = ""

    for attachment in st.session_state.attachments:
        name     = attachment.get("name", "unknown")
        mime_type = attachment.get("type", "")
        data_bytes = attachment.get("data", b"")

        if mime_type.startswith("image/"):
            fmt = mime_type.split("/")[-1].upper()
            if fmt == "JPEG":
                fmt = "JPEG"
            elif fmt == "PNG":
                fmt = "PNG"
            else:
                ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
                fmt = ext.upper() if ext in ("jpg", "jpeg", "png") else "PNG"

            buffered = io.BytesIO(data_bytes)
            with Image.open(buffered) as img:
                img = img.convert("RGB")
                output_buf = io.BytesIO()
                img.save(output_buf, format=fmt)
                b64_data = base64.b64encode(output_buf.getvalue()).decode("utf-8")

            documents["dynamic_images"].append(b64_data)

        elif mime_type == "application/pdf":
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(data_bytes)
                temp_path = tmp.name
            try:
                loader = PyPDFLoader(temp_path)
                pages = []
                for page in loader.lazy_load():
                    pages.append(page)
                text_content = "".join(doc.page_content for doc in pages)
            finally:
                os.unlink(temp_path)

            documents["dynamic_files"] += f"\n--- {name} ---\n\n{text_content}\n\n"

        else:
            try:
                text_content = data_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text_content = data_bytes.decode("latin-1")
                # pylint: disable-next=broad-exception-caught   # Specifics being handled
                except Exception:
                    text_content = "<binary content, could not decode>"

            documents["dynamic_files"] += f"\n--- {name} ---\n\n{text_content}\n\n"

    return documents
# pylint: enable=redefined-outer-name

# ── Streamlit UI ───────────────────────────────────────────────────
chat: Chat = get_chat_session()
renderer = chat.session.renderer

st.html("""
<style>
/* ── Kill all borders and shadows on st.chat_input ───────────────── */
[data-testid="stChatInput"] {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

[data-testid="stChatInput"] > div {
    border: none !important;
    box-shadow: none !important;
}

/* Kill the focus-induced red ring */
[data-testid="stChatInput"]:focus-within,
[data-testid="stChatInput"]:focus {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

/* Target the inner input element */
[data-testid="stChatInput"] input {
    border: none !important;
    outline: none !important;
}

/* ── Chat layout ────────────────────────────────────────────────── */
[data-testid="stHorizontalBlock"] {
    gap: 0;
}

[data-testid="stChatMessageContent"] {
    border-radius: 0;
    background: transparent !important;
    box-shadow: none;
}

[data-testid="stChatMessage"] {
    padding: 2px 0;
}

/* ── Custom sticky title ────────────────────────────────────────── */
.custom-chat-title {
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    position: sticky;
    top: 0;
    background: inherit;
    z-index: 10;
}

/* ── Chat layout — role-based positioning ───────────────────────────── */
.stChatMessage.user {
    justify-content: flex-end;
}

.stChatMessage.assistant {
    justify-content: flex-start;
}

/* ── Message bubbles — tapers face each other ─────────────────────── */
.user-bubble {
    background: #1a1a2e !important;
    border-radius: 4px 18px 4px 18px; /* rounded LEFT, tapered RIGHT */
    padding: 12px 16px;
    max-width: 75%;
}

.assistant-bubble {
    background: #252525 !important;
    border-radius: 18px 4px 18px 4px; /* rounded RIGHT, tapered LEFT */
    padding: 12px 16px;
    max-width: 75%;
}
</style>
""")

if "messages" not in st.session_state:
    persistent = get_history(chat) or []
    persistent = persistent[-chat.opts.history_sessions:]
    st.session_state.messages = [
        {
            "role": m.get("role", "assistant"),
            "content": m.get("content", ""),
        }
        for m in persistent
        if isinstance(m, dict) and m.get("content")
    ]
    for msg in persistent:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "assistant")
        # Optional: map any extra fields
        st.session_state.messages.append({
            "role": role,
            "content": msg.get("content", ""),
            # "reasoning": msg.get("reasoning"),   # if I ever store it
        })

st.markdown(
    "<h5 class='custom-chat-title'>Dynamic RAG Chat</h5>",
    unsafe_allow_html=True,
)
st.set_page_config(page_title="Dynamic RAG Chat", page_icon="💬", layout="wide")
#st.title("Dynamic RAG Chat")

# ── Sidebar: Attachments + Metrics ────────────────────────────────
st.sidebar.header("📎 Attachments")
uploaded_files = st.sidebar.file_uploader(
    "Attach files",
    type=["png", "jpg", "jpeg", "pdf", "txt", "py"],
    help="Attach images, PDFs, or text files to your message",
)

if uploaded_files:
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]
    st.session_state.attachments = [
        {"name": f.name, "type": f.type, "data": f.getvalue()}
        for f in uploaded_files
    ]
else:
    st.session_state.attachments = []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        if msg.get("attachments"):
            for fdata in msg["attachments"]:
                st.markdown(f"📎 **{fdata.get('name', 'attached')}**")
        if "reasoning" in msg and msg["reasoning"]:
            with st.expander("🤭 Reasoning"):
                st.markdown(msg["reasoning"])
        st.markdown(msg["content"])

# ── Input & Streaming ─────────────────────────────────────────────
if prompt := st.chat_input("Type your message…"):

    with st.chat_message("user"):
        st.markdown(prompt)

    # Attachments for this turn
    if st.session_state.attachments:
        for f in st.session_state.attachments:
            st.markdown(f"📎 **{f['name']}**")

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "attachments": st.session_state.attachments,
    })

    documents, meta_data = chat.prepare_turn(prompt)
    documents = handle_attachments(documents)

    with st.chat_message("assistant"):
        history = get_history(chat)
        metrics = StreamMetrics()
        metrics.turn_count = len(history)

        pre_worker         = st.empty()
        thinking_indicator = st.empty()
        answer_container   = st.empty()
        reasoning_buffer   = ""

        stream = call_llm_stream(renderer, documents, meta_data, metrics, pre_worker)
        full_response = ""

        for chunk in stream:
            if is_thinking(chunk):
                reasoning_buffer += chunk.content
            else:
                full_response += chunk.content

            answer_container.markdown(full_response)

            if is_thinking(chunk):
                thinking_indicator.markdown(
                    f"🤔 **Reasoning…**\n`{metrics.model}`"
                )
            else:
                thinking_indicator.empty()

        # Expander for full reasoning block
        if reasoning_buffer:
            thinking_indicator.empty()
            with st.expander("🤭 Reasoning"):
                st.markdown(reasoning_buffer)

        # tokens per second — generation only (post-TTFT)
        gen_duration = metrics.generation_time - metrics.ttft
        tps = metrics.token_count / gen_duration if gen_duration > 0 else 0

        st.caption(
            f"⏱ TTFT: {metrics.ttft:.2f}s | "
            f"Gen: {gen_duration:.2f}s | "
            f"Tokens: {metrics.token_count} | "
            f"{tps:.1f} T/s | "
            f"`{metrics.model}`"
        )

    # Save after streaming completes
    save_messages(renderer, documents, full_response)
