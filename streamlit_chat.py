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
from src .chat_utils import RAGTag

# REASONING/THINKING RE
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
    """ model performance object data """
    model: str = ""
    token_count: int = 0
    generation_time: float = 0.0
    turn_count: int = 0
    prompt_tokens: int = 0
    token_savings: int = 0
    pre_process_time: float = 0.0

console = Console(highlight=True)
current_dir = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def get_chat_session()->Chat:
    """ Instantiate Chat """
    opts = ChatOptions.from_yaml(current_dir)
    args = parse_args(sys.argv[1:], opts)
    _opts = ChatOptions.from_args(current_dir, args, opts)
    _opts.seed = seed_from_string(_opts.seed)
    session = SessionContext.from_args(console, _opts)
    return Chat(session, _opts)

def call_llm_stream(o_renderer,
                    o_documents: dict,
                    o_meta: RAGTag,
                    o_metrics: StreamMetrics)->Generator:
    """Stream LLM response and collect performance metrics."""

    o_renderer.set_llm(o_meta, o_documents)
    messages = o_renderer.get_messages(o_meta, o_documents)

    o_metrics.model = o_renderer.llm.model_name
    o_metrics.prompt_tokens = o_documents["prompt_tokens"]
    o_metrics.token_savings = o_documents["token_savings"]

    start_time = time.time()

    for token in o_renderer.stream_response(messages):
        o_metrics.token_count += o_renderer.response_count(token.content)
        yield token

    o_metrics.generation_time = time.time() - start_time

def save_messages(o_renderer: Chat,
                  o_documents: dict,
                  response: str)->None:
    """ Save Turn """
    # pylint: disable-next=global-statement
    global NEVER_THINK, SHADOW_THINK, IS_THINKING
    IS_THINKING = False
    NEVER_THINK = False
    SHADOW_THINK = False
    o_renderer.save_response(o_documents, response)

def get_history(o_chat: Chat)->list:
    """ return chat history """
    o_history = o_chat.session.common.load_chat()
    if o_chat.opts.assistant_mode:
        branch = 'assistant'
    else:
        branch = o_history.get('current', 'default')
    return o_history[branch]

def is_thinking(o_chunk)->bool:
    """ Return whether or not model is reasoning/thinking """
    # pylint: disable-next=global-statement
    global NEVER_THINK, SHADOW_THINK, IS_THINKING
    if NEVER_THINK:
        return False

    is_start_tag = bool(THINK_START_RE.search(o_chunk.content))
    is_end_tag   = bool(THINK_END_RE.search(o_chunk.content))
    # -------- FIRST/(AND SHADOW LAST) REASON TOKEN DISCOVERY IF/ELIF
    # First chunk has reasoning content: ('' || <*ing> || <*think>)
    if not IS_THINKING and (not o_chunk.content or is_start_tag):
        if not o_chunk.content:
            SHADOW_THINK = True
        IS_THINKING = True

    # While shadow thinking, the next non-empty chunk of any kind ends thinking
    elif SHADOW_THINK and o_chunk.content:
        IS_THINKING = False
        NEVER_THINK = True
        return False

    # First token is a non-thinking token. Prevent future thinking discovery.
    elif not IS_THINKING:
        NEVER_THINK = True
        return False

    # -------- MODEL IS IN THE MIDDLE OF REASONING (we are looking for a is_end_tag)
    if IS_THINKING and is_end_tag:
        NEVER_THINK = True
        return False
    return True


# pylint: disable=redefined-outer-name   # this is desired behavior in this case
def handle_attachments(documents: dict) -> dict:
    """Process attachments from session state and add to documents.

    Modifies documents in-place: adds base64-encoded images to 'dynamic_images',
    appends extracted text from PDFs/text files to 'dynamic_files'.
    """

    if "attachments" not in st.session_state or not st.session_state.attachments:
        return documents

    # Ensure target keys exist
    if "dynamic_images" not in documents:
        documents["dynamic_images"] = []
    if "dynamic_files" not in documents:
        documents["dynamic_files"] = ""

    for attachment in st.session_state.attachments:
        name = attachment.get("name", "unknown")
        mime_type = attachment.get("type", "")
        data_bytes = attachment.get("data", b"")

        # ── Image processing ───────────────────────────────────────────
        if mime_type.startswith("image/"):
            # Infer format from MIME or filename extension
            fmt = mime_type.split("/")[-1].upper()
            if fmt == "JPEG":
                fmt = "JPEG"
            elif fmt == "PNG":
                fmt = "PNG"
            else:
                # Fallback: guess from filename
                ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
                fmt = ext.upper() if ext in ("jpg", "jpeg", "png") else "PNG"

            buffered = io.BytesIO(data_bytes)
            with Image.open(buffered) as img:
                img = img.convert("RGB")
                output_buf = io.BytesIO()
                img.save(output_buf, format=fmt)
                b64_data = base64.b64encode(output_buf.getvalue()).decode("utf-8")

            documents["dynamic_images"].append(b64_data)

        # ── PDF processing ─────────────────────────────────────────────
        elif mime_type == "application/pdf":
            # Write bytes to temp file for PyPDFLoader
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
                os.unlink(temp_path)  # clean up even on failure
            documents["dynamic_files"] += f"\n--- {name} ---\n\n{text_content}\n\n"

        # ── Text/code file processing ───────────────────────────────────
        else:
            try:
                text_content = data_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text_content = data_bytes.decode("latin-1")
                # pylint: disable-next=broad-exception-caught   # specifics being handled
                except Exception:
                    text_content = "<binary content, could not decode>"

            documents["dynamic_files"] += f"\n--- {name} ---\n\n{text_content}\n\n"

    return documents
# pylint: enable=redefined-outer-name

# ──────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────
chat: Chat = get_chat_session()
renderer = chat.session.renderer


st.html("""
<style>
[data-testid="stChatInput"] div:first-child {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}
[data-testid="stChatInput"]:focus-within {
    border: none !important;
    box-shadow: none !important;
}
/* iMessage layout: user on right, assistant on left */
[data-testid="stChatMessageContent"]:has([data-testid="stAvatar"]) {
    display: flex !important;
}

/* user messages → right side */
.user-message [data-testid="stChatMessageContent"] {
    justify-content: flex-end !important;
}

/* assistant messages → left side */
.assistant-message [data-testid="stChatMessageContent"] {
    justify-content: flex-start !important;
}
</style>
""")

st.set_page_config(page_title="LLM Chat", page_icon="💬", layout="wide")
st.title("Dynamic RAG Chat")

# ── Sidebar: Attachment Zone ──────────────────────────────────────
st.sidebar.header("Attachments")

uploaded_files = st.sidebar.file_uploader(
    "Attach files",
    type=["png", "jpg", "jpeg", "pdf", "txt", "py"],
    help="Attach images, PDFs, or text files to your message",
)

# Store attachments in session state so they persist across reruns
if uploaded_files:
    # Normalize: single file → wrap in list
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    st.session_state.attachments = [
        {"name": f.name, "type": f.type, "data": f.getvalue()}
        for f in uploaded_files
    ]
else:
    # Ensure no stale attachments
    st.session_state.attachments = []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Attachment chips
        if msg.get("attachments"):
            for fdata in msg["attachments"]:
                st.markdown(f"📎 **{fdata.get('name', 'attached')}**")
        # Reasoning expander
        if "reasoning" in msg and msg["reasoning"]:
            with st.expander("Reasoning", expanded=False):
                st.markdown(msg["reasoning"])
        # Content
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your message…"):
    with st.chat_message("user"):
        st.markdown(prompt)
    # Show attached file chips alongside the user's message
    if st.session_state.attachments:
        for f in st.session_state.attachments:  # uploaded_files still exists in this branch
            st.markdown(f"📎 **{f['name']}**")
    # Append user message WITH attachments attached
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "attachments": st.session_state.attachments,
    })
    documents, meta_data = chat.prepare_turn(prompt)
    documents = handle_attachments(documents)

    with st.chat_message("assistant"):
        history = get_history(chat)
        thinking_indicator = st.empty()
        thinking_indicator.markdown("**Processing Prompt…**")
        answer_container = st.empty()
        reasoning_buffer = ""
        metrics = StreamMetrics()
        metrics.turn_count = len(history)
        stream = call_llm_stream(renderer, documents, meta_data, metrics)
        full_response = ""
        for chunk in stream:
            if is_thinking(chunk):
                reasoning_buffer += chunk.content
            else:
                full_response += chunk.content
            answer_container.markdown(full_response)

            # ── Sidebar indicator (shows only while reasoning) ───────────
            if is_thinking(chunk):
                thinking_indicator.markdown(f"🤔 **Reasoning… [{renderer.llm.model_name}]**"
                                            "\n\n_Pausing output_")
            else:
                thinking_indicator.empty()

        # ── Final render — expander with full reasoning block ───────────
        if reasoning_buffer:
            # Clean up the ephemeral indicator
            thinking_indicator.empty()

            with st.expander("🤔 Reasoning", expanded=False):
                st.markdown(reasoning_buffer)

        # Post-stream metrics
        st.caption(
            f"Turn: {metrics.turn_count} • "
            f"{metrics.model} • "
            f"{metrics.generation_time:.2f}s • "
            f"Tokens: {metrics.token_count} • "
            f"{metrics.token_count / metrics.generation_time:.1f} T/s"
        )

    # Save after streaming completes
    save_messages(renderer, documents, full_response)
