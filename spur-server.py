#!/usr/bin/env python3
"""
Spur adapter — drop this next to chat.py.

The React UI is only a view. This process owns Chat / SessionContext /
RenderWindow: branches, pickle history, RAG clone/reset, agent tools,
prepare_turn, stream_response, save_history. LM Studio (or whatever is
in .chat.yaml) is reached the same way the terminal app already does.

  uv run --with fastapi --with uvicorn spur-server.py
  # or: pip install fastapi uvicorn && python spur-server.py

Then point the UI at this origin:

  VITE_CHAT_API=http://127.0.0.1:8765
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import shutil
import sys
import time
from copy import deepcopy
from typing import Any, Iterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from rich.console import Console

ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(ROOT) == "public":
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from chat import (  # noqa: E402
    Chat,
    ChatOptions,
    SessionContext,
    parse_args,
    parse_user_input,
    seed_from_string,
)
from src.think_tags import ThinkFeed  # noqa: E402

LOCKED_BRANCHES = frozenset({"assistant", "story"})
# Metadata keys in the pickle — not message lists.
HISTORY_META_KEYS = frozenset({"current", "assistant_mode", "branch_modes"})
RESERVED_NAMES = frozenset(
    {"current", "assistant", "story", "assistant_mode", "branch_modes"}
)

app = FastAPI(title="Spur")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

console = Console(highlight=True)
_chat: Chat | None = None


def get_chat() -> Chat:
    global _chat
    if _chat is None:
        opts = ChatOptions.from_yaml(ROOT)
        args = parse_args(sys.argv[1:], opts)
        built = ChatOptions.from_args(ROOT, args, opts)
        built.seed = seed_from_string(str(built.seed))
        session = SessionContext.from_args(console, built)
        _chat = Chat(session, built)
        _sync_chat_object(_chat)
    return _chat


def sse(obj: dict[str, Any]) -> str:
    # Escape < > so a proxy/browser that HTML-parses the SSE body cannot
    # treat a MiniMax inner <think> token as a real tag and swallow the rest
    # of the stream.
    payload = json.dumps(obj).replace("<", "\\u003c").replace(">", "\\u003e")
    return f"data: {payload}\n\n"


def _history(chat: Chat) -> dict:
    return chat.session.common.load_chat()


def _vector_dir() -> str:
    if _chat is not None:
        return _chat.opts.vector_dir
    opts = ChatOptions.from_yaml(ROOT)
    vdir = opts.vector_dir
    if not os.path.isabs(vdir):
        vdir = os.path.join(ROOT, vdir)
    return vdir


def read_hist() -> dict:
    """Load pickle directly so the branch list does not wait on Chat/Chroma."""
    path = os.path.join(_vector_dir(), "chat_history.pkl")
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            return data
    except FileNotFoundError:
        pass
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    if _chat is not None:
        return _history(_chat)
    return {
        "story": [],
        "assistant": [],
        "current": "story",
        "branch_modes": {},
        "assistant_mode": False,
    }


def _canonical_mode(name: str) -> bool | None:
    if name == "assistant":
        return True
    if name == "story":
        return False
    return None


def _persisted_mode(hist: dict, name: str, fallback: bool = False) -> bool:
    canon = _canonical_mode(name)
    if canon is not None:
        return canon
    return bool(hist.get("branch_modes", {}).get(name, fallback))


def _turn_count(msgs: list) -> int:
    if not msgs:
        return 0
    return sum(1 for m in msgs if isinstance(m, dict) and m.get("role") == "user")


def _refresh_mode_runtime(chat: Chat) -> None:
    mode = bool(chat.opts.assistant_mode)
    renderer = chat.session.renderer
    context = chat.session.context
    for holder in (
        getattr(renderer, "opts", None),
        getattr(renderer, "args", None),
        getattr(renderer, "state", None),
        getattr(getattr(renderer, "prompts", None), "args", None),
        getattr(context, "opts", None),
        getattr(context, "args", None),
        getattr(getattr(context, "prompts", None), "args", None),
    ):
        if holder is not None and hasattr(holder, "assistant_mode"):
            holder.assistant_mode = mode
    if hasattr(renderer, "state"):
        renderer.state.assistant_mode = mode
    renderer.assistant_prompt = mode
    context.assistant_prompt = mode
    context.mode = "document_topics" if mode else "entity"
    if hasattr(renderer, "prompts") and hasattr(renderer.prompts, "build_prompts"):
        renderer.prompts.prompt_model = chat.opts.model
        renderer.prompts.build_prompts()
    if hasattr(renderer, "build_prompts"):
        renderer.prompt_model = chat.opts.model
        renderer.build_prompts()
    if hasattr(context, "prompts") and hasattr(context.prompts, "build_prompts"):
        context.prompts.prompt_model = getattr(
            chat.opts, "preconditioner", chat.opts.model
        )
        context.prompts.build_prompts()


def _sync_chat_object(chat: Chat, hist: dict | None = None) -> tuple[str, bool]:
    hist = hist or _history(chat)
    branch = hist.get("current", "story")
    mode = _persisted_mode(
        hist, branch, fallback=bool(hist.get("assistant_mode", False))
    )
    chat.chat_branch = branch
    chat.opts.assistant_mode = mode
    hist["assistant_mode"] = mode
    hist.setdefault("branch_modes", {})
    if _canonical_mode(branch) is None:
        hist["branch_modes"][branch] = mode
    chat.session.common.save_chat(hist)
    _refresh_mode_runtime(chat)
    return branch, mode


def _message_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(
                    _message_text(item.get("text") or item.get("content") or "")
                )
        return "".join(parts)
    return str(value)


def _mode_label(assistant: bool) -> str:
    return "assistant" if assistant else "story"


def session_payload(chat: Chat | None = None) -> dict[str, Any]:
    hist = _history(chat) if chat is not None else read_hist()
    if chat is not None:
        current, mode = _sync_chat_object(chat, hist)
    else:
        current = hist.get("current", "story")
        mode = _persisted_mode(
            hist, current, fallback=bool(hist.get("assistant_mode", False))
        )
    branches: dict[str, Any] = {}
    for name, msgs in hist.items():
        if name in HISTORY_META_KEYS or not isinstance(msgs, list):
            continue
        locked = name in LOCKED_BRANCHES
        assistant = _persisted_mode(hist, name, fallback=mode)
        clean = []
        for m in msgs:
            if isinstance(m, dict):
                row = {
                    "role": "user" if m.get("role") == "user" else "assistant",
                    "content": _message_text(m.get("content")),
                }
                if m.get("reasoning"):
                    row["reasoning"] = m["reasoning"]
                if m.get("metrics"):
                    row["metrics"] = m["metrics"]
                if m.get("attachments"):
                    row["attachments"] = m["attachments"]
                clean.append(row)
            elif isinstance(m, str) and m.strip():
                clean.append({"role": "assistant", "content": m})
        branches[name] = {
            "id": name,
            "name": name,
            "mode": _mode_label(assistant),
            "locked": locked,
            "messages": clean,
        }
    if "story" not in branches:
        branches["story"] = {
            "id": "story",
            "name": "story",
            "mode": "story",
            "locked": True,
            "messages": [],
        }
    if "assistant" not in branches:
        branches["assistant"] = {
            "id": "assistant",
            "name": "assistant",
            "mode": "assistant",
            "locked": True,
            "messages": [],
        }
    if current not in branches:
        current = "story"
    return {"currentId": current, "branches": branches}


def switch_branch(chat: Chat, name: str) -> tuple[bool, str]:
    hist = _history(chat)
    if not isinstance(hist.get(name), list):
        return False, f"Branch '{name}' does not exist."
    old = hist.get("current", "story")
    if name == old:
        return True, f"Already on '{name}'."
    new_mode = _persisted_mode(hist, name, fallback=bool(chat.opts.assistant_mode))
    hist["current"] = name
    hist["assistant_mode"] = new_mode
    if _canonical_mode(name) is None:
        hist.setdefault("branch_modes", {})[name] = new_mode
    chat.session.common.save_chat(hist)
    chat.chat_branch = name
    chat.opts.assistant_mode = new_mode
    if hasattr(chat.session.renderer, "clear_ooc"):
        chat.session.renderer.clear_ooc()
    _refresh_mode_runtime(chat)
    return True, f"Switched to '{name}'."


def create_branch(chat: Chat, name: str, cut_turns: int | None) -> tuple[bool, str]:
    name = (name or "").strip()
    if not name:
        return False, "Branch name cannot be empty."
    if name in RESERVED_NAMES:
        return False, f"'{name}' is reserved."
    hist = _history(chat)
    src = hist.get("current", "story")
    base = hist.get(src, [])
    if not isinstance(base, list):
        return False, f"Source branch '{src}' is invalid."
    if name in hist:
        return switch_branch(chat, name)
    if cut_turns is not None:
        cut_idx = max(0, min(int(cut_turns) * 2, len(base)))
        new_list = deepcopy(base[:cut_idx])
    else:
        new_list = deepcopy(base)
    source_mode = _persisted_mode(hist, src, fallback=bool(chat.opts.assistant_mode))
    hist[name] = new_list
    hist["current"] = name
    hist.setdefault("branch_modes", {})[name] = source_mode
    hist["assistant_mode"] = source_mode
    try:
        chat.session.common.save_chat(hist)
        if hasattr(chat.session.renderer, "clear_ooc"):
            chat.session.renderer.clear_ooc()
        if cut_turns is None:
            chat.session.rag.clone_collection(src, name, overwrite=False)
        elif hasattr(chat.session.rag, "build_collection_from_texts"):
            texts = [
                m.get("content", "") if isinstance(m, dict) else str(m)
                for m in new_list
            ]
            chat.session.rag.build_collection_from_texts(name, texts, overwrite=True)
        chat.chat_branch = name
        chat.opts.assistant_mode = source_mode
        _refresh_mode_runtime(chat)
        return True, f"Branched to '{name}'."
    except Exception as exc:  # pylint: disable=broad-exception-caught
        hist.pop(name, None)
        hist["current"] = src
        hist.setdefault("branch_modes", {}).pop(name, None)
        chat.session.common.save_chat(hist)
        chat.chat_branch = src
        return False, f"RAG sync failed: {exc}"


def delete_branch(chat: Chat, name: str) -> tuple[bool, str]:
    if name in RESERVED_NAMES or name in LOCKED_BRANCHES:
        return False, f"Cannot delete protected branch '{name}'."
    hist = _history(chat)
    if hist.get("current") == name:
        return False, "Cannot delete the branch you are on."
    if name not in hist or not isinstance(hist.get(name), list):
        return False, f"Unknown branch '{name}'."
    hist.pop(name, None)
    hist.get("branch_modes", {}).pop(name, None)
    chat.session.common.save_chat(hist)
    if hasattr(chat.session.rag, "delete_collection"):
        chat.session.rag.delete_collection(name)
    vector_dir = getattr(chat.opts, "vector_dir", "")
    if vector_dir:
        for path in glob.glob(f"{vector_dir}{os.path.sep}{name}*"):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
    return True, f"Deleted '{name}'."


def reset_branch(chat: Chat) -> tuple[bool, str]:
    hist = _history(chat)
    branch = hist.get("current", "story")
    if hasattr(chat.session.rag, "delete_collection"):
        chat.session.rag.delete_collection(branch)
    hist[branch] = []
    chat.session.common.save_chat(hist)
    vector_dir = getattr(chat.opts, "vector_dir", "")
    if vector_dir:
        for path in glob.glob(f"{vector_dir}{os.path.sep}{branch}*"):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
    if hasattr(chat.session.renderer, "clear_ooc"):
        chat.session.renderer.clear_ooc()
    return True, f"Reset '{branch}'."


def delete_last_turn(chat: Chat) -> tuple[bool, str]:
    hist = _history(chat)
    branch = hist.get("current", "story")
    msgs = hist.get(branch, [])
    if not msgs:
        return False, "History empty."
    if msgs and msgs[-1].get("role") == "assistant":
        msgs.pop()
    if msgs and msgs[-1].get("role") == "user":
        msgs.pop()
    hist[branch] = msgs
    chat.session.common.save_chat(hist)
    if hasattr(chat.session.renderer, "clear_ooc"):
        chat.session.renderer.clear_ooc()
    return True, "Deleted last turn."


def pop_last_assistant(chat: Chat) -> tuple[bool, str]:
    hist = _history(chat)
    branch = hist.get("current", "story")
    msgs = hist.get(branch, []) or []
    if msgs and msgs[-1].get("role") == "assistant":
        msgs.pop()
        hist[branch] = msgs
        chat.session.common.save_chat(hist)
    if hasattr(chat.session.renderer, "clear_ooc"):
        chat.session.renderer.clear_ooc()
    return True, "Popped last assistant."


def rewind_to(chat: Chat, n: int) -> tuple[bool, str]:
    hist = _history(chat)
    branch = hist.get("current", "story")
    msgs = hist.get(branch, [])
    total = _turn_count(msgs)
    if not 1 <= n <= total:
        return False, f"Rewind needs 1 ≤ N ≤ {total}."
    hist[branch] = msgs[: n * 2]
    chat.session.common.save_chat(hist)
    if hasattr(chat.session.renderer, "clear_ooc"):
        chat.session.renderer.clear_ooc()
    return True, f"Rewound to turn {n} of {total}."


def set_assistant_mode(chat: Chat, enabled: bool) -> tuple[bool, str]:
    hist = _history(chat)
    current = hist.get("current", "story")
    canon = _canonical_mode(current)
    target = canon if canon is not None else bool(enabled)
    if canon is None:
        hist.setdefault("branch_modes", {})[current] = target
    hist["assistant_mode"] = target
    chat.opts.assistant_mode = target
    chat.chat_branch = current
    chat.session.common.save_chat(hist)
    _refresh_mode_runtime(chat)
    return True, _mode_label(target)


def persist_turn(
    renderer,
    documents: dict,
    response: str,
    reasoning: str = "",
    metrics: dict | None = None,
    attachments: list | None = None,
) -> None:
    documents["llm_response"] = response
    renderer.save_history(documents, response)
    common = renderer.common
    hist = common.load_chat()
    branch = hist.get("current", "story")
    msgs = hist.get(branch) or []
    if (
        len(msgs) >= 3
        and msgs[-1].get("role") == "assistant"
        and msgs[-2].get("role") == "user"
        and msgs[-3].get("role") == "user"
    ):
        msgs.pop(-2)
        hist[branch] = msgs
        common.save_chat(hist)
    if attachments:
        for m in reversed(msgs):
            if m.get("role") == "user":
                m["attachments"] = attachments
                break
    extra = {}
    if reasoning and reasoning.strip():
        extra["reasoning"] = reasoning
    if metrics:
        extra["metrics"] = metrics
    if msgs and msgs[-1].get("role") == "assistant":
        # save_history runs sanitize_response which strips ``` fences.
        # Restore the streamed answer so reload still highlights.
        msgs[-1]["content"] = response
        msgs[-1].update(extra)
    common.save_chat(hist)


def apply_includes(chat: Chat, documents: dict, raw: str) -> dict:
    try:
        parsed = parse_user_input(raw)
    except Exception:  # pylint: disable=broad-exception-caught
        return documents
    if not parsed.includes or not hasattr(chat, "load_content_as_context"):
        return documents
    wrapped = " ".join(f"{{{{{item}}}}}" for item in parsed.includes)
    try:
        extra = chat.load_content_as_context(wrapped)
    except Exception:  # pylint: disable=broad-exception-caught
        return documents
    documents.setdefault("dynamic_files", "")
    documents.setdefault("dynamic_images", [])
    if extra.get("dynamic_files"):
        documents["dynamic_files"] += extra["dynamic_files"]
    if extra.get("dynamic_images"):
        documents["dynamic_images"].extend(extra["dynamic_images"])
    return documents


def fold_uploads(documents: dict, images: list, files: list) -> dict:
    documents.setdefault("dynamic_images", [])
    documents.setdefault("dynamic_files", "")
    for img in images or []:
        raw = img
        if isinstance(raw, str) and "," in raw and raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        if raw:
            documents["dynamic_images"].append(raw)
    for item in files or []:
        name = item.get("name", "file")
        text = item.get("text", "")
        documents["dynamic_files"] += f"\n--- {name} ---\n\n{text}\n\n"
    return documents


def include_branch(chat: Chat, documents: dict, name: str) -> dict:
    hist = _history(chat)
    msgs = hist.get(name)
    if not isinstance(msgs, list):
        return documents
    lines = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = "USER" if m.get("role") == "user" else "AI"
        lines.append(f"{role}: {m.get('content', '')}")
    documents.setdefault("dynamic_files", "")
    documents["dynamic_files"] += f"\n=== include:{name} ===\n" + "\n".join(lines)
    return documents


def _op(fn, *args) -> JSONResponse:
    ok, msg = fn(*args)
    return JSONResponse({"ok": ok, "error": None if ok else msg, "message": msg})


@app.get("/api/session")
def api_session() -> dict[str, Any]:
    # Read pickle directly — do not construct Chat/Chroma just to list branches.
    return session_payload(_chat)


@app.post("/api/branches/switch")
async def api_switch(request: Request) -> JSONResponse:
    body = await request.json()
    return _op(switch_branch, get_chat(), str(body.get("id") or ""))


@app.post("/api/branches")
async def api_create(request: Request) -> JSONResponse:
    body = await request.json()
    cut = body.get("cutTurns")
    cut_n = int(cut) if cut is not None else None
    return _op(create_branch, get_chat(), str(body.get("name") or ""), cut_n)


@app.post("/api/branches/delete")
async def api_delete(request: Request) -> JSONResponse:
    body = await request.json()
    return _op(delete_branch, get_chat(), str(body.get("id") or ""))


@app.post("/api/history/reset")
def api_reset() -> JSONResponse:
    return _op(reset_branch, get_chat())


@app.post("/api/history/delete-last")
def api_delete_last() -> JSONResponse:
    return _op(delete_last_turn, get_chat())


@app.post("/api/history/rewind")
async def api_rewind(request: Request) -> JSONResponse:
    body = await request.json()
    return _op(rewind_to, get_chat(), int(body.get("n") or 0))


@app.post("/api/history/pop-assistant")
def api_pop() -> JSONResponse:
    return _op(pop_last_assistant, get_chat())


@app.post("/api/session/mode")
async def api_mode(request: Request) -> JSONResponse:
    body = await request.json()
    enabled = str(body.get("mode") or "") == "assistant"
    return _op(set_assistant_mode, get_chat(), enabled)


@app.post("/api/chat")
async def api_chat(request: Request) -> StreamingResponse:
    body = await request.json()
    chat = get_chat()
    renderer = chat.session.renderer
    prompt = str(body.get("text") or "")
    use_agent = bool(body.get("useAgent"))
    no_context = bool(body.get("noContext"))
    regenerate = bool(body.get("regenerate"))

    def generate() -> Iterator[bytes]:
        try:
            _sync_chat_object(chat)
            if regenerate:
                pop_last_assistant(chat)
            yield sse({"type": "status", "message": "Working — RAG / agent / prompt…"}).encode()
            if no_context:
                parsed = parse_user_input(prompt)
                documents = chat.no_context(parsed.clean_text or prompt)
                documents["no_context"] = True
                documents["in_line_commands"] = "Meta: [no-context]"
                meta = []
            else:
                parsed = parse_user_input(prompt)
                documents, meta = chat.prepare_turn(parsed.clean_text or prompt)
                documents = apply_includes(chat, documents, prompt)
            documents = fold_uploads(
                documents, body.get("images") or [], body.get("files") or []
            )
            if body.get("includeBranch"):
                documents = include_branch(chat, documents, str(body["includeBranch"]))
            if use_agent and chat.opts.assistant_mode:
                documents["use_agent"] = True
                documents["agent_ran"] = False
                documents["in_line_commands"] = "Meta: [agent]"
                yield sse({"type": "status", "message": "Agent tool web search…"}).encode()
            if body.get("rare"):
                documents["system_addendum"] = (
                    "Story controls for this turn: " + ", ".join(body["rare"])
                )

            renderer.set_llm(meta, documents)
            packed = renderer.get_messages(meta, documents)
            # Stay on Processing Prompt while LM Studio / Ollama loads.
            # Flip to Streaming on the first LLM chunk, even if content is blank.
            yield sse({"type": "status", "message": "Processing Prompt…"}).encode()
            # reveal_thinking is a Rich TUI helper: it zeros chunk.content,
            # starts a console thread, and TypeErrors when MiniMax sends
            # content=None. Split tags here instead. ThinkFeed also covers
            # gpt-oss-style blank first tokens (shadow think → never_think).
            stream_state = getattr(getattr(renderer, "state", None), "stream", None)
            if stream_state is not None:
                stream_state.never_think = False
                stream_state.shadow_think = False
                stream_state.thinking = False
                if hasattr(stream_state, "think_ns"):
                    stream_state.think_ns = ""
            started = time.time()
            first = True
            ttft = 0.0
            tokens = 0
            answer = ""
            reasoning = ""
            parser = ThinkFeed()
            model = getattr(renderer.llm, "model_name", "")

            def bump(n: int = 1) -> None:
                nonlocal first, ttft, tokens
                if first:
                    ttft = time.time() - started
                    first = False
                tokens += max(1, n)

            for chunk in renderer.stream_response(packed):
                if first:
                    # Null/empty first tokens still count as "the stream started".
                    ttft = time.time() - started
                    first = False
                    yield sse({"type": "status", "message": "Streaming…"}).encode()
                visible, thought = parser.feed_chunk(chunk)
                if thought:
                    bump(len(thought.split()))
                    reasoning += thought
                    yield sse({"type": "reasoning", "content": thought}).encode()
                if visible:
                    bump(renderer.response_count(visible))
                    answer += visible
                    yield sse({"type": "token", "content": visible}).encode()
            gen = time.time() - started
            yield sse(
                {
                    "type": "usage",
                    "model": model,
                    "promptTokens": documents.get("prompt_tokens", 0),
                    "completionTokens": tokens,
                    "ttft": ttft,
                }
            ).encode()
            if (answer or reasoning) and not documents.get("no_context"):
                persist_turn(
                    renderer,
                    documents,
                    answer,
                    reasoning,
                    metrics={
                        "model": model,
                        "tokenCount": tokens,
                        "generationTime": gen,
                        "promptTokens": documents.get("prompt_tokens", 0),
                        "tokenSavings": 0,
                        "ttft": ttft,
                    },
                    attachments=body.get("attachments") or None,
                )
            yield sse({"type": "done"}).encode()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            yield sse({"type": "error", "error": str(exc)}).encode()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": True, "backend": "chat.py"})


STATIC = os.environ.get("SPUR_STATIC") or os.path.join(ROOT, "spur-ui")
if os.path.isdir(STATIC):
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)
