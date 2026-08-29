# 🧠 dynamic-rag-chat

**A terminal-first, orchestrated, context-aware chat system powered by LLMs, RAG, and a tagging pre-processor.**
Built for long-running story work and a capable local assistant — with memory that actually persists.

Also ships two GUIs for the same backend: **Streamlit** (`streamlit_chat.py`) and **[Spur](https://github.com/milljm/spur)**. Spur is the prettier one — a React view that talks to `spur-server.py`, a small FastAPI adapter around `chat.py`.

```bash
# Terminal
./chat.py --assistant-mode
./chat.py                       # story mode

# Streamlit (same flags after --)
streamlit run streamlit_chat.py -- --assistant-mode
streamlit run streamlit_chat.py

# Spur — adapter in this repo, UI in milljm/spur
uv run --with fastapi --with uvicorn spur-server.py
# then: VITE_CHAT_API=http://127.0.0.1:8765 npm run dev
```

---

## What it is

Most chat UIs are a sliding token window. When the window fills, facts fall out or get invented.

`dynamic-rag-chat` keeps a tagged memory store next to the conversation:

1. A **lightweight pre-conditioner** reads the turn and emits metadata tags (entities, topics, scene, whether to search the web, …).
2. Those tags **field-filter** Chroma collections (user / AI / optional gold “canon”).
3. Hits are mixed with **similarity + BM25 + parent-document** retrieval, then **deduped** against recent chat history.
4. History itself is **staggered** (recent turns intact, older turns sampled with exponential decay).
5. The resulting context is routed to the model that fits the job (story, coder, vision, agent, …).

That is the whole product: targeted context, not a bigger window.

---

## Features

- **Terminal UI** — `prompt_toolkit` + `rich` (Markdown in the terminal, optional `--light-mode`)
- **Streamlit UI** — branch cards, mode toggle, attachments, slash commands, reasoning panel (same `Chat` / `RenderWindow` stack)
- **Spur UI** — React front-end ([milljm/spur](https://github.com/milljm/spur)). Same stack via `spur-server.py`; just nicer to look at.
- **Streaming** — token-level generation; Spur shows `Processing Prompt…` / `Streaming…` with `[model] [route] [12.4k]`
- **Persistent history** — JSON on disk (`vector_dir/chat_history.json`, atomic write + `.bak`). Role/content per branch, including reasoning blocks.
- **Branches** — fork / switch / delete; RAG collections clone with the fork (`\branch`, `\dbranch`, or the Streamlit / Spur cards)
- **Two flavors**
  - **Story** — role-play prompts, scene grounding, optional NPC sheets + polisher
  - **Assistant** — tool-style prompts, optional vision + web-search agent, RAG on (pass `--no-rags` to disable)
- **Pre-processor** — lightweight LLM for tags *and* model routing (casual → general → coder → analysis)
- **Optional post-process** — threaded RAG write-back; story mode can mint entity files
- **Gold / canon import** — pre-load a read-oriented collection from `.md`, `.html`, `.txt`, `.pdf`, `.template` (`--import-dir`)
- **Documents cabinet** — assistant paperclips land as whole files in `vector_dir/attachments/` *and* as gold chunks. Mention the filename, or the model emits `<NEED_GOLD:file>` (even while thinking). Spur status: `Recalling Document… [README.md, …]`
- **Inline context** — files, images, and URLs in the message:
  ```text
  Compare {{/home/user/a.txt}} and {{/home/user/b.txt}}
  What is this? {{/Users/me/Pictures/tree.png}}
  Summarize {{https://example.com/article}}
  ```
- **Agents** — pre-processor can request a web search (threshold via `--distrust-confidence`), or force it with `\agent …`. The agent may re-search once more (cap 2) if the first dump looks thin.
- **Think tags** — MiniMax `<mm:think>`, `<think>`, and null-token reasoners are split out of the visible stream. Spur has a Reasoning disclosure.
- **Debug** — `--debug` / `--prompts-debug` dumps prompts, tags, and RAG payloads

### In-line commands

```text
\?                          help
\regenerate                 regenerate last turn
\no-context msg             query with no RAG / history context
\agent msg                  force web-search agent
\delete-last                drop the last user+assistant pair
\turn                       print current turn count
\rewind N                   keep turns 1..N
\branch                     list branches
\branch NAME                switch or create a full fork
\branch NAME@N              fork from the first N turns
\dbranch NAME               delete a non-protected branch (+ its RAG)
\include branch             attach another branch as context
\reset                      wipe history + RAG for the current branch
\history [N]                last N user inputs (default 5)
```

Protected branch names: `story`, `assistant`. Metadata keys (`current`, `assistant_mode`, `branch_modes`) are not branches.

Terminal shortcuts: Ctrl-W / U / K / A / E / L.

<img width="764" alt="light_mode" src="https://github.com/user-attachments/assets/df7bd018-0354-45e7-8451-903d2834fcfd" />

https://github.com/user-attachments/assets/07976c98-3935-4b24-a1c0-e09dcd8bf07b

---

## How context is built

```
[User input]
  ↳ {{path}} / {{url}} / \agent
        file → inject text
        image → base64 for a vision route
        URL → BeautifulSoup text
        \agent → tool model web search
     ↓
[Pre-conditioner]  tags + route
     ↓
[RAG]  field filter → similarity → BM25 → parent docs
     ↓
[Context manager]  dedupe vs history, stagger old turns,
                   scene / entity sheets (story), explicit flag
     ↓
[Routed generator]  vision / NSFW / coder / … / default story or assistant model
     ↓
[Optional polisher]  story mode only
     ↓
[Screen]
     ↓  (background thread)
[Write AI turn into the AI collection; maybe mint an NPC sheet]
```

### RAG layout (what the code actually does)

Each **branch** owns collections named `{branch}_user_documents` and `{branch}_ai_documents`. Story mode can also read an un-prefixed **gold** collection (import-only; not cloned on fork). Assistant gold is `assistant_gold_documents`.

Two jobs, two stores:

| | Gold RAG (chunks) | `vector_dir/attachments/` (whole files) |
|---|---|---|
| `--import-dir` | yes | no |
| Paperclip / `{{path}}` in assistant mode | yes (search) | yes (the file itself) |
| Spur **Documents** widget | — | list + delete |
| Filename in the query (`look at spur-server.py`) | — | inject the whole file |
| `<NEED_GOLD:filename>` mid-reply | — | fetch, resume same turn (cap 2) |

Story gold stays import-only. Paperclips this turn are `THIS_TURN_ATTACHMENTS` / `FILES` in the prompt (full text); after the turn they become Documents.

Chunking is parent/child, not a single 100/50 split:

| Mode      | Parent chunk / overlap / split | Child chunk / overlap / split |
|-----------|--------------------------------|-------------------------------|
| Story     | 1000 / 500 / `\n\n`            | 100 / 50 / `.`                |
| Assistant | 2000 / 1000 / `\n\n`           | 1000 / 500 / `.`              |

Retrieval is an ensemble (filter + similarity), then BM25 on that set, then parent-document expansion. Dedupe drops chunks that overlap history or each other (~65% containment).

`--rag-matches 0` disables retrieval. `--no-rags` skips retrieve/store entirely (tagging and routing still run). RAG is on by default in both story and assistant mode.

### Models

**Required (3):**

- Generation model (`--model`)
- Pre-conditioner (`--pre-llm`)
- Embeddings (`--embedding-llm`)

**Optional routes** (only used if configured and the pre-processor / flags say so):

- Vision, agent/web-search, casual, general, coder, structured
- Story extras: polisher, NSFW, entity/NPC writer

You do not need seven models running. Three is enough; the rest are sockets you can fill.

---

## GUIs

The terminal is the source of truth. Two optional fronts wrap it.

**Streamlit** (`streamlit_chat.py`) is the original GUI. Same flags as `chat.py` after `--`. Fine if you already live in Python.

**Spur** is a React UI that never imports LangChain. `spur-server.py` sits next to `chat.py` and exposes the same session: branches, JSON history (`chat_history.json`, migrated from pickle on first load), RAG, Documents cabinet, agent tools, SSE tokens (including reasoning vs visible, and `Recalling Document…` when the model asks for a file). The UI is only a view — that is why the adapter lives in *this* repo, not in [milljm/spur](https://github.com/milljm/spur). Point Spur at it with `VITE_CHAT_API=http://127.0.0.1:8765`. OpenAPI is at `http://127.0.0.1:8765/docs`.

Spur’s sidebar is **Documents** (permanent cabinet) and **Downloadable Files** (named code fences still in history). Paperclip lives on the composer. Status line: `Processing Prompt…` / `Streaming…` with `[model] [route] [12.4k]`.

Streamlit still works. Spur is just prettier.

## Getting started

### Install

Conda + `uv` is the intended path. System Python will fight you.

```bash
conda create -n dynamic-rag python=3.13 uv pip nodejs
conda activate dynamic-rag
git clone https://github.com/milljm/dynamic-rag-chat.git
cd dynamic-rag-chat
uv pip install -r requirements.txt
```

Python 3.10+ (3.13 is what the conda line above uses). This tree now tracks **LangChain 1.x** ([issue #23](https://github.com/milljm/dynamic-rag-chat/issues/23)). Retrievers / `AgentExecutor` live in `langchain-classic`; prompts, messages, and tools in `langchain-core`.

Chroma jumped with it (`chromadb` 0.6 → 1.x). Existing `vector_data/` written under 0.6 may not open. If retrieve errors after the upgrade, `\reset` the branch or re-import gold.

### Tests

```bash
python test_harness.py
```

That runs every `src/*_test.py` as a top-level module (so `src/__init__.py` is not imported). You can still run one file with `python src/think_tags_test.py`.

### Ollama (local models)

You need a generator, a small pre-conditioner, and an embedding model.

```bash
conda activate dynamic-rag
conda install ollama=0.24.0          # or install Ollama yourself
export OLLAMA_MAX_LOADED_MODELS=3
ollama serve
```

Other terminal, once:

```bash
conda activate dynamic-rag
ollama pull nomic-embed-text
ollama pull gemma3:1b                # pre-processor
ollama pull gemma3:12b               # generator that fits most boxes
```

Then:

```bash
./chat.py \
  --model gemma3:12b \
  --pre-llm gemma3:1b \
  --embedding-llm nomic-embed-text \
  --model-server http://localhost:11434/v1
```

Add `--assistant-mode` for the utility flavor. `./chat.py --help` lists every flag. Defaults can live in `.chat.yaml` (see `.chat.yaml.example`).

LM Studio works the same way — point `--model-server` at its OpenAI-compatible URL. Handy if you want Hugging Face weights Ollama does not ship.

### OpenAI for generation, local for the rest

`.chat.yaml`:

```yaml
chat:
  model: gpt-4o
  llm_server: https://api.openai.com/v1
  pre_llm: gemma3:1b
  pre_server: http://localhost:11434/v1
  embedding_llm: nomic-embed-text
  embedding_server: http://localhost:11434/v1
  time_zone: America/Denver
  name: Mr. Knowitall
  context_window: 16384
  api_key: YOUR_API_KEY
```

```bash
conda activate dynamic-rag
./chat.py
```

---

## Why this exists

Sliding-window chats forget or hallucinate once the window is full. This tool keeps **user** and **AI** memory in separate, tagged collections, prunes duplicates, and only pulls what the pre-processor tagged as relevant — so a 12B local model can stay coherent across a long story or a working assistant session on one machine.
