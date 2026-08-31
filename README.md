# dynamic-rag-chat

Local chat that **remembers**. Not a sliding window that forgets. Not “dump a folder of PDFs and hope.” Tagged memory, gold documents the model can pull mid-turn, branches, and a router that picks the right model for the job — on your machine.

Windows is not supported natively, but works perfectly fine through [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install) (WSL).

The face of it is **Spur**.

![Spur — dark](spur/docs/screenshot.jpg)

Light mode (washed dust / sand). Code fences have a **theme dropdown** — pick coffee, fruity, stata-light, whatever you like. It sticks.

![Spur — light](spur/docs/screenshot-light.jpg)

---

## Features

- **Spur** — dark / light / system, live status (`RAG Processing…` → `Streaming… [model] [route]`), reasoning disclosure, settings gear.
- **Story and Assistant** — switch in the UI. Story is role-play with scene grounding. Assistant is tools, documents, and optional web search.
- **Memory that is not a window** — tagged RAG (user / AI / gold), deduped against history, older turns staggered so the important bits stay.
- **Documents** — paperclip a file; later mention the name, or the model can recall it with `<NEED_GOLD:file>` even while thinking.
- **URL scrape** — wrap a link in double braces and the page text joins the turn:
  `{{https://example.com/article}}`
- **Web agent** — `\agent` or the Agent toggle. Live search, then the answering model.
- **Coding / Projects** — `\coding` or the Coding toggle (assistant). One row per project root. Persistent **tools** live outside the project (`vector_dir/tools/`); write `tool:uv_setup.py`, run `<TOOL:uv_setup.py>` (cwd is the project, so uv/conda/pip installs land there). Local git agent. Image and Coding are exclusive.
- **Stable Diffusion** — Settings → Automatic1111 URL. **Image** on in assistant (toggle + New ↻). In story, Image is a one-click still of the current beat. No follow-up LLM after the PNG.
- **Model routes** — casual, coder, vision, agent… fill in only what you run. Three models is enough (generator, pre-conditioner, embeddings).
- **Branches** — fork, switch, rewind. `story` and `assistant` are protected.

---

## Getting started

Conda + `uv`. Node is required for Spur (`nodejs` in the conda line).

```bash
conda create -n dynamic-rag python=3.13 uv pip nodejs
conda activate dynamic-rag
# optional: borders / captions after Stable Diffusion
conda install -c conda-forge imagemagick
git clone https://github.com/milljm/dynamic-rag-chat.git
cd dynamic-rag-chat
uv pip install -r requirements.txt
```

Then:

```bash
./chat.py --spur
./chat.py --spur --serve          # same Wi-Fi: iPad / phone. No login — don't port-forward.
```
*Your favorite browser should open. If not, visit [http://localhost:8765](http://localhost:8765).*

On WSL, auto-open is skipped (`gio: … Operation not supported` is harmless). Open the URL in your Windows browser.

First run builds the UI (needs Node). After that it's just Python. Rebuild with `./chat.py --spur --spur-rebuild`. `Ctrl-C` stops it. Open **Settings** (gear) for the server URL and models — next turn uses them. Defaults can live in `.chat.yaml` (see `.chat.yaml.example`).

### A local stack (Ollama)

You need a generator, a small pre-conditioner, and embeddings.

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
ollama pull gemma3:12b               # generator
```

```bash
./chat.py --spur \
  --model gemma3:12b \
  --pre-llm gemma3:1b \
  --embedding-llm nomic-embed-text \
  --model-server http://localhost:11434/v1
```

LM Studio works the same way — point `--model-server` at its OpenAI-compatible URL.

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
./chat.py --spur
```

---

## Also here: terminal and Streamlit

```bash
./chat.py                       # story, terminal
./chat.py --assistant-mode      # assistant, terminal
streamlit run streamlit_chat.py -- --assistant-mode
```

Same engine. Spur is just the nicer face.

The old terminal look, if you care:

<img width="764" alt="light_mode" src="https://github.com/user-attachments/assets/df7bd018-0354-45e7-8451-903d2834fcfd" />

https://github.com/user-attachments/assets/07976c98-3935-4b24-a1c0-e09dcd8bf07b

---

## Slash commands

Same in Spur, Streamlit, and the terminal.

```text
\?                          help
\regenerate                 regenerate last turn
\no-context msg             query with no RAG / history context
\agent msg                  force web-search agent
\coding msg                 write/run local workers in the Projects workspace
\image msg                  force Stable Diffusion (Automatic1111)
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

{{https://example.com}}     scrape that page into this turn
{{/absolute/path/to/file}}  include a local file
```

Protected branch names: `story`, `assistant`.

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
                   scene / entity sheets (story)
     ↓
[Routed generator]  vision / NSFW / coder / … / default
     ↓
[Optional polisher]  story mode only
     ↓
[Screen]
     ↓  (background thread)
[Write AI turn into the AI collection; maybe mint an NPC sheet]
```

Most chat UIs are a sliding token window. When it fills, facts fall out. This keeps a tagged store next to the conversation: a small pre-conditioner emits metadata, those tags field-filter Chroma (user / AI / gold), hits mix with similarity + BM25 + parent retrieval, then dedupe against recent history. Older turns are sampled, not dumped. The result goes to the model that fits the job.

### RAG layout

Each **branch** owns `{branch}_user_documents` and `{branch}_ai_documents`. Story can also read an un-prefixed **gold** collection (import-only). Assistant gold is `assistant_gold_documents`.

| | Gold RAG (chunks) | `vector_dir/attachments/` (whole files) |
|---|---|---|
| `--import-dir` | yes | no |
| Paperclip / `{{path}}` in assistant mode | yes (search) | yes (the file itself) |
| Spur **Documents** widget | — | list + delete |
| Filename in the query | — | inject the whole file |
| `<NEED_GOLD:filename>` mid-reply | — | fetch, resume same turn (cap 2) |

**Projects** (Coding toggle / `\coding`): default `vector_dir/projects/workspace/`. **Add project dir** registers an existing directory in place — that directory is one project. Named fences persist in the active root. Own-line `<RUN:file.py args>`, `<READ:file.py>`, `<GIT:status>`, or `<TOOL:uv_setup.py>` mid-reply (cap 8). **Tools** (`vector_dir/tools/`) are the model's own namespace — outside the git project, reused across projects. Write with ` ```python tool:name.py `; cwd when run is the project so installs go there. Git is local only. If `git: no`, the model must ask before `<GIT:init>`.

Chunking is parent/child:

| Mode      | Parent chunk / overlap / split | Child chunk / overlap / split |
|-----------|--------------------------------|-------------------------------|
| Story     | 1000 / 500 / `\n\n`            | 100 / 50 / `.`                |
| Assistant | 2000 / 1000 / `\n\n`           | 1000 / 500 / `.`              |

`--rag-matches 0` disables retrieval. `--no-rags` skips retrieve/store (tagging and routing still run). RAG is on by default.

### Models

**Required (3):** generator (`--model`), pre-conditioner (`--pre-llm`), embeddings (`--embedding-llm`).

**Optional routes** (only if configured): vision, agent/web-search, casual, general, coder, structured. Story extras: polisher, NSFW, entity/NPC writer.

You do not need seven models running. Three is enough.

Stable Diffusion: Settings → A1111 URL (`--sd-server`). Image must be on. ImageMagick is optional (`conda install -c conda-forge imagemagick`). A1111 needs `--api`.

### Tests

```bash
python test_harness.py
```

Runs every `src/*_test.py`. See [CONTRIBUTING.md](CONTRIBUTING.md).

This tree tracks **LangChain 1.x** ([issue #23](https://github.com/milljm/dynamic-rag-chat/issues/23)). Chroma is 1.x; old `vector_data/` from 0.6 may need `\reset` or a gold re-import.

`--debug` / `--prompts-debug` dumps prompts, tags, and RAG payloads. `./chat.py --help` lists every flag.
