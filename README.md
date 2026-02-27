# 🧠 dynamic-rag-chat

**A Terminal-first, orchestrated, context-aware chat system powered by LLMs, RAGs, and context management.**
*Built for immersive role-playing with evolving memory and rich, relevant context.*

---

## ✨ What is it?

`dynamic-rag-chat` is a Terminal UI, open-source chat tool built around retrieval-augmented generation (RAG), using metadata field filtering and context tagging. A lightweight pre-conditioner extracts and tags relevant information based on the user's query, enabling a highly targeted context window. The resulting context is then routed to specialized LLMs based on the task at hand.

This allows the LLM to:

- Recall plot points, characters, and lore across long sessions
- Provide narrative nuance often lost in general-purpose chat tools
- Avoid clutter and hallucination while generating responses more quickly

Perfect for storytelling, world-building, AI role-play, and narrative design — or just a powerful tool for tinkering with LLMs and RAGs.

---

## 🧩 Features

- ⌨️ **Terminal-first UI**: Clean CLI using `prompt_toolkit` and `rich` (Markdown in Terminal)
- 🔁 **Streaming responses**: Token-level generation, async-ready
- 🧾 **Persistent chat history**: Your context survives between runs
- 🧠 **Multiple RAGs**: Retrieval is triggered by user input and LLM output
- ♻️ **Assistant Swap**: Switch between story-teller and assistant mode with an argument (In assistant mode, the chat behaves more like a utility tool, with vision and web-search agent support enabled.)
- ✍️ **Preconditioning layer**: Lightweight LLM summarizes RAG/Chat History before sending to the larger model (saves tokens while retaining depth)
- 🧩 **Recursive RAG import**: Pre-populate your RAG with "Gold" documents or "Canon Lore"
- 🧪 **Debug mode**: View prompt assembly, RAG matches, and context composition, LLM raw output, etc
- 🛠️ **Agents**: Agent tool support for web search (`\agent How are the stocks doing today`)
- 📂 **Inline file & image context aware loading**:
    The chat tool supports inline resource references, letting you embed files, images, or URL(s) content directly in your message using double braces:

    ```text
    images: {{/path/to/image.png}}
    files: {{/path/to/textfile.txt}}
    url: {{https://example.com}}
    ```

    Example:

    ```text
    Compare these two docs: {{/home/user/doc1.txt}} and {{/home/user/doc2.txt}}
    What do you make of this photo? {{/Users/me/Pictures/tree.png}}
    Summarize this page: {{https://somenewssite.com/article123}}
    ```

    Supported file types:

    - ✅ `.txt`, `.md`, `.html`, `.pdf`
    - ✅ `.png`, `.jpg`, `.jpeg` (base64 encoded and injected for vision models)
    - ✅ URLs (scraped via BeautifulSoup for readable text)
- ？**In-line commands**: An extensive in-line command system:
```pre
>>> \?
in-command switches you can use:

        \regenerate                  - regenerate last turn
        \no-context msg              - perform a query with no context
        \agent msg                   - enable agent (web search)
        \delete-last                 - delete last message from history
        \turn                        - show turn/status
        \rewind N                    - rewind to turn N (keep 0..N)
        \branch NAME@N               - set/fork branch name, if empty list branches;
                                       optional @N to fork from first N turns
        \dbranch NAME                - delete chat history branch
        \seed N                      - set RNG seed (or omit to clear)
        \history [N]                 - show last N user inputs (default 5)
        \include branch              - include branch as attachment
        \reset                       - resets history/RAG for current branch

context injection
    {{/absolute/path/to/file}}       - include a file as context
    {{https://somewebsite.com/}}     - include URL as context

keyboard shortcuts (terminal):

    Ctrl-W - delete word left of cursor
    Ctrl-U - delete everything left of cursor
    Ctrl-K - delete everything right of cursor
    Ctrl-A - move to beginning of line
    Ctrl-E - move to end of line
    Ctrl-L - clear screen
```

<img width="764" alt="light_mode" src="https://github.com/user-attachments/assets/df7bd018-0354-45e7-8451-903d2834fcfd" />

https://github.com/user-attachments/assets/07976c98-3935-4b24-a1c0-e09dcd8bf07b

---

## 🚀 Getting Started

### 🔧 Installation

The recommended setup is a Conda environment with `uv` for clean dependency management.

> 🛑 You *can* use your system Python, but that’s a quick path to dependency conflicts. Don't do it.

1. Install [Miniforge](https://github.com/conda-forge/miniforge)
2. Create your environment:

```bash
conda create -n dynamic-rag python=3.13 uv pip
conda activate dynamic-rag
```
_you will need to activate this environment each time you wish to use this tool_

_and then proceed to do the following_
```bash
git clone https://github.com/milljm/dynamic-rag-chat.git
cd dynamic-rag-chat
uv pip install -r requirements.txt
```
*Reminder: you’ll need to `conda activate dynamic-rag` before using the tool each time.*

### 🦙 Ollama (Recommended for Local Models)

This tool requires three LLMs at a minimum: a model for response generation, a pre-conditioner lightweight model for metadata tag extraction, and an embedding model for RAG work.

Install and run Ollama (via Conda or manual method):
```bash
conda activate dynamic-rag
conda install ollama
export OLLAMA_MAX_LOADED_MODELS=3
ollama serve
```
*`OLLAMA_MAX_LOADED_MODELS=3` is encouraged, as this tool uses three+ models simultaneously*

Note: You will need to launch Ollama each time you wish to use chat:
```bash
conda activate dynamic-rag
export OLLAMA_MAX_LOADED_MODELS=3
ollama serve
```

Then, in another terminal (do only once):
```bash
conda activate dynamic-rag
ollama list  # Will either return all your hosted models, or nothing. But should NOT fail
ollama pull nomic-embed-text
ollama pull gemma3:1b   # lightweight pre-processor model
ollama pull gemma3:12b  # heavyweight model that should work on most hardware
```

#### Experiment

There are thousands of models to choose from. I encourage you to experiment! Mix'n match, explore and have fun! Search the internet for Ollama library, or head on over to https://huggingface.co and begin your journey into LLMs. If you are already a fan of HuggingFace, I recommend using this chat tool with LM Studio instead of Ollama (more models to choose from).

#### ⚙️ Example usage

Once Ollama is running, and you have pulled the models you want to use or if you have already pulled the default models above, you only need to launch `./chat.py` without arguments:
```bash
conda activate dynamic-rag
./chat.py
./chat.py --help  # for more details on available options
```
You can also configure arguments in .chat.yaml. See .chat.yaml.example for examples.

### 🧠 Using OpenAI or ChatGPT

If you have an OpenAI account (note: **not** https://chatgpt.com, but https://platform.openai.com), create a .chat.yaml like this:

```pre
chat:
  model: gpt-4o
  llm_server: https://api.openai.com/v1
  pre_llm: gemma3:1b
  pre_server: http://localhost:11434/v1
  embedding_llm: nomic-embed-text
  embedding_server: http://localhost:11434/v1
  time_zone: 'America/Denver'
  debug: False
  name: Mr. Knowitall
  context_window: 16384
  api_key: YOUR_API_KEY
```
The above will leverage the powerful GPT-4o model, while using your local machine to provide pre-processing and embeddings through Ollama. With the above set, you would simple run:
```bash
conda activate dynamic-rag
./chat.py
```

### Under the orchestration process

```
[User Input]
     ↳ [Inline resource detected: {{/path/to/file}}, {{https://url}}, etc]
          ↳ If text file: open and inject content into context
          ↳ If image: base64-encode and embed for LLM vision support
          ↳ If URL: fetch and extract readable text via BeautifulSoup
          ↳ [If \agent: perform web search using dedicated Tool Model]
     ↓
[Pre-conditioner (Lightweight LLM extracts metadata tags)]
     ↓
[Metadata tags parsed → RAG retrieval: field-filtered, BM25, similarity matching]
     ↓
[Context Manager: RAG result deduplication, chat history, scene/meta injection]
          ↳ [If one-shot enabled - Use dedicated summarization model to produce a summary of chat history]
          ↓
     ↳ [If Image detected: Use dedicated Vision Model to produce response]
     ↳ [If content rating is NSFW: use dedicated NSFW Model]
     ↳ [Else: Heavyweight LLM Generates Response]
          ↓
[If Polisher enabled - post-process output using dedicated polisher model for additional quality prose]
     ↓
[Output to Screen]
     ↓ ↳ (threaded non-blocking: new metadata extracted → NPC entity creation from output → RAG collections updated)
[User Input]
```

In all, this tool allows for 8 different models to be used:
- Pre-conditioner metadata extraction
- Summarizer
- Agent Tool (web search)
- Vision model
- NSFW model
- General Heavyweight model
- Polisher post-process model
- Post-Process entity extraction/generation model (the LLM created an NPC that we dedicate creating a permanent character sheet for)

### ❓ Why Use This

Most chat tools treat conversation as a sliding window of tokens. Once the window fills, memory collapses, the model forgets key facts, or worse: invents new ones you never discussed...

I wanted to create a tool that basically felt like talking to world-class tools such as ChatGPT, which makes use of RAGs and multiple LLMs to achieve specialized relevant output.

- RAG Uses 100/50 '.' split chunking for high-granularity tagging suitable for chatting/story telling
     - Both USER and AI utilize their own RAG.
     - RAGs are maintained dynamically (\branch cool_story_so_far) will copy existing RAG artifacts into a new RAG collection as you branch (fork) from a current conversation.
- Employs preconditioned filtering to keep only high-value content for a pruned healthy context token count. (deduplication/fuzzy match removal)
- Utilizes multiple LLMs for their individual strengths (model orchestration).

