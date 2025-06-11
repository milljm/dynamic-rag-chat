# 🧠 dynamic-rag-chat

**A dynamic, context-aware chat system powered by LLMs, RAGs, and context management.**
*Built for immersive role-playing with evolving memory and rich, dynamic context.*

---

## ✨ What is it?

`dynamic-rag-chat` is an open-source chat tool built around retrieval-augmented generation (RAG), using metadata field filtering and context tagging. A lightweight pre-conditioner 'tags' relevant information based on the user's query, enabling a highly targeted context window.

This allows the LLM to:

- Recall plot points, characters, and lore across long sessions
- Provide narrative nuance often lost in general-purpose RAG pipelines
- Avoid clutter and hallucination while generating responses more quickly

Perfect for storytelling, world-building, AI role-play, and narrative design — or just a powerful tool for tinkering with LLMs and RAGs.

---

## 🧩 Features

- ⌨️ **Terminal-first UI**: Clean CLI using `prompt_toolkit` and `rich`
- 🔁 **Streaming responses**: Token-level generation, async-ready
- 🧾 **Persistent chat history**: Your context survives between runs
- 🧠 **Dynamic RAGs**: Retrieval is triggered by user input or LLM output
- ✍️ **Preconditioning layer**: Lightweight LLM summarizes RAG output before the larger model (saves tokens while retaining depth)
    *(Note: in active development)*
- 🧩 **Recursive RAG import**:
    `./chat.py --import-dir /path/to/dir` scans, tags, and loads `.txt`, `.md`, `.html`, and `.pdf` files. HTML is parsed using BeautifulSoup.
- 🧪 **Debug mode**: View prompt assembly, RAG matches, and context composition
- 📂 **Inline File & Image Loading**:
    The chat tool supports inline resource references, letting you embed file, image, or URL content directly in your message using double braces:

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
conda create -n dynamic-rag python uv pip
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

This tool uses three LLMs: a heavyweight model for response generation, a lightweight model for tagging/summarizing, and a dedicated embedding model for RAG. If you're not using OpenAI for all three, you'll need a local host like [Ollama](https://ollama.com/).

Install and run Ollama (via Conda or manual method):
```bash
conda activate dynamic-rag-chat
conda install ollama
export OLLAMA_MAX_LOADED_MODELS=3
ollama serve
```
In another terminal:
```bash
conda activate dynamic-rag-chat
ollama list  # Will either return all your hosted models, or nothing. But should NOT fail
ollama pull nomic-embed-text
ollama pull gemma3:1b
ollama pull gemma3:27b  # Only if you’re not using OpenAI
```
Recommended Ollama-hosted models:
- Heavy Weight LLM [gemma-3-27b-it](https://ollama.com/library/gemma3:27b)
- Light Weight LLM (preprosser) [gemma-3-1b-it](https://ollama.com/library/gemma3:1b)
- Embedding LLM (for RAG work) [nomic-embed-text](https://ollama.com/library/nomic-embed-text)

#### ⚙️ Example usage
```bash
conda activate dynamic-rag-chat
./chat.py --model gemma3:27b \
          --pre-llm gemma3:1b \
          --embedding-llm nomic-embed-text \
          --llm-server http://localhost:11434/v1
./chat.py --help  # for more details on available options
```
You can also configure arguments in .chat.yaml. See .chat.yaml.example for a template.

### 🧠 Using OpenAI or ChatGPT

If you have an OpenAI account (note: **not** https://chatgpt.com, but https://platform.openai.com), create a .chat.yaml like this:

```pre
chat:
  model: gpt-4o
  llm_server: https://api.openai.com/v1
  pre_llm: gemma-3-1b-it
  pre_server: http://localhost:11434/v1
  embedding_llm: nomic-embed-text
  embedding_server: http://localhost:11434/v1
  time_zone: 'America/Denver'
  debug: False
  name: Mr. Knowitall
  context_window: 16384
  api_key: YOUR_API_KEY
```
The above will leverage the powerful GPT-4o model, while using your local machine to provide
pre-processing and embeddings through Ollama. With the above set, you would simple run:
```bash
conda activate dynamic-rag-chat
./chat.py
```

### Under the hood design process

```
[User Input]
     ↳ [Inline resource detected: {{/path/to/file}}, {{https://url}}, etc]
          ↓
          → If text file: open and inject content into context
          → If image: base64-encode and embed for LLM vision support
          → If URL: fetch and extract readable text via BeautifulSoup
     ↓
[Pre-conditioner (Lightweight LLM applies metadata tags)]
     ↓
[Metadata tags parsed → field-filtered RAG retrieval]
     ↓
[Context Manager: deduplication, chat history, scene/meta injection]
     ↓
[Prompt Template Constructed]
     ↓
[Heavyweight LLM Generates Response]
     ↓
[Chat + Context Saved]
     ↳ (threaded: new metadata tags → RAG collections updated)
```

### ❓Why This?

Most RAG frameworks focus on Q&A or document retrieval using 1000/200 token chunking. This tool takes a different route:

- Uses 100/50 '.' split chunking for high-granularity tagging
- Applies LLMs to generate their own memory scaffolding ({{lore:...}})
- Employs preconditioned filtering to keep only high-value context
- Balances narrative freedom with factual continuity

This is ideal for storytelling — keeping the LLM imaginative, but grounded.
