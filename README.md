# 🧠 dynamic-rag-chat

**A dynamic, context-aware chat system powered by LLMs, RAGs, and memory.**
_Built for immersive role-playing experiences with evolving knowledge and deep context._

---

## ✨ What is it?

`dynamic-rag-chat` is an open-source chat tool making use of several interesting technologies surounding retrieval-augmented generation (RAG) sources, based on context, and uses lightweight pre-conditioners to summarize relevant information before handing it off to a heavyweight LLM.

This allows the LLM model to:

- Remember plot points, characters, and lore across long sessions
- Provides nuances to chats
- Efficiently use local or remote LLMs in a modular setup

Perfect for storytelling, worldbuilding, AI roleplay, and narrative design tools.

---

## 🧩 Features

- 🧠 **Dynamic RAGs**: Contextual retrieval is triggered by LLM output or user actions
- ✍️ **Preconditioning layer**: Light LLMs summarize fetched data before handing off to larger models (decreasing context box without losing details)
- ⌨️ **Terminal-first UI**: Clean and rich CLI interface using `prompt_toolkit` and `rich`
- 🔁 **Streaming responses**: Get tokens as the model generates them, async-ready
- 🧾 **Chat history tracking**: Maintains memory across turns for better long-term interactions
- 🧪 **Debug mode**: Visualize what the model sees, including RAG hits and prompt stages

---

## 🚀 Getting Started

### 🔧 Installation
The easiest method is to create yourself an environment using Conda, and then using uv pip install for the rest

You _could_ also just throw everything into your current environment, but sooner or later this is a recipe for disaster.

My advice, go to: https://github.com/conda-forge/miniforge, and install Miniforge. Then create an environment soley for the purpse of this project.

```bash
conda create -n dynamic-rag python uv pip
conda activate dynamic-rag
```
_you will need to activate this environment each time you wish to use this tool_

Next, you're going to need Ollama running (or at some location you can access remotely), and hosting several LLMs. I have found the following works very well:

- Heavy Weight LLM [gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it)
- Light Weight LLM (preprosser) [gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)
- Embedding LLM (for RAG work) [nomic-embed-text](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF)

_and then proceed to do the following_
```bash
git clone https://github.com/milljm/dynamic-rag-chat.git
cd dynamic-rag-chat
pip install -r requirements.txt
./chat.py <your-favorite-llm-on-ollama>
./chat.py --help  # for more details on available options
```

### Under the hood design process

```pre
[User Input] → [Regex Tags Parsed] → [Matching RAG Collection Queried]
                                        ↓
[Pre-conditioner Model Summarizes RAG Output]
     ↓
[Final Prompt Constructed with Summarized Context]
     ↓
[Heavyweight LLM Responds]
     ↓
[Chat History + Context Saved] → (treaded non-blocking [Regex Tags Parsed] → [RAG Collection Extended])
```

### Why am I doing this?
Most RAG systems focus on question answering or document retrieval (1000/200 chunk size/overlap). This project takes a different approach — 200/50 chunk size/overlap, using LLMs to manage their own context through natural output cues (like {{lore:dragon_king}}), and pre-conditioning that knowledge before engaging in deeper conversation. If you're after a story telling LLM, you want to keep details while allowing your LLM to embellish.

My hope is for the retrieval of pertinent information for the task at hand allowing the LLM to 'never forget' the details that matter most.

The result? A responsive, evolving, story-aware model that remembers people, events, and places — just like a good Dungeon Master.
