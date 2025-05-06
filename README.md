# 🧠 dynamic-rag-chat

**A dynamic, context-aware chat system powered by LLMs, RAGs, and Contex Management.**
_Built for immersive role-playing experiences with evolving knowledge and deep context._

---

## ✨ What is it?

`dynamic-rag-chat` is an open-source chat tool making use of several interesting technologies surrounding retrieval-augmented generation (RAG) sources, based on context tagging metadata field filtering. By using a lightweight pre-conditioners to 'tag' relevant information along the users query, we can populate the context window with highly relevant data.

This allows the LLM model to:

- Remember plot points, characters, and lore across long sessions
- Provides nuances that would otherwise be missed in a general RAG retrieval
- Clutter removed to help the LLM avoid hallucinations while achieving a quicker response

Perfect for storytelling, world building, AI role-play, and narrative design tools. Or just a tool to tinker with RAGs and LLMs.

---

## 🧩 Features

- ⌨️ **Terminal-first UI**: Clean and rich CLI interface using `prompt_toolkit` and `rich`
- 🔁 **Streaming responses**: Get tokens as the model generates them, async-ready
- 🧾 **Chat history tracking**: Maintains history after you exit the tool
- 🧠 **Dynamic RAGs**: Contextual retrieval is triggered by LLM output or user actions
- ✍️ **Preconditioning layer**: Light LLMs summarize fetched data before handing off to larger models (decreasing context box without losing details) -NOT Complete yet!
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

_and then proceed to do the following_
```bash
git clone https://github.com/milljm/dynamic-rag-chat.git
cd dynamic-rag-chat
uv pip install -r requirements.txt
```

### Optional. Obtain/Use Ollama how you see fit

Next, you're going to need Ollama running (or can access remotely), and hosting several LLMs. I have found the following work very well:

- Heavy Weight LLM [gemma-3-27b-it](https://ollama.com/library/gemma3:27b)
- Light Weight LLM (preprosser) [gemma-3-1b-it](https://ollama.com/library/gemma3:1b)
- Embedding LLM (for RAG work) [nomic-embed-text](https://ollama.com/library/nomic-embed-text)

If you choose to run your own:
conda-forge has Ollama pre-built, and up to date. You can choose the following method, or obtain/use Ollama any way you wish.
```bash
conda install ollama
export OLLAMA_MAX_LOADED_MODELS=3  # We are working with three LLMs simultaneously!
ollama serve
```
*This will launch the Ollama server, serving on localhost:11434*
Leave Ollama running, and open a new terminal, activate the same environment (remember you have to do this each time you wish to use this chat tool) and perform the following:
```bash
conda activate dynamic-rag
# as a test, see if Ollama responds:
ollama list
# should either produce a list of your LLMs or an emty table.
ollama pull nomic-embed-text
ollama pull gemma3:1b
ollama pull gemma3:27b
ollama list
```
`ollama list` needs to display all the models necessary to run this utility. That being: A heavy LLM (whatever your machine can afford), a light-weight LLM (used for TAG gathering and in the future when I get the system prompts working: a summarizer filter), and an embedding model (this is used when dealing with the many RAG collections this tool will generate).

```bash
./chat.py --model gemma3:27b \
          --pre-llm gemma3:1b \
          --embedding-llm nomic-embed-text \
          --server localhost:11434
./chat.py --help  # for more details on available options
```
You can manage all those arguments by creating a `.chat.yaml` file. See `.chat.yaml.example` for details.


### Under the hood design process

```pre
[User Input] → [Regex Tags Parsed] → [Matching RAG Collection Queried]
                                        ↓
                                     [Contextual Management]
                                        ↓
[Pre-conditioner Model Summarizes RAG Output] # not complete, its too aggressive at present
     ↓
[Final Prompt Constructed with Summarized Context]
     ↓
[Heavyweight LLM Responds]
     ↓
[Chat History + Context Saved] → (treaded non-blocking [Regex Tags Parsed] → [RAG Collection Extended])
```

### Why am I doing this?
Most RAG systems focus on question answering or document retrieval (1000/200 chunk size/overlap). This project takes a different approach — 100/50 chunk size/overlap, using LLMs to manage their own context through natural output cues (like {{lore:dragon_king}}), and pre-conditioning that knowledge before engaging in deeper conversation. If you're after a story telling LLM, you want to keep details while allowing your LLM to embellish.

My hope is for the retrieval of pertinent information for the task at hand allowing the LLM to 'never forget' the details that matter most.

The result? A responsive, evolving, story-aware model that remembers people, events, and places — just like a good Dungeon Master.
