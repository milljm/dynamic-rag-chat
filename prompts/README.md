# Prompts
This directly should contain your prized prompts for any given LLM you plan on using. This allows for highly specialized prompts that work well with one LLM architecture over another.

When a specialized prompt is not found, defaults are used. You are of course free to modify the default prompt (I do).

The naming convention happens thusly:

- plot_prompt_LLM-NAME_system.md
    - The system prompt for LLM-NAME, where LLM-NAME is matched with what ever is being provided with `--model`. Example: `--model qwen3-235B-A22B` will match `qwen`, and thus, if the file exists, will load the contents of file `plot_prompt_qwen_system.md`
- plot_prompt_LLM-NAME_human.md
    - Follows the same rules for the system prompt above.

- tagging_prompt_LLM-NAME_human.md
    - The tagging prompt, for the pre-processor. While the chat tool will attempt to load a prompt for the system prompt, it is encouraged not to have one. Most small LLMs do not even support a system prompt. LLM-NAME will match with what ever is being provided with `--pre-llm` (same rules applies as established above).

- pre_conditioner_prompt_LLM-NAME_system.md
- pre_conditioner_prompt_LLM-NAME_system.md
    - both are not used as of yet.

Assistant (`nostory`) plot prompts are composed per turn:

- `plot_prompt_nostory_system.md` — always (identity, buckets, output)
- `plot_prompt_nostory_need_gold.md` — only when DOCUMENTS_INDEX is non-empty and this is *not* a GOLD_RESUME relaunch
- `plot_prompt_nostory_resume.md` — prepended on a NEED_GOLD relaunch (cookbook is omitted so the model does not copy the example tag)
- `plot_prompt_nostory_human.md` — Jinja drops empty THIS_TURN / INDEX / GOLD_RESUME / RAG blocks

Tagging (`tagging_prompt_nostory_human.md`) never offers `vision`. Pixels on the turn append `tagging_prompt_nostory_images.md` (HAS_IMAGE). The orchestrator routes vision only when `dynamic_images` is non-empty; a tagger `vision` without pixels falls back to `structured`.


