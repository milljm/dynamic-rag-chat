# Prompts

Prompt templates are organized by **flavor → role → slot**:

```
prompts/
├── story/
│   ├── pre_processor/     # light LLM: metadata tagging, entity sheets
│   └── heavy/             # main LLM: canon, ooc, polish, addenda
└── assistant/
    ├── pre_processor/     # light LLM: metadata tagging
    └── heavy/             # main LLM: canon + retrieval/search fragments
```

- **flavor** — `story` or `assistant`. Selected at runtime by
  `--assistant-mode` / `args.assistant_mode`; see `PromptManager.flavor()`.
- **role** — which model reads the prompt. `pre_processor` is the light
  LLM (`--pre-llm`, used for tagging and entity extraction). `heavy` is
  the answering model (`--model`).
- **slot** — the filename stem. `*_system.md` fills the system message,
  `*_human.md` fills the human message, and any other name is an
  optional fragment composed in for the turn.

## Slot conventions

The two prompts that every flavor needs are the `canon` pair:

- `heavy/canon_system.md` — the always-on spine (identity, rules, style).
- `heavy/canon_human.md` — the Jinja2 data payload (documents, history, scene).

Fragments are appended to the spine only when the turn calls for them, so
the model is not handed instructions for a mode it is not in. Files are
plain markdown with Jinja2 `{{ variable }}` / `{% if %}` support.

## Current tree

### `story/pre_processor/`
- `tagging_human.md` — scene metadata extraction: `entity`, `creature`,
  `audience`, `content_rating`, `nsfw_reason`, `player_location`,
  `npc_locations`, `moving_confidence`, plus `prompt_stack`. Human prompt
  only — many light LLMs do not support a system prompt, so
  `tagging_system.md` stays empty by design.
  **A name is what earns RAG scoping.** `entity` is people (named or not),
  plus any creature that has been given a name — a dog called Spot, a cat
  called Pepper, a swarm called the Wailing Death — because `entity` is the
  RAG field filter (`ContextManager.mode`). `creature` is unnamed,
  non-thinking beasts: scene colour and continuity that never drives
  retrieval.
- `mood_router_human.md` — the six-family mood router (see below).
- `entity_human.md` — sheets for characters (`name`, `gender`, `race`,
  `appearance`). The humanoid shape covers goblins and orcs as readily as
  humans.
- `creature_human.md` — sheets for non-sentient beasts (`name`, `kind`,
  `size`, `demeanour`, `danger`, `appearance`). Never a gender or a race: a
  rat snake must not be asked its ancestry.
- `entity_system.md`, `preconditioner_system.md`,
  `preconditioner_human.md` — reserved.

### `story/heavy/`

The story system prompt is a **control stack** composed per turn by
`compose_story_plot()`. One file per concern, so a turn loads only the
behaviour it needs.

Always sent, in order (unless a mood's `replaces` displaces one):
- `canon_system.md` — `<ROOT_PRIMER>`: identity, setting, what you control.
- `agency_system.md` — player agency, input contract, continuation.
- `camera_system.md` — POV and pronoun law.
- `echo_system.md` — anti-echo rules for NPC speech.
- `style_system.md` — prose style.
- `npc_system.md` — baseline NPC behaviour.
- `world_system.md` — world response and location law.
- `plot_system.md` — plot advancement pressure.
- `initiation_system.md` — when the world may move first.
- `checklist_system.md` — final self-check.
- `canon_human.md` — the Jinja data payload (documents, history, scene).
  Its SCENE_STATE block lists `present` (people) and `creatures`
  (non-people) on separate lines.

Conditional:
- `addendum_system.md` — only when the turn is **explicit** (`explicit` /
  `content_rating == nsfw`, from the pre-processor's `content_rating`) *and*
  `additional_content` is non-empty. A SFW turn never carries `nsfw.md`,
  however long that file happens to be.
- `mood_*_system.md` — only when the turn selects that mood.

Sent instead of the entire stack above when the turn is out-of-character:
- `ooc_system.md`

Not part of the story stack:
- `nsfw.md` — read verbatim by `ContextManager.get_explicit()` and
  injected as `{{ additional_content }}`. **Not** template-processed.
- `ooc_human.md` — currently unreferenced; `canon_human.md` carries the
  `<OOC_INSTRUCTIONS>` block.
- `polish_system.md`, `polish_human.md` — the polisher pass.

### Story control stack

`PromptManager.story_stack()` returns the ordered control names for a turn:

- a normal turn: `STORY_HEAD` + this turn's moods + `STORY_TAIL`
  + `addendum` (only if `additional_content` is set) + `checklist`
- an OOC turn: `['ooc']` and nothing else

### Control headers

A control file may describe itself with an HTML comment at the top. The
header is stripped before the text reaches the LLM.

```
<!-- control
desc: A fight, or violence that has already started. Steel out, blood in the air.
replaces: plot, initiation
-->
```

- `desc` — one line. **Only a control with a `desc` is offered to the
  tagging LLM**, so core files stay out of the menu just by omitting the
  header.
- `replaces` — comma-separated core controls this one makes redundant,
  restricted to `REPLACEABLE` = `echo, style, npc, world, plot, initiation`.
  `canon`, `agency`, and `camera` are invariants and can never be dropped.

So a tense scene never carries `PLOT_ADVANCEMENT` instructions telling it
to create pressure — it is already inside the plot.

### Moods (`mood_*_system.md`)

47 moods. Each declares a `family:` in its header, plus a `desc` whose
closing clause is a "use when …" trigger:

| family | n | moods |
|---|---|---|
| `danger` | 8 | ambush, combat, duel, hunt, pursuit, siege, stealth, tense |
| `fallout` | 7 | aftermath, betrayal, consequence, desperation, grief, horror, mercy |
| `schemes` | 6 | deception, escape, heist, interrogation, mystery, recruitment |
| `society` | 12 | first_contact, formal, gambling, negotiation, plea, reunion, sedition, social, teaching, trade, trial, war_council |
| `quiet` | 8 | awe, celebration, downtime, dream, exploration, travel, vigil, weather |
| `interior` | 6 | anger, compulsion, courage, intimacy, intoxication, reverie |

Every mood is a draft meant to be edited — behavioural rules only, with no
invented lore. Adding one needs **no code change**: drop in
`mood_X_system.md` with `desc:` and `family:` headers and it joins that
family's shortlist.

Family names and blurbs live in `MOOD_FAMILIES` (`src/prompt_manager.py`),
because the router needs a description of the *group*, not its members. A
family with no moods on disk is never offered.

### How the stack is chosen (per turn)

1. **Router.** `ContextManager._route_mood_families()` makes one small
   pre-LLM call against `story/pre_processor/mood_router_human.md` (~1.6KB)
   offering only the six families, and reads back `{"families": [...]}`. A
   ~2B model choosing among six groups is far more reliable than the same
   model choosing from 47 moods, and the call that follows then sees a
   short menu. A failed, empty, or unknown-family route falls back to the
   full menu: routing can never break a turn or silently disable moods.
2. **Tagging.** `pre_processor()` renders
   `story/pre_processor/tagging_human.md` (human prompt only — many light
   LLMs do not support a system prompt) with `documents['mood_menu']`
   narrowed to the routed families. The menu is supplied **only for
   `direction='query'`**; reply and import pass an empty menu, which
   Jinja-gates the whole PROMPT_STACK section out (~5KB saved) and stops
   the tagger emitting a stack nothing reads. The AI reply never picks
   controls.
   `documents['tag_direction']` (`query` | `response` | `import`) is passed
   too: the prompt uses it to state that on a `response` pass `INPUT_TEXT`
   is the model's **own narration**, so the PC cannot have moved. That is
   what stops a room merely mentioned in prose (a cabin door opening) from
   relocating the player. `SceneManager` enforces the same rule from its
   side — a location only changes with `moving_confidence > 0.7`.
3. The tagger returns `{"metadata": {…}, "prompt_stack": [mood_x, …]}`.
   `prompt_stack` is a **sibling** of `metadata`, never inside it: a
   metadata key becomes a `RAGTag`, so it would be written to Chroma and
   merged into the scene file. `CommonUtils.get_prompt_stack()` reads only
   the sibling key.
4. `ContextManager._tag_user_query()` stores the parsed list on
   `documents['prompt_stack']`.
5. `compose_story_plot()` validates those names against the **full** menu
   (a typo or a core name is ignored, never fatal), applies `replaces`,
   and assembles the stack.

### OOC turns

Detected from the same `OOC:` / `SYSTEM:` / `OOC>` prefix that
`save_response()` uses to keep the turn out of history and RAG. The story
stack is dropped entirely, so the model is never handed rules for a mode
it is not in — the old `If OOC_MODE = TRUE` conditional prose is gone;
routing *is* the conditional.

The OOC reply is stashed in `documents` and returned on the **next** turn
as `ooc_diagnostics`, rendered in `canon_human.md` under
`<OOC_INSTRUCTIONS>`.

### `assistant/pre_processor/`
- `tagging_human.md` — metadata extraction (assistant flavor).
- `tagging_images.md`, `tagging_files.md`, `tagging_sd.md` — appended by
  turn signals: pixels attached (`HAS_IMAGE`), text paperclips
  (`HAS_FILES`), or a generated picture already on screen.
- `preconditioner_system.md`, `preconditioner_human.md` — reserved.

### `assistant/heavy/`
- `canon_system.md`, `canon_human.md` — the assistant spine.
- `need_gold.md` — how to emit `<NEED_GOLD:file>` from `DOCUMENTS_INDEX`.
- `need_search.md` — how to emit `<NEED_SEARCH:query>` for live facts.
- `search.md` — this turn already ran a live lookup; the `WEB_SEARCH`
  block is the model's own knowledge.
- `search_resume.md` — prepended on a `NEED_SEARCH` relaunch.
- `resume.md` — prepended on a `NEED_GOLD` relaunch (cookbook omitted so
  the model does not copy the example tag).
- `images.md` — pixels are attached this turn.
- `sd_last.md` — a generated picture is already on screen.

`compose_assistant_plot()` in `src/prompt_manager.py` decides which heavy
fragments are appended; `compose_assistant_tag()` does the same for the
tagging fragments.

## Overlays (user edits)

Edit prompts from the Spur prompt editor, or drop a file into:

```
<vector_dir>/prompt_overrides/<flavor>/<role>/<slot>.md
```

The overlay path **mirrors** the stock tree, so a story slot and an
assistant slot may share a filename without colliding. Repo templates are
never modified by the editor — `restore` simply deletes the overlay.
`PromptManager.overlay_path()` builds the mirror, and `optional_slot()`
resolves overlay → stock → `''`.

## Adding a prompt

1. Drop the file into the right `flavor/role/` tree. No code change is
   needed for the manifest to see it (`PromptManager.slots()`).
2. Read it with `pm.slot(role, name)` when it is required, or
   `pm.optional_slot(role, name)` when it may be absent.
3. Never read `prompts/...` paths directly — always go through the
   resolver so overlays and missing-file handling stay consistent.
