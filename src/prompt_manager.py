""" An inherited class for handling prompts """
import os
import re
import sys
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
# ChatOptions is a type hint only — avoid importing chat_utils at module load.
try:
    from .sd_client import has_generated_images
    from .search_fetch import MAX_SEARCH_FETCHES
except ImportError:
    from sd_client import has_generated_images
    from search_fetch import MAX_SEARCH_FETCHES

FLAVORS = ('story', 'assistant')
ROLES = ('pre_processor', 'heavy')
KINDS = ('system', 'human')

# Story control slots (bare names → ``heavy/<name>_system.md``), in send order.
# Turn-selected controls (moods) splice in where STORY_HEAD ends, right after
# NPC_BEHAVIOR; ADDITIONAL_CONTENT and the checklist close the stack.
STORY_HEAD = ('canon', 'agency', 'camera', 'echo', 'style', 'npc')
STORY_TAIL = ('world', 'plot', 'initiation')
STORY_OOC = ('ooc',)
_CONTROL_SPLIT = re.compile(r'[,;\s]+')

# Core controls a mood may displace. canon / agency / camera are the
# invariants the whole system leans on, so they are never droppable.
REPLACEABLE = ('echo', 'style', 'npc', 'world', 'plot', 'initiation')

# Optional self-description at the top of a control file:
#   <!-- control
#   desc: one line, shown to the tagging LLM
#   replaces: plot, initiation
#   -->
# Only a control that declares a ``desc`` is offered to the tagger, so core
# files stay out of the menu just by omitting the header.
_CONTROL_HEADER = re.compile(
    r'\A\s*<!--\s*control\b(?P<body>.*?)-->\s*', re.DOTALL
)

# Mood families, used by the router that narrows the menu before the
# metadata tagger sees it. Names must match the ``family:`` header on the
# mood files; a family with no moods on disk is never offered.
MOOD_FAMILIES = (
    ('danger', 'Violence, pursuit, and physical risk — a fight, a chase, '
               'a hunt, a siege, hiding, a standoff, an ambush, a duel.'),
    ('fallout', 'Paying for what already happened — aftermath, betrayal, '
                'desperation, grief, horror, mercy, a debt called in.'),
    ('schemes', 'Working an angle — deception, heists, interrogation, '
                'investigation, escaping, recruiting.'),
    ('society', 'People and power — company, etiquette, deals, judgement, '
                'bargains, unrest, instruction, games of chance.'),
    ('quiet', 'Calm and scale — rest, celebration, travel, discovery, '
              'wonder, dreams, weather, a long watch.'),
    ('interior', 'Inside the player — anger, compulsion, intimacy, '
                 'intoxication, doubt, resolve.'),
)


class PromptManager():
    """
    ### PromptManager

    Load, overlay, and compose the ``prompts/<flavor>/<role>/*.md``
    templates. The directory tree *is* the manifest: both flavors own a
    ``pre_processor`` tree (light LLM: tagging, entity, preconditioner)
    and a ``heavy`` tree (main LLM: canon, ooc, moods, polish,
    fragments). Spur edits write overlays under
    ``vector_dir/prompt_overrides`` mirroring that tree, so repo
    templates stay intact.

    *Class init args:*
        .. code-block:: python
            console: Console
            current_dir: str      # repo root (prompts/ lives here)
            args: ChatOptions     # args.assistant_mode picks the flavor

    *Usage:*
        - construct (also called by ContextManager / RenderWindow):
            .. code-block:: python
                pm = PromptManager(console, current_dir, args)

        - a required prompt for the live LLM:
            .. code-block:: python
                system = pm.slot('heavy', 'canon_system')

        - an optional fragment ('' when absent):
            .. code-block:: python
                extra = pm.optional_slot('heavy', 'need_search')

        - plot prompt for the live LLM:
            .. code-block:: python
                system, human = pm.compose_assistant_plot(documents)

        - Spur editor:
            .. code-block:: python
                slot = pm.read_plot('assistant', 'system')
                pm.write_plot('assistant', 'system', text)
                pm.restore_plot('assistant', 'system')
    """
    def __init__(self, console, current_dir, args):
        self.console = console
        self.assistant_prompt = args.assistant_mode
        self.args = args
        self.debug = args.debug
        self.current_dir = current_dir
        self.manifest: dict[str, dict[str, list[str]]] = {}
        # instance build_prompts to kick start member availability
        self.build_prompts()

    def flavor(self) -> str:
        """Live flavor. Story unless ``args.assistant_mode`` is set."""
        return 'assistant' if self.args.assistant_mode else 'story'

    def _prompt_dir(self, flavor: str, role: str) -> str:
        """Absolute directory of one ``prompts/<flavor>/<role>`` tree."""
        if flavor not in FLAVORS:
            raise ValueError(f'Unknown prompt flavor: {flavor}')
        if role not in ROLES:
            raise ValueError(f'Unknown prompt role: {role}')
        return os.path.abspath(
            os.path.join(self.current_dir, 'prompts', flavor, role)
        )

    def _stock_path(self, flavor: str, role: str, slot: str) -> str:
        """Absolute path of a shipped slot file."""
        return os.path.join(self._prompt_dir(flavor, role), f'{slot}.md')

    def build_prompts(self) -> None:
        """
        Discover the slot manifest from disk.

        ``prompts/<flavor>/<role>/*.md`` becomes
        ``self.manifest[flavor][role] = [slot, ...]``. Contents are read
        on demand so Spur edits are live; this only indexes the tree.
        """
        self.manifest = {
            flavor: {role: self._discover(flavor, role) for role in ROLES}
            for flavor in FLAVORS
        }

    def _discover(self, flavor: str, role: str) -> list[str]:
        """Slot names (filename stems) present in one flavor/role tree."""
        try:
            names = os.listdir(self._prompt_dir(flavor, role))
        except OSError:
            return []
        return sorted(
            name[:-3] for name in names
            if name.endswith('.md') and not name.startswith('.')
        )

    def slots(self, role: str, flavor: str | None = None) -> list[str]:
        """Selectable slot names for a role (live flavor by default)."""
        return list(self.manifest.get(flavor or self.flavor(), {}).get(role, []))

    @staticmethod
    def _parse_control_header(text: str) -> dict[str, str]:
        """Parse a control file's optional ``<!-- control … -->`` header.

        Keys are lowercased; values keep their case. A file with no header
        (every core control) yields ``{}``.
        """
        match = _CONTROL_HEADER.match(text or '')
        if not match:
            return {}
        meta: dict[str, str] = {}
        for line in match.group('body').splitlines():
            key, sep, value = line.partition(':')
            if sep and value.strip():
                meta[key.strip().lower()] = value.strip()
        return meta

    @staticmethod
    def _strip_control_header(text: str) -> str:
        """Drop the control header so it never reaches the LLM."""
        return _CONTROL_HEADER.sub('', text or '', count=1)

    def _control_meta(self, slot: str, role: str = 'heavy',
                     flavor: str | None = None) -> dict[str, str]:
        """Header metadata for one control slot (overlay wins)."""
        return self._parse_control_header(self.optional_slot(role, slot, flavor))

    def mood_menu(self, families: list[str] | None = None,
                  flavor: str | None = None) -> list[tuple[str, str]]:
        """Selectable ``(name, description)`` pairs for the tagging LLM.

        A control is offered when it declares a ``desc`` and is not a
        reserved core/one-shot name. Pass ``families`` to narrow the list
        to the router's choice. Drop in ``mood_x_system.md`` with a header
        and the tagger can pick it — no code change.
        """
        reserved = set(STORY_HEAD) | set(STORY_TAIL) | set(STORY_OOC)
        wanted = set(families) if families else None
        menu: list[tuple[str, str]] = []
        for slot in self.slots('heavy', flavor):
            if not slot.endswith('_system'):
                continue
            name = slot[:-len('_system')]
            if name in reserved:
                continue
            meta = self._control_meta(slot, 'heavy', flavor)
            desc = meta.get('desc', '')
            if not desc:
                continue
            if wanted is not None and meta.get('family', '') not in wanted:
                continue
            menu.append((name, desc))
        return menu

    def mood_family_menu(self, flavor: str | None = None) -> list[tuple[str, str]]:
        """Families that actually have moods on disk, with their blurbs.

        The router picks from this short list; only the chosen families'
        moods are then offered to the metadata tagger.
        """
        present = {
            self._control_meta(slot, 'heavy', flavor).get('family', '')
            for slot in self.slots('heavy', flavor)
            if slot.endswith('_system')
        }
        return [(name, desc) for name, desc in MOOD_FAMILIES if name in present]

    def compose_mood_router(self, documents: dict) -> list:
        """Formatted messages for the mood router's family pick.

        Returns ``[]`` when the router prompt is absent, so a missing
        optional file degrades to "offer every mood" instead of exiting.
        Note ``PromptTemplate`` has no ``format_messages``; building it the
        other way raised AttributeError inside a bare except, which once
        silently disabled routing with no log to show for it.
        """
        template = self.optional_slot('pre_processor', 'mood_router_human')
        if not template.strip():
            return []
        tmpl = PromptTemplate(template=template, template_format='jinja2')
        return ChatPromptTemplate.from_messages(
            [HumanMessagePromptTemplate(prompt=tmpl)]
        ).format_messages(**documents)

    def compose_tagging_messages(self, documents: dict,
                                 direction: str = 'query',
                                 menu=None) -> list:
        """Pre-processor chat messages for one tagging run.

        Owned here with the rest of prompt composition, so the tagging call
        can be tested without building a ContextManager. Pass ``menu`` as
        None/[] for the reply and import directions: an empty menu
        Jinja-gates the PROMPT_STACK section out of the tagging prompt.

        Sets ``tag_direction`` and ``mood_menu`` on ``documents`` **in
        place** — the caller formats with that same dict.
        """
        if self.args.assistant_mode:
            tmpl = PromptTemplate(
                template=self.compose_assistant_tag(documents),
                template_format='jinja2',
            )
            return [HumanMessagePromptTemplate(prompt=tmpl)]
        documents['tag_direction'] = direction
        documents['mood_menu'] = list(menu or [])
        tmpl = PromptTemplate(
            template=self.slot('pre_processor', 'tagging_human'),
            template_format='jinja2',
        )
        messages = [HumanMessagePromptTemplate(prompt=tmpl)]
        system_msg = self._tagging_system_message()
        if system_msg is not None:
            messages.insert(0, system_msg)
        return messages

    def _tagging_system_message(self):
        """Optional tagging system prompt (story only, usually absent).

        Many light LLMs support a human prompt and nothing else, so
        ``tagging_system.md`` ships empty and this returns ``None``.
        """
        system_prompt = self.optional_slot('pre_processor', 'tagging_system')
        if not system_prompt or not system_prompt.strip():
            return None
        return SystemMessagePromptTemplate(prompt=PromptTemplate(
            template=system_prompt, template_format='jinja2',
        ))

    def reload(self) -> None:
        """Re-scan the prompt tree (Spur editor / live edits)."""
        self.build_prompts()

    def overlay_root(self) -> str:
        """User prompt edits live here so repo templates stay intact."""
        vd = getattr(self.args, 'vector_dir', None) or ''
        if not vd:
            return ''
        return os.path.join(os.path.abspath(vd), 'prompt_overrides')

    def _prompts_root(self) -> str:
        """Absolute ``prompts/`` directory the slot tree lives under."""
        return os.path.abspath(os.path.join(self.current_dir, 'prompts'))

    def overlay_path(self, stock_path: str) -> str:
        """Overlay path mirroring the stock path under ``prompts/``.

        ``prompt_overrides/<flavor>/<role>/<slot>.md`` (not just the
        basename) so a story slot and an assistant slot of the same name
        cannot collide.
        """
        root = self.overlay_root()
        if not root or not stock_path:
            return ''
        rel = os.path.relpath(os.path.abspath(stock_path), self._prompts_root())
        if rel.startswith(os.pardir):
            rel = os.path.basename(stock_path)
        return os.path.join(root, rel)

    def read_prompt(self, path: str) -> str | None:
        """Overlay then stock contents, or None when neither exists."""
        overlay = self.overlay_path(path)
        if overlay and os.path.isfile(overlay):
            with open(overlay, 'r', encoding='utf-8') as prompt:
                return prompt.read()
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as prompt:
                return prompt.read()
        return None

    def get_prompt(self, path: str) -> str:
        """ Required prompt file (overlay wins). Exits when missing """
        content = self.read_prompt(path)
        if content is not None:
            return content
        print(f'Prompt not found! I expected to find it at:\n\n\t{path}')
        sys.exit(1)

    def slot(self, role: str, slot: str, flavor: str | None = None) -> str:
        """Read a required slot for the live (or given) flavor."""
        return self.get_prompt(
            self._stock_path(flavor or self.flavor(), role, slot)
        )

    def optional_slot(self, role: str, slot: str, flavor: str | None = None) -> str:
        """Read an optional fragment. '' when neither overlay nor stock exists."""
        content = self.read_prompt(
            self._stock_path(flavor or self.flavor(), role, slot)
        )
        return content if content is not None else ''

    def plot_file(self, flavor: str, kind: str) -> str:
        """Absolute path for the plot system/human file Spur should edit.

        Independent of the live assistant_mode flag so the editor can open
        the other flavor without switching branches.
        """
        if flavor not in FLAVORS or kind not in KINDS:
            raise ValueError('Unknown prompt slot')
        return self._stock_path(flavor, 'heavy', f'canon_{kind}')

    def read_plot(self, flavor: str, kind: str) -> dict:
        """Stock file plus optional overlay contents for the Spur editor."""
        stock = self.plot_file(flavor, kind)
        overlay = self.overlay_path(stock)
        if overlay and os.path.isfile(overlay):
            with open(overlay, 'r', encoding='utf-8') as handle:
                content = handle.read()
            return {
                'stock': stock,
                'path': overlay,
                'overlaid': True,
                'content': content,
            }
        if not os.path.isfile(stock):
            raise FileNotFoundError(stock)
        with open(stock, 'r', encoding='utf-8') as handle:
            content = handle.read()
        return {
            'stock': stock,
            'path': stock,
            'overlaid': False,
            'content': content,
        }

    def write_plot(self, flavor: str, kind: str, content: str) -> str:
        """Write an overlay. Never clobbers the shipped template."""
        if kind != 'system':
            raise ValueError('Human prompt is not editable')
        stock = self.plot_file(flavor, kind)
        overlay = self.overlay_path(stock)
        if not overlay:
            raise RuntimeError('No vector_dir for prompt overrides')
        os.makedirs(os.path.dirname(overlay), exist_ok=True)
        with open(overlay, 'w', encoding='utf-8') as handle:
            handle.write(content)
        return overlay

    def restore_plot(self, flavor: str, kind: str) -> dict:
        """Drop the overlay so the shipped template is used again."""
        stock = self.plot_file(flavor, kind)
        overlay = self.overlay_path(stock)
        if overlay and os.path.isfile(overlay):
            os.remove(overlay)
        return self.read_plot(flavor, kind)

    @staticmethod
    def is_ooc(documents: dict) -> bool:
        """True when this turn is an out-of-character aside.

        ``ooc_mode_bool`` is set per turn by RenderWindow from the same
        prefix that drops OOC turns out of history and RAG.
        """
        return str(documents.get('ooc_mode_bool') or '').strip().upper() == 'TRUE'

    def _selected_controls(self, documents: dict) -> list[str]:
        """Turn-requested controls, validated against the on-disk menu.

        ``documents['prompt_stack']`` takes a list or a separated string.
        Only controls that advertise themselves (a ``desc`` header) are
        selectable, so a typo — or a core name — is ignored, not fatal.
        """
        raw = documents.get('prompt_stack') or []
        if isinstance(raw, str):
            raw = _CONTROL_SPLIT.split(raw)
        offered = {name for name, _ in self.mood_menu()}
        chosen: list[str] = []
        for name in raw:
            key = str(name).strip().lower()
            if key in offered and key not in chosen:
                chosen.append(key)
        return chosen

    def _replaced_controls(self, documents: dict) -> set[str]:
        """Core controls the selected moods displace this turn.

        A mood declares ``replaces: plot, initiation`` when it makes those
        instructions redundant — a tense scene is already deep in a plot,
        so PLOT_ADVANCEMENT is just noise.
        """
        dropped: set[str] = set()
        for name in self._selected_controls(documents):
            raw = self._control_meta(f'{name}_system').get('replaces', '')
            for target in _CONTROL_SPLIT.split(raw):
                if target in REPLACEABLE:
                    dropped.add(target)
        return dropped

    @staticmethod
    def is_explicit_turn(documents: dict) -> bool:
        """True when the pre-processor rated this turn explicit.

        ``explicit`` comes from ``FilterBuilder.tags_are_nsfw`` (a
        content_rating or scene_mode of ``nsfw``). The NSFW addendum must
        never ride along on a SFW turn, however much text the file holds.
        """
        if str(documents.get('explicit') or '').strip().lower() == 'true':
            return True
        return str(documents.get('content_rating') or '').strip().lower() == 'nsfw'

    def story_stack(self, documents: dict) -> list[str]:
        """Ordered bare control names for this story turn.

        OOC turns get the OOC protocol and nothing else: no rules for a
        mode the model is not in. Everything else gets the surviving core
        behaviour, the selected moods spliced in after NPC_BEHAVIOR, then
        the checklist last.
        """
        if self.is_ooc(documents):
            return list(STORY_OOC)
        dropped = self._replaced_controls(documents)
        stack = [
            *[name for name in STORY_HEAD if name not in dropped],
            *self._selected_controls(documents),
            *[name for name in STORY_TAIL if name not in dropped],
        ]
        if (self.is_explicit_turn(documents)
                and str(documents.get('additional_content') or '').strip()):
            stack.append('addendum')
        stack.append('checklist')
        return stack

    def compose_story_plot(self, documents: dict) -> tuple[str, str]:
        """Story control stack + human payload.

        Each control is an independent ``heavy`` slot, so a turn loads only
        the behaviour it needs and a mood can spend far more tokens than
        the old monolithic canon had room for. Control headers are stripped
        before the text is sent.
        """
        human = self.slot('heavy', 'canon_human')
        parts: list[str] = []
        for name in self.story_stack(documents):
            text = self._strip_control_header(
                self.optional_slot('heavy', f'{name}_system')
            ).strip()
            if text:
                parts.append(text)
        return '\n'.join(parts), human

    def compose_assistant_plot(self, documents: dict) -> tuple[str, str]:
        """Spine + event fragments. Resume turns omit the matching cookbook."""
        spine = self.slot('heavy', 'canon_system')
        human = self.slot('heavy', 'canon_human')
        gold_resume = bool(str(documents.get('gold_resume') or '').strip())
        search_resume = bool(str(documents.get('search_resume') or '').strip())
        has_index = bool(documents.get('has_documents_index'))
        has_images = bool(
            documents.get('has_images') or documents.get('dynamic_images')
        )
        search_used = (
            int(documents.get('search_fetches') or 0)
            + int(documents.get('agent_calls') or 0)
        )
        parts: list[str] = []
        if gold_resume:
            parts.append(self.slot('heavy', 'resume'))
        if search_resume:
            parts.append(self.optional_slot('heavy', 'search_resume'))
        parts.append(spine)
        if has_images:
            parts.append(self.optional_slot('heavy', 'images'))
        if has_index and not gold_resume:
            parts.append(self.slot('heavy', 'need_gold'))
        if not search_resume and search_used < MAX_SEARCH_FETCHES:
            parts.append(self.optional_slot('heavy', 'need_search'))
        searched = (
            search_used > 0
            or 'WEB_SEARCH' in str(documents.get('dynamic_files') or '')
            or 'AGENT_TOOL_RESULT' in str(documents.get('dynamic_files') or '')
        )
        if searched:
            parts.append(self.optional_slot('heavy', 'search'))
        if documents.get('has_last_image') or has_generated_images(
                getattr(self.args, 'vector_dir', '') or ''
        ):
            parts.append(self.optional_slot('heavy', 'sd_last'))
        return '\n'.join(parts), human

    def compose_assistant_tag(self, documents: dict) -> str:
        """Tagging human prompt. Attach fragments only when files/pixels exist."""
        human = self.slot('pre_processor', 'tagging_human')
        has_images = bool(
            documents.get('has_images') or documents.get('dynamic_images')
        )
        has_files = bool(
            documents.get('has_files')
            or documents.get('attached_files_note')
            or documents.get('attachment_texts')
            or documents.get('attached_filenames')
        )
        extra = ''
        if has_images:
            extra = 'tagging_images'
        elif has_files:
            extra = 'tagging_files'
        if extra:
            human = human.rstrip() + '\n' + self.optional_slot('pre_processor', extra)
        if has_generated_images(getattr(self.args, 'vector_dir', '') or ''):
            human = (human.rstrip() + '\n'
                     + self.optional_slot('pre_processor', 'tagging_sd'))
        return human
