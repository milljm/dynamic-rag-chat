""" An inherited class for handling prompts """
import os
import sys
# ChatOptions is a type hint only — avoid importing chat_utils at module load.
try:
    from .sd_client import has_generated_images
except ImportError:
    from sd_client import has_generated_images

# pylint: disable=no-member  # (build_prompts construct members on class init)
class PromptManager():
    """
    ### PromptManager

    Load, overlay, and compose the ``prompts/*.md`` templates. Story
    picks a model-matched file (gemma/llama/qwen/…); assistant always
    uses the nostory pair. Spur edits write overlays under
    ``vector_dir/prompt_overrides`` so repo templates stay intact.

    *Class init args:*
        .. code-block:: python
            console: Console
            current_dir: str           # repo root (prompts/ lives here)
            args: ChatOptions
            prompt_model: str = 'default'  # --model / --pre-llm name

    *Usage:*
        - construct (also called by ContextManager / RenderWindow):
            .. code-block:: python
                pm = PromptManager(console, current_dir, args, prompt_model)

        - plot prompt for the live LLM:
            .. code-block:: python
                system, human = pm.compose_nostory_plot(documents)

        - Spur editor:
            .. code-block:: python
                slot = pm.read_plot('assistant', 'system')
                pm.write_plot('assistant', 'system', text)
                pm.restore_plot('assistant', 'system')
    """
    def __init__(self, console, current_dir, args, prompt_model: str = 'default'):
        self.console = console
        self.assistant_prompt = args.assistant_mode
        self.args = args
        self.debug = args.debug
        self.prompt_model = prompt_model
        self.model = self._match_model(prompt_model)
        self.current_dir = current_dir
        # instance build_prompts to kick start member availability
        self.build_prompts()

    def _match_model(self, model: str)->str:
        """ attempt to match model, default to 'default' """
        if self.args.assistant_mode:
            return 'nostory'
        supported = ['gemma', 'llama', 'qwen', 'deepseek', 'mixtral']
        return next((x for x in supported if x in model.lower()), 'default')

    def build_prompts(self) -> None:
        """
        A way to manage a growing number of prompt templates
        {key : value} pairs become self.key_* : contents-of-file
        file naming convention: {value}_system.md / {value}_human.md
        """
        self.model = self._match_model(self.prompt_model)
        prompt_files = {
            'ooc_prompt'    :f'ooc_{self.model}',
            'pre_prompt'    :f'pre_conditioner_prompt_{self.model}',
            'tag_prompt'    :f'tagging_prompt_{self.model}',
            'plot_prompt'   :f'plot_prompt_{self.model}',
            'polish_prompt' :f'polish_prompt_{self.model}',
            'entity_prompt' :f'entity_prompt_{self.model}',
        }
        for prompt_key, prompt_base in prompt_files.items():
            prompt_dir = os.path.join('prompts', prompt_base)
            setattr(self, f'{prompt_key}_file', os.path.join(self.current_dir, prompt_dir))
            setattr(self, f'{prompt_key}_system',
                    self.get_prompt(f'{prompt_dir}_system.md'))
            setattr(self, f'{prompt_key}_human',
                    self.get_prompt(f'{prompt_dir}_human.md'))

    def overlay_root(self) -> str:
        """User prompt edits live here so repo templates stay intact."""
        vd = getattr(self.args, 'vector_dir', None) or ''
        if not vd:
            return ''
        return os.path.join(os.path.abspath(vd), 'prompt_overrides')

    def overlay_path(self, stock_path: str) -> str:
        root = self.overlay_root()
        if not root or not stock_path:
            return ''
        return os.path.join(root, os.path.basename(stock_path))

    def reload(self) -> None:
        """Re-read prompt files from disk (Spur editor / live edits)."""
        self.build_prompts()

    def plot_file(self, flavor: str, kind: str) -> str:
        """Absolute path for the plot system/human file Spur should edit.

        Assistant always uses the nostory pair. Story uses the model-matched
        file when it exists, otherwise plot_prompt_default_*.md.
        Independent of the live assistant_mode flag so the editor can open
        the other flavor without switching branches.
        """
        if flavor not in ('assistant', 'story') or kind not in ('system', 'human'):
            raise ValueError('Unknown prompt slot')
        if flavor == 'assistant':
            stem = 'plot_prompt_nostory'
        else:
            model = next(
                (name for name in ('gemma', 'llama', 'qwen', 'deepseek', 'mixtral')
                 if name in (self.prompt_model or '').lower()),
                'default',
            )
            stem = f'plot_prompt_{model}'
        path = os.path.abspath(
            os.path.join(self.current_dir, 'prompts', f'{stem}_{kind}.md')
        )
        if os.path.isfile(path):
            return path
        if flavor == 'story' and 'default' not in stem:
            fallback = os.path.abspath(
                os.path.join(self.current_dir, 'prompts', f'plot_prompt_default_{kind}.md')
            )
            if os.path.isfile(fallback):
                return fallback
        return path

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

    def get_prompt(self, path):
        """ Keep the prompts as files for easier manipulation """
        self.model = self._match_model(self.prompt_model)
        overlay = self.overlay_path(path)
        if overlay and os.path.isfile(overlay):
            with open(overlay, 'r', encoding='utf-8') as prompt:
                return prompt.read()
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as prompt:
                return prompt.read()
        elif path.find('default') == -1:
            default_path = path.replace(self.model, 'default')
            return self.get_prompt(default_path)
        print(f'Prompt not found! I expected to find it at:\n\n\t{path}')
        sys.exit(1)

    def compose_nostory_plot(self, documents: dict) -> tuple[str, str]:
        """Spine + event fragments. Resume turns do not see the NEED_GOLD cookbook."""
        spine = self.get_prompt(f'{self.plot_prompt_file}_system.md')
        human = self.get_prompt(f'{self.plot_prompt_file}_human.md')
        resume = bool(str(documents.get('gold_resume') or '').strip())
        has_index = bool(documents.get('has_documents_index'))
        has_images = bool(
            documents.get('has_images') or documents.get('dynamic_images')
        )
        parts: list[str] = []
        if resume:
            parts.append(self.get_prompt(f'{self.plot_prompt_file}_resume.md'))
        parts.append(spine)
        if has_images:
            extra = f'{self.plot_prompt_file}_images.md'
            if os.path.exists(extra):
                parts.append(self.get_prompt(extra))
        if has_index and not resume:
            parts.append(self.get_prompt(f'{self.plot_prompt_file}_need_gold.md'))
        searched = (
            int(documents.get('agent_calls') or 0) > 0
            or 'WEB_SEARCH' in str(documents.get('dynamic_files') or '')
            or 'AGENT_TOOL_RESULT' in str(documents.get('dynamic_files') or '')
        )
        if searched:
            extra = f'{self.plot_prompt_file}_search.md'
            if os.path.exists(extra):
                parts.append(self.get_prompt(extra))
        if documents.get('has_last_image') or has_generated_images(
                getattr(self.args, 'vector_dir', '') or ''
        ):
            extra = f'{self.plot_prompt_file}_sd_last.md'
            if os.path.exists(extra):
                parts.append(self.get_prompt(extra))
        return '\n'.join(parts), human

    def compose_nostory_tag(self, documents: dict) -> str:
        """Tagging human prompt. Attach fragments only when files/pixels exist."""
        human = self.get_prompt(f'{self.tag_prompt_file}_human.md')
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
            extra = f'{self.tag_prompt_file}_images.md'
        elif has_files:
            extra = f'{self.tag_prompt_file}_files.md'
        if extra and os.path.exists(extra):
            human = human.rstrip() + '\n' + self.get_prompt(extra)
        if has_generated_images(getattr(self.args, 'vector_dir', '') or ''):
            sd_tag = f'{self.tag_prompt_file}_sd.md'
            if os.path.exists(sd_tag):
                human = human.rstrip() + '\n' + self.get_prompt(sd_tag)
        return human
