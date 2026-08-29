""" An inherited class for handling prompts """
import os
import sys
from .chat_utils import ChatOptions # For Type Hinting

class PromptManager():
    """
    Handle all the possible prompt files we may introduce with RAG/Tagging

    Most can be handled by a default. But This class is here so we can support possibly\n
    more nuanced LLMs.
    """
    def __init__(self, console, current_dir, args: ChatOptions, prompt_model: str = 'default'):
        self.console = console
        self.assistant_prompt = args.assistant_mode
        self.args = args
        self.debug = args.debug
        self.prompt_model = prompt_model
        self.model = self._match_model(prompt_model)
        self.current_dir = current_dir

    def _match_model(self, model: str)->str:
        """ attempt to match model, default to 'default' """
        if self.args.assistant_mode:
            return 'nostory'
        supported = ['gemma', 'llama', 'qwen', 'deepseek', 'mixtral']
        return next((x for x in supported if x in model.lower()), 'default')

    def build_prompts(self):
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

    def get_prompt(self, path):
        """ Keep the prompts as files for easier manipulation """
        self.model = self._match_model(self.prompt_model)
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
        parts: list[str] = []
        if resume:
            parts.append(self.get_prompt(f'{self.plot_prompt_file}_resume.md'))
        parts.append(spine)
        if has_index and not resume:
            parts.append(self.get_prompt(f'{self.plot_prompt_file}_need_gold.md'))
        return '\n'.join(parts), human
