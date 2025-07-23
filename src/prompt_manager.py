""" An inherited class for handling prompts """
import os
import sys
from .chat_utils import ChatOptions # For Type Hinting

class PromptManager():
    """
    Handle all the possible prompt files we may introduce with RAG/Tagging

    Most can be handled by a default. But This class is here so we can support possibly\n
    more naunced LLMs.
    """
    def __init__(self, console, current_dir, args: ChatOptions, prompt_model: str = 'default'):
        self.console = console
        self.assistant_prompt = args.assistant_mode
        self.debug = args.debug
        self.model = self._match_model(prompt_model)
        self.current_dir = current_dir

    def _match_model(self, model: str)->str:
        """ attempt to match model, default to 'default' """
        if self.assistant_prompt:
            return 'nostory'
        supported = ['gemma', 'llama', 'qwen', 'deepseek', 'mixtral']
        return next((x for x in supported if x in model.lower()), 'default')

    def build_prompts(self):
        """
        A way to manage a growing number of prompt templates
        {key : value} pairs become self.key_* : contents-of-file
        filenaming convention: {value}_system.md / {value}_human.md
        """
        prompt_files = {
            'pre_prompt'  :f'pre_conditioner_prompt_{self.model}',
            'tag_prompt'  :f'tagging_prompt_{self.model}',
            'plot_prompt' :f'plot_prompt_{self.model}'
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
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as prompt:
                return prompt.read()
        elif path.find('default') == -1:
            default_path = path.replace(self.model, 'default')
            return self.get_prompt(default_path)
        print(f'Prompt not found! I expected to find it at:\n\n\t{path}')
        sys.exit(1)
