""" An inherited class for handling prompts """
import os
import sys
class PromptManager():
    """ Handle all the possible prompt files we may introduce with RAG/Tagging """
    def __init__(self, console, current_dir, model: str = 'default', debug=False):
        self.console = console
        self.debug = debug
        self.model = self._match_model(model)
        self.current_dir = current_dir

    @staticmethod
    def _match_model(model: str)->str:
        """ attempt to match model, default to 'default' """
        supported = ['gemma', 'llama', 'qwen', 'deepseek']
        return next((x for x in supported if x in model.lower()), 'default')

    def build_prompts(self):
        """
        A way to manage a growing number of prompt templates
        {key : value} pairs become self.key_* : contents-of-file
        filenaming convention: {value}_system.txt / {value}_human.txt
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
                    self.get_prompt(f'{prompt_dir}_system.txt'))
            setattr(self, f'{prompt_key}_human',
                    self.get_prompt(f'{prompt_dir}_human.txt'))

    def get_prompt(self, path):
        """ Keep the prompts as files for easier manipulation """
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as prompt:
                return prompt.read()
        else:
            print(f'Prompt not found! I expected to find it at:\n\n\t{path}')
            sys.exit(1)
