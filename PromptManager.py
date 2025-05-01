""" An inherited class for handling prompts """
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
class PromptManager():
    """ Handle all the possible prompt files we may introduce with RAG/Tagging """
    def __init__(self, console, debug=False):
        self.console = console
        self.debug = debug

    def build_prompts(self):
        """
        A way to manage a growing number of prompt templates dynamic RAG/Tagging
        might introduce...

          {key : value} pairs become self.key_* : contents-of-file
          filenaming convention: {value}_system.txt / {value}_human.txt
        """
        if self.debug:
            self.console.print('[italic dim grey50]Debug mode enabled. I will re-read the '
                          'prompt files each time[/]\n')
        prompt_files = {
            'pre_prompt':  'pre_conditioner_prompt',
            'tag_prompt': 'tagging_prompt',
            'plot_prompt': 'plot_prompt'
        }
        for prompt_key, prompt_base in prompt_files.items():
            prompt_dir = os.path.join('prompts', prompt_base)
            setattr(self, f'{prompt_key}_file', os.path.join(current_dir, prompt_dir))
            setattr(self, f'{prompt_key}_system', self.get_prompt(f'{prompt_dir}_system.txt'))
            setattr(self, f'{prompt_key}_human', self.get_prompt(f'{prompt_dir}_human.txt'))

    def get_prompt(self, path):
        """ Keep the prompts as files for easier manipulation """
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as prompt:
                return prompt.read()
        else:
            print(f'Prompt not found! I expected to find it at:\n\n\t{path}')
            sys.exit(1)
