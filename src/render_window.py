""" module responsible for rendering output to the screen """
import re
import time
from threading import Thread
import requests
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text
from rich.console import Group
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from .context_manager import ContextManager
from .prompt_manager import PromptManager
class AnimationThread(Thread):
    """ Allow pulsing animation to run as a thread """
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def run(self):
        while self.owner.animation_active:
            self.owner.update_animation()
            time.sleep(0.5)

class RenderWindow(PromptManager):
    """ Responsible for printing Rich Text/Markdown Live to the screen """
    def __init__(self, console, common, current_dir, **kwargs):
        super().__init__(console, current_dir)
        self.debug = kwargs['debug']
        self.console = console
        self.host = kwargs['host']
        self.model = kwargs['model']
        self.num_ctx = kwargs['num_ctx']
        self.verbose = kwargs['verbose']
        self.light_mode = kwargs['light_mode']
        self.color = 245 if self.light_mode else 233
        self.model_re = re.compile(r'(\w+)\W+')
        self.common = common
        self.cm = ContextManager(console, self.common, current_dir, **kwargs)
        self.prompts = PromptManager(console, current_dir, model=self.model, debug=self.debug)
        self.llm = ChatOllama(base_url=self.host,
                              model=self.model,
                              temperature=0.9,
                              repeat_penalty=1.1,
                              num_ctx=self.num_ctx,
                              streaming=True,
                              keep_alive='1h')
        self.prompts.build_prompts()
        self.meta_capture = ''

        # Ollama hot
        self.ollama_ping = False

        # Thinking animation
        self.pulsing_chars = ["⠇", "⠋", "⠙", "⠸", "⠴", "⠦"]
        self.do_once = False
        self.pulse_index = 0
        self.thinking = False
        self.meta_hiding = False
        self.animation_active = False
        self.animation_thread = Thread(target=self.update_animation)

    @staticmethod
    def ollama_keepalive(host):
        """ keep those pesky models warm """
        while True:
            requests.head(f'http://{host}/', timeout=300)
            time.sleep(5)

    @staticmethod
    def response_count(response)->int:
        """
        Attempt to return a token count in response. Caveats: Some models 'think'
        before responding. Allow this response to not count against the token/s
        performance. Make an assumption: Any return should be considered as 1 token
        at minimum. See the for loop in self.stream_response for details why response
        is empty.
        """
        if response:
            return len(response.split())
        return 1

    def hide_reasons(self, chunk_content)->bool:
        """ return bool on reasons to hide token """
        reasons = ['<meta', '<th', '<status']
        for reason in reasons:
            if chunk_content.find(f'>{reason}') != -1 or chunk_content.find(reason) != -1:
                return True
        return False

    def if_hiding(self, chunk, verbose)->object:
        """ hide meta tags from user (unless in verbose mode) """
        if verbose:
            return chunk
        content = str(chunk.content)
        if self.meta_hiding:
            if self.hide_reasons(content):
                self.meta_capture += content
                content = ''
            elif content.find('>') != -1:
                self.meta_hiding = False
                self.stop_thinking()
            self.meta_capture += content
            chunk.content = ''
        elif self.hide_reasons(content):
            self.start_thinking()
            self.meta_hiding = True
            # Capture multiple during this response
            if self.meta_capture:
                self.meta_capture += content
            else:
                self.meta_capture = content
            chunk.content = ''
        return chunk

    def reveal_thinking(self, chunk, show: bool = False)->object:
        """ return thinking chunk if verbose """
        content = chunk.content
        if self.thinking and '</think>' in chunk.content:
            chunk.content = ''
            self.thinking = False
            self.stop_thinking()
        elif not self.thinking and '<think>' in chunk.content:
            self.do_once = True
            self.thinking = True
            self.start_thinking()
            chunk.content = 'AI thinking...'
        elif self.thinking:
            chunk.content = content if show else ''
        return chunk

    def start_thinking(self):
        """ method to start thinking animation """
        if hasattr(self, 'animation_thread') and self.animation_thread.is_alive():
            self.animation_thread.join(timeout=0.1)
        self.animation_active = True
        self.animation_thread = AnimationThread(self)
        self.animation_thread.daemon = True
        self.animation_thread.start()

    def stop_thinking(self):
        """ method to stop thinking animation """
        self.animation_active = False

    def update_animation(self):
        """ a threaded method to run the thinking animation """
        while self.animation_active:
            time.sleep(0.1)  # Adjust speed (0.1 seconds per frame)
            self.pulse_index = (self.pulse_index + 1) % len(self.pulsing_chars)
            self.render_chat()

    # Stream response as chunks
    def stream_response(self, documents: dict):
        """ Parse LLM Prompt """
        prompts = self.prompts
        # pylint: disable=no-member # dynamic prompts (see self.__build_prompts)
        system_prompt = (prompts.get_prompt(f'{prompts.plot_prompt_file}_system.txt')
                     if self.debug else prompts.plot_prompt_system)

        human_prompt = (prompts.get_prompt(f'{prompts.plot_prompt_file}_human.txt')
                     if self.debug else prompts.plot_prompt_human)
        # pylint: enable=no-member
        prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])

        if self.debug:
            self.console.print(f'LLM DOCUMENTS: {documents.keys()}\n'
                               f'{documents["performance"]}\n',
                               style=f'color({self.color})',
                               highlight=False)
        prompt = prompt_template.format_messages(**documents)
        # pylint: enable=no-member
        if self.debug:
            self.console.print(f'HEAVY LLM PROMPT (llm.stream()):\n{prompt}\n\n',
                          style=f'color({self.color})',
                          highlight=False)
        for chunk in self.llm.stream(prompt):
            chunk = self.reveal_thinking(chunk, self.verbose)
            chunk = self.if_hiding(chunk, self.verbose)
            yield chunk

    def render_footer(self, time_taken: float = 0, **kwargs)->Text:
        """ Handle the footer statistics """
        prompt_tokens = kwargs['prompt_tokens']
        token_count = kwargs['token_count']
        cleaned_color = kwargs['cleaned_color']
        token_savings = kwargs['token_savings']
        pre_processing_time = kwargs['pre_process_time']

        # Calculate real-time heat maps
        context_size = [v for k,v in self.common.prompt_map.items() if k<=prompt_tokens][-1:][0]
        produced = [v for k,v in self.common.heat_map.items() if token_count>=k][-1:][0]

        # Calculate Tokens/s
        if time_taken > 0:
            tokens_per_second = token_count / time_taken
        else:
            tokens_per_second = 0

        # Implement a thinking emoji
        emoji = f' {self.pulsing_chars[self.pulse_index]} ' if self.animation_active else ' '

        foot_color = self.color-6 if self.light_mode else self.color
        # Create the footer text with model info, time, and token count
        model = '-'.join(self.model_re.findall(self.model)[:2])
        footer = Text('\n', style=f'color({foot_color})')
        footer.append(f'{model}', style='color(202)')
        footer.append(emoji, style=f'color({12 if self.light_mode else 51})')
        footer.append(f'{time_taken:.2f}', style='color(94)')
        footer.append('s Tokens(deduplication:', style=f'color({foot_color})')
        footer.append(f'{token_savings}', style=f'color({cleaned_color})')
        footer.append(' context:', style=f'color({foot_color})')
        footer.append(f'{prompt_tokens}', style=f'color({context_size})')
        footer.append(f':{pre_processing_time}', style=f'color({foot_color})')
        footer.append(' completion:', style=f'color({foot_color})')
        footer.append(f'{token_count}', style=f'color({produced})')
        footer.append(f') {tokens_per_second:.1f}T/s', style=f'color({foot_color})')
        return footer

    # Compose the full chat display with footer (model name, time taken, token count)
    def render_chat(self, current_stream: str = "")->Text|Markdown:
        """ render and return markdown/syntax """
        if self.thinking and self.verbose:
            chat_content = Text(current_stream, style=f'color({self.color})')
        else:
            chat_content = Markdown(current_stream)

        return chat_content

    def live_stream(self, documents: dict,
                          token_savings: int,
                          prompt_tokens: int,
                          cleaned_color: int,
                          preprocessing)->None:
        """ Handle the Rich Live updating process """
        current_response = ''
        footer_meta = {'token_savings'   : token_savings,
                       'prompt_tokens'   : prompt_tokens,
                       'cleaned_color'   : cleaned_color,
                       'pre_process_time': documents['pre_process_time'],
                       'token_count'     : 0}

        if not self.ollama_ping:
            self.ollama_ping = True
            thread = Thread(target=self.ollama_keepalive,
                            args=(self.host,),
                            daemon=True)
            thread.start()

        start_time = 0
        query = Markdown(f'**You:** {documents["user_query"]}\n\n---\n\n')
        color = self.color-5 if self.light_mode else self.color
        with Live(refresh_per_second=20, console=self.console) as live:
            live.console.clear(home=True)
            live.update(query)
            self.console.print(f'\n[italic]Gathered {footer_meta["prompt_tokens"]} context '
                               f'tokens (took {preprocessing}). Submitted to LLM. Awaiting '
                               'response...[/italic]',
                               style=f'color({color})', highlight=True)
            for piece in self.stream_response(documents):
                if start_time == 0:
                    start_time = time.time()
                current_response += piece.content
                footer_meta['token_count'] += self.response_count(piece.content)
                response = self.render_chat(current_response)
                footer = self.render_footer(time.time()-start_time, **footer_meta)
                # create our theme in the following order
                rich_content = Group(query, response, footer)
                # replace 'thinking' output with Mode's Markdown response
                if isinstance(response, Markdown) and self.do_once:
                    self.do_once = False
                    # Reset (erase) the thinking output
                    current_response = ''
                    rich_content = Group(query, response, footer)
                live.update(rich_content)

        # Finish by saving chat history, finding and storing new RAG/Tags or
        # llm_prompt changes, then reset it.
        current_response += f'{self.meta_capture}'
        self.meta_capture = ''
        self.common.chat_history_session.append(f'DATE TIMESTAMP:{documents["date_time"]}'
                                         f'\nUSER:{documents["user_query"]}\n'
                                         f'AI:{current_response}\n\n')
        self.cm.handle_context([current_response],
                                direction='store')
        self.common.save_chat(self.common.history_dir, self.common.chat_history_session)
        self.common.check_prompt(current_response)
