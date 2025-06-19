""" module responsible for rendering output to the screen """
import os
from dataclasses import dataclass, field
from typing import Any
import time
from threading import Thread
from rich.live import Live
from rich.markdown import Markdown, CodeBlock
from rich.syntax import Syntax
from rich.text import Text
from rich.align import Align
from rich.console import Group
from langchain.prompts import ChatPromptTemplate
from langchain.schema.messages import HumanMessage
from langchain.schema import BaseMessage   # for Type Hinting
from langchain_openai import ChatOpenAI
from .prompt_manager import PromptManager

# pylint: disable=too-many-instance-attributes  # this is what a dataclass is for
@dataclass
class StreamState:
    """ RenderWindow animation (thinking) dataclass attributes """
    partial_chunk: str = ''
    meta_capture: str = ''
    meta_hiding: bool = False
    thinking: bool = False
    do_once: bool = False
    pulse_index: int = 0
    pulsing_chars: list[str] = field(default_factory=lambda: ["⠇", "⠋", "⠙", "⠸", "⠴", "⠦"])

@dataclass
class RenderWindowState:
    """ RenderWindow dataclass attributes """
    debug: bool
    assistant_mode: bool
    no_rags: bool
    verbose: bool
    light_mode: bool
    model: str
    host: str
    api_key: str
    num_ctx: int
    syntax_theme: str
    console: Any
    common: Any
    context: Any
    current_dir: str
    color: int = field(init=False)
    pulse_colors: list[int] = field(default_factory=lambda: list(
                                              range(234,254)) + list(range(252,233,-1))
                                              )
    pulse_color_index: int = 0
    stream: StreamState = field(default_factory=StreamState)

    def __post_init__(self):
        self.color = 245 if self.light_mode else 233
# pylint: enable=too-many-instance-attributes

class ThinkingThread(Thread):
    """ Allow pulsing animation to run as a thread """
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def run(self):
        while self.owner.thinking_active:
            self.owner.animate_thinking()
            time.sleep(0.5)

class NamepulseThread(Thread):
    """ Allow pulsing animation to run as a thread """
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def run(self):
        while self.owner.namepulse_active:
            self.owner.animate_namepulse()
            time.sleep(0.5)

class RenderWindow(PromptManager):
    """ Responsible for printing Rich Text/Markdown Live to the screen """
    def __init__(self, console, common, context, current_dir, **kwargs):
        super().__init__(console, current_dir, **kwargs)

        # Load config into dataclass
        self.state = RenderWindowState(
            debug=kwargs['debug'],
            assistant_mode=kwargs['assistant_mode'],
            no_rags=kwargs['use_rags'],
            verbose=kwargs['verbose'],
            light_mode=kwargs['light_mode'],
            model=kwargs['model'],
            host=kwargs['host'],
            api_key=kwargs['api_key'],
            num_ctx=kwargs['num_ctx'],
            syntax_theme=kwargs['syntax_theme'],
            console=console,
            common=common,
            context=context,
            current_dir=current_dir
        )

        self.thinking_active: bool = False
        self.thinking_thread = Thread(target=self.animate_thinking)
        self.namepulse_active: bool = False
        self.namepulse_thread = Thread(target=self.animate_namepulse)

        self.console = self.state.console
        self.common = self.state.common

        self.prompts = PromptManager(
            console,
            current_dir,
            prompt_model=self.state.model,
            **kwargs
        )

        self.llm = ChatOpenAI(
            base_url=self.state.host,
            model=self.state.model,
            temperature=0.4 if self.state.assistant_mode else '1.1',
            streaming=True,
            max_tokens=self.state.num_ctx,
            api_key=self.state.api_key
        )

        self.prompts.build_prompts()
        class SimpleCodeBlock(CodeBlock):
            """ Code Block Syntax injection """
            state = self.state
            def __rich_console__(self, console, options):
                code = str(self.text).rstrip()
                syntax = Syntax(code,
                                self.lexer_name,
                                theme=self.state.syntax_theme,
                                word_wrap=True,
                                padding=(1,0))
                yield syntax
        Markdown.elements["fence"] = SimpleCodeBlock

    def _format_model_name(self) -> str:
        """Extracts a cleaned model identifier from full model name."""
        return '-'.join(self.common.model_re.findall(self.state.model)[:2])

    def _pulse_emoji(self) -> str:
        stream = self.state.stream
        return f' {stream.pulsing_chars[stream.pulse_index]} ' if self.thinking_active else ' '

    def _calc_tokens_per_sec(self, tokens: int, duration: float) -> float:
        return tokens / duration if duration > 0 else 0

    def _color_for_context(self, prompt_tokens: int) -> int:
        return [v for k, v in self.common.prompt_map.items() if k <= prompt_tokens][-1]

    def _color_for_completion(self, token_count: int) -> int:
        return [v for k, v in self.common.heat_map.items() if token_count * 4 >= k][-1]

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

    def if_metatags(self, chunk: object, verbose: bool)->object:
        """
        Hide the LLM's response when performing <meta_tags operation
        """
        if verbose:
            return chunk
        content = str(chunk.content)
        stream = self.state.stream  # shorthand

        # === CASE 1: Chunk has '<' – buffer it
        # print(f'DEBUG START>{content}<END')
        if '<' in content and not stream.meta_hiding:
            # print(f'DEBUG: < found {content}')
            stream.partial_chunk = content
            chunk.content = ''

        # === CASE 2: We're continuing from a previous partial '<'
        elif stream.partial_chunk and not stream.meta_hiding:
            # print('DEBUG: partial_chunk true')
            combined = stream.partial_chunk + content
            stream.partial_chunk = ''

            if self.common.meta_start_re.search(combined):
                # print(f'DEBUG: partial_chunk match {combined}')
                stream.meta_capture = combined
                stream.meta_hiding = True
                self.start_thinking()
                chunk.content = ''
                return chunk

            chunk.content = combined

        # === CASE 3: Normal hiding mode
        elif stream.meta_hiding:
            # print(f'DEBUG HIDING: waiting for ">" curently:{content}:{">" in content}')
            stream.meta_capture += content
            if '>' in content:
                stream.meta_hiding = False
                self.stop_thinking()
            chunk.content = ''

        # === CASE 4: Tag started cleanly with no split '<'
        elif self.common.meta_start_re.search(content):
            # print(f'DEBUG NOT HIDING: waiting for "reason to hide" curently:{content}')
            stream.meta_capture = content
            stream.meta_hiding = True
            self.start_thinking()
            chunk.content = ''

        return chunk

    def reveal_thinking(self, chunk: object, show: bool = False)->object:
        """
        Intercept <think> tags in streamed content and optionally hide or reveal them.

        If `show` is True, actual thinking content is shown.
        If `show` is False, replaces it with 'AI thinking...' at start, then hides remaining.
        """
        stream = self.state.stream
        content = str(chunk.content)

        # End of <think> block
        if stream.thinking and '</think>' in content:
            stream.thinking = False
            self.stop_thinking()
            chunk.content = ''
            return chunk

        # Start of <think> block
        if not stream.thinking and '<think>' in content:
            stream.thinking = True
            stream.do_once = True
            self.start_thinking()
            chunk.content = 'AI thinking...'
            return chunk

        # Middle of thinking stream
        if stream.thinking:
            chunk.content = content if show else ''
            return chunk

        return chunk

    def start_thinking(self):
        """ method to start thinking animation """
        if hasattr(self, 'thinking_thread') and self.thinking_thread.is_alive():
            self.thinking_thread.join(timeout=0.1)
        self.thinking_active = True
        self.thinking_thread = ThinkingThread(self)
        self.thinking_thread.daemon = True
        self.thinking_thread.start()

    def stop_thinking(self):
        """ method to stop thinking animation """
        self.thinking_active = False

    def start_namepulse(self):
        """ method to start thinking animation """
        if hasattr(self, 'namepulse_thread') and self.namepulse_thread.is_alive():
            self.namepulse_thread.join(timeout=0.1)
        self.namepulse_active = True
        self.namepulse_thread = NamepulseThread(self)
        self.namepulse_thread.daemon = True
        self.namepulse_thread.start()

    def stop_namepulse(self):
        """ method to stop thinking animation """
        self.namepulse_active = False

    def animate_namepulse(self):
        """ animate the assistants name """
        state = self.state
        while self.namepulse_active:
            time.sleep(0.1)
            state.pulse_color_index = (state.pulse_color_index  + 1) % len(state.pulse_colors)
            self.render_chat()  # Re-render chat with updated pulse

    def animate_thinking(self):
        """ a threaded method to run the thinking animation """
        stream = self.state.stream  # shorthand
        while self.thinking_active:
            time.sleep(0.1)  # Adjust speed (0.1 seconds per frame)
            stream.pulse_index = (stream.pulse_index + 1) % len(stream.pulsing_chars)
            self.render_chat()

    @staticmethod
    def add_image_block(messages: list[BaseMessage], images: list)->list[BaseMessage]:
        """ add/return image block if images are present """
        # Optional: inject images into HumanMessage if present
        if images:
            image_blocks = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    }
                }
                for img_b64 in images
            ]
            # Replace or extend HumanMessage with image blocks
            for i, msg in enumerate(messages):
                if isinstance(msg, HumanMessage):
                    if isinstance(msg.content, str):
                        messages[i] = HumanMessage(content=[
                            {"type": "text", "text": msg.content},
                            *image_blocks
                        ])
                    elif isinstance(msg.content, list):
                        messages[i].content.extend(image_blocks)
                    break  # Only handle first HumanMessage
        return messages

    # Stream response as chunks
    def stream_response(self, documents: dict):
        """ Parse LLM Prompt """
        prompts = self.prompts
        # pylint: disable=no-member # dynamic prompts (see self.__build_prompts)
        system_prompt = (prompts.get_prompt(f'{prompts.plot_prompt_file}_system.md')
                     if self.debug else prompts.plot_prompt_system)

        human_prompt = (prompts.get_prompt(f'{prompts.plot_prompt_file}_human.md')
                     if self.debug else prompts.plot_prompt_human)
        # pylint: enable=no-member
        prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])

        if self.debug:
            self.console.print(f'LLM DOCUMENTS: {documents.keys()}\n'
                               f'{documents["performance"]}\n',
                               style=f'color({self.state.color})',
                               highlight=False)
        else:
            with open(os.path.join(self.common.history_dir, 'debug.log'),
                      'w', encoding='utf-8') as f:
                f.write(f'LLM DOCUMENTS: {documents.keys()}')

        # Format text messages from template
        images = documents.pop('dynamic_images', [])
        formatted_messages = prompt_template.format_messages(**documents)

        # Optional: inject images into HumanMessage if present
        messages = self.add_image_block(formatted_messages, images)

        # pylint: enable=no-member
        if self.debug:
            self.console.print(f'HEAVY LLM PROMPT (llm.stream()):\n{formatted_messages}\n\n',
                          style=f'color({self.state.color})',
                          highlight=False)
        else:
            with open(os.path.join(self.common.history_dir, 'debug.log'),
                      'w', encoding='utf-8') as f:
                f.write(f'HEAVY LLM PROMPT (llm.stream()): {formatted_messages}')
        for chunk in self.llm.stream(messages):
            chunk = self.reveal_thinking(chunk, self.state.verbose)
            chunk = self.if_metatags(chunk, self.state.verbose)
            yield chunk

    def render_footer(self, time_taken: float = 0, **kwargs) -> Text:
        """Render footer stats with heatmapped colors and token metrics."""
        prompt_tokens = kwargs['prompt_tokens']
        token_count = kwargs['token_count']
        cleaned_color = kwargs['cleaned_color']
        token_savings = kwargs['token_savings']
        pre_processing_time = kwargs['pre_process_time']

        foot_color = self.state.color - 6 if self.state.light_mode else self.state.color

        footer = Text('\n', style=f'color({foot_color})')
        footer.append(self._format_model_name(), style='color(202)')
        footer.append(self._pulse_emoji(), style=f'color({12 if self.state.light_mode else 51})')
        footer.append(f'{time_taken:.2f}', style='color(94)')
        footer.append('s Tokens(deduplication:', style=f'color({foot_color})')
        footer.append(f'{token_savings}', style=f'color({cleaned_color})')
        footer.append(' context:', style=f'color({foot_color})')
        footer.append(f'{prompt_tokens}', style=f'color({self._color_for_context(prompt_tokens)})')
        footer.append(f':{pre_processing_time}', style=f'color({foot_color})')
        footer.append(' completion:', style=f'color({foot_color})')
        footer.append(f'{token_count}', style=f'color({self._color_for_completion(token_count)})')
        footer.append(f') {self._calc_tokens_per_sec(token_count, time_taken):.1f}T/s',
                      style=f'color({foot_color})')

        return footer

    # Compose the full chat display with footer (model name, time taken, token count)
    def render_chat(self, current_stream: str = "")->Text|Markdown:
        """ render and return markdown/syntax """
        if self.state.stream.thinking and self.state.verbose:
            chat_content = Text(current_stream, style=f'color({self.state.color})')
        else:
            chat_content = Markdown(current_stream)
        return chat_content

    def live_stream(self, documents: dict,
                          token_savings: int,
                          prompt_tokens: int,
                          cleaned_color: int,
                          preprocessing)->None:
        """ Handle the Rich Live updating process """
        stream = self.state.stream  # shorthand
        current_response = ''
        footer_meta = {'token_savings'   : token_savings,
                       'prompt_tokens'   : prompt_tokens,
                       'cleaned_color'   : cleaned_color,
                       'pre_process_time': documents['pre_process_time'],
                       'token_count'     : 0}

        start_time = 0
        color = self.state.color-5 if self.state.light_mode else self.state.color
        _rag = '' if not self.state.no_rags and self.state.assistant_mode else 'RAG+'
        header = Text(f'Submitting relevant {_rag}History tokens: {footer_meta["prompt_tokens"]} '
                        f'(took {preprocessing})...', style=f'color({color})')
        seperator = Markdown('---')
        query = Markdown(f'**You:** {documents["user_query"]}')
        with Live(refresh_per_second=20, console=self.console) as live:
            live.console.clear(home=True)
            live.update(Group(header,
                              query,
                              seperator,
                              Align.right(Text(documents["name"],
                                               style='bold color(208)'))))

            self.start_namepulse()
            for piece in self.stream_response(documents):
                if start_time == 0:
                    start_time = time.time()
                current_response += piece.content
                footer_meta['token_count'] += self.response_count(piece.content)
                response = self.render_chat(current_response)
                footer = self.render_footer(time.time()-start_time, **footer_meta)
                # replace 'thinking' output with Mode's Markdown response
                if isinstance(response, Markdown) and stream.do_once:
                    stream.do_once = False
                    # Reset (erase) the thinking output
                    current_response = ''
                name_color = self.state.pulse_colors[self.state.pulse_color_index]
                rich_content = Group(header,
                                     query,
                                     seperator,
                                     Align.right(Text(documents["name"],
                                                      style=f'bold color({name_color})')),
                                     response,
                                     footer)
                live.update(rich_content)
            self.stop_namepulse()
            rich_content = Group(header,
                                     query,
                                     seperator,
                                     Align.right(Text(documents["name"],
                                                      style='bold color(208)')),
                                     response,
                                     footer)
            live.update(rich_content)

        # Finish by saving chat history, finding and storing new RAG/Tags or
        # llm_prompt changes, then reset it.
        current_response += stream.meta_capture
        if self.state.assistant_mode and not self.state.no_rags:
            self.common.chat_history_session.append(f'\nUSER: {documents["user_query"]}\n\n'
                                                f'AI: {current_response}')
            if self.state.verbose:
                self.console.print('Info: Nothing saved (assistant-mode)',
                                style=f'color({self.state.color})',highlight=False)
            return
        # Pesky LLM forgot to close meta_tags with '>'
        if (stream.thinking and "<meta_tags:" in current_response
            and not current_response.strip().endswith(">")):
            current_response += '>'
            stream.meta_hiding = False
            self.stop_thinking()
            self.console.print('Warn: I had to help the LLM close meta_tags',
                               style=f'color({self.state.color})',highlight=False)
        stream.meta_capture = ''
        self.state.context.handle_context([str(current_response)],
                                          direction='store')
        current_response = self.common.sanatize_response(current_response)
        self.common.chat_history_session.append(f'\nUSER: {documents["user_query"]}\n\n'
                                                f'AI: {current_response}')
        self.common.save_chat()
        self.common.check_prompt(current_response)
