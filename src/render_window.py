""" module responsible for rendering output to the screen """
from dataclasses import dataclass, field
import time
from threading import Thread
from rich.live import Live
from rich.markdown import Markdown, CodeBlock
from rich.syntax import Syntax
from rich.text import Text
from rich.align import Align
from rich.console import Group
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.schema.messages import HumanMessage
from langchain.schema import BaseMessage, Document   # For Type Hinting
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
from .prompt_manager import PromptManager
from .context_manager import ContextManager # For Type Hinting
from .chat_utils import CommonUtils, ChatOptions # For Type Hinting
from .agent_tools import DuckDuckGoSearchTool

# pylint: disable=too-many-instance-attributes  # this is what a dataclass is for
@dataclass
class StreamState:
    """ RenderWindow animation (thinking) dataclass attributes """
    partial_chunk: str = ''
    meta_capture: str = ''
    meta_brace_count: int = 0
    meta_hide_attempt_count: int = 0
    meta_hiding: bool = False
    thinking: bool = False
    no_think_bug: bool = False
    do_once: bool = False
    pulse_index: int = 0
    pulsing_chars: list[str] = field(default_factory=lambda: ["⠇", "⠋", "⠙", "⠸", "⠴", "⠦"])

@dataclass
class RenderWindowState:
    """ RenderWindow dataclass attributes """
    debug: bool
    verbose: bool
    assistant_mode: bool
    disable_thinking: bool
    no_rags: bool
    light_mode: bool
    no_think_tag: bool
    model: str
    agent_llm: str
    agent_host: str
    vision_llm: str
    vision_host: str
    polisher: str
    polisher_host: str
    polisher_cnt: int
    nsfw_model: str
    host: str
    api_key: str
    completion_tokens: int
    syntax_theme: str
    context: ContextManager
    current_dir: str
    seed: int|None = None
    color: int = field(init=False)
    pulse_colors: list[int] = field(default_factory=lambda: list(
                                              range(234,254)) + list(range(252,233,-1))
                                              )
    pulse_color_index: int = 0
    stream: StreamState = field(default_factory=StreamState)

    def __post_init__(self):
        self.color = 245 if self.light_mode else 236

@dataclass
class Renderables:
    """ Rich Live renderables dataclass object """
    header: Text
    query: Markdown
    separator: Markdown
    assistant: Text
    response: Text|Markdown
    footer: Text

    @property
    def full_window(self) -> Group:
        """ return Live Group """
        return Group(self.header,
                     self.query,
                     self.separator,
                     Align.right(self.assistant),
                     self.response,
                     self.footer
                     )
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
    def __init__(self, console,
                 common: CommonUtils,
                 context: ContextManager,
                 current_dir,
                 args: ChatOptions):
        super().__init__(console, current_dir, args)
        self.console = console
        self.common = common
        self.opts = args
        self.think_once = True
        self.use_vision = False
        self.thinking_chunk = ''
        self.ooc_response = ''

        # populate dataclasses, setup
        self._load_states(current_dir, context, args)

        disable_thinking = self.state.disable_thinking
        think = not disable_thinking
        extra_body = {
                       "thinking": {"type": "disabled" if not think else "enabled"},
                       "chat_template_kwargs": {
                         "enable_thinking": think,
                         "include_reasoning": think,
                         "think": think,
                       }
                     }

        # Our heavy-weight LLMs
        self.llm = {
                'sfw'  : ChatOpenAI(base_url=self.state.host,
                                    model=self.state.model,
                                    temperature=0.6 if self.state.assistant_mode
                                                    else args.temperature,
                                    top_p=args.top_p,
                                    frequency_penalty=args.frequency_penalty,
                                    presence_penalty=args.presence_penalty,
                                    streaming=True,
                                    max_completion_tokens=self.state.completion_tokens,
                                    stop_sequences=["<END_BEAT>", "<END_TURN>"],
                                    api_key=self.state.api_key,
                                    extra_body = extra_body,
                                    seed = self.state.seed,
                                    ),

                'nsfw' : ChatOpenAI(base_url=self.state.host,
                                    model=self.state.nsfw_model,
                                    temperature=0.6 if self.state.assistant_mode
                                                    else args.temperature,
                                    top_p=args.top_p,
                                    frequency_penalty=args.frequency_penalty,
                                    presence_penalty=args.presence_penalty,
                                    streaming=True,
                                    max_completion_tokens=self.state.completion_tokens,
                                    stop_sequences=["<END_BEAT>", "<END_TURN>"],
                                    api_key=self.state.api_key,
                                    extra_body = extra_body,
                                    seed = self.state.seed,
                                    ),
                'polish' : ChatOpenAI(base_url=self.state.polisher_host,
                                      model=('None' if self.state.polisher is None
                                             else self.state.polisher),
                                      temperature=0.9 if self.state.assistant_mode
                                                        else args.temperature,
                                      top_p=args.top_p,
                                      frequency_penalty=args.frequency_penalty,
                                      presence_penalty=args.presence_penalty,
                                      streaming=True,
                                      max_completion_tokens=self.state.completion_tokens,
                                      stop_sequences=["<END_BEAT>", "<END_TURN>"],
                                      api_key=self.state.api_key,
                                      extra_body = extra_body,
                                      seed = self.state.seed,
                                      ),
                'tool' :    ChatOpenAI(base_url=self.state.agent_host,
                                      model=('None' if self.state.agent_llm is None
                                             else self.state.agent_llm),
                                      temperature=0.5,
                                      top_p=args.top_p,
                                      frequency_penalty=args.frequency_penalty,
                                      presence_penalty=args.presence_penalty,
                                      streaming=False,
                                      max_completion_tokens=self.state.completion_tokens,
                                      stop_sequences=["<END_BEAT>", "<END_TURN>"],
                                      api_key=self.state.api_key,
                                      seed = self.state.seed,
                                      ),
                'vision' :  ChatOpenAI(base_url=self.state.vision_host,
                                      model=('None' if self.state.vision_llm is None
                                             else self.state.vision_llm),
                                      temperature=0.5,
                                      top_p=args.top_p,
                                      frequency_penalty=args.frequency_penalty,
                                      presence_penalty=args.presence_penalty,
                                      streaming=False,
                                      max_completion_tokens=self.state.completion_tokens,
                                      stop_sequences=["<END_BEAT>", "<END_TURN>"],
                                      api_key=self.state.api_key,
                                      seed = self.state.seed,
                                      ),
            }

        # Agent Prompt
        self.agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant."""),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
        # Prompts
        self.prompts = PromptManager(
            console,
            current_dir,
            args,
            prompt_model=self.state.model
        )
        self.prompts.build_prompts()

        self.thinking_active: bool = False
        self.thinking_thread = Thread(target=self.animate_thinking)
        self.namepulse_active: bool = False
        self.namepulse_thread = Thread(target=self.animate_namepulse)
        self.agent_tools = [DuckDuckGoSearchTool()]

    def _load_states(self, current_dir, context, args):
        """ Load the assorted dataclass objects in use throughout this module """
        self.state = RenderWindowState(
            debug = args.debug,
            verbose = args.verbose,
            assistant_mode = args.assistant_mode,
            disable_thinking = args.disable_thinking,
            no_rags=args.no_rags,
            light_mode = args.light_mode,
            no_think_tag = args.no_think_tag,
            model = args.model,
            agent_llm = args.agent_llm,
            agent_host = args.agent_host,
            vision_llm = args.vision_llm,
            vision_host = args.vision_host,
            polisher = args.polisher,
            polisher_host = args.polisher_host,
            polisher_cnt = args.polisher_cnt,
            nsfw_model = args.nsfw_model,
            host = args.host,
            api_key = args.api_key,
            completion_tokens = args.completion_tokens,
            syntax_theme = args.syntax_theme,
            context = context,
            current_dir = current_dir,
            seed = args.seed,
        )
        self.renderable = Renderables(
            header = Text(''),
            query = Markdown(''),
            separator = Markdown('---'),
            assistant = Text('', style='bold color(208)'),
            response = Markdown(''),
            footer = Text('')
        )

        # Use SimpleCodeBlock instead of CodeBlock (CodeBlock fencing will strip trailing
        # character from a word if that word fits perfectly within fenced window)
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

    def _format_model_name(self, is_nsfw: bool) -> str:
        """Extract a cleaned model identifier and prefix with sfw/nsfw."""
        model = self.state.nsfw_model if is_nsfw else self.state.model
        label = 'nsfw' if is_nsfw else 'sfw'
        matches = self.common.regex.model_re.findall(model)[:2]
        model_short = '-'.join(matches)
        return f'{label}-{model_short[:10]}'

    def _pulse_emoji(self) -> str:
        stream = self.state.stream
        return f' {stream.pulsing_chars[stream.pulse_index]} ' if self.thinking_active else ' '

    def _calc_tokens_per_sec(self, tokens: int, duration: float) -> float:
        return tokens / duration if duration > 0 else 0

    def _color_for_context(self, prompt_tokens: int) -> int:
        return [v for k, v in self.common.prompt_map.items() if k <= prompt_tokens][-1]

    def _color_for_completion(self, token_count: int) -> int:
        return [v for k, v in self.common.heat_map.items() if token_count * 4 >= k][-1]

    def clear_ooc(self):
        """ clear the OOC Response """
        self.ooc_response = ''

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

    def reveal_thinking(self, chunk: object, show: bool = False)->object:
        """
        Intercept <think> tags in streamed content and optionally hide or reveal them.

        If `show` is True, actual thinking content is shown.
        If `show` is False, replaces it with '' at start, then hides remaining.
        """
        stream = self.state.stream
        content = str(chunk.content)
        # Allow the model to print <think> </think> tags after it is finished reasoning
        if self.think_once is False:
            return chunk
        # print(f'DEBUG: think_once: {self.think_once}, {stream.thinking} TOK:>{content}<')

        # End of <think> block
        if stream.thinking and ('</think>' in content or '</thinking>' in content):
            self.common.save_thinking(self.thinking_chunk)
            self.thinking_chunk = ''
            stream.thinking = False
            self.think_once = False
            self.stop_thinking()
            chunk.content = ''
            return chunk

        # Start of <think> block
        if not stream.thinking and ('<think>' in content
                                    or '<thinking>' in content
                                    or stream.no_think_bug):
            stream.no_think_bug = False
            stream.thinking = True
            stream.do_once = True
            self.start_thinking()
            chunk.content = ''
            return chunk

        # Middle of thinking stream
        if stream.thinking:
            chunk.content = content if show else ''
            self.thinking_chunk += content
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
            self.build_content()  # Re-render chat with updated pulse

    def animate_thinking(self):
        """ a threaded method to run the thinking animation """
        stream = self.state.stream  # shorthand
        while self.thinking_active:
            time.sleep(0.1)  # Adjust speed (0.1 seconds per frame)
            stream.pulse_index = (stream.pulse_index + 1) % len(stream.pulsing_chars)
            self.build_content()

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

    def get_messages(self, documents: dict, polish: bool = False)->list[Document]:
        """ return formatted message to be sent to LLM stream """
        prompts = self.prompts

        # One shot OOC population
        diag = (self.ooc_response or '').strip()
        documents['ooc_diagnostics'] = (
            'CRITICAL: Previous turn generated invalid output. You are to study the previous turn'
            f' and and understand your folly/error, and follow these correction rules:\n{diag}')
        documents['ooc_diagnostics_bool'] = 'TRUE' if diag else 'FALSE'
        documents['ooc_mode_bool'] = (
            'TRUE' if documents['user_query'].strip().lower().startswith("ooc:") else 'FALSE')
        self.ooc_response = ''

        if self.state.disable_thinking and not polish:
            documents['user_query'] = f'{documents["user_query"]} </think> </think> '

        # pylint: disable=no-member # dynamic prompts (see self.__build_prompts)
        if polish:
            system_prompt = (prompts.get_prompt(f'{prompts.polish_prompt_file}_system.md')
                        if self.debug or self.opts.prompts_debug else prompts.polish_prompt_system)

            human_prompt = (prompts.get_prompt(f'{prompts.polish_prompt_file}_human.md')
                        if self.debug or self.opts.prompts_debug else prompts.polish_prompt_human)
        else:
            system_prompt = (prompts.get_prompt(f'{prompts.plot_prompt_file}_system.md')
                        if self.debug or self.opts.prompts_debug else prompts.plot_prompt_system)

            human_prompt = (prompts.get_prompt(f'{prompts.plot_prompt_file}_human.md')
                        if self.debug or self.opts.prompts_debug else prompts.plot_prompt_human)
        # pylint: enable=no-member

        # Prompt conversions/templates
        system_tmpl = PromptTemplate(template=system_prompt,
                                     template_format="jinja2")
        human_tmpl  = PromptTemplate(template=human_prompt,
                                     template_format="jinja2")

        system_msg = SystemMessagePromptTemplate(prompt=system_tmpl)
        human_msg  = HumanMessagePromptTemplate(prompt=human_tmpl)

        if polish:
            prompt_template = ChatPromptTemplate.from_messages([
                human_msg
            ])
        else:
            prompt_template = ChatPromptTemplate.from_messages([
                system_msg,
                human_msg,
            ])

        # Does this post require SFW, or NSFW
        llm = 'nsfw' if documents['explicit'] else 'sfw'
        if polish:
            llm = 'polish'

        if self.debug:
            self.console.print(f'LLM DOCUMENTS: {documents.keys()}\n'
                               f'{documents["performance"]}\n',
                               style=f'color({self.state.color})',
                               highlight=False)
        self.common.write_debug(self.llm[llm].model_name, documents.keys())

        # Format text messages from template
        images = documents.pop('dynamic_images', [])
        if images:
            self.use_vision = True
        formatted_messages = prompt_template.format_messages(**documents)

        # Optional: inject images into HumanMessage if present
        messages = self.add_image_block(formatted_messages, images)

        if self.debug:
            self.console.print(f'Content rating: {llm}',
                          style=f'color({self.state.color})',
                          highlight=False)

        # pylint: enable=no-member
        if self.debug:
            self.console.print(f'HEAVY LLM PROMPT (llm.stream()):\n{formatted_messages}\n\n',
                          style=f'color({self.state.color})',
                          highlight=False)
        self.common.write_debug(self.llm[llm].model_name, formatted_messages)
        if documents.get('use_agent', False) and not documents.get('agent_ran', False):
            # Let LangChain create the proper prompt template for the agent
            llm = 'tool' if self.state.agent_llm is not None else llm
            agent = create_openai_tools_agent(self.llm[llm], self.agent_tools, self.agent_prompt)
            documents['agent_ran'] = True
            agent_executor = AgentExecutor(agent=agent, tools=self.agent_tools, verbose=False)
            self.console.print('Agent Tool running...', style=f'color({self.state.color})',
                                highlight=False)
            try:
                result = agent_executor.invoke({"input": documents['user_query']})
                documents['dynamic_files'] += f'\n=== AGENT_TOOL_RESULT ===\n{result}\n\n'
                return self.get_messages(documents, polish=polish)
            # pylint: disable-next=bare-except # too many ways an LLM can go wrong
            except:
                self.console.print('Error running agent!', style=f'color({self.state.color})',
                                highlight=False)
                documents['dynamic_files'] += '\n=== AGENT_TOOL_RESULT ===\nERROR RUNNING AGENT\n\n'
                return self.get_messages(documents, polish=polish)
        return messages

    # Stream response as chunks
    def stream_response(self, documents: dict, messages: Document, polish: bool = False):
        """ Parse LLM Prompt """
        # Priorities of which model to use depending on the situation
        # I feel this should be way more betterer
        llm = 'nsfw' if documents['explicit'] else 'sfw'
        llm = 'polish' if polish else llm
        llm = 'vision' if self.use_vision else llm
        for chunk in self.llm[llm].stream(messages):
            chunk = self.reveal_thinking(chunk, self.state.verbose)
            yield chunk

    def render_footer(self, time_taken: float = 0, **kwargs) -> Text:
        """ Render footer stats with heatmap colors and token metrics. """
        prompt_tokens = kwargs['prompt_tokens']
        token_count = kwargs['token_count']
        cleaned_color = kwargs['cleaned_color']
        token_savings = kwargs['token_savings']
        pre_processing_time = kwargs['pre_process_time']
        # pylint: disable-next=consider-using-f-string # no. this is how its done
        formatted_time = '{:.1f}s'.format(pre_processing_time)
        rating = kwargs['content_rating']
        turn = kwargs['turn_count']

        foot_color = self.state.color - 6 if self.state.light_mode else self.state.color

        footer = Text('\nTurn:', style=f'color({foot_color})')
        footer.append(f'{turn} ', style='color(123)')
        footer.append(self._format_model_name(rating), style='color(202)')
        footer.append(self._pulse_emoji(), style=f'color({12 if self.state.light_mode else 51})')
        footer.append(f'{time_taken:.2f}', style='color(94)')
        footer.append('s Tokens(dedup:', style=f'color({foot_color})')
        footer.append(f'{token_savings}', style=f'color({cleaned_color})')
        footer.append(' context:', style=f'color({foot_color})')
        footer.append(f'{prompt_tokens}', style=f'color({self._color_for_context(prompt_tokens)})')
        footer.append(f':{formatted_time}', style=f'color({foot_color})')
        footer.append(' completion:', style=f'color({foot_color})')
        footer.append(f'{token_count}', style=f'color({self._color_for_completion(token_count)})')
        footer.append(f') {self._calc_tokens_per_sec(token_count, time_taken):.1f}T/s',
                      style=f'color({foot_color})')

        return footer

    def render_chat(self, live: Live)->None:
        """ update the screen using Rich Live with all Rich renderables """
        live.update(self.renderable.full_window)

    # Compose the full chat display with footer (model name, time taken, token count)
    def build_content(self, current_stream: str = '')->Text|Markdown:
        """ render and return markdown/syntax """
        stream = self.state.stream # shorthand
        if stream.thinking and self.state.verbose:
            chat_content = Text(current_stream, style=f'color({self.state.color})')
        elif stream.do_once and (stream.thinking or stream.meta_hiding):
            color = self.state.color-5 if self.state.light_mode else self.state.color
            chat_content = Text('Thinking...', style=f'color({color}')
        else:
            chat_content = Markdown(current_stream)
        return chat_content

    def live_stream(self, documents: dict)->None:
        """ Handle the Rich Live updating process """
        stream = self.state.stream   # shorthand
        context = self.state.context # shorthand
        self.think_once = True
        self.use_vision = False

        # pesky LLMs that have reasoning and don't generate a <think> token,
        # yet generate an ending </think> token!
        if self.opts.no_think_tag:
            stream.no_think_bug = True

        history = self.common.load_chat()
        pre_process_time = float(documents['pre_process_time'])
        start_time = time.time()
        messages = self.get_messages(documents)
        pre_process_time += time.time()-start_time
        token_total = documents['prompt_tokens']
        for message in messages:
            token_total += context.token_retriever(message.content)
        current_response = ''
        if self.opts.assistant_mode:
            branch = 'assistant'
        else:
            branch = history.get('current', 'default')
        footer_meta = {'token_savings'   : documents['token_savings'],
                       'prompt_tokens'   : token_total,
                       'cleaned_color'   : documents['cleaned_color'],
                       'pre_process_time': pre_process_time,
                       'token_count'     : 0,
                       'content_rating'  : documents['explicit'],
                       'turn_count'      : len(history[branch])+1}
        color = self.state.color-5 if self.state.light_mode else self.state.color
        _rag = '' if not self.state.no_rags and self.state.assistant_mode else 'RAG+'
        self.renderable.header = Text(f'Submitting relevant {_rag}History tokens: '
                                      f'{footer_meta["prompt_tokens"]} '
                                      f'{documents.get("in_line_commands", "")} '
                                      f"(took {'{:.1f}s'.format(pre_process_time)})...",
                                      style=f'color({color})')
        self.renderable.query = Markdown(f'**You:** {documents["user_query"]}')
        self.renderable.assistant = Text(documents["name"], style='bold color(208)')
        self.renderable.response = Text('Inference/Loading...', style=f'color({color}')
        self.renderable.footer = self.render_footer(0.0, **footer_meta)
        start_time = 0
        with Live(refresh_per_second=20, console=self.console) as live:
            live.console.clear(home=True)
            self.render_chat(live)
            self.start_namepulse()
            for piece in self.stream_response(documents, messages):
                if start_time == 0:
                    start_time = time.time()
                current_response += piece.content
                footer_meta['token_count'] += self.response_count(piece.content)
                if (self.state.polisher is None
                     or documents['user_query'].find('OOC:') != -1
                     or self.opts.assistant_mode):
                    self.renderable.response = self.build_content(current_response)
                else:
                    self.renderable.response = Text('Receiving message to polish...',
                                                     style=f'color({color}')
                self.renderable.footer = self.render_footer(time.time()-start_time, **footer_meta)
                # replace 'thinking' output with Model's Markdown response
                if isinstance(self.renderable.response, Markdown) and stream.do_once:
                    stream.do_once = False
                    # Reset (erase) the thinking output
                    current_response = ''
                name_color = self.state.pulse_colors[self.state.pulse_color_index]
                self.renderable.assistant = Text(documents["name"],
                                                 style=f'bold color({name_color})')
                self.render_chat(live)

            # Polisher + polishing cnt
            if (self.state.polisher is not None
                    and documents['user_query'].find('OOC:') == -1
                    and not self.opts.assistant_mode):
                self.renderable.response = Text('Loading Polisher...',
                                                     style=f'color({color}')
                self.render_chat(live)
                documents['llm_response'] = current_response
                for pass_num in range(int(self.state.polisher_cnt)):
                    documents['llm_response'] = current_response
                    messages = self.get_messages(documents, polish=True)
                    current_response = ''
                    for piece in self.stream_response(documents, messages, polish=True):
                        if start_time == 0:
                            start_time = time.time()
                        current_response += piece.content
                        footer_meta['token_count'] += self.response_count(piece.content)
                        if int(self.state.polisher_cnt) == pass_num+1:
                            self.renderable.response = self.build_content(current_response)
                        else:
                            self.renderable.response = Text(
                                f'Polishing pass {pass_num+1} of'
                                f' {int(self.state.polisher_cnt)-1} before final...',
                                style=f'color({color}')
                        self.renderable.footer = self.render_footer(time.time()-start_time,
                                                                     **footer_meta)
                        name_color = self.state.pulse_colors[self.state.pulse_color_index]
                        self.renderable.assistant = Text(documents["name"],
                                                    style=f'bold color({name_color})')
                        self.render_chat(live)

            self.stop_namepulse()
            if not current_response or current_response == ' ':
                self.renderable.response = self.build_content('Error: received no response '
                                                              'from LLM')

            self.renderable.assistant = Text(documents["name"], style='bold color(208)')
            self.render_chat(live)

        # Do not save any output if \no-context was used
        if documents.get('no_context', False):
            return

        # Finish by saving chat history, finding and storing new RAG/Tags or
        # llm_prompt changes, then reset it.
        if self.debug and self.state.no_rags and self.state.assistant_mode:
            self.console.print(f'DEBUG: storing meta\n{stream.meta_capture}\n\n',
                                style=f'color({self.state.color})',highlight=False)
        # current_response += f'\n\n{stream.meta_capture}'

        # Attache OOC's response for next system/human prompt message usage
        ooc_prefix = self.common.regex.ooc_prefix  # shorthand
        if ooc_prefix.search(current_response) or ooc_prefix.search(documents['user_query']):
            if not ooc_prefix.search(current_response):
                self.console.print(
                    '\nNOTE:\tBad LLM response. LLM ignored OOC request. This turn not saved.',
                    style=f'color({self.state.color})', highlight=False)
                return
            self.ooc_response = current_response
            return

        documents['llm_response'] = current_response
        stream.meta_capture = ''
        if self.debug:
            self.console.print('DEBUG: saving to RAG...',
                                style=f'color({self.state.color})',highlight=False)
        self.common.write_debug('last', current_response)
        self.state.context.handle_context(documents, direction='store')
        current_response = self.common.sanitize_response(current_response)
        self.common.write_debug('sanitized', current_response)

        if self.state.disable_thinking:
            documents["user_query"] = documents["user_query"].replace('</think>', '')

        history[branch].append(
            f'\n⬇  TURN {len(history[branch])+1}  ⬇\n'
            f'TIMESTAMP: {self.common.get_time(self.opts.time_zone)}\n'
            f'USER: {documents["user_query"]}\n\n'
            f'AI: {current_response}')
        self.common.save_chat(history)

        if self.debug:
            self.console.print('DEBUG: live finished',
                                style=f'color({self.state.color})',highlight=False)
        return
