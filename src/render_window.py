""" module responsible for rendering output to the screen """
from dataclasses import dataclass, field
import time
import re
import traceback
from datetime import datetime
from threading import Thread
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text
from rich.align import Align
from rich.console import Group
from rich.rule import Rule
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_classic.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI # For Type Hinting
from .prompt_manager import PromptManager
from .context_manager import ContextManager # For Type Hinting
from .chat_utils import CommonUtils, ChatOptions, RAGTag # For Type Hinting
from .model_orchestrator import Orchestration, MAX_AGENT_CALLS
from .agent_tools import DuckDuckGoSearchTool
from .think_tags import ThinkFeed, chunk_text, split_think
from .gold_fetch import MAX_GOLD_FETCHES, take_need_gold, recall_status


def _abort_llm_stream(stream) -> None:
    """Close an in-flight LLM HTTP stream so the next completion can start.

    Breaking a ``for chunk in llm.stream()`` does not close the connection
    until GC. LM Studio then keeps generating while the resume call waits.
    """
    cur = stream
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        closer = getattr(cur, 'close', None)
        if callable(closer):
            try:
                closer()
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        nxt = None
        for attr in ('response', '_response', 'http_response'):
            cand = getattr(cur, attr, None)
            if cand is not None and cand is not cur:
                nxt = cand
                break
        cur = nxt


# pylint: disable=too-many-instance-attributes  # this is what a dataclass is for
@dataclass
class StreamState:
    """ RenderWindow animation (thinking) dataclass attributes """
    partial_chunk: str = ''
    meta_capture: str = ''
    meta_brace_count: int = 0
    meta_hide_attempt_count: int = 0
    thinking: bool = False
    think_ns: str = ''
    shadow_think: bool = False
    never_think: bool = False
    do_once: bool = False
    pulse_index: int = 0
    pulsing_chars: list[str] = field(default_factory=lambda: ['⠇', '⠋', '⠙', '⠸', '⠴', '⠦'])

@dataclass
class RenderWindowState:
    """ RenderWindow dataclass attributes """
    debug: bool
    verbose: bool
    assistant_mode: bool
    disable_thinking: bool
    no_rags: bool
    light_mode: bool
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
                 orchestration: Orchestration,
                 args: ChatOptions):
        super().__init__(console, current_dir, args)
        self.console = console
        self.common = common
        self.opts = args
        self.thinking_chunk = ''
        self.ooc_response = ''
        self.llm = None

        # populate dataclasses, setup
        self._load_states(current_dir, context, args)
        self.orchestrator = orchestration

        # Agent Prompt
        self.agent_prompt = ChatPromptTemplate.from_messages([
            ('system', ("You are a helpful research assistant. Today\'s date is "
                        f'{datetime.today().strftime("%B %d, %Y")}. Use web search to find '
                        'accurate, up-to-date information. If this is a follow-up search, '
                        'run a new query that fills the gaps — do not repeat the first '
                        'search verbatim unless the first results were empty.'))
            ,
            ('user', '{input}'),
            MessagesPlaceholder(variable_name='agent_scratchpad'),
            ])
        # Prompts
        self.prompts = PromptManager(
            console,
            current_dir,
            args,
            prompt_model=self.opts.model
        )
        self.prompts.build_prompts()

        self.thinking_active: bool = False
        self.thinking_thread = Thread(target=self.animate_thinking)
        self.namepulse_active: bool = False
        self.namepulse_thread = Thread(target=self.animate_namepulse)
        key = (self.opts.tavily_key or '').strip().lower()
        self.agent_tools = (
                [TavilySearch(tavily_api_key=self.opts.tavily_key)]
                if key and key != 'none'
                else [DuckDuckGoSearchTool()]
            )

    def _load_states(self, current_dir, context, args):
        """ Load the assorted dataclass objects in use throughout this module """
        self.state = RenderWindowState(
            debug = args.debug,
            verbose = args.verbose,
            assistant_mode = args.assistant_mode,
            disable_thinking = args.disable_thinking,
            no_rags=args.no_rags,
            light_mode = args.light_mode,
            completion_tokens = args.completion_tokens,
            syntax_theme = args.syntax_theme,
            context = context,
            current_dir = current_dir,
            seed = args.seed,
        )
        self.renderable = Renderables(
            header = Text(''),
            query = Markdown('', code_theme=self.state.syntax_theme),
            separator=Rule(style='bold color(208)'),
            assistant = Text('', style='bold color(208)'),
            response = Markdown('', code_theme=self.state.syntax_theme),
            footer = Text('')
        )

    def _get_version(self, model: str) -> str:
        """Extract version-like pattern, preferring ones with letter suffixes."""
        # Match: optional letter + number + optional decimal + optional letter suffix
        matches = re.findall(r'([a-z]?\d+(?:\.\d+)?[a-z]?)', model, re.IGNORECASE)

        if not matches:
            parts = re.split(r'[-_/]', model)
            return parts[-1][:4]

        # Rank matches: prefer those with a letter suffix
        def has_suffix(m):
            return bool(re.search(r'[a-z]$', m, re.IGNORECASE))

        ranked = sorted(matches, key=lambda m: (has_suffix(m), matches.index(m)), reverse=True)
        return ranked[0][:4]

    def _format_model_name(self, model) -> str:
        clean = model.lower().replace('_', '-')

        patterns = {
            'gpt': ('🤖', 'GPT'),
            'llama': ('🦙', 'Meta'),
            'maverick': ('🦙', 'Meta'),
            'meta': ('🦙', 'Meta'),
            'sapphira': ('🦙', 'Meta'),
            'mixtral': ('🌪️ ', 'Mistral'),
            'mistral': ('🌪️ ', 'Mistral'),
            'midnight': ('🌪️ ', 'Mistral'),
            'qwen': ('🐉', 'Qwen'),
            'glm': ('🔮', 'GLM'),
            'deepseek': ('🔍', 'DeepSeek'),
            'minimax': ('🎯', 'MiniMax'),
            'gemma': ('💎', 'Gemma'),
            'claude': ('🧠', 'Claude'),
        }

        for keyword, (icon, title) in patterns.items():
            if keyword in clean:
                version = self._get_version(model)
                return f'{icon} {title} [{version}]'

        parts = re.split(r'[-_/]', model)
        version = self._get_version(model)
        return f"📚 {' '.join(parts[:1])} [{version}]".title()[:20]

    def _pulse_emoji(self) -> str:
        stream = self.state.stream
        return f' {stream.pulsing_chars[stream.pulse_index]} ' if self.thinking_active else ' '

    def _calc_tokens_per_sec(self, tokens: int, generation_time: float) -> float:
        return tokens / generation_time if generation_time > 0 else 0

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
        Intercept <think> / <mm:think> tags in streamed content and optionally
        hide or reveal them.

        Blank first tokens (gpt-oss-120b) are shadow-think until the first
        non-blank token, which latches never_think. Tag-based models still
        open on <think>/<mm:think>; closers must match the opener namespace.
        Reset on the next user turn.
        """
        stream = self.state.stream
        piece, extra = chunk_text(chunk)
        try:
            chunk.content = piece
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        if show:
            return chunk

        was_busy = stream.thinking or stream.shadow_think
        feed = ThinkFeed(
            in_think=stream.thinking and not stream.shadow_think,
            ns=stream.think_ns,
            never_think=stream.never_think,
            shadow_think=stream.shadow_think,
        )
        visible, thought = feed.feed(piece)
        if extra:
            thought = extra + thought
        busy = feed.in_think or feed.shadow_think
        if thought and not was_busy:
            stream.do_once = True
            self.start_thinking()
            self.thinking_chunk = ''
        elif busy and not was_busy:
            stream.do_once = True
            self.start_thinking()
            self.thinking_chunk = ''
        if thought:
            self.thinking_chunk += thought
        if was_busy and not busy:
            stream.do_once = False
            self.stop_thinking()
        stream.thinking = busy
        stream.shadow_think = feed.shadow_think
        stream.think_ns = feed.ns
        stream.never_think = feed.never_think
        chunk.content = visible
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
        if images:
            image_blocks = [
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': RenderWindow._as_image_data_url(img_b64),
                    }
                }
                for img_b64 in images
            ]
            for i, msg in enumerate(messages):
                if isinstance(msg, HumanMessage):
                    if isinstance(msg.content, str):
                        messages[i] = HumanMessage(content=[
                            {'type': 'text', 'text': msg.content},
                            *image_blocks
                        ])
                    elif isinstance(msg.content, list):
                        messages[i].content.extend(image_blocks)
                    break
        return messages

    @staticmethod
    def _as_image_data_url(raw) -> str:
        """Keep data URLs (and their mime). Wrap bare base64 as jpeg."""
        text = str(raw or '').strip()
        if text.startswith('data:'):
            return text
        return f'data:image/jpeg;base64,{text}'

    @staticmethod
    def _llm_text(resp) -> str:
        piece = getattr(resp, 'content', None)
        if piece is None or piece == '' or piece == []:
            extra = getattr(resp, 'reasoning_content', None) or ''
            if not extra:
                kwargs = getattr(resp, 'additional_kwargs', None) or {}
                extra = (kwargs.get('reasoning_content') or '') if isinstance(kwargs, dict) else ''
            piece = extra or ''
        if not isinstance(piece, str):
            piece = str(piece)
        visible, _, _, _, _ = split_think(piece, False)
        return (visible or piece).strip()

    @staticmethod
    def _parse_agent_followup(text: str) -> str | None:
        """Return a refined query from YES: …, or None for NO / garbage."""
        lines = [ln.strip() for ln in (text or '').splitlines() if ln.strip()]
        if not lines:
            return None
        for ln in reversed(lines):
            cleaned = ln.strip('`').strip()
            upper = cleaned.upper()
            if upper == 'NO' or upper.startswith('NO:') or upper.startswith('NO '):
                return None
            if upper.startswith('YES:'):
                query = cleaned.split(':', 1)[1].strip().strip("\"'")
                return query or None
        return None

    def _agent_followup_query(self, documents: dict) -> str | None:
        """Ask the agent model if another web search is worth it."""
        if int(documents.get('agent_calls', 0)) >= MAX_AGENT_CALLS:
            return None
        original = documents.get('original_user_query') or documents.get('user_query', '')
        evidence = str(documents.get('dynamic_files') or '')[-6000:]
        prompt = (
            'You already ran a web search for this user query:\n'
            f'{original}\n\n'
            'Search results:\n'
            f'{evidence}\n\n'
            'If the results are enough to answer the user, reply with exactly: NO\n'
            'If you need one more search, reply with exactly:\n'
            'YES: <refined search query>\n'
            'No explanation.'
        )
        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
        except Exception:  # pylint: disable=broad-exception-caught
            return None
        return self._parse_agent_followup(self._llm_text(resp))

    def _invoke_web_agent(self, documents: dict, polish: bool, meta_data) -> list:
        """Run AgentExecutor once, then recurse so a follow-up search can happen."""
        if int(documents.get('agent_calls', 0)) >= MAX_AGENT_CALLS:
            documents['agent_ran'] = True
            return self.get_messages(meta_data, documents, polish=polish)
        documents.setdefault('original_user_query', documents['user_query'])
        documents.setdefault('dynamic_files', '')
        documents['agent_calls'] = int(documents.get('agent_calls', 0)) + 1
        call_n = documents['agent_calls']
        documents['agent_ran'] = True
        agent = create_openai_tools_agent(self.llm, self.agent_tools, self.agent_prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.agent_tools,
            verbose=False,
            max_iterations=4,
        )
        agent_input = (
            documents.pop('agent_followup_query', None)
            or documents['original_user_query']
        )
        label = f'({call_n}/{MAX_AGENT_CALLS})'
        try:
            self.console.print(
                f'Agent Tool Web Search {label} (ctl-c to cancel)...',
                style=f'color({self.state.color})',
                highlight=False,
            )
            result = agent_executor.invoke({'input': agent_input})
            documents['dynamic_files'] += (
                f'\n=== AGENT_TOOL_RESULT {label} ===\n{result}\n\n'
            )
        except KeyboardInterrupt:
            documents['dynamic_files'] += (
                '\n=== AGENT_TOOL_RESULT ===\nUSER CANCELED SEARCH\n\n'
            )
            return self.get_messages(meta_data, documents, polish=polish)
        except Exception:  # pylint: disable=broad-exception-caught
            self.console.print(
                'Error running agent!',
                style=f'color({self.state.color})',
                highlight=False,
            )
            documents['dynamic_files'] += (
                '\n=== AGENT_TOOL_RESULT ===\n'
                'ERROR: Tool execution failed.\n'
                'INSTRUCTION: You must inform the user that the web/tool search failed '
                'and that you cannot answer reliably without it. '
                'Do NOT fabricate or guess.\n\n'
            )
            documents['agent_error'] = '<AGENT_ERROR: TRUE>'
            return self.get_messages(meta_data, documents, polish=polish)

        if call_n < MAX_AGENT_CALLS:
            follow = self._agent_followup_query(documents)
            if follow:
                documents['agent_followup_query'] = follow
                documents['agent_ran'] = False
                documents['use_agent'] = True
                self.console.print(
                    f'Agent follow-up search: {follow}',
                    style=f'color({self.state.color})',
                    highlight=False,
                )
        return self.get_messages(meta_data, documents, polish=polish)

    def get_messages(self,
                     meta_data: RAGTag,
                     documents: dict,
                     polish: bool = False)->list[Document]:
        """ return formatted message to be sent to LLM stream """
        prompts = self.prompts
        if polish:
            self.llm = self.orchestrator.get_model('polisher')
        if self.debug:
            self.console.print(f'Model Chosen: {self.llm.model_name}',
                          style=f'color({self.state.color})',
                          highlight=False)

        # Populate current selected model
        documents['model_name'] = self.llm.model_name

        # One shot OOC population
        diag = (self.ooc_response or '').strip()
        if diag:
            documents['ooc_diagnostics'] = (
                'CRITICAL: Previous turn generated invalid output. You are to '
                'study the previous turn and understand your folly/error, and '
                f'follow these correction_rules:\n{diag}\n'
                '\nend correction_rules.'
            )
        else:
            documents['ooc_diagnostics'] = ''
        documents['ooc_diagnostics_bool'] = 'TRUE' if diag else 'FALSE'
        documents['ooc_mode_bool'] = (
            'TRUE' if documents['user_query'].strip().lower().startswith('ooc:') else 'FALSE')
        self.ooc_response = ''

        # One shot VISION population
        documents['vision_capable'] = 'FALSE'
        has_pixels = bool(documents.get('dynamic_images'))
        if (has_pixels
                or self.orchestrator.get_rout_name(meta_data, documents) == 'vision'):
            documents['vision_capable'] = ('TRUE - YOU ARE A VISION CAPABLE MODEL AND BEING '
                                           'PROVIDED ATTACHED IMAGES THIS TURN')

        if self.state.disable_thinking and not polish:
            documents['user_query'] = f'{documents["user_query"]} </think> </think> '

        if hasattr(self.state, 'context'):
            self.state.context.fill_documents_index(documents)
        documents.setdefault('documents_index', '')
        documents.setdefault('has_documents_index', False)
        documents.setdefault('gold_resume', '')
        documents.setdefault('attached_files_note', '')
        documents.setdefault('dynamic_files', '')
        documents.setdefault('include_branch', '')
        documents.setdefault('gold_documents', '')
        documents.setdefault('user_documents', '')
        documents.setdefault('ai_documents', '')
        documents.setdefault('chat_history', '')
        documents.setdefault('agent_error', '<AGENT_ERROR: FALSE>')
        documents['has_agent_error'] = 'TRUE' in str(documents.get('agent_error') or '')

        # pylint: disable=no-member # dynamic prompts (see self.__build_prompts)
        if polish:
            system_prompt = prompts.get_prompt(f'{prompts.polish_prompt_file}_system.md')
            human_prompt = prompts.get_prompt(f'{prompts.polish_prompt_file}_human.md')
        elif self.opts.assistant_mode:
            system_prompt, human_prompt = prompts.compose_nostory_plot(documents)
        else:
            system_prompt = prompts.get_prompt(f'{prompts.plot_prompt_file}_system.md')
            human_prompt = prompts.get_prompt(f'{prompts.plot_prompt_file}_human.md')
        # pylint: enable=no-member

        # Prompt conversions/templates
        system_tmpl = PromptTemplate(template=system_prompt,
                                     template_format='jinja2')
        human_tmpl  = PromptTemplate(template=human_prompt,
                                     template_format='jinja2')

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

        if self.debug:
            self.console.print(f'LLM DOCUMENTS: {documents.keys()}\n'
                               f'{documents["performance"]}\n',
                               style=f'color({self.state.color})',
                               highlight=False)

        # Format text messages from template
        images = documents.pop('dynamic_images', [])
        formatted_messages = prompt_template.format_messages(**documents)
        # Optional: inject images into HumanMessage if present
        messages = self.add_image_block(formatted_messages, images)
        documents['dynamic_images'] = images

        if self.debug:
            self.console.print(f'HEAVY LLM PROMPT (llm.stream()):\n{formatted_messages}\n\n',
                          style=f'color({self.state.color})',
                          highlight=False)

        documents.setdefault('agent_error', '<AGENT_ERROR: FALSE>')
        if self.orchestrator.requires_agent(meta_data, documents):
            return self._invoke_web_agent(documents, polish, meta_data)
        self.common.write_debug(f'live_stream-{self.llm.model_name}', messages)
        documents['prompt_tokens'] = self.packed_prompt_tokens(messages)
        return messages

    def packed_prompt_tokens(self, messages) -> int:
        """Word-count estimate of the packed prompt the LLM will actually see."""
        total = 0
        retriever = self.state.context.token_retriever
        for message in messages or []:
            total += self._content_tokens(getattr(message, 'content', ''), retriever)
        return total

    @staticmethod
    def _content_tokens(content, retriever) -> int:
        """Count text parts; skip image_url blocks (base64 is not context)."""
        if content is None:
            return 0
        if isinstance(content, str):
            return retriever(content)
        if isinstance(content, list):
            n = 0
            for part in content:
                if isinstance(part, dict):
                    if part.get('type') == 'image_url':
                        continue
                    n += retriever(str(part.get('text') or ''))
                elif part is not None:
                    n += retriever(str(part))
            return n
        return retriever(str(content))

    # Stream response as chunks
    def stream_response(self, messages: Document)->object:
        """Invoke LLM and stream response. Always abort the HTTP body on exit."""
        stream = self.llm.stream(messages)
        try:
            for chunk in stream:
                yield chunk
        finally:
            _abort_llm_stream(stream)

    def render_footer(self, time_taken: float = 0, generation_time: float = 0, **kwargs) -> Text:
        """ Render footer stats with heatmap colors and token metrics. """
        prompt_tokens = kwargs['prompt_tokens']
        token_count = kwargs['token_count']
        cleaned_color = kwargs['cleaned_color']
        token_savings = kwargs['token_savings']
        pre_processing_time = kwargs['pre_process_time']
        # pylint: disable-next=consider-using-f-string # no. this is how its done
        formatted_time = '{:.1f}s'.format(pre_processing_time)
        model = self.llm.model_name
        turn = kwargs['turn_count']

        foot_color = self.state.color - 6 if self.state.light_mode else self.state.color

        footer = Text('\nTurn: ', style=f'color({foot_color})')
        footer.append(f'{turn} ', style='color(123)')
        footer.append(self._format_model_name(model), style='color(202)')
        footer.append(self._pulse_emoji(), style=f'color({12 if self.state.light_mode else 51})')
        footer.append(f'{time_taken:.2f}', style='color(94)')
        footer.append('s Tokens(dedup:', style=f'color({foot_color})')
        footer.append(f'{token_savings}', style=f'color({cleaned_color})')
        footer.append(' context:', style=f'color({foot_color})')
        footer.append(f'{prompt_tokens}', style=f'color({self._color_for_context(prompt_tokens)})')
        footer.append(f':{formatted_time}', style=f'color({foot_color})')
        footer.append(' completion:', style=f'color({foot_color})')
        footer.append(f'{token_count}', style=f'color({self._color_for_completion(token_count)})')
        footer.append(f') {self._calc_tokens_per_sec(token_count, generation_time):.1f}T/s',
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
        elif stream.do_once and stream.thinking:
            color = self.state.color-5 if self.state.light_mode else self.state.color
            chat_content = Text('Thinking...', style=f'color({color}')
        else:
            chat_content = Markdown(current_stream, code_theme=self.state.syntax_theme)
        return chat_content

    def set_llm(self, meta_data: RAGTag, documents: dict)->ChatOpenAI:
        """ Set and Return Orchestrated LLM """
        self.llm = self.orchestrator.route(meta_data, documents)
        return self.llm

    def _reset_think_state(self) -> None:
        """Clear think-tag latches at the start of a turn."""
        stream = self.state.stream
        stream.never_think = False
        stream.shadow_think = False
        stream.thinking = False
        stream.think_ns = ''
        self.thinking_chunk = ''

    def _skip_polisher(self, documents: dict) -> bool:
        """True when the raw stream should be shown (no polish pass)."""
        return (
            self.opts.polisher_llm == 'None'
            or documents['user_query'].find('OOC:') != -1
            or self.opts.assistant_mode
        )

    def _paint_token(self, documents, footer_meta, color, live, inference_start,
                     start_time, current_response) -> None:
        """Update the live footer/name after one stream piece."""
        stream = self.state.stream
        if self._skip_polisher(documents):
            self.renderable.response = self.build_content(current_response)
        else:
            self.renderable.response = Text(
                'Receiving message to polish...', style=f'color({color}',
            )
        self.renderable.footer = self.render_footer(
            time.time() - inference_start,
            time.time() - start_time,
            **footer_meta,
        )
        if (isinstance(self.renderable.response, Markdown)
                and stream.do_once
                and not stream.shadow_think):
            stream.do_once = False
            cleared = True
        else:
            cleared = False
        name_color = self.state.pulse_colors[self.state.pulse_color_index]
        self.renderable.assistant = Text(
            documents['name'], style=f'bold color({name_color})',
        )
        self.render_chat(live)
        return cleared

    def _consume_model_stream(self, messages, documents, footer_meta, color,
                              live, inference_start):
        """Read the LLM stream into the live panel.

        Returns (assembled text, first-token timestamp).
        """
        current_response = ''
        first_token_at = 0
        for piece in self.stream_response(messages):
            piece = self.reveal_thinking(piece, self.state.verbose)
            if first_token_at == 0:
                first_token_at = time.time()
            current_response += piece.content
            footer_meta['token_count'] += self.response_count(piece.content)
            if self._paint_token(
                documents, footer_meta, color, live, inference_start,
                first_token_at, current_response,
            ):
                current_response = ''
        return current_response, first_token_at

    def _resume_gold_fetches(
        self, assembled: str, documents: dict, meta_data, messages,
        footer_meta, color, live, inference_start, first_token_at,
    ) -> str:
        """Fetch gold files the model asked for and continue this turn."""
        del messages
        if not self.opts.assistant_mode:
            visible, _ = take_need_gold(assembled)
            return visible
        fetches = 0
        recalled: list[str] = []
        while fetches < MAX_GOLD_FETCHES:
            visible, fname = take_need_gold(assembled)
            assembled = visible
            if not fname:
                break
            if not self.state.context.fetch_gold_file(documents, fname):
                break
            fetches += 1
            recalled.append(fname)
            documents['gold_resume'] = visible
            self.renderable.response = Text(
                recall_status(recalled), style=f'color({color}',
            )
            self.render_chat(live)
            packed = self.get_messages(meta_data, documents)
            more, later = self._consume_model_stream(
                packed, documents, footer_meta, color, live, inference_start,
            )
            if later and not first_token_at:
                first_token_at = later
            assembled = (visible.rstrip() + '\n' + more).strip()
        return assembled

    def _run_polisher(self, documents, meta_data, footer_meta, color, live,
                      inference_start, first_token_at, current_response) -> str:
        """Optional polish passes; returns the last pass text."""
        self.renderable.response = Text('Loading Polisher...', style=f'color({color}')
        self.render_chat(live)
        self.common.write_debug(
            f'polisher_input-{self.llm.model_name}', current_response,
        )
        passes = int(self.opts.polisher_cnt)
        for pass_num in range(passes):
            documents['llm_response'] = current_response
            messages = self.get_messages(meta_data, documents, polish=True)
            current_response = ''
            for piece in self.stream_response(messages):
                piece = self.reveal_thinking(piece, self.state.verbose)
                current_response += piece.content
                footer_meta['token_count'] += self.response_count(piece.content)
                if passes == pass_num + 1:
                    self.renderable.response = self.build_content(current_response)
                else:
                    self.renderable.response = Text(
                        f'Polishing pass {pass_num+1} of {passes-1} before final...',
                        style=f'color({color}',
                    )
                self.renderable.footer = self.render_footer(
                    time.time() - inference_start,
                    time.time() - first_token_at,
                    **footer_meta,
                )
                name_color = self.state.pulse_colors[self.state.pulse_color_index]
                self.renderable.assistant = Text(
                    documents['name'], style=f'bold color({name_color})',
                )
                self.render_chat(live)
        return current_response

    def _prime_live_panel(self, documents, meta_data, footer_meta, color) -> None:
        """Set header/query/assistant/response before Live starts."""
        rag = '' if self.state.no_rags else 'RAG+'
        took = '{:.1f}s'.format(footer_meta['pre_process_time'])
        self.renderable.header = Text(
            f'Submitting relevant {rag}History tokens: '
            f'{footer_meta["prompt_tokens"]} '
            f'{documents.get("in_line_commands", "")} '
            f'(took {took})...',
            style=f'color({color})',
        )
        self.renderable.query = Markdown(
            f'**You:** {documents["user_query"]}',
            code_theme=self.state.syntax_theme,
        )
        self.renderable.assistant = Text(documents['name'], style='bold color(208)')
        route = self.orchestrator.get_rout_name(meta_data, documents)
        self.renderable.response = Text(
            f'Inference/Loading ({route} LLM)...', style=f'color({color}',
        )
        self.renderable.footer = self.render_footer(0.0, 0.0, **footer_meta)

    def live_stream(self, documents: dict, meta_data: RAGTag)->None:
        """Stream one turn into the Rich Live display and persist it."""
        self._reset_think_state()
        history = self.common.load_chat()
        pre_process_time = float(documents['pre_process_time'])
        start_time = time.time()
        self.llm = self.orchestrator.route(meta_data, documents)
        messages = self.get_messages(meta_data, documents)
        self.llm = self.orchestrator.route(meta_data, documents)
        pre_process_time += time.time() - start_time
        token_total = documents.get('prompt_tokens') or self.packed_prompt_tokens(messages)
        branch = 'assistant' if self.opts.assistant_mode else history.get(
            'current', 'story',
        )
        footer_meta = {
            'token_savings': documents['token_savings'],
            'prompt_tokens': token_total,
            'cleaned_color': documents['cleaned_color'],
            'pre_process_time': pre_process_time,
            'token_count': 0,
            'content_rating': documents['explicit'],
            'turn_count': len(history[branch]) + 1,
        }
        color = self.state.color - 5 if self.state.light_mode else self.state.color
        self._prime_live_panel(documents, meta_data, footer_meta, color)
        inference_start = time.time()
        current_response = ''
        with Live(refresh_per_second=30, console=self.console) as live:
            live.console.clear(home=True)
            self.render_chat(live)
            self.start_namepulse()
            try:
                current_response, first_token_at = self._consume_model_stream(
                    messages, documents, footer_meta, color, live,
                    inference_start,
                )
                current_response = self._resume_gold_fetches(
                    current_response, documents, meta_data, messages,
                    footer_meta, color, live, inference_start, first_token_at,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                error_text = (
                    f'**LLM Error**: {exc}\n\n'
                    'The model backend may need to be reloaded.'
                )
                self.renderable.response = self.build_content(error_text)
                self.render_chat(live)
                if self.state.debug:
                    traceback.print_exc()
                return None
            documents['llm_response'] = current_response
            if not self._skip_polisher(documents):
                current_response = self._run_polisher(
                    documents, meta_data, footer_meta, color, live,
                    inference_start, first_token_at or inference_start,
                    current_response,
                )
            self.stop_namepulse()
            if not current_response or current_response == ' ':
                self.renderable.response = self.build_content(
                    'Error: received no response from LLM',
                )
            self.renderable.assistant = Text(
                documents['name'], style='bold color(208)',
            )
            self.render_chat(live)
        if documents.get('no_context', False) or not current_response:
            return None
        self.save_history(documents, current_response)
        return None


    def save_history(self, documents: dict, current_response: str,
                     reasoning: str = '') -> None:
        """Save turn as role/content messages. Reasoning is optional extra."""
        stream = self.state.stream
        history = self.common.load_chat()

        # ── Resolve active branch (now branches work in both modes) ─────
        branch = history.get('current', 'story')
        if not isinstance(history.get(branch), list):
            history[branch] = []

        # ── Persist mode metadata alongside the save ────────────────────
        history['assistant_mode'] = bool(self.opts.assistant_mode)
        history.setdefault('branch_modes', {})
        history['branch_modes'][branch] = bool(self.opts.assistant_mode)

        # ── OOC handling (unchanged) ─────────────────────────────────────
        ooc_prefix = self.common.regex.ooc_prefix
        if ooc_prefix.search(current_response) or ooc_prefix.search(
            documents['user_query']
        ):
            if not ooc_prefix.search(current_response):
                self.console.print(
                    '\nNOTE:\tBad LLM response. LLM ignored OOC request.',
                    style=f'color({self.state.color})',
                )
                return
            self.ooc_response = current_response
            return

        # ── Context handling (unchanged) ────────────────────────────────
        documents['llm_response'] = current_response
        self.state.context.handle_context(documents, direction='store')
        current_response = self.common.sanitize_response(current_response)

        if self.state.disable_thinking:
            documents['user_query'] = documents['user_query'].replace('', '')

        # ── Append user/assistant pair to active branch ─────
        assistant_msg: dict = {'role': 'assistant', 'content': current_response}
        thought = (reasoning or getattr(self, 'thinking_chunk', '') or '').strip()
        if thought:
            assistant_msg['reasoning'] = thought
        history[branch].append({'role': 'user', 'content': documents['user_query']})
        history[branch].append(assistant_msg)

        stream.meta_capture = ''
        if self.debug:
            self.console.print('DEBUG: saving to RAG...', highlight=False)

        self.common.save_chat(history)

        if self.debug:
            self.console.print('DEBUG: live finished', highlight=False)
