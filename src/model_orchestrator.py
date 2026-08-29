""" Model Orchestration """
import re
from langchain_openai import ChatOpenAI
from .chat_utils import ChatOptions, RAGTag # For Type Hinting

MAX_AGENT_CALLS = 2
# Tagger often scores these 1.0 anyway; force a search when the query is live.
_LIVE_QUERY = re.compile(
    r'(?ix)'
    r'(stock\s+price|share\s+price|ticker\b|price\s+of\b'
    r'|weather\b|current\s+events?'
    r'|right\s+now|as\s+of\b|just\s+released'
    r'|latest\s+(version|release|news|price))'
)

class Orchestration():
    """ Responsible for instantiating all ChatOpenAI objects """
    def __init__(self, console, args: ChatOptions):
        """Build one ChatOpenAI client per orchestrated role."""
        self.console = console
        self.args = args
        disable_thinking = args.disable_thinking
        think = not disable_thinking
        # Keep model-side thinking on (chat template). Do NOT let the server
        # (LM Studio / llama.cpp) parse generic <think> as a stream delimiter —
        # MiniMax-M3 uses <mm:think> and often *mentions* <think> in prose,
        # which used to abort the OpenAI stream at that token.
        extra_body = {
            'thinking': {'type': 'disabled' if not think else 'enabled'},
            'reasoning_format': 'none',
            'chat_template_kwargs': {
                'enable_thinking': think,
                'include_reasoning': think,
                'think': think,
                'top_k': 50,
            },
        }
        self.__llm = {}
        for name, spec in self._model_specs(args).items():
            self.__llm[name] = ChatOpenAI(
                **spec,
                **self._shared_llm_kwargs(args, extra_body),
            )

    @staticmethod
    def _model_specs(args: ChatOptions) -> dict:
        """Per-role base_url / model / temperature / top_p."""
        return {
            'story': {
                'base_url': args.host, 'model': args.model,
                'temperature': args.model_temp, 'top_p': args.model_topp,
            },
            'polisher': {
                'base_url': args.polisher_host, 'model': args.polisher_llm,
                'temperature': args.polisher_temp, 'top_p': args.polisher_topp,
            },
            'vision': {
                'base_url': args.vision_host, 'model': args.vision_llm,
                'temperature': args.vision_temp, 'top_p': args.vision_topp,
            },
            'agent': {
                'base_url': args.agent_host, 'model': args.agent_llm,
                'temperature': args.agent_temp, 'top_p': args.agent_topp,
            },
            'nsfw': {
                'base_url': args.nsfw_host, 'model': args.nsfw_llm,
                'temperature': args.nsfw_temp, 'top_p': args.nsfw_topp,
            },
            'casual': {
                'base_url': args.casual_host, 'model': args.casual_llm,
                'temperature': args.casual_temp, 'top_p': args.casual_topp,
            },
            'coding': {
                'base_url': args.coder_host, 'model': args.coder_llm,
                'temperature': args.coder_temp, 'top_p': args.coder_topp,
            },
            'structured': {
                'base_url': args.structured_host, 'model': args.structured_llm,
                'temperature': args.structured_temp, 'top_p': args.structured_topp,
            },
            'general': {
                'base_url': args.general_host, 'model': args.general_llm,
                'temperature': args.general_temp, 'top_p': args.general_topp,
            },
        }

    @staticmethod
    def _shared_llm_kwargs(args: ChatOptions, extra_body: dict) -> dict:
        """Constructor kwargs shared by every ChatOpenAI role."""
        return {
            'frequency_penalty': args.frequency_penalty,
            'presence_penalty': args.presence_penalty,
            'streaming': True,
            'max_completion_tokens': args.completion_tokens,
            'stop_sequences': ['<END_BEAT>', '<END_TURN>'],
            'api_key': args.api_key,
            'extra_body': extra_body,
            'seed': args.seed,
            'output_version': 'v0',
            'use_responses_api': False,
        }

    def _route_story(self, documents)->ChatOpenAI:
        """Pick NSFW vs story for non-assistant turns."""
        if documents.get('explicit', False):
            return self.get_model('nsfw')
        return self.get_model('story')

    @staticmethod
    def _requires_vision(documents)->bool:
        """True when the turn includes attached images."""
        if documents.get('dynamic_images', []):
            return True
        return False

    def requires_agent(self, meta_tags: list[RAGTag], documents)->bool:
        """True when this turn should run AgentExecutor (capped at 2)."""
        if not self.args.assistant_mode or self.__llm['agent'].model_name == 'None':
            return False
        answer_confidence = float(0.0)
        for tag in meta_tags:
            if tag.tag == 'answer_confidence':
                answer_confidence = float(tag.content)

        # Hard cap: get_messages may run the agent twice (initial + follow-up)
        if int(documents.get('agent_calls', 0)) >= MAX_AGENT_CALLS:
            return False
        # Agent previously invoked (get_messages recurses after the search)
        if documents.get('agent_ran', False):
            return False
        # Explicit: \agent, Spur Agent toggle, or pre-processor search_internet
        if documents.get('use_agent'):
            return True
        query = str(documents.get('user_query') or '')
        if _LIVE_QUERY.search(query) and not (
                documents.get('has_images') or documents.get('dynamic_images')
                or documents.get('has_files')
                or documents.get('attached_files_note')):
            return True
        # Agent requested
        if (answer_confidence <= float(self.args.distrust_confidence)
            or 'agent' in documents.get('in_line_commands', [])):
            return True

        return False

    @staticmethod
    def _extract_mode(meta_tags: list[RAGTag],
                      documents: dict | None = None)->str:
        """Read assistant_mode from pre-processor tags (default general).

        Vision is never a tagger decision — only attached pixels force it.
        """
        assistant_mode = 'general'
        for tag in meta_tags:
            if tag.tag == 'assistant_mode':
                assistant_mode = tag.content.lower()
        if assistant_mode == 'vision' and not Orchestration._requires_vision(
                documents or {},
        ):
            return 'structured'
        return assistant_mode

    def _route_assistant(self, meta_tags, documents)->ChatOpenAI:
        """Pick agent / vision / tagged assistant model."""
        if self.requires_agent(meta_tags, documents):
            return self.get_model('agent')

        if self._requires_vision(documents):
            return self.get_model('vision')

        assistant_mode = self._extract_mode(meta_tags, documents)
        if self.args.debug:
            self.console.print(f'DEBUG: DYNAMIC MODEL CHOSEN: {assistant_mode}',
                                style=f'color({self.args.color})',highlight=False)
        return self.__llm.get(assistant_mode, self.__llm['general'])

    def name_of(self, llm) -> str:
        """Role key for a ChatOpenAI instance (story, nsfw, coding, …)."""
        for name, client in self.__llm.items():
            if client is llm:
                return name
        return ''

    def get_route_name(self, meta_tags: list[RAGTag],
                       documents: dict | None = None) -> str:
        """Return the orchestrator role chosen for this turn."""
        documents = documents or {}
        if not self.args.assistant_mode:
            if documents.get('explicit', False):
                return 'nsfw'
            return 'story'
        if self.requires_agent(meta_tags, documents):
            return 'agent'
        if self._requires_vision(documents):
            return 'vision'
        return self._extract_mode(meta_tags, documents)

    def get_rout_name(self, meta_tags: list[RAGTag],
                      documents: dict | None = None) -> str:
        """Alias kept for the TUI footer."""
        return self.get_route_name(meta_tags, documents)

    def route(self, meta_tags: list[RAGTag], documents: dict | None = None)->ChatOpenAI:
        """
        Return suitable LLM based on chat mode and/or special context in documents
        """
        documents = documents or {}

        if not self.args.assistant_mode:
            return self._route_story(documents)

        return self._route_assistant(meta_tags, documents)

    def get_model(self, model)->ChatOpenAI:
        """ return a specific ChatOpenAI model object """
        return self.__llm[model]
