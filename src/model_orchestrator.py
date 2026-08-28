""" Model Orchestration """
from langchain_openai import ChatOpenAI
from .chat_utils import ChatOptions, RAGTag # For Type Hinting

MAX_AGENT_CALLS = 2

class Orchestration():
    """ Responsible for instantiating all ChatOpenAI objects """
    def __init__(self, console, args: ChatOptions):
        self.console = console
        self.args = args
        disable_thinking = args.disable_thinking
        think = not disable_thinking
        # Keep model-side thinking on (chat template). Do NOT let the server
        # (LM Studio / llama.cpp) parse generic <think> as a stream delimiter —
        # MiniMax-M3 uses <mm:think> and often *mentions* <think> in prose,
        # which used to abort the OpenAI stream at that token.
        extra_body = {
                       "thinking": {"type": "disabled" if not think else "enabled"},
                       "reasoning_format": "none",
                       "chat_template_kwargs": {
                         "enable_thinking": think,
                         "include_reasoning": think,
                         "think": think,
                         "top_k": 50,
                       }
                     }
        model_specs = {
            "story": {
                "base_url": args.host,
                "model": args.model,
                "temperature": args.model_temp,
                "top_p": args.model_topp,
            },
            "polisher": {
                "base_url": args.polisher_host,
                "model": args.polisher_llm,
                "temperature": args.polisher_temp,
                "top_p": args.polisher_topp
            },
            "vision": {
                "base_url": args.vision_host,
                "model": args.vision_llm,
                "temperature": args.vision_temp,
                "top_p": args.vision_topp
            },
            "agent": {
                "base_url": args.agent_host,
                "model": args.agent_llm,
                "temperature": args.agent_temp,
                "top_p": args.agent_topp
            },
            "nsfw": {
                "base_url": args.nsfw_host,
                "model": args.nsfw_llm,
                "temperature": args.nsfw_temp,
                "top_p": args.nsfw_topp
            },
            "casual": {
                "base_url": args.casual_host,
                "model": args.casual_llm,
                "temperature": args.casual_temp,
                "top_p": args.casual_topp
            },
            "coding": {
                "base_url": args.coder_host,
                "model": args.coder_llm,
                "temperature": args.coder_temp,
                "top_p": args.coder_topp
            },
            "structured": {
                "base_url": args.structured_host,
                "model": args.structured_llm,
                "temperature": args.structured_temp,
                "top_p": args.structured_topp
            },
            "general": {
                "base_url": args.general_host,
                "model": args.general_llm,
                "temperature": args.general_temp,
                "top_p": args.general_topp
            },
        }
        self.__llm = {}
        for model, dict_meta in model_specs.items():
            self.__llm[model] = ChatOpenAI(**dict_meta,
                                        frequency_penalty=args.frequency_penalty,
                                        presence_penalty=args.presence_penalty,
                                        streaming=True,
                                        max_completion_tokens=args.completion_tokens,
                                        stop_sequences=["<END_BEAT>", "<END_TURN>"],
                                        api_key=args.api_key,
                                        extra_body = extra_body,
                                        seed = args.seed,
                                        output_version="v0",
                                        use_responses_api=False,
                                    )

    def _route_story(self, documents)->ChatOpenAI:
        if documents.get('explicit', False):
            return self.get_model('nsfw')
        return self.get_model('story')

    @staticmethod
    def _requires_vision(documents)->bool:
        if documents.get('dynamic_images', []):
            return True
        return False

    def requires_agent(self, meta_tags: list[RAGTag], documents)->bool:
        if not self.args.assistant_mode or self.__llm['agent'].model_name == 'None':
            return False
        answer_confidence = float(0.0)
        for tag in meta_tags:
            if tag.tag == "answer_confidence":
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
        # Agent requested
        if (answer_confidence <= float(self.args.distrust_confidence)
            or 'agent' in documents.get('in_line_commands', [])):
            return True

        return False

    @staticmethod
    def _extract_mode(meta_tags: list[RAGTag])->str:
        assistant_mode = "general"
        for tag in meta_tags:
            if tag.tag == "assistant_mode":
                assistant_mode = tag.content.lower()
        return assistant_mode

    def _route_assistant(self, meta_tags, documents)->ChatOpenAI:
        if self.requires_agent(meta_tags, documents):
            return self.get_model("agent")

        if self._requires_vision(documents):
            return self.get_model("vision")

        assistant_mode = self._extract_mode(meta_tags)
        if self.args.debug:
            self.console.print(f'DEBUG: DYNAMIC MODEL CHOSEN: {assistant_mode}',
                                style=f'color({self.args.color})',highlight=False)
        return self.__llm.get(assistant_mode, self.__llm["general"])

    def get_rout_name(self, meta_tags: list[RAGTag], documents: dict | None = None)->str:
        """ Return route name """
        if self.requires_agent(meta_tags, documents):
            return 'agent'
        if self._requires_vision(documents):
            return 'vision'
        assistant_mode = self._extract_mode(meta_tags)
        return assistant_mode

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
