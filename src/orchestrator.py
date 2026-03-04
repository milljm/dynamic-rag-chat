""" Model Orchestration """
from langchain_openai import ChatOpenAI
from .chat_utils import ChatOptions, RAGTag # For Type Hinting

class Orchestration():
    """ Responsible for instantiating all ChatOpenAI objects """
    def __init__(self, console, args: ChatOptions):
        self.console = console
        self.args = args
        disable_thinking = args.disable_thinking
        think = not disable_thinking
        extra_body = {
                       "thinking": {"type": "disabled" if not think else "enabled"},
                       "chat_template_kwargs": {
                         "enable_thinking": think,
                         "include_reasoning": think,
                         "think": think,
                       }
                     }
        model_specs = {
            "story": {
                "base_url": args.host,
                "model": args.model,
                "temperature": 0.7,
            },
            "polisher": {
                "base_url": args.polisher_host,
                "model": args.polisher_llm,
                "temperature": 0.9,
            },
            "vision": {
                "base_url": args.vision_host,
                "model": args.vision_llm,
                "temperature": 0.5,
            },
            "agent": {
                "base_url": args.agent_host,
                "model": args.agent_llm,
                "temperature": 0.7,
            },
            "nsfw": {
                "base_url": args.nsfw_host,
                "model": args.nsfw_llm,
                "temperature": 0.7,
            },
            "casual": {
                "base_url": args.casual_host,
                "model": args.casual_llm,
                "temperature": 0.7,
            },
            "coding": {
                "base_url": args.coder_host,
                "model": args.coder_llm,
                "temperature": 0.2,
            },
            "structured": {
                "base_url": args.structured_host,
                "model": args.structured_llm,
                "temperature": 0.2,
            },
            "general": {
                "base_url": args.general_host,
                "model": args.general_llm,
                "temperature": 0.6,
            },
        }
        self.__llm = {}
        for model, dict_meta in model_specs.items():
            self.__llm[model] = ChatOpenAI(**dict_meta,
                                        top_p=args.top_p,
                                        frequency_penalty=args.frequency_penalty,
                                        presence_penalty=args.presence_penalty,
                                        streaming=True,
                                        max_completion_tokens=args.completion_tokens,
                                        stop_sequences=["<END_BEAT>", "<END_TURN>"],
                                        api_key=args.api_key,
                                        extra_body = extra_body,
                                        seed = args.seed,
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

    def _requires_agent(self, meta_tags: list[RAGTag], documents)->bool:
        if not self.args.assistant_mode:
            return False
        search_internet = 'false'
        for tag in meta_tags:
            if tag.tag == "search_internet":
                search_internet = tag.content

        # Agent previously invoked
        if documents.get('agent_ran', False):
            return False
        # Agent requested
        if search_internet.lower() == 'true' or 'agent' in documents.get('in_line_commands', []):
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
        if self._requires_agent(meta_tags, documents):
            return self.get_model("agent")

        if self._requires_vision(documents):
            return self.get_model("vision")

        assistant_mode = self._extract_mode(meta_tags)
        if self.args.debug:
            self.console.print(f'DEBUG: DYNAMIC MODEL CHOSEN: {assistant_mode}',
                                style=f'color({self.args.color})',highlight=False)
        return self.__llm.get(assistant_mode, self.__llm["general"])

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

