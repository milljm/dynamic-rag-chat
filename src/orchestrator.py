""" Model Orchestration """
from langchain_openai import ChatOpenAI
from .chat_utils import ChatOptions, RAGTag # For Type Hinting

class Orchestration():
    """ Responsible for instantiating all ChatOpenAI objects """
    def __init__(self, args: ChatOptions):
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
            "analysis": {
                "base_url": args.analysis_host,
                "model": args.analysis_llm,
                "temperature": 0.4,
            },
            "reasoning": {
                "base_url": args.reasoning_host,
                "model": args.reasoning_llm,
                "temperature": 0.3,
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

    def route(self, meta_tags: list[RAGTag], documents: dict=None)->ChatOpenAI:
        """
        Decide which model to use based on RAGTags,
        and documents during RAG/Context population based on USER Query.
        """
        if documents is None:
            documents = {}
        # Hard stop for story telling
        if not self.args.assistant_mode:
            if 'nsfw' in documents.get('explicit', []):
                return self.get_model('nsfw')
            return self.get_model('story')

        # Determine best assistant mode model
        assistant_mode = "general"
        search_internet = False
        for tag in meta_tags:
            if tag.tag == "assistant_mode":
                assistant_mode = tag.content.lower()
            if tag.tag == "search_internet":
                search_internet = tag.content

        # If web search required, optionally override model
        if search_internet or 'agent' in documents.get('in_line_commands', []):
            return self.get_model("agent")

        # If user uploaded an image, override model
        if documents.get('dynamic_images', []):
            return self.get_model("vision")

        # Sanity check model availability, default to general
        if assistant_mode not in self.__llm:
            assistant_mode = "general"

        return self.get_model(assistant_mode)

    def get_model(self, model)->ChatOpenAI:
        """ return orchestrated model """
        return self.__llm[model]
