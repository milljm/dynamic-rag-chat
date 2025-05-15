"""
ContextManager aims at handeling everything relating to the context
being supplied to the LLM. It utilizing several methods:

    Emoji removal
    Fuzzy match sentences
    list[] -> set() removes any matches from the RAG.
"""
import re
import os
import threading
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from ragtag_manager import RAGTagManager, RAG, RAGTag
from ollama_model import OllamaModel
from prompt_manager import PromptManager
from filter_builder import FilterBuilder
current_dir = os.path.dirname(os.path.abspath(__file__))

class ContextManager(PromptManager):
    """ A collection of methods aimed at producing/reducing the context """
    def __init__(self, console, common, **kwargs):
        super().__init__(console)
        self.console = console
        self.common = common
        self.host = kwargs['host']
        self.matches = kwargs['matches']
        self.preconditioner = kwargs['preconditioner']
        self.debug = kwargs['debug']
        self.name = kwargs['name']
        self.rag = RAG(console, self.common, **kwargs)
        self.rag_tagger = RAGTagManager(console, self.common, **kwargs)
        self.prompts = PromptManager(self.console,
                                     model=self.preconditioner,
                                     debug=self.debug)
        self.filter_builder = FilterBuilder()
        self.prompts.build_prompts()

    @staticmethod
    def token_retreiver(context: str)->int:
        """ iterate over list and do a word count (token) """
        _token_cnt = 0
        if isinstance(context, list):
            for sentence in context:
                _token_cnt += len(sentence.split())
        else:
            _token_cnt += len(context.split(' '))
        return _token_cnt

    @staticmethod
    def normalize_for_dedup(text: str) -> str:
        """ remove emojis and other markdown """
        text = re.sub(r'[\U0001F600-\U0001F64F\u2600-\u26FF\u2700-\u27BF]', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return ' '.join(text.lower().split())

    def pre_processor(self, query)->tuple[str,list[RAGTag]]:
        """
        lightweight LLM as a tagging pre-processor
        """
        pre_llm = OllamaModel(self.host)
        prompts = self.prompts
        # pylint: disable=no-member # dynamic prompts (see self.__build_prompts)
        human_prompt = (prompts.get_prompt(f'{prompts.tag_prompt_file}_human.txt')
                        if self.debug else prompts.tag_prompt_human)
        # pylint: enable=no-member
        prompt_template = ChatPromptTemplate.from_messages([
                    ("human", human_prompt)
                ])
        prompt = prompt_template.format_messages(context=query)
        if self.debug:
            self.console.print(f'PRE-PROCESSOR PROMPT:\n{prompt}\n\n',
                                style='color(233)', highlight=False)
        content = pre_llm.llm_query(self.preconditioner, prompt).content
        if self.debug:
            self.console.print(f'PRE-PROCESSOR RESPONSE:\n{content}\n\n',
                                style='color(233)', highlight=False)

        tags = self.rag_tagger.get_tags(content, debug=self.debug)
        return (content, tags)

    def post_process(self, response)->None:
        """ Start a thread to process LLMs response """
        threading.Thread(target=self.rag_tagger.update_rag, args=(response,),
                         kwargs={'debug': self.debug},
                         daemon=True).start()

    def build_filter(
        self,
        tags: list[tuple[str, str]],
        skip_fields: set[str] = None,
        composite_fields: set[str] = None,
        multi_delimiters: str = ",/",
        **field_overrides
    ) -> dict | None:
        """
        Build a ChromaDB-compatible filter from a list of (tag, value) tuples.
        Supports composite fields (slash-separated), multi-fragment fields (comma-separated),
        and custom per-field logic via kwargs.

        Params:
            tags: list of (tag, value)
            skip_fields: tags to exclude (e.g., time, date)
            composite_fields: treat these as hierarchical (e.g., "combat/melee")
            multi_delimiters: delimiters that indicate multi-value fragments
            field_overrides: optional mapping of tag -> {"type": "composite"|"multi"|"exact"}

        Returns:
            dict filter or None
        """
        if skip_fields is None:
            skip_fields = {"time", "date"}
        if composite_fields is None:
            composite_fields = {"focus", "tone"}

        conditions = []
        delimiters = set(multi_delimiters)

        for tag, value in tags:
            # Skip unwanted fields or self-referential tags
            if tag in skip_fields or self.name in value:
                continue

            # Field behavior override
            field_type = field_overrides.get(tag, {}).get("type")

            # Composite fields like "combat/melee"
            if field_type == "composite" or (tag in composite_fields and '/' in value):
                parts = [p.strip() for p in value.split('/')]
                expanded = ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]
                conditions.append({tag: {"$in": expanded}})

            # Multi-value fields (comma-separated, slash fallback)
            elif field_type == "multi" or any(d in value for d in delimiters):
                for d in delimiters:
                    value = value.replace(d, ',')
                fragments = [v.strip() for v in value.split(',') if v.strip()]
                conditions.append({tag: {"$in": fragments}})

            # Simple exact match
            else:
                conditions.append({tag: value})

        return {"$and": conditions} if conditions else None

    @staticmethod
    def clean_tags(tags: list[RAGTag]) -> list[RAGTag]:
        """ clean filters """
        return [ tag for tag in tags if tag.content and
                 tag.content.lower() not in {'null', 'none', ''} and
                 not any(char in tag.content for char in [';)', "I'd say!"])]

    def gather_context(self, query: str,
                             collection: str,
                             tags: list[RAGTag[str,str]])->list[Document]:
        """
        Perform metadata field filtering matching
        """
        filter_dict = self.filter_builder.build(tags)
        # Combined filter retrieval (highly relevant information)
        documents = self.rag.retrieve_data(query,
                                           collection,
                                           meta_data=filter_dict,
                                           matches=self.matches)
        return documents

    def handle_context(self, data_set: list,
                             direction='query')->tuple[dict[str,list], int]:
        """ Method to handle all the lovely context """
        # Retrieve context from AI and User RAG
        if direction == 'query':
            pre_tokens = 0
            post_tokens = 0
            collection_list = ['ai_documents', 'user_documents']
            documents = {key: [] for key in collection_list}
            if data_set:
                query = self.common.stringify_lists(data_set[0])
                # Try to tagify the users query
                (_, meta_tags) = self.pre_processor(query)
                if self.debug:
                    self.console.print(f'TAG RETREIVAL:\n{meta_tags}\n\n',
                                       style='color(233)',
                                       highlight=False)
                for collection in collection_list:
                    storage = []
                    # Extensive RAG retreival: field filter dictionary, highly relevant
                    storage.extend(self.gather_context(query,
                                                       collection,
                                                       meta_tags))

                    # General RAG retreival: if Extensive retrieval above is low, figure the
                    # difference of allowed maximum, and use that number for matches without
                    # field filter searches
                    balance = self.matches - len(storage)
                    if self.debug:
                        self.console.print(f'BALANCE: {balance}', style='color(233)')
                    storage.extend(self.rag.retrieve_data(query,
                                                          collection,
                                                          matches=int(max(1, balance))))

                    # Record pre-token counts
                    pages = list(map(lambda doc: doc.page_content, storage))
                    for page in pages:
                        pre_tokens += self.token_retreiver(page)

                    # Remove duplicates RAG matches
                    documents[collection] = (list(set(pages)))

                    # Record post-token counts
                    for page in documents[collection]:
                        post_tokens += self.token_retreiver(page)

                if self.debug:
                    self.console.print(f'CONTEXT RETRIEVAL:\n{documents}\n\n',
                                       style='color(233)',
                                       highlight=False)

            # Store the users query to their RAG, now that we are done pre-processing
            # (so as not to bring back identical information in their query)
            # A little unorthodox, but the first item in the list is ther user's query
            self.rag.store_data(self.common.stringify_lists(data_set[0]),
                                tags_metadata=meta_tags,
                                collection='user_documents',
                                chunk_size=100,
                                chunk_overlap=50)
            # Return data collected
            return (documents, pre_tokens, post_tokens)
        # Store data (non-blocking)
        return self.post_process(data_set[0])
