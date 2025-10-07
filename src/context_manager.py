"""
ContextManager aims at handling everything relating to the context
being supplied to the LLM. It utilizing several methods:

    Emoji removal.
    list[] -> set() removes any matches from the RAG.
    Staggered History.
    ParentDocument/ChildDocument retrieval (return one large response with many small one)
"""
import os
from difflib import SequenceMatcher
import threading
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from openai import APITimeoutError
from .rag_manager import RAG, RAGTag
from .chat_utils import CommonUtils, ChatOptions
from .prompt_manager import PromptManager
from .filter_builder import FilterBuilder
from .scene_manager import SceneManager

class ContextManager(PromptManager):
    """ A collection of methods aimed at producing/reducing the context """
    # pylint: disable=too-many-positional-arguments, too-many-arguments
    def __init__(self,
                 console,
                 common: CommonUtils,
                 rag: RAG,
                 scene: SceneManager,
                 current_dir,
                 args: ChatOptions):
        super().__init__(console, current_dir, args)
        self.console = console
        self.common = common
        self.rag = rag
        self.scene = scene
        self.opts = args
        self.mode = 'document_topics' if args.assistant_mode else 'entity'
        self.prompts = PromptManager(self.console,
                                     current_dir,
                                     args,
                                     prompt_model=args.preconditioner)
        self.pre_llm = ChatOpenAI(base_url=args.pre_host,
                                  model=args.preconditioner,
                                  temperature=0.3,
                                  streaming=False,
                                  max_tokens=4096,
                                  api_key=args.api_key,
                                  request_timeout=15)
        self.filter_builder = FilterBuilder()
        self.prompts.build_prompts()

    # pylint: enable=too-many-positional-arguments,too-many-arguments
    def deduplication(self, base_reference: list[str],
                            response_list: list[str],
                            threshold: float = 0.92) -> list[str]:
        """
        Deduplicate response_list using semantic similarity against base_reference.
        Returns cleaned RAG chunks.
        """
        def is_similar(a: str, b: str) -> bool:
            return SequenceMatcher(None,
                                   a.lower().strip(),
                                   b.lower().strip()).ratio() > threshold
        cleaned_chunks = []
        for chunk in response_list:
            if any(is_similar(chunk, base) for base in base_reference):
                continue
            if any(is_similar(chunk, prior) for prior in cleaned_chunks):
                continue
            cleaned_chunks.append(chunk)
        return cleaned_chunks

    @staticmethod
    def token_retriever(context: str|list[str])->int:
        """ iterate over string or list of strings and do a word count (token) """
        _token_cnt = 0
        if not isinstance(context, list|str):
            return _token_cnt
        if isinstance(context, list):
            for sentence in context:
                if sentence is not None:
                    _token_cnt += len(sentence.split())
        else:
            _token_cnt += len(context.split(' '))
        return int(_token_cnt * 1.3)

    def no_entity(self, tags: list[RAGTag])->bool:
        """ Bool check for entity == None """
        entity_tag = next((item for item in tags if item.tag == self.mode), None)
        if not entity_tag:
            return True
        entities = ''.join(entity_tag.content)
        if entities.lower() in ['none', 'no ent', 'null']:
            return True
        return False

    def pre_processor(self,
                      query: str,
                      documents: dict,
                      do_scene: bool=True)->tuple[str,list[RAGTag]]:
        """
        lightweight LLM as a tagging pre-processor
        Returns LLM's response, meta_tags, bool (general failure or not)
        """
        prompts = self.prompts
        query = self.common.normalize_for_dedup(query)
        # pylint: disable=no-member # dynamic prompts (see self.__build_prompts)
        human_prompt = (prompts.get_prompt(f'{prompts.tag_prompt_file}_human.md')
                        if self.debug or self.opts.prompts_debug else prompts.tag_prompt_human)
        # pylint: enable=no-member
        human_tmpl = PromptTemplate(template=human_prompt,
                                    template_format="jinja2")
        human_msg = HumanMessagePromptTemplate(prompt=human_tmpl)

        prompt_template = ChatPromptTemplate.from_messages([human_msg])

        prompt = prompt_template.format_messages(**documents)
        if self.debug:
            self.console.print(f'PRE-PROCESSOR PROMPT:\n{prompt}\n\n',
                                style=f'color({self.opts.color})', highlight=False)
        try:
            content = self.pre_llm.invoke(prompt).content
            self.common.write_debug(self.pre_llm.model_name, content)
        except APITimeoutError:
            return ('APITimeoutError', [], False)
        if self.debug:
            self.console.print(f'PRE-PROCESSOR RESPONSE:\n{content}\n\n',
                                style=f'color({self.opts.color})', highlight=False)

        # Parse tags (JSON) response from LLM
        tags = self.common.get_tags(content)

        if do_scene:
            tags = self.scene.ground_scene(tags, query)
            if self.debug:
                self.console.print(f'SCENE MANAGER OVERRIDE:\n{tags}\n\n',
                                style=f'color({self.opts.color})', highlight=False)
        return (content, tags, True)

    def post_process(self, documents: dict)->None:
        """ Start a thread to process LLMs response """
        threading.Thread(target=self.save_response, args=(documents,),
                         daemon=True).start()

    def save_response(self, documents: dict, collection: str='')->None:
        """
        ### Save Response

        Attempt to parse any metadata in LLM's `response`, and then store this
        information along with the response itself to the RAG's `collection`. If
        collection is not specified, the default will be used. There is also a
        hard-coded protection if an attempt to write to the gold collection is
        made.

        *Key init args:*
            .. code-block:: python
                documents: dict       # document object containing LLMs response
                collection: str = ''  # Defaults to AI Document collection
        *Returns None:*
            .. code-block:: python
                return None
        """
        # tagify the AI response
        response = documents['llm_response']
        (_, list_rag_tags, _) = self.pre_processor(response, documents)

        # Handle triggers
        self.scene.finalize_turn(response)
        self.scene.save_scene()

        collections = self.common.attributes.collections  # short hand
        # protect against empty or gold RAG (read-only) collections
        if not collection or collection == collections['gold']:
            collection = collections['ai']
        # list_rag_tags: list[RAGTag] = self.common.get_tags(response)
        self.scene.ground_scene(list_rag_tags, response)
        if self.debug:
            self.console.print(f'THREADED META TAGS PARSED: {list_rag_tags}',
                               style=f'color({self.opts.color})',
                               highlight=False)
        else:
            self.common.write_debug('meta_tags', list_rag_tags)
        rag = RAG(self.console, self.common, self.opts)
        rag.store_data(response, tags_metadata=list_rag_tags, collection=collection)

    @staticmethod
    def stagger_history(history_size: int,
                        max_elements: int = 20,
                        recent_tail: int = 4) -> list[int]:
        """
        Returns a list of indices from chat history with decaying density.
        - Guarantees `recent_tail` most recent indices.
        - Remaining slots are staggered across earlier history.
        """
        if history_size <= max_elements:
            return list(range(history_size))

        recent_tail = min(recent_tail, max_elements)
        base_count = max_elements - recent_tail
        earlier = history_size - recent_tail

        # Spread earlier indices linearly
        step = max(1, earlier) / max(1, base_count)
        indices = [int(i * step) for i in range(base_count)]

        # Add tail
        indices += list(range(history_size - recent_tail, history_size))
        return sorted(set(indices))

    def get_chat_history(self, history_list)->list:
        """ just return n previous turns """
        return '\n\n'.join(history_list[-self.opts.history_sessions:])

    def handle_topics(self,
                      meta_tags: list[RAGTag],
                      query: str,
                      collection: str,
                      field: str)->list[Document]:
        """
        ### Main Topic Retrieval

        Perform topic context matching routines focused on RAGTag key: `field`. Each content
        value in RAGTag(key=`field`, content=[]) will be searched individually in Chroma. The
        method will split the amount of returned matches evenly based on `--history-matches`.

        *Key init args:*
            .. code-block:: python
                meta_tags: list[RAGTag]
                query: str
                collection: str
                field: str
        *Returns:*
            .. code-block:: python
                return list[Document]
        """
        storage = []
        # Return immediately if we somehow have no topic field available
        topic_field = [x for x in meta_tags if x.tag == field]
        if not topic_field:
            return storage

        _meta = list(meta_tags)
        entity_weights = max(1, int(self.opts.matches * .75))

        # Grab the list of topics, then remove the RAGTag from the list, as Chroma does not
        # support searching a metadata tag with a list of values in it
        entities = topic_field[0].content
        _meta.remove(topic_field[0])

        # In case the pre-processor supplied a string of space separated items or one item
        if isinstance(entities, str):
            entities = entities.split(',')

        # Perform a balanced search for each entity/document_topics
        for a_entity in entities:
            for _ in range(max(1, int(entity_weights / len(entities)))):
                storage.extend(self.gather_context(
                                            query,
                                            collection,
                                            [RAGTag(tag=field, content=a_entity.lower()), *_meta],
                                            field)
                                            )
        return storage

    @staticmethod
    def is_explicit(meta_tags: list[RAGTag[str,str]]) -> bool:
        """ Return bool if content detected is NSFW """
        try:
            rating = [x.content for x in meta_tags if x.tag == 'scene_mode'] # shorthand
            return 'nsfw' in rating[0].lower()
        #pylint: disable=bare-except   # LLMs can get so many things wrong
        except:
            pass
        #pylint: enable=bare-except
        return False

    def prompt_entities(self, meta_tags: list[RAGTag[str,str]]) -> list[str]:
        """
        Return list of strings with grounding info for each entity detected in meta_tags.
        Handles entity content as list or delimiter-separated string.
        """
        # Collect raw values of all entity tags
        raw_entities = [x.content for x in meta_tags if x.tag == self.mode]
        if not raw_entities:
            return ['']

        seen = set()
        _entity_prompt = []

        for entry in raw_entities:
            # Step 1: normalize to list of strings
            if isinstance(entry, list):
                candidates = entry
            elif isinstance(entry, str):
                # Remove brackets and normalize
                entry = entry.strip().lstrip("[").rstrip("]")
                # Split on common delimiters
                candidates = self.common.regex.entities.split(entry)
            else:
                candidates = [str(entry)]

            # Step 2: load file for each unique, non-empty candidate
            for candidate in candidates:
                name = candidate.strip().lower()
                if not name or name in seen:
                    continue
                seen.add(name)
                safe_name = self.common.regex.safe_name.sub('_', name).strip('_')
                entity_file = os.path.join(
                    self.current_dir, 'prompts', 'entities', f'{safe_name}.txt'
                )
                if self.debug:
                    self.console.print(f'Loading Entity File:\n{entity_file}\n\n',
                                       style=f'color({self.opts.color})',
                                       highlight=False)
                if os.path.exists(entity_file):
                    with open(entity_file, 'r', encoding='utf-8') as f:
                        _entity_prompt.append(f.read())

        return _entity_prompt or ['']

    def gather_context(self, query: str,
                             collection: str,
                             tags: list[RAGTag],
                             field: str)->list[Document]:
        """
        Perform metadata field filtering matching
        """
        filter_dict = self.filter_builder.build(tags, field)
        # Combined filter retrieval (highly relevant information)
        documents = self.rag.retrieve(query,
                                      collection,
                                      metadatas=filter_dict)
        return documents

    def get_explicit(self):
        """ read and return nsfw.md file """
        nsfw_file = os.path.join(self.current_dir, 'prompts', 'nsfw.md')
        if os.path.exists(nsfw_file):
            with open(nsfw_file, 'r', encoding='utf-8') as f:
                return f.read()

    def handle_context(self, documents: dict,
                             direction='query')->tuple[dict[str,list], int]:
        """ Method to handle all the lovely context """
        # Retrieve context from AI and User RAG and Chat History
        if direction == 'query':
            pre_tokens = 0
            post_tokens = 0
            collection_list = [self.common.attributes.collections[x] for
                                x in self.common.attributes.collections]
            # documents = {key: [] for key in collection_list}

            # populate chat history
            history = documents['history'] # shorthand
            documents['chat_history'] = self.get_chat_history(
                    history[history.get('current',
                                        'default')])

            if self.opts.assistant_mode and not self.opts.no_rags:
                return (documents, pre_tokens, post_tokens)

            query = documents.get('user_query', '')

            # tag the users query
            (_, meta_tags, _) = self.pre_processor(query, documents)

            # grab entities and perform another tagging process (with character sheets)
            documents['entities'] = '\n\n'.join(self.prompt_entities(meta_tags))

            # Populate explicit content if triggered
            documents['explicit'] = self.is_explicit(meta_tags)

            # documents['nsfw_content'] = (self.get_explicit() if documents['explicit'] else '')
            documents['nsfw_content'] = self.get_explicit()

            # Known characters the user has encountered during the story
            # documents['known_characters'] = self.scene.render_known_characters_for_prompt()

            # Make all meta_tags available for prompt templating operations
            # Note: This *might* have the side-effect of breaking necessary keys, though rare.
            #       as in, we are allowing any tags that the pre-processor *might* invent on its
            #       own, could overwrite a required on. Still, the benefits outweigh the negs.
            #       Allowing it provides a synergy pipeline. Allowing us to stream from LLM to LLM
            documents.update(meta_tags)

            # If content_type is populated, instruct the LLM to respond in kind
            if documents.get('content_type', False):
                documents['content_type'] = ('- Respond in the following format: ',
                                                f'{documents["content_type"]}')

            if self.debug:
                self.console.print(f'TAG RETRIEVAL:\n{meta_tags}\n\n',
                                    style=f'color({self.opts.color})',
                                    highlight=False)

            for collection in collection_list:
                storage = []

                # field-filtering RAG retrieval specific for document_topics
                storage.extend(self.handle_topics(meta_tags,
                                                  query,
                                                  collection,
                                                  self.mode))

                # general retrieval
                storage.extend(self.rag.retrieve(query, collection))

                # Record pre-token counts
                pages = list(map(lambda doc: doc.page_content, storage))
                for page in pages:
                    pre_tokens += self.token_retriever(page)

                # Remove duplicates RAG matches
                documents[collection] = self.deduplication(documents['chat_history'],
                                                           pages)

                # Record post-token counts
                for page in documents[collection]:
                    post_tokens += self.token_retriever(page)

                # Stringify RAG retrieval lists
                documents[collection] = self.common.stringify_lists(documents[collection])

            # Stringify lists in chat_history
            documents['chat_history'] = self.common.stringify_lists(documents['chat_history'])

            # Store the users query to their RAG, now that we are done pre-processing
            # (so as not to bring back identical information in their query)
            # A little unorthodox, but the first item in the list is the user's query
            self.rag.store_data(query,
                                tags_metadata=meta_tags,
                                collection=self.common.attributes.collections['user'])
            # Return data collected
            return (documents, pre_tokens, post_tokens)

        # Store data (non-blocking)
        return self.post_process(documents)
