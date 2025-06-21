"""
ContextManager aims at handeling everything relating to the context
being supplied to the LLM. It utilizing several methods:

    Emoji removal.
    list[] -> set() removes any matches from the RAG.
    Staggered History.
    ParentDocument/ChildDocument retrieval (return one large response with many small one)
"""
import os
import re
from difflib import SequenceMatcher
import threading
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import APITimeoutError
from .ragtag_manager import RAGTagManager, RAG, RAGTag
from .chat_utils import CommonUtils, ChatOptions # For type hinting
from .prompt_manager import PromptManager
from .filter_builder import FilterBuilder

class ContextManager(PromptManager):
    """ A collection of methods aimed at producing/reducing the context """
    # pylint: disable=too-many-positional-arguments, too-many-arguments
    def __init__(self,
                 console,
                 common: CommonUtils,
                 rag: RAG,
                 rag_tag: RAGTagManager,
                 current_dir,
                 args: ChatOptions):
        super().__init__(console, current_dir, args)
        self.console = console
        self.common = common
        self.rag = rag
        self.rag_tagger = rag_tag
        self.opts = args
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
    def token_retreiver(context: str|list[str])->int:
        """ iterate over string or list of strings and do a word count (token) """
        _token_cnt = 0
        if isinstance(context, list):
            for sentence in context:
                _token_cnt += len(sentence.split())
        else:
            _token_cnt += len(context.split(' '))
        return _token_cnt

    @staticmethod
    def no_entity(tags: list[RAGTag])->bool:
        """ Bool check for entity == None """
        entity = ''.join([x.content for x in tags if x.tag == 'entity'])
        if not entity:
            return True
        return entity.lower().find('none') != -1

    def pre_processor(self, query: str,
                      previous: str='',
                      do_scene: bool=True)->tuple[str,list[RAGTag],bool]:
        """
        lightweight LLM as a tagging pre-processor
        Returns LLM's response, meta_tags, bool (general failure or not)
        """
        prompts = self.prompts
        query = self.common.normalize_for_dedup(query)
        # pylint: disable=no-member # dynamic prompts (see self.__build_prompts)
        human_prompt = (prompts.get_prompt(f'{prompts.tag_prompt_file}_human.md')
                        if self.debug else prompts.tag_prompt_human)
        # pylint: enable=no-member
        prompt_template = ChatPromptTemplate.from_messages([
                    ("human", human_prompt)
                ])
        prompt = prompt_template.format_messages(context=query, previous=previous)
        if self.debug:
            self.console.print(f'PRE-PROCESSOR PROMPT:\n{prompt}\n\n',
                                style=f'color({self.opts.color})', highlight=False)
        else:
            with open(os.path.join(self.opts.vector_dir, 'debug.log'),
                      'w', encoding='utf-8') as f:
                f.write(f'PRE-PROCESSOR PROMPT: {prompt}')
        try:
            content = self.pre_llm.invoke(prompt).content
        except APITimeoutError:
            return ('APITimeoutError', [], self.common.no_scene(), False)
        if self.debug:
            self.console.print(f'PRE-PROCESSOR RESPONSE:\n{content}\n\n',
                                style=f'color({self.opts.color})', highlight=False)
        else:
            with open(os.path.join(self.opts.vector_dir, 'debug.log'),
                      'w', encoding='utf-8') as f:
                f.write(f'PRE-PROCESSOR RESPONSE: {content}')
        tags = self.common.get_tags(content)
        if self.no_entity(tags) and previous and not self.common.if_importing:
            try:
                last_contents = self.common.chat_history_session[-1:]
            except IndexError:
                last_contents = ''
            return self.pre_processor(query, previous=last_contents)
        # when importing documents directly into the RAG, we don't want to produce a scene
        scene_consistency = self.common.no_scene()
        if do_scene:
            scene_consistency = self.common.scene_tracker_from_tags(tags)
        return (content, tags, scene_consistency, True)

    def post_process(self, response)->None:
        """ Start a thread to process LLMs response """
        threading.Thread(target=self.rag_tagger.update_rag, args=(response,),
                         daemon=True).start()

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
        step = earlier / base_count
        indices = [int(i * step) for i in range(base_count)]

        # Add tail
        indices += list(range(history_size - recent_tail, history_size))
        return sorted(set(indices))

    def get_chat_history(self)->list:
        """
        Limit Chat History to 550 * int(history_sessions) tokens (defaults = 2750 tokens)
        """
        splice_list = self.stagger_history(len(self.common.chat_history_session),
                                           self.opts.history_sessions)
        spliced = [self.common.chat_history_session[x] for x in splice_list]
        abridged = []
        _tk_cnt = 0
        max_tokens = 550 * max(1, int(self.opts.history_sessions)) # Defaults = 2750 tokens
        for response in spliced:
            _tk_cnt += self.token_retreiver(response)
            if _tk_cnt > max_tokens:
                if self.debug:
                    self.console.print(f'MAX CHAT TOKENS: {_tk_cnt}',
                                       style=f'color({self.opts.color})',
                                       highlight=False)
                else:
                    with open(os.path.join(self.opts.vector_dir, 'debug.log'),
                              'w', encoding='utf-8') as f:
                        f.write(f'MAX CHAT TOKENS: {_tk_cnt}')
                return abridged
            abridged.append(response)
        return abridged

    def gather_context(self, query: str,
                             collection: str,
                             tags: list[RAGTag[str,str]],
                             field: str)->list[Document]:
        """
        Perform metadata field filtering matching
        """
        filter_dict = self.filter_builder.build(tags, field)
        # Combined filter retrieval (highly relevant information)
        documents = self.rag.retrieve_data(query,
                                           collection,
                                           meta_data=filter_dict,
                                           matches=self.opts.matches)
        return documents

    def prompt_entities(self, meta_tags: list[RAGTag[str,str]]) -> list[str]:
        """
        Return list of strings with grounding info for each entity detected in meta_tags.
        Handles entity content as list or delimiter-separated string.
        """
        # Collect raw values of all entity tags
        raw_entities = [x.content for x in meta_tags if x.tag == 'entity']
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
                candidates = re.split(r'[;,|\n]+|\s{2,}|(?<!\w)\s(?!\w)', entry)
            else:
                candidates = [str(entry)]

            # Step 2: load file for each unique, non-empty candidate
            for candidate in candidates:
                name = candidate.strip().lower()
                if not name or name in seen:
                    continue
                seen.add(name)
                entity_file = os.path.join(
                    self.current_dir, 'prompts', 'entities', f'{name}.txt'
                )
                if self.debug:
                    print(f'DEBUG: {entity_file}')
                if os.path.exists(entity_file):
                    with open(entity_file, 'r', encoding='utf-8') as f:
                        _entity_prompt.append(f.read())

        return _entity_prompt or ['']

    def hanlde_entity(self,
                      meta_tags: list[RAGTag],
                      query: str,
                      collection: str,
                      field: str)->list:
        """ perform entity context matching routines """
        flat_entities = []
        storage = []
        entity_weights = max(1, int(self.opts.matches * .75))
        # Obtain the entities tag, and it's contents
        entities = [x.content for x in meta_tags if x.tag == 'entity']

        # Clean the entities tag and store its contents
        past_multi_entity = []
        for key, tag in enumerate(meta_tags):
            if entities and tag.tag == 'entity':
                past_multi_entity = meta_tags.pop(key)

        # Generate a list of entities
        for entity in entities:
            if isinstance(entity, list):
                flat_entities.extend(entity)
            elif isinstance(entity, str):
                flat_entities.extend([e.strip() for e in entity.split(',')
                                                if e.strip()])
        # Perform a balanced search for each entity
        for a_entity in flat_entities:
            meta_tags.append(RAGTag(tag='entity', content=a_entity.lower()))
            for _ in range(max(1, int(entity_weights / len(flat_entities)))):
                storage.extend(self.gather_context(query,
                                                   collection,
                                                   meta_tags,
                                                   field))
            # Remove the entities RAGTag from the list each time
            for key, tag in enumerate(meta_tags):
                if entities and tag.tag == 'entity':
                    meta_tags.pop(key)

        # add the pruned entity tag back in (allow a search for combination)
        if past_multi_entity:
            meta_tags.append(past_multi_entity)
            storage.extend(self.gather_context(query,
                                               collection,
                                               meta_tags,
                                               field))
        return storage

    def handle_context(self, data_set: list,
                             direction='query')->tuple[dict[str,list], int]:
        """ Method to handle all the lovely context """
        # Retrieve context from AI and User RAG and Chat History
        if direction == 'query':
            pre_tokens = 0
            post_tokens = 0
            collection_list = ['ai_documents', 'user_documents']
            documents = {key: [] for key in collection_list}
            documents['chat_history'] = self.get_chat_history()
            if self.opts.assistant_mode and not self.opts.no_rags:
                return (documents, pre_tokens, post_tokens)
            if data_set:
                query = self.common.stringify_lists(data_set[0])
                # Try to tagify the users query
                (_, meta_tags, documents['scene_meta'], _) = self.pre_processor(query)

                # set entities. We will use this to load a grounded character sheet
                # if it exists in promts/entities/entity.txt
                documents['entities'] = '\n\n'.join(self.prompt_entities(meta_tags))

                if self.debug:
                    self.console.print(f'TAG RETREIVAL:\n{meta_tags}\n\n',
                                       style=f'color({self.opts.color})',
                                       highlight=False)
                else:
                    with open(os.path.join(self.opts.vector_dir, 'debug.log'),
                              'w', encoding='utf-8') as f:
                        f.write(f'TAG RETREIVAL: {meta_tags}')
                for collection in collection_list:
                    storage = []
                    # Extensive RAG retreival: field filter dictionary, highly relevant
                    # Loop until we've exhausted fields or collected maximum
                    for field in ['entity', 'focus', 'tone', 'emotion', 'other']:
                        # Entity is very important, as it represents an NPC/Character. We
                        # will therefor spend 75% of our allotted context window budget
                        # on Entity field-filtering matches. We will also include a full
                        # file dedicated to said character, if the file exists.
                        if field == 'entity':
                            storage.extend(self.hanlde_entity(meta_tags,
                                                              query,
                                                              collection,
                                                              field))
                        # 25% other field-filters
                        storage.extend(self.gather_context(query,
                                                           collection,
                                                           meta_tags,
                                                           field))

                        balance = max(0, self.opts.matches - len(storage))
                        if balance == 0 or len(storage) >= self.opts.matches:
                            break

                    if self.debug:
                        self.console.print(f'BALANCE: {balance}',
                                           style=f'color({self.opts.color})')
                    else:
                        with open(os.path.join(self.opts.vector_dir, 'debug.log'),
                                  'w', encoding='utf-8') as f:
                            f.write(f'BALANCE: {balance}')
                    # Fallback to simularity search based on the difference. Always allow one.
                    storage.extend(self.rag.retrieve_data(query,
                                                          collection,
                                                          matches=int(max(1, balance))))

                    # Record pre-token counts
                    pages = list(map(lambda doc: doc.page_content, storage))
                    for page in pages:
                        pre_tokens += self.token_retreiver(page)

                    # Remove duplicates RAG matches
                    documents[collection] = self.deduplication(documents['chat_history'],
                                                               pages)

                    # Record post-token counts
                    for page in documents[collection]:
                        post_tokens += self.token_retreiver(page)

                if self.debug:
                    self.console.print(f'CONTEXT RETRIEVAL:\n{documents}\n\n',
                                       style=f'color({self.opts.color})',
                                       highlight=False)
                else:
                    with open(os.path.join(self.opts.vector_dir, 'debug.log'),
                              'w', encoding='utf-8') as f:
                        f.write(f'CONTEXT RETRIEVAL: {documents}')

            # Store the users query to their RAG, now that we are done pre-processing
            # (so as not to bring back identical information in their query)
            # A little unorthodox, but the first item in the list is ther user's query
            self.rag.store_data(self.common.stringify_lists(data_set[0]),
                                tags_metadata=meta_tags,
                                collection='user_documents')
            # Return data collected
            return (documents, pre_tokens, post_tokens)
        # Store data (non-blocking)
        return self.post_process(data_set[0])
