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
                                  temperature=args.pre_temp,
                                  streaming=False,
                                  max_tokens=8096,
                                  api_key=args.api_key,
                                  seed = args.seed,
                                  request_timeout=150)

        self.entity_llm = ChatOpenAI(base_url=args.entity_host,
                                  model=args.entity_llm,
                                  temperature=args.entity_temp,
                                  streaming=False,
                                  max_tokens=4096,
                                  api_key=args.api_key,
                                  seed = args.seed,
                                  request_timeout=150)

        self.filter_builder = FilterBuilder()
        self.prompts.build_prompts()

    # Helper methods for history schema migration
    @staticmethod
    def _is_message_list(history: list) -> bool:
        """True when history is the new role/content format."""
        return bool(history) and isinstance(history[0], dict) and "role" in history[0]

    @staticmethod
    def _turn_count(messages: list) -> int:
        if not messages:
            return 0
        if ContextManager._is_message_list(messages):
            return sum(1 for m in messages if m.get("role") == "user")
        return len(messages)

    @staticmethod
    def _messages_for_last_n_turns(messages: list, n: int) -> list:
        """Return the message list corresponding to the last n turns."""
        if not messages or n <= 0:
            return []
        if ContextManager._is_message_list(messages):
            return messages[-(n * 2):]
        return messages[-n:]

    # pylint: enable=too-many-positional-arguments,too-many-arguments
    def deduplication(self, base_reference: list, response_list: list) -> list[str]:
        """
        Deduplicate response_list by checking for overlap containment.
        Accepts either list[str] or list[dict] (role/content format).
        Returns cleaned RAG chunks (list[str]).
        """
        def to_text(item) -> str:
            if isinstance(item, dict):
                return item.get("content", "") or ""
            return str(item) if item is not None else ""

        def is_overlap_duplicate(a: str, b: str) -> bool:
            s, l = (a, b) if len(a) < len(b) else (b, a)
            if not s.strip():
                return False
            matcher = SequenceMatcher(None, s, l)
            match = matcher.find_longest_match(0, len(s), 0, len(l))
            containment_ratio = match.size / len(s)
            return containment_ratio > 0.65

        # Normalize the base reference once
        base_texts = [to_text(x) for x in base_reference]

        cleaned_chunks = []
        for chunk in response_list:
            chunk_text = to_text(chunk)
            if any(is_overlap_duplicate(chunk_text, base) for base in base_texts):
                continue
            if any(is_overlap_duplicate(chunk_text, prior) for prior in cleaned_chunks):
                continue
            cleaned_chunks.append(chunk_text)   # always store as string
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
                    _token_cnt += len(str(sentence).split())
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
        documents = dict(documents)
        # Make only one previous turn available in chat_history when working with pre-conditioning
        if documents.get('chat_history'):
            documents['chat_history'] = self._messages_for_last_n_turns(
                                                        documents['chat_history'], 1)
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
            self.common.write_debug('pre_processor', content)
        except APITimeoutError:
            return ('APITimeoutError', [], False)
        # pylint: disable-next=bare-except  # can't handle everything
        except:
            return('Failed Return', [], False)

        if self.debug:
            self.console.print(f'PRE-PROCESSOR RESPONSE:\n{content}\n\n',
                                style=f'color({self.opts.color})', highlight=False)

        # Parse tags (JSON) response from LLM
        tags = self.common.get_tags(content)

        if do_scene and not self.opts.assistant_mode:
            tags = self.scene.ground_scene(tags)
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
        # Handle Scene State
        history = documents['history'] # shorthand
        if not collection:
            collection = self.common.attributes.collections['ai']

        # Swap rolls, feeding the LLM's response back at the pre-processor for tagging
        response = documents['llm_response']
        roll_reversal = {'user_query'   : documents['llm_response'],
                         'chat_history' : documents['chat_history'],
                         'user_name'    : documents['user_name']}
        if self.debug:
            self.console.print(f'ROLL REVERSAL PRE-PROCESSOR:\n{roll_reversal}\n\n',
                style=f'color({self.opts.color})', highlight=False)
        (_, list_rag_tags, error) = self.pre_processor(response, roll_reversal)
        if not error:
            self.console.print('ERROR running pre-processor. Generated output not saved.'
                               r' Advised to run `\regenerate` to try again.',
                style=f'color({self.opts.color})', highlight=False)
            return

        scene = dict(self.scene.get_scene())
        for char in scene['entity']:
            if self.debug:
                self.console.print(f'CHAR CHECK:\n{char}\n\n',
                                style=f'color({self.opts.color})', highlight=False)
            if self.scene.is_new_character(char):
                if self.debug:
                    self.console.print(f'ADD CHAR:\n{char}\n\n',
                                style=f'color({self.opts.color})', highlight=False)
                self.create_character(char, roll_reversal)

        if not self.opts.assistant_mode:
            self.scene.save_scene()

        # protect against empty or gold RAG (read-only) collections
        if self.opts.assistant_mode:
            branch = 'assistant'
        else:
            branch = history.get('current', 'default')
        collection = f'{branch}_{collection}'

        # list_rag_tags: list[RAGTag] = self.common.get_tags(response)
        if not self.opts.assistant_mode:
            self.scene.ground_scene(list_rag_tags)
        if self.debug:
            self.console.print(f'THREADED META TAGS PARSED: {list_rag_tags}',
                               style=f'color({self.opts.color})',
                               highlight=False)
        rag = RAG(self.console, self.common, self.opts)
        rag.store_data(response, tags_metadata=list_rag_tags, collection=collection)

    def create_character(self, char: str, documents: dict)->None:
        """ Query the Entity LLM to generate a character file based on chat_history """
        if self.opts.assistant_mode or self.entity_llm.model_name == 'None':
            return
        if not os.path.exists(os.path.join(self.opts.vector_dir, 'entities')):
            os.makedirs(os.path.join(self.opts.vector_dir, 'entities'))

        safe_name = self.common.regex.safe_name.sub('_', char).strip('_')
        entity_file = os.path.join(self.opts.vector_dir, 'entities', f'{safe_name}.txt')

        # Entity already exists
        if os.path.exists(entity_file):
            if self.debug:
                self.console.print(f'Character Already Exists:\n{char}\n\n',
                    style=f'color({self.opts.color})', highlight=False)
            return

        prompts = self.prompts
        populated = {'character_name' : char} | documents

        # pylint: disable=no-member # dynamic prompts (see self.__build_prompts)
        human_prompt = (prompts.get_prompt(f'{prompts.entity_prompt_file}_human.md')
                        if self.debug or self.opts.prompts_debug else prompts.entity_prompt_human)
        # pylint: enable=no-member
        human_tmpl = PromptTemplate(template=human_prompt,
                                    template_format="jinja2")
        human_msg = HumanMessagePromptTemplate(prompt=human_tmpl)

        prompt_template = ChatPromptTemplate.from_messages([human_msg])

        prompt = prompt_template.format_messages(**populated)
        if self.debug:
            self.console.print(f'ENTITY-PROCESSOR PROMPT:\n{prompt}\n\n',
                                style=f'color({self.opts.color})', highlight=False)
        try:
            content = self.entity_llm.invoke(prompt).content
        except APITimeoutError:
            self.console.print('ENTITY-PROCESSOR API ERROR\n\n',
                                style=f'color({self.opts.color})', highlight=False)
            return
        if self.debug:
            self.console.print(f'ENTITY-PROCESSOR RESPONSE:\n{content}\n\n',
                                style=f'color({self.opts.color})', highlight=False)

        with open(os.path.join(self.opts.vector_dir,
                                'character_llm_debug.log'), 'w', encoding='utf-8') as f:
            f.write('\n\n'.join([str(prompt),str(content)]))

        if self.debug:
            self.console.print(f'Generating New Character:\n{content}\n\n',
                            style=f'color({self.opts.color})', highlight=False)
        with open(entity_file, 'w', encoding='utf-8') as f:
            f.write(content)

    @staticmethod
    def stagger_indices(history_size: int,
                        max_elements: int = 20,
                        recent_tail: int = 4,
                        lookback: int = None) -> list[int]:
        """
        Returns a list of indices from chat history with exponentially growing gaps.
        - Guarantees `recent_tail` most recent indices.
        - Older turns get progressively sparser representation.

        Args:
            history_size: Total number of turns in chat history.
            max_elements: Maximum indices to return (including tail).
            recent_tail: Number of guaranteed unmolested most-recent turns.
            lookback: Optional window. When set, hard floor excludes older turns.

        Returns:
            Sorted list of indices with decaying density.
        """
        if history_size <= max_elements:
            return list(range(history_size))

        if lookback is None:
            lookback = history_size

        floor = max(0, history_size - lookback)
        recent_tail = min(recent_tail, max_elements)

        earlier_end = history_size - recent_tail
        tail_indices = set(range(earlier_end, history_size))

        if earlier_end <= floor:
            return sorted(tail_indices)

        # Cap base_count to available slots before tail
        max_base_slots = earlier_end - floor
        base_count = min(max_elements - recent_tail, max_base_slots)

        earlier_span = max(1, earlier_end - floor)

        base_indices = set()
        for i in range(base_count):
            progress = i / max(1, base_count - 1)
            idx = int(floor + (1 - progress ** 2) * earlier_span)
            idx = min(idx, earlier_end - 1)
            base_indices.add(idx)

        return sorted(base_indices | tail_indices)

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

        # Perhaps the user does not want to use RAG
        if self.opts.matches == 0:
            return storage

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
        #pylint: disable-next=bare-except   # LLMs can get so many things wrong
        except:
            pass
        return False

    @staticmethod
    def use_agent(meta_tags: list[RAGTag[str, str]]) -> bool:
        """ Return bool if content benefits from a web search """
        try:
            for tag in meta_tags:
                if tag.tag == "search_internet":
                    return tag.content is True
        #pylint: disable-next=broad-exception-caught   # LLMs can get so many things wrong
        except Exception:
            pass
        return False

    def prompt_entities(self, meta_tags: list[RAGTag]) -> list[str]:
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
                    self.opts.vector_dir, 'entities', f'{safe_name}.txt'
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

    def get_explicit(self)->str:
        """ read and return nsfw.md file """
        nsfw_file = os.path.join(self.current_dir, 'prompts', 'nsfw.md')
        if os.path.exists(nsfw_file):
            with open(nsfw_file, 'r', encoding='utf-8') as f:
                return f.read()
        return ''

    def get_ooc(self)->str:
        """ read and return ooc_default_system.md """
        # this is temporary until I develop a separate OOC LLM calling method
        ooc_file = os.path.join(self.current_dir, 'prompts', 'ooc_default_system.md')
        if os.path.exists(ooc_file):
            with open(ooc_file, 'r', encoding='utf-8') as f:
                return f.read()
        return ''

    def stagger_history(self, documents) -> list:
        """
        Return a list of messages that includes:
        - the last `unmolested_sessions` turns completely intact
        - older turns sampled with exponential decay
        """
        full_history = documents.get('chat_history', [])
        if not full_history:
            return []

        # How many *turns* we want to keep in total
        max_turns = self.opts.history_sessions
        unmolested_turns = self.opts.unmolested_sessions

        # Fast path: keep everything recent
        if unmolested_turns == 0:
            return self._messages_for_last_n_turns(full_history, max_turns)

        total_turns = self._turn_count(full_history)
        if total_turns <= unmolested_turns:
            return full_history

        # Convert turn-based numbers → message indices
        is_new = self._is_message_list(full_history)
        msg_per_turn = 2 if is_new else 1

        # Guaranteed recent messages (the unmolested tail)
        unmolested_msgs = unmolested_turns * msg_per_turn
        recent = full_history[-unmolested_msgs:]

        # Everything before the unmolested tail
        older = full_history[:-unmolested_msgs]
        if not older:
            return recent

        # How many additional *turns* we are allowed to pull from the older part
        remaining_turns = max(0, max_turns - unmolested_turns)
        if remaining_turns == 0:
            return recent

        # Use the existing stagger logic, but on *turn* counts
        older_turn_count = self._turn_count(older)
        indices = self.stagger_indices(
            history_size=older_turn_count,
            max_elements=remaining_turns,
            recent_tail=0,                  # we already handled the real tail
            lookback=self.opts.lookback
        )

        # Map turn indices back to message slices
        selected = []
        for turn_idx in indices:
            start = turn_idx * msg_per_turn
            end = start + msg_per_turn
            selected.extend(older[start:end])

        return selected + recent

    def handle_context(self, documents: dict,
                             direction='query')->tuple[dict[str,list], int, list]:
        """ Method to handle all the lovely context """
        # Retrieve context from AI and User RAG and Chat History
        if direction == 'query':
            pre_tokens = 0
            post_tokens = 0
            history = documents['history'] # shorthand
            if self.opts.assistant_mode:
                branch = 'assistant'
            else:
                branch = history.get('current', 'default')
            collection_list = [self.common.attributes.collections[x] for
                                x in self.common.attributes.collections]

            # column cnt
            documents['terminal_width'] = int(os.get_terminal_size().columns) - 5
            # populate chat history
            documents['chat_history'] = history[branch]

            documents['additional_content'] = self.get_explicit()
            documents['ooc_system'] = self.get_ooc()

            if self.opts.assistant_mode and not self.opts.no_rags:
                return (documents, pre_tokens, post_tokens, [])

            query = documents.get('user_query', '')

            # tag the users query
            self.console.print('Processing query (meta tagging for RAG)...',
                               style=f'color({self.opts.color})',
                               highlight=False)
            (_, meta_tags, error) = self.pre_processor(query, documents)
            self.common.write_debug(f'handle_context_preprocess-{self.pre_llm.model_name}',
                                     meta_tags)
            if self.debug:
                self.console.print(f'TAG RETRIEVAL:\n{meta_tags}\n\n',
                                    style=f'color({self.opts.color})',
                                    highlight=False)
            if not error:
                return ([],0,0,[])

            # Add tags so they can be passed around
            documents['RAGTags'] = meta_tags

            # Populate explicit content if triggered
            documents['explicit'] = self.is_explicit(meta_tags)

            # Use agent if pre-processor believes that would help
            if self.use_agent(meta_tags) and self.opts.assistant_mode:
                documents['use_agent'] = True
                documents['agent_ran'] = False

            # grab entities and perform another tagging process (with character sheets)
            documents['entities'] = '---\n\n'.join(self.prompt_entities(meta_tags))
            documents['known_characters'] = ','.join(
                           self.scene.get_scene().get('known_characters', [])
                       )

            if self._turn_count(documents['chat_history']) > self.opts.unmolested_sessions:
                documents['chat_history'] = self.stagger_history(documents)

            # Make all meta_tags available for prompt templating operations, without overwriting
            # important already established keys.
            _gold = dict(documents)
            documents.update(meta_tags)
            documents.update(_gold)

            # If content_type is populated, instruct the LLM to respond in kind
            if documents.get('content_type', False):
                documents['content_type'] = ('- Respond in the following format: ',
                                                f'{documents["content_type"]}')
            self.console.print('Gathering RAG data...',
                               style=f'color({self.opts.color})',
                               highlight=False)
            for collection in collection_list:
                if self.opts.assistant_mode:
                    g_branch = 'assistant_'
                else:
                    g_branch = f'{branch}_'

                if not self.opts.assistant_mode and collection == 'gold_documents':
                    g_branch = ''

                if self.debug:
                    self.console.print(f'Collection: {g_branch}{collection}',
                                       style=f'color({self.opts.color})',
                                       highlight=False)
                storage = []
                # field-filtering RAG retrieval specific for document_topics
                storage.extend(self.handle_topics(meta_tags,
                                                  query,
                                                  f'{g_branch}{collection}',
                                                  self.mode))

                # general retrieval
                _retrieved = self.rag.retrieve(query, f'{g_branch}{collection}')
                if self.debug:
                    self.console.print(f'Data:\n{_retrieved}\n\n',
                                       style=f'color({self.opts.color})',
                                       highlight=False)
                storage.extend(_retrieved)

                # Record pre-token counts
                pages = list(map(lambda doc: doc.page_content, storage))
                for page in pages:
                    pre_tokens += self.token_retriever(page)

                # Remove duplicate RAG matches
                documents[collection] = self.deduplication(documents['chat_history'], pages)

                # Record post-token counts
                for page in documents[collection]:
                    post_tokens += self.token_retriever(page)

                # Stringify RAG retrieval lists
                documents[collection] = self.common.stringify_lists(documents[collection])

            # Stringify lists in chat_history
            chat_lines = []
            for msg in documents['chat_history']:
                if isinstance(msg, dict):
                    role = 'USER' if msg.get('role') == 'user' else 'AI'
                    content = msg.get('content', '')
                    chat_lines.append(f'{role}: {content}')
                else:
                    # safety net for any remaining old-format items
                    chat_lines.append(str(msg))
            documents['chat_history'] = '\n'.join(chat_lines)
            # Store the users query to their RAG, now that we are done pre-processing
            # (so as not to bring back identical information in their query)
            # A little unorthodox, but the first item in the list is the user's query
            self.rag.store_data(query,
                                tags_metadata=meta_tags,
                                collection=f'{branch}_{self.common.attributes.collections["user"]}')
            # Return data collected
            return (documents, pre_tokens, post_tokens, meta_tags)

        # Store data (non-blocking)
        return self.post_process(documents)
