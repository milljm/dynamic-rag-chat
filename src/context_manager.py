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
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
from openai import APITimeoutError
from .rag_manager import RAG, RAGTag
from .chat_utils import CommonUtils, ChatOptions
from .prompt_manager import PromptManager
from .filter_builder import FilterBuilder
from .scene_manager import SceneManager
from .gold_fetch import MAX_GOLD_FETCHES
from .attachment_store import list_attachments

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

        # output_version=v0 keeps chunk.content as a string so ThinkFeed
        # still sees MiniMax / gpt-oss reasoning tokens. use_responses_api=False
        # keeps LM Studio / Ollama on Chat Completions.
        self.pre_llm = ChatOpenAI(base_url=args.pre_host,
                                  model=args.preconditioner or args.model,
                                  temperature=args.pre_temp,
                                  streaming=False,
                                  max_tokens=8096,
                                  api_key=args.api_key,
                                  seed = args.seed,
                                  request_timeout=150,
                                  output_version='v0',
                                  use_responses_api=False)

        self.entity_llm = ChatOpenAI(base_url=args.entity_host,
                                  model=args.entity_llm or 'None',
                                  temperature=args.entity_temp,
                                  streaming=False,
                                  max_tokens=4096,
                                  api_key=args.api_key,
                                  seed = args.seed,
                                  request_timeout=150,
                                  output_version='v0',
                                  use_responses_api=False)

        self.filter_builder = FilterBuilder()
        self.prompts.build_prompts()

    # Helper methods for history schema migration
    @staticmethod
    def _is_message_list(history: list) -> bool:
        """True when history is the new role/content format."""
        return bool(history) and isinstance(history[0], dict) and 'role' in history[0]

    @staticmethod
    def _turn_count(messages: list) -> int:
        if not messages:
            return 0
        if ContextManager._is_message_list(messages):
            return sum(1 for m in messages if m.get('role') == 'user')
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
                return item.get('content', '') or ''
            return str(item) if item is not None else ''

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
        if not self.opts.assistant_mode:
            history = documents.get('history') or {}
            if isinstance(history, dict):
                self.scene.set_branch(self._active_branch(history))
        # Make only one previous turn available in chat_history when working with pre-conditioning
        if documents.get('chat_history'):
            documents['chat_history'] = self._messages_for_last_n_turns(
                                                        documents['chat_history'], 1)
        # pylint: disable-next=no-member # dynamic prompts (see self.__build_prompts)
        if self.opts.assistant_mode:
            human_prompt = prompts.compose_nostory_tag(documents)
        else:
            human_prompt = prompts.get_prompt(f'{prompts.tag_prompt_file}_human.md')
        human_tmpl = PromptTemplate(template=human_prompt,
                                    template_format='jinja2')
        human_msg = HumanMessagePromptTemplate(prompt=human_tmpl)
        messages = [human_msg]
        if not self.opts.assistant_mode:
            # pylint: disable-next=no-member
            system_prompt = prompts.get_prompt(f'{prompts.tag_prompt_file}_system.md')
            if system_prompt and system_prompt.strip():
                sys_tmpl = PromptTemplate(
                    template=system_prompt, template_format='jinja2',
                )
                messages.insert(0, SystemMessagePromptTemplate(prompt=sys_tmpl))
        prompt_template = ChatPromptTemplate.from_messages(messages)

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
        if not self.opts.assistant_mode:
            self.scene.set_branch(self._active_branch(history))

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

        self._mint_new_characters(roll_reversal)
        if not self.opts.assistant_mode:
            list_rag_tags = self.scene.ground_scene(list_rag_tags)

        branch = self._active_branch(history)
        collection = f'{branch}_{collection}'
        if self.debug:
            self.console.print(f'THREADED META TAGS PARSED: {list_rag_tags}',
                               style=f'color({self.opts.color})',
                               highlight=False)
        rag = RAG(self.console, self.common, self.opts)
        rag.store_data(response, tags_metadata=list_rag_tags, collection=collection)

    def _mint_new_characters(self, documents: dict) -> None:
        """Write NPC sheets for anyone new in the current scene entity list."""
        present = self.scene.get_scene().get('entity') or []
        if isinstance(present, str):
            present = [present]
        for char in present:
            if self.scene.is_new_character(char):
                self.create_character(char, documents)

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

        # pylint: disable-next=no-member # dynamic prompts (see self.__build_prompts)
        human_prompt = prompts.get_prompt(f'{prompts.entity_prompt_file}_human.md')
        human_tmpl = PromptTemplate(template=human_prompt,
                                    template_format='jinja2')
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
        """Retrieve by must-field: `entity` in story, `document_topics` in assistant."""
        storage = []

        # Perhaps the user does not want to use RAG
        if self.opts.matches == 0:
            return storage

        values = self.filter_builder.values_for(meta_tags, field)
        if not values:
            return storage

        for value in values:
            storage.extend(self.gather_context(
                query,
                collection,
                [RAGTag(tag=field, content=value)],
                field,
            ))
        return storage

    @staticmethod
    def is_explicit(meta_tags: list[RAGTag[str, str]]) -> bool:
        """Return bool if content_rating or scene_mode is nsfw."""
        return FilterBuilder.tags_are_nsfw(meta_tags)

    @staticmethod
    def use_agent(meta_tags: list[RAGTag[str, str]]) -> bool:
        """ Return bool if content benefits from a web search """
        try:
            for tag in meta_tags:
                if tag.tag == 'search_internet':
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
                entry = entry.strip().lstrip('[').rstrip(']')
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

    def _active_branch(self, history: dict) -> str:
        """Return the history branch name for this turn."""
        return self.common.active_branch(history)

    def _tag_user_query(self, query: str, documents: dict):
        """Run the pre-processor and print debug output. None on failure."""
        self.console.print(
            'Processing query (meta tagging for RAG)...',
            style=f'color({self.opts.color})',
            highlight=False,
        )
        (_, meta_tags, error) = self.pre_processor(query, documents)
        self.common.write_debug(
            f'handle_context_preprocess-{self.pre_llm.model_name}',
            meta_tags,
        )
        if self.debug:
            self.console.print(
                f'TAG RETRIEVAL:\n{meta_tags}\n\n',
                style=f'color({self.opts.color})',
                highlight=False,
            )
        if not error:
            return None
        return meta_tags

    def _apply_meta_to_documents(self, documents: dict, meta_tags) -> None:
        """Copy tags onto documents without clobbering existing keys."""
        documents['RAGTags'] = meta_tags
        documents['explicit'] = self.is_explicit(meta_tags)
        if self.use_agent(meta_tags) and self.opts.assistant_mode:
            documents['use_agent'] = True
            documents['agent_ran'] = False
        documents['entities'] = '---\n\n'.join(self.prompt_entities(meta_tags))
        documents['known_characters'] = ','.join(
            self.scene.get_scene().get('known_characters', []),
        )
        if self._turn_count(documents['chat_history']) > self.opts.unmolested_sessions:
            documents['chat_history'] = self.stagger_history(documents)
        gold = dict(documents)
        documents.update(meta_tags)
        documents.update(gold)
        for tag in meta_tags:
            val = tag.content
            if isinstance(val, list):
                documents[tag.tag] = ', '.join(str(x) for x in val)
            else:
                documents[tag.tag] = val
        documents.setdefault('player_location', '')
        documents.setdefault('entity', '')
        documents.setdefault('audience', '')
        documents.setdefault('npc_locations', '')
        if documents.get('content_type', False):
            documents['content_type'] = (
                '- Respond in the following format: ',
                f'{documents["content_type"]}',
            )

    def gold_collection_name(self, documents: dict) -> str:
        """Chroma collection that holds gold for this turn."""
        history = documents.get('history') or self.common.load_chat()
        branch = self._active_branch(history)
        prefix = self._collection_prefix(branch, 'gold_documents')
        return f'{prefix}gold_documents'

    def fetch_gold_file(self, documents: dict, filename: str) -> bool:
        """Inject a whole gold file into this turn. Assistant mode only.

        Returns True when the streamer should resume (found or not-found
        notice). False when we are out of fetches / not assistant.
        """
        if not self.opts.assistant_mode or self.opts.no_rags:
            return False
        name = (filename or '').strip()
        if not name:
            return False
        used = int(documents.get('gold_fetches', 0))
        if used >= MAX_GOLD_FETCHES:
            return False
        documents['gold_fetches'] = used + 1
        coll = self.gold_collection_name(documents)
        found = self.rag.retrieve_named_files(name, coll)
        blob = '\n\n'.join(
            getattr(doc, 'page_content', '') for doc in found if doc is not None
        )
        documents.setdefault('dynamic_files', '')
        documents.setdefault('gold_documents', '')
        label = f'GOLD_FETCH ({used + 1}/{MAX_GOLD_FETCHES}): {name}'
        if blob.strip():
            documents['gold_documents'] = (
                str(documents.get('gold_documents') or '') + '\n\n' + blob
            )
            documents['dynamic_files'] += f'\n=== {label} ===\n{blob}\n'
        else:
            documents['dynamic_files'] += (
                f'\n=== {label} ===\n'
                'Not in gold. Do not ask the user to attach it. '
                'Say you do not have that file.\n'
            )
        return True

    def _collection_prefix(self, branch: str, collection: str) -> str:
        """Chroma collection name prefix for this branch."""
        if self.opts.assistant_mode:
            return 'assistant_'
        if collection == 'gold_documents':
            return ''
        return f'{branch}_'

    def list_gold_files(self) -> list[dict]:
        """Whole files in vector_dir/attachments (Documents widget)."""
        return list_attachments(self.opts.vector_dir)

    def fill_documents_index(self, documents: dict) -> None:
        """Basenames the model may NEED_GOLD. Empty in story mode."""
        documents['has_documents_index'] = False
        documents['documents_index'] = ''
        if not self.opts.assistant_mode:
            return
        names = [
            str(row.get('name') or '').strip()
            for row in self.list_gold_files()
        ]
        names = [n for n in names if n]
        if not names:
            return
        documents['has_documents_index'] = True
        documents['documents_index'] = '\n'.join(f'- {n}' for n in names)

    def delete_gold_file(self, filename: str) -> bool:
        """Drop a named file from the cabinet and gold chunks."""
        history = self.common.load_chat()
        branch = self._active_branch(history)
        gold = f'{self._collection_prefix(branch, "gold_documents")}gold_documents'
        return self.rag.delete_named_file(gold, filename)

    def _fill_rag_collections(self, documents, meta_tags, query, branch):
        """Retrieve, dedupe, and stringify each RAG collection.

        Returns (pre_tokens, post_tokens).
        """
        pre_tokens = 0
        post_tokens = 0
        collections = [
            self.common.attributes.collections[x]
            for x in self.common.attributes.collections
        ]
        self.console.print(
            'Gathering RAG data...',
            style=f'color({self.opts.color})',
            highlight=False,
        )
        for collection in collections:
            prefix = self._collection_prefix(branch, collection)
            name = f'{prefix}{collection}'
            if self.debug:
                self.console.print(
                    f'Collection: {name}',
                    style=f'color({self.opts.color})',
                    highlight=False,
                )
            storage = []
            storage.extend(self._retrieve_collection(
                meta_tags, query, name, collection,
            ))
            pages = [doc.page_content for doc in storage]
            pre_tokens += sum(self.token_retriever(page) for page in pages)
            documents[collection] = self.deduplication(
                documents['chat_history'], pages,
            )
            post_tokens += sum(
                self.token_retriever(page) for page in documents[collection]
            )
            documents[collection] = self.common.stringify_lists(
                documents[collection],
            )
        return pre_tokens, post_tokens

    def _retrieve_collection(self, meta_tags, query: str, name: str,
                             collection: str) -> list:
        """Gold: whole file on filename mention, then topic/similarity as usual."""
        storage = []
        named = []
        if collection == 'gold_documents':
            named = self.rag.retrieve_named_files(query, name)
            storage.extend(named)
        extra = self.handle_topics(meta_tags, query, name, self.mode)
        if named:
            blob = '\n'.join(doc.page_content for doc in named)
            extra = [doc for doc in extra if doc.page_content not in blob]
            storage.extend(extra)
        else:
            storage.extend(extra)
            if not storage:
                storage.extend(self.rag.retrieve(query, name))
        return storage

    @staticmethod
    def _stringify_chat_history(documents: dict) -> None:
        """Flatten role/content messages into USER:/AI: lines."""
        chat_lines = []
        for msg in documents['chat_history']:
            if isinstance(msg, dict):
                chat_lines.append(CommonUtils.history_line(msg))
            else:
                chat_lines.append(str(msg))
        documents['chat_history'] = '\n'.join(chat_lines)

    def ingest_user_attachments(self, documents: dict, meta_tags: list | None) -> None:
        """Store this turn's *text* files in gold (assistant mode only).

        Attachments arrive *after* handle_context (Spur fold_uploads / {{path}}).
        Images stay this-turn-only for vision — no stub in Documents (clipboard
        pastes land as image.png and the 139-byte stub is not the picture).
        Agent search dumps in dynamic_files are not ingested.
        """
        documents.setdefault('attached_files_note', '')
        files = documents.get('attachment_texts') or []
        if not files:
            return
        gold_names: list[str] = []
        image_names: list[str] = []
        bodies: list[tuple[str, str]] = []
        for rec in files:
            if not isinstance(rec, dict):
                continue
            name = str(rec.get('name') or 'file')
            kind = str(rec.get('kind') or 'text')
            text = str(rec.get('text') or '')
            if kind == 'image':
                image_names.append(name)
                continue
            gold_names.append(name)
            clipped = text[:400_000]
            if clipped.strip():
                bodies.append((name, f'ATTACHED FILE: {name}\n\n{clipped}'))
        notes = []
        if image_names:
            notes.append(
                'THIS TURN the user attached image(s): '
                + ', '.join(image_names)
                + '. The pixels are in this message. Look at them. '
                'Do not write code to open the file. Not saved to Documents.'
            )
        if gold_names:
            notes.append(self._attachment_note(gold_names))
        if notes:
            documents['attached_files_note'] = ' '.join(notes)
        if not bodies or self.opts.no_rags or not self.opts.assistant_mode:
            return
        history = documents.get('history') or self.common.load_chat()
        branch = self._active_branch(history)
        gold = f'{self._collection_prefix(branch, "gold_documents")}gold_documents'
        for name, body in bodies:
            tags = list(meta_tags or [])
            tags.append(RAGTag('source', f'attachment:{name}'))
            tags.append(RAGTag('filename', name.lower()))
            self.rag.store_data(body, tags_metadata=tags, collection=gold, quiet=True)
            self.rag.store_full_file(gold, name, body)

    def _attachment_note(self, names: list[str]) -> str:
        """Tell the plot prompt these files are already in hand."""
        listed = ', '.join(names)
        if self.opts.assistant_mode:
            return (
                'THIS TURN the user paperclipped: ' + listed
                + '. Full text is in THIS_TURN_ATTACHMENTS / FILES right now. '
                'A copy is being saved to DOCUMENTS (permanent cabinet). '
                'Never ask them to attach these files again. Do not NEED_GOLD them this turn.'
            )
        return (
            'THIS TURN the user paperclipped: ' + listed
            + '. Full text is in THIS_TURN_ATTACHMENTS / FILES. '
            'Never ask them to attach these files again.'
        )

    def handle_context(self, documents: dict,
                             direction='query')->tuple[dict[str,list], int, list]:
        """Assemble RAG + history for a query, or post-process a reply."""
        if direction != 'query':
            return self.post_process(documents)

        history = documents['history']
        branch = self._active_branch(history)
        if not self.opts.assistant_mode:
            self.scene.set_branch(branch)
        documents['terminal_width'] = int(os.get_terminal_size().columns) - 5
        documents['chat_history'] = history[branch]
        documents['additional_content'] = self.get_explicit()
        documents['ooc_system'] = self.get_ooc()

        query = documents.get('user_query', '')
        meta_tags = self._tag_user_query(query, documents)
        if meta_tags is None:
            return ([], 0, 0, [])

        self._apply_meta_to_documents(documents, meta_tags)
        pre_tokens, post_tokens = 0, 0
        if not self.opts.no_rags:
            pre_tokens, post_tokens = self._fill_rag_collections(
                documents, meta_tags, query, branch,
            )
            self.rag.store_data(
                query,
                tags_metadata=meta_tags,
                collection=f'{branch}_{self.common.attributes.collections["user"]}',
            )
        else:
            documents.setdefault('user_documents', '')
            documents.setdefault('ai_documents', '')
            documents.setdefault('gold_documents', '')
        self._stringify_chat_history(documents)
        return (documents, pre_tokens, post_tokens, meta_tags)
