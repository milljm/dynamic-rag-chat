"""
ContextManager aims at handeling everything relating to the context
being supplied to the LLM. It utilizing several methods:

    Emoji removal
    Fuzzy match sentences
    list[] -> set() removes any matches from the RAG.
"""
import re
import os
import hashlib
import threading
from langchain.prompts import ChatPromptTemplate
from RAGTagManager import RAGTagManager, RAG
from OllamaModel import OllamaModel
from PromptManager import PromptManager
current_dir = os.path.dirname(os.path.abspath(__file__))
class ContextManager(PromptManager):
    """ A collection of methods aimed at producing/reducing the context """
    def __init__(self, console, **kwargs):
        super().__init__(console, kwargs)
        self.console = console
        self.prompts = PromptManager(console, self.debug)
        self.rag = RAG(console, **kwargs)
        self.rag_tagger = RAGTagManager(console, **kwargs)
        self.host = kwargs['host']
        self.matches = kwargs['matches']
        self.preconditioner = kwargs['preconditioner']
        self.debug = kwargs['debug']
        self.prompts.build_prompts()

    def pre_processor(self, query, collection='default')->str:
        """
        lightweight LLM as a summarization/tagging pre-processor
        """
        prompts = self.prompts
        pre_llm = OllamaModel(self.host)
        docs = self.rag.retrieve_data(query, collection, matches=self.matches)
        context = "\n\n".join(doc.page_content for doc in docs if doc.page_content.strip())
        if self.debug:
            self.console.print(f'DEBUG VECTOR:\n{context}', style='color(233)')
        # pylint: disable=no-member # dynamic prompts (see self.__build_prompts)
        prompt_template = ChatPromptTemplate.from_messages([
                    ("system", (prompts.get_prompt(f'{prompts.pre_prompt_file}_system.txt')
                                if self.debug else prompts.pre_prompt_system)),
                    ("human", (prompts.get_prompt(f'{prompts.pre_prompt_file}_human.txt')
                               if self.debug else prompts.pre_prompt_human))
                ])
        # pylint: enable=no-member
        prompt = prompt_template.format_messages(context=context, question='')
        content = pre_llm.llm_query(self.preconditioner, prompt).content
        tags = self.rag_tagger.get_tags(content, debug=self.debug)
        return (content, tags)

    def post_processing(self, response):
        """
        Send LLM's resoonse off for post processing as a thread. This allows the
        user to begin formulating a response (and even sending a new message
        before this has completed). This step will begin to shine as more and
        more vectors are established, allowing the pre-processing step to draw
        in more nuanced context.
        """
        prompts = self.prompts
        # pylint: disable=no-member # dynamic prompts (see self.__build_prompts)
        post_prompt = (prompts.get_prompt(f'{prompts.post_prompt_file}_system.txt')
                        if self.debug else prompts.post_prompt_system)
        prompt_template = ChatPromptTemplate.from_messages([
                    ("system", post_prompt),
                    ("human", '{context}')
                ])
        prompt = prompt_template.format_messages(context=response, question='')
        # pylint: enable=no-member
        threading.Thread(target=self.rag_tagger.update_rag,
                         args=(self.host, self.preconditioner, prompt),
                         kwargs={'debug': self.debug}).start()

    @staticmethod
    def token_retreiver(context_list: list)->int:
        """ iterate over list and do a word count (token) """
        _token_cnt = 0
        if isinstance(context_list, list):
            for sentence in context_list:
                _token_cnt += len(sentence.split())
        return _token_cnt

    @staticmethod
    def normalize_for_dedup(text: str) -> str:
        """ remove emojis and other markdown """
        text = re.sub(r'[\U0001F600-\U0001F64F\u2600-\u26FF\u2700-\u27BF]', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return ' '.join(text.lower().split())

    def deduplicate_texts(self, texts: list[str]) -> list[str]:
        """ attempt to remove fuzzy duplicate sentences """
        seen = set()
        unique = []
        for text in texts:
            norm = self.normalize_for_dedup(text)
            key = hashlib.md5(norm.encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                unique.append(text)
        return unique

    def handle_context(self, data, direction='query')->tuple:
        """ Method to handle all the lovely context """
        # Retrieve context from AI and User RAG
        if direction == 'query':
            documents = []
            for collection in ['ai_response', 'user_queries']:
                documents.extend(self.rag.retrieve_data(data,
                                                        collection,
                                                        matches=self.matches))

            # Retrieve data from fast pre-processor and query the RAG once more
            # This is where things get interesting. Has a knack for bringing
            # in otherwise missed but relevant context at the cost of ~1-2 seconds.
            (pre_query, rag_tags) = self.pre_processor(data, 'ai_response')
            for tag in rag_tags:
                documents.extend(self.rag.retrieve_data(tag.content,
                                                        tag.tag,
                                                        matches=self.matches))

            documents.extend(self.rag.retrieve_data(pre_query,
                                                    'ai_response',
                                                    matches=self.matches))

            # Lambda function to extract page_content from each document, then
            # a set() to remove any duplicates(you'd be surpised how many tokens this saves).
            token_reduction = 0
            _pre = self.token_retreiver(list(map(lambda doc: doc.page_content,documents)))
            context = list(set(list(map(lambda doc: doc.page_content, documents))))
            context = self.deduplicate_texts(context)
            _post = self.token_retreiver(context)
            token_reduction = _pre - _post
            if token_reduction and self.debug:
                self.console.print(f'RAG TOKEN REDUCTION:\n{_pre} --> '
                              f'{_post} = {token_reduction}\n\n', style='color(233)')

            # If the context is empty (no documents found), well, wow. Nothing.
            if not context:
                return ('', token_reduction)

            # LLMs prefer strings separated by \n\n
            return ('\n\n'.join(context), token_reduction)

        # Store Context to AI RAG
        # Aggresively fragment the response from heavy-weight LLM responses
        # We can afford to do this due to the dynamic RAG/Tagging in post_processing
        self.rag.store_data(data, collection='ai_response', chunk_size=150, chunk_overlap=50)
        # dynamic RAG creation starts here (non-blocking)
        self.post_processing(data)

